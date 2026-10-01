#!/usr/bin/env python3
"""B6：连接算子的耗时分解（真算，不是估计）。

在同一几何（0.6B 宽度）下测 fwd+bwd 的 ms/step：
  plain Qwen3 / GDAR 主行 / 白化关 / 单读头 / 全秩 / 低秩 r64（默认）/ 不同块粒度
输出各档相对 plain 的倍数，并给出「要写融合内核得先赢过什么」的结论依据。

    python -m shensi.recipes.paper.gated_delta_attn_res.train.bench_connection
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, "/home/louzo/code/shensi/shensi/src")
os.environ.setdefault("SHENSI_ROOT", "/home/louzo/code/shensi/shensi")
os.environ.setdefault("SHENSI_FS", "/home/louzo/fsdata")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29571")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.parallel_state import initialize_model_parallel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

initialize_model_parallel(1, 1)


def make_config(num_layers: int, hidden: int, heads: int, kv: int, seq: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_hidden_size=4 * hidden,
        num_attention_heads=heads,
        num_query_groups=kv,
        kv_channels=hidden // heads,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        gated_linear_unit=True,
        init_method_std=0.02,
        add_bias_linear=False,
        qk_layernorm=True,
        transformer_impl="local",
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )


def build(cfg, spec, vocab=4096, seq=2048):
    return GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=vocab,
        max_sequence_length=seq,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=1000000,
    ).cuda()


def time_step(model, ids, *, warmup=3, iters=8) -> float:
    """fwd+bwd 的中位毫秒数。"""
    opts = torch.optim.SGD(model.parameters(), lr=0.0)
    for _ in range(warmup):
        out = model(ids, position_ids=None, attention_mask=None)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out.float().pow(2).mean().backward()
        opts.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model(ids, position_ids=None, attention_mask=None)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out.float().pow(2).mean().backward()
        opts.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="连接算子耗时分解（0.6B 宽度）")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args(argv)

    import shensi.recipes.paper.gated_delta_attn_res.models.megatron as M

    cfg = make_config(args.layers, args.hidden, args.heads, args.kv, args.seq)
    ids = torch.randint(0, 4096, (args.batch, args.seq)).cuda()

    specs = {
        "plain Qwen3": get_gpt_layer_local_spec(
            num_experts=None, moe_grouped_gemm=False, qk_layernorm=True, normalization="RMSNorm"
        ),
        "GDAR 主行（B=4, r64, 8 头, 全白化）": M.gdar_layer_spec_paper,
        "GDAR 子层形态（B=1）": M.gdar_layer_spec_paper_sublayer,
        "GDAR 白化关（read_whiten=off）": M.gdar_layer_spec_paper_whiten_off,
        "GDAR 白化对角（diag）": M.gdar_layer_spec_paper_whiten_diag,
        "GDAR 单读头（read_heads=1）": M.gdar_layer_spec_paper_heads1,
        "GDAR 全秩（无低秩投影）": M.gdar_layer_spec_paper_rankfull,
        "GDAR r16（参数匹配）": M.gdar_layer_spec_paper_rank16,
        "RealFormer（参考：只加注意力残差）": __import__(
            "shensi.recipes.paper.gated_delta_attn_res.models.megatron.realformer_spec",
            fromlist=["realformer_layer_spec"],
        ).realformer_layer_spec,
    }

    base_ms = None
    print(
        f"几何：{args.layers} 层 × hidden {args.hidden}，{args.batch}×{args.seq} tokens，bf16，fwd+bwd 中位"
    )
    print(f"{'spec':<38} {'ms/step':>9} {'vs plain':>9}  参数增量")
    plain_params = None
    for name, spec in specs.items():
        model_parallel_cuda_manual_seed(0)
        model = build(cfg, spec)
        n_params = sum(p.numel() for p in model.parameters())
        plain_params = plain_params or n_params
        ms = time_step(model, ids, iters=args.iters)
        base_ms = base_ms or ms
        delta = n_params - plain_params
        print(f"{name:<38} {ms:>9.1f} {ms / base_ms:>8.2f}×  {delta:>+12,}")
        del model
        torch.cuda.empty_cache()

    print(
        "\n结论口径：这一列的倍数是**整步**的（骨干占大头），连接算子的增量 ≈ 倍数 − 1；"
        "\n低秩是主要省钱项、白化（每步一次 eigh）是主要花钱项。融合内核要赢过的是"
        "\n「连接增量 × 主行 token 吞吐」那部分，而不是整步。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
