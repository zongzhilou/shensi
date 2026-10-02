#!/usr/bin/env python3
"""白化实现的在训练内（带常驻显存压力）计时台。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
os.environ.setdefault("SHENSI_ROOT", str(Path(__file__).resolve().parents[7]))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29573")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)

from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.parallel_state import initialize_model_parallel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

initialize_model_parallel(1, 1)

import shensi.recipes.paper.gated_delta_attn_res.common.models.megatron as M  # noqa: E402
from shensi.recipes.paper.gated_delta_attn_res.common.kernels import whiten_ns  # noqa: E402
from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (  # noqa: E402
    gdar_connection as gc,
)

STATS: dict = {}


def _reset() -> None:
    STATS.update(calls=0, shapes={}, ms=0.0, nan=0)


def build(layers: int, hidden: int, ffn: int, heads: int, kv: int, seq: int) -> GPTModel:
    cfg = TransformerConfig(
        num_layers=layers,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
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
    model_parallel_cuda_manual_seed(0)
    return GPTModel(
        config=cfg,
        transformer_layer_spec=M.gdar_layer_spec_paper,
        vocab_size=151936,
        max_sequence_length=seq,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=1000000,
    ).cuda()


def wrap(impl: str):
    orig = gc._whitening_transform

    def timed(values, mode, ridge, return_inverse=False):
        if impl == "eager":
            fn = orig
        elif impl == "ns":
            fn = whiten_ns.whitening_transform_ns
        else:  # pragma: no cover
            raise ValueError(impl)
        t0 = time.perf_counter()
        out = fn(values, mode, ridge)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        STATS["ms"] += (time.perf_counter() - t0) * 1e3
        STATS["calls"] += 1
        shape = tuple(values.shape)
        STATS["shapes"][shape] = STATS["shapes"].get(shape, 0) + 1
        if torch.isnan(out).any():
            STATS["nan"] += 1
        return out

    gc._whitening_transform = timed
    return orig


def run(model: GPTModel, ids: torch.Tensor, impl: str, steps: int, warmup: int = 1) -> None:
    wrap(impl)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    step_ms = []
    for i in range(warmup + steps):
        if i == warmup:
            _reset()
        t0 = time.perf_counter()
        out = model(ids, position_ids=None, attention_mask=None)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out.float().pow(2).mean().backward()
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= warmup:
            step_ms.append((time.perf_counter() - t0) * 1e3)
    n = len(step_ms)
    total_step = sum(step_ms) / n
    print(
        f"  {impl:<6} 步 {n} 次：平均 {total_step:.0f} ms/微步｜白化 {STATS['ms'] / n:.0f} ms"
        f"（{STATS['calls'] / n:.0f} 次/微步，{STATS['ms'] / max(STATS['calls'], 1):.1f} ms/次）"
        f"｜占整步 {STATS['ms'] / n / max(total_step, 1e-9) * 100:.0f}%｜NaN {STATS['nan']}",
        flush=True,
    )
    shapes = sorted(STATS["shapes"].items(), key=lambda kv_: -kv_[1])[:3]
    print(f"         主要形状：{shapes}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="白化在真实训练路径上的计量")
    ap.add_argument("--layers", type=int, default=19)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=2816)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--impls", default="eager,ns")
    args = ap.parse_args(argv)

    model = build(args.layers, args.hidden, args.ffn, args.heads, args.kv, args.seq)
    ids = torch.randint(0, 151936, (1, args.seq)).cuda()
    print(
        f"几何：{args.layers} 层 × {args.hidden} 宽、micro-batch 1 × seq {args.seq}、"
        f"论文主行 spec（B=4、白化 full）"
    )
    for impl in [x for x in args.impls.split(",") if x]:
        run(model, ids, impl, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
