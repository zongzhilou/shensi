#!/usr/bin/env python3
"""闸门：HF ↔ mcore 权重通路（表 + 转换 + 装载 + logits 对拍），不经 Bridge。

python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.test_convert_tiny
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/home/louzo/code/shensi/shensi/src")
os.environ.setdefault("SHENSI_ROOT", "/home/louzo/code/shensi/shensi")
os.environ.setdefault("SHENSI_FS", "/home/louzo/fsdata")

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")


def main() -> int:
    # ---------------- 1) tiny HF GDAR（非默认旋钮，逼表走满分支）
    from shensi.recipes.paper.gated_delta_attn_res.models.transformers.configuration_qwen3_gdar import (
        Qwen3GDARConfig,
    )
    from shensi.recipes.paper.gated_delta_attn_res.models.transformers.modeling_qwen3_gdar import (
        Qwen3GDARForCausalLM,
    )

    V, H, L, HEADS = 256, 64, 2, 2
    hf_cfg = Qwen3GDARConfig(
        vocab_size=V,
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=L,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attn_res_block_size=1,
        attn_res_gate_rank=16,
        attn_res_q_rank=16,
        attn_res_k_rank=16,
        attn_res_gate_param="deviation",
        attn_res_update="objective",
        attn_res_address="delta",
        attn_res_decay_ladder=8,
        attn_res_read_heads=HEADS,
        attn_res_read_null=True,
        attn_res_read_whiten="diag",
    )
    torch.manual_seed(0)
    hf_model = Qwen3GDARForCausalLM(hf_cfg).eval()
    hf_sd = {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    print(f"HF: {len(hf_sd)} tensors, {sum(v.numel() for v in hf_sd.values()):,} params")

    # ---------------- 2) 名字表 + 转换
    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert import (
        AttnLayout,
        build_table,
        hf_to_mcore,
    )

    table = build_table("gdar", hf_cfg, L)
    layout = AttnLayout.from_config(hf_cfg)
    mcore_sd, report = hf_to_mcore(hf_sd, table, layout=layout, padded_vocab_size=V)
    check(
        "转换报告无缺口（HF 侧全覆盖）",
        bool(getattr(report, "ok", False)),
        getattr(report, "describe", lambda: "")()[:160] if hasattr(report, "describe") else "",
    )
    print(f"  → mcore: {len(mcore_sd)} tensors")
    if getattr(report, "synthesized", None):
        print(f"  → mcore 侧合成（HF 无对应）：{sorted(report.synthesized)}")

    # 往返位级：mcore → HF 转回，与原 HF 逐位比。
    # 这是"映射正确性"的判据；后面对拍 logits 相差的是跨实现的核数值，两者分开看。
    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert import mcore_to_hf

    hf_back, _report2 = mcore_to_hf(mcore_sd, table, layout=layout, vocab_size=V)
    diffs = []
    for k, v in hf_sd.items():
        back = hf_back.get(k)
        if back is None:
            diffs.append((k, "missing"))
        elif back.shape != v.shape or not torch.equal(back.to(v.dtype), v):
            diffs.append((k, f"max|Δ|={(back.float() - v.float()).abs().max().item():.2e}"))
    check(
        "往返位级（mcore→HF 与原 HF 逐位相等）", not diffs, f"{len(hf_sd)} 张量，差异 {diffs[:3]}"
    )

    # ---------------- 3) 按 HF 旋钮建 mcore 模型（TE 子模块 = 表的布局前提）
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29597")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    from megatron.core.models.gpt import GPTModel
    from megatron.core.parallel_state import initialize_model_parallel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    cfg = TransformerConfig(
        num_layers=L,
        hidden_size=H,
        ffn_hidden_size=128,
        num_attention_heads=HEADS,
        num_query_groups=1,
        kv_channels=32,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        gated_linear_unit=True,
        init_method_std=0.02,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        params_dtype=torch.float32,
        add_bias_linear=False,
        qk_layernorm=True,  # 表假定 mcore 侧带 qk-norm（synth 常量 1，等价于无）
        transformer_impl="transformer_engine",
    )
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron.gdar_spec import make_gdar_spec

    spec = make_gdar_spec(
        gdar_block_size=1,
        gdar_gate_param="deviation",
        gdar_update="objective",
        gdar_address="delta",
        gdar_decay_ladder=8,
        gdar_gate_rank=16,
        gdar_q_rank=16,
        gdar_k_rank=16,
        gdar_read_heads=HEADS,
        gdar_read_null=True,
        gdar_read_whiten="diag",
        gdar_output_route=True,
        gdar_write_carrier_bias=-4.0,
    )
    mcore = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=V,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=10000,
    ).eval()

    missing, unexpected = mcore.load_state_dict(mcore_sd, strict=False)
    miss = [k for k in missing if not k.endswith("_extra_state")]
    unexp = [k for k in unexpected if not k.endswith("_extra_state")]
    check("mcore 装载：无缺键", not miss, f"missing={miss[:4]}")
    check("mcore 装载：无意外键", not unexp, f"unexpected={unexp[:4]}")

    # ---------------- 3b) 合成行必须"永远是零"：钉死 + 不可训练
    pinned = {
        name: param
        for name, param in mcore.named_parameters()
        if name.endswith((".q_proj.up.bias", ".k_proj.up.bias"))
    }
    drifting = [n for n, par in pinned.items() if par.requires_grad or par.count_nonzero().item()]
    check(
        "合成行零且不可训练（训练后不会与 HF 分叉）",
        pinned and not drifting,
        f"{len(pinned)} 个 up.bias，漂移/可训练 {drifting[:3]}",
    )

    # ---------------- 4) 前向对拍
    torch.manual_seed(1)
    ids = torch.randint(0, V, (1, 16))
    # mcore 的并行层会把参数放到当前 CUDA 设备：显式各就各位（mcore 与它的输入上 GPU，HF 留 CPU）
    mcore = mcore.cuda()
    with torch.no_grad():
        hf_logits = hf_model(ids).logits.float().cpu()
        mc_logits = mcore(ids.cuda(), position_ids=None, attention_mask=None)
    mc_logits = mc_logits[0] if isinstance(mc_logits, (tuple, list)) else mc_logits
    mc_logits = mc_logits.float().cpu()
    mc_logits = (
        mc_logits.transpose(0, 1) if mc_logits.shape[:2] != hf_logits.shape[:2] else mc_logits
    )
    d = (hf_logits.float() - mc_logits.float()).abs().max().item()
    # 判据口径：权重正确性已由"往返位级"钉死；这里比的是 HF 与 TE-mcore 两个**不同实现**
    # 的核数值（TE 融合核 vs torch 逐算子），与实测的 TE 底噪 9.8e-3 同阶，阈值取 2e-2。
    check(
        "logits 对拍（HF vs TE-mcore，核数值底噪口径 |Δ| ≤ 2e-2）", d <= 2e-2, f"max|Δ| = {d:.3e}"
    )
    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
