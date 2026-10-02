#!/usr/bin/env python3
"""verl 桥的闸门：登记、装载零缺键与与 HF 的单步对拍。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

RECIPE = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Looma：verl 桥的端到端闸门")
    ap.add_argument(
        "--ckpt", required=True, help="HF 目录（export_hf.py 的产物，或 tiny_checkpoint）"
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args(argv)
    ckpt = Path(args.ckpt)
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"[looma·bridge] {ckpt} 不是 HF 目录（缺 config.json）")

    sys.path.insert(0, str(RECIPE.parents[3]))
    from shensi.recipes.paper.looma.common.train.export_hf import (
        _bind_pg_collection,
        _init_distributed,
    )
    from shensi.recipes.paper.looma.stage2_rl import looma_bridge  # noqa: F401

    _init_distributed(args.device)

    from megatron.bridge import AutoBridge
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    hf_cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)

    bridge = AutoBridge.from_hf_pretrained(ckpt, trust_remote_code=True)
    inner = bridge._model_bridge  # noqa: SLF001
    provider = bridge.to_megatron_provider(load_weights=False)
    spec = provider.transformer_layer_spec
    ok_b1 = type(inner).__name__ == "LoomaForCausalLMBridge" and (
        getattr(spec.module, "__name__", "") == "LoomaTransformerLayer"
    )
    print(
        f"[looma·bridge] B1 注册与分发：{'PASS' if ok_b1 else 'FAIL'}"
        f"（桥 {type(inner).__name__}，层规格 {getattr(spec.module, '__name__', spec)}）"
    )
    if not ok_b1:
        return 1

    overrides = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": False,
        "seq_length": int(hf_cfg.max_position_embeddings),
        "gradient_accumulation_fusion": False,
        "attention_backend": "unfused",
        "persist_layer_norm": False,
        "masked_softmax_fusion": False,
        "apply_rope_fusion": False,
        "bias_activation_fusion": False,
        "bias_dropout_fusion": False,
    }
    if hasattr(provider, "apply_overrides_and_finalize"):
        provider.apply_overrides_and_finalize(dtype=torch_dtype, overrides=overrides)
    _bind_pg_collection(provider)
    model = provider.provide()
    if args.device == "cuda":
        model = model.cuda()

    report = bridge.load_hf_weights(model, str(ckpt))
    missing = list(getattr(report, "missing_keys", []) or [])
    real_missing = [k for k in missing if not k.endswith("_extra_state")]
    ok_b2 = not real_missing
    print(
        f"[looma·bridge] B2 装载零缺键：{'PASS' if ok_b2 else 'FAIL'}"
        f"（missing={len(missing)}，其中真缺 {len(real_missing)}）"
        + (f"；前几个：{real_missing[:5]}" if real_missing else "")
    )
    if not ok_b2:
        return 1

    model = model.to(torch_dtype)
    reloaded = (
        AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch_dtype)
        .to(args.device)
        .eval()
    )
    saved = [(layer, layer.looma_cfg.max_iter) for layer in model.decoder.layers]
    saved_hf = [(layer, layer.solver_max_iter) for layer in reloaded.model.layers]

    def compare(tag: str, max_iter: int) -> float:
        for layer, _ in saved:
            layer.looma_cfg.max_iter = max_iter
        for layer, _ in saved_hf:
            layer.solver_max_iter = max_iter
        worst, worst_len = 0.0, 0
        for seq_len in (1, 2, 4, 8, 16):
            ids = torch.randint(0, int(hf_cfg.vocab_size), (1, seq_len), device=args.device)
            positions = torch.arange(seq_len, device=args.device).unsqueeze(0)
            with torch.no_grad():
                ref = model(ids.clone(), positions, None)
                ref = (ref[0] if isinstance(ref, tuple) else ref).float()
                got = reloaded(ids.clone()).logits.float()
            delta = float((ref[..., : got.shape[-1]] - got).abs().max())
            if delta > worst:
                worst, worst_len = delta, seq_len
        print(f"[looma·bridge] {tag}：各长度最差 max|Δ| = {worst:.3e}（seq={worst_len}）")
        return worst

    one_step = compare("B3 单步对拍（两侧 max_iter=1）", 1)
    for layer, value in saved:
        layer.looma_cfg.max_iter = value
    for layer, value in saved_hf:
        layer.solver_max_iter = value
    compare("B3 报数（参考设置 max_iter=8）", hf_cfg.looma_max_iter)

    tol = 1e-3 if args.dtype == "fp32" else 5e-2
    ok = one_step < tol
    print(
        f"\n[looma·bridge] {'全部通过' if ok else '有失败项'}"
        f"（B3 判据：两侧迭代次数钉成 1、精度同为 {args.dtype}，各长度 logits 都应在 {tol:g} 内；"
        "逐位级的接线判据请用 --dtype fp32）"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
