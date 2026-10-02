#!/usr/bin/env python3
"""unified（encoder-free，对齐 Gemma 4）的闸门：无 ViT、占位契约、预算、前馈/反传。

python -m shensi.recipes.paper.deeprecur.common.models.transformers.smoke_test
"""

from __future__ import annotations

import torch

from ...data import _mock_canvas
from ...paths import TOKENIZER_DIR
from .configuration import Qwen3VLUnifiedConfig, budget_to_max_pixels
from .modeling_qwen3_vl_unified import build_unified, build_unified_processor

BUDGET = 1120
TEXT = {
    "vocab_size": 151669,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 4096,
}
VISION = {
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "mm_embed_dim": 64,
    "output_proj_dims": 64,
    "mm_posemb_size": 64,
}

PROMPT = (
    "<|im_start|>user\n<|image_pad|>Describe this image.<|im_end|>\n"
    "<|im_start|>assistant\nA checkerboard.<|im_end|>\n"
)


def _model():
    torch.manual_seed(0)
    return build_unified(
        tiny_config={"text": dict(TEXT), "vision": dict(VISION)}, dtype=torch.float32
    )[0]


def _processor():
    return build_unified_processor(str(TOKENIZER_DIR), BUDGET)


def _image(seed: int = 1):
    return _mock_canvas({"width": 192, "height": 192, "cells": 6, "seed": seed})


def check_encoder_free() -> str:
    """视觉侧没有 ViT / 注意力 / deepstack：参数恰为嵌入器的 LN/Dense/位置/投影。"""
    model = _model()
    names = [n for n, _ in model.named_parameters()]
    assert not any("visual" in n or "deepstack" in n for n in names), "仍有视觉塔/deepstack 参数"
    embedder = model.model.vision_embedder
    cfg = model.config.vision_config
    patch_dim = cfg.model_patch_size**2 * 3
    mm = cfg.mm_embed_dim
    hidden = model.config.text_config.hidden_size
    expected = (
        patch_dim * 2  # LN1(w,b)
        + patch_dim * mm
        + mm  # Dense(w,b)
        + mm * 2  # LN2(w,b)
        + cfg.mm_posemb_size * 2 * mm  # 因子化 2D 位置
        + mm * 2  # pos_norm(w,b)
        + cfg.output_proj_dims * hidden  # 投影（无 bias）
    )
    actual = sum(p.numel() for p in embedder.parameters())
    assert actual == expected, f"嵌入器参数量 {actual} != 预期 {expected}"
    return (
        f"无 ViT/注意力/deepstack；视觉侧仅嵌入器 {actual} 参数"
        f"（{cfg.model_patch_size}px 合并 patch → {patch_dim} 维 → Dense → 位置 → 投影）"
    )


def check_placeholder_contract() -> str:
    """占位契约：soft token 数 == 占位 span 数；不一致显式报错。"""
    model, processor = _model(), _processor()
    batch = processor(text=[PROMPT], images=[_image()])
    n_placeholder = int((batch["input_ids"] == processor.image_token_id).sum())
    n_valid = int((batch["image_position_ids"] != -1).all(-1).sum())
    assert n_placeholder == n_valid, f"占位 {n_placeholder} != 有效 patch {n_valid}"
    out = model(**batch)
    assert torch.isfinite(out.logits).all()

    bad = dict(batch)
    bad_input = batch["input_ids"].clone()
    row, col = (bad_input == processor.image_token_id).nonzero()[0].tolist()
    bad_input[row, col] = 0  # 抹掉一个真实占位 → soft token 数少 1，必须报错
    bad["input_ids"] = bad_input
    try:
        model(**bad)
    except SystemExit as exc:
        assert "不一致" in str(exc)
        return f"占位 {n_placeholder} == soft token {n_valid}；缺一个占位时显式报错"
    raise AssertionError("占位不匹配没有被拦住")


def check_budget() -> str:
    """Gemma 4 预算：像素上限 = 预算×m²（m=48）；图像处理器的 soft token ≤ 预算。"""
    assert budget_to_max_pixels(BUDGET) == BUDGET * 48**2
    processor = _processor()
    big = _mock_canvas({"width": 1024, "height": 1024, "cells": 8, "seed": 3})
    out = processor.image_processor(images=[big], return_tensors="pt")
    tokens = int((out["image_position_ids"] != -1).all(-1).sum())
    assert tokens <= BUDGET, f"1024² 图产出 {tokens} > 预算 {BUDGET}"
    return f"预算 {BUDGET} → 像素上限 {BUDGET * 48**2}；1024² 图落地 {tokens} soft token ≤ 预算"


def check_forward_backward() -> str:
    """前馈/反传：loss 有限；视觉嵌入器（含位置表）拿到非零梯度。"""
    model, processor = _model(), _processor()
    batch = processor(text=[PROMPT, PROMPT], images=[[_image(1)], [_image(2)]])
    batch["labels"] = batch["input_ids"].clone()
    model.train()
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    vision_params = dict(model.model.vision_embedder.named_parameters())
    missing = [n for n, p in vision_params.items() if p.grad is None]
    total = sum(float(p.grad.abs().sum()) for p in vision_params.values() if p.grad is not None)
    assert not missing, f"视觉嵌入器无梯度：{missing}"
    assert total > 0
    return f"loss={float(out.loss.detach()):.3f}；嵌入器全参数有梯度（|grad| 合计 {total:.3f}）"


def check_text_only() -> str:
    """纯文本路径（无图）可用；config 的模型类型与合并 patch 边长自洽。"""
    model, processor = _model(), _processor()
    batch = processor(text=["<|im_start|>user\nhi<|im_end|>\n"])
    out = model(**batch)
    assert torch.isfinite(out.logits).all()
    cfg = Qwen3VLUnifiedConfig()
    assert cfg.model_type == "qwen3_vl_unified" and cfg.model_patch_size == 48
    return "纯文本前向可用；config.model_type=qwen3_vl_unified、model_patch_size=48"


CHECKS = (
    check_encoder_free,
    check_placeholder_contract,
    check_budget,
    check_forward_backward,
    check_text_only,
)


def main() -> int:
    print("[deeprecur·unified] 逐项闸门（encoder-free，对齐 Gemma 4）：")
    for check in CHECKS:
        print(f"  ✓ {check.__name__}: {check()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
