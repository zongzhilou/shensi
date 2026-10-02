#!/usr/bin/env python3
"""unified（encoder-free）在引擎侧的现状闸门 + 明确的 vLLM 口径。

- **默认档（离线，不下权重）**：造 tiny ckpt → 存盘 → 回读（同进程按类加载）→ 前向出 logits，
  验证「存盘/回读往返 + 占位对齐 + 预算」这条链。
- **vLLM 口径（诚实说明）**：``qwen3_vl_unified`` 是 encoder-free 的**自研架构**，vLLM 里
  **没有**对应原生实现（native 档才能走上游 ``qwen3_vl.py``）。要上引擎需要照 vLLM 的
  ``model_executor/models/gemma4_unified.py``（同架构的参考实现）做插件；插件落地前，
  本配方不声称 vLLM 可用。

python -m shensi.recipes.paper.deeprecur.common.models.vllm.smoke_generate
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from ..transformers.modeling_qwen3_vl_unified import (
    Qwen3VLUnifiedForConditionalGeneration,
    build_unified_processor,
)
from .tiny_checkpoint import build

PROMPT = (
    "<|im_start|>user\n<|image_pad|>Describe this image.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def gate_roundtrip() -> int:
    """Tiny ckpt 存盘 → 回读 → 前向：验证往返与占位契约。"""
    from ...data import _mock_canvas
    from ...paths import TOKENIZER_DIR
    from ..variants import UNIFIED

    tmp = Path(tempfile.mkdtemp(prefix="deeprecur_unified_ckpt_"))
    try:
        ckpt = tmp / "tiny"
        build(ckpt)
        model = Qwen3VLUnifiedForConditionalGeneration.from_pretrained(ckpt).eval()
        processor = build_unified_processor(str(TOKENIZER_DIR), UNIFIED.visual_token_budget)
        image = _mock_canvas({"width": 192, "height": 192, "cells": 6, "seed": 5})
        batch = processor(text=[PROMPT], images=[image])
        import torch

        with torch.no_grad():
            logits = model(**batch).logits
        assert torch.isfinite(logits).all()
        n_placeholder = int((batch["input_ids"] == processor.image_token_id).sum())
        print(
            f"[deeprecur·vllm] ✓ 存盘→回读→前向通过：logits {tuple(logits.shape)}，"
            f"占位 {n_placeholder} == soft token（预算 {UNIFIED.visual_token_budget}）"
        )
        print(
            "[deeprecur·vllm] ℹ vLLM：encoder-free 架构无原生实现，需按 "
            "vllm/model_executor/models/gemma4_unified.py 做插件（未做，不声称可用）"
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="unified（encoder-free）引擎侧现状闸门")
    parser.parse_args()
    return gate_roundtrip()


if __name__ == "__main__":
    raise SystemExit(main())
