#!/usr/bin/env python3
"""引擎闸门：三臂 tiny ckpt → 登记 → vLLM 生成（主干原生 + 自研件走桥）。

- **native**：vLLM 原生 ``Qwen3VLForConditionalGeneration``（不用桥）——本闸门不覆盖，见 README；
- **unified / gdar / deeprecur**：桥登记后由 vLLM 的 Transformers 后端加载，跑出 token 即通过。

python -m shensi.recipes.paper.deeprecur.common.models.vllm.engine_smoke --arm deeprecur
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from .tiny_checkpoint import build

PROMPT_RAW = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _image():
    from ...data import _mock_canvas

    return _mock_canvas({"width": 256, "height": 256, "cells": 8, "seed": 7})


def run(arm: str, tokens: int = 8) -> int:
    from transformers import AutoProcessor

    from .registration import describe, register_all

    report = register_all()
    assert not report["failed"], f"登记失败：{report['failed']}"
    tmp = Path(tempfile.mkdtemp(prefix=f"deeprecur_{arm}_engine_"))
    try:
        ckpt = tmp / "tiny"
        build(ckpt, arm=arm)
        print(f"[deeprecur·vllm] 解析：{describe()}")

        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        prompt = PROMPT_RAW
        if getattr(processor, "species", "") == "qwen3_vl_unified":
            from ...paths import TOKENIZER_DIR
            from ..transformers.modeling_qwen3_vl_unified import build_unified_processor
            from ..variants import UNIFIED

            processor = build_unified_processor(str(TOKENIZER_DIR), UNIFIED.visual_token_budget)
        image = _image()

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(ckpt),
            dtype="float32",
            enforce_eager=True,
            max_model_len=2048,
            gpu_memory_utilization=0.45,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        out = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            SamplingParams(max_tokens=tokens, temperature=0.0),
        )
        ids = list(out[0].outputs[0].token_ids)
        assert ids, "引擎没有产出 token"
        print(
            f"[deeprecur·vllm] ✓ arm={arm} 由 vLLM 出 token {len(ids)}/{tokens}：{ids}"
        )
        print(
            "[deeprecur·vllm] ℹ 主干（注意力/KV/融合）走 vLLM 原生；自研件（AR/reinject/feedback）"
            "在 HF 侧实现（桥）"
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="三臂 tiny ckpt 的引擎闸门")
    parser.add_argument("--arm", default="deeprecur", choices=["unified", "gdar", "deeprecur"])
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()
    return run(args.arm, args.tokens)


if __name__ == "__main__":
    raise SystemExit(main())
