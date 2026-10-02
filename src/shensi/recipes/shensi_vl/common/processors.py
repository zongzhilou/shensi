"""Kimi K3 的 processor / image_processor（论文配方只借它的**图像侧**）。

KimiK3Processor = KimiK3VisionProcessor（图像）+ 自带 tokenizer（文本）。本配方的文本侧
（DSV4F tokenizer + DSv4 chat 模板）不走它的 tokenizer，所以只加载 image_processor：
  - ``preprocess(pil_images)`` → pixel_values / grid_thws（patch 14、merge 2×2、mean/std 0.5）
  - 有 ``make_image_prompt(w, h)`` 时直接拿它产出的图像 token 串，用 DSV4F tokenizer 计数；
    老版本没有这个方法时，退回 grid_thws 的 t*h*w 估 token 数（等价口径）。
加载走 trust_remote_code（auto_map 指向发布件里的 kimi_k3_vision_processing / kimi_k3_processor）。
"""

from __future__ import annotations

from pathlib import Path


def load_image_processor(processor_path: str | Path):
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(str(processor_path), trust_remote_code=True)
    if not hasattr(proc, "preprocess"):
        raise SystemExit(f"[processors] {processor_path} 不是可用的 image_processor")
    return proc


def image_token_text(iproc, width: int, height: int) -> str:
    """该尺寸图像在序列里占的图像 token 串（Kimi 口径：merge 后每格 1 token）。"""
    make = getattr(iproc, "make_image_prompt", None)
    if callable(make):
        return make(width, height)
    raise SystemExit(
        "[processors] image_processor 没有 make_image_prompt(w,h)："
        "确认加载的是 KimiK3VisionProcessor（trust_remote_code）"
    )


def image_token_count(iproc, width: int, height: int, tok=None) -> int:
    """图像占多少个文本槽位：优先拿 prompt 串数一遍，退回 t*h*w。"""
    make = getattr(iproc, "make_image_prompt", None)
    if callable(make) and tok is not None:
        prompt = make(width, height)
        if isinstance(prompt, str):
            return len(tok(prompt, add_special_tokens=False)["input_ids"])
        if isinstance(prompt, (list, tuple)):
            return len(prompt)
    grid = getattr(iproc, "grid_thws", None)
    if grid is not None:
        t, h, w = (int(v) for v in list(grid)[-1])
        return int(t * h * w)
    raise SystemExit("[processors] 算不出图像 token 数（没有 make_image_prompt / grid_thws）")


def preprocess_images(iproc, images: list) -> dict:
    """PIL 图 → pixel_values / grid_thws（张量形状与字段名以 KimiK3VisionProcessor 为准）。"""
    out = iproc.preprocess(images, return_tensors="pt")
    for key in ("pixel_values", "grid_thws"):
        if key not in out:
            raise SystemExit(f"[processors] preprocess 输出缺 {key}（keys={list(out)}）")
    return out
