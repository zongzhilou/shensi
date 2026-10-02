"""多模态对话数据：messages jsonl ↔ Qwen3-VL processor 输入，assistant-only loss mask。

中间态是引擎无关的 messages jsonl，一行一个样本：

```json
{"id": "...", "images": ["images/x.jpg"],
 "messages": [{"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
              {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
```

``{"type": "image"}`` 部件按出现顺序消费 ``images`` 列表；路径相对 jsonl 所在目录解析，
绝对路径原样使用。落真实模型时这套 schema 可以直接换 engine。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rows(jsonl: str | Path, limit: int | None = None) -> list[dict]:
    """读 messages jsonl（``--limit`` 取前 N 条；空行跳过）。"""
    rows = []
    with open(jsonl, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("messages"):
                raise SystemExit(f"[deeprecur] jsonl 行缺 messages：{row.get('id')}")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"[deeprecur] jsonl 是空的：{jsonl}")
    return rows


def _resolve(row: dict, base_dir: Path) -> list[Image.Image]:
    images = []
    for path in row.get("images") or []:
        p = Path(path)
        if not p.is_absolute():
            p = base_dir / p
        images.append(Image.open(p).convert("RGB"))
    return images


class VLChatDataset(Dataset):
    """messages jsonl 的 torch Dataset：返回原始行（collator 里再做 processor）。"""

    def __init__(self, rows: list[dict], jsonl_dir: Path):
        self.rows = rows
        self.jsonl_dir = Path(jsonl_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class MockVLDataset(Dataset):
    """合成冒烟数据：确定性画布（棋盘 + 编号），不碰磁盘与网络。

    messages 也按真实 schema 走（image 部件 + 文本部件），只是图像内容是画出来的。
    """

    def __init__(self, n: int = 16, seed: int = 42):
        self.n = n
        self.seed = seed
        rng = random.Random(seed)
        self.rng = rng
        self.prompts = [
            "Describe this image.",
            "What color dominates the picture?",
            "Caption the image in one sentence.",
            "How many patches does this grid have?",
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        prompt = self.prompts[idx % len(self.prompts)]
        answer = f"Mock answer {idx}: a {8 + idx % 4}x8 checkerboard canvas."
        return {
            "id": f"mock-{idx}",
            "mock_image": {"width": 448, "height": 448, "cells": 8, "seed": rng.randint(0, 2**31)},
            "images": [],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ],
        }


def _mock_canvas(spec: dict) -> Image.Image:
    rng = random.Random(spec["seed"])
    img = Image.new("RGB", (spec["width"], spec["height"]))
    cells = spec["cells"]
    cw, ch = spec["width"] // cells, spec["height"] // cells
    for y in range(cells):
        for x in range(cells):
            color = (rng.randint(32, 255), rng.randint(32, 255), rng.randint(32, 255))
            for py in range(y * ch, (y + 1) * ch):
                for px in range(x * cw, (x + 1) * cw):
                    img.putpixel((px, py), color)
    return img


def _expanded_ids(processor, text: str, tokens_per_image: list[int]) -> list[int]:
    """渲染文本 → token ids，并把每个 ``<|image_pad|>`` 展开成该图的真实 token 数。

    processor 编码时会做同样的展开，loss 区间必须在这一口径上算才不错位。
    """
    tokenizer = processor.tokenizer
    ids = tokenizer(text)["input_ids"]
    image_token_id = processor.image_token_id
    out: list[int] = []
    index = 0
    for tid in ids:
        if tid == image_token_id and index < len(tokens_per_image):
            out.extend([image_token_id] * tokens_per_image[index])
            index += 1
        else:
            out.append(tid)
    return out


def assistant_spans(processor, messages: list[dict], tokens_per_image: list[int]) -> list[tuple[int, int]]:
    """按轮次算 assistant 内容在展开后序列里的 token 区间 [start, im_end 含]。

    做法：完整渲染展开一次得总长；对每个 assistant 轮 j，用
    ``apply_chat_template(messages[:j], add_generation_prompt=True)`` 得前缀（同样展开），
    再从该位置向后找第一个 ``<|im_end|>``。模板无关，不依赖 generation 标记。
    """
    tokenizer = processor.tokenizer
    full = _expanded_ids(
        processor,
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        tokens_per_image,
    )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    spans = []
    for j, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix = _expanded_ids(
            processor,
            processor.apply_chat_template(
                messages[:j], tokenize=False, add_generation_prompt=True
            ),
            tokens_per_image,
        )
        start = len(prefix)
        end = next((i for i in range(start, len(full)) if full[i] == im_end), len(full) - 1)
        spans.append((start, end))
    return spans


class VLCollator:
    """batch 化：渲染模板 → processor → labels（assistant 区间外 + 图像 + padding 置 -100）。"""

    def __init__(self, processor, jsonl_dir: Path | None, max_length: int = 2048):
        self.processor = processor
        self.jsonl_dir = Path(jsonl_dir) if jsonl_dir else None
        self.max_length = max_length

    def __call__(self, rows: list[dict]) -> dict:
        proc = self.processor
        texts, images = [], []
        for row in rows:
            if row.get("mock_image"):
                images.append([_mock_canvas(row["mock_image"])])
            else:
                images.append(_resolve(row, self.jsonl_dir))
            texts.append(proc.apply_chat_template(row["messages"], tokenize=False))
        n_images_per_row = [len(group) for group in images]

        if getattr(proc, "species", "") == "qwen3_vl_unified":
            # encoder-free 处理器：按行分组传图，产出 pixel_values / image_position_ids
            batch = proc(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            counts_per_row = _unified_counts(batch, n_images_per_row)
        else:
            flat = [img for group in images for img in group]
            batch = proc(
                text=texts,
                images=flat or None,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            counts_per_row = _grid_counts(batch.get("image_grid_thw"), n_images_per_row, proc)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        if proc.image_token_id is not None:
            labels[batch["input_ids"] == proc.image_token_id] = -100
        for i, row in enumerate(rows):
            spans = assistant_spans(proc, row["messages"], counts_per_row[i])
            keep = torch.zeros_like(labels[i], dtype=torch.bool)
            for start, end in spans:
                keep[start : min(end + 1, len(keep))] = True
            labels[i][~keep] = -100
        batch["labels"] = labels
        return {k: v for k, v in batch.items() if hasattr(v, "shape")}


def _grid_counts(grid_rows, n_images_per_row: list[int], proc) -> list[list[int]]:
    """Qwen 原生口径：每图占位数 = grid_thw 之积 / merge²。"""
    merge = proc.image_processor.merge_size**2
    counts, cursor = [], 0
    for n_images in n_images_per_row:
        row_counts = []
        if grid_rows is not None and n_images:
            for grid in grid_rows[cursor : cursor + n_images]:
                row_counts.append(int(grid.prod()) // merge)
            cursor += n_images
        counts.append(row_counts)
    return counts


def _unified_counts(batch, n_images_per_row: list[int]) -> list[list[int]]:
    """encoder-free 口径：每图占位数 = 该图 soft token 数。

    优先用处理器给的 ``num_soft_tokens_per_image``（摊平后仍是逐图计数）；退化路径按
    ``image_position_ids`` 的有效行数算（padded 口径）。
    """
    field = batch.get("num_soft_tokens_per_image")
    if field is not None:
        flat = [int(c) for c in (field.tolist() if hasattr(field, "tolist") else field)]
    else:
        position_ids = batch["image_position_ids"]
        flat = [int(c) for c in (position_ids != -1).all(-1).sum(-1).tolist()]
    counts, cursor = [], 0
    for n_images in n_images_per_row:
        counts.append(flat[cursor : cursor + n_images])
        cursor += n_images
    return counts
