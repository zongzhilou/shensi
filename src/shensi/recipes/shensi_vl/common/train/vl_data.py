"""VL 数据集与 collator：jsonl 行 →（pixel_values, input_ids, labels, image_positions）。

统一样本 schema（各 data_prep 的产物、冷启动数据、RFT 数据都归一到它）：
  {"image": 落地路径或 null, "question": str,
   "thinking": str（可空；原语占位用数据侧写法 <box>/<point>）,
   "response": str, "source": 数据集名}

文本侧 = DSV4F tokenizer + DSv4 模板（render.py）；图像侧 = Kimi K3 image_processor
（processors.py）。assistant 段（thinking + response）参训，脚手架与图像位屏蔽。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from shensi.recipes.shensi_vl.common import processors as proc_mod
from shensi.recipes.shensi_vl.common import render, vl_tokens


def iter_jsonl(paths: list[str | Path], limit: int | None = None):
    n = 0
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
                    n += 1
                    if limit and n >= limit:
                        return


class VLDataset(torch.utils.data.Dataset):
    """jsonl → 训练样本。图像在 __getitem__ 里读（多 worker 下不堵主进程）。"""

    def __init__(self, jsonl_paths: list[str | Path], tokenizer, image_processor,
                 *, max_len: int = 8192, limit: int | None = None, with_image: bool = True):
        self.rows = list(iter_jsonl(jsonl_paths, limit))
        if not self.rows:
            raise SystemExit(f"[vl_data] 样本为空：{jsonl_paths}")
        self.tok = tokenizer
        self.iproc = image_processor
        self.max_len = max_len
        self.with_image = with_image
        self.img_token_id = vl_tokens.token_ids(tokenizer)[vl_tokens.IMAGE_TOKEN]

    def __len__(self) -> int:
        return len(self.rows)

    def _image_tokens(self, img: Image.Image) -> int:
        w, h = img.size
        return proc_mod.image_token_count(self.iproc, w, h, self.tok)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        messages = [
            {
                "role": "user",
                "content": render.build_user_content(
                    render.finalize_text(row["question"]),
                    1 if (self.with_image and row.get("image")) else 0,
                    trigger=row.get("trigger", True),
                ),
            }
        ]
        assistant_content = render.finalize_text(row.get("response") or "")
        thinking = row.get("thinking")
        image = None
        image_counts = []
        if self.with_image and row.get("image"):
            image = Image.open(row["image"]).convert("RGB")
            image_counts = [self._image_tokens(image)]
        if thinking:
            messages.append(
                {"role": "assistant", "reasoning_content": render.finalize_text(thinking),
                 "content": assistant_content}
            )
        else:
            messages.append({"role": "assistant", "content": assistant_content})
        enc = render.encode_sample(self.tok, messages, image_counts)
        if len(enc["input_ids"]) > self.max_len:  # 右截断（图像位在前面，不受影响）
            enc = {k: v[: self.max_len] for k, v in enc.items()}
        pixel_values = None
        if image is not None:
            pixel_values = proc_mod.preprocess_images(self.iproc, [image])
        return {**enc, "pixel_values": pixel_values}


def collate(batch: list[dict], pad_id: int) -> dict:
    """右 padding 拼 batch：input_ids / labels(-100) / attention_mask / image_positions / pixel_values。"""
    n = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    labels = torch.full((n, L), -100, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    img_pos = torch.zeros((n, L), dtype=torch.bool)
    for i, b in enumerate(batch):
        m = len(b["input_ids"])
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        lab = torch.tensor(
            [t if mk else -100 for t, mk in zip(b["input_ids"], b["loss_mask"])],
            dtype=torch.long,
        )
        input_ids[i, :m] = ids
        labels[i, :m] = lab
        attn[i, :m] = 1
        pos = torch.tensor(b["image_positions"], dtype=torch.long)
        img_pos[i, pos] = True
    pixels = None
    if batch[0].get("pixel_values") is not None:
        pixels = torch.cat([b["pixel_values"] for b in batch if b["pixel_values"] is not None], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "image_positions": img_pos,
        "pixel_values": pixels,
    }
