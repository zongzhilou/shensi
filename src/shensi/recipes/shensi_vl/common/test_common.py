"""test_train 公共件：tiny LM 定位 + 假数据（带图样本）+ 扩展 tokenizer。"""

from __future__ import annotations

import json
from pathlib import Path

from shensi.recipes.shensi_vl.common import primitives as P
from shensi.recipes.shensi_vl.common import vl_tokens

TINY_CANDIDATES = ("$SHENSI_FS/shensi/models/tiny-rl", "Qwen/Qwen3-0.6B")


def tiny_llm() -> str:
    """冒烟语言骨干：优先本地 tiny-rl（shensi 的 tiny_artifacts 产物），否则 hub 小模型。"""
    import os

    for cand in TINY_CANDIDATES:
        p = Path(os.path.expandvars(cand))
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
    return TINY_CANDIDATES[-1]


def make_fake_grounding(out_jsonl: Path, n: int = 8, with_images: bool = True) -> Path:
    """造 n 条假 grounding 样本（PIL 色块图 + 真实框标注），结构统一 schema。"""
    from PIL import Image, ImageDraw

    rng = __import__("random").Random(0)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out_jsonl.parent / "images_tiny"
    rows = []
    for i in range(n):
        img = Image.new("RGB", (256, 192), (240, 240, 240))
        dr = ImageDraw.Draw(img)
        boxes, labels = [], []
        for k in range(rng.randint(1, 3)):
            x1, y1 = rng.randint(4, 140), rng.randint(4, 100)
            bw, bh = rng.randint(30, 90), rng.randint(30, 80)
            color = rng.choice([(200, 60, 60), (60, 120, 200), (60, 160, 90)])
            dr.rectangle([x1, y1, x1 + bw, y1 + bh], fill=color)
            labels.append(f"block{color[0]}")
            boxes.append([x1, y1, x1 + bw, y1 + bh])
        target = labels[0]
        sel = [b for b, lbl in zip(boxes, labels) if lbl == target]
        rows.append(
            {
                "task": "grounding",
                "image": str(_save(img, img_dir / f"tiny_{i:03d}.jpg")) if with_images else None,
                "question": f"Locate {target} in this image and report its bounding box coordinates.",
                "thinking": "",
                "response": P.render_box_primitive(target, P.boxes_from_px(sel, img.width, img.height)),
                "source": "tiny_fake",
            }
        )
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_jsonl


def _save(img, fp: Path) -> Path:
    fp.parent.mkdir(parents=True, exist_ok=True)
    img.save(fp, quality=90)
    return fp


def tiny_tokenizer(llm_path: str, out_dir: Path) -> tuple[str, int]:
    """给 tiny LM 的 tokenizer 加扩展 token（冒烟专用副本）；返回 (目录, 词表长)。"""
    out = vl_tokens.extend_tokenizer(llm_path, out_dir)
    from transformers import AutoTokenizer

    return str(out), len(AutoTokenizer.from_pretrained(str(out)))
