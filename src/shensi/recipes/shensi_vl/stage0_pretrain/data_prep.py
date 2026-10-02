#!/usr/bin/env python3
"""预训练语料准备：落地的检测/点标注数据集 → 统一 schema 的 grounding jsonl。

适配三类数据（blend 里按数据集选 adapter 与字段名，`--discover` 打实际列）：
  detection  检测框（objects.bbox + 类别名）→ box 原语（Locate TARGET … 模板）
  points     点标注（PixMo-Points 类）      → point 原语（Help me find TARGET … 模板）
  caption    图文对（描述类）               → 纯文本 VQA（Describe this image.）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common
from shensi.recipes.shensi_vl.common import paths as vl_paths
from shensi.recipes.shensi_vl.common import primitives as P

BOX_TEMPLATES = [
    "Locate {t} in this image and report its bounding box coordinates.",
    "Find every instance of {t} and give me their bounding boxes, ordered from left to right.",
]
POINT_TEMPLATES = [
    "Help me find {t}. Give me the center point for each instance.",
    "Point out all {t} in this image with their center points.",
]


def rows_of(path: Path, limit: int | None):
    import pyarrow.parquet as pq

    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield row
            n += 1
            if limit and n >= limit:
                return


def _img_wh(row: dict, d: dict) -> tuple[object, int, int] | None:
    img = row.get(d.get("image_key", "image"))
    w = row.get(d.get("width_key", "width")) or 0
    h = row.get(d.get("height_key", "height")) or 0
    if isinstance(img, dict) and img.get("bytes"):  # HF parquet 里图通常是 {bytes, path}
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(img["bytes"])).convert("RGB")
        w = w or img.width
        h = h or img.height
    if not w or not h:
        return None
    return img, int(w), int(h)


def _save_image(img, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{name}.jpg"
    if not fp.exists():
        img.save(fp, quality=90)
    return str(fp)


def adapt_detection(row: dict, d: dict, rng: random.Random, img_dir: Path, tag: str):
    got = _img_wh(row, d)
    if got is None:
        return None
    img, w, h = got
    objs = row.get(d.get("objects_key", "objects")) or {}
    boxes = objs.get(d.get("bbox_key", "bbox")) or []
    labels_key = d.get("labels_key") or ("label" if "label" in objs else "category")
    labels = objs.get(labels_key) or []
    if not boxes or not labels or len(boxes) != len(labels):
        return None
    xyxy = []
    for b in boxes:
        if d.get("bbox_format", "xywh") == "xywh":
            b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        xyxy.append(b)
    target = str(rng.choice(labels))
    idx = [i for i, lbl in enumerate(labels) if str(lbl) == target]
    sel = [xyxy[i] for i in idx]
    if not sel:
        return None
    return {
        "task": "grounding",
        "image": _save_image(img, img_dir, tag),
        "question": rng.choice(BOX_TEMPLATES).format(t=target),
        "thinking": "",
        "response": P.render_box_primitive(target, P.boxes_from_px(sel, w, h)),
        "source": d["name"],
    }


def adapt_points(row: dict, d: dict, rng: random.Random, img_dir: Path, tag: str):
    got = _img_wh(row, d)
    if got is None:
        return None
    img, w, h = got
    pts = row.get(d.get("points_key", "points")) or []
    label = row.get(d.get("label_key", "label")) or row.get(d.get("labels_key", "labels"))
    flat = []
    for p in pts if isinstance(pts, list) else []:
        if isinstance(p, dict):
            flat.append([p.get("x", p.get("x_norm", 0)), p.get("y", p.get("y_norm", 0))])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            flat.append([p[0], p[1]])
    if not flat:
        return None
    # 点标注可能已是 0–1 归一：小于 1.01 就按归一口径放大
    if all(abs(v) <= 1.01 for p in flat for v in p):
        flat = [[v * w if i == 0 else v * h for i, v in enumerate(p)] for p in flat]
    target = str(label) if label and not isinstance(label, list) else str(label[0] if label else "objects")
    # 论文：point 任务不要求输出对象名，方便扩展成轨迹等抽象引用
    return {
        "task": "pointing",
        "image": _save_image(img, img_dir, tag),
        "question": rng.choice(POINT_TEMPLATES).format(t=target),
        "thinking": "",
        "response": P.render_point_primitive(P.points_from_px(flat, w, h)),
        "source": d["name"],
    }


def adapt_caption(row: dict, d: dict, rng: random.Random, img_dir: Path, tag: str):
    got = _img_wh(row, d)
    if got is None:
        return None
    img, _, _ = got
    text = row.get(d.get("text_key", "text"))
    if not isinstance(text, str) or len(text) < 8:
        return None
    return {
        "task": "caption",
        "image": _save_image(img, img_dir, tag),
        "question": rng.choice(["Describe this image.", "What is in this image?"]),
        "thinking": "",
        "response": text,
        "source": d["name"],
    }


ADAPTERS = {"detection": adapt_detection, "points": adapt_points, "caption": adapt_caption}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage0 语料准备")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--blend", default=None)
    ap.add_argument("--root", default=None, help="语料根，默认 $SHENSI_FS/datasets/llm/pre-training")
    ap.add_argument("--out", default=None, help="产物目录，默认 <data>/stage0_pretrain")
    ap.add_argument("--limit", type=int, default=None, help="每数据集最多取多少条（冒烟）")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    root = Path(args.root or paths["pre"])
    out = Path(args.out or paths["data"] / "stage0_pretrain")
    spec = common.load_blend_spec(
        Path(args.blend) if args.blend
        else Path(__file__).parent / "config/data_prep/data_blend_raw.json"
    )
    datasets = [d for d in spec["datasets"] if not args.only or args.only in d["name"]]
    if args.discover:
        print(f"[stage0] 语料根：{root}")
        for d in datasets:
            base = root / d["name"]
            files = [f for f in sorted(base.glob("**/*")) if f.suffix == ".parquet"] if base.is_dir() else []
            cols = "<无文件>"
            if files:
                import pyarrow.parquet as pq

                cols = pq.ParquetFile(files[0]).schema_arrow.names
            print(f"  {'✅' if files else '❌'} {d['name']:<40} adapter={d.get('adapter'):<10} 列={cols}")
        return 0
    if not args.prepare:
        ap.error("至少给一个：--discover / --prepare")

    rng = random.Random(args.seed)
    rows, img_dir = [], out / "images"
    for d in datasets:
        base_dir = root / d["name"]
        files = sorted(base_dir.glob("**/*.parquet")) if base_dir.is_dir() else []
        if not files:
            msg = f"{d['name']}: 在 {root} 下没找到 parquet"
            if args.skip_missing:
                print(f"[stage0] 跳过（--skip-missing）：{msg}")
                continue
            raise SystemExit(f"{msg}，先 download_manifest.json 落数据或跑 --discover")
        fn = ADAPTERS[d.get("adapter", "detection")]
        n = 0
        for f in files:
            for row in rows_of(f, args.limit):
                rec = fn(row, d, rng, img_dir, f"{d['name']}_{n:09d}")
                if rec:
                    rows.append(rec)
                    n += 1
        print(f"[stage0] {d['name']}: {n} 条")
    if not rows:
        raise SystemExit("[stage0] 一条都没解析出来：先 --discover 对字段名，blend 里改 adapter/keys")
    rng.shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pretrain.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stage0] {len(rows)} 条 → {out / 'pretrain.jsonl'}（图像在 {img_dir}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
