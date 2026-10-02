#!/usr/bin/env python3
"""Specialized SFT 冷启动数据（论文 §2.4）。

四个任务族 → 两类原语：
  box（thinking with grounding）
    - coarse 计数：密集检测集（coco/crowdhuman/objects365 落地）按元数据**程序化**合成
      "意图 → 批量 grounding → 统计求和"三段思维链（论文用 MLLM 合成；这里程序化生成后
      过同一套严格校验：框与标注严格对齐、语法正确、与计数一致）；
    - 细粒度计数 / 空间推理：GQA 场景图（属性/空间关系约束）→ 带多跳 grounding 的思维链，
      并按论文构造"对象/关系不存在"的负样本（faithful refusal）；
  point（thinking with pointing）
    - 迷宫导航（common/synth_maze.py）、路径追踪（common/synth_trace.py）。

混合：70% 通用多模态（图文对/VQA）+ 30% 专项，按 blend 权重合成 sft_box.jsonl / sft_point.jsonl
（两份专家档各吃一份，论文 §2.5.1：分开训防止小数据下的模式冲突）。
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
from shensi.recipes.shensi_vl.common import synth_maze, synth_trace
from shensi.recipes.shensi_vl.stage0_pretrain import data_prep as dp0

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------- 计数（box）：密集检测集 → 三段式思维链 ----------------


def count_sample(row: dict, d: dict, rng: random.Random, img_dir: Path, tag: str) -> dict | None:
    got = dp0._img_wh(row, d)
    if got is None:
        return None
    img, w, h = got
    objs = row.get(d.get("objects_key", "objects")) or {}
    boxes = objs.get(d.get("bbox_key", "bbox")) or []
    labels_key = d.get("labels_key") or ("label" if "label" in objs else "category")
    labels = objs.get(labels_key) or []
    if not boxes or len(boxes) != len(labels):
        return None
    by_label: dict[str, list] = {}
    for b, lbl in zip(boxes, labels):
        by_label.setdefault(str(lbl), []).append(b)
    # 论文的过滤三准则：别太密 / 框要够大能看清 / 标注召回高（这里用元数据可算的前两条）
    cands = [
        (lbl, bs) for lbl, bs in by_label.items()
        if 3 <= len(bs) <= 30  # 别太密（batch grounding 一次框完不重复枚举）
    ]
    if not cands:
        return None
    target, sel = rng.choice(cands)
    xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in sel] if d.get("bbox_format", "xywh") == "xywh" else sel
    if any((x2 - x1) * (y2 - y1) / (w * h) < 0.001 for x1, y1, x2, y2 in xyxy):
        return None  # 框太小看不清
    norm = P.boxes_from_px(xyxy, w, h)
    count = len(norm)
    thinking = (
        f"1. **Deconstructing the query**\nThe user wants to count the number of {target} in the image.\n"
        f"2. **Sweeping the image for {target}**\n"
        f"Locating all of them at once: {P.render_box_primitive(target, norm)} — "
        f"this covers the {target} instances from left to right.\n"
        f"3. **Statistical summation**\nTallying the anchored boxes: {count}."
    )
    return {
        "task": "count_coarse",
        "image": dp0._save_image(img, img_dir, tag),
        "question": rng.choice([
            f"Count the number of {target} in this image.",
            f"How many {target} are there?",
        ]),
        "thinking": thinking,
        "response": f"There are {count} {target} in this image.",
        "ground_truth": {"count": count, "boxes": norm, "target": target},
        "source": d["name"],
    }


# ---------------- GQA 场景图：细粒度计数 / 空间推理（含负样本）----------------

SPATIAL_PREDS = {
    "left of": lambda a, b: a[0] < b[0],
    "right of": lambda a, b: a[0] > b[0],
    "above": lambda a, b: a[1] < b[1],
    "below": lambda a, b: a[1] > b[1],
}


def _gqa_objects(entry: dict) -> list[dict]:
    """GQA scene graph 的 objects 容器 → [{name, x,y,w,h, attributes}]（字段名宽容）。"""
    out = []
    objs = entry.get("objects") or {}
    if isinstance(objs, dict):
        it = objs.values()
    elif isinstance(objs, list):
        it = objs
    else:
        return out
    for o in it:
        box = o.get("bounding_box") or o.get("bbox") or o.get("box")
        if not box or len(box) < 4:
            continue
        out.append(
            {
                "name": str(o.get("name") or o.get("title") or "object"),
                "x": float(box[0]),
                "y": float(box[1]),
                "w": float(box[2]),
                "h": float(box[3]),
                "attributes": [str(a) for a in (o.get("attributes") or {}).values()]
                if isinstance(o.get("attributes"), dict)
                else [str(a) for a in (o.get("attributes") or [])],
            }
        )
    return out


def _img_of(entry: dict, d: dict):
    img = entry.get(d.get("image_key", "image"))
    if isinstance(img, dict) and img.get("bytes"):
        from io import BytesIO

        from PIL import Image

        return Image.open(BytesIO(img["bytes"])).convert("RGB")
    return None


def gqa_sample(entry: dict, d: dict, rng: random.Random, img_dir: Path, tag: str) -> dict | None:
    """从一条场景图记录出细粒度计数或空间推理题（带多跳 grounding 思维链；负样本穿插）。"""
    img = _img_of(entry, d)
    if img is None:
        return None
    objs = _gqa_objects(entry)
    if len(objs) < 3:
        return None
    w, h = img.width, img.height
    center = lambda o: [o["x"] + o["w"] / 2, o["y"] + o["h"] / 2]  # noqa: E731

    if rng.random() < 0.5:  # --- 细粒度计数：属性约束（白狗/左边的狗 这类） ---
        target = rng.choice([o for o in objs if o["attributes"]] or objs)
        attrs = target["attributes"][:2]
        phrase = " ".join(attrs + [target["name"]]) if attrs else target["name"]
        same_name = [o for o in objs if o["name"] == target["name"]]
        match = [o for o in same_name if all(a in o["attributes"] for a in attrs)]
        if not match:
            if rng.random() < 0.5:
                match, neg = [], True  # count=0 负样本（论文：增强幻觉鲁棒性）
            else:
                match, attrs, phrase = same_name, [], target["name"]  # 退化为裸类目计数
        else:
            neg = False
        pred_boxes = P.boxes_from_px(
            [[o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"]] for o in match], w, h
        )
        scan = [
            f"- I see {P.render_box_primitive(o['name'], P.boxes_from_px([[o['x'], o['y'], o['x'] + o['w'], o['y'] + o['h']]], w, h))} — "
            f"{'matches' if o in match else 'does not match'} the constraint {phrase}."
            for o in rng.sample(same_name, min(len(same_name), 6))
        ]
        thinking = (
            f"1. **What am I looking for**\n{phrase}: I must scan the scene and check each candidate "
            f"against the constraints.\n2. **Sequential scan**\n" + "\n".join(scan)
            + f"\n3. **Tally**\n{'No instance satisfies the constraints.' if neg else f'{len(match)} instance(s) qualify.'}"
        )
        return {
            "task": "count_fine",
            "image": dp0._save_image(img, img_dir, tag),
            "question": f"How many {phrase} are in the image?",
            "thinking": thinking,
            "response": (
                "Based on the constraints, there are none in the image." if neg
                else f"Based on the constraints, there are {len(match)} {phrase} in the image."
            ),
            "ground_truth": {"count": len(match), "boxes": pred_boxes, "target": phrase},
            "source": d["name"],
        }
    # --- 空间推理：定位两个对象 → 坐标比较 → 结论（正负样本） ---
    a, b = rng.sample(objs, 2)
    rel = rng.choice(list(SPATIAL_PREDS))
    holds = SPATIAL_PREDS[rel](center(a), center(b))
    if rng.random() < 0.3:  # 负样本：挑一个不成立的问法（faithful refusal）
        holds = not holds
    ba = P.boxes_from_px([[a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]]], w, h)
    bb = P.boxes_from_px([[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]], w, h)
    thinking = (
        f"1. **Analyzing the request**\nCheck whether the {a['name']} is {rel} the {b['name']}.\n"
        f"2. **Grounding both objects**\n"
        f"- {a['name']}: {P.render_box_primitive(a['name'], ba)} (center {center(a)}).\n"
        f"- {b['name']}: {P.render_box_primitive(b['name'], bb)} (center {center(b)}).\n"
        f"3. **Relational inference**\nComparing coordinates: the statement is "
        f"{'true' if holds else 'false'} based on the anchored positions."
    )
    return {
        "task": "spatial",
        "image": dp0._save_image(img, img_dir, tag),
        "question": f"Is the {a['name']} {rel} the {b['name']}?",
        "thinking": thinking,
        "response": f"Yes, the {a['name']} is {rel} the {b['name']}." if holds
        else f"No, the {a['name']} is not {rel} the {b['name']}.",
        "ground_truth": {"answer": holds, "boxes": ba + bb},
        "source": d["name"],
    }


# ---------------- 合成器（point 家族）----------------


def general_sample(row: dict, d: dict, rng: random.Random, img_dir: Path, tag: str) -> dict | None:
    """通用多模态档（70% 那部分）：图文对 / 通用 VQA，无思维链。"""
    got = dp0._img_wh(row, d)
    if got is None:
        return None
    img, _, _ = got
    text = row.get(d.get("text_key", "text")) or row.get(d.get("answer_key", "answer")) or ""
    question = row.get(d.get("question_key", "question")) or "Describe this image."
    if not isinstance(text, str) or len(text) < 4:
        return None
    return {
        "task": "general",
        "image": dp0._save_image(img, img_dir, tag),
        "question": question,
        "thinking": "",
        "response": text,
        "source": d["name"],
    }


def synth_point_family(rng: random.Random, out: Path, n_maze: int, n_trace: int, render: bool) -> list[dict]:
    rows = []
    mazer = synth_maze
    for i in range(n_maze):
        diff = rng.choice(list(mazer.DIFFICULTY))
        lo, hi = mazer.DIFFICULTY[diff]
        topo = rng.choice(mazer.TOPOLOGIES)
        gw = rng.randint(lo, hi)
        gh = rng.randint(3, 8) if topo == "ring" else rng.randint(lo, hi)
        maze = mazer.gen_maze(rng, rng.choice(mazer.ALGOS), topo, gw, gh)
        cells = list(maze["centers"])
        start, goal = cells[0], cells[-1]
        solvable = rng.random() < 0.7
        if not solvable:
            mazer.make_unsolvable(rng, maze, start, goal)
        thinking, path_cells = mazer.explore_narration(rng, maze, start, goal)
        row = {
            "task": "maze",
            "difficulty": diff,
            "question": "Is there a feasible way from the green marker to the red circle? "
                        "Explain the path if any. Output \\boxed{True} or \\boxed{False}.",
            "thinking": thinking,
            "response": (
                f"The maze is solvable. The verified path is: "
                f"{P.render_point_primitive([maze['centers'][c] for c in path_cells])}\n\\boxed{{True}}"
                if solvable else "No path exists.\n\\boxed{False}"
            ),
            "spec": P.serialize_maze({**maze, "start": start, "goal": goal, "solvable": solvable}),
            "source": "synth_maze",
        }
        if render:
            img_dir = out / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            fp = img_dir / f"maze_{i:07d}.png"
            mazer.render_maze(maze, start, goal, path_cells, rng.randint(384, 768), mazer.random_style(rng)).save(fp)
            row["image"] = str(fp)
        rows.append(row)
    for i in range(n_trace):
        diff = rng.choice(["easy", "normal", "hard", "nightmare"])
        row = synth_trace.synth_one(rng, diff)
        row.pop("curves", None)
        row.pop("style", None)
        row.setdefault("image", None)
        row["source"] = "synth_trace"
        rows.append(row)
    return rows


# ---------------- GQA 两份落地（图像 parquet + 场景图）合并 ----------------


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def join_gqa(root: Path, out_dir: Path, *, images_dir: str, graph_dir: str,
             id_key: str = "id", limit: int | None = None) -> Path:
    """lmms-lab/GQA（图像+QA，行键 imageId/question_id…）与场景图 parquet 按图像 id 合并成
    单份 jsonl（{"id","image","objects"}），gqa_sample 直接吃。id 字段名以 --discover 实测为准。"""
    import pyarrow.parquet as pq

    graphs: dict[str, dict] = {}
    for f in sorted((root / graph_dir).glob("**/*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                gid = str(row.get(id_key) or row.get("imageId") or row.get("image_id") or "")
                if gid and row.get("objects"):
                    graphs[gid] = row
    out_file = out_dir / f"{images_dir}__joined.jsonl"
    n = 0
    with open(out_file, "w", encoding="utf-8") as fh:
        for f in sorted((root / images_dir).glob("**/*.parquet")):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    gid = str(row.get(id_key) or row.get("imageId") or row.get("image_id") or "")
                    g = graphs.get(gid)
                    if g is None:
                        continue
                    img = row.get("image")
                    fh.write(
                        json.dumps(
                            {"id": gid, "image": img, "objects": g.get("objects")},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n += 1
                    if limit and n >= limit:
                        print(f"[sft_vl] join-gqa：{n} 条 → {out_file}")
                        return out_file
    print(f"[sft_vl] join-gqa：{n} 条 → {out_file}")
    return out_file


# ---------------- 主流程 ----------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage1_sft 冷启动数据")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--blend", default=None)
    ap.add_argument("--root", default=None, help="语料根，默认 $SHENSI_FS/datasets/llm/post-training")
    ap.add_argument("--out", default=None, help="产物目录，默认 <data>/stage1_sft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-render", action="store_true", help="合成迷宫/路径时不落图（冒烟）")
    ap.add_argument(
        "--join-gqa", action="store_true",
        help="只做 GQA 图像×场景图合并（--images-dir/--graph-dir），不产冷启动数据",
    )
    ap.add_argument("--images-dir", default="gqa-images")
    ap.add_argument("--graph-dir", default="gqa-scene-graph")
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    root = Path(args.root or paths["post"])
    out = Path(args.out or paths["data"] / "stage1_sft")
    if args.join_gqa:
        out.mkdir(parents=True, exist_ok=True)
        join_gqa(root, out, images_dir=args.images_dir, graph_dir=args.graph_dir, limit=args.limit)
        return 0
    spec = common.load_blend_spec(
        Path(args.blend) if args.blend
        else Path(__file__).parent / "config/data_prep/data_blend_raw.json"
    )
    datasets = [d for d in spec["datasets"] if not args.only or args.only in d["name"]]
    rng = random.Random(args.seed)

    if args.discover:
        print(f"[sft_vl] 语料根：{root}")
        for d in datasets:
            base = root / d["name"]
            files = [f for f in sorted(base.glob("**/*")) if f.suffix in (".parquet", ".json", ".jsonl")] if base.is_dir() else []
            cols = "<无文件>"
            if files and files[0].suffix == ".parquet":
                import pyarrow.parquet as pq

                cols = pq.ParquetFile(files[0]).schema_arrow.names
            elif files:
                cols = list(json.loads(files[0].read_text(encoding="utf-8").splitlines()[0]).keys())
            print(f"  {'✅' if files else '❌'} {d['name']:<28} family={d.get('family', '-'):<8} 列={cols}")
        return 0
    if not args.prepare:
        ap.error("至少给一个：--discover / --prepare")

    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list] = {"box": [], "point": [], "general": []}
    for d in datasets:
        family = d.get("family", "general")
        n_total = 0
        if d.get("source") == "synth":  # 迷宫/路径：程序化合成
            base_n = args.limit or 5000
            n_maze = int(d.get("n_maze", base_n))
            n_trace = int(d.get("n_trace", base_n))
            rows = synth_point_family(rng, out, n_maze, n_trace, not args.no_render)
            buckets["point"] += rows
            n_total = len(rows)
        else:
            ref = d.get("adapter_ref", d["name"])  # 复用别的落地目录（计数复用检测数据落地）
            base = root / ref
            files = (
                [f for f in sorted(base.glob("**/*.parquet"))] + [f for f in sorted(base.glob("**/*.jsonl"))]
                if base.is_dir()
                else []
            )
            if not files:
                msg = f"{d['name']}: 在 {root} 下没找到 parquet/jsonl（找的目录：{ref}）"
                if args.skip_missing:
                    print(f"[sft_vl] 跳过（--skip-missing）：{msg}")
                    continue
                raise SystemExit(f"{msg}，先 --discover（GQA 先跑 --join-gqa）")
            builder = {
                "count": count_sample,
                "gqa": gqa_sample,
                "general": general_sample,
            }[d.get("kind", "general")]
            for f in files:
                n = 0
                it = (
                    dp0.rows_of(f, args.limit)
                    if f.suffix == ".parquet"
                    else _iter_jsonl(f)
                )
                for row in it:
                    rec = builder(row, d, rng, out / "images", f"{d['name']}_{n:09d}")
                    if rec and rec.get("response"):
                        buckets[family].append(rec)
                        n += 1
                n_total += n
        print(f"[sft_vl] {d['name']} (family={family}): {n_total} 条")

    # 70/30 混合（论文 §2.5.1）：两份专家档 = 通用多模态 70% + 专项 30%（按条数配比）
    general = buckets["general"]
    rng.shuffle(general)
    for fam in ("box", "point"):
        spec_rows = buckets[fam]
        if not spec_rows:
            print(f"[sft_vl] family={fam} 没有专项样本（对应数据没落地？）——只写 general")
            target = general
        else:
            k = min(len(general), int(len(spec_rows) * 7 / 3))
            target = spec_rows + general[:k]
        rng.shuffle(target)
        tag = "box" if fam == "box" else "point"
        fp = out / f"sft_{tag}.jsonl"
        with open(fp, "w", encoding="utf-8") as fh:
            for r in target:
                r2 = {k: v for k, v in r.items() if k not in ("ground_truth", "spec")}
                fh.write(json.dumps(r2, ensure_ascii=False) + "\n")
        # 任务池（带 spec / ground_truth）：stage2_rl 的难度分层与奖励判分从这里读
        with open(out / f"{tag}_tasks.jsonl", "w", encoding="utf-8") as fh:
            for r in buckets[fam]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[sft_vl] family={fam}: SFT {len(target)} 条 → {fp}；任务池 {len(buckets[fam])} 条 → {tag}_tasks.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
