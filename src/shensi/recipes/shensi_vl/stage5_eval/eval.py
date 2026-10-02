#!/usr/bin/env python3
"""评测入口：HF generate 本地判分（不走 vLLM——自定义 VL 架构没注册进 vLLM）。

套件（对齐论文 §3.2）：
  counting    Pixmo-Count（官方 test 划分）+ 自建 DS_Finegrained_Counting（GQA 出题）
  spatial     GQA 关系题（自建 DS_Spatial_Reasoning 口径：true/false）
  maze        DS_Maze_Navigation：合成 2000 题（可解/不可解 × 难度），判 \\boxed{True/False}
  trace       DS_Path_Tracing：合成 2000 题，判 \\boxed{端点标签}
模型目录用训练产物；指标打 summary.json（EM / ACC）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shensi import runtime  # noqa: F401,E402
from shensi.recipes.shensi_vl.common import paths as vl_paths  # noqa: E402
from shensi.recipes.shensi_vl.common import synth_maze, synth_trace  # noqa: E402
from shensi.recipes.shensi_vl.common.train import train_loop  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import grpo  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        yield from fh


def load_model(model_dir: str, tokenizer_dir: str, processor_path: str):
    cfg = {"train": {"model": {"tokenizer_dir": tokenizer_dir, "llm_path": model_dir,
                               "vision_path": model_dir, "processor_path": processor_path}}}
    tok = train_loop.build_tokenizer(cfg)
    model = train_loop.make_model(cfg, 0).eval()
    iproc = proc_mod_loader(processor_path)
    return model, tok, iproc


def proc_mod_loader(processor_path: str):
    from shensi.recipes.shensi_vl.common import processors as proc_mod

    return proc_mod.load_image_processor(processor_path)


@torch.no_grad()
def answer(model, tok, iproc, question: str, image_path: str | None,
           max_new_tokens: int = 1024) -> str:
    prompt = grpo.encode_prompt(tok, iproc, {"image": image_path, "question": question})
    pb = grpo.to_device(grpo.batch_prompts(tok, [prompt]), model.device)
    gen = model.generate(pb["input_ids"], pb["attention_mask"], pb["pixel_values"],
                         pb["image_positions"], max_new_tokens=max_new_tokens,
                         temperature=0.0, do_sample=False)
    return tok.decode(gen[0], skip_special_tokens=True)


def boxed(sol: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", sol or "")
    return m[-1].strip() if m else ""


def eval_synth(model, tok, iproc, kind: str, n: int, seed: int, tmp: Path) -> dict:
    """DS_Maze_Navigation / DS_Path_Tracing：合成即评测（同分布不同种子）。"""
    rng = __import__("random").Random(seed)
    hit = 0
    for i in range(n):
        if kind == "maze":
            diff = rng.choice(list(synth_maze.DIFFICULTY))
            lo, hi = synth_maze.DIFFICULTY[diff]
            topo = rng.choice(synth_maze.TOPOLOGIES)
            gw = rng.randint(lo, hi)
            gh = rng.randint(3, 8) if topo == "ring" else rng.randint(lo, hi)
            maze = synth_maze.gen_maze(rng, rng.choice(synth_maze.ALGOS), topo, gw, gh)
            cells = list(maze["centers"])
            start, goal = cells[0], cells[-1]
            solvable = rng.random() < 0.5
            if not solvable:
                synth_maze.make_unsolvable(rng, maze, start, goal)
            fp = tmp / f"eval_maze_{i}.png"
            synth_maze.render_maze(maze, start, goal, [], 640, synth_maze.random_style(rng)).save(fp)
            q = ("Is there a feasible way from the green marker to the red circle? "
                 "Output \\boxed{True} or \\boxed{False}.")
            pred = boxed(answer(model, tok, iproc, q, str(fp))).lower()
            hit += int(pred == ("true" if solvable else "false"))
        else:
            row = synth_trace.synth_one(rng, rng.choice(["easy", "normal", "hard"]))
            fp = tmp / f"eval_trace_{i}.png"
            synth_trace.render_trace(row.pop("curves"), 640, row.pop("style")).save(fp)
            label = row["spec"]["labels"][row["spec"]["target"]]
            pred = boxed(answer(model, tok, iproc, row["question"], str(fp)))
            hit += int(pred == label)
        if (i + 1) % 20 == 0:
            print(f"[eval:{kind}] {i + 1}/{n} acc={hit / (i + 1):.3f}", flush=True)
    return {"n": n, "acc": hit / max(1, n)}


def eval_counting(model, tok, iproc, pool: Path, n: int) -> dict:
    """Pixmo-Count 落地数据（question/answer 列）→ EM（答案里最后一个数）。"""
    rows = []
    for f in sorted(pool.glob("**/*.parquet")) + sorted(pool.glob("**/*.jsonl")):
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            for b in pq.ParquetFile(f).iter_batches(batch_size=128):
                rows += b.to_pylist()
        else:
            rows += [json.loads(line) for line in _read_jsonl(f) if line.strip()]
        if len(rows) >= n:
            break
    rows = rows[:n]
    hit = 0
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("query") or ""
        want = str(r.get("answer") or "")
        img = r.get("image")
        fp = None
        if isinstance(img, dict) and img.get("bytes"):
            from io import BytesIO

            from PIL import Image

            fp = pool / f"_eval_{i}.png"
            Image.open(BytesIO(img["bytes"])).convert("RGB").save(fp)
        elif isinstance(img, str):
            fp = Path(img)
        sol = answer(model, tok, iproc, q, str(fp) if fp and Path(fp).exists() else None)
        got = re.findall(r"-?\d+", boxed(sol) or sol)
        hit += int(bool(got) and got[-1] == want)
        if (i + 1) % 20 == 0:
            print(f"[eval:count] {i + 1}/{len(rows)} em={hit / (i + 1):.3f}", flush=True)
    return {"n": len(rows), "em": hit / max(1, len(rows))}


def main() -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage5_eval")
    ap.add_argument("--model", default=None, help="模型目录（默认 stage4_opd final）")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--suite", default="synth", choices=("synth", "count", "all"))
    ap.add_argument("--n", type=int, default=200, help="每套件题数（论文：DS_* 各 2000）")
    ap.add_argument("--count-pool", default=None, help="Pixmo-Count 落地目录")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = vl_paths.env_paths()
    model_dir = args.model or str(Path(paths["ckpt"]) / "stage4_opd/final")
    tokenizer_dir = args.tokenizer or paths["vl_tokenizer"]
    model, tok, iproc = load_model(model_dir, tokenizer_dir, paths["processor"])
    tmp = Path(paths["data"]) / "stage5_eval/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    summary = {"model": model_dir}
    if args.suite in ("synth", "all"):
        summary["DS_Maze_Navigation"] = eval_synth(model, tok, iproc, "maze", args.n, args.seed, tmp)
        summary["DS_Path_Tracing"] = eval_synth(model, tok, iproc, "trace", args.n, args.seed + 1, tmp)
    if args.suite in ("count", "all"):
        pool = Path(args.count_pool or paths["post"] / "pixmo-count")
        if pool.is_dir():
            summary["Pixmo-Count"] = eval_counting(model, tok, iproc, pool, args.n)
        else:
            print(f"[eval] Pixmo-Count 未落地（{pool}），跳过")
    out = Path(args.out or Path(paths["data"]) / "stage5_eval/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval] summary → {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
