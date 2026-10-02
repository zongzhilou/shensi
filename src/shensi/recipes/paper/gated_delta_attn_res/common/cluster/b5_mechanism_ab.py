#!/usr/bin/env python3
"""B5(a)(b)：机制曲线上的主表 A/B 与多 seed 消融。

同一份数据顺序、同一个种子、只换 `--model-algo`，用固定步数比 loss（机制级对照，
不是质量结论）。本机在 220M 档上跑得动；集群把 `--geom geoms/qwen3_1p04b` 一换就是
同一个脚本。

    python cluster/b5_mechanism_ab.py --steps 150 \
        --arms qwen3_gdar_main,base --seeds 42 \
        --out $SHENSI_FS/shensi/runs/gated_delta_attn_res/b5_ab.json

多 seed（给出误差棒，判断臂间差是否过种子噪声）：

    python cluster/b5_mechanism_ab.py --steps 150 \
        --arms qwen3_gdar_main,qwen3_gdar_noladder --seeds 42,43,44 \
        --out $SHENSI_FS/shensi/runs/gated_delta_attn_res/b5_seeds.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

RECIPE = Path(__file__).resolve().parents[2]
REPO = Path(
    os.environ.get("SHENSI_ROOT") or RECIPE.parents[4]
).resolve()  # …/src/shensi/recipes/paper/gated_delta_attn_res → 仓库根
PY = sys.executable
TRAIN = RECIPE / "stage0_pretrain/stage1_pretrain/train.py"
LOSS_RE = re.compile(r"iteration\s+(\d+)/\s*(\d+).*?lm loss(?:\s+value)?:\s*([0-9.eE+-]+)")
MS_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")


def geom_shape(geom: str) -> tuple[int, int]:
    """按档名读出 (global_batch_size, seq_length)，把步数换算成 `--tokens`。"""
    path = RECIPE / "stage0_pretrain/stage1_pretrain/config" / f"{geom}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = (cfg.get("train") or {}).get("model") or {}
    return int(model.get("global_batch_size") or 1), int(model.get("seq_length") or 1)


def parse_log(log: Path) -> dict:
    if not log.is_file():
        return {}
    losses, ms = [], []
    for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LOSS_RE.search(ln)
        if m:
            losses.append((int(m.group(1)), float(m.group(3))))
        t = MS_RE.search(ln)
        if t:
            ms.append(float(t.group(1)))
    if not losses:
        return {}
    vals = [v for _, v in losses]
    tail = 10 if len(vals) >= 10 else len(vals)
    return {
        "steps_logged": len(vals),
        "last_iter": losses[-1][0],
        "loss_first10": statistics.mean(vals[:tail]),
        "loss_last10": statistics.mean(vals[-tail:]),
        "loss_min": min(vals),
        "ms_per_iter": statistics.median(ms) if ms else None,
        "loss_curve": [round(v, 4) for v in vals],
    }


def run_one(arm: str, seed: int, steps: int, geom: str, out_root: Path, extra: list[str]) -> dict:
    gb, seq = geom_shape(geom)
    tokens = steps * gb * seq
    exp_dir = out_root / f"{Path(geom).name}-{arm}-s{seed}"
    cmd = [
        PY,
        str(TRAIN),
        "--config",
        geom,
        "--model-algo",
        arm,
        "--tokens",
        str(tokens),
        "--set",
        f"experiment.seed={seed}",
        "--set",
        f"experiment.exp_dir={exp_dir}",
        "--no-early-stop",
        *extra,
    ]
    t0 = time.time()
    run = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=3600 * 12)
    wall = time.time() - t0
    log = exp_dir / "logs/host_0_localhost.output"
    got = parse_log(log)
    row = {
        "arm": arm,
        "seed": seed,
        "geom": geom,
        "steps": steps,
        "tokens": tokens,
        "rc": run.returncode,
        "wall_s": round(wall, 1),
        **got,
    }
    if run.returncode != 0:
        tail = [ln for ln in (run.stdout + run.stderr).splitlines() if ln.strip()][-6:]
        row["tail"] = tail
    print(
        f"  {arm:<22} seed={seed:<4} rc={run.returncode} "
        f"steps={got.get('last_iter', '-')} loss(末10)={got.get('loss_last10', float('nan')):.4f} "
        f"{row['wall_s']:.0f}s",
        flush=True,
    )
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="机制曲线上的主表 A/B 与多 seed 消融")
    ap.add_argument(
        "--geom", default="geoms/qwen3_0p22b", help="几何档（本机 220M，集群换 1p04b/0p6b）"
    )
    ap.add_argument("--arms", default="qwen3_gdar_main,base", help="逗号分隔的臂名")
    ap.add_argument("--seeds", default="42", help="逗号分隔的种子")
    ap.add_argument(
        "--steps", type=int, default=150, help="每臂步数（按档的 gbs×seq 换算 --tokens）"
    )
    ap.add_argument("--out", default=None, help="结果 JSON（默认写 runs/b5_mechanism.json）")
    ap.add_argument("--set", dest="extra", action="append", default=[], help="透传的点号覆写")
    args = ap.parse_args(argv)

    runs_root = (
        Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
        / "shensi/runs/gated_delta_attn_res"
    )
    out_root = runs_root / "b5"
    out_json = Path(args.out) if args.out else runs_root / "b5_mechanism.json"
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    extra_flags = [x for v in args.extra for x in ("--set", v)]

    print(
        f"[b5] 几何 {args.geom}｜臂 {arms}｜种子 {seeds}｜每臂 {args.steps} 步｜输出 {out_root}",
        flush=True,
    )
    rows = []
    for arm in arms:
        for seed in seeds:
            rows.append(run_one(arm, seed, args.steps, args.geom, out_root, extra_flags))

    print("\n臂                      种子数  末10步 loss（均值±标准差）  单步 ms  首10步 loss")
    summary = []
    for arm in arms:
        got = [r for r in rows if r["arm"] == arm and r.get("loss_last10") is not None]
        if not got:
            print(f"{arm:<22} （没有可用读数：rc={[r['rc'] for r in rows if r['arm'] == arm]}）")
            continue
        last = [r["loss_last10"] for r in got]
        first = [r["loss_first10"] for r in got]
        ms = [r["ms_per_iter"] for r in got if r.get("ms_per_iter")]
        entry = {
            "arm": arm,
            "n_seeds": len(got),
            "loss_last10_mean": statistics.mean(last),
            "loss_last10_std": statistics.stdev(last) if len(last) > 1 else 0.0,
            "loss_first10_mean": statistics.mean(first),
            "ms_per_iter_median": statistics.median(ms) if ms else None,
        }
        summary.append(entry)
        print(
            f"{arm:<22} {len(got):<6} {entry['loss_last10_mean']:.4f} ± {entry['loss_last10_std']:.4f}"
            f"        {entry['ms_per_iter_median'] or float('nan'):<8.0f} {entry['loss_first10_mean']:.4f}"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "geom": args.geom,
                "steps": args.steps,
                "arms": arms,
                "seeds": seeds,
                "runs": rows,
                "summary": summary,
                "note": "同一数据顺序与种子下的固定步数对照；样本语料有限时会重复多轮，只作机制级比较",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n[b5] 明细写 {out_json}")
    return 0 if all(r["rc"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
