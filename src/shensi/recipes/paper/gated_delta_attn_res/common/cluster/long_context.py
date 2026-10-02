#!/usr/bin/env python3
"""长上下文能力测法：在长度 L 的上下文里放 K 个位置已知的键，让模型回答其中一个，逐长度档记录准确率并与随机基线对照。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[2]
EVAL = RECIPE / "stage4_eval"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="受控检索的多长度曲线")
    ap.add_argument("--model", required=True, help="HF 目录（export_hf 的产物）")
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--n", type=int, default=40, help="每个长度多少题")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--max-length", type=int, default=None, help="评分时的截断上限（默认跟随最长档）"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs_root = (
        Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
        / "shensi/runs/gated_delta_attn_res/cluster"
    )
    out_json = Path(args.out) if args.out else runs_root / "long_context.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    curves = []
    for ln in [x for x in args.lengths.split(",") if x]:
        length = int(ln)
        data = out_json.parent / f"dr_L{length}.jsonl"
        score = out_json.parent / f"score_L{length}.json"
        print(f"\n=== 长度 {length}：生成 {args.n} 题 → {data}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(EVAL / "make_depth_retrieval.py"),
                "--out",
                str(data),
                "--n",
                str(args.n),
                "--lengths",
                str(length),
                "--ks",
                args.ks,
                "--seed",
                "7",
                "--filler-random",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        cmd = [
            sys.executable,
            str(EVAL / "run_depth_retrieval.py"),
            "--model",
            args.model,
            "--data",
            str(data),
            "--device",
            args.device,
            "--out-json",
            str(score),
        ]
        if args.max_length:
            cmd += ["--max-length", str(args.max_length)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not score.is_file():
            tail = [x for x in (r.stdout + r.stderr).splitlines() if x.strip()][-5:]
            print("  评分失败：", "\n  ".join(tail))
            curves.append({"length": length, "error": tail})
            continue
        got = json.loads(score.read_text())
        v = got["verdict"]
        curves.append(
            {
                "length": length,
                "n": got["meta"]["data_n"],
                "pooled_acc": v["pooled_acc"],
                "chance": v["chance"],
                "wilson95": v["wilson95_acc"],
                "position_bias_index": v["position_bias_index"],
                "usable": v["usable"],
                "p_value": v["p_value"],
            }
        )
        print(
            f"  pooled_acc={v['pooled_acc']:.3f}（chance {v['chance']:.2f}，"
            f"Wilson95 {v['wilson95_acc'][0]:.3f}–{v['wilson95_acc'][1]:.3f}）"
            f"｜位置偏差 {v['position_bias_index']:.2f}｜usable={v['usable']}",
            flush=True,
        )

    print("\n长度        题数  命中率   Wilson95        位置偏差  usable")
    for c in curves:
        if "error" in c:
            print(f"{c['length']:<10} 评分失败")
            continue
        print(
            f"{c['length']:<10} {c['n']:<5} {c['pooled_acc']:.3f}   "
            f"{c['wilson95'][0]:.3f}–{c['wilson95'][1]:.3f}   "
            f"{c['position_bias_index']:.2f}      {c['usable']}"
        )
    out_json.write_text(
        json.dumps(
            {"model": args.model, "n_per_length": args.n, "curves": curves},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n[cluster] 曲线写 {out_json}")
    return 0 if all("error" not in c for c in curves) else 1


if __name__ == "__main__":
    raise SystemExit(main())
