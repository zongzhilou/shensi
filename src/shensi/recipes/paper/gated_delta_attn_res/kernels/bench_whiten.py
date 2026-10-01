#!/usr/bin/env python3
"""白化各种实现的计时对比（本目录 README 表里那些数的出处）。

    python kernels/bench_whiten.py --seq 2048 --sources 5 --hidden 1024

对每档给出：变换耗时（中位 ms）、相对参考实现的 max 误差、以及**折算到整步**的收益
（用 `train/bench_connection.py` 量出的口径：主行 351.0 ms、关白化 129.7 ms ⇒ 白化占
整步 221.0/351.0 ≈ 63%）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

RECIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
sys.path.insert(0, str(RECIPE))

from kernels import whiten_ns, whiten_triton  # noqa: E402

from shensi.recipes.paper.gated_delta_attn_res.models.megatron import (  # noqa: E402
    gdar_connection as gc,
)

WHITEN_SHARE = 221.0 / 351.0  # 白化在整步里占的比例（bench_connection 实测）


def timed(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="白化实现的计时对比")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--sources", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ridge", type=float, default=1.0e-3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    values = torch.randn(args.seq, args.sources, args.hidden, device="cuda")
    print(
        f"几何：values [{args.seq}, {args.sources}, {args.hidden}]｜ridge {args.ridge}｜"
        f"每档 {args.iters} 次取中位"
    )
    ref = gc._whitening_transform(values, "full", args.ridge)

    rows = [("eigh（参考实现）", lambda: gc._whitening_transform(values, "full", args.ridge), None)]
    rows.append(
        (
            "NS（无 LAPACK）",
            lambda: whiten_ns.whitening_transform_ns(values, "full", args.ridge),
            ref,
        )
    )
    if whiten_triton.HAS_TRITON:
        rows.append(
            (
                "Triton 协方差 + NS（ieee）",
                lambda: whiten_triton.whitening_transform_triton(
                    values, "full", args.ridge, "ieee"
                ),
                ref,
            )
        )
        rows.append(
            (
                "Triton 协方差 + NS（tf32）",
                lambda: whiten_triton.whitening_transform_triton(
                    values, "full", args.ridge, "tf32"
                ),
                ref,
            )
        )
    else:
        print("（没有 Triton，跳过两档内核）")

    base_ms = None
    print(f"\n{'实现':<30} {'ms':>8} {'倍速':>7} {'max 误差':>11} {'折算整步节省':>13}")
    out = []
    for name, fn, cmp in rows:
        ms = timed(fn, args.iters)
        base_ms = base_ms or ms
        err = (
            "-"
            if cmp is None
            else f"{((fn() - cmp).abs().amax() / cmp.abs().amax().clamp_min(1e-12)).item():.2e}"
        )
        saving = (1 - ms / base_ms) * WHITEN_SHARE * 100
        print(f"{name:<30} {ms:>8.3f} {base_ms / ms:>6.2f}× {err:>11} {saving:>12.1f}%")
        out.append((name, ms))

    print(
        f"\n口径：白化占整步 {WHITEN_SHARE * 100:.0f}%（train/bench_connection.py：主行 351.0 ms、"
        "关白化 129.7 ms）；『折算整步节省』= 变换省下的比例 × 这个占比。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
