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

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(COMMON))

from kernels import whiten_fused, whiten_ns, whiten_triton  # noqa: E402

from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (  # noqa: E402
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
        try:
            ms = timed(fn, args.iters)
        except whiten_ns.NSNotConverged as exc:
            print(f"{name:<30} {'—':>8} {'—':>7} {'—':>11}   NS 不达容差：{exc}")
            continue
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

    # ---- 融合读：只比"读的那一段"（白化矩阵同源，保证是等价对比）----
    T = args.seq
    q = torch.randn(T, args.hidden, device="cuda")
    heads = 8
    ref_read = gc._depth_read(
        values, q, 1e-6, heads=heads, null=True, whiten="full", ridge=args.ridge
    )

    def eager_read():
        return gc._depth_read(
            values, q, 1e-6, heads=heads, null=True, whiten="full", ridge=args.ridge
        )

    def fused_read():
        return whiten_fused.fused_depth_read(
            values,
            q,
            1e-6,
            heads=heads,
            null=True,
            whiten="full",
            ridge=args.ridge,
            impl="eager_whiten",
        )

    def fused_apply_read():
        return whiten_fused.fused_depth_read(
            values,
            q,
            1e-6,
            heads=heads,
            null=True,
            whiten="full",
            ridge=args.ridge,
            impl="eager_whiten",
            fuse_apply=True,
        )

    ms_e = timed(eager_read, args.iters)
    ms_f = timed(fused_read, args.iters)
    ms_a = timed(fused_apply_read, args.iters)
    err_a = (
        (fused_apply_read() - ref_read).abs().amax() / ref_read.abs().amax().clamp_min(1e-12)
    ).item()
    err = ((fused_read() - ref_read).abs().amax() / ref_read.abs().amax().clamp_min(1e-12)).item()
    print(
        f"\n{'读的那一段（白化矩阵同源=等价对比）':<30} {'ms':>8}\n"
        f"{'参考实现（~15 个 torch 算子）':<30} {ms_e:>8.3f}\n"
        f"{'融合读（Triton，3 次 kernel）':<30} {ms_f:>8.3f}"
        f"   倍速 {ms_e / ms_f:.2f}×｜max 误差 {err:.2e}\n"
        f"{'融合读 + 融合应用（2 次 kernel）':<30} {ms_a:>8.3f}"
        f"   倍速 {ms_e / ms_a:.2f}×｜max 误差 {err_a:.2e}"
    )

    # ---- per_head 档：开关前后（上游默认档，逐头白化）----
    from kernels import whiten_per_head

    pfx = torch.randn(T, args.hidden, device="cuda")
    blk = torch.randn(T, 4, args.hidden, device="cuda")
    cfg = gc.GdarConfig(read_heads=heads, read_null=True, read_whiten="per_head").validated()
    mod = gc.AttentionResidual(args.hidden, cfg).cuda().float().eval()

    def per_head_eager():
        with torch.no_grad():
            return mod.read(pfx, blk)[0]

    ref_ph = per_head_eager()
    ms_e2 = timed(per_head_eager, args.iters, warmup=1)
    try:
        whiten_per_head.install()
        fast_ph = per_head_eager()
        ms_f2 = timed(per_head_eager, args.iters, warmup=1)
        err2 = ((fast_ph - ref_ph).abs().amax() / ref_ph.abs().amax().clamp_min(1e-12)).item()
        print(
            f"\n{'per_head 档的读（上游默认，白化逐头）':<30} {'ms':>8}\n"
            f"{'参考实现（逐头 eigh）':<30} {ms_e2:>8.3f}\n"
            f"{'Triton 逐头协方差 + NS':<30} {ms_f2:>8.3f}"
            f"   倍速 {ms_e2 / ms_f2:.2f}×｜max 误差 {err2:.2e}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n（per_head 档跳过：{type(exc).__name__}: {exc}）")
    finally:
        whiten_per_head.uninstall()

    # ---- 批量 eigh：39 次单算 vs 一次批量（要调用方把请求交出来，见 whiten_batched）----
    from kernels import whiten_batched

    ntok = max(64, args.seq // 4)  # 只看 eigh 吞吐；全量 39 个 [., S, H] 会吃掉 1 GB 显存
    items = [torch.randn(ntok, s, args.hidden, device="cuda") for s in (2, 3, 4, 5, 6) * 8][:39]
    ms_one = timed(
        lambda: [gc._whitening_transform(v, "full", args.ridge) for v in items], 3, warmup=1
    )
    ms_b = timed(lambda: whiten_batched.batched_whitening(items, "full", args.ridge), 3, warmup=1)
    samp = items[0]
    err3 = (
        (
            whiten_batched.batched_whitening([samp], "full", args.ridge)[0]
            - gc._whitening_transform(samp, "full", args.ridge)
        )
        .abs()
        .amax()
        .item()
    )
    print(
        f"\n{'批量白化（39 次请求）':<30} {'ms':>8}\n"
        f"{'逐个（参考实现）':<30} {ms_one:>8.3f}\n"
        f"{'一次批量 eigh':<30} {ms_b:>8.3f}   倍速 {ms_one / ms_b:.2f}×｜单项 max 误差 {err3:.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
