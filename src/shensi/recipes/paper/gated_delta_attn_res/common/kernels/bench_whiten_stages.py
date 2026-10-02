#!/usr/bin/env python3
"""白化各阶段的分解计时（协方差 / 逆平方根 / 应用）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))


def timed(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def stages_full(S: torch.Tensor, Q: torch.Tensor, ridge: float) -> dict:
    d = S.shape[-1]
    eye = torch.eye(d, device=S.device, dtype=torch.float32)

    def cov():
        return (S.transpose(0, 1) @ S) / S.shape[0] + ridge * eye

    A = cov()
    out = {}
    out["cov（GEMM）"] = timed(cov)
    out["eigh（LAPACK）"] = timed(lambda: torch.linalg.eigh(A))
    evals, evecs = torch.linalg.eigh(A)

    def form():
        return evecs @ torch.diag(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)

    out["组装 W（2×GEMM）"] = timed(form)
    W = form()

    def apply():
        return S @ W, Q @ W

    out["应用（2×GEMM）"] = timed(apply)
    out["合计"] = sum(out.values())
    return out


def stages_per_head(S: torch.Tensor, Q: torch.Tensor, heads: int, eps: float) -> dict:
    n, d = S.shape
    dh = d // heads
    V = S.reshape(n, heads, dh)
    out = {}

    def cov():
        c = torch.einsum("nhd,nhe->hde", V, V) / n
        c.diagonal(dim1=-2, dim2=-1).add_(eps)
        return c

    A = cov()
    out["cov（逐头 einsum）"] = timed(cov)
    out["eigh（逐头 batched）"] = timed(lambda: torch.linalg.eigh(A))
    evals, evecs = torch.linalg.eigh(A)

    def form():
        return evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(eps))) @ evecs.transpose(-1, -2)

    out["组装 W（逐头）"] = timed(form)
    W = form()

    def apply():
        Ss = torch.einsum("nhd,hde->nhe", V, W).reshape(n, d)
        Qs = torch.einsum("nhd,hde->nhe", Q.reshape(Q.shape[0], heads, dh), W).reshape(
            Q.shape[0], d
        )
        return Ss, Qs

    out["应用（逐头）"] = timed(apply)
    out["合计"] = sum(out.values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="白化分段计时")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--sources", type=int, default=5, help="B+1（主行 B=4）")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0e-3)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    n = args.seq * args.sources
    S = torch.randn(n, args.hidden, device="cuda", dtype=torch.float32)
    Q = torch.randn(args.seq, args.hidden, device="cuda", dtype=torch.float32)
    print(
        f"几何：values [{n}, {args.hidden}]（seq {args.seq} × {args.sources} 来源）"
        f"，heads {args.heads}，fp32，中位 ms"
    )
    full = stages_full(S, Q, args.ridge)
    per = stages_per_head(S, Q, args.heads, 1e-3)
    print(f"\n{'阶段':<22} {'full（全矩阵）':>14} {'per_head（逐头）':>16}")
    for k in ("cov（GEMM）", "cov（逐头 einsum）"):
        pass
    keys = [
        ("cov", "cov（GEMM）", "cov（逐头 einsum）"),
        ("eigh", "eigh（LAPACK）", "eigh（逐头 batched）"),
        ("组装 W", "组装 W（2×GEMM）", "组装 W（逐头）"),
        ("应用", "应用（2×GEMM）", "应用（逐头）"),
        ("合计", "合计", "合计"),
    ]
    for label, kf, kp in keys:
        print(f"{label:<22} {full[kf]:>14.3f} {per[kp]:>16.3f}")
    share = full["eigh（LAPACK）"] / full["合计"]
    print(
        f"\nfull 里 eigh 占 {share * 100:.0f}%、GEMM 三步占 {(1 - share) * 100:.0f}%；"
        f"per_head 的 eigh 占 {per['eigh（逐头 batched）'] / per['合计'] * 100:.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
