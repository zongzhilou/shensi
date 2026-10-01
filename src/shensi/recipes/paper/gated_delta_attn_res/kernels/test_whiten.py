#!/usr/bin/env python3
"""白化内核的闸门：协方差、逆平方根、以及"换进连接模块后读出来的东西一样"。

    python kernels/test_whiten.py

四类检查（每条都打印 max|Δ| 与其相对量）：
  1. Triton 协方差 vs torch（ieee 档要求 ~1e-6，tf32 档只要求 ~1e-3）
  2. NS 逆平方根 vs eigh（含条件数差的输入，如实报误差）
  3. 白化变换 vs 参考实现（full / diag / per_head 三档）
  4. 端到端：把变换换进 `gdar_connection` 后，`AttentionResidual.read` 的输出与参考一致，
     并且 `uninstall()` 之后逐位回到参考实现
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

RECIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
sys.path.insert(0, str(RECIPE))

from kernels import whiten_ns, whiten_triton  # noqa: E402

from shensi.recipes.paper.gated_delta_attn_res.models.megatron import (  # noqa: E402
    gdar_connection as gc,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILS: list[str] = []


def report(name: str, ok: bool, extra: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().amax() / b.abs().amax().clamp_min(1e-12)).item()


def main() -> int:
    torch.manual_seed(0)
    print(f"设备 {DEV}｜Triton {'可用' if whiten_triton.HAS_TRITON else '不可用（回退 torch）'}")

    # 真实一点的输入：连接里的 state 过 RMSNorm，尺度 ~1
    n, h, heads = 2048 * 5, 512, 8
    S = torch.randn(n, h, device=DEV, dtype=torch.float32) * 0.7 + 0.3

    # 1. 协方差
    ref_cov = (S.transpose(0, 1) @ S) / S.shape[0] + 1e-3 * torch.eye(h, device=DEV)
    got = whiten_triton.cov_symmetric(S, 1e-3, precision="ieee")
    e = rel(got, ref_cov)
    report("Triton 协方差（ieee）", e < 5e-6, f"max rel = {e:.2e}")
    got_tf32 = whiten_triton.cov_symmetric(S, 1e-3, precision="tf32")
    e32 = rel(got_tf32, ref_cov)
    report("Triton 协方差（tf32）", e32 < 5e-3, f"max rel = {e32:.2e}")

    V = S.reshape(n, heads, h // heads)
    per_ref = torch.einsum("nhd,nhe->hde", V, V) / n
    sc_ref = per_ref.diagonal(dim1=-2, dim2=-1).mean(-1)
    per_ref.diagonal(dim1=-2, dim2=-1).add_(
        (max(h // heads, n) * torch.finfo(per_ref.dtype).eps * sc_ref).unsqueeze(-1)
    )
    per_got = whiten_triton.cov_per_head(S, heads, 1e-3, precision="ieee")
    e = rel(per_got, per_ref)
    report("Triton 逐头协方差（ieee）", e < 5e-6, f"max rel = {e:.2e}")

    # 2. 逆平方根
    for ridge, label in ((1e-3, "ridge=1e-3（论文档）"), (1e-6, "ridge=1e-6（病态档）")):
        A = (S.transpose(0, 1) @ S) / n + ridge * torch.eye(h, device=DEV)
        evals, evecs = torch.linalg.eigh(A)
        w_ref = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)
        w_ns = whiten_ns.inv_sqrt_ns(A)
        e = rel(w_ns, w_ref)
        res = (w_ns @ A @ w_ns - torch.eye(h, device=DEV)).abs().amax().item()
        report(f"NS 逆平方根 {label}", e < 5e-4, f"max rel = {e:.2e}｜残差|WAW−I| = {res:.2e}")

    # 3. 白化变换 vs 参考实现（三档）
    values = torch.randn(2048, 5, h, device=DEV)
    for mode in ("diag", "full"):
        r = gc._whitening_transform(values, mode, 1e-3)
        g = whiten_ns.whitening_transform_ns(values, mode, 1e-3)
        e = rel(g, r)
        report(f"变换 vs 参考（{mode}）", e < 5e-4, f"max rel = {e:.2e}")
    if whiten_triton.HAS_TRITON:
        r = gc._whitening_transform(values, "full", 1e-3)
        for prec, tol in (("ieee", 5e-4), ("tf32", 5e-3)):
            g = whiten_triton.whitening_transform_triton(values, "full", 1e-3, precision=prec)
            e = rel(g, r)
            report(f"变换 vs 参考（full/triton-{prec}）", e < tol, f"max rel = {e:.2e}")
    w_h, ridge_h = whiten_triton.whitening_transform_per_head(values, heads)
    Vv = values.reshape(-1, heads, h // heads).float()
    cov = torch.einsum("nhd,nhe->hde", Vv, Vv) / Vv.shape[0]
    sc = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
    cov.diagonal(dim1=-2, dim2=-1).add_(
        (max(h // heads, Vv.shape[0]) * torch.finfo(cov.dtype).eps * sc).unsqueeze(-1)
    )
    ev, evec = torch.linalg.eigh(cov)
    wh_ref = evec @ torch.diag_embed(torch.rsqrt(ev.clamp_min(1e-12))) @ evec.transpose(-1, -2)
    e = rel(w_h, wh_ref)
    report("逐头白化 vs 参考", e < 5e-4, f"max rel = {e:.2e}｜ridge_h 形状 {tuple(ridge_h.shape)}")

    # 4. 端到端：换进连接模块后读出来一样；卸掉后逐位一致
    cfg = gc.GdarConfig(
        read_heads=heads, read_null=True, read_whiten="full", read_ridge=1e-3, block_size=None
    ).validated()
    mod = gc.AttentionResidual(256, cfg).to(DEV).float().eval()
    T = 64
    # 各向异性输入：逐通道量级差 4 个数量级，白化才真的在做功
    # （各向同性时 W ≈ c·I，两个实现会给出"看起来逐位相同"的假象）
    scale = torch.logspace(-2, 2, 256, device=DEV)
    prefix = torch.randn(T, 256, device=DEV) * scale
    blocks = torch.randn(T, 4, 256, device=DEV) * scale
    with torch.no_grad():
        ref_out, _ = mod.read(prefix, blocks)
    swapped = whiten_ns.install()
    with torch.no_grad():
        ns_out, _ = mod.read(prefix, blocks)
    e = rel(ns_out, ref_out)
    report(
        "端到端 read（NS 换入）",
        swapped and e < 5e-3,
        f"max rel = {e:.2e}（各向异性输入；来源数小时 softmax 近打平，W 的 1e-5 级差异会被放大）",
    )
    whiten_ns.uninstall()
    with torch.no_grad():
        back_out, _ = mod.read(prefix, blocks)
    report("卸掉后逐位回参考", bool((back_out == ref_out).all()), "逐位相等")

    print(f"\n{'全部闸门通过 ✓' if not FAILS else '失败：' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
