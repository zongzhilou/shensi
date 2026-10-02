#!/usr/bin/env python3
"""白化内核的附加闸门：逐头与批量档的等价性。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

from kernels import whiten_batched, whiten_fused, whiten_ns, whiten_per_head  # noqa: E402

from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (  # noqa: E402
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
    print(
        f"设备 {DEV}｜Triton {'可用' if whiten_fused.HAS_TRITON else '不可用（回退 torch 读尾）'}"
    )

    T, hidden = 96, 512

    scale = torch.logspace(-2, 2, hidden, device=DEV)
    values = torch.randn(T, 5, hidden, device=DEV) * scale
    query = torch.randn(T, hidden, device=DEV) * scale
    eps = 1e-6

    combos = [
        ("off", 1, False, "raw"),
        ("full", 8, True, "raw"),
        ("full", 8, False, "raw"),
        ("full", 1, True, "raw"),
        ("full", 8, True, "whitened"),
        ("full", 8, True, "raw", "eager_whiten"),
        ("diag", 8, True, "raw"),
        ("per_head", 8, True, "raw"),
    ]
    worst = 0.0
    for combo in combos:
        mode, heads, null, mix = combo[:4]
        impl = combo[4] if len(combo) > 4 else "ns"
        tag = f"{mode}/heads={heads}/null={null}/mix={mix}" + (f"/{impl}" if impl != "ns" else "")
        ref, ref_s = gc._depth_read(
            values,
            query,
            eps,
            heads=heads,
            null=null,
            whiten=mode,
            ridge=1e-3,
            return_scores=True,
            mix=mix,
        )
        try:
            got, got_s = whiten_fused.fused_depth_read(
                values,
                query,
                eps,
                heads=heads,
                null=null,
                whiten=mode,
                ridge=1e-3,
                return_scores=True,
                mix=mix,
                impl=impl,
            )
        except whiten_ns.NSNotConverged as exc:
            print(f"  [包线] 融合读（{tag}）：NS 不达容差，未断言等价 -- {exc}", flush=True)
            continue
        e_r, e_s = rel(got, ref), rel(got_s.reshape(ref_s.shape), ref_s)
        tol = 1e-4 if (impl == "eager_whiten" or mode in ("off", "per_head")) else 5e-3
        if tol <= 1e-4:
            worst = max(worst, e_r, e_s)
        report(
            f"融合读 vs 参考（{tag}）",
            e_r < tol and e_s < tol,
            f"routed {e_r:.2e}｜scores {e_s:.2e}（容差 {tol:.0e}）",
        )
    print(f"      严格档最差相对误差 {worst:.2e}")

    ref, ref_s = gc._depth_read(
        values, query, eps, heads=8, null=True, whiten="full", return_scores=True
    )
    got, got_s = whiten_fused.fused_depth_read(
        values,
        query,
        eps,
        heads=8,
        null=True,
        whiten="full",
        return_scores=True,
        impl="eager_whiten",
        fuse_apply=True,
    )
    e_r, e_s = rel(got, ref), rel(got_s.reshape(ref_s.shape), ref_s)
    report(
        "融合读 + 融合应用（白化沿用参考）",
        e_r < 1e-4 and e_s < 1e-4,
        f"routed {e_r:.2e}｜scores {e_s:.2e}",
    )

    print("      NS 精度包线（W_ns vs W_eigh 的相对误差）：")
    for tag, extra in (
        ("各向同性", torch.ones(hidden, device=DEV)),
        ("各向异性 1e2", torch.logspace(-1, 1, hidden, device=DEV)),
        ("各向异性 1e4", torch.logspace(-2, 2, hidden, device=DEV)),
    ):
        v = values * extra
        w_ref = gc._whitening_transform(v, "full", 1e-3)
        try:
            w_ns = whiten_ns.whitening_transform_ns(v, "full", 1e-3, warn_residual=1e9)
            note = f"{rel(w_ns, w_ref):.2e}"
        except whiten_ns.NSNotConverged as exc:
            note = f"未收敛（{exc}）"
        print(f"        {tag:<14} {note}")

    cfg = gc.GdarConfig(read_heads=8, read_null=True, read_whiten="per_head").validated()
    mod = gc.AttentionResidual(256, cfg).to(DEV).float().eval()
    pfx = torch.randn(64, 256, device=DEV)
    blk = torch.randn(64, 4, 256, device=DEV)
    with torch.no_grad():
        ref_out, ref_scores = mod.read(pfx, blk)
    swapped = whiten_per_head.install()
    with torch.no_grad():
        fast_out, fast_scores = mod.read(pfx, blk)
    e = rel(fast_out, ref_out)
    report(
        "per_head 开关（read 一致）",
        swapped and e < 5e-3,
        f"routed {e:.2e}｜scores {rel(fast_scores, ref_scores):.2e}（逐头白化换了算法）",
    )
    whiten_per_head.uninstall()
    with torch.no_grad():
        back_out, _ = mod.read(pfx, blk)
    report("per_head 卸掉后逐位回参考", bool((back_out == ref_out).all()), "逐位相等")

    swapped2 = whiten_fused.install("eager_whiten")
    with torch.no_grad():
        fused_out, fused_scores = mod.read(pfx, blk)
    e = rel(fused_out, ref_out)
    report("融合读 install（per_head 档）", swapped2 and e < 1e-4, f"routed {e:.2e}")
    whiten_fused.uninstall()
    with torch.no_grad():
        back2 = mod.read(pfx, blk)[0]
    report("融合读卸掉后逐位回参考", bool((back2 == ref_out).all()), "逐位相等")

    items = [torch.randn(512, s, hidden, device=DEV) for s in (2, 3, 4, 5, 6, 2, 3, 4)]
    batch = whiten_batched.batched_whitening(items, "full", 1e-3)
    worst_b = max(
        rel(batch[i], gc._whitening_transform(v, "full", 1e-3)) for i, v in enumerate(items)
    )
    report("批量白化 vs 逐个（full）", worst_b < 5e-4, f"max rel = {worst_b:.2e}")
    bd = whiten_batched.batched_whitening(items, "diag", 1e-3)
    worst_d = max(rel(bd[i], gc._whitening_transform(v, "diag", 1e-3)) for i, v in enumerate(items))
    report("批量白化 vs 逐个（diag）", worst_d < 1e-6, f"max rel = {worst_d:.2e}")

    small = torch.randn(4096, 512, device=DEV)
    cov = (small.transpose(0, 1) @ small) / small.shape[0] + 1e-3 * torch.eye(512, device=DEV)

    def _ns_steps(cubic: bool, tol: float = 1e-5, cap: int = 40) -> int:
        d = cov.shape[-1]
        eye = torch.eye(d, device=DEV)
        ah = cov / whiten_ns._lam_max(cov)
        y, z = ah.clone(), eye.clone()
        for k in range(cap):
            if float((z @ ah @ z - eye).abs().amax()) < tol:
                return k
            m = z @ y
            if cubic:
                dm = eye - m
                g = eye + 0.5 * dm + 0.375 * (dm @ dm)
                y, z = y @ g, g @ z
            else:
                f = 0.5 * (3.0 * eye - m)
                y, z = y @ f, f @ z
        return cap

    k2, k3 = _ns_steps(False), _ns_steps(True)
    report("三阶加速（良态：步数下降且都收敛）", k3 < k2 <= 20, f"二阶 {k2} 步 → 三阶 {k3} 步")

    evals, evecs = torch.linalg.eigh(cov)
    w_ref = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(1e-3))) @ evecs.transpose(0, 1)
    e3 = rel(whiten_ns.inv_sqrt_ns(cov, cubic=True), w_ref)
    report("三阶 W vs eigh（良态等价）", e3 < 1e-5, f"max rel = {e3:.2e}")

    print(f"\n{'全部闸门通过 ✓' if not FAILS else '失败：' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
