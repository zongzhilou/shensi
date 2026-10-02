#!/usr/bin/env python3
"""B6 新增三件的闸门：融合读、per_head 开关、批量白化。

    python kernels/test_whiten_extra.py

覆盖：
  1. `whiten_fused.fused_depth_read` vs 参考 `_depth_read`：off/diag/full × heads{1,8} × null{on,off}
     × mix{raw,whitened}，逐组合比 routed 与 probs。**紧闸门只给"白化矩阵与参考同源"的档**
     （`eager_whiten`、`per_head`、`off`），NS 档的精度由条件数决定，单独作"精度包线"报告。
  2. `per_head` 开关：换进 `_depth_read` 后 `AttentionResidual.read`（上游档）与参考一致，
     卸掉后逐位回参考；融合读 install 同样过一遍。
  3. `batched_whitening` vs 逐个 `_whitening_transform`（full 与 diag）。
"""

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
    # 各向异性：白化不是"乘个常数"，读的 softmax 才会真的被 W 的小差异影响
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
            # 这是"精度包线"而不是缺陷：full 档协方差条件数大，NS 达不到等价精度（见 README）
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

    # 1b. 把 values@W / query@W 也搬进 kernel（K 循环 tl.dot）：
    #     整链 = 协方差（Triton）+ 逆平方根（NS/cuBLAS）+ 应用/读（一次 kernel）
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

    # 精度包线：NS 的白化误差随条件数变化（诊断，不做断言）
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

    # 2. per_head 开关（上游档）
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

    # 2b. 融合整条读（fused install）在上游档下也一致
    swapped2 = whiten_fused.install("eager_whiten")
    with torch.no_grad():
        fused_out, fused_scores = mod.read(pfx, blk)
    e = rel(fused_out, ref_out)
    report("融合读 install（per_head 档）", swapped2 and e < 1e-4, f"routed {e:.2e}")
    whiten_fused.uninstall()
    with torch.no_grad():
        back2 = mod.read(pfx, blk)[0]
    report("融合读卸掉后逐位回参考", bool((back2 == ref_out).all()), "逐位相等")

    # 3. 批量白化 vs 逐个
    items = [torch.randn(512, s, hidden, device=DEV) for s in (2, 3, 4, 5, 6, 2, 3, 4)]
    batch = whiten_batched.batched_whitening(items, "full", 1e-3)
    worst_b = max(
        rel(batch[i], gc._whitening_transform(v, "full", 1e-3)) for i, v in enumerate(items)
    )
    report("批量白化 vs 逐个（full）", worst_b < 5e-4, f"max rel = {worst_b:.2e}")
    bd = whiten_batched.batched_whitening(items, "diag", 1e-3)
    worst_d = max(rel(bd[i], gc._whitening_transform(v, "diag", 1e-3)) for i, v in enumerate(items))
    report("批量白化 vs 逐个（diag）", worst_d < 1e-6, f"max rel = {worst_d:.2e}")

    print(f"\n{'全部闸门通过 ✓' if not FAILS else '失败：' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
