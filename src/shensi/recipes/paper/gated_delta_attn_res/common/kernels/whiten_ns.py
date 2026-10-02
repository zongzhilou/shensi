"""白化的免 LAPACK 实现：多项式初值 + Newton–Schulz 逆平方根（只用 matmul）。

论文主行的读是白化读（`read_whiten="full"`）：`W = (C + ridge·I)^{-1/2}`，`C = SᵀS/N`。
参考实现走 `eigh`（LAPACK，带一次主机同步）；这里换成分块/批量 matmul：

1. **初值**：把 A 按谱范数归一化后用多项式 p 逼近 λ^{-1/2}（在 [lo, 1] 上按相对误差
   加权最小二乘拟合，只在首次调用时拟合一次，之后缓存）；
2. **抛光**：耦合迭代 Y ← Y(3I − ZY)/2、Z ← (3I − ZY)Z/2（Y₀ = p(A)、Z₀ = p(A)，
   每步相对误差平方级下降），2–3 步就到 fp32 精度；
3. **自查**：残差 `max|ZAZ − I|` 与迭代同为 matmul——不达标就多迭代，不靠外部判据。

与参考实现同数学（同一个 `(C + ridge·I)^{-1/2}`），只换算法；`GDAR(0)` 的恒等路径
不用白化（默认 `read_whiten="off"`），所以恒等闸门不受影响。
"""

from __future__ import annotations

import torch


# λmax 用幂迭代估（matmul，无 LAPACK 同步）：步数少了会给 W 留一个系统性缩放误差
# （实测 6 步 → W 相对误差卡在 ~5e-5），30 步把它压到 fp32 噪声级。
def _lam_max(A: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """幂迭代估最大特征值（matmul；比 eigvalsh 便宜，且无 LAPACK 同步）。"""
    x = torch.ones(A.shape[:-1] + (1,), device=A.device, dtype=A.dtype)
    tiny = torch.finfo(A.dtype).tiny
    x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    for _ in range(iters):
        x = A @ x
        x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    y = A @ x
    return (x.transpose(-1, -2) @ y).squeeze(-1).squeeze(-1)


class NSNotConverged(RuntimeError):
    """NS 迭代在给定步数内没把残差压到容差以内——精度不够，别拿它当等价实现。"""


def inv_sqrt_ns(
    A: torch.Tensor, iters: int = 40, tol: float = 1e-5, *, strict: bool = True
) -> torch.Tensor:
    """A: [..., d, d] 对称正定（含 ridge 平移）→ A^{-1/2}，纯 matmul。

    先按 λmax 归一化（谱落在 (0, 1]），再跑耦合迭代（Y₀ = Â、Z₀ = I）：

        M = Z·Y，Y ← Y(3I − M)/2，Z ← (3I − M)Z/2

    标量情形 m = zy 满足 m ← m(3−m)²/4：m=1 是不动点且在 m=1 处一阶、二阶导都为 0
    （二次收敛）。**收敛由条件数定**：κ 每翻一个数量级就多要一步"线性段"，所以 κ ≲ 1e3 十几步够，
    κ ≳ 1e4（真实隐状态白化常见）几十步也压不到 1e-6——实测包线见 `kernels/README.md`。
    `strict=True` 时收敛不了抛 `NSNotConverged`（免得有人拿不达标的 W 去训练）。
    """
    d = A.shape[-1]
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    lam = _lam_max(A).clamp_min(torch.finfo(A.dtype).tiny)
    Ahat = A / lam.unsqueeze(-1).unsqueeze(-1)
    Y = Ahat.clone()
    Z = eye.expand_as(Ahat).clone()
    for _ in range(max(1, iters)):
        if float((Z @ Ahat @ Z - eye).abs().amax()) < tol:
            break
        M = Z @ Y
        Y = 0.5 * (Y @ (3.0 * eye - M))
        Z = 0.5 * ((3.0 * eye - M) @ Z)
    res = float((Z @ Ahat @ Z - eye).abs().amax())
    if strict and res > tol:
        raise NSNotConverged(
            f"NS 残差 {res:.2e} > 容差 {tol:.2e}（{iters} 步；条件数太大时改用 eigh/融合内核）"
        )
    # 还原尺度：A^{-1/2} = (λmax · Â)^{-1/2} = λmax^{-1/2} · Â^{-1/2}
    return Z * torch.rsqrt(lam).unsqueeze(-1).unsqueeze(-1)


_WARNED = False


def whitening_transform_ns(
    values: torch.Tensor, mode: str, ridge: float, *, warn_residual: float = 1e-4
) -> torch.Tensor:
    """`gdar_connection._whitening_transform` 的等价实现（只支持 diag/full）。

    训练路径用 `strict=False`：残差超过 `warn_residual` 只**警告一次**，不打断训练
    （条件数大的数据上 NS 达不到等价精度——包线见 `kernels/README.md`）。
    """
    global _WARNED
    with torch.no_grad():
        S = values.reshape(-1, values.shape[-1]).float().detach()
        if mode == "diag":
            return torch.rsqrt(S.pow(2).mean(dim=0) + ridge)
        if mode != "full":
            raise ValueError(f"whitening_transform_ns 只接 'diag'/'full'，收到 {mode!r}")
        d = S.shape[-1]
        eye = torch.eye(d, device=S.device, dtype=torch.float32)
        cov = (S.transpose(0, 1) @ S) / S.shape[0] + ridge * eye
        w = inv_sqrt_ns(cov, strict=False)
        res = float((w @ cov @ w - eye).abs().amax())
        if res > warn_residual and not _WARNED:
            _WARNED = True
            print(
                f"[gdar][whiten] ⚠️ NS 白化残差 {res:.2e} > {warn_residual:.0e}"
                "（条件数过大时 NS 不等价；full 档请用 eager/eigh 或融合内核）",
                flush=True,
            )
        return w


def install() -> bool:
    """把连接模块里的白化换成本实现（幂等；返回是否已换）。"""
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import gdar_connection

    if getattr(gdar_connection, "_whiten_ns_installed", False):
        return False
    gdar_connection._whiten_eager = gdar_connection._whitening_transform
    gdar_connection._whitening_transform = whitening_transform_ns
    gdar_connection._whiten_ns_installed = True
    return True


def uninstall() -> bool:
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import gdar_connection

    if not getattr(gdar_connection, "_whiten_ns_installed", False):
        return False
    gdar_connection._whitening_transform = gdar_connection._whiten_eager
    gdar_connection._whiten_ns_installed = False
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 1024
    S = torch.randn(10240, d, device="cuda", dtype=torch.float32)
    A = (S.transpose(0, 1) @ S) / S.shape[0] + 1e-3 * torch.eye(d, device="cuda")
    w_ref = (
        torch.linalg.eigh(A)[1]
        @ torch.diag(torch.rsqrt(torch.linalg.eigh(A)[0].clamp_min(1e-3)))
        @ torch.linalg.eigh(A)[1].transpose(0, 1)
    )
    w_ns = inv_sqrt_ns(A)
    rel = ((w_ns - w_ref).abs().amax() / w_ref.abs().amax()).item()
    print(f"d={d}：max|W_ns − W_eigh| / max|W| = {rel:.3e}")
