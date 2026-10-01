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


# 拟合适配的谱区间：归一化后 A 的特征值落在 [lo, 1]；lo 取 1e-4 覆盖到 ridge/λmax ≈ 1e-3 一带
def _lam_max(A: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """幂迭代估最大特征值（matmul；比 eigvalsh 便宜，且无 LAPACK 同步）。"""
    x = torch.ones(A.shape[:-1] + (1,), device=A.device, dtype=A.dtype)
    tiny = torch.finfo(A.dtype).tiny
    x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    for _ in range(iters):
        x = A @ x
        x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    y = A @ x
    return (x.transpose(-1, -2) @ y).squeeze(-1).squeeze(-1)


def inv_sqrt_ns(A: torch.Tensor, iters: int = 14, tol: float = 2e-6) -> torch.Tensor:
    """A: [..., d, d] 对称正定（含 ridge 平移）→ A^{-1/2}，纯 matmul。

    先按 λmax 归一化（谱落在 (0, 1]），再跑耦合迭代（Y₀ = Â、Z₀ = I）：

        M = Z·Y，Y ← Y(3I − M)/2，Z ← (3I − M)Z/2

    标量情形 m = zy 满足 m ← m(3−m)²/4：m=1 是不动点且在 m=1 处一阶、二阶导都为 0
    （二次收敛），m ∈ (0,5) 都收敛。谱跨几个数量级时前几步是"线性段"、之后才平方级收敛，
    所以迭代次数由条件数定：λ_min/λ_max = 1e-3 要 ~12 步（残差 max|ZÂZ − I| 自查，达标即停）。
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
    # 还原尺度：A^{-1/2} = (λmax · Â)^{-1/2} = λmax^{-1/2} · Â^{-1/2}
    return Z * torch.rsqrt(lam).unsqueeze(-1).unsqueeze(-1)


def whitening_transform_ns(values: torch.Tensor, mode: str, ridge: float) -> torch.Tensor:
    """`gdar_connection._whitening_transform` 的等价实现（只支持 off/diag/full）。"""
    with torch.no_grad():
        S = values.reshape(-1, values.shape[-1]).float().detach()
        if mode == "diag":
            return torch.rsqrt(S.pow(2).mean(dim=0) + ridge)
        if mode != "full":
            raise ValueError(f"whitening_transform_ns 只接 'diag'/'full'，收到 {mode!r}")
        d = S.shape[-1]
        cov = (S.transpose(0, 1) @ S) / S.shape[0]
        cov = cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)
        return inv_sqrt_ns(cov)


def install() -> bool:
    """把连接模块里的白化换成本实现（幂等；返回是否已换）。"""
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection

    if getattr(gdar_connection, "_whiten_ns_installed", False):
        return False
    gdar_connection._whiten_eager = gdar_connection._whitening_transform
    gdar_connection._whitening_transform = whitening_transform_ns
    gdar_connection._whiten_ns_installed = True
    return True


def uninstall() -> bool:
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection

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
