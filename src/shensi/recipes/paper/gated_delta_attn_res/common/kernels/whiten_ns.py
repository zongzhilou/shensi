"""牛顿–舒尔茨求逆平方根：幂迭代估谱带、三阶加速迭代、不收敛时的显式报错。"""

from __future__ import annotations

import math

import torch


def _lam_max(A: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """幂迭代估计最大特征值（返回 [B] 形状的逐批结果）。"""
    x = torch.ones(A.shape[:-1] + (1,), device=A.device, dtype=A.dtype)
    tiny = torch.finfo(A.dtype).tiny
    x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    for _ in range(iters):
        x = A @ x
        x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    y = A @ x
    return (x.transpose(-1, -2) @ y).squeeze(-1).squeeze(-1)


def _lam_min(A: torch.Tensor, lam_max: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """用幂迭代估最小特征值：对 (λmax·I − A) 做主特征值估计，再换算回 λmin。"""
    d = A.shape[-1]
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    shifted = lam_max.unsqueeze(-1).unsqueeze(-1) * eye - A
    return (lam_max - _lam_max(shifted, iters)).clamp_min(0.0)


class NSNotConverged(RuntimeError):
    """牛顿–舒尔茨在给定步数内没有达到容差（病态输入下不要当成等价实现用）。"""


def _predicted_steps(lo: float, tol: float) -> int:
    """按三阶收敛的粗略预测：初始缺陷 (1−l) 的 3^k 次方降到 tol 所需的 k。"""
    e0 = max(1.0 - lo, 1e-12)
    if e0 >= 1.0:
        return 1 << 30
    need = math.log(max(tol, 1e-30)) / math.log(e0)
    if need <= 1.0:
        return 1
    return max(1, math.ceil(math.log(need) / math.log(3.0)))


def inv_sqrt_ns(
    A: torch.Tensor, iters: int = 40, tol: float = 1e-5, *, strict: bool = True, cubic: bool = True
) -> torch.Tensor:
    """牛顿–舒尔茨迭代求逆平方根（先按 λmax 缩放，再迭代；``cubic`` 时用三阶校正）。

    收敛判据是归一化后的残差 ``max|Z·Â·Z − I|``；达不到容差时抛 ``NSNotConverged``，
    并把估到的谱带（λmin/λmax/κ）与按三阶收敛预测的步数一并给出。
    """
    d = A.shape[-1]
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    tiny = torch.finfo(A.dtype).tiny
    lam = _lam_max(A).clamp_min(tiny)
    Ahat = A / lam.unsqueeze(-1).unsqueeze(-1)
    Y = Ahat.clone()
    Z = eye.expand_as(Ahat).clone()
    for _ in range(max(1, iters)):
        if float((Z @ Ahat @ Z - eye).abs().amax()) < tol:
            break
        M = Z @ Y
        if cubic:
            D = eye - M
            G = eye + 0.5 * D + 0.375 * (D @ D)
            Y = Y @ G
            Z = G @ Z
        else:
            F = 0.5 * (3.0 * eye - M)
            Y = Y @ F
            Z = F @ Z
    res = float((Z @ Ahat @ Z - eye).abs().amax())
    if strict and res > tol:
        lo = float((_lam_min(Ahat, torch.ones_like(lam)) / lam.clamp_min(tiny)).amin())
        raise NSNotConverged(
            f"NS 残差 {res:.2e} > 容差 {tol:.2e}（{iters} 步；估到 λmin/λmax ≈ {lo:.2e}，"
            f"三阶收敛的理想预测约 {_predicted_steps(lo, tol)} 步，实际受 fp32 残差地板限制；"
            "病态档请改用 eigh/融合内核）"
        )
    return Z * torch.rsqrt(lam).unsqueeze(-1).unsqueeze(-1)


_WARNED = False


def whitening_transform_ns(
    values: torch.Tensor, mode: str, ridge: float, *, warn_residual: float = 1e-4
) -> torch.Tensor:
    """用牛顿–舒尔茨结果构造白化变换（含对角线兜底档）；残差超阈值只警告一次。"""
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
    """把连接模块的逆平方根换成 NS 实现，返回是否安装成功。"""
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import gdar_connection

    if getattr(gdar_connection, "_whiten_ns_installed", False):
        return False
    gdar_connection._whiten_eager = gdar_connection._whitening_transform
    gdar_connection._whitening_transform = whitening_transform_ns
    gdar_connection._whiten_ns_installed = True
    return True


def uninstall() -> bool:
    """还原成参考（eigh）实现。"""
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import gdar_connection

    if not getattr(gdar_connection, "_whiten_ns_installed", False):
        return False
    gdar_connection._whitening_transform = gdar_connection._whiten_eager
    gdar_connection._whiten_ns_installed = False
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 1024
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.randn(10240, d, device=dev, dtype=torch.float32)
    A = (S.transpose(0, 1) @ S) / S.shape[0] + 1e-3 * torch.eye(d, device=dev)
    evals, evecs = torch.linalg.eigh(A)
    w_ref = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(1e-3))) @ evecs.transpose(0, 1)
    for cubic in (False, True):
        w_ns = inv_sqrt_ns(A, cubic=cubic)
        rel = ((w_ns - w_ref).abs().amax() / w_ref.abs().amax()).item()
        print(f"d={d} cubic={cubic}：max|W_ns − W_eigh| / max|W| = {rel:.3e}")
