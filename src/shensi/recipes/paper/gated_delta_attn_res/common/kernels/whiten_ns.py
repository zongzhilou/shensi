"""牛顿–舒尔茨求逆平方根：含幂迭代估计最大特征值与不收敛时的显式报错。"""

from __future__ import annotations

import torch


def _lam_max(A: torch.Tensor, iters: int = 30) -> torch.Tensor:
    x = torch.ones(A.shape[:-1] + (1,), device=A.device, dtype=A.dtype)
    tiny = torch.finfo(A.dtype).tiny
    x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    for _ in range(iters):
        x = A @ x
        x = x / x.norm(dim=-2, keepdim=True).clamp_min(tiny)
    y = A @ x
    return (x.transpose(-1, -2) @ y).squeeze(-1).squeeze(-1)


class NSNotConverged(RuntimeError):
    pass


def inv_sqrt_ns(
    A: torch.Tensor, iters: int = 40, tol: float = 1e-5, *, strict: bool = True
) -> torch.Tensor:
    """牛顿–舒尔茨迭代求逆平方根：先用幂迭代估最大特征值做缩放，再迭代到收敛；达不到容差时抛 ``NSNotConverged``（病态输入下不要当成等价实现用）。"""
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

    return Z * torch.rsqrt(lam).unsqueeze(-1).unsqueeze(-1)


_WARNED = False


def whitening_transform_ns(
    values: torch.Tensor, mode: str, ridge: float, *, warn_residual: float = 1e-4
) -> torch.Tensor:
    """用牛顿–舒尔茨结果构造白化变换（含对角线兜底档）。"""
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
