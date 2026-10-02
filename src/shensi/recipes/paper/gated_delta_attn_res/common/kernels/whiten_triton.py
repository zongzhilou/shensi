"""Triton 实现的协方差与逐头白化。"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None
    tl = None
    HAS_TRITON = False

from . import whiten_ns

if HAS_TRITON:

    @triton.jit
    def _cov_kernel(
        S,
        COV,
        n_rows,
        h,
        ridge,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        PREC: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_n = pid_n * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros((BLOCK_H, BLOCK_H), dtype=tl.float32)
        for k0 in range(0, n_rows, BLOCK_N):
            offs_k = k0 + tl.arange(0, BLOCK_N)
            kmask = offs_k < n_rows
            a = tl.load(S + offs_k[:, None] * h + offs_m[None, :], mask=kmask[:, None], other=0.0)
            b = tl.load(S + offs_k[:, None] * h + offs_n[None, :], mask=kmask[:, None], other=0.0)
            if PREC == "tf32":
                acc = tl.dot(tl.trans(a), b, acc, input_precision="tf32")
            else:
                acc = tl.dot(tl.trans(a), b, acc, input_precision="ieee")
        acc = acc / n_rows
        acc = tl.where(offs_m[:, None] == offs_n[None, :], acc + ridge, acc)
        tl.store(COV + offs_m[:, None] * h + offs_n[None, :], acc)

    @triton.jit
    def _cov_per_head_kernel(
        S,
        COV,
        n_rows,
        heads,
        dh,
        eps_d,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PREC: tl.constexpr,
    ):
        head = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        dmask = offs_d < dh
        base = head * dh
        acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        for k0 in range(0, n_rows, BLOCK_N):
            offs_k = k0 + tl.arange(0, BLOCK_N)
            kmask = offs_k < n_rows
            pa = S + offs_k[:, None] * (heads * dh) + base + offs_d[None, :]
            a = tl.load(pa, mask=kmask[:, None] & dmask[None, :], other=0.0)
            if PREC == "tf32":
                acc = tl.dot(tl.trans(a), a, acc, input_precision="tf32")
            else:
                acc = tl.dot(tl.trans(a), a, acc, input_precision="ieee")
        acc = acc / n_rows
        diag = tl.sum(tl.where(offs_d[:, None] == offs_d[None, :], acc, 0.0), axis=1)
        scale = tl.sum(diag) / dh
        ridge_h = tl.maximum(dh, n_rows) * eps_d * scale
        acc = tl.where(offs_d[:, None] == offs_d[None, :], acc + ridge_h, acc)
        tl.store(
            COV + head * dh * dh + offs_d[:, None] * dh + offs_d[None, :],
            acc,
            mask=dmask[:, None] & dmask[None, :],
        )

    def cov_symmetric(
        S: torch.Tensor, ridge: float, precision: str = "ieee", block_n: int = 64
    ) -> torch.Tensor:
        n_rows, h = S.shape
        out = torch.empty((h, h), device=S.device, dtype=torch.float32)
        grid = (triton.cdiv(h, 64), triton.cdiv(h, 64))
        _cov_kernel[grid](S, out, n_rows, h, ridge, BLOCK_N=block_n, BLOCK_H=64, PREC=precision)
        return out

    def cov_per_head(
        S: torch.Tensor, heads: int, eps: float = 1e-3, precision: str = "ieee", block_n: int = 64
    ) -> torch.Tensor:
        n_rows, h = S.shape
        dh = h // heads
        out = torch.empty((heads, dh, dh), device=S.device, dtype=torch.float32)
        _cov_per_head_kernel[(heads,)](
            S,
            out,
            n_rows,
            heads,
            dh,
            torch.finfo(torch.float32).eps,
            BLOCK_N=block_n,
            BLOCK_D=triton.next_power_of_2(dh),
            PREC=precision,
        )
        return out

else:  # pragma: no cover - 没有 Triton 的环境

    def cov_symmetric(S: torch.Tensor, ridge: float, precision: str = "ieee", block_n: int = 64):
        cov = (S.transpose(0, 1) @ S) / S.shape[0]
        d = cov.shape[0]
        return cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)

    def cov_per_head(
        S: torch.Tensor, heads: int, eps: float = 1e-3, precision: str = "ieee", block_n: int = 64
    ):
        n, h = S.shape
        dh = h // heads
        V = S.reshape(n, heads, dh)
        cov = torch.einsum("nhd,nhe->hde", V, V) / n
        scale = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
        ridge_h = max(dh, n) * torch.finfo(cov.dtype).eps * scale
        cov.diagonal(dim1=-2, dim2=-1).add_(ridge_h.unsqueeze(-1))
        return cov


def whitening_transform_triton(
    values: torch.Tensor, mode: str, ridge: float, precision: str = "ieee"
):
    """Triton 实现的协方差 + 变换（full 档）。"""
    with torch.no_grad():
        S = values.reshape(-1, values.shape[-1]).float().detach()
        if mode == "diag":
            return torch.rsqrt(S.pow(2).mean(dim=0) + ridge)
        if mode != "full":
            raise ValueError(
                f"whitening_transform_triton 只接 'diag'/'full'，收到 {mode!r}（逐头档走 whitening_transform_per_head）"
            )
        cov = cov_symmetric(S, ridge, precision=precision)
        return whiten_ns.inv_sqrt_ns(cov)


def whitening_transform_per_head(
    values: torch.Tensor, heads: int, precision: str = "ieee"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton 实现的逐头协方差 + 变换。"""
    with torch.no_grad():
        S = values.reshape(-1, values.shape[-1]).float().detach()
        n, h = S.shape
        dh = h // heads
        cov = cov_per_head(S, heads, precision=precision)
        scale = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
        ridge_h = max(dh, n) * torch.finfo(cov.dtype).eps * scale

        return whiten_ns.inv_sqrt_ns(cov), ridge_h
