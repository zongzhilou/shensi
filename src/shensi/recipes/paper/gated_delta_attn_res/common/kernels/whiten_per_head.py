"""逐头白化：每个注意力头独立算协方差与逆平方根。"""

from __future__ import annotations

import torch

from . import whiten_ns, whiten_triton

_INSTALLED = False


def whiten_per_head_apply(
    values: torch.Tensor, query: torch.Tensor, heads: int, *, mix: str = "raw"
):
    """逐头白化：每个注意力头独立算协方差与逆平方根后应用。"""
    values_f, query_f = values.float(), query.float()
    hidden = values_f.shape[-1]
    n = values_f.reshape(-1, hidden).shape[0]
    dh = hidden // heads
    if whiten_triton.HAS_TRITON:
        cov = whiten_triton.cov_per_head(values_f.reshape(n, hidden), heads)
    else:
        flat = values_f.reshape(n, heads, dh)
        cov = torch.einsum("nhd,nhe->hde", flat, flat) / n
        scale0 = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
        cov.diagonal(dim1=-2, dim2=-1).add_(
            (max(dh, n) * torch.finfo(cov.dtype).eps * scale0).unsqueeze(-1)
        )

    whiten_h = whiten_ns.inv_sqrt_ns(cov, strict=False)
    flat_v = values_f.reshape(n, heads, dh)
    flat_q = query_f.reshape(query_f.shape[0], heads, dh)
    values_s = torch.einsum("nhd,hde->nhe", flat_v, whiten_h).reshape(-1, hidden)
    query_s = torch.einsum("nhd,hde->nhe", flat_q, whiten_h).reshape(query_f.shape[0], hidden)
    mix_values = values_s if mix == "whitened" else values_f
    return values_s, query_s, mix_values


def install() -> bool:
    """把逐头白化换进连接模块，返回是否安装成功。"""
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (
        gdar_connection as gc,
    )

    global _INSTALLED
    if _INSTALLED:
        return False

    def patched(
        values,
        query,
        eps,
        heads=1,
        null=False,
        whiten="off",
        ridge=1e-3,
        return_scores=False,
        mix="raw",
    ):
        if whiten != "per_head":
            return gc._depth_read_eager(
                values,
                query,
                eps,
                heads=heads,
                null=null,
                whiten=whiten,
                ridge=ridge,
                return_scores=return_scores,
                mix=mix,
            )
        from . import whiten_fused

        return whiten_fused.fused_depth_read(
            values,
            query,
            eps,
            heads=heads,
            null=null,
            whiten="per_head",
            ridge=ridge,
            return_scores=return_scores,
            mix=mix,
        )

    if not hasattr(gc, "_depth_read_eager"):
        gc._depth_read_eager = gc._depth_read
    gc._depth_read = patched
    _INSTALLED = True
    return True


def uninstall() -> bool:
    """还原成参考实现。"""
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (
        gdar_connection as gc,
    )

    global _INSTALLED
    if not _INSTALLED:
        return False
    gc._depth_read = gc._depth_read_eager
    _INSTALLED = False
    return True
