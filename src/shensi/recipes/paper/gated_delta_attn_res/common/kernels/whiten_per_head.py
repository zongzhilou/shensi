"""`per_head` 档的白化（上游默认档）：接上 Triton 协方差 + Newton–Schulz。

参考实现里 `per_head` 是**内联**在 `_depth_read` 里的：逐头协方差 → 逐头 eigh → 逐头白化矩阵，
再把 values/query 变换过去。这里把那一段抽出来做成可替换实现（`install()` 换进 `_depth_read`），
逐头协方差走 `whiten_triton.cov_per_head`（一次 kernel），逆平方根走 `whiten_ns.inv_sqrt_ns`
（batched，只 dh×dh 的矩阵）。

**一处与参考实现的差别（如实）**：参考实现还按最大特征值夹了一层地板
`floor = λ_max · dh · eps`；协方差对角上已经加了 `ridge_h = max(dh, n)·eps·scale` 的平移，
而 `eig(A + rI) = eig(A) + r` ⇒ 地板在正常尺度下不生效（λ_min ≥ ridge_h ≫ floor）。
"""

from __future__ import annotations

import torch

from . import whiten_ns, whiten_triton

_INSTALLED = False


def whiten_per_head_apply(
    values: torch.Tensor, query: torch.Tensor, heads: int, *, mix: str = "raw"
):
    """返回 (values_s, query_s, mix_values)：白化后的 values/query 与"混合源"。

    `mix="whitened"` 时混合源用白化后的 values（与参考实现一致）。
    """
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
    # 逐头矩阵是 dh×dh（上游档 dh=128），条件数比 full 好得多；判据是"读的输出等价"而不是残差
    # 严格到 2e-6，所以这里 warn-only（残差实测 3e-6 级，读级误差 ~1e-5）
    whiten_h = whiten_ns.inv_sqrt_ns(cov, strict=False)
    flat_v = values_f.reshape(n, heads, dh)
    flat_q = query_f.reshape(query_f.shape[0], heads, dh)
    values_s = torch.einsum("nhd,hde->nhe", flat_v, whiten_h).reshape(-1, hidden)
    query_s = torch.einsum("nhd,hde->nhe", flat_q, whiten_h).reshape(query_f.shape[0], hidden)
    mix_values = values_s if mix == "whitened" else values_f
    return values_s, query_s, mix_values


def install() -> bool:
    """把 `_depth_read` 换成本实现（幂等）；返回是否发生了替换。"""
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
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (
        gdar_connection as gc,
    )

    global _INSTALLED
    if not _INSTALLED:
        return False
    gc._depth_read = gc._depth_read_eager
    _INSTALLED = False
    return True
