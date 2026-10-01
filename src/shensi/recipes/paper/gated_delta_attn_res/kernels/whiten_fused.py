"""白化读的融合实现：把"应用 + 逐头读"落成少量 kernel。

参考实现的读（`gdar_connection._depth_read` 的 whitened 分支）在 PyTorch 里是十来个算子：
`values @ W`、`query @ W`、逐头 reshape/点积、`square().mean()`、`softmax`、加权和……
每一步都要把 `[T, S, H]`（论文档 T=1024、S≤6、H=1024 ⇒ 每次 25 MB fp32）整块读一遍或写一遍。

这里把它压成：

    kernel 1  Triton 协方差（`whiten_triton.cov_symmetric` / `cov_per_head`）
    kernel 2  Newton–Schulz 逆平方根（`whiten_ns.inv_sqrt_ns`，纯 matmul；d=1024 的矩阵塞不进 kernel）
    kernel 3  Triton **融合应用 + 逐头读**（本文件）：`values @ W`、`query @ W`、归一化点积、
              softmax1/softmax、加权和，一次过

即整条白化读从 ~15 个 torch 算子压到 3 次 kernel 调用（其中一次是 cuBLAS 的 matmul 链）。
d=1024 的逆平方根没法进 kernel（4 MB 矩阵远超 shared memory），这条边界写在 README 里。

需要 Triton；没有时 `fused_depth_read` 会回退到参考实现（同语义）。
"""

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


#: 读的来源数上限（主行 B=4 ⇒ S ≤ 6；留一点余量）
MAX_SOURCES = 8


if HAS_TRITON:

    @triton.jit
    def _read_kernel(
        VS,
        QS,
        MIX,
        OUT,
        SCORES,
        T,
        S,
        heads,
        dh,
        eps,
        NULL: tl.constexpr,
        RETURN_SCORES: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_S: tl.constexpr,
    ):
        """VS/QS: [T, (S), heads, dh]；MIX: [T, S, heads, dh]；OUT: [T, heads, dh]。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        offs_s = tl.arange(0, MAX_S)
        tmask = (offs_t < T)[:, None]
        dmask = (offs_d < dh)[None, :]

        q = tl.load(
            QS + offs_t[:, None] * heads * dh + pid_h * dh + offs_d[None, :],
            mask=tmask & dmask,
            other=0.0,
        )
        logits = tl.full((BLOCK_T, MAX_S), float("-inf"), tl.float32)
        for s in tl.static_range(MAX_S):
            smask = (offs_s == s)[None, :] & (s < S)
            v = tl.load(
                VS + (offs_t[:, None] * S + s) * heads * dh + pid_h * dh + offs_d[None, :],
                mask=tmask & dmask & (s < S),
                other=0.0,
            )
            dot = tl.sum(v * q, axis=1)
            msq = tl.sum(v * v, axis=1) / dh
            logit = dot * tl.rsqrt(msq + eps)
            logits = tl.where(smask, logit[:, None], logits)

        if NULL:
            # Softmax₁：分母 1 + Σexp（= logsumexp 后过 softplus）；Triton 没有 logsumexp，手写稳定式
            mx = tl.max(logits, axis=1)
            lse = mx + tl.log(tl.sum(tl.exp(logits - mx[:, None]), axis=1))
            softplus = tl.maximum(lse, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(lse)))
            probs = tl.exp(logits - softplus[:, None])
        else:
            mx = tl.max(logits, axis=1)
            e = tl.exp(logits - mx[:, None])
            probs = e / tl.sum(e, axis=1)[:, None]

        acc = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for s in tl.static_range(MAX_S):
            ps = tl.sum(tl.where((offs_s == s)[None, :], probs, 0.0), axis=1)
            mv = tl.load(
                MIX + (offs_t[:, None] * S + s) * heads * dh + pid_h * dh + offs_d[None, :],
                mask=tmask & dmask & (s < S),
                other=0.0,
            )
            acc += tl.where(s < S, ps[:, None] * mv, 0.0)

        tl.store(
            OUT + offs_t[:, None] * heads * dh + pid_h * dh + offs_d[None, :],
            acc,
            mask=tmask & dmask,
        )
        if RETURN_SCORES:
            for s in tl.static_range(MAX_S):
                ps = tl.sum(tl.where((offs_s == s)[None, :], probs, 0.0), axis=1)
                tl.store(
                    SCORES + offs_t * S * heads + s * heads + pid_h,
                    ps,
                    mask=(offs_t < T) & (s < S),
                )


def _apply_W(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """[T, S, H] @ [H, H]（W 为向量时按通道缩放）。"""
    if W.dim() == 1:
        return x * W
    return x @ W


def fused_depth_read(
    values: torch.Tensor,
    query: torch.Tensor,
    eps: float,
    *,
    heads: int = 1,
    null: bool = False,
    whiten: str = "full",
    ridge: float = 1e-3,
    return_scores: bool = False,
    mix: str = "raw",
    impl: str = "ns",
):
    """与 `_depth_read`（whiten ∈ {off, diag, full, per_head}）等价的融合实现。

    `impl`：`ns`（协方差走 Triton、逆平方根走 NS）或 `eager_whiten`（白化矩阵沿用参考实现，
    只把后半段读融合——用来单独量"融合"这一步，也是 `enable("fused")` 用的档）。
    """
    num_tokens, num_sources, hidden = values.shape
    if whiten in ("off", "diag", "full"):
        values = values.float()
        query = query.float()
        if whiten == "off":
            values_s, query_s = values, query
        elif impl == "eager_whiten":
            from shensi.recipes.paper.gated_delta_attn_res.models.megatron import (
                gdar_connection as gc,
            )

            w = gc._whitening_transform(values, whiten, ridge).to(values.dtype)
            values_s = _apply_W(values, w)
            query_s = _apply_W(query, w)
        else:
            from . import whiten_triton

            w = whiten_triton.whitening_transform_triton(values, whiten, ridge).to(values.dtype)
            values_s = _apply_W(values, w)
            query_s = _apply_W(query, w)
        mix_values = values_s if mix == "whitened" else values
    elif whiten == "per_head":
        from . import whiten_per_head

        values_s, query_s, mix_values = whiten_per_head.whiten_per_head_apply(
            values, query, heads, mix=mix
        )
    else:
        raise ValueError(f"whiten must be off/diag/full/per_head, got {whiten!r}")

    if not HAS_TRITON or num_sources > MAX_SOURCES:
        return _read_tail_torch(values_s, query_s, mix_values, eps, heads, null, return_scores)

    dh = hidden // heads
    out = torch.empty((num_tokens, heads, dh), device=values.device, dtype=torch.float32)
    scores = (
        torch.empty((num_tokens, num_sources, heads), device=values.device, dtype=torch.float32)
        if return_scores
        else out
    )
    block_t = 32
    _read_kernel[(triton.cdiv(num_tokens, block_t), heads)](
        values_s,
        query_s,
        mix_values,
        out,
        scores,
        num_tokens,
        num_sources,
        heads,
        dh,
        eps,
        NULL=bool(null),
        RETURN_SCORES=bool(return_scores),
        BLOCK_T=block_t,
        BLOCK_D=triton.next_power_of_2(dh),
        MAX_S=MAX_SOURCES,
    )
    routed = out.reshape(num_tokens, hidden)
    if return_scores:
        return routed, scores
    return routed


def _read_tail_torch(values_s, query_s, mix_values, eps, heads, null, return_scores):
    """参考实现的后半段（回退路径，语义与 Triton 版一致）。"""
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection as gc

    num_tokens, num_sources, hidden = values_s.shape
    dh = hidden // heads
    if heads > 1:
        vs = values_s.view(num_tokens, num_sources, heads, dh)
        qs = query_s.view(num_tokens, heads, dh)
        recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
        logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip
        probs = gc._softmax1(logits, dim=1) if null else logits.softmax(dim=1)
        routed = (probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)).sum(
            dim=1
        )
        routed = routed.reshape(num_tokens, hidden)
    else:
        recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
        logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip
        probs = gc._softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
        routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
    if return_scores:
        return routed, probs
    return routed


_INSTALLED = False


def install(impl: str = "ns") -> bool:
    """把 `_depth_read` 换成本实现（幂等）；返回是否发生了替换。"""
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection as gc

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
        return fused_depth_read(
            values,
            query,
            eps,
            heads=heads,
            null=null,
            whiten=whiten,
            ridge=ridge,
            return_scores=return_scores,
            mix=mix,
            impl=impl,
        )

    if not hasattr(gc, "_depth_read_eager"):
        gc._depth_read_eager = gc._depth_read
    gc._depth_read = patched
    _INSTALLED = True
    return True


def uninstall() -> bool:
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection as gc

    global _INSTALLED
    if not _INSTALLED:
        return False
    gc._depth_read = gc._depth_read_eager
    _INSTALLED = False
    return True
