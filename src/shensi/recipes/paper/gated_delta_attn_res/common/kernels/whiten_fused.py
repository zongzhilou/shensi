"""融合读与融合应用：把协方差、逆平方根与读出合成更少的 kernel 调用。"""

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

    @triton.jit
    def _read_kernel_fused_apply(
        VALUES,
        QUERY,
        W,
        MIX,
        OUT,
        SCORES,
        T,
        S,
        H,
        heads,
        dh,
        eps,
        NULL: tl.constexpr,
        RETURN_SCORES: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_S: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        offs_s = tl.arange(0, MAX_S)
        tmask = (offs_t < T)[:, None]
        dmask = (offs_d < dh)[None, :]

        q_acc = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for k0 in range(0, H, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            qk = tl.load(
                QUERY + offs_t[:, None] * H + offs_k[None, :],
                mask=tmask & (offs_k < H)[None, :],
                other=0.0,
            )
            wk = tl.load(
                W + offs_k[:, None] * H + pid_h * dh + offs_d[None, :],
                mask=(offs_k < H)[:, None] & dmask,
                other=0.0,
            )
            # 两处 tl.dot 都必须 ieee：tf32 会让分数超出与参考实现的等价容差
            q_acc = tl.dot(qk, wk, q_acc, input_precision="ieee")

        logits = tl.full((BLOCK_T, MAX_S), float("-inf"), tl.float32)
        for s in tl.static_range(MAX_S):
            v_acc = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
            for k0 in range(0, H, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                vk = tl.load(
                    VALUES + (offs_t[:, None] * S + s) * H + offs_k[None, :],
                    mask=tmask & (offs_k < H)[None, :] & (s < S),
                    other=0.0,
                )
                wk = tl.load(
                    W + offs_k[:, None] * H + pid_h * dh + offs_d[None, :],
                    mask=(offs_k < H)[:, None] & dmask,
                    other=0.0,
                )
                v_acc = tl.dot(vk, wk, v_acc, input_precision="ieee")
            dot = tl.sum(v_acc * q_acc, axis=1)
            msq = tl.sum(v_acc * v_acc, axis=1) / dh
            logit = dot * tl.rsqrt(msq + eps)
            logits = tl.where((offs_s == s)[None, :] & (s < S), logit[:, None], logits)

        if NULL:
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
                MIX + (offs_t[:, None] * S + s) * H + pid_h * dh + offs_d[None, :],
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
    fuse_apply: bool = False,
):
    """融合读：协方差、逆平方根与读出合并进更少的 kernel 调用。"""
    num_tokens, num_sources, hidden = values.shape
    if fuse_apply and whiten == "full" and HAS_TRITON and num_sources <= MAX_SOURCES:
        if impl == "eager_whiten":
            from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (
                gdar_connection as gc,
            )

            w = gc._whitening_transform(values, "full", ridge).float()
        else:
            from . import whiten_triton

            w = whiten_triton.whitening_transform_triton(values, "full", ridge).float()
        return _fused_apply_read(
            values.float(), query.float(), w, eps, heads, null, return_scores, mix, num_sources
        )
    if whiten in ("off", "diag", "full"):
        values = values.float()
        query = query.float()
        if whiten == "off":
            values_s, query_s = values, query
        elif impl == "eager_whiten":
            from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (
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
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (
        gdar_connection as gc,
    )

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


def install(impl: str = "ns", *, fuse_apply: bool = False) -> bool:
    """把连接模块的深度读换成融合实现，返回是否安装成功。"""
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
            fuse_apply=fuse_apply,
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


def _fused_apply_read(values, query, w, eps, heads, null, return_scores, mix, num_sources):
    if not HAS_TRITON:
        return _read_tail_torch(
            _apply_W(values, w), _apply_W(query, w), values, eps, heads, null, return_scores
        )
    num_tokens, _, hidden = values.shape
    dh = hidden // heads
    out = torch.empty((num_tokens, heads, dh), device=values.device, dtype=torch.float32)
    scores = (
        torch.empty((num_tokens, num_sources, heads), device=values.device, dtype=torch.float32)
        if return_scores
        else out
    )
    block_t, block_k = 32, 64
    _read_kernel_fused_apply[(triton.cdiv(num_tokens, block_t), heads)](
        values,
        query,
        w,
        values,
        out,
        scores,
        num_tokens,
        num_sources,
        hidden,
        heads,
        dh,
        eps,
        NULL=bool(null),
        RETURN_SCORES=bool(return_scores),
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        BLOCK_D=triton.next_power_of_2(dh),
        MAX_S=MAX_SOURCES,
    )
    routed = out.reshape(num_tokens, hidden)
    if return_scores:
        return routed, scores
    return routed
