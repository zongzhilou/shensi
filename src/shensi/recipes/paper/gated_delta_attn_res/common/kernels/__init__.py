"""白化算子的可切换实现：eager / fused / fused_apply / per_head / ns / triton 六档。"""

from __future__ import annotations

from . import whiten_ns, whiten_triton

IMPLS = ("eager", "fused", "fused_apply", "per_head", "ns", "triton")

__all__ = ["IMPLS", "enable", "whiten_ns", "whiten_triton"]


def enable(impl: str) -> str:
    """切换白化实现档（eager / fused / fused_apply / per_head / ns / triton），返回是否认识该档。"""
    if impl not in IMPLS:
        raise ValueError(f"whiten impl must be one of {IMPLS}, got {impl!r}")
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import gdar_connection

    from . import whiten_fused, whiten_per_head

    whiten_ns.uninstall()
    whiten_per_head.uninstall()
    whiten_fused.uninstall()
    if impl == "eager":
        return "eager"
    if impl == "fused":
        whiten_fused.install("eager_whiten")
        return "fused"
    if impl == "fused_apply":
        whiten_fused.install("eager_whiten", fuse_apply=True)
        return "fused_apply"
    if impl == "per_head":
        whiten_per_head.install()
        return "per_head"
    if impl == "ns":
        whiten_ns.install()
        print(
            "[gdar][whiten] ns 档只在良态协方差上等价（full 档残差不达容差会正在训练里报出来）",
            flush=True,
        )
        return "ns"

    def _triton_transform(values, mode, ridge):
        return whiten_triton.whitening_transform_triton(values, mode, ridge)

    whiten_ns.install()
    gdar_connection._whitening_transform = _triton_transform
    print("[gdar][whiten] triton 档同上：full 档有精度包线，别拿它出论文数", flush=True)
    return "triton"
