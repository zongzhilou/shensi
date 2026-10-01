"""白化的内核侧实现（B6）：Triton 协方差 + Newton–Schulz 逆平方根。

用法（训练入口的开关）：

    python train.py --config ... --gdar-whiten-impl ns      # 免 LAPACK 的等价实现
    python train.py --config ... --gdar-whiten-impl triton  # 再叠加 Triton 协方差

闸门：`python kernels/test_whiten.py`；计时：`python kernels/bench_whiten.py`。
"""

from __future__ import annotations

from . import whiten_ns, whiten_triton

IMPLS = ("eager", "ns", "triton")

__all__ = ["IMPLS", "enable", "whiten_ns", "whiten_triton"]


def enable(impl: str) -> str:
    """把连接模块的白化换成 `impl`；返回实际生效的实现名。"""
    if impl not in IMPLS:
        raise ValueError(f"whiten impl must be one of {IMPLS}, got {impl!r}")
    whiten_ns.uninstall()
    if impl == "eager":
        return "eager"
    if impl == "ns":
        whiten_ns.install()
        return "ns"
    if not whiten_triton.HAS_TRITON:
        raise RuntimeError("要 triton 档得先装 triton（>=3.0）；本机没有 Triton 就用 ns 档")

    def _triton_transform(values, mode, ridge):
        return whiten_triton.whitening_transform_triton(values, mode, ridge)

    whiten_ns.install()
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection

    gdar_connection._whitening_transform = _triton_transform
    return "triton"
