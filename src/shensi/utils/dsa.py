"""DSA 融合内核（cudnn / tilelang）的可用性判断与缺内核时的回退口径。

上游 mcore 给 `dsv4_hybrid` 的默认是 `dsa_kernel_backend="cudnn"`，那要 `flash_mla` +
`nvidia-cudnn-frontend[cutedsl]`；本机（以及任何没装融合内核的环境）构造配置时会直接抛错。
训练入口与 RL 侧（verl → Bridge provider）都要按同一口径回退到 `"none"`（PyTorch 实现，数值同口径）。
"""

from __future__ import annotations

__all__ = ["dsa_backend_fallback", "fused_dsa_kernels_available"]


def fused_dsa_kernels_available() -> bool:
    """cudnn（tilelang 之外的默认）融合内核要的两个包是否都在。"""
    try:
        import flash_mla  # noqa: F401
        from cudnn import DSA  # noqa: F401
    except Exception:
        return False
    return True


def dsa_backend_fallback(configured: str | None) -> str | None:
    """没显式配置（None）且融合内核缺失时给出 `"none"`，否则原样返回。"""
    if configured:
        return configured
    if fused_dsa_kernels_available():
        return configured
    return "none"
