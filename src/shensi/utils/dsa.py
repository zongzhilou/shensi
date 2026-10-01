"""DSA 融合内核的可用性判断与后端回退。"""

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
