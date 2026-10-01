"""Looma 的 vLLM 实现（rollout/评测侧）：复用 vLLM 原生 Llama 的算子，只新写连接与不动点。"""

from .variants import BY_ARCH, BY_KEY, BY_MODEL_TYPE, VARIANTS, LoomaVariant

__all__ = ["BY_ARCH", "BY_KEY", "BY_MODEL_TYPE", "VARIANTS", "LoomaVariant"]
