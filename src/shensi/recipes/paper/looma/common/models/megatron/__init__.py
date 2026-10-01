"""Looma 的 mcore 实现（训练侧）：不动点 block + 深度连接，骨架全部 mcore 原生件。"""

from .looma_connection import LoomaAttentionResidual, LoomaConfig, looma_knobs_from_kwargs
from .looma_layer import LoomaTransformerLayer, build_looma_submodules
from .looma_spec import looma_layer_spec

__all__ = [
    "LoomaAttentionResidual",
    "LoomaConfig",
    "LoomaTransformerLayer",
    "build_looma_submodules",
    "looma_knobs_from_kwargs",
    "looma_layer_spec",
]
