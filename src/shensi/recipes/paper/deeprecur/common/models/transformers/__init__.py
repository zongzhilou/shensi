"""transformers 镜像：引用上游 Qwen3-VL；unified 是 encoder-free（对齐 Gemma 4）；GDAR 版换两塔的层。"""

from .configuration import Qwen3VLUnifiedConfig, budget_to_max_pixels
from .modeling_qwen3_vl_unified import (
    Qwen3VLUnifiedForConditionalGeneration,
    Qwen3VLUnifiedModel,
    Qwen3VLUnifiedProcessor,
    Qwen3VLUnifiedVisionEmbedder,
    build_unified,
    build_unified_processor,
)

__all__ = [
    "Qwen3VLUnifiedConfig",
    "Qwen3VLUnifiedForConditionalGeneration",
    "Qwen3VLUnifiedModel",
    "Qwen3VLUnifiedProcessor",
    "Qwen3VLUnifiedVisionEmbedder",
    "budget_to_max_pixels",
    "build_unified",
    "build_unified_processor",
]

# GDAR 两塔 + DeepRecur 顶层（按需显式导入，避免把重依赖挂到轻路径上）
try:  # pragma: no cover - 依赖 gated_delta_attn_res 配方与上游 Qwen3-VL
    from .configuration_qwen3_vl_gdar import Qwen3VLGdarConfig
    from .modeling_qwen3_vl_gdar import (
        Qwen3VLGdarDeepRecurForConditionalGeneration,
        Qwen3VLGdarDeepRecurModel,
        Qwen3VLGdarForConditionalGeneration,
        Qwen3VLGdarModel,
        Qwen3VLGdarTextModel,
        Qwen3VLGdarVisionModel,
        block_boundaries,
    )

    __all__ += [
        "Qwen3VLGdarConfig",
        "Qwen3VLGdarDeepRecurForConditionalGeneration",
        "Qwen3VLGdarDeepRecurModel",
        "Qwen3VLGdarForConditionalGeneration",
        "Qwen3VLGdarModel",
        "Qwen3VLGdarTextModel",
        "Qwen3VLGdarVisionModel",
        "block_boundaries",
    ]
except ImportError:  # pragma: no cover
    pass
