"""兼容垫片：unified 的实现已迁到 ``modeling_qwen3_vl_unified.py``（encoder-free 口径）。

保留本模块只为旧导入路径（``from .modeling import build_unified``）不断链。
"""

from .modeling_qwen3_vl_unified import (
    Qwen3VLUnifiedForConditionalGeneration,
    Qwen3VLUnifiedModel,
    Qwen3VLUnifiedProcessor,
    Qwen3VLUnifiedVisionEmbedder,
    build_unified,
    build_unified_processor,
)

__all__ = [
    "Qwen3VLUnifiedForConditionalGeneration",
    "Qwen3VLUnifiedModel",
    "Qwen3VLUnifiedProcessor",
    "Qwen3VLUnifiedVisionEmbedder",
    "build_unified",
    "build_unified_processor",
]
