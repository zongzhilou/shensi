"""qwen3_vl_unified 的配置：Qwen3-VL text 塔 + **Gemma 4 式 encoder-free 视觉**。

对齐 Gemma 4（arXiv 2607.02770 与 transformers 的 ``Gemma4Unified``）：

- **没有 ViT**：48×48×3 的"合并后原始 patch"（图像处理器 patch 16 × 3×3 merge 产出）经
  单个大 matmul 直接投影进 LLM 空间；
- 视觉侧 knobs 直接复用 ``Gemma4UnifiedVisionConfig``（``patch_size=16``、
  ``pooling_kernel_size=3`` → ``model_patch_size=48``、``mm_embed_dim``、
  ``mm_posemb_size=1120``、``output_proj_dims``），不另造一套；
- 视觉 token 预算沿用 Gemma 4 的离散档（``{70,140,280,560,1120}``），像素上限 = 预算 × m²。
"""

from __future__ import annotations

import contextlib

from transformers import AutoConfig
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedVisionConfig,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from ..variants import UNIFIED_MODEL_PATCH_SIZE, budget_to_max_pixels

__all__ = ["Qwen3VLUnifiedConfig", "budget_to_max_pixels"]


def _strict_config(cls):
    try:
        from huggingface_hub.dataclasses import strict
    except ImportError:  # pragma: no cover - huggingface_hub without dataclasses
        return cls
    try:
        return strict(cls)
    except Exception:  # pragma: no cover - 环境不支持时退回普通类
        return cls


@_strict_config
class Qwen3VLUnifiedConfig(PreTrainedConfig):
    """encoder-free 的 Qwen3-VL：text 塔用 Qwen3-VL，视觉用 Gemma 4 的单 matmul 路径。"""

    model_type = "qwen3_vl_unified"

    auto_map = {
        "AutoConfig": "configuration_qwen3_vl_unified.Qwen3VLUnifiedConfig",
        "AutoModel": "modeling_qwen3_vl_unified.Qwen3VLUnifiedModel",
        "AutoModelForCausalLM": "modeling_qwen3_vl_unified.Qwen3VLUnifiedForConditionalGeneration",
        "AutoModelForConditionalGeneration": (
            "modeling_qwen3_vl_unified.Qwen3VLUnifiedForConditionalGeneration"
        ),
    }

    sub_configs = {"text_config": Qwen3VLTextConfig, "vision_config": Gemma4UnifiedVisionConfig}

    text_config: Qwen3VLTextConfig | dict | None = None
    vision_config: Gemma4UnifiedVisionConfig | dict | None = None
    #: 图像占位 token（与 vendored Qwen3 词表对齐的 ``<|image_pad|>``）
    image_token_id: int = 151655
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = Qwen3VLTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen3VLTextConfig()
        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma4UnifiedVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Gemma4UnifiedVisionConfig()
        super().__post_init__(**kwargs)

    @property
    def model_patch_size(self) -> int:
        """合并后 patch 边长（像素）：patch × pooling，默认 48。"""
        vision = self.vision_config
        return getattr(vision, "model_patch_size", None) or UNIFIED_MODEL_PATCH_SIZE


with contextlib.suppress(ValueError):  # 重复注册（多进程/多次导入）
    AutoConfig.register("qwen3_vl_unified", Qwen3VLUnifiedConfig)
