"""deeprecur 的模型变体注册表：同一 Qwen3-VL 基座，三种口径。

- ``qwen3_vl``（native）：Qwen 原生——deepstack 多层视觉注入（ViT 第 8/16/24 层特征经
  ``deepstack_merger_list`` 加到 LLM 中间层），动态分辨率。
- ``qwen3_vl_unified``：**对齐 Gemma 4 的 encoder-free 版**——*删掉整个 ViT*，48×48×3 的
  合并后原始 patch 经"单个大 matmul"（LN→Dense→LN→+因子化 2D 位置→LN→无缩放 RMSNorm→Linear）
  直接投影进 LLM 空间；视觉 token 预算沿用 Gemma 4 的离散档 {70,140,280,560,1120}。
  视觉侧 knobs 用 ``Gemma4UnifiedVisionConfig``。
- ``qwen3_vl_gdar``：GDAR/DeepRecur 版（两塔单流 attn-res + 顶层交织），见
  ``configuration_qwen3_vl_gdar.py`` / ``modeling_qwen3_vl_gdar.py``。
"""

from __future__ import annotations

from dataclasses import dataclass

#: Gemma 4 技术报告（arXiv 2607.02770）§视觉 token 预算的离散档
GEMMA4_BUDGETS = (70, 140, 280, 560, 1120)

#: Qwen3-VL 原生的 deepstack 注入点（ViT 层序号，transformers / vLLM / mbridge 三方一致）
NATIVE_DEEPSTACK_INDEXES = (8, 16, 24)

#: Gemma 4 encoder-free 的合并后 patch 边长：patch 16 × pooling 3 = 48 像素
UNIFIED_MODEL_PATCH_SIZE = 48


@dataclass(frozen=True)
class VLVariant:
    """一个变体 = deepstack 开关 + 视觉 token 预算。"""

    key: str
    deepstack_visual_indexes: tuple[int, ...]
    visual_token_budget: int | None  # None = Qwen 原生动态分辨率（不设预算）

    @property
    def is_unified(self) -> bool:
        return not self.deepstack_visual_indexes


#: Qwen3-VL 原生口径（deepstack 开、无预算）
NATIVE = VLVariant("qwen3_vl", NATIVE_DEEPSTACK_INDEXES, None)
#: unified 口径（encoder-free、Gemma 4 最大预算档）
UNIFIED = VLVariant("qwen3_vl_unified", (), 1120)

BY_KEY = {v.key: v for v in (NATIVE, UNIFIED)}


def budget_to_max_pixels(budget: int, model_patch_size: int = UNIFIED_MODEL_PATCH_SIZE) -> int:
    """Gemma 4 Algorithm 1 的预算→像素换算：T = N_max · m²，m = patch × pooling。

    encoder-free 路径没有 pooling 核（合并已由图像处理器完成），所以 m 就是合并后
    patch 的边长（默认 48）。每个 soft token 恰好对应一个 m×m 的原始 patch。
    """
    return budget * model_patch_size**2
