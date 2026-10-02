"""Qwen3-VL 的 GDAR 配置：两塔共用一套 attn-res 旋钮，递归块数是一个共享数（无 HC）。

设计口径（DeepRecur 的前两步）：

- Text / Vision 两塔的层都换成**单流** GDAR（行银行 + 门控 delta + 白化读）；
- ``recur_blocks`` 是两塔**共享**的"递归次数"：各塔按 ``floor(i × depth / n)`` 切块，
  块数都是 n（DeepRecur 的 reinject/feedback 依赖两塔块数一致）；
- ``recur_blocks=None`` 关闭 GDAR——模型完全回上游 Qwen3-VL 行为（可做逐位对照）。

``attn_res_*`` 旋钮在本配置**单点持有**（默认值与 ``gated_delta_attn_res`` 配方一致，
消融预置可直接搬）；塔内用适配器把「旋钮 + 该塔的 hidden_size / rms_norm_eps」合给
``AttentionResidual``。
"""

from __future__ import annotations

import contextlib

from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

__all__ = ["Qwen3VLGdarConfig"]


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
class Qwen3VLGdarConfig(Qwen3VLConfig):
    """上游 ``Qwen3VLConfig`` + 一组 attn-res 旋钮（两塔共用，单点持有）。"""

    model_type = "qwen3_vl_gdar"

    auto_map = {
        "AutoConfig": "configuration_qwen3_vl_gdar.Qwen3VLGdarConfig",
        "AutoModel": "modeling_qwen3_vl_gdar.Qwen3VLGdarModel",
        "AutoModelForConditionalGeneration": (
            "modeling_qwen3_vl_gdar.Qwen3VLGdarForConditionalGeneration"
        ),
    }

    # ---- 递归结构 ----
    #: 两塔共享的递归块数（块数 = 递归次数）；None = 关闭 GDAR，回上游行为
    recur_blocks: int | None = None
    #: 塔末的深度读（DepthRead）收口；vision 与 language 各一次，都在最后一块
    attn_res_output_route: bool = True
    #: DeepRecur：每块把最新视觉硬覆盖进语言的图像位（reinject ↓）
    recur_reinject: bool = True
    #: DeepRecur：语言回注下一块的视觉（feedback ↑；最后一块没有回边）
    recur_feedback: bool = True
    #: feedback 回注注意力的头数
    recur_feedback_heads: int = 8

    # ---- attn-res 旋钮（与 gated_delta_attn_res 配方同名同默认） ----
    attn_res_gate_rank: int | None = None
    attn_res_q_rank: int | None = None
    attn_res_k_rank: int | None = None
    attn_res_gate_init: str = "paper"
    attn_res_gate_init_bias: float = 4.0
    attn_res_gate_channels: str = "dew"
    attn_res_gate_param: str = "sigmoid"
    attn_res_write_carrier_bias: float = -4.0
    attn_res_decay_ladder: int = 0
    attn_res_decay_tau_max: float = 100.0
    attn_res_update: str = "shensi"
    attn_res_read_heads: int = 1
    attn_res_read_null: bool = False
    attn_res_read_whiten: str = "off"
    attn_res_read_ridge: float = 1e-3
    attn_res_address: str = "state"
    attn_res_gate_source: str = "state"
    attn_res_decay_positivity: str = "free"
    attn_res_lambda_clamp: float | None = -0.5
    attn_res_read_mix: str = "raw"

    def to_dict(self):
        output = super().to_dict()
        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name, None))
        return output


_EXTRA_CONFIG_FIELDS = tuple(Qwen3VLGdarConfig.__annotations__)

with contextlib.suppress(ValueError):  # 重复注册（多进程/多次导入）
    AutoConfig.register("qwen3_vl_gdar", Qwen3VLGdarConfig)
