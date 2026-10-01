# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RealFormer 的 mcore 层：Qwen3 的 ``TransformerLayer`` + 残差注意力（见 realformer_attention.py）。

与其它深度连接的层（``gdar_layer.py`` / ``depth_layer.py`` / ``hc_layer.py``）不同，RealFormer
**不动残差流**：它的跨层状态是每层的注意力分数矩阵 ``[b, heads, s, s]``，走一份共享的
:class:`~.realformer_attention.RealFormerCarry`（由 spec 的 ``params`` 交给各层）。两个直接后果
在 ``__init__`` 里被拒绝而不是留到运行时：

* ``pipeline_model_parallel_size > 1``：``[b, heads, s, s]`` 过不了 stage 边界（p2p 形状由 config
  决定），打包进 ``hidden_states`` 的路子（其它变体用的那个）在这里不成立；
* ``recompute_granularity='full'`` 与 ``fp32_residual_connection``：前者会让"层 1 重置、之后顺序
  消费"的时序不再成立，后者假定残差流恒为 H 宽。
"""

from __future__ import annotations

from megatron.core.transformer.transformer_layer import TransformerLayer

from .realformer_attention import build_realformer_submodules, realformer_knobs_from_kwargs

__all__ = ["RealFormerTransformerLayer", "build_realformer_submodules", "realformer_knobs_from_kwargs"]


class RealFormerTransformerLayer(TransformerLayer):
    """一个 decoder 层：注意力步骤换成 RealFormer 的残差注意力。"""

    def __init__(
        self,
        config,
        submodules=None,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        pg_collection=None,
        vp_stage: int | None = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        carry = kwargs.pop("realformer_carry", None)  # spec 建的共享对象（不是旋钮）
        knobs = realformer_knobs_from_kwargs(kwargs, config)
        if config.pipeline_model_parallel_size > 1:
            raise NotImplementedError(
                "RealFormer 的跨层状态是注意力分数矩阵 [b, heads, s, s]，过不了 pipeline 边界"
                "（p2p 的张量形状由 config 决定，装不下这层状态）。请用 "
                "pipeline_model_parallel_size=1。"
            )
        if getattr(config, "recompute_granularity", None) == "full":
            raise NotImplementedError(
                "RealFormer 不支持 recompute_granularity='full'：残差注意力要求层按序"
                "「层 1 重置 → 每层顺序消费」，整层重算会打乱这个时序。"
            )
        if getattr(config, "fp32_residual_connection", False):
            raise NotImplementedError("RealFormer 不支持 fp32_residual_connection。")
        if submodules is None:
            submodules = build_realformer_submodules(
                config,
                gate_mode=knobs["gate"],
                use_running_mean=knobs["mean"],
                carry=carry,
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )
