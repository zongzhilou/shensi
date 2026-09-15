# Copyright (c) 2026 Zongzhi Lou. All rights reserved.
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

"""Shensi 的解码层：三态传递 + 两处 AttentionResidual + 超连接读写。

与 HF ``ShensiDecoderLayer`` 一一对应，但按上游 mcore 的层骨架实现：继承
``HyperConnectionTransformerLayer``（是为了拿到它的 ``forward`` 骨架与两个超连接槽位），
只覆写四个钩子，从而**保留上游的 checkpoint / 卸载 / 重算 / CUDA graph 记账**：

* :meth:`_run_input_layernorm`：解包三态 → 第一个 AttentionResidual → 超连接塌缩；
* :meth:`_apply_self_attn_bda_step`：注意力输出写回多流、更新 ``prefix_sum``、重新打包；
* :meth:`_pre_mlp_layernorm_and_residual`：解包 → 第二个 AttentionResidual → 超连接塌缩；
* :meth:`_apply_mlp_bda_step`：MLP 输出写回、更新 ``prefix_sum``、重新打包。

两个要点与 HF 一致、且与上游惯例不同：

1. 层的 ``input_layernorm`` / ``pre_mlp_layernorm`` **不施加归一化**，它们的 **权重** 被传进
   AttentionResidual（HF 就是这么做的：归一化发生在块记忆内部）；
2. 第一个钩子拿到的是**未打包**的单流张量（块的展开被我们改成恒等，见
   :mod:`shensi.core.transformer.shensi_block`），此时按 HF 的初始状态就地补齐三态。
"""

from __future__ import annotations

import torch

from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

from .shensi_attention_residual import ShensiAttentionResidual
from .state import pack_state, unpack_state

__all__ = ["ShensiTransformerLayer"]


class ShensiTransformerLayer(HyperConnectionTransformerLayer):
    """Shensi 解码层（三态传递）。"""

    #: 声明本层实现 mHC 残差流契约（上游在构建块时会检查）。
    supports_mhc_connections = True

    def __init__(self, config, submodules, layer_number: int = 1, **kwargs):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        self.num_streams = int(config.mhc_num_residual_streams)
        self.hidden = int(config.hidden_size)
        self.num_blocks = config.attn_res_block_layer_types.count("block_write_layer")

        plan = config.attn_res_block_layer_types
        index = layer_number - 1
        # 全局层号索引：PP>1 时本 stage 的局部下标不从 0 开始（虽然超连接模型当前不支持 PP>1）。
        self.block_write_idx = sum(1 for role in plan[:index] if role == "block_write_layer")
        self.is_block_write_layer = plan[index] == "block_write_layer"
        self.is_hash = config.mlp_layer_types[index] == "hash_moe"

        # AttentionResidual 没有上游槽位，按 HF 的规则在本层直接构建（参数名与 HF 逐字一致）。
        self.self_attention_attn_res = ShensiAttentionResidual(config, self.block_write_idx > 0)
        self.mlp_attn_res = ShensiAttentionResidual(
            config, self.block_write_idx + self.is_block_write_layer > 0
        )

    def _materialize_state(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把进入本层的张量解释成三态。

        首层拿到的是单流 embedding（块展开被改成恒等），此时按 HF ``ShensiModel.forward``
        的初始状态补齐：``streams = prefix_sum = 复制的 embedding``、``library = 0``。
        其余层拿到的是打包三态，直接解包。
        """
        if hidden_states.shape[-1] == self.hidden:
            s, b, _ = hidden_states.shape
            streams = hidden_states.unsqueeze(2).expand(s, b, self.num_streams, self.hidden).contiguous()
            library = streams.new_zeros(s, b, self.num_streams, self.num_blocks, self.hidden)
            return streams, streams, library
        return unpack_state(hidden_states, self.num_streams, self.num_blocks, self.hidden)

    def _input_norm_weight(self) -> torch.Tensor:
        """归一化权重：独立 norm 时取它的 ``weight``，被融合进线性层时取融合权重。"""
        if isinstance(self.input_layernorm, IdentityOp):
            return self.get_qkv_layer_norm_weights()
        return self.input_layernorm.weight

    def _pre_mlp_norm_weight(self) -> torch.Tensor:
        """同 :meth:`_input_norm_weight`，用于 MLP 侧。"""
        if isinstance(self.pre_mlp_layernorm, IdentityOp):
            return self.get_mlp_layer_norm_weights()
        return self.pre_mlp_layernorm.weight

    def _run_input_layernorm(self, hidden_states):
        """注意力前：第一个 AttentionResidual + 超连接塌缩。

        Returns:
            ``(collapsed, streams, (prefix_sum, library))``。第二项是塌缩前的多流状态，它带着
            本层写回的语义流到 :meth:`_apply_self_attn_bda_step`；第三项承载 AttentionResidual
            更新后的 ``prefix_sum`` 与块库。
        """
        streams, prefix_sum, library = self._materialize_state(hidden_states)
        # 首层的 streams 与 prefix_sum 是同一次复制的 embedding，相减为 0 —— 与 HF 的
        # `delta = None` 等价（HF 里 None 表示"没有增量"）。
        delta = streams - prefix_sum

        hidden, prefix_sum, library = self.self_attention_attn_res(
            delta,
            library,
            prefix_sum,
            output_norm_weight=self._input_norm_weight(),
            num_blocks=self.block_write_idx,
        )
        if self.is_block_write_layer:
            library = torch.cat(
                [
                    library[..., : self.block_write_idx, :],
                    hidden.unsqueeze(-2),
                    library[..., self.block_write_idx + 1 :, :],
                ],
                dim=-2,
            )
            prefix_sum = None

        # 我们不经过上游的 norm checkpoint / 卸载路径（归一化已在 AttentionResidual 内完成）。
        self._input_layernorm_checkpoint_active = False
        self.attn_norm_manager = None
        collapsed = self.self_attention_hyper_connection(hidden)
        return collapsed, hidden, (prefix_sum, library)

    def _apply_self_attn_bda_step(self, attention_output_with_bias, residual, attn_state=()):
        """注意力后：把注意力输出写回多流、更新 ``prefix_sum``、重新打包三态。"""
        if not attn_state:
            raise ValueError(
                "ShensiTransformerLayer._apply_self_attn_bda_step 需要 "
                "_run_input_layernorm 产出的 (prefix_sum, library)；说明它被一条不传 mHC "
                "中间量的上游路径调用了。"
            )
        prefix_sum, library = attn_state
        attention_output = (
            attention_output_with_bias[0]
            if isinstance(attention_output_with_bias, tuple)
            else attention_output_with_bias
        )
        streams = self.self_attention_hyper_connection.write_back(residual, attention_output)
        prefix_sum = streams if prefix_sum is None else prefix_sum + streams
        return pack_state(streams, prefix_sum, library)

    def _pre_mlp_layernorm_and_residual(self, hidden_states):
        """MLP 前：第二个 AttentionResidual + 超连接塌缩。

        HF 传给第二个 AttentionResidual 的"增量"就是 ``prefix_sum`` 本身，因此第一个与第三个
        参数相同。
        """
        streams, prefix_sum, library = unpack_state(
            hidden_states, self.num_streams, self.num_blocks, self.hidden
        )
        hidden, prefix_sum, library = self.mlp_attn_res(
            prefix_sum,
            library,
            prefix_sum,
            output_norm_weight=self._pre_mlp_norm_weight(),
            num_blocks=self.block_write_idx + int(self.is_block_write_layer),
        )
        self.mlp_norm_manager = None
        collapsed = self.mlp_hyper_connection(hidden)
        return collapsed, hidden, (prefix_sum, library)

    def _apply_mlp_bda_step(self, mlp_output_with_bias, residual, mlp_state=()):
        """MLP 后：写回多流、更新 ``prefix_sum``、重新打包三态。"""
        if not mlp_state:
            raise ValueError(
                "ShensiTransformerLayer._apply_mlp_bda_step 需要 _pre_mlp_layernorm_and_residual "
                "产出的 (prefix_sum, library)；说明它被一条不传 mHC 中间量的上游路径调用了。"
            )
        prefix_sum, library = mlp_state
        mlp_output = (
            mlp_output_with_bias[0] if isinstance(mlp_output_with_bias, tuple) else mlp_output_with_bias
        )
        streams = self.mlp_hyper_connection.write_back(residual, mlp_output)
        prefix_sum = prefix_sum + streams
        return pack_state(streams, prefix_sum, library)
