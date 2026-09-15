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

"""Shensi 的块级注意力残差（块记忆）。

对应 HF ``ShensiAttentionResidual``。它是一段 delta-rule 线性注意力记忆 + 块库软路由：

1. ``state = RMSNorm(prefix_sum + delta)``；
2. ``decay/erase/write = sigmoid(gate_proj(state))`` 三分量；
3. ``forgotten = decay * prefix_sum``；``khat = normalize(k_proj(state))``；
   ``updated = forgotten - khat * (khat · (erase * forgotten)) + write * delta``；
4. 若 ``num_blocks > 0``：把已写入的 ``num_blocks`` 个块与 ``updated`` 拼成候选，
   用 ``q_proj(state)`` 打分（按行均方根归一化后 softmax）做加权检索，得到 ``routed``；
5. ``output = updated + routed``，可选地再按调用方传入的权重做一次 RMSNorm。

上游 mcore 没有对应实现（``megatron/core/transformer/*`` 里没有块记忆/检索类模块），因此按
HF 逐行实现；张量约定为 sequence-first，块库形状 ``[s, b, hc, blocks, H]``。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule

from .shensi_norms import ShensiUnweightedRMSNorm

__all__ = ["ShensiAttentionResidual"]


class ShensiAttentionResidual(MegatronModule):
    """块级记忆：写入/擦除/衰减 + 块库检索。

    Args:
        config: Megatron 配置；读取 ``hidden_size``、``layernorm_epsilon``、``params_dtype``。
        has_router: 是否构造 ``q_proj``。为 ``False`` 时表示"当前还没有任何块"，此时调用方
            必须传 ``num_blocks=0``（HF 的调用方用 ``prev_valid_blocks > 0`` 与之保持一致）。
    """

    def __init__(self, config, has_router: bool = True) -> None:
        super().__init__(config)
        hidden = int(config.hidden_size)
        dtype = config.params_dtype
        self.norm = ShensiUnweightedRMSNorm(config)
        self.gate_proj = nn.Linear(hidden, 3 * hidden, bias=True, dtype=dtype)
        self.q_proj = (
            nn.Parameter(torch.empty(hidden, hidden, dtype=dtype)) if has_router else None
        )
        self.k_proj = nn.Parameter(torch.empty(hidden, hidden, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一次块记忆更新。

        Args:
            hidden_states: ``[s, b, hc, H]`` 的增量 ``delta``（首层为 ``None``，等价于 0）。
            residual: ``[s, b, hc, blocks, H]`` 块库。
            prefix_sum: ``[s, b, hc, H]`` 累加记忆。
            output_norm_weight: 出口 RMSNorm 的权重（来自 ``input_layernorm`` /
                ``post_attention_layernorm``）；``None`` 表示不再施加归一化（模型出口）。
            num_blocks: 参与检索的块数（已写入的块个数）。

        Returns:
            ``(output, updated, blocks)``：出口张量（可能已归一化）、更新后的记忆、块库。
            三者 dtype 与 ``prefix_sum`` 一致。
        """
        delta = hidden_states
        blocks = residual
        state = self.norm(prefix_sum.float() + (delta.float() if delta is not None else 0.0))
        decay, erase, write = torch.sigmoid(
            F.linear(
                state,
                self.gate_proj.weight.float(),
                self.gate_proj.bias.float() if self.gate_proj.bias is not None else None,
            ).reshape(*state.shape[:-1], 3, -1)
        ).unbind(-2)
        forgotten = decay * prefix_sum.float()
        khat = F.normalize(F.linear(state, self.k_proj.float()), dim=-1)
        r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
        updated = forgotten - khat * r + write * (delta.float() if delta is not None else 0.0)
        if num_blocks > 0:
            values = torch.cat(
                [blocks[..., :num_blocks, :].float(), updated.unsqueeze(-2)], dim=-2
            )
            # HF 此处刻意没有 keepdim：reciprocal_std 形状为 [*, K+1]，逐候选乘到 logits 上。
            reciprocal_std = torch.rsqrt(values.square().mean(dim=-1) + self.norm.eps)
            query = F.linear(state, self.q_proj.float())
            logits = (values * query.unsqueeze(-2)).sum(dim=-1) * reciprocal_std
            scores = F.softmax(logits, dim=-1)
            routed = scores.unsqueeze(-1).mul(values).sum(dim=-2)
        else:
            routed = torch.zeros_like(updated)
        output = updated + routed
        if output_norm_weight is not None:
            output = (
                output
                * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self.norm.eps)
                * output_norm_weight.float()
            )
        return output.to(prefix_sum.dtype), updated.to(prefix_sum.dtype), blocks
