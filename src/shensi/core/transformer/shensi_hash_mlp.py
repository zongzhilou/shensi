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

"""Shensi 的 hash 层：dense MLP × token embedding 门控。

对应 HF ``ShensiHashMLP``：中间维取潜维（``routed_expert_hidden_size``，本仓库读 Megatron 的
``moe_latent_size``），输出再逐元素乘上 ``deepemb[input_ids]``。

注意与上游的差别：上游 DSv4 的 hash 层是**哈希路由 MoE**（``moe_n_hash_layers`` / ``tid2eid``），
而在当前 mcore 修订里那套机制没有任何消费者；Shensi 侧 HF 把它换成了"dense MLP × deepemb"，
没有路由表、没有共享专家，因此这里按 HF 实现，不复用上游的 hash 路由。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from .context import get_current_input_ids

__all__ = ["ShensiHashMLP"]


class ShensiHashMLP(MegatronModule):
    """clamp 过的 SwiGLU MLP，输出乘 ``deepemb(input_ids)``。

    Args:
        config: Megatron 配置；读取 ``hidden_size``、``moe_latent_size``、
            ``activation_func``、``activation_func_clamp_value``、``add_bias_linear``、
            ``vocab_size`` / ``actual_vocab_size`` 与 ``params_dtype``。
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        del kwargs
        dtype = config.params_dtype
        hidden = int(config.hidden_size)
        intermediate = int(config.moe_latent_size)
        bias = bool(config.add_bias_linear)
        self.limit = float(config.activation_func_clamp_value)
        self.act_fn = config.activation_func
        self.gate_proj = nn.Linear(hidden, intermediate, bias=bias, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=bias, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias, dtype=dtype)
        # 用真实词表：mcore 的 vocab_size 可能为并行切分补齐，deepemb 的行数必须与 HF 一致。
        vocab = int(getattr(config, "actual_vocab_size", 0) or config.vocab_size)
        self.deepemb = nn.Embedding(vocab, hidden, dtype=dtype)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """前向。

        Args:
            hidden_states: ``[s, b, H]``（sequence-first）。
            input_ids: ``[b, s]``；``None`` 时取当前 step 暂存的 ids（见
                :mod:`shensi.core.transformer.context`，这也是反向重算路径能取到 ids 的原因）。

        Returns:
            ``[s, b, H]``。
        """
        del kwargs
        if input_ids is None:
            input_ids = get_current_input_ids()
        if input_ids is None:
            raise ValueError(
                "ShensiHashMLP 需要 input_ids：请先调用 "
                "shensi.core.transformer.context.set_current_input_ids（ShensiGPTModel.forward "
                "会设置），或显式传入。"
            )
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(min=-self.limit, max=self.limit)
        # hidden_states 是 sequence-first 而 input_ids 是 [b, s]，门控必须转置后再乘，
        # 否则会广播成 [s, s, H]。
        deepemb = self.deepemb(input_ids).transpose(0, 1)
        return self.down_proj(self.act_fn(gate) * up) * deepemb
