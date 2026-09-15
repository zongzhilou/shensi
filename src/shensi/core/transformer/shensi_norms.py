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

"""Shensi 的两个 RMSNorm（HF ``ShensiRMSNorm`` / ``ShensiUnweightedRMSNorm`` 的对应物）。

两者都在 float32 下计算方差再转回输入 dtype（与 HF 一致，是数值对齐的必要条件）；
带权重的那支把输出乘上权重。注意本仓库另有 :class:`ShensiOutputContract`，它在出口处直接
用无权重 RMSNorm 的算式，因此这里不额外抽象。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

__all__ = ["ShensiWeightedRMSNorm", "ShensiUnweightedRMSNorm"]


class ShensiWeightedRMSNorm(MegatronModule):
    """带权重的 RMSNorm，对应 HF ``ShensiRMSNorm``。

    Args:
        config: Megatron 配置，读取 ``hidden_size``、``layernorm_epsilon``、``params_dtype``。
    """

    def __init__(self, config, hidden_size: int | None = None) -> None:
        super().__init__(config)
        self.hidden_size = int(hidden_size or config.hidden_size)
        self.variance_epsilon = float(config.layernorm_epsilon)
        self.weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=config.params_dtype)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """归一化并乘权重。

        Args:
            hidden_states: ``[..., hidden_size]``。

        Returns:
            与输入同形状、同 dtype 的张量。
        """
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class ShensiUnweightedRMSNorm(MegatronModule):
    """无权重 RMSNorm，对应 HF ``ShensiUnweightedRMSNorm``。

    只做 ``x * rsqrt(mean(x^2) + eps)``（float32 内部计算）。``eps`` 同时被
    :class:`~shensi.core.transformer.shensi_attention_residual.ShensiAttentionResidual`
    用作块库的路由归一化项，因此对外暴露为属性 ``eps``。
    """

    def __init__(self, config, eps: float | None = None) -> None:
        super().__init__(config)
        self.eps = float(config.layernorm_epsilon if eps is None else eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """按 float32 计算方差后归一化，返回输入 dtype。"""
        return hidden_states * torch.rsqrt(
            hidden_states.float().square().mean(-1, keepdim=True) + self.eps
        ).to(hidden_states.dtype)
