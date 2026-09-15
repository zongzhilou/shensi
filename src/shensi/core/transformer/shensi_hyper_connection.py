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

"""Shensi 的超连接：把多流隐状态塌缩成单流、再把子层输出写回选中的流。

对应 HF ``ShensiHyperConnection``（``transformers/src/transformers/models/shensi/
modeling_shensi.py``），参数名与 HF 逐字一致以便权重一一对应；张量约定为本仓库统一的
**sequence-first**（HF 是 batch-first）。

与上游 ``megatron/core/transformer/hyper_connection.py``（mHC）不是一回事：上游用
``mapping_proj`` + ``alpha_pre/post/res`` + Sinkhorn 生成 ``h_pre/h_post/h_res``、且 ``h_post``
把子层输出广播到**所有**流；Shensi 是 ``pre/route/post_fn`` 三组独立门控 + **固定流 + 路由
top-k** 的局部写回，MLP 侧还要把子层输出与因果深度卷积的 Gram–Schmidt 正交基一起做增广。
数学不同、不可互换，因此这里按 HF 逐行实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule

from .shensi_norms import ShensiUnweightedRMSNorm

__all__ = ["ShensiHyperConnection"]


class ShensiHyperConnection(MegatronModule):
    """Shensi 的超连接模块。

    Args:
        config: Megatron 配置；读取 ``mhc_num_residual_streams``、``hidden_size``、
            ``layernorm_epsilon``、``params_dtype``，以及由 provider 透传的
            ``hc_active_streams`` / ``hc_fixed_streams`` / ``hc_conv_kernels``。
        is_mlp: ``True`` 表示这是 MLP 侧的超连接（多一组时序卷积、``kr > 1``）。
        layer_number: 由 ``build_module`` 传入，这里只用于接口兼容。
    """

    def __init__(self, config, is_mlp: bool = False, layer_number: int | None = None, **kwargs):
        super().__init__(config)
        del layer_number, kwargs
        dtype = config.params_dtype
        hidden = int(config.hidden_size)
        self.hc_mult = int(config.mhc_num_residual_streams)
        self.active_streams = int(config.hc_active_streams)
        self.fixed_streams = int(config.hc_fixed_streams)
        self.routed_streams = self.active_streams - self.fixed_streams

        self.input_norm = ShensiUnweightedRMSNorm(config)
        self.pre_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden, dtype=dtype))
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult, dtype=dtype))
        self.pre_scale = nn.Parameter(torch.empty(1, dtype=dtype))

        self.route_norm = nn.LayerNorm(self.hc_mult * hidden, dtype=dtype)
        self.route_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden, dtype=dtype))
        self.route_base = nn.Parameter(torch.empty(self.hc_mult, dtype=dtype))
        self.route_scale = nn.Parameter(torch.empty(1, dtype=dtype))

        self.is_mlp = is_mlp
        self.kr = (len(config.hc_conv_kernels) + 1) if is_mlp else 1
        if is_mlp:
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden,
                        hidden,
                        kernel_size,
                        padding=kernel_size - 1,
                        groups=hidden,
                        bias=False,
                        dtype=dtype,
                    )
                    for kernel_size in config.hc_conv_kernels
                ]
            )
        self.post_fn = nn.Parameter(
            torch.empty(self.active_streams * self.kr, self.active_streams * hidden, dtype=dtype)
        )
        self.post_base = nn.Parameter(torch.empty(self.active_streams * self.kr, dtype=dtype))
        self.post_scale = nn.Parameter(torch.empty(1, dtype=dtype))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """按 ``pre_fn`` 的 sigmoid 门控把多流塌缩成单流。

        Args:
            hidden_streams: ``[s, b, hc, H]``。

        Returns:
            ``[s, b, H]``。
        """
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre = torch.sigmoid(
            F.linear(flat, self.pre_fn.float()) * self.pre_scale.float() + self.pre_base.float()
        )
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)

    def write_back(
        self, hidden_streams: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        """把子层输出写回选中的流，其余流保持不变。

        选中集合 = 前 ``fixed_streams`` 个流（权重恒为 1）与按 route 打分取 top-k 的流
        （权重为对应打分，softmax 之外的 sigmoid 值）。MLP 侧的 ``kr > 1``：子层输出先与
        因果深度卷积的 Gram–Schmidt 正交基拼成增广基，再由 ``post`` 门控线性组合。

        Args:
            hidden_streams: ``[s, b, hc, H]``。
            sublayer_output: ``[s, b, H]`` 子层输出（注意力或 MLP）。

        Returns:
            写回后的 ``[s, b, hc, H]``。
        """
        s, b, hc, hidden = hidden_streams.shape

        flat = self.route_norm(
            hidden_streams.flatten(start_dim=2).to(self.route_norm.weight.dtype)
        ).float()
        route_scores = torch.sigmoid(
            F.linear(flat, self.route_fn.float()) * self.route_scale.float()
            + self.route_base.float()
        )
        fixed_mask = torch.arange(hc, device=route_scores.device) < self.fixed_streams
        route_scores = route_scores.masked_fill(fixed_mask.view(1, 1, -1), -float("inf"))
        fixed_idx = torch.arange(self.fixed_streams, device=hidden_streams.device)
        fixed_idx = fixed_idx.view(1, 1, -1).expand(s, b, -1)
        routed_idx = route_scores.topk(self.routed_streams, dim=-1).indices
        active_idx = torch.cat([fixed_idx, routed_idx], dim=-1)
        p = torch.cat(
            [
                torch.ones_like(fixed_idx, dtype=route_scores.dtype),
                route_scores.gather(-1, routed_idx),
            ],
            dim=-1,
        )

        if self.is_mlp:
            # 卷积沿序列方向；把 [s, b, H] 转成 conv1d 需要的 [b, H, s]
            x = sublayer_output.transpose(0, 1).transpose(1, 2).to(
                self.temporal_convs[0].weight.dtype
            )
            conv_outs = [conv(x)[..., :s] for conv in self.temporal_convs]
            ortho = []
            prevs = [x]
            for conv_out in conv_outs:
                basis = conv_out
                for prev in prevs:
                    denom = (prev * prev).sum(dim=1, keepdim=True).clamp_min(self.input_norm.eps)
                    basis = basis - ((prev * basis).sum(dim=1, keepdim=True) / denom) * prev
                ortho.append(basis)
                prevs.append(basis)
            out_aug = (
                torch.cat([x] + ortho, dim=1)
                .transpose(1, 2)
                .reshape(b, s, self.kr, hidden)
                .transpose(0, 1)
                .float()
            )
        else:
            out_aug = sublayer_output.float().unsqueeze(-2)

        active_streams = hidden_streams.gather(
            2, active_idx.unsqueeze(-1).expand(-1, -1, -1, hidden)
        )
        post = 2 * torch.sigmoid(
            F.linear(
                self.input_norm(active_streams.flatten(start_dim=2).float()), self.post_fn.float()
            ).view(s, b, self.active_streams, self.kr)
            * self.post_scale.float()
            + self.post_base.float().view(self.active_streams, self.kr)
        )

        delta = torch.einsum("sbkr,sbrh->sbkh", post, out_aug) * p.unsqueeze(-1)
        updated_active = delta.to(hidden_streams.dtype)
        return hidden_streams.scatter(
            2, active_idx.unsqueeze(-1).expand(-1, -1, -1, hidden), updated_active
        )
