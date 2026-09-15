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

"""Shensi 的 Transformer 块：出口收缩 + MoE 参数组共享。

对应 HF ``ShensiModel``：层循环结束后先做一次不带输出归一化的 ``output_attn_res``，再用
``hc_head`` 把多流收缩成单流，最后接共享的 ``norm``。

这里的实现刻意**不覆写块的 forward**（上游的 forward 里有 checkpoint / 卸载 / 重算 /
CUDA graph 等大量记账逻辑），而是：

* 把块的 ``mhc_num_residual_streams`` 置为 1 —— 上游在块首尾调用的
  ``HyperConnectionModule.input_expand`` / ``output_contract`` 在 ``n = 1`` 时都是**恒等**
  （前者复制一份、后者对自己的切片求均值），于是三态打包由层自己负责、出口收缩则交给
  :class:`ShensiOutputContract`；
* 把 :class:`ShensiOutputContract` 放进块的 ``layer_norm`` 槽位（spec 工厂里替换），
  上游在 post_process 阶段会 `apply_module(self.final_layernorm)(hidden_states)`，
  正好是我们需要的"收缩 + 最终归一化"。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock

from .shensi_attention_residual import ShensiAttentionResidual
from .shensi_norms import ShensiUnweightedRMSNorm, ShensiWeightedRMSNorm
from .state import unpack_state

__all__ = ["ShensiHyperHead", "ShensiOutputContract", "ShensiTransformerBlock"]


class ShensiHyperHead(MegatronModule):
    """出口的学习型流收缩，对应 HF ``ShensiHyperHead``。

    与 :class:`~shensi.core.transformer.shensi_hyper_connection.ShensiHyperConnection` 只差
    参数集：这里是 ``hc_fn`` / ``hc_base`` / ``hc_scale``（没有 ``pre_scale`` 作为独立缩放，
    而是把 ``hc_scale`` 直接乘在打分上），且**不加 eps**（HF 的 Shensi 版把 DSv4 的 ``+eps``
    删掉了）。

    Args:
        config: Megatron 配置。
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        dtype = config.params_dtype
        self.num_streams = int(config.mhc_num_residual_streams)
        hidden = int(config.hidden_size)
        self.input_norm = ShensiUnweightedRMSNorm(config)
        self.hc_fn = nn.Parameter(
            torch.empty(self.num_streams, self.num_streams * hidden, dtype=dtype)
        )
        self.hc_base = nn.Parameter(torch.empty(self.num_streams, dtype=dtype))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=dtype))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """``[s, b, hc, H]`` → ``[s, b, H]``。"""
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)


class ShensiOutputContract(MegatronModule):
    """块出口：``output_attn_res`` → ``hc_head`` → 最终 norm。

    它被放在块的最终归一化槽位（``decoder.final_layernorm``），输入是打包三态
    ``[s, b, (2 * hc + blocks * hc) * H]``，输出是 ``[s, b, H]``。

    Args:
        config: Megatron 配置。
        hidden_size: 单流宽度（由块按上游惯例传入）。
        eps: 归一化 epsilon（由块按上游惯例传入）。
    """

    def __init__(self, config, hidden_size: int | None = None, eps: float | None = None) -> None:
        super().__init__(config)
        del hidden_size, eps
        self.num_streams = int(config.mhc_num_residual_streams)
        self.hidden = int(config.hidden_size)
        self.num_blocks = config.attn_res_block_layer_types.count("block_write_layer")
        self.output_attn_res = ShensiAttentionResidual(config)
        self.hc_head = ShensiHyperHead(config)
        self.norm = ShensiWeightedRMSNorm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """收缩三态并做最终归一化。"""
        streams, prefix_sum, library = unpack_state(
            hidden_states, self.num_streams, self.num_blocks, self.hidden
        )
        streams, _, _ = self.output_attn_res(
            streams,
            library,
            prefix_sum,
            output_norm_weight=None,
            num_blocks=self.num_blocks,
        )
        return self.norm(self.hc_head(streams))


class ShensiTransformerBlock(TransformerBlock):
    """Shensi 解码块。

    只做两件事：把流数改成 1（让上游的展开/收缩退化成恒等）与按 HF 的规则共享 MoE 组的
    router / experts 参数。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 上游据此决定块首尾的 input_expand / output_contract；n = 1 时两者都是恒等，
        # 三态打包与出口收缩分别由层和 ShensiOutputContract 负责。
        self.mhc_num_residual_streams = 1
        self.tie_moe_groups()

    def tie_moe_groups(self) -> None:
        """把同一块内读层的 router / experts 别名到该块的持有层（与 HF ``tie_moe_groups`` 一致）。

        HF 的规则：每个 AttentionResidual 块里，**第一个非 hash 层**持有 ``gate`` 与
        ``experts``，其后直到下一个写库层的读层共享同一份参数（HF 检查点保存时会被去重，
        因此读层没有自己的键）。层号取**全局**层号：本 stage 的局部下标在 PP>1 时不从 0 开始。
        """
        plan = self.config.attn_res_block_layer_types
        mlp_types = self.config.mlp_layer_types
        write_layers = [i for i, role in enumerate(plan) if role == "block_write_layer"]

        shared: dict[int, object] = {}
        for layer in self.layers:
            index = layer.layer_number - 1
            if mlp_types[index] == "hash_moe":
                continue
            block_id = max(w for w in write_layers if w <= index)
            if block_id not in shared:
                shared[block_id] = layer.mlp
                continue
            source = shared[block_id]
            if source is not layer.mlp:
                layer.mlp.router = source.router
                layer.mlp.experts = source.experts
