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

"""Shensi 的层 spec 工厂。

做法是"先取上游产物、再就地替换"：调用
``megatron.core.models.gpt.experimental_attention_variant_module_specs.
get_transformer_block_with_experimental_attention_variant_spec`` 拿到 DSv4 hybrid 的
``TransformerBlockSubmodules``（注意力 / 压缩器 / Indexer / MoE 全部来自上游），然后只替换三处：

1. 每层的 ``module``：换成 :class:`~shensi.core.transformer.shensi_layer.ShensiTransformerLayer`；
2. 两个超连接槽位：换成 :class:`~shensi.core.transformer.shensi_hyper_connection.ShensiHyperConnection`
   （上游那份是 mHC 的另一套参数化，不复用）；
3. hash 层的 ``mlp``：换成 :class:`~shensi.core.transformer.shensi_hash_mlp.ShensiHashMLP`；
4. 块的 ``layer_norm``：换成 :class:`~shensi.core.transformer.shensi_block.ShensiOutputContract`
   （承担出口收缩 + 最终归一化）。
"""

from __future__ import annotations

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from ...transformer.shensi_block import ShensiOutputContract
from ...transformer.shensi_hash_mlp import ShensiHashMLP
from ...transformer.shensi_hyper_connection import ShensiHyperConnection
from ...transformer.shensi_layer import ShensiTransformerLayer

__all__ = ["get_shensi_spec"]


def get_shensi_spec(config, vp_stage: int | None = None):
    """构建 Shensi 的块 submodules。

    Args:
        config: Megatron 配置（需带 ``mlp_layer_types`` 等 Shensi 专有字段）。
        vp_stage: 虚拟流水段号，透传给上游工厂。

    Returns:
        ``TransformerBlockSubmodules``。
    """
    block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config, vp_stage=vp_stage
    )

    mlp_layer_types = list(config.mlp_layer_types)
    for index, layer_spec in enumerate(block_spec.layer_specs):
        layer_spec.module = ShensiTransformerLayer
        layer_spec.submodules.self_attention_hyper_connection = ModuleSpec(
            module=ShensiHyperConnection, params={"is_mlp": False}
        )
        layer_spec.submodules.mlp_hyper_connection = ModuleSpec(
            module=ShensiHyperConnection, params={"is_mlp": True}
        )
        if mlp_layer_types[index] == "hash_moe":
            layer_spec.submodules.mlp = ModuleSpec(module=ShensiHashMLP)

    block_spec.layer_norm = ModuleSpec(module=ShensiOutputContract)
    return block_spec
