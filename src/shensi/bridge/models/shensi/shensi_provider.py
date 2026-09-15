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

"""Shensi 的 model provider。

上游 ``GPTModelProvider.provide()`` 直接使用模块级的 ``MCoreGPTModel`` 与
``gpt_model.TransformerBlock`` 构造模型，没有可覆写的类属性；在不改动上游源码的前提下，
这里在调用父类实现期间把这两个符号临时指向 Shensi 的实现，结束后恢复（构造参数与父类完全
一致）。Shensi 专有字段（``attn_res_block_layer_types`` / ``mlp_layer_types`` / ``hc_*``）
由 bridge 在 ``provider_bridge`` 里设置，provider 只负责把它们带进构造出的模型。
"""

from __future__ import annotations

from megatron.bridge.models.mla_provider import MLAModelProvider

from ....core.models.gpt.shensi_model import ShensiGPTModel
from ....core.transformer.shensi_block import ShensiTransformerBlock

__all__ = ["ShensiModelProvider"]


class ShensiModelProvider(MLAModelProvider):
    """Shensi provider：构造模型时换上 Shensi 的模型类与块类。"""

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """构造模型；期间替换两个模块级符号（见模块 docstring）。"""
        from megatron.bridge.models import gpt_provider
        from megatron.core.models.gpt import gpt_model as mcore_gpt_model

        original_model = gpt_provider.MCoreGPTModel
        original_block = mcore_gpt_model.TransformerBlock
        gpt_provider.MCoreGPTModel = ShensiGPTModel
        mcore_gpt_model.TransformerBlock = ShensiTransformerBlock
        try:
            return super().provide(
                pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
            )
        finally:
            gpt_provider.MCoreGPTModel = original_model
            mcore_gpt_model.TransformerBlock = original_block
