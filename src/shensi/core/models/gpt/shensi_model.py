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

"""Shensi 的 GPT 模型：只补一件事 —— 把 ``input_ids`` 交给 hash 层。

上游 ``GPTModel`` 的其余部分（embedding、块、输出层、损失组装、PP 处理）全部继承。
"""

from __future__ import annotations

from megatron.core.models.gpt.gpt_model import GPTModel

from ...transformer.context import set_current_input_ids

__all__ = ["ShensiGPTModel"]


class ShensiGPTModel(GPTModel):
    """Shensi 的 GPT 模型。"""

    def forward(self, *args, **kwargs):
        """记录 ``input_ids`` 后走上游实现。

        hash 层的 ``deepemb`` 门控需要 token id（见 :mod:`shensi.core.transformer.context`），
        而上游的 MLP 调用链没有传递它的通道，因此在模型入口处暂存一次。
        """
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        set_current_input_ids(input_ids)
        return super().forward(*args, **kwargs)
