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

"""把当前 step 的 ``input_ids`` 交给 hash 层。

HF 的 ``ShensiHashMLP`` 用 ``deepemb(input_ids)`` 做门控，而 Megatron 的 MLP 槽位只接收
hidden states（上游在 ``_run_mlp`` 里按 ``apply_module(self.mlp)`` 调用，没有传 ``input_ids``
的通道）。这里用一个**模块级**暂存把当前 step 的 ids 放进去，供 :class:`ShensiHashMLP` 取用。

为什么不是 ``with`` 上下文：激活重算在**反向**期间重新执行前向，那时上下文早已退出（实测
hash 层会拿到空值）。模块级暂存保留的是最近一次前向的 ids，而重算总是紧随该 microbatch 的
前向之后发生，因此取到的是正确的 microbatch。代价是同一进程内不能交错执行两个 microbatch
的前向与反向（PP>1 的异步调度会这样，但超连接模型在上游本就不支持 PP>1）。
"""

from __future__ import annotations

import torch

__all__ = ["set_current_input_ids", "get_current_input_ids"]

_current_input_ids: torch.Tensor | None = None


def set_current_input_ids(input_ids: torch.Tensor | None) -> None:
    """记录当前 step 的 ``input_ids``（``[b, s]``）。"""
    global _current_input_ids
    _current_input_ids = input_ids


def get_current_input_ids() -> torch.Tensor | None:
    """取出最近一次前向的 ``input_ids``；未设置时返回 ``None``。"""
    return _current_input_ids
