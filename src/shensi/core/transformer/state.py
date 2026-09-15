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

"""Shensi 层间三态（多流隐状态 / prefix_sum / 块库）的打包与解包。

HF 的 ``ShensiDecoderLayer`` 在层与层之间传三个张量（形状均为 batch-first）：

* ``hidden_states`` ``[b, s, hc, H]``：当前的多流隐状态；
* ``prefix_sum`` ``[b, s, hc, H]``：沿层累加的 delta-rule 记忆；
* ``residual`` ``[b, s, hc, blocks, H]``：每个 AttentionResidual 块写入的块库。

Megatron 的层接口只给一个 ``hidden_states``，且是 sequence-first。因此这里把三态拼成一个
张量 ``[s, b, (2 * hc + blocks * hc) * H]`` 让它随层循环流动：拼接顺序固定为
``streams | prefix_sum | library``，两部分各占 ``hc * H`` 宽，库占 ``blocks * hc * H``。
"""

from __future__ import annotations

import torch

__all__ = ["pack_state", "unpack_state", "packed_width"]


def packed_width(num_streams: int, num_blocks: int, hidden_size: int) -> int:
    """打包后每个 token 的特征宽度。

    Args:
        num_streams: 多流隐状态的流数 ``hc``。
        num_blocks: 块库中的块数（即 block_write_layer 的个数）。
        hidden_size: 单流的隐层宽度 ``H``。

    Returns:
        打包张量的最后一维宽度。
    """
    return (2 * num_streams + num_blocks * num_streams) * hidden_size


def pack_state(
    streams: torch.Tensor, prefix_sum: torch.Tensor, library: torch.Tensor
) -> torch.Tensor:
    """把三态拼成 ``[s, b, (2 * hc + blocks * hc) * H]``。

    Args:
        streams: ``[s, b, hc, H]`` 多流隐状态。
        prefix_sum: ``[s, b, hc, H]`` 沿层累加的记忆。
        library: ``[s, b, hc, blocks, H]`` 块库。

    Returns:
        拼接后的张量。
    """
    s, b, _, _ = streams.shape
    return torch.cat(
        [streams.reshape(s, b, -1), prefix_sum.reshape(s, b, -1), library.reshape(s, b, -1)],
        dim=-1,
    )


def unpack_state(
    packed: torch.Tensor, num_streams: int, num_blocks: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`pack_state` 的逆操作。

    Args:
        packed: ``[s, b, (2 * hc + blocks * hc) * H]`` 的打包张量。
        num_streams: 流数 ``hc``。
        num_blocks: 块数。
        hidden_size: 单流宽度 ``H``。

    Returns:
        ``(streams, prefix_sum, library)``，形状分别为 ``[s, b, hc, H]``、
        ``[s, b, hc, H]``、``[s, b, hc, blocks, H]``。
    """
    s, b, _ = packed.shape
    hidden = num_streams * hidden_size
    streams = packed[..., :hidden].reshape(s, b, num_streams, hidden_size)
    prefix_sum = packed[..., hidden : 2 * hidden].reshape(s, b, num_streams, hidden_size)
    library = packed[..., 2 * hidden :].reshape(s, b, num_streams, num_blocks, hidden_size)
    return streams, prefix_sum, library
