# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
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
"""RealFormer 的层规格（``--spec`` 对象），与其它变体同构。

上游：He, Ravula, Kanagal & Ainslie, "RealFormer: Transformer Likes Residual Attention"
(arXiv:2012.11747, Findings of ACL-IJCNLP 2021)；算子语义与三类 gate 见
``realformer_attention.py`` 的模块 docstring。

四个预设：:

    realformer_layer_spec            gate="deviation"（默认）：恒等初始化 + 可学习加法强度
    realformer_layer_spec_identity   gate="zero"      ：恒等锚点（RealFormer(0) == Qwen3 逐位）
    realformer_layer_spec_reference  gate="one"       ：**上游原样**（scores + prev，无 gate）
    realformer_layer_spec_mean       gate="deviation" + running mean（上游 use_running_mean）

用法（与其它变体一致）::

    python train.py --model-algo qwen3_realformer            # deviation，论文对照臂
    python train.py --model-algo qwen3_realformer_reference  # 上游原样
"""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .realformer_attention import RealFormerCarry
from .realformer_layer import RealFormerTransformerLayer

__all__ = [
    "make_realformer_spec",
    "realformer_layer_spec",
    "realformer_layer_spec_identity",
    "realformer_layer_spec_mean",
    "realformer_layer_spec_reference",
]

#: 默认：gate = 0 + δ（零初始化）—— init 时逐位恒等，训练中学加法强度
_DEFAULT = dict(realformer_gate="deviation", realformer_mean=False)


def make_realformer_spec(**knobs) -> ModuleSpec:
    """RealFormer 层规格，``realformer_*`` 旋钮覆盖默认值。

    **carry 在这里建**（一份 spec 一个对象），各层通过 ``params`` 拿到同一份 —— 跨层状态必须
    是同一个对象，layer 自己建的话每层一份、上一层的分数传不过去。副作用是"同一份 spec 对象
    并发喂两个模型"会撞 carry 的顺序断言（见 ``realformer_attention.RealFormerCarry``）：本配方
    的每个入口都是"一个模型一份 spec"，断言是为了让误用当场暴露而不是静默算错。
    """
    params = dict(_DEFAULT)
    params.update(knobs)
    params.setdefault("realformer_carry", RealFormerCarry())
    return ModuleSpec(module=RealFormerTransformerLayer, submodules=None, params=params)


#: 论文对照臂的默认形态（恒等初始化 + 可学习）
realformer_layer_spec: ModuleSpec = make_realformer_spec()

#: 恒等锚点：加法恒为 0，用于 tiny 的"RealFormer(0) == Qwen3 逐位"验收与"连接关掉"的对照
realformer_layer_spec_identity: ModuleSpec = make_realformer_spec(realformer_gate="zero")

#: **上游原样**：`cur = scores + prev`（没有 gate），用于与官方实现对拍
realformer_layer_spec_reference: ModuleSpec = make_realformer_spec(realformer_gate="one")

#: 上游的 running mean（对累加 logits 除以层数；论文说 30 层以上的深模型用它更好）
realformer_layer_spec_mean: ModuleSpec = make_realformer_spec(
    realformer_gate="deviation", realformer_mean=True
)
