# coding=utf-8
# Copyright 2026 The FlagOS Contributors and HuggingFace Inc. team. All rights reserved.
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
"""Qwen3 + RealFormer（残差注意力）配置。

**对齐上游**：``google-research/google-research/realformer/realformer.py``（ACL-IJCNLP 2021
Findings，"RealFormer: Transformer Likes Residual Attention"，arXiv:2012.11747）。官方文件已按原样
vendored 在 ``upstream/realformer_realformer.py``（sha256 见 ``upstream/PROVENANCE.md``），
可执行的 PyTorch 转写在 ``upstream/realformer_torch_reference.py``。

上游算子（``residual_attention_layer``，vendored 文件 820-836 行）::

    attention_scores = QK^T / sqrt(d_head)                 # 缩放后的分数
    cur_attention   = attention_scores
    if prev_attention is not None:
        cur_attention += prev_attention                    # 跨层累加，softmax **之前**
    attention_logits = cur_attention
    if use_running_mean:
        attention_logits /= (num_prev_layers + 1.0)        # 温度 = 已走过的层数（深模型用）
    attention_logits += (1 - mask) * -10000.0
    attention_probs = softmax(attention_logits)
    context = attention_probs @ V
    return context, cur_attention                          # ← 交给下一层当 prev_attention

模型侧（``realformer_model``，919-935 行）：``prev_attention`` 从 ``None`` 起，逐层
``attention_output, prev_attention = residual_attention_layer(..., prev_attention=prev_attention,
num_prev_layers=layer_idx, use_running_mean=...)``。也就是说 **layer 0 就开始往后传**
（layer 0 的 ``cur`` 就等于它自己的分数），从 layer 1 起才真正"加上上一层"。

与上游的唯一差异（本配方加的身份锚点）：上游没有 gate，我们给加法项乘一个逐层 gate::

    cur_attention = attention_scores + gate * prev_attention

``attn_res_realformer_gate`` 三档：

* ``"deviation"``（默认）：``gate = 0 + delta``，``delta`` 零初始化 —— 初始化时加的是 0，
  于是整条连接**逐位恒等**（``RealFormer(0) == Qwen3``），训练中学习"加多少残差注意力"；
* ``"zero"``：``gate ≡ 0``（恒等锚点，不学习），tiny 恒等验收与"连接关掉"的对照用它；
* ``"one"``：``gate ≡ 1``，即**上游算子原样**（``scores + prev``，逐位等于上游公式）。

``attn_res_realformer_mean`` 对应上游 ``use_running_mean``：对累加后的 logits 除以
``(num_prev_layers + 1)``（论文说 30 层以上的深模型用 running mean 更好）。注意除法只作用在
本层的 logits 上，**往下一层传的仍是未除的累加和**（上游代码就是这么写的）。
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3RealFormerConfig(Qwen3Config):
    """Qwen3 骨干 + RealFormer 残差注意力。"""

    model_type = "qwen3_realformer"

    #: 残差注意力的 gate 档位：``"deviation"``（默认，恒等初始化 + 可学习）/ ``"zero"``（恒等）/
    #: ``"one"``（上游原样）
    attn_res_realformer_gate: str = "deviation"
    #: 上游 ``use_running_mean``：logits 除以已走过的层数
    attn_res_realformer_mean: bool = False

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` 在 ``trust_remote_code=True`` 时靠它解析；
    #: 两个文件由 ``save_pretrained`` 拷到权重旁边。
    auto_map = {
        "AutoConfig": "configuration_qwen3_realformer.Qwen3RealFormerConfig",
        "AutoModel": "modeling_qwen3_realformer.Qwen3RealFormerModel",
        "AutoModelForCausalLM": "modeling_qwen3_realformer.Qwen3RealFormerForCausalLM",
    }

    def to_dict(self):
        """把本变体的旋钮也序列化（理由同其它变体：v4 的 ``to_dict`` 会丢类属性）。"""
        output = super().to_dict()
        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output


#: 本变体在 ``Qwen3Config`` 之上新增的字段（``to_dict`` 读它）
_EXTRA_CONFIG_FIELDS = tuple(Qwen3RealFormerConfig.__annotations__)

try:
    AutoConfig.register("qwen3_realformer", Qwen3RealFormerConfig)
except ValueError:
    pass

try:
    Qwen3RealFormerConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):
    pass
