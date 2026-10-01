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
"""RealFormer（残差注意力）的 mcore 实现：把跨层累加的注意力分数加到 softmax 之前。

上游：``google-research/google-research/realformer/realformer.py``（ACL-IJCNLP 2021 Findings，
arXiv:2012.11747），已按原样 vendored 在 ``models/transformers/upstream/realformer_realformer.py``
（sha256 见该目录的 ``PROVENANCE.md``），PyTorch 转写在
``models/transformers/upstream/realformer_torch_reference.py``。

上游算子（vendored 文件 ``residual_attention_layer``，820-851 行）::

    attention_scores = QK^T / sqrt(d_head)                 # 820-822
    cur_attention    = attention_scores                    # 824
    if prev_attention is not None:
        cur_attention += prev_attention                    # 825-826
    attention_logits = cur_attention                       # 828
    if use_running_mean:
        attention_logits /= (num_prev_layers + 1.0)        # 829-830
    attention_logits += (1 - mask) * -10000.0              # 836-840
    attention_probs   = softmax(attention_logits)          # 845
    context_layer     = attention_probs @ V                # 849
    return context_layer, cur_attention                    # 851  ← 传给下一层

本文件与它的关系，逐条说清：

1. **加法顺序一字未改**：`scores → (+prev) → (/层数) → (+mask) → softmax`，其中 mask 用 mcore 的
   ``attention_mask_func``（``True`` 的位置填 −10000，与上游的 ``(1-mask)*-10000`` 同效）。
2. **多出来的 gate**：上游没有 gate。这里把加的那一项乘一个逐层 gate（``0`` = 恒等锚点、
   ``0+δ``（零初始化，默认）= 恒等起手但可学习、``1`` = 上游原样）。layer 1（第一个没有
   ``prev`` 的层）不建 gate —— 上游在那里也整句跳过加法，建了只有死梯度。
3. **必须 eager**：残差注意力**就是**那份分数矩阵（上游每层都物化 ``[b, heads, s, s]``），
   所以 flash/paged 那类不吐分数的内核用不了；本实现在
   ``DotProductAttention``（mcore 的 torch/eager core attention）上做，body 与原实现逐行一致，
   只在标注处插入两段。
4. **跨层状态怎么传**：分数矩阵没法塞进 ``hidden_states``（宽度会变成 ``heads·s·s``），所以
   用一份 :class:`RealFormerCarry` 在**同一 model 的各层之间**共享（由 spec 的 ``params`` 交给
   每层）。每层把 ``layer_number`` 记进 carry：层 ``l`` 只接受 ``l-1`` 写下的分数，层 1 每
   次前向重置 —— 于是"一个 spec 对象对应一个前向中的模型"这个不变量**由断言守着**，
   错用（两个模型交错前向）会当场报错而不是静默算错。
5. **pp > 1 不支持**：``[b, heads, s, s]`` 的跨层状态过不了 stage 边界（p2p 的张量形状由
   config 决定），layer 里直接拒绝并说明原因；``recompute_granularity='full'`` 同理拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [
    "RealFormerCarry",
    "RealFormerCoreAttention",
    "build_realformer_submodules",
    "realformer_knobs_from_kwargs",
    "REALFORMER_GATE_MODES",
]

#: gate 三档（与 HF 侧的 ``attn_res_realformer_gate`` 同名同义）
REALFORMER_GATE_MODES = ("deviation", "zero", "one")


@dataclass
class RealFormerCarry:
    """跨层共享的残差注意力状态（上游的 ``attention_scores`` 累加和，softmax **之前**）。

    ``last_layer`` 是"这份状态是谁写的"：层 ``l`` 只接受 ``l-1`` 写的（层 1 不受限，它每次
    前向重置）。这个断言把"两个模型交错前向却共用一份 spec"这种误用变成当场报错。
    """

    scores: Tensor | None = None
    last_layer: int = 0

    def take(self, layer_number: int) -> Tensor | None:
        """取走上一层写的分数；层 1 重置后返回 ``None``（上游 layer 0 的 ``prev_attention``）。"""
        if layer_number == 1:
            self.scores = None
            self.last_layer = 0
            return None
        if self.scores is None or self.last_layer != layer_number - 1:
            raise RuntimeError(
                f"RealFormer carry 的顺序不成立：层 {layer_number} 期待层 {layer_number - 1} 写下的"
                f"分数，实际 last_layer={self.last_layer}，scores={'None' if self.scores is None else '有'}。"
                " 这份 carry 是同一 spec 的各层共享的；出现这个错误通常意味着**两个模型**（例如 actor 与"
                " ref）共用了同一个 spec 对象并交错前向 —— 给每个模型各自的 spec（本配方的 bridge 与"
                " 训练入口都是这么做的）。"
            )
        return self.scores

    def store(self, scores: Tensor, layer_number: int) -> None:
        self.scores = scores
        self.last_layer = layer_number


class RealFormerCoreAttention(DotProductAttention):
    """``DotProductAttention`` + 上游 residual_attention_layer 的两处插入。

    body 从 mcore 的 ``megatron/core/transformer/dot_product_attention.py::DotProductAttention.forward``
    逐行照抄（含 GQA 的 ``repeat_interleave``、``baddbmm`` 的 ``alpha=softmax_scale``、fp32 softmax、
    dropout 的 RNG tracker 分支），两处插入用 ``# --- RealFormer ---`` 标出。
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        softmax_scale: float | None = None,
        carry: RealFormerCarry | None = None,
        gate_mode: str = "deviation",
        use_running_mean: bool = False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            softmax_scale=softmax_scale,
            **kwargs,
        )
        if gate_mode not in REALFORMER_GATE_MODES:
            raise ValueError(f"realformer gate_mode={gate_mode!r} 不在 {REALFORMER_GATE_MODES}")
        self.carry = carry if carry is not None else RealFormerCarry()
        self.gate_mode = gate_mode
        self.use_running_mean = bool(use_running_mean)
        # layer 1 没有可加的上一层（上游 prev_attention=None），所以不建 gate。
        if self.layer_number >= 2:
            if gate_mode == "deviation":
                # 零初始化：init 时整条连接恒等（加 0），训练中学"加多少"。
                self.carry_gate = torch.nn.Parameter(torch.zeros(1))
            else:
                value = 0.0 if gate_mode == "zero" else 1.0
                # 非持久：常量不进检查点（转换表只认 deviation 档的 carry_gate 参数）
                self.register_buffer(
                    "carry_gate_const", torch.full((1,), value), persistent=False
                )

    # -- gate ---------------------------------------------------------------
    def _gate(self) -> Tensor:
        """逐层 gate（layer 1 没有可加项，恒 0）。"""
        if self.layer_number < 2:
            return torch.zeros((), device=self._device())
        if self.gate_mode == "deviation":
            return self.carry_gate
        return self.carry_gate_const

    def reset_parameters(self) -> None:
        """恒等锚点：``deviation`` 档的 gate 归零（``zero``/``one`` 是常量 buffer）。"""
        if self.layer_number >= 2 and self.gate_mode == "deviation":
            with torch.no_grad():
                self.carry_gate.zero_()

    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
    ):
        """见类 docstring；插入点在下文以 ``RealFormer`` 注释标出。"""
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        import megatron.core.parallel_state as parallel_state

        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),
            key.transpose(0, 1).transpose(1, 2),
            beta=0.0,
            alpha=self.softmax_scale,
        )
        # [b, np, sq, sk] —— 上游的 `attention_scores`（= QK^T / sqrt(d_head)）
        attention_scores = matmul_result.view(*output_size)

        if self.softcap is not None:
            attention_scores = self.softcap * torch.tanh(attention_scores / self.softcap)

        # --- RealFormer 插入 1（上游 824-826）：加上上一层传下来的累加分数 ---
        prev = self.carry.take(self.layer_number)
        if prev is not None:
            attention_scores = attention_scores + self._gate().to(attention_scores.dtype) * prev
        cur_attention = attention_scores
        logits = cur_attention
        # --- RealFormer 插入 2（上游 829-830）：running mean 只除本层 logits ---
        if self.use_running_mean:
            logits = logits / float(self.layer_number)
        # 之后的 mask + softmax + dropout + 加权求和与父类逐行一致（uppstream 836-849）。
        attention_probs: Tensor = self.scale_mask_softmax(
            logits, attention_mask, self.softmax_offset
        )
        if not self.config.sequence_parallel:
            from megatron.core import tensor_parallel

            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        output_size = (value.size(1), value.size(2), query.size(0), value.size(3))
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context = torch.bmm(attention_probs, value.transpose(0, 1))
        context = context.view(*output_size)
        context = context.permute(2, 0, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        # --- RealFormer 插入 3（上游 851）：把这份累加分数交给下一层 ---
        self.carry.store(cur_attention, self.layer_number)
        return context


def build_realformer_submodules(
    config: TransformerConfig,
    *,
    gate_mode: str = "deviation",
    use_running_mean: bool = False,
    carry: RealFormerCarry | None = None,
):
    """local 子模块 + 把 ``core_attention`` 换成 :class:`RealFormerCoreAttention`。

    除 core attention 之外与 :func:`gdar_layer.build_gdar_submodules` 完全同一次调用、同参数、
    同顺序（RNG 抽取一致 ⇒ ``gate=0`` 时与 plain Qwen3 逐位相同的恒等成立）。
    """
    submodules = get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,  # 0.20 的第 5 位是 fp8 槽；本配方走稠密 local 路径
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
    )
    old = submodules.self_attention.submodules.core_attention
    params = dict(getattr(old, "params", {}) or {})
    params.update(
        gate_mode=gate_mode,
        use_running_mean=use_running_mean,
        carry=carry if carry is not None else RealFormerCarry(),
    )
    submodules.self_attention.submodules.core_attention = ModuleSpec(
        module=RealFormerCoreAttention, params=params
    )
    return submodules


def realformer_knobs_from_kwargs(kwargs: dict, config: TransformerConfig | None = None):
    """从 spec 的 ``params`` 里取 RealFormer 旋钮（未知键当场报错）。"""
    known = {"gate", "mean"}
    values = {"gate": "deviation", "mean": False}
    for key, value in kwargs.items():
        if key == "realformer_carry":  # spec 传下来的共享状态，不是旋钮
            continue
        if not key.startswith("realformer_"):
            continue
        name = key[len("realformer_") :]
        if name not in known:
            raise TypeError(f"未知 RealFormer 旋钮 {key!r}；可用：{sorted('realformer_' + k for k in known)}")
        values[name] = value
    if values["gate"] not in REALFORMER_GATE_MODES:
        raise ValueError(f"realformer_gate={values['gate']!r} 不在 {REALFORMER_GATE_MODES}")
    return values
