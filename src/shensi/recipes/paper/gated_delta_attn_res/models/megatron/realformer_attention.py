"""RealFormer 的注意力实现：共享 carry + eager core attention。"""

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

REALFORMER_GATE_MODES = ("deviation", "zero", "one")


@dataclass
class RealFormerCarry:

    scores: Tensor | None = None
    last_layer: int = 0

    # 顺序不变量：层 l 只接受 l-1 写下的分数，层 1 每次前向重置
    def take(self, layer_number: int) -> Tensor | None:
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
        if self.layer_number >= 2:
            if gate_mode == "deviation":
                self.carry_gate = torch.nn.Parameter(torch.zeros(1))
            else:
                value = 0.0 if gate_mode == "zero" else 1.0
                self.register_buffer(
                    "carry_gate_const", torch.full((1,), value), persistent=False
                )

    def _gate(self) -> Tensor:
        if self.layer_number < 2:
            return torch.zeros((), device=self._device())
        if self.gate_mode == "deviation":
            return self.carry_gate
        return self.carry_gate_const

    def reset_parameters(self) -> None:
        if self.layer_number >= 2 and self.gate_mode == "deviation":
            with torch.no_grad():
                self.carry_gate.zero_()

    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

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
        attention_scores = matmul_result.view(*output_size)

        if self.softcap is not None:
            attention_scores = self.softcap * torch.tanh(attention_scores / self.softcap)

        # 残差注意力就是这份分数矩阵，所以只能走 eager（flash/paged 不吐分数）
        prev = self.carry.take(self.layer_number)
        if prev is not None:
            attention_scores = attention_scores + self._gate().to(attention_scores.dtype) * prev
        cur_attention = attention_scores
        logits = cur_attention
        if self.use_running_mean:
            logits = logits / float(self.layer_number)
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

        self.carry.store(cur_attention, self.layer_number)
        return context


def build_realformer_submodules(
    config: TransformerConfig,
    *,
    gate_mode: str = "deviation",
    use_running_mean: bool = False,
    carry: RealFormerCarry | None = None,
):
    submodules = get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,
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
    known = {"gate", "mean"}
    values = {"gate": "deviation", "mean": False}
    for key, value in kwargs.items():
        if key == "realformer_carry":
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
