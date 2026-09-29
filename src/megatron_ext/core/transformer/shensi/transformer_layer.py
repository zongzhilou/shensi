# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


from dataclasses import dataclass
from typing import Any, Optional, Union
import torch
from torch import Tensor
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from .attn_res import (
    ShensiAttentionResidual,
    attn_res_block_layer_types,
    prev_valid_blocks,
)


@dataclass
class ShensiTransformerLayerSubmodules(TransformerLayerSubmodules):
    self_attention_attn_res: Union[ModuleSpec, type] = IdentityOp
    mlp_attn_res: Union[ModuleSpec, type] = IdentityOp


def attn_res_to_streams(
    hidden_states: Tensor, *, hc_mult: int, hidden_size: int, is_first_in_block: bool
) -> Tensor:
    s, b = hidden_states.shape[0], hidden_states.shape[1]
    if hidden_states.dim() == 4:
        return hidden_states
    if is_first_in_block:
        if hidden_states.dim() == 3 and hidden_states.shape[-1] == hidden_size:
            return hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
        if (
            hidden_states.dim() == 3
            and hidden_states.shape[-1] == hc_mult * hidden_size
        ):
            return hidden_states.view(s, b, hc_mult, hidden_size)
        raise ValueError(
            f"block 第一层的输入应为 1 流 [s, b, {hidden_size}]、扁平 n 流 "
            f"[s, b, {hc_mult * hidden_size}] 或已展开的 4D n 流，"
            f"得到 {tuple(hidden_states.shape)}"
        )
    if hidden_states.dim() == 4:
        return hidden_states
    if hidden_states.dim() == 3 and hidden_states.shape[-1] == hc_mult * hidden_size:
        return hidden_states.view(s, b, hc_mult, hidden_size)
    raise ValueError(
        f"层输入应为扁平 n 流 [s, b, {hc_mult * hidden_size}]，得到 "
        f"{tuple(hidden_states.shape)}"
    )


def attn_res_to_flat(streams: Tensor) -> Tensor:
    return streams.reshape(streams.shape[0], streams.shape[1], -1)


def shensi_attn_res_layer_forward(
    *,
    state,
    hidden_states: Tensor,
    layer_number: int,
    hc_mult: int,
    hidden_size: int,
    is_first_in_block: bool,
    is_block_write_layer: bool,
    prev_valid_blocks: int,
    attn_res,
    mlp_attn_res,
    attn_hc,
    ffn_hc,
    input_norm_weight: Tensor,
    pre_mlp_norm_weight: Tensor,
    attn_fn,
    mlp_fn,
    handoff=None,
    packed_seq_params=None,
) -> Tensor:
    layer_idx = int(layer_number) - 1
    streams = attn_res_to_streams(
        hidden_states,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        is_first_in_block=is_first_in_block,
    )
    if handoff is not None:
        handoff.import_layer(state, layer_idx, hidden_flat=attn_res_to_flat(streams))
    if is_first_in_block:
        state.begin(streams)
        delta = None
    else:
        state.check_ready(layer_number)
        delta = streams - state.prefix_sum
    if is_block_write_layer:
        state.write_block(prev_valid_blocks, state.prefix_sum)
    attn_in, prefix_sum = attn_res(
        state.prefix_sum,
        delta,
        state.residual,
        output_norm_weight=input_norm_weight,
        num_blocks=prev_valid_blocks,
    )
    if is_block_write_layer:
        prefix_sum = None
    collapsed = attn_hc(attn_in)
    attn_output = attn_fn(collapsed)
    streams = attn_hc.write_back(attn_in, attn_output)
    prefix_sum = streams if prefix_sum is None else prefix_sum + streams
    mlp_in, prefix_sum = mlp_attn_res(
        prefix_sum,
        prefix_sum,
        state.residual,
        output_norm_weight=pre_mlp_norm_weight,
        num_blocks=prev_valid_blocks + int(is_block_write_layer),
    )
    collapsed = ffn_hc(mlp_in)
    mlp_output = mlp_fn(collapsed)
    streams = ffn_hc.write_back(mlp_in, mlp_output, packed_seq_params)
    state.prefix_sum = prefix_sum + streams
    flat = attn_res_to_flat(streams)
    if handoff is not None:
        flat = handoff.export_layer(state, layer_idx, stage_output=flat)
    return flat


class ShensiTransformerLayer(TransformerLayer):
    def __init__(
        self,
        config,
        submodules: ShensiTransformerLayerSubmodules,
        layer_number: int = 1,
        is_mtp_layer: bool = False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            is_mtp_layer=is_mtp_layer,
            **kwargs,
        )
        self.hc_mult = int(config.num_residual_streams)
        self.n_hash_layers = int(config.moe_n_hash_layers)
        self.attn_res_block_size = int(config.attn_res_block_size)
        layer_idx = self.layer_number - 1
        if not 0 <= layer_idx < config.num_layers:
            raise ValueError(
                f"layer_number={self.layer_number} 超出 decoder 范围（num_layers={config.num_layers}）"
            )
        types = attn_res_block_layer_types(
            config.num_layers, self.n_hash_layers, self.attn_res_block_size
        )
        self.is_mtp_layer = bool(is_mtp_layer)
        if self.is_mtp_layer:
            self.is_block_write_layer = True
            self.prev_valid_blocks = 0
            self.is_first_in_block = True
            self.is_hash = False
        else:
            self.is_block_write_layer = types[layer_idx] == "block_write_layer"
            self.prev_valid_blocks = prev_valid_blocks(
                layer_idx,
                config.num_layers,
                self.n_hash_layers,
                self.attn_res_block_size,
            )
            self.is_first_in_block = self.layer_number == 1
            self.is_hash = layer_idx < self.n_hash_layers
        for name, norm in (
            ("input_layernorm", self.input_layernorm),
            ("pre_mlp_layernorm", self.pre_mlp_layernorm),
        ):
            if not hasattr(norm, "weight"):
                raise RuntimeError(
                    f"ShensiTransformerLayer 需要 {name} 提供 .weight（AttnRes 的 "
                    "output_norm_weight）；当前是 "
                    f"{type(norm).__name__}。请勿把该 layernorm fuse 成 IdentityOp。"
                )
        self.attn_hc = build_module(
            submodules.self_attention_hyper_connection,
            config=config,
            layer_number=self.layer_number,
        )
        self.ffn_hc = build_module(
            submodules.mlp_hyper_connection,
            config=config,
            layer_number=self.layer_number,
        )
        if (
            not isinstance(self.attn_hc, torch.nn.Module)
            or self.attn_hc.__class__ is ShensiAttentionResidual
        ):
            raise TypeError(
                "self_attention_hyper_connection 必须是 ShensiHyperConnection"
            )
        self.self_attention_attn_res = build_module(
            submodules.self_attention_attn_res, config=config
        )
        self.mlp_attn_res = build_module(submodules.mlp_attn_res, config=config)
        self.attn_res_state = None
        self._attn_res_recompute_plan = None
        self.attn_res_handoff = None

    def _to_streams(self, hidden_states: Tensor) -> Tensor:
        return attn_res_to_streams(
            hidden_states,
            hc_mult=self.hc_mult,
            hidden_size=int(self.config.hidden_size),
            is_first_in_block=self.is_first_in_block,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager: Optional[Any] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        if getattr(self, "_attn_res_recompute_plan", None) is None:
            from .attn_res import build_attn_res_recompute_plan

            self._attn_res_recompute_plan = build_attn_res_recompute_plan(self.config)
        if not self._attn_res_recompute_plan.safe:
            raise NotImplementedError(
                "[shensi][attn_res] 检测到与 AttnRes 状态机**不兼容**的 recompute(full) 形状："
                f"{self._attn_res_recompute_plan.describe()}。详见 "
                "attn_res.check_attn_res_recompute_support 的说明（上游依据与安全形状清单）。"
            )
        if mhc_recompute_manager is not None:
            raise NotImplementedError(
                "ShensiTransformerLayer 不支持 mhc_recompute（AttnRes 状态是 block 内部"
                "状态，需要与 mHC 的重算管理一起设计）；请勿在 recompute_modules 里加 'mhc'。"
            )
        state = self.attn_res_state
        if state is None:
            raise RuntimeError(
                "AttnRes 状态未绑定：请在模型构造后调用 "
                "attn_res.bind_attn_res_state(model.decoder, config, num_blocks)。"
            )

        def _attn_fn(collapsed):
            attention_output_with_bias = self.self_attention(
                collapsed,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            attn_output, attn_bias = (
                attention_output_with_bias
                if isinstance(attention_output_with_bias, tuple)
                else (attention_output_with_bias, None)
            )
            if attn_bias is not None:
                raise NotImplementedError(
                    "Shensi 的注意力不带 bias（add_bias_linear=False）"
                )
            return attn_output

        def _mlp_fn(collapsed):
            mlp_output_with_bias = (
                self.mlp(collapsed, input_ids=input_ids)
                if self.is_hash
                else self.mlp(collapsed)
            )
            mlp_output, mlp_bias = (
                mlp_output_with_bias
                if isinstance(mlp_output_with_bias, tuple)
                else (mlp_output_with_bias, None)
            )
            if mlp_bias is not None:
                raise NotImplementedError(
                    "Shensi 的 MLP/MoE 不带 bias（add_bias_linear=False）"
                )
            return mlp_output

        flat = shensi_attn_res_layer_forward(
            state=state,
            hidden_states=hidden_states,
            layer_number=self.layer_number,
            hc_mult=self.hc_mult,
            hidden_size=int(self.config.hidden_size),
            is_first_in_block=self.is_first_in_block,
            is_block_write_layer=self.is_block_write_layer,
            prev_valid_blocks=self.prev_valid_blocks,
            attn_res=self.self_attention_attn_res,
            mlp_attn_res=self.mlp_attn_res,
            attn_hc=self.attn_hc,
            ffn_hc=self.ffn_hc,
            input_norm_weight=self.input_layernorm.weight,
            pre_mlp_norm_weight=self.pre_mlp_layernorm.weight,
            attn_fn=_attn_fn,
            mlp_fn=_mlp_fn,
            handoff=self.attn_res_handoff,
            packed_seq_params=packed_seq_params,
        )
        return flat, context
