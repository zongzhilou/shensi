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

"""Qwen3 + MUDD（多路动态稠密）的 HF 参考实现。"""

from __future__ import annotations

import copy
import functools
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from .configuration_qwen3_mudd import Qwen3MUDDConfig

__all__ = [
    "Qwen3MUDDConfig",
    "RMSNormNoScale",
    "MultiwayDynamicDense",
    "Qwen3MUDDDecoderLayer",
    "Qwen3MUDDModel",
    "Qwen3MUDDForCausalLM",
]


class RMSNormNoScale(nn.Module):

    def __init__(self, dim: int = -1, eps: float = 1.0e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=self.dim, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype)







class MultiwayDynamicDense(nn.Module):

    def __init__(self, config: Qwen3MUDDConfig, num_states: int, last_layer: bool = False):
        super().__init__()
        self.num_ways = (
            1 if (last_layer and config.mudd_fix_last_layer) else int(config.mudd_num_ways)
        )
        self.num_states = int(num_states)
        self.last_layer = bool(last_layer)
        self.param_mode = config.mudd_param
        self.scale_dw = bool(config.mudd_scale_dw)
        self.dw_norm = config.mudd_dw_norm

        out_dim = self.num_ways * self.num_states
        hidden = out_dim * (int(config.mudd_last_layer_expand) if last_layer else 1)
        round_to = int(config.mudd_hidden_round or 0)
        if round_to > 0:
            hidden = (hidden // round_to + 1) * round_to
        self.hidden = hidden

        eps = config.rms_norm_eps


        self.norm = (
            Qwen3RMSNorm(config.hidden_size, eps=eps)
            if config.mudd_pre_norm
            else RMSNormNoScale(eps=eps)
        )
        self.w1 = nn.Linear(config.hidden_size, hidden, bias=False)
        self.act = ACT2FN[config.mudd_act]
        self.w2 = nn.Linear(hidden, out_dim, bias=False)

        self.zero_prior = bool(config.mudd_pre_norm and config.mudd_post_norm)
        identity = torch.zeros(self.num_ways, self.num_states)
        if not self.zero_prior:
            identity[:, -1] = 1.0
        if self.param_mode == "official":
            self.prior = nn.Parameter(identity.clone())
        elif self.param_mode == "random":
            self.prior = nn.Parameter(torch.randn(self.num_ways, self.num_states))
        else:




            self.prior_delta = nn.Parameter(torch.zeros(self.num_ways, self.num_states))
        self.reset_parameters()

    def effective_prior(self) -> torch.Tensor:
        if self.param_mode in ("official", "random"):
            return self.prior
        identity = torch.zeros(
            self.num_ways,
            self.num_states,
            dtype=self.prior_delta.dtype,
            device=self.prior_delta.device,
        )
        if not self.zero_prior:
            identity[:, -1] = 1.0
        return identity + self.prior_delta

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.w2.weight)
            if self.param_mode == "official":
                target = torch.zeros_like(self.prior)
                if not self.zero_prior:
                    target[:, -1] = 1.0
                self.prior.copy_(target)
            elif self.param_mode == "deviation":
                self.prior_delta.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = self.act(self.w1(self.norm(x)))
        dw = self.w2(act)
        if self.scale_dw:
            dw = dw / math.sqrt(self.hidden)
        dw = dw.view(*x.shape[:-1], self.num_ways, self.num_states) + self.effective_prior().to(
            dw.dtype
        )
        if self.dw_norm == "softmax":
            dw = torch.softmax(dw, dim=-1)
        return dw

    @staticmethod
    def aggregate(dw: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = torch.einsum("tcn,tnd->ctd", dw.float(), states.float()).to(states.dtype)
        return out.unbind(0)


def record_mudd_stats(stats, layer_idx, dw=None, n_sources=None, period=None):
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": "mudd"}
    with torch.no_grad():
        if dw is not None:
            w = dw.detach().float()
            flat = w.reshape(-1, w.shape[-1])
            last = float(flat[:, -1].mean())
            entry["n_sources"] = int(w.shape[-1])
            entry["alpha_last"] = last
            entry["dyn_mean"] = float(w.mean())
            p = flat.abs()
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            entry["sharpness"] = float(p.max(dim=-1).values.mean())
            entry["entropy"] = float(-(p * (p + 1e-8).log()).sum(dim=-1).mean())
            entry["offdiag_mass"] = float(1.0 - p[:, -1].mean())
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
        if period is not None:
            entry["period"] = int(period)
    stats.append(entry)


def _fused_qkv_sizes(attn: nn.Module, fused: nn.Module) -> tuple[int, int, int]:
    total = int(fused.weight.shape[0])
    sizes = getattr(fused, "output_sizes", None)
    if sizes is not None and len(sizes) == 3:
        tp = int(getattr(fused, "tp_size", 1) or 1)
        per_rank = [int(size) // tp for size in sizes]
        if sum(per_rank) == total and min(per_rank) > 0:
            return per_rank[0], per_rank[1], per_rank[2]
    config = getattr(attn, "config", None)
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    heads = int(getattr(config, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads)
    if head_dim and heads and kv_heads and total:
        expected = (heads + 2 * kv_heads) * head_dim
        if expected % total == 0:
            tp = expected // total
            if heads % tp == 0 and kv_heads % tp == 0:
                return (
                    heads // tp * head_dim,
                    kv_heads // tp * head_dim,
                    kv_heads // tp * head_dim,
                )
    raise ValueError(
        f"cannot split the fused {type(fused).__name__} of {type(attn).__name__} into q/k/v: "
        f"{total} rows do not factor as ({heads} q + 2 x {kv_heads} kv) heads of width {head_dim}"
    )


def _qkv_projections(attn: nn.Module) -> tuple[Any, Any, Any]:
    if hasattr(attn, "q_proj"):
        return attn.q_proj, attn.k_proj, attn.v_proj
    fused = getattr(attn, "qkv_proj", None)
    if fused is None:
        raise AttributeError(
            f"{type(attn).__name__} holds neither q_proj/k_proj/v_proj nor a fused qkv_proj, "
            "so the multiway attention cannot project its q/k/v streams"
        )
    weight, bias = fused.weight, getattr(fused, "bias", None)
    projections, start = [], 0
    for size in _fused_qkv_sizes(attn, fused):
        stop = start + size
        projections.append(
            functools.partial(
                F.linear,
                weight=weight[start:stop],
                bias=None if bias is None else bias[start:stop],
            )
        )
        start = stop
    return tuple(projections)  # type: ignore[return-value]


class Qwen3MUDDDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MUDDConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.num_ways = max(1, int(config.mudd_num_ways))

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = self._make_mlp(config, layer_idx)
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals and self.num_ways > 1:

            self.input_layernorm_q = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.mudd_sepln:
                self.input_layernorm_k = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.input_layernorm_v = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:

            self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dense_conn = None
        self.dense_post_norm = None
        self.state_norm = None
        if self.use_attn_residuals:
            self.period = int(config.attn_res_block_size)
            self.is_da_layer = self._is_da_layer()
            if self.is_da_layer:
                last_da = self.layer_idx == config.num_hidden_layers - 1
                self.dense_conn = MultiwayDynamicDense(
                    config, self.layer_idx + 2, last_layer=last_da
                )
                if config.mudd_post_norm:
                    self.dense_post_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    with torch.no_grad():
                        self.dense_post_norm.weight.fill_(0.001)
                if config.mudd_pre_norm:
                    self.state_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _make_mlp(config: Qwen3MUDDConfig, layer_idx: int):
        if not config.mudd_ffn_depth_scaling:
            return Qwen3MLP(config)
        depth = max(1, config.num_hidden_layers - 1)
        factor = layer_idx / depth + 0.5
        round_to = max(1, int(config.mudd_ffn_round or 1))
        dim = max(round_to, int(round(config.intermediate_size * factor / round_to) * round_to))
        layer_config = copy.deepcopy(config)
        layer_config.intermediate_size = dim
        return Qwen3MLP(layer_config)

    def _is_da_layer(self) -> bool:
        if (self.layer_idx + 1) % self.period == 0:
            return True
        return (
            bool(self.config.attn_res_output_route)
            and self.layer_idx == self.config.num_hidden_layers - 1
        )


    def _attention(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        position_embeddings,
    ):
        output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        return output[0] if isinstance(output, tuple) else output

    def _forward_plain(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        position_embeddings,
    ):
        residual = hidden_states
        hidden_states = self._attention(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states

    def _multiway_attention(
        self, xq, xk, xv, attention_mask, past_key_values, use_cache, position_embeddings
    ):
        attn = self.self_attn
        input_shape = xq.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q_proj, k_proj, v_proj = _qkv_projections(attn)
        query_states = attn.q_norm(q_proj(xq).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(k_proj(xk).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(xv).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output)

    def _stream_norms(self):
        if self.num_ways == 1:
            return (self.input_layernorm,) * 3
        if self.config.mudd_sepln:
            return self.input_layernorm_q, self.input_layernorm_k, self.input_layernorm_v
        return (self.input_layernorm_q,) * 3


    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        depth_states: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if not self.use_attn_residuals:

            return self._forward_plain(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                position_embeddings,
            )

        if isinstance(hidden_states, (tuple, list)):
            xq, xk, xv, xr = hidden_states
        else:

            xq = xk = xv = xr = hidden_states


        ln_q, ln_k, ln_v = self._stream_norms()
        attn_out = self._multiway_attention(
            ln_q(xq),
            ln_k(xk),
            ln_v(xv),
            attention_mask,
            past_key_values,
            use_cache,
            position_embeddings,
        )
        block_out = attn_out + xr
        block_out = block_out + self.mlp(self.post_attention_layernorm(block_out))


        state = self.state_norm(block_out) if self.state_norm is not None else block_out
        flat = state.reshape(-1, state.shape[-1])
        depth_states = (
            flat.unsqueeze(1)
            if depth_states is None
            else torch.cat([depth_states, flat.unsqueeze(1)], dim=1)
        )

        if self.dense_conn is None:
            return block_out, depth_states


        dw = self.dense_conn(block_out.reshape(-1, self.hidden_size))
        streams = MultiwayDynamicDense.aggregate(dw, depth_states)
        record_mudd_stats(attn_res_stats, self.layer_idx, dw=dw, period=self.period)
        if self.dense_post_norm is not None:
            streams = tuple(
                (block_out.reshape(-1, self.hidden_size) + self.dense_post_norm(s)).view(
                    block_out.shape
                )
                for s in streams
            )
        if self.num_ways == 1:
            return streams[0].view(block_out.shape), depth_states
        batch_size, seq_len = block_out.shape[0], block_out.shape[1]
        return tuple(s.view(batch_size, seq_len, self.hidden_size) for s in streams), depth_states


class Qwen3MUDDModel(Qwen3PreTrainedModel):

    config_class = Qwen3MUDDConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3MUDDConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3MUDDDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):




        super()._init_weights(module)

    def post_init(self):
        super().post_init()


        for module in self.modules():
            if isinstance(module, MultiwayDynamicDense):
                module.reset_parameters()

    @merge_with_config_defaults
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        depth_states = None
        if self.config.attn_res_block_size is not None:

            depth_states = hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(1)

        for decoder_layer in self.layers:
            if self.config.attn_res_block_size is not None:
                hidden_states, depth_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    depth_states=depth_states,
                    attn_res_stats=attn_res_stats,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

        if isinstance(hidden_states, (tuple, list)):

            hidden_states = hidden_states[-1]
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3MUDDForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3MUDDConfig

    def __init__(self, config: Qwen3MUDDConfig):
        super().__init__(config)
        self.model = Qwen3MUDDModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_attn_res_stats: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        stats = [] if return_attn_res_stats else None
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            attn_res_stats=stats,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        out = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values
        )
        if stats is not None:
            out.attn_res_stats = stats
        return out


for _register, _args in (
    (AutoConfig.register, ("qwen3_mudd", Qwen3MUDDConfig)),
    (AutoModel.register, (Qwen3MUDDConfig, Qwen3MUDDModel)),
    (AutoModelForCausalLM.register, (Qwen3MUDDConfig, Qwen3MUDDForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass











for _cls, _auto in (
    (Qwen3MUDDConfig, "AutoConfig"),
    (Qwen3MUDDModel, "AutoModel"),
    (Qwen3MUDDForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
