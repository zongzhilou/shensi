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

"""Qwen3 + RealFormer（残差注意力）的 HF 参考实现。"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from .configuration_qwen3_realformer import Qwen3RealFormerConfig

__all__ = [
    "Qwen3RealFormerAttention",
    "Qwen3RealFormerDecoderLayer",
    "Qwen3RealFormerForCausalLM",
    "Qwen3RealFormerModel",
    "residual_attention",
]





class GateScale(nn.Module):

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "deviation":
            self.delta = nn.Parameter(torch.zeros(1))
        elif mode in ("zero", "one"):

            self.register_buffer(
                "_const", torch.full((1,), 0.0 if mode == "zero" else 1.0), persistent=False
            )
        else:
            raise ValueError(
                f"attn_res_realformer_gate={mode!r} is not one of 'deviation' / 'zero' / 'one'"
            )

    def value(self) -> torch.Tensor:
        if self.mode == "deviation":
            return self.delta
        return self._const

    def reset_parameters(self) -> None:
        if self.mode == "deviation":
            with torch.no_grad():
                self.delta.zero_()

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


def residual_attention(
    attention_scores: torch.Tensor,
    prev_attention: torch.Tensor | None,
    gate: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    use_running_mean: bool = False,
    num_prev_layers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_attention = attention_scores
    if prev_attention is not None:
        cur_attention = cur_attention + gate * prev_attention
    logits = cur_attention
    if use_running_mean:
        logits = logits / (num_prev_layers + 1.0)
    if attention_mask is not None:
        logits = logits + attention_mask
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(
        attention_scores.dtype
    )
    return probs, cur_attention


class Qwen3RealFormerAttention(Qwen3Attention):

    def __init__(self, config: Qwen3RealFormerConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)


        self.realformer_gate = (
            GateScale(getattr(config, "attn_res_realformer_gate", "deviation"))
            if layer_idx >= 1
            else None
        )
        self.realformer_mean = bool(getattr(config, "attn_res_realformer_mean", False))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_attention: torch.Tensor | None = None,
        num_prev_layers: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )



        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        gate = self.realformer_gate.value() if self.realformer_gate is not None else None
        probs, cur_attention = residual_attention(
            attention_scores,
            prev_attention,
            gate if gate is not None else attention_scores.new_ones(()),
            attention_mask,
            use_running_mean=self.realformer_mean,
            num_prev_layers=num_prev_layers,
        )
        dropout_p = 0.0 if not self.training else self.attention_dropout
        probs = nn.functional.dropout(probs, p=dropout_p, training=self.training)

        attn_output = torch.matmul(probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, cur_attention


class Qwen3RealFormerDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3RealFormerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3RealFormerAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_attention: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cur_attention = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            prev_attention=prev_attention,
            num_prev_layers=self.layer_idx,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, cur_attention


class Qwen3RealFormerModel(Qwen3PreTrainedModel):

    config_class = Qwen3RealFormerConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3RealFormerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3RealFormerDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def post_init(self):
        super().post_init()

        for module in self.modules():
            if isinstance(module, GateScale):
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



        prev_attention: torch.Tensor | None = None
        for decoder_layer in self.layers:
            hidden_states, prev_attention = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_attention=prev_attention,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3RealFormerForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3RealFormerConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3RealFormerConfig):
        super().__init__(config)
        self.model = Qwen3RealFormerModel(config)
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
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )



try:
    AutoConfig.register("qwen3_realformer", Qwen3RealFormerConfig)
except ValueError:
    pass
for _cls, _auto in (
    (Qwen3RealFormerModel, "AutoModel"),
    (Qwen3RealFormerForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):  # pragma: no cover
        pass
del _cls, _auto
