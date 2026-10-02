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

"""Qwen3 + DAR（delta 注意力残差）的 HF 参考实现。"""

from __future__ import annotations

from typing import Any

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
)
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from .configuration_qwen3_dar import Qwen3DARConfig

__all__ = ["Qwen3DARConfig", "Qwen3DARDecoderLayer", "Qwen3DARModel", "Qwen3DARForCausalLM"]







def _delta_attn_res_kernel(V, partial_block, query, norm):
    K = norm(V)
    logits = torch.einsum("d, n t d -> n t", query, K)
    weights = logits.softmax(dim=0)
    selected = torch.einsum("n t, n t d -> t d", weights, V)
    return partial_block + selected


def delta_attn_res(deltas, partial_block, proj, norm, null_source=None, return_weights=False):
    sources = list(deltas)
    if null_source is not None:
        sources = [null_source.to(partial_block.dtype).expand_as(partial_block)] + sources
    if not sources:
        if return_weights:
            return partial_block, partial_block.new_zeros((0, partial_block.shape[0]))
        return partial_block

    V = torch.stack(sources, dim=0)
    query = proj.weight.view(-1)
    out = _delta_attn_res_kernel(V, partial_block, query, norm)
    if return_weights:
        K = norm(V)
        logits = torch.einsum("d, n t d -> n t", query, K)
        return out, logits.softmax(dim=0)
    return out


def record_router_stats(stats, layer_idx, sublayer, probs=None, n_sources=None):
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": sublayer}
    with torch.no_grad():
        if probs is not None:
            w = probs.detach().float()
            entry["n_sources"] = int(w.shape[0])
            entry["sharpness"] = float(w.max(dim=0).values.mean())
            entry["entropy"] = float(-(w * (w + 1e-8).log()).sum(dim=0).mean())
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
    stats.append(entry)


class Qwen3DARDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3DARConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size

            self.self_attention_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

            self.use_null_source = config.attn_res_use_null_source
            if self.use_null_source:
                self.self_attention_null_source = nn.Parameter(torch.zeros(config.hidden_size))
                self.mlp_null_source = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        delta_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                position_embeddings,
                delta_residual,
                attn_res_stats,
            )
        return self._forward_plain(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
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

    def _route(self, prefix_sum, deltas, proj, norm, sublayer, stats, null_source=None):
        batch_size, seq_len, hidden_size = prefix_sum.shape
        flat = prefix_sum.view(-1, hidden_size)
        if not deltas and null_source is None:
            record_router_stats(stats, self.layer_idx, sublayer, n_sources=0)
            return flat.view(batch_size, seq_len, hidden_size)
        if stats is None:
            routed = delta_attn_res(deltas, flat, proj, norm, null_source=null_source)
            record_router_stats(stats, self.layer_idx, sublayer, n_sources=len(deltas))
        else:
            routed, probs = delta_attn_res(
                deltas, flat, proj, norm, null_source=null_source, return_weights=True
            )
            record_router_stats(stats, self.layer_idx, sublayer, probs=probs)
        return routed.view(batch_size, seq_len, hidden_size)

    def _source_list(self, delta_residual):
        if delta_residual is None or delta_residual.shape[1] == 0:
            return []
        entries = list(delta_residual.unbind(dim=1))
        if self.attn_res_block_size == 1:
            return entries
        return [entries[i + 1] - entries[i] for i in range(len(entries) - 1)]

    def _append_source(self, delta_residual, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1])
        return torch.cat([delta_residual, flat.unsqueeze(1)], dim=1)

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        delta_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states
        per_sublayer_sources = self.attn_res_block_size == 1


        deltas = self._source_list(delta_residual)
        if self.use_null_source:
            attn_null = self.self_attention_null_source
        else:
            attn_null = None
        if deltas or attn_null is not None:
            hidden_states = self._route(
                prefix_sum,
                deltas,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                "attn",
                attn_res_stats,
                null_source=attn_null,
            )
        else:
            record_router_stats(attn_res_stats, self.layer_idx, "attn", n_sources=0)

        if not per_sublayer_sources and self.layer_idx % self.attn_res_block_size == 0:

            delta_residual = self._append_source(delta_residual, prefix_sum)

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._attention(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )
        prefix_sum = prefix_sum + hidden_states
        if per_sublayer_sources:
            if delta_residual is None or delta_residual.shape[1] == 0:
                delta_residual = self._append_source(delta_residual, prefix_sum - hidden_states)
            delta_residual = self._append_source(delta_residual, hidden_states)


        deltas = self._source_list(delta_residual)
        mlp_null = self.mlp_null_source if self.use_null_source else None
        hidden_states = self._route(
            prefix_sum,
            deltas,
            self.mlp_res_proj,
            self.mlp_res_norm,
            "mlp",
            attn_res_stats,
            null_source=mlp_null,
        )
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        prefix_sum = prefix_sum + hidden_states
        if per_sublayer_sources:
            delta_residual = self._append_source(delta_residual, hidden_states)

        return prefix_sum, delta_residual


class Qwen3DARModel(Qwen3PreTrainedModel):

    config_class = Qwen3DARConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3DARConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3DARDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.use_attn_residuals = config.attn_res_block_size is not None
        self.output_attn_res = self.use_attn_residuals and config.attn_res_output_route
        if self.output_attn_res:
            self.output_attn_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):



        super()._init_weights(module)

    def _apply_output_attn_res(self, hidden_states, delta_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        deltas = list(delta_residual.unbind(dim=1)) if delta_residual.shape[1] else []
        if not deltas:
            return hidden_states
        return delta_attn_res(
            deltas,
            hidden_states.view(-1, hidden_size),
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(batch_size, seq_len, hidden_size)

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
        delta_residual = None
        if self.use_attn_residuals:
            delta_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2]
            )

        for decoder_layer in self.layers:
            if self.use_attn_residuals:
                hidden_states, delta_residual = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    delta_residual=delta_residual,
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

        if self.output_attn_res:
            hidden_states = self._apply_output_attn_res(hidden_states, delta_residual)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3DARForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3DARConfig

    def __init__(self, config: Qwen3DARConfig):
        super().__init__(config)
        self.model = Qwen3DARModel(config)
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
    (AutoConfig.register, ("qwen3_dar", Qwen3DARConfig)),
    (AutoModel.register, (Qwen3DARConfig, Qwen3DARModel)),
    (AutoModelForCausalLM.register, (Qwen3DARConfig, Qwen3DARForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass











for _cls, _auto in (
    (Qwen3DARConfig, "AutoConfig"),
    (Qwen3DARModel, "AutoModel"),
    (Qwen3DARForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
