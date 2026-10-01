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
"""Qwen3 + Attention Residuals (AR).

This file mirrors ``moonshotai/Kimi-K3:modeling_kimi_linear.py``: its own norm
implementation, a module-level depth-routing function, a decoder layer exposing
``_forward_attn_residual`` that returns ``(prefix_sum, block_residual)``, and a
backbone that carries that tensor state plus one output routing pass before the
final norm.

The depth connection is the one from that file, copied verbatim:
``_apply_attn_res`` routes over ``[completed blocks..., current accumulator]``
with scores ``<rmsnorm(v), norm.weight * proj.weight>``; the routed vector
*replaces* the sublayer input while the accumulator keeps growing additively.

    prefix_sum     : the in-flight accumulator; the layer's hidden state.
    block_residual : ``(num_tokens, num_blocks, hidden_size)``, one row per closed
                     block; a block closes whenever
                     ``layer_idx % attn_res_block_size == 0``.
    output routing : one final ``_apply_attn_res`` pass before the last norm.

``attn_res_block_size=1`` -> one source per layer (the "full" variant);
``N > 1`` -> blocks of N layers; ``None`` -> stock Qwen3 (no depth routing).

Attention, MLP, RoPE and the LM head are the stock Qwen3 modules, so an
AR-vs-baseline comparison isolates the depth connection.

Paper: Attention Residuals, Kimi Team, arXiv:2603.15031.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from transformers.utils.generic import merge_with_config_defaults

logger = logging.get_logger(__name__)

from .configuration_qwen3_ar import Qwen3ARConfig

__all__ = ["Qwen3ARConfig", "Qwen3ARDecoderLayer", "Qwen3ARModel", "Qwen3ARForCausalLM"]


class RMSNorm(nn.Module):
    """RMSNorm with a learnable weight (the Kimi norm, used by the AR routing)."""

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def _apply_attn_res(prefix_sum, block_residual, proj, norm, return_probs=False):
    """Softmax over completed blocks + the current accumulator.

    Verbatim from ``modeling_kimi_linear.py``; ``return_probs`` is an additive
    diagnostic hook that does not change the computation.

    Args:
        prefix_sum: ``(num_tokens, hidden_size)`` -- the in-flight accumulator.
        block_residual: ``(num_tokens, num_blocks, hidden_size)``.
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1)
    hidden_states = torch.matmul(probs.unsqueeze(1), v_float).squeeze(1)
    if return_probs:
        return hidden_states.to(v.dtype), probs
    return hidden_states.to(v.dtype)


def record_router_stats(stats, layer_idx, sublayer, probs=None, n_sources=None):
    """Scalar routing diagnostics for the routing-collapse analysis."""
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": sublayer}
    with torch.no_grad():
        if probs is not None:
            w = probs.detach().float()
            entry["n_sources"] = int(w.shape[-1])
            entry["sharpness"] = float(w.max(dim=-1).values.mean())
            entry["entropy"] = float(-(w * (w + 1e-8).log()).sum(dim=-1).mean())
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
    stats.append(entry)


class Qwen3ARDecoderLayer(nn.Module):
    """Qwen3 decoder layer with Attention Residuals."""

    def __init__(self, config: Qwen3ARConfig, layer_idx: int):
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
            self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_residual: torch.Tensor | None = None,
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
                block_residual,
                attn_res_stats,
            )
        return self._forward_plain(
            hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings
        )

    def _attention(self, hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings):
        output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        return output[0] if isinstance(output, tuple) else output

    def _forward_plain(self, hidden_states, attention_mask, position_ids, past_key_values, use_cache,
                       position_embeddings):
        residual = hidden_states
        hidden_states = self._attention(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_values, use_cache,
            position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states

    def _route(self, prefix_sum, block_residual, proj, norm, sublayer, stats):
        batch_size, seq_len, hidden_size = prefix_sum.shape
        flat = prefix_sum.view(-1, hidden_size)
        if stats is None:
            routed = _apply_attn_res(flat, block_residual, proj, norm)
        else:
            routed, probs = _apply_attn_res(flat, block_residual, proj, norm, return_probs=True)
            record_router_stats(stats, self.layer_idx, sublayer, probs=probs)
        return routed.view(batch_size, seq_len, hidden_size)

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        # ---- attention sublayer ----
        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = self._route(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                "attn",
                attn_res_stats,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat([block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._attention(
            hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings
        )
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        # ---- MLP sublayer ----
        hidden_states = self._route(
            prefix_sum, block_residual, self.mlp_res_proj, self.mlp_res_norm, "mlp", attn_res_stats
        )
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        return prefix_sum, block_residual


class Qwen3ARModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Attention Residuals."""

    config_class = Qwen3ARConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3ARConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3ARDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.use_attn_residuals = config.attn_res_block_size is not None
        self.output_attn_res = self.use_attn_residuals and config.attn_res_output_route
        if self.output_attn_res:
            self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

        if config._attn_implementation not in ("sdpa", "eager", "flash_attention_2", "flex_attention"):
            logger.warning_once(f"Unknown attention implementation {config._attn_implementation}; using the default.")
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        # Delegate to the transformers default: it also rebuilds the non-persistent RoPE
        # buffers, which is what makes from_pretrained work (the checkpoint is built on the
        # meta device, so any buffer this method skips stays uninitialised).
        super()._init_weights(module)

    def _apply_output_attn_res(self, hidden_states, block_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        return _apply_attn_res(
            hidden_states.view(-1, hidden_size),
            block_residual,
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
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
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
        block_residual = None
        if self.use_attn_residuals:
            block_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2]
            )

        for decoder_layer in self.layers:
            if self.use_attn_residuals:
                hidden_states, block_residual = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    block_residual=block_residual,
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
            hidden_states = self._apply_output_attn_res(hidden_states, block_residual)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3ARForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 + Attention Residuals, causal LM head."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3ARConfig

    def __init__(self, config: Qwen3ARConfig):
        super().__init__(config)
        self.model = Qwen3ARModel(config)
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
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        out = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)
        if stats is not None:
            out.attn_res_stats = stats
        return out


for _register, _args in (
    (AutoConfig.register, ("qwen3_ar", Qwen3ARConfig)),
    (AutoModel.register, (Qwen3ARConfig, Qwen3ARModel)),
    (AutoModelForCausalLM.register, (Qwen3ARConfig, Qwen3ARForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass  # already registered (module imported more than once)

# ``register_for_auto_class`` is what makes a *fresh* process able to resolve
# ``model_type = "qwen3_ar"`` straight from a checkpoint directory: it sets
# ``_auto_class``, which makes ``save_pretrained`` write ``auto_map`` into
# ``config.json`` and copy these modules next to the weights, and it is the flag
# the ``trust_remote_code=True`` path checks.  Together with the
# ``AutoConfig.register`` / ``AutoModelForCausalLM.register`` calls above it
# covers both routes -- imported package and checkpoint-local code -- because
# verl's MegatronWorker does
# ``AutoConfig.from_pretrained(local_path, trust_remote_code=...)`` on a
# checkpoint whose directory this package is not on ``sys.path`` for.
for _cls, _auto in (
    (Qwen3ARConfig, "AutoConfig"),
    (Qwen3ARModel, "AutoModel"),
    (Qwen3ARForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
