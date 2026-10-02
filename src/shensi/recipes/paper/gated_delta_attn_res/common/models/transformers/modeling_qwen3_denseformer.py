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

"""Qwen3 + DenseFormer（深度加权平均）的 HF 参考实现。"""

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
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from .configuration_qwen3_denseformer import Qwen3DenseFormerConfig

__all__ = [
    "Qwen3DenseFormerConfig",
    "DepthWeightedAverage",
    "Qwen3DenseFormerDecoderLayer",
    "Qwen3DenseFormerModel",
    "Qwen3DenseFormerForCausalLM",
]







class DepthWeightedAverage(nn.Module):

    def __init__(self, config: Qwen3DenseFormerConfig, num_sources: int):
        super().__init__()
        self.num_sources = int(num_sources)
        self.param_mode = getattr(config, "attn_res_dwa_param", "deviation")
        if self.param_mode == "official":
            self.alpha = nn.Parameter(torch.zeros(self.num_sources))
        else:




            self.alpha_delta = nn.Parameter(torch.zeros(self.num_sources))
        self.reset_parameters()

    def effective_alpha(self) -> torch.Tensor:
        if self.param_mode == "official":
            return self.alpha
        one_hot = torch.zeros(
            self.num_sources, dtype=self.alpha_delta.dtype, device=self.alpha_delta.device
        )
        one_hot[-1] = 1.0
        return one_hot + self.alpha_delta

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.param_mode == "official":
                self.alpha.zero_()
                self.alpha[-1] = 1.0
            else:
                self.alpha_delta.zero_()

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        alpha = self.effective_alpha().to(sources.dtype)
        return torch.einsum("tnd,n->td", sources.float(), alpha.float()).to(sources.dtype)


def record_dwa_stats(stats, layer_idx, alpha=None, n_sources=None, period=None, dilation=None):
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": "dwa"}
    with torch.no_grad():
        if alpha is not None:
            w = alpha.detach().float().abs()
            total = float(w.sum())
            entry["n_sources"] = int(w.numel())
            entry["sharpness"] = float(w.max())
            entry["alpha_sum"] = float(alpha.detach().float().sum())
            entry["alpha_last"] = float(alpha.detach().float()[-1])
            if total > 0:
                p = w / total
                entry["entropy"] = float(-(p * (p + 1e-8).log()).sum())

                entry["offdiag_mass"] = float(1.0 - p[-1])
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
        if period is not None:
            entry["period"] = int(period)
        if dilation is not None:
            entry["dilation"] = int(dilation)
    stats.append(entry)


class Qwen3DenseFormerDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3DenseFormerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = config.attn_res_block_size is not None
        self.dwa = None
        if self.use_attn_residuals:
            self.dwa_period = int(config.attn_res_block_size)
            self.dwa_dilation = max(1, int(getattr(config, "attn_res_dwa_dilation", 1) or 1))
            self.dwa_layer = self._is_dwa_layer()
            if self.dwa_layer:
                self.dwa = DepthWeightedAverage(config, len(self.source_indices()))


    def _is_dwa_layer(self) -> bool:
        if (self.layer_idx + 1) % self.dwa_period == 0:
            return True
        return (
            bool(self.config.attn_res_output_route)
            and self.layer_idx == self.config.num_hidden_layers - 1
        )

    def source_indices(self) -> list[int]:
        i = self.layer_idx + 1
        k = self.dwa_dilation
        return list(range(i % k, i + 1, k))


    def forward(
        self,
        hidden_states: torch.Tensor,
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
        return self._forward_dwa(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
            depth_states,
            attn_res_stats,
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

    @staticmethod
    def _append_state(depth_states, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1])
        if depth_states is None:
            return flat.unsqueeze(1)
        return torch.cat([depth_states, flat.unsqueeze(1)], dim=1)

    def _forward_dwa(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Cache | None,
        use_cache: bool | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        depth_states: torch.Tensor | None,
        attn_res_stats: list | None,
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape


        block_out = self._forward_plain(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )


        depth_states = self._append_state(depth_states, block_out)

        if self.dwa is None:
            return block_out, depth_states


        sources = depth_states[:, self.source_indices(), :]
        routed = self.dwa(sources).view(batch_size, seq_len, hidden_size)
        record_dwa_stats(
            attn_res_stats,
            self.layer_idx,
            alpha=self.dwa.effective_alpha(),
            period=self.dwa_period,
            dilation=self.dwa_dilation,
        )
        return routed, depth_states


class Qwen3DenseFormerModel(Qwen3PreTrainedModel):

    config_class = Qwen3DenseFormerConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3DenseFormerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3DenseFormerDecoderLayer(config, layer_idx)
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
            if isinstance(module, DepthWeightedAverage):
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

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3DenseFormerForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3DenseFormerConfig

    def __init__(self, config: Qwen3DenseFormerConfig):
        super().__init__(config)
        self.model = Qwen3DenseFormerModel(config)
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
    (AutoConfig.register, ("qwen3_denseformer", Qwen3DenseFormerConfig)),
    (AutoModel.register, (Qwen3DenseFormerConfig, Qwen3DenseFormerModel)),
    (AutoModelForCausalLM.register, (Qwen3DenseFormerConfig, Qwen3DenseFormerForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass











for _cls, _auto in (
    (Qwen3DenseFormerConfig, "AutoConfig"),
    (Qwen3DenseFormerModel, "AutoModel"),
    (Qwen3DenseFormerForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
