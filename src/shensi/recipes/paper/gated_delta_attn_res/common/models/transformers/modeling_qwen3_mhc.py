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

"""Qwen3 + mHC（流形约束超连接）的 HF 参考实现。"""

from __future__ import annotations

import math
from typing import Any, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from .configuration_qwen3_mhc import Qwen3MHCConfig

__all__ = [
    "Qwen3MHCConfig",
    "HyperConnection",
    "sinkhorn_knopp_project",
    "Qwen3MHCDecoderLayer",
    "Qwen3MHCModel",
    "Qwen3MHCForCausalLM",
]







def sinkhorn_knopp_project(
    logits: torch.Tensor, num_iterations: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    matrix = logits.softmax(dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(num_iterations) - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def _read_weights(logits: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    if mode == "simplex":
        return logits.softmax(dim=-1)
    if mode == "sigmoid":
        return logits.sigmoid() + eps
    if mode == "linear":
        return logits
    raise ValueError(f"unknown hc_read: {mode}")


def _write_weights(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "sigmoid2":
        return logits.sigmoid() * 2
    if mode == "linear":
        return logits
    raise ValueError(f"unknown hc_write: {mode}")


def _identity_index(layer_index: int, sublayer: str, num_streams: int) -> int:
    offset = 0 if sublayer == "attn" else 1
    return (2 * layer_index + offset) % num_streams







class HyperConnection(nn.Module):

    def __init__(self, config: Qwen3MHCConfig, layer_index: int, sublayer: str = "attn"):
        super().__init__()
        self.n = int(config.hc_num_streams)
        self.hidden_size = int(config.hidden_size)
        self.layer_index = int(layer_index)
        self.sublayer = sublayer

        self.init_mode = config.hc_init
        self.dynamic = bool(config.hc_dynamic)
        self.read_mode = config.hc_read
        self.write_mode = config.hc_write
        self.manifold = config.mhc_manifold
        self.gating_factor = float(config.mhc_init_gating_factor)
        self.compute_h_eps = float(config.mhc_compute_h_eps)
        self.sinkhorn_iterations = int(config.mhc_sinkhorn_iterations)


        self.sinkhorn_eps = (
            0.0
            if (self.init_mode == "identity" and self.manifold == "doubly_stochastic")
            else float(config.mhc_sinkhorn_eps)
        )

        num_coeffs = self.n * self.n + 2 * self.n
        self.num_coeffs = num_coeffs

        self.mapping_proj = (
            nn.Linear(self.n * self.hidden_size, num_coeffs, bias=False) if self.dynamic else None
        )

        self.alpha_pre = nn.Parameter(torch.full((1,), self.gating_factor))
        self.alpha_post = nn.Parameter(torch.full((1,), self.gating_factor))
        self.alpha_res = nn.Parameter(torch.full((1,), self.gating_factor))

        self.bias = nn.Parameter(torch.zeros(num_coeffs))


        self._force_identity = False

        self.reset_parameters()


    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.mapping_proj is not None:
            nn.init.xavier_uniform_(self.mapping_proj.weight)

        n = self.n
        if self.init_mode == "official":
            self.alpha_pre.fill_(self.gating_factor)
            self.alpha_post.fill_(self.gating_factor)
            self.alpha_res.fill_(self.gating_factor)
            self.bias.zero_()
            if self.manifold == "none":
                self.bias[2 * n :].view(n, n).copy_(torch.eye(n, dtype=self.bias.dtype))
            return

        if self.init_mode != "identity":
            raise ValueError(f"unknown hc_init: {self.init_mode}")


        self.alpha_pre.zero_()
        self.alpha_post.zero_()
        self.alpha_res.zero_()
        self.bias.zero_()


        pre = self.bias[:n]
        if self.read_mode == "linear":

            pre[_identity_index(self.layer_index, self.sublayer, n)] = 1.0
        elif self.read_mode == "simplex":
            pre.zero_()
        else:
            target = 1.0 / n - self.compute_h_eps
            pre.fill_(math.log(target / (1.0 - target)))


        post = self.bias[n : 2 * n]
        post.fill_(0.0 if self.write_mode == "sigmoid2" else 1.0)




        res = self.bias[2 * n :].view(n, n)
        if self.manifold == "none":
            res.copy_(torch.eye(n, dtype=self.bias.dtype))
        else:
            res.zero_()


    def compute_mappings(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, hidden = self.n, self.hidden_size
        shape = streams.shape[:-2]
        x = streams.reshape(*shape, n * hidden)
        dtype = x.dtype

        x = x.float() if dtype != torch.float32 else x

        if self.mapping_proj is not None:
            proj = F.linear(x, self.mapping_proj.weight.float())
            r = 1.0 / (x.norm(dim=-1, keepdim=True) / math.sqrt(n * hidden) + self.compute_h_eps)
            alpha = torch.cat(
                [
                    self.alpha_pre.expand(n),
                    self.alpha_post.expand(n),
                    self.alpha_res.expand(n * n),
                ]
            ).float()
            h = r * proj * alpha + self.bias.float()
        else:
            h = self.bias.float().expand(*shape, self.num_coeffs)

        h_pre = _read_weights(h[..., :n], self.read_mode, self.compute_h_eps)
        h_post = _write_weights(h[..., n : 2 * n], self.write_mode)
        h_res = h[..., 2 * n :].reshape(*shape, n, n)
        if self.manifold == "doubly_stochastic":
            h_res = sinkhorn_knopp_project(h_res, self.sinkhorn_iterations, self.sinkhorn_eps)
        elif self.manifold != "none":
            raise ValueError(f"unknown mhc_manifold: {self.manifold}")
        return h_pre.to(dtype), h_post.to(dtype), h_res.to(dtype)


    @staticmethod
    def _reference(x: torch.Tensor) -> torch.Tensor:
        return x[..., 0, :]

    def aggregate(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        x_ref = self._reference(streams)
        w_sum = h_pre.sum(dim=-1, keepdim=True)
        correction = ((streams - x_ref.unsqueeze(-2)) * h_pre.unsqueeze(-1)).sum(dim=-2)
        return x_ref * w_sum + correction

    def apply_h_res(self, h_res: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        x_ref = self._reference(streams)
        col_sum = h_res.sum(dim=-2)
        correction = torch.einsum("...ji,...jc->...ic", h_res, streams - x_ref.unsqueeze(-2))
        return x_ref.unsqueeze(-2) * col_sum.unsqueeze(-1) + correction

    @staticmethod
    def apply_h_post(x: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
        return h_post.unsqueeze(-1) * x.unsqueeze(-2)

    def forward(
        self, streams: torch.Tensor, stats: list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._force_identity:


            n = self.n
            shape = streams.shape[:-2]
            eye = torch.eye(n, dtype=streams.dtype, device=streams.device).expand(*shape, n, n)
            ones = torch.ones(*shape, n, dtype=streams.dtype, device=streams.device)
            return self._reference(streams), eye, ones, streams
        h_pre, h_post, h_res = self.compute_mappings(streams)
        if stats is not None:
            entry = self.mapping_stats(h_pre, h_post, h_res)
            entry.update({"layer": self.layer_index, "sublayer": self.sublayer})
            stats.append(entry)
        return self.aggregate(streams, h_pre), h_res, h_post, streams

    def fuse(
        self,
        h_res: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_h_res(h_res, residual) + self.apply_h_post(layer_output, h_post)


    def mapping_stats(self, h_pre: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> dict:
        with torch.no_grad():
            w = h_pre.float()
            row = h_res.float().abs().sum(dim=-1)
            col = h_res.float().abs().sum(dim=-2)
            return {
                "n_streams": self.n,
                "h_pre_sum": float(w.sum(dim=-1).mean()),
                "h_pre_min": float(w.min()),
                "h_post_mean": float(h_post.float().mean()),
                "h_res_row_sum_max": float(row.max()),
                "h_res_col_sum_max": float(col.max()),
                "h_res_min": float(h_res.float().min()),
                "sinkhorn_eps": float(self.sinkhorn_eps),
            }







class Qwen3MHCDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MHCConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_hyper_connections = config.attn_res_block_size is not None
        if self.use_hyper_connections:
            self.self_attention_hyper_connection = HyperConnection(config, layer_idx, "attn")
            self.mlp_hyper_connection = HyperConnection(config, layer_idx, "mlp")

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_res_stats: list | None = None,
        composite: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if not self.use_hyper_connections:
            return self._forward_plain(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                position_embeddings,
            ), composite

        batch_size, seq_len, _ = hidden_states.shape
        n, hidden = self.config.hc_num_streams, self.hidden_size
        streams = hidden_states.view(batch_size, seq_len, n, hidden)


        module = self.self_attention_hyper_connection
        read, h_res, h_post, residual = module(streams, stats=attn_res_stats)
        attn_out = self._attention(
            self.input_layernorm(read),
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )
        if composite is not None:
            composite = h_res.reshape(-1, n, n).transpose(-1, -2) @ composite
        streams = module.fuse(h_res, residual, h_post, attn_out)


        module = self.mlp_hyper_connection
        read, h_res, h_post, residual = module(streams, stats=attn_res_stats)
        mlp_out = self.mlp(self.post_attention_layernorm(read))
        if composite is not None:
            composite = h_res.reshape(-1, n, n).transpose(-1, -2) @ composite
        streams = module.fuse(h_res, residual, h_post, mlp_out)

        return streams.reshape(batch_size, seq_len, n * hidden), composite

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


class Qwen3MHCModel(Qwen3PreTrainedModel):

    config_class = Qwen3MHCConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3MHCConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3MHCDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.use_hyper_connections = config.attn_res_block_size is not None
        self.num_streams = int(config.hc_num_streams)
        self.stream_chunk = int(config.attn_res_block_size) if self.use_hyper_connections else 0
        self.output_contract = config.hc_output_contract
        if self.use_hyper_connections and self.output_contract == "learned":


            n, hidden = self.num_streams, config.hidden_size
            self.hc_head_fn = nn.Parameter(torch.randn(n, n * hidden))
            self.hc_head_base = nn.Parameter(torch.zeros(n))
            self.hc_head_scale = nn.Parameter(torch.ones(1))
            nn.init.xavier_uniform_(self.hc_head_fn)
        self.gradient_checkpointing = False
        self.post_init()


    def _init_weights(self, module):
        if isinstance(module, HyperConnection):



            module.reset_parameters()
            return




        super()._init_weights(module)


    def _input_expand(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden = hidden_states.shape
        n = self.num_streams
        return (
            hidden_states.unsqueeze(2)
            .expand(batch_size, seq_len, n, hidden)
            .reshape(batch_size, seq_len, n * hidden)
        )

    def _output_contract(self, streams: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = streams.shape
        n, hidden = self.num_streams, self.config.hidden_size
        x = streams.view(batch_size, seq_len, n, hidden)
        x_ref = x[..., 0, :]
        if self.output_contract == "mean":



            return x_ref + (x - x_ref.unsqueeze(-2)).mean(dim=-2)
        if self.output_contract == "sum":

            return x.sum(dim=-2)
        if self.output_contract == "learned":


            flat = streams.float()
            rsqrt = torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.config.hc_output_contract_eps
            )
            mixes = F.linear(flat, self.hc_head_fn.float()) * rsqrt
            pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float())
            pre = pre + self.config.hc_output_contract_eps
            correction = ((x - x_ref.unsqueeze(-2)) * pre.unsqueeze(-1)).sum(dim=-2)
            return (x_ref * pre.sum(dim=-1, keepdim=True) + correction).to(streams.dtype)
        raise ValueError(f"unknown hc_output_contract: {self.output_contract}")

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

        if (
            past_key_values is not None
            and past_key_values.get_seq_length() > 0
            and inputs_embeds.shape[1] == 1
        ):



            raise NotImplementedError(
                "Qwen3MHC/Qwen3HC carries an n-stream residual as depth state, which is not "
                "cached: incremental decoding would silently drop it.  Call the model with the "
                "full sequence (use_cache=False) instead."
            )

        hidden_states = inputs_embeds
        num_layers = len(self.layers)
        chunk = self.stream_chunk if self.use_hyper_connections else num_layers

        for start in range(0, num_layers, chunk):
            if self.use_hyper_connections:

                hidden_states = self._input_expand(hidden_states)
                composite = (
                    torch.eye(
                        self.num_streams, dtype=hidden_states.dtype, device=hidden_states.device
                    )
                    .expand(
                        hidden_states.shape[0] * hidden_states.shape[1],
                        self.num_streams,
                        self.num_streams,
                    )
                    .clone()
                    if attn_res_stats is not None
                    else None
                )
            else:
                composite = None
            for layer in self.layers[start : start + chunk]:
                hidden_states, composite = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    attn_res_stats=attn_res_stats,
                    composite=composite,
                )
            if self.use_hyper_connections:

                if attn_res_stats is not None and composite is not None:
                    row = composite.detach().abs().sum(dim=-1).amax(dim=-1)
                    col = composite.detach().abs().sum(dim=-2).amax(dim=-1)
                    attn_res_stats.append(
                        {
                            "layer": start,
                            "sublayer": "composite",
                            "amax_gain_fwd": float(row.mean()),
                            "amax_gain_fwd_max": float(row.max()),
                            "amax_gain_bwd": float(col.mean()),
                            "amax_gain_bwd_max": float(col.max()),
                        }
                    )
                hidden_states = self._output_contract(hidden_states)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3MHCForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3MHCConfig

    def __init__(self, config: Qwen3MHCConfig):
        super().__init__(config)
        self.model = Qwen3MHCModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, HyperConnection):
            module.reset_parameters()
            return




        super()._init_weights(module)

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
    (AutoConfig.register, ("qwen3_mhc", Qwen3MHCConfig)),
    (AutoModel.register, (Qwen3MHCConfig, Qwen3MHCModel)),
    (AutoModelForCausalLM.register, (Qwen3MHCConfig, Qwen3MHCForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass











for _cls, _auto in (
    (Qwen3MHCConfig, "AutoConfig"),
    (Qwen3MHCModel, "AutoModel"),
    (Qwen3MHCForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
