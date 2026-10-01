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
"""Qwen3 + DenseFormer (Depth-Weighted Averaging).

Baseline requested by the reviewers: DenseFormer, Pagliardini, Mohtashami,
Fleuret & Jaggi, "DenseFormer: Enhancing Information Flow in Transformers via
Depth Weighted Averaging" (arXiv:2402.02622, NeurIPS 2024).

**This file is aligned with the official implementation**, which was downloaded
and read while writing it (``epfml/DenseFormer``):

    * ``denseformer/denseformer.py``        -> ``DWAModules`` (the released package, the
      canonical "3 steps to a DenseFormer" API used by the paper's experiments);
    * ``experiments/models/denseformer.py`` -> the GPT-2 style experiment model
      (``self.weights`` = one ``nn.Linear(..., 1, bias=False)`` per DWA event).

The formulations below are copied from those files / the paper (Eq. numbers are
the paper's)::

    X_0 := Embedding(X),          Y_0 := X_0
    X_i := B_i(Y_{i-1})                                  # i = 1..d, an ordinary Qwen3 block
    Y_i := DWA_i({X_0, ..., X_i}) = sum_{j=0}^{i} alpha_{i,j} * X_j     # paper Sec. 3
    DenseFormer(X) := Y_d

  * the ``alpha_{i,j}`` are the only added parameters (one scalar per (i, j) pair,
    no weight sharing between DWA modules);
  * **initialisation** (paper "Initializing the DWA modules", official
    ``DWAModules._init_weights``): ``alpha.weight.data.zero_()`` followed by
    ``alpha.weight.data[0, -1] = 1.``, i.e. the DWA is the identity and
    DenseFormer reduces *exactly* to the standard Transformer at step 0.  Here the
    last entry is the current block's own output ``X_i``, which is the last source
    in the list, so ``Y_i = X_i`` and the residual chain is untouched;
  * **dilation** (paper Sec. 3.2, ``DWAModules(dilation=k)``): DWA_i only averages
    ``{X_j | j <= i, j = i (mod k)}``; official implementation writes state ``X_j``
    into accumulator group ``j % k`` at slot ``j // k``, which is exactly this
    subset in increasing-``j`` order (the last entry is again ``X_i``);
  * **periodicity** (paper Sec. 3.3, ``DWAModules(period=p)``): a DWA is applied
    only when ``(i + 1) % p == 0``; other blocks just update the stream to their own
    output.  ``attn_res_block_size`` maps onto this knob (``1`` = full DenseFormer).

Parameter count (paper Sec. 3.1): a DWA at depth ``i`` has ``i + 1`` weights, so a
depth-``d`` DenseFormer adds ``sum_{i=1}^{d} (i + 1) = d(d + 3) / 2`` scalars.  The
``d(d+1)/2`` figure quoted in the plan is the same sum without the ``alpha_{i,i}``
self-weight of every module (and without the extra module of the last layer):
``sum_{i=1}^{d} i``.  For ``d = 28``: 434 (official / this file) vs 406.

Differences from the official code, all of them deliberate and labelled:

  1. The backbone is Qwen3 (the project's shared trunk): the DWA modules are the
     only architectural change, so DenseFormer-vs-GDAR is a clean comparison.
  2. ``attn_res_dwa_param="deviation"`` (default) keeps the *identical* forward
     value of the official initialisation but writes the weights as
     ``one_hot(last) + delta`` with ``delta`` zero-initialised.  The paper's
     "alpha_{i,i}=1, rest 0" is then a construction-level guarantee rather than a
     one-off initialisation that any later re-init could destroy.  Setting
     ``attn_res_dwa_param="official"`` reproduces the released code literally.
  3. Incremental decoding (``use_cache``) does not carry the depth states across
     forward calls -- same limitation as the AR / DAR / GDAR files in this
     directory.
"""

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


# ---------------------------------------------------------------------------
# The DWA module (official ``DWAModules``, one alpha vector per event)
# ---------------------------------------------------------------------------


class DepthWeightedAverage(nn.Module):
    """``Y = sum_j alpha_j * X_j`` over states ordered by increasing depth.

    ``sources`` has shape ``(num_tokens, num_sources, hidden)`` with the *last*
    entry being the current block's own output, which is where the identity
    initialisation puts its ``1``.  The official code stores one
    ``nn.Linear(n, 1, bias=False)`` per event and applies it with
    ``torch.tensordot(weight.view(-1), slice, dims=1)``; a 1-D ``nn.Parameter``
    plus ``einsum`` is the same operator (there is no bias and no activation).

    ``attn_res_dwa_param="deviation"`` keeps ``alpha = one_hot(last) + delta``
    with ``delta == 0`` at init so that (i) the forward value is *bit-exactly* the
    standard residual stream, independently of how the module got initialised,
    and (ii) every ``delta_j`` still has a non-vanishing gradient at that point
    (``dL/d delta_j = <dL/dY, X_j> != 0``), which is what makes the identity
    *escapable*.  A multiplicative zero-init gate -- ``Y = X + s * (sum a X - X)``
    as in the read side of GDAR -- would instead freeze the connection forever:
    at the identity ``sum a X - X == 0``, so ``dL/ds = 0`` and, because the whole
    deviation is multiplied by ``s = 0``, ``dL/da = 0`` as well.
    """

    def __init__(self, config: Qwen3DenseFormerConfig, num_sources: int):
        super().__init__()
        self.num_sources = int(num_sources)
        self.param_mode = getattr(config, "attn_res_dwa_param", "deviation")
        if self.param_mode == "official":
            self.alpha = nn.Parameter(torch.zeros(self.num_sources))
        else:
            # The identity target is *derived* in ``effective_alpha`` rather than kept in a
            # non-persistent buffer: such a buffer is not checkpointed, so under
            # ``from_pretrained`` (meta-device init) it stays uninitialised and would corrupt
            # the model on reload.  Same reasoning as ``decay_tau`` in modeling_qwen3_gdar.py.
            self.alpha_delta = nn.Parameter(torch.zeros(self.num_sources))
        self.reset_parameters()

    def effective_alpha(self) -> torch.Tensor:
        """The ``alpha`` vector actually used by the forward pass."""
        if self.param_mode == "official":
            return self.alpha
        one_hot = torch.zeros(
            self.num_sources, dtype=self.alpha_delta.dtype, device=self.alpha_delta.device
        )
        one_hot[-1] = 1.0
        return one_hot + self.alpha_delta

    def reset_parameters(self) -> None:
        """Identity initialisation, see the class docstring."""
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
    """Diagnostics for the depth-flow analysis (``return_attn_res_stats=True``)."""
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
                # mass on *earlier* layers (the point of a dense connection)
                entry["offdiag_mass"] = float(1.0 - p[-1])
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
        if period is not None:
            entry["period"] = int(period)
        if dilation is not None:
            entry["dilation"] = int(dilation)
    stats.append(entry)


class Qwen3DenseFormerDecoderLayer(nn.Module):
    """Qwen3 decoder layer + DWA after the block (official ``Block`` + ``DWAModules``)."""

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

    # -- bookkeeping --------------------------------------------------------
    def _is_dwa_layer(self) -> bool:
        """A DWA event happens at layers ``period, 2*period, ...`` (paper Sec. 3.3).

        ``attn_res_output_route`` additionally forces one at the very last layer so
        that the model output is a DWA output (``DenseFormer(X) := Y_d``) even when
        the depth is not a multiple of the period.
        """
        if (self.layer_idx + 1) % self.dwa_period == 0:
            return True
        return (
            bool(self.config.attn_res_output_route)
            and self.layer_idx == self.config.num_hidden_layers - 1
        )

    def source_indices(self) -> list[int]:
        """Indices of the states this DWA averages, in increasing depth order.

        States are ``S_0 = embedding``, ``S_j = output of block j`` (1-based block
        index), so the DWA after block ``i = layer_idx + 1`` reads ``S_0..S_i`` and
        dilation ``k`` keeps ``{j <= i : j = i (mod k)}`` -- the official
        accumulator-group semantics -- whose last element is ``S_i`` again.
        """
        i = self.layer_idx + 1
        k = self.dwa_dilation
        return list(range(i % k, i + 1, k))

    # -- forward ------------------------------------------------------------
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
        """Append a block output as the newest depth state (``InPlaceSetSlice`` in the official code)."""
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
        """``X_i = B_i(Y_{i-1})`` then, at a DWA event, ``Y_i = sum_j alpha_j * X_j``."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # ---- the block itself: exactly the stock Qwen3 residual block ----
        block_out = self._forward_plain(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )

        # ---- append the block output as the newest depth state ----
        depth_states = self._append_state(depth_states, block_out)

        if self.dwa is None:
            return block_out, depth_states

        # ---- DWA: Y_i = sum_j alpha_j X_j (last source = the block we just ran) ----
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
    """Qwen3 backbone with Depth-Weighted Averaging between blocks."""

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
        # Delegate to the transformers default: it also rebuilds the non-persistent RoPE
        # buffers, which is what makes from_pretrained work (the checkpoint is built on the
        # meta device, so any buffer this method skips stays uninitialised).  The connection
        # anchors are re-established in ``post_init`` after this generic pass.
        super()._init_weights(module)

    def post_init(self):
        super().post_init()
        # HF's generic _init_weights may be re-applied (and would zero nothing here,
        # but a later base-class change could); the DWA identity is a construction
        # guarantee, so re-establish it explicitly, exactly like GDAR does.
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
            # X_0 = the embedding seeds the state list (``DWAModules.init_accumulators``
            # writes it at index 0 of accumulator 0); every block then appends its own
            # output, so the list is ``{X_0, ..., X_i}`` when block ``i`` looks at it.
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
    """Qwen3 + DenseFormer, causal LM head."""

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
        pass  # already registered (module imported more than once)

# ``register_for_auto_class`` is what makes a *fresh* process able to resolve
# ``model_type = "qwen3_denseformer"`` straight from a checkpoint directory: it sets
# ``_auto_class``, which makes ``save_pretrained`` write ``auto_map`` into
# ``config.json`` and copy these modules next to the weights, and it is the flag
# the ``trust_remote_code=True`` path checks.  Together with the
# ``AutoConfig.register`` / ``AutoModelForCausalLM.register`` calls above it
# covers both routes -- imported package and checkpoint-local code -- because
# verl's MegatronWorker does
# ``AutoConfig.from_pretrained(local_path, trust_remote_code=...)`` on a
# checkpoint whose directory this package is not on ``sys.path`` for.
for _cls, _auto in (
    (Qwen3DenseFormerConfig, "AutoConfig"),
    (Qwen3DenseFormerModel, "AutoModel"),
    (Qwen3DenseFormerForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
