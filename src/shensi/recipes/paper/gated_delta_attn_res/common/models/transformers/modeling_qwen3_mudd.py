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
"""Qwen3 + MUDDFormer (Multiway Dynamic Dense connections).

Baseline requested by the reviewers: MUDDFormer, Xiao, Meng, Li & Yuan,
"MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic
Dense Connections" (arXiv:2502.12170, ICML 2025).

**This file is aligned with the official implementation**, which was downloaded
and read while writing it (``Caiyun-AI/MUDDFormer``):

    * ``jax/MaxText/layers/mudd.py``        -> ``Mlp`` + ``Compose``: the reference
      the released checkpoints were trained with (static prior as the ``dense_proj2``
      *bias*, its kernel initialised to 0, bias one-hot on the newest state);
    * ``pytorch/muddformer/modeling_muddformer.py`` -> ``MultiwayDynamicDenseBlock`` +
      the ``dense_bs`` parameter list (this release initialises the static prior
      with ``torch.randn``, i.e. *not* at the identity);
    * ``jax/MaxText/exp.py:MUDDLlama2Medium`` -> the canonical flag values.

Local copies of all of the above (fetched 2026-09-28) are kept next to the paper
for cross-checking: ``tmp/official_refs/muddformer/`` (``modeling_muddformer.py``,
``modeling_muddpythia.py``, ``layers_mudd.py`` -- the JAX reference -- and
``exp_mudd_configs.py``, i.e. ``jax/MaxText/exp.py``) plus
``tmp/official_refs/mudd_paper.txt`` (arXiv:2502.12170v2 HTML converted to text;
every quotation below is verbatim from there).

The formulation below is the paper's, quoted with its equation numbers::

    # dynamic connection weights (Sec. 2.2, Eq. 5/6) -- one function, per layer
    A_i = A_i(X_i) = GELU(RMSNorm(X_i) W1) W2 + a_i          # (T, i+1)
    Xbar_i = sum_j A_{i,j} * X_j                              # Eq. 5, broadcasting

    # multiway (Sec. 2.3, Eq. 7/8): the block input is decoupled into
    # (query, key, value, residual) streams and each gets its own DA module
    X_A'  = MHA(LN(X^Q), LN(X^K), LN(X^V)) + X^R
    B'(.) = FFN(LN(X_A')) + X_A'
    Xbar_i^Q, .., Xbar_i^R = DA_i^Q(X_:i), .., DA_i^R(X_:i),   MUDDFormer(X) = Xbar_L^R

    # static prior a_i (the "learnable prior for dense connectivity")
    a_i = one_hot(last)      at initialisation (official; see below)
    W2  = 0                  at initialisation (official)

Points where the redo plan's assumptions do **not** match the official code, all
verified against the sources above rather than assumed:

  1. **No softmax, no learned query/key dot product.**  The paper's Appendix A
     states: "Softmax is removed.  Instead, GELU activation is applied to query
     ... we empirically found that adding more sophisticated ingredients in DA
     (e.g. input dependent keys, softmax) does not bring improvement and slow down
     training".  ``W2^T`` plays the role of *static, input-independent* keys and
     ``a_i`` is a learnable positional bias.  ``mudd_dw_norm="softmax"`` is
     available to reproduce the softmax-routing variant the plan described, but it
     is off by default because (a) it is not MUDDFormer and (b) it destroys the
     identity initialisation (softmax of a one-hot vector is not one-hot).
  2. **A relay mechanism does not exist.**  The word does not appear in the paper
     (v1 or v2) nor anywhere in the official repository; the O(L^2) cost is instead
     bounded analytically (Sec. 2.6 / Appendix C: ``R_dparams ~ eta/6``,
     ``R_dFLOPs ~ eta/(3 + rho/4)``) and the states are cached for inference
     (``LayerCache``, an O(L) buffer).  Nothing here implements or needs a relay.
  3. **Identity initialisation is already the official one**, so no extra
     zero-init gate is needed -- and a multiplicative one (GDAR's ``read_scale``
     style, ``Y = X + s * (DA - X)``) would be *harmful*: at the identity point
     ``DA - X == 0`` and ``s == 0``, so ``dL/ds = <dL/dY, DA - X> = 0`` and every
     DA parameter gets ``dL/dtheta = s * (...) = 0``.  The connection would then be
     frozen at the identity forever, silently turning the baseline into plain
     Qwen3.  What is implemented instead (``mudd_param="deviation"``) is the same
     construction *without* the dead gate: the static prior is
     ``one_hot(last) + delta`` with ``delta = 0``, so at init ``DA_i = X_i``
     (bit-exact, `torch.equal`) while ``dL/d delta_j = <dL/dY, X_j> != 0`` keeps the
     dense path trainable.  ``test_baselines_mudd_df.py`` asserts both properties.
     The official initialisation (``mudd_param="official"``) is one and the same
     point; ``mudd_param="random"`` reproduces the released PyTorch init instead.

Deviations from the official code that are deliberate, all documented in
``configuration_qwen3_mudd.py``: the backbone is the project's shared Qwen3 trunk
(only the connection module differs), ``mudd_pre_norm`` / ``mudd_post_norm`` /
``mudd_ffn_depth_scaling`` default to *off* (they are the official JAX flags but
they change the trunk; use ``Qwen3MUDDConfig.official_preset()`` to switch them
on), and ``attn_res_block_size=N>1`` (periodic DA modules) is an extension.  The
official leaves an *unused* extra ``attention_norm`` on every layer > 0 when
``sepln=True``; that dead parameter is not replicated here.

One difference in *how the module is executed*, not in what it computes: an inference
engine is free to **fuse the trunk's three projections into one linear** (vLLM does:
``q_proj + k_proj + v_proj -> qkv_proj``, and it deletes the three attributes).
``_multiway_attention`` reads those projections directly, so it accepts the fused
layout as well and slices it back into q/k/v -- see ``_qkv_projections``.  Fusing is
arithmetically neutral (a fused linear is the concatenation of the three along its
output axis), so both layouts produce the same numbers;
``rollout/mudd_fused_qkv_repro.py`` checks that bit-exactly, fused against unfused.
"""

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
    """``RMSnormNoscale`` from the official code: RMS normalisation, no weight."""

    def __init__(self, dim: int = -1, eps: float = 1.0e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=self.dim, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype)


# ---------------------------------------------------------------------------
# The DA module (official ``MultiwayDynamicDenseBlock`` + static prior ``a_i``)
# ---------------------------------------------------------------------------


class MultiwayDynamicDense(nn.Module):
    """Generates the dense-connection weights ``A_i`` of one layer (Eq. 6).

    ``forward(x)`` returns ``(num_tokens, C, num_states)``: for every token, the
    weight of every source state ``X_0 .. X_i`` for each of the ``C`` decoupled
    streams.  The operator is literally the official one::

        dw = W2(GELU(norm(x) @ W1)) + a  # (T, C * (i+2))
        dw = dw.view(T, C, i + 2)  # 'B T (C L) -> C B T L' in the release

    ``a`` is the static prior (the official ``dense_proj2.bias`` / ``dense_bs``),
    with three initialisation modes -- see ``mudd_param`` in the configuration:

      * ``"deviation"`` (default): ``a = one_hot(last) + a_delta``, ``a_delta = 0``.
        Exactly the official forward value at init, with a live gradient.
      * ``"official"``: ``a`` is a raw parameter initialised to that same one-hot
        vector (the JAX release: zero kernel + one-hot bias).
      * ``"random"``: ``torch.randn`` as in the released PyTorch port.

    ``mudd_post_norm=True`` follows the official ``dense2_bias_init_value`` and
    initialises the prior to 0 instead: the DA output is then 0 and identity comes
    from the post-norm residual ``X + Norm(DA)`` (``Norm(0) = 0``).
    """

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
        if round_to > 0:  # official dynamic_dense_hidden_round: (K // 64 + 1) * 64
            hidden = (hidden // round_to + 1) * round_to
        self.hidden = hidden

        eps = config.rms_norm_eps
        # official: the weight-generating MLP always normalises its own input; the
        # norm is weighted only on the mudd_prenorm path (``pre_dense_proj1_norm``)
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
            identity[:, -1] = 1.0  # the newest state is X_i, the block we just ran
        if self.param_mode == "official":
            self.prior = nn.Parameter(identity.clone())
        elif self.param_mode == "random":
            self.prior = nn.Parameter(torch.randn(self.num_ways, self.num_states))
        else:
            # NOTE: the identity target is *derived* in ``effective_prior`` rather than kept
            # in a non-persistent buffer -- such a buffer is not checkpointed, so under
            # ``from_pretrained`` (meta-device init) it stays uninitialised and would corrupt
            # the model on reload.  Same reasoning as ``decay_tau`` in modeling_qwen3_gdar.py.
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
        """Official initialisation: ``W2 = 0`` and ``a`` = the identity target."""
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
        """``(T, D)`` -> ``(T, C, num_states)`` generated connection weights."""
        act = self.act(self.w1(self.norm(x)))
        dw = self.w2(act)
        if self.scale_dw:  # official dynamic_dense_scale_dw
            dw = dw / math.sqrt(self.hidden)
        dw = dw.view(*x.shape[:-1], self.num_ways, self.num_states) + self.effective_prior().to(
            dw.dtype
        )
        if self.dw_norm == "softmax":
            dw = torch.softmax(dw, dim=-1)
        return dw

    @staticmethod
    def aggregate(dw: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``Xbar_c = sum_j dw[t, c, j] * states[t, j]`` for every stream ``c``."""
        out = torch.einsum("tcn,tnd->ctd", dw.float(), states.float()).to(states.dtype)
        return out.unbind(0)


def record_mudd_stats(stats, layer_idx, dw=None, n_sources=None, period=None):
    """Connection-weight diagnostics (``return_attn_res_stats=True``)."""
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": "mudd"}
    with torch.no_grad():
        if dw is not None:
            w = dw.detach().float()
            flat = w.reshape(-1, w.shape[-1])  # (T * C, num_states)
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
    """Per-rank output widths of ``(q, k, v)`` inside a fused ``qkv_proj``.

    An engine that fuses the three projections publishes its own geometry; anything
    else is derived from the attention module's head counts and ``head_dim``, which
    are the same numbers.  The slice is only ever taken when it adds up to the fused
    weight's row count, so a layout this does not understand raises instead of
    silently returning the wrong heads.
    """
    total = int(fused.weight.shape[0])
    sizes = getattr(fused, "output_sizes", None)  # vLLM: the unsharded widths
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
    """``attn``'s q/k/v projections, as three callables: separate (HF) or fused (engine).

    The trunk this file was written against holds ``q_proj`` / ``k_proj`` / ``v_proj``
    as three modules, and ``_multiway_attention`` needs them separately because MUDD
    feeds a *different* stream to each (``MHA(LN(X^Q), LN(X^K), LN(X^V))``).  An
    inference engine is entitled to merge them -- vLLM's ``QKVFuser`` builds one
    ``qkv_proj`` and **deletes** the three attributes, so the read above fails with
    ``AttributeError: 'Qwen3Attention' object has no attribute 'q_proj'``.

    Merging is arithmetically neutral: one fused linear is the concatenation of the
    three along its output axis (per rank: ``q`` rows, then ``k``, then ``v``), so the
    fused weight sliced back apart *is* the three original projections, at the same
    cost.  Returning callables keeps ``_multiway_attention`` identical either way.
    """
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
    """Qwen3 block with MUDD connections (official ``TransformerBlock`` + ``Compose``)."""

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
            # Eq. (7): LN(X^Q), LN(X^K), LN(X^V)
            self.input_layernorm_q = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.mudd_sepln:
                self.input_layernorm_k = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.input_layernorm_v = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            # plain Qwen3 (connection off) or the single-stream variant
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
                if config.mudd_post_norm:  # official PostDANorm: X + Norm(DA), scale 0.001
                    self.dense_post_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    with torch.no_grad():
                        self.dense_post_norm.weight.fill_(0.001)
                if config.mudd_pre_norm:  # official PreDANorm: the state list is normalised
                    self.state_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _make_mlp(config: Qwen3MUDDConfig, layer_idx: int):
        """FFN, optionally with the paper's parameter re-allocation (Eq. 9).

        Official ``FeedForward(..., scale_with_layer=True)``::
            hid = intermediate_size * (lidx / (n_layer - 1) + 0.5);  hid = round(hid / 128) * 128
        The rounding is kept, but clamped to at least one 128-block: the official
        expression collapses to a zero-width FFN for very narrow models (which is
        not worth reproducing in the toy configs used here).
        """
        if not config.mudd_ffn_depth_scaling:
            return Qwen3MLP(config)
        depth = max(1, config.num_hidden_layers - 1)
        factor = layer_idx / depth + 0.5  # official scale_with_layer: 0.5 -> 1.5
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

    # -- the block ----------------------------------------------------------
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
        """``MHA(LN(X^Q), LN(X^K), LN(X^V))`` -- Qwen3 attention, three inputs.

        The arithmetic is a line-for-line copy of ``Qwen3Attention.forward`` with the
        three projections fed from different streams (and the same ``q_norm`` /
        ``k_norm`` / RoPE / attention interface); when the three streams coincide it
        is *bit-identical* to the trunk's attention, which
        ``test_baselines_mudd_df.py`` checks numerically.

        The projections come from ``_qkv_projections``, so the block runs whether the
        trunk's three projections are still three modules or were fused into one by the
        engine serving the model.
        """
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

    # -- forward ------------------------------------------------------------
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
            # ``hidden_states`` is a plain tensor here (no stream decoupling)
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
            # layer 0, single-stream mode, or a block that follows a non-DA layer
            xq = xk = xv = xr = hidden_states

        # ---- the block (Eq. 7) -------------------------------------------
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

        # ---- append the newest depth state (official Compose) -------------
        state = self.state_norm(block_out) if self.state_norm is not None else block_out
        flat = state.reshape(-1, state.shape[-1])
        depth_states = (
            flat.unsqueeze(1)
            if depth_states is None
            else torch.cat([depth_states, flat.unsqueeze(1)], dim=1)
        )

        if self.dense_conn is None:
            return block_out, depth_states

        # ---- DA: dynamic weights -> aggregate all states (Eq. 5/6/8) ------
        dw = self.dense_conn(block_out.reshape(-1, self.hidden_size))
        streams = MultiwayDynamicDense.aggregate(dw, depth_states)
        record_mudd_stats(attn_res_stats, self.layer_idx, dw=dw, period=self.period)
        if self.dense_post_norm is not None:  # official PostDANorm: X_i + Norm(DA)
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
    """Qwen3 backbone with Multiway Dynamic Dense connections."""

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
        # Delegate to the transformers default: it also rebuilds the non-persistent RoPE
        # buffers, which is what makes from_pretrained work (the checkpoint is built on the
        # meta device, so any buffer this method skips stays uninitialised).  The connection
        # anchors are re-established in ``post_init`` after this generic pass.
        super()._init_weights(module)

    def post_init(self):
        super().post_init()
        # The DA initialisation (W2 = 0, identity prior) is a construction
        # guarantee; re-establish it after the generic weight init, as GDAR does.
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
            # X_0 = the embedding seeds the state list (official hiddens = [x])
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
            # no DA on the last layer: MUDDFormer(X) = Xbar_L^R, the residual stream
            hidden_states = hidden_states[-1]
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3MUDDForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 + MUDD connections, causal LM head."""

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
        pass  # already registered (module imported more than once)

# ``register_for_auto_class`` is what makes a *fresh* process able to resolve
# ``model_type = "qwen3_mudd"`` straight from a checkpoint directory: it sets
# ``_auto_class``, which makes ``save_pretrained`` write ``auto_map`` into
# ``config.json`` and copy these modules next to the weights, and it is the flag
# the ``trust_remote_code=True`` path checks.  Together with the
# ``AutoConfig.register`` / ``AutoModelForCausalLM.register`` calls above it
# covers both routes -- imported package and checkpoint-local code -- because
# verl's MegatronWorker does
# ``AutoConfig.from_pretrained(local_path, trust_remote_code=...)`` on a
# checkpoint whose directory this package is not on ``sys.path`` for.
for _cls, _auto in (
    (Qwen3MUDDConfig, "AutoConfig"),
    (Qwen3MUDDModel, "AutoModel"),
    (Qwen3MUDDForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
