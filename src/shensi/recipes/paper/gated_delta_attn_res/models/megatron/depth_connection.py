# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
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
"""Depth-connection operators of the "snapshot list" family -- Megatron-Core port.

Line-for-line ports of the HuggingFace reference files in ``code/models/``:

===========================  =========================================
``modeling_qwen3_ar.py``     Kimi Attention Residuals (cumulative sources,
                             *replacement* read)
``modeling_qwen3_dar.py``    Delta Attention Residuals (delta sources,
                             *additive* read; optional learnable null source)
``modeling_qwen3_denseformer.py``  DenseFormer depth-weighted average
                             (``DWA``: one scalar per (event, source) pair)
``modeling_qwen3_mudd.py``   MUDDFormer multiway dynamic dense connection,
                             static-key form (``DA``: GELU MLP + static prior)
===========================  =========================================

All four share one state model: the layer carries a *stream* ``prefix``
``[T, H]`` plus a list of *sources* ``[T, N, H]``.  What differs is where the
sources come from (cumulative snapshots vs deltas vs block outputs), how the
read is applied (replacement vs additive vs weighted average) and how the
sublayer output is written back.  Each class below implements exactly one of
those operations, in fp32 where the reference is in fp32, and the layer
(``depth_layer.py``) owns the bookkeeping.

Deviations from the reference files, all of them labelled in the code:

* Megatron ``nn.Module`` plumbing; ``reset_parameters()`` reproduces HF's
  ``_init_weights`` (Linear: ``N(0, init_method_std)``; RMSNorm weight: ones)
  followed by the module-specific init, inside the caller's forked RNG.
* ``identity`` modes: the reference operators are *not* the identity at step 0
  (a randomly initialised router replaces / perturbs the sublayer input).  A
  zero-initialised scalar gate is added so that a variant can be pinned to the
  plain Megatron residual bit-exactly at step 0; with the gate at 1 the operator
  is the reference one again (for DAR/DenseFormer/MUDD even bit-exactly, see the
  class docstrings).  ``force_identity`` is the run-time guard: it short-circuits
  the connection to the plain residual path regardless of the weights
  (the same semantics as ``_force_identity`` in ``modeling_qwen3_hc.py`` /
  ``modeling_qwen3_mhc.py``).
* The reference's ``record_router_stats`` diagnostics are not ported (no
  equivalent sink in the Megatron forward signature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DepthConnectionConfig",
    "ARRouter",
    "DeltaRouter",
    "DepthWeightedAverage",
    "MultiwayDynamicDense",
    "RMSNormNoScale",
    "WeightedRMSNorm",
]


# ---------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------


class WeightedRMSNorm(nn.Module):
    """``models/modeling_qwen3_ar.py::RMSNorm`` (the Kimi norm) verbatim.

    ``x -> weight * (x/float32 * rsqrt(mean(x^2)+eps)).to(input_dtype)``: the
    normalised tensor is rounded back to the *input* dtype before the weight is
    applied, exactly as the reference writes it, and the result of that multiply
    is fp32 (bf16 x * fp32 weight promotes).  AR and DAR both use it for the
    routing keys.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNormNoScale(nn.Module):
    """``RMSnormNoscale`` from the official MUDDFormer code: RMS norm, no weight."""

    def __init__(self, dim: int = -1, eps: float = 1.0e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=self.dim, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype)


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------


@dataclass
class DepthConnectionConfig:
    """Knobs shared by the four snapshot-list variants.

    ``variant``
        ``"ar"`` | ``"dar"`` | ``"denseformer"`` | ``"mudd"``.
    ``block_size``
        The reference ``attn_res_block_size``.  ``None`` disables the connection
        entirely (plain Qwen3).  For AR/DAR it is the snapshot *period* (``1`` =
        one source per sublayer write, the default in this port); for
        DenseFormer/MUDD it is the *event* period (``1`` = one DWA/DA module per
        block, i.e. the published method).
    ``output_route``
        One extra read/DWA/DA at the last layer so that the model output is a
        connection output (the reference's ``attn_res_output_route``).
    ``identity``
        Pin the operator to the plain residual at step 0 through a
        zero-initialised scalar gate (AR) or the identity target of the
        published initialisation (DenseFormer / MUDD / DAR-additive).  See each
        class.
    ``force_identity``
        Run-time guard: the connection returns its input unchanged (plain
        residual), regardless of the learned weights.
    """

    variant: str = "dar"
    block_size: int | None = 1
    output_route: bool = True

    # --- AR ---------------------------------------------------------------
    #: how the accumulator behaves at a block boundary.  ``"zero"`` is the Kimi
    #: reference (the block's state is the sum of its own sublayer outputs);
    #: ``"keep"`` seeds it with the incoming stream, which is what makes
    #: ``AR(0) == plain Qwen3`` a bit-exact statement.
    ar_reset: str = "keep"

    # --- DAR --------------------------------------------------------------
    #: the reference's learnable null source (zeros at init, prepended to the
    #: source list so a random query still has a near-identity option).
    use_null_source: bool = False

    # --- DenseFormer / MUDD ----------------------------------------------
    #: DWA weight parameterisation: ``"deviation"`` = ``one_hot(last) + delta``
    #: with ``delta = 0`` (identity anchor with a live gradient);
    #: ``"official"`` = a raw vector initialised to the same one-hot.
    dwa_param: str = "deviation"
    #: official ``DWAModules(dilation=k)``: keep every k-th source.
    dwa_dilation: int = 1

    # --- MUDD -------------------------------------------------------------
    #: number of decoupled streams (``4`` = the paper's qkvr multiway variant).
    #: Only ``1`` -- the official ``dense_type='l'`` single-stream DA -- is
    #: ported here; see the note in ``MultiwayDynamicDense``.
    mudd_num_ways: int = 1
    mudd_param: str = "deviation"
    mudd_act: str = "gelu"
    mudd_hidden_round: int = 64
    mudd_scale_dw: bool = False
    mudd_use_post_norm: bool = False
    mudd_use_pre_norm: bool = False

    # --- init -------------------------------------------------------------
    #: pin the operator to the plain residual at step 0 through a
    #: zero-initialised scalar gate (AR / DAR).  DenseFormer and MUDD are exact
    #: at their *official* initialisation and do not use a gate (one would kill
    #: their gradients) -- see the class docstrings.
    identity: bool = True
    #: run-time guard: the connection is a no-op whatever the weights are.
    force_identity: bool = False
    #: HF ``initializer_range`` / Megatron ``init_method_std``.
    init_std: float = 0.02
    #: dropout applied to the sublayer output before the write (the connection
    #: replaces ``bias_dropout_add``).  ``None`` = follow ``config.hidden_dropout``
    #: (off by default, which is what keeps every run deterministic).
    residual_dropout: float | None = None

    def validated(self) -> DepthConnectionConfig:
        if self.variant not in ("ar", "dar", "denseformer", "mudd"):
            raise ValueError(f"unknown depth variant {self.variant!r}")
        if self.block_size is not None and int(self.block_size) < 1:
            raise ValueError(f"block_size must be >= 1 or None, got {self.block_size}")
        if self.ar_reset not in ("zero", "keep"):
            raise ValueError(f"ar_reset must be 'zero' or 'keep', got {self.ar_reset!r}")
        if self.dwa_param not in ("deviation", "official"):
            raise ValueError(f"dwa_param must be 'deviation' or 'official', got {self.dwa_param!r}")
        if self.mudd_param not in ("deviation", "official", "random"):
            raise ValueError(
                f"mudd_param must be deviation/official/random, got {self.mudd_param!r}"
            )
        if int(self.dwa_dilation) < 1:
            raise ValueError(f"dwa_dilation must be >= 1, got {self.dwa_dilation}")
        return self

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


# ---------------------------------------------------------------------------
# AR: cumulative sources, replacement read (Kimi ``_apply_attn_res``)
# ---------------------------------------------------------------------------


class ARRouter(nn.Module):
    """``_apply_attn_res(prefix_sum, block_residual, proj, norm)`` + identity gate.

    Reference (``modeling_qwen3_ar.py``)::

        v = cat([blocks, prefix.unsqueeze(1)], dim=1)  # (T, N+1, H)
        k = v.float() * rsqrt(mean(v.float() ** 2) + eps)
        score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
        probs = (k * score_weight).sum(-1).softmax(-1)
        routed = probs @ v.float()  # (T, H)

    and the routed vector **replaces** the sublayer input.  With
    ``identity=True`` the returned value is ``prefix + s * (routed - prefix)``
    with ``s`` a zero-initialised scalar, so ``s = 0`` gives back exactly the
    stream (``x + 0 == x`` bit-for-bit) and ``s = 1`` is the reference operator.
    ``proj``/``norm`` are skipped entirely when there is nothing to route over,
    as in the reference.
    """

    def __init__(self, hidden: int, cfg: DepthConnectionConfig, eps: float = 1e-6):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.identity = bool(cfg.identity)
        self.force_identity = bool(cfg.force_identity)

        self.norm = WeightedRMSNorm(hidden, eps=eps)
        self.proj = nn.Linear(hidden, 1, bias=False)
        if self.identity:
            self.read_scale = nn.Parameter(torch.zeros(1))
        else:
            self.read_scale = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """HF ``_init_weights``: Linear ``N(0, init_std)``, RMSNorm weight ones."""
        std = float(self.cfg.init_std)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=std)
            nn.init.ones_(self.norm.weight)
            if self.read_scale is not None:
                self.read_scale.zero_()

    def forward(self, prefix: torch.Tensor, sources: torch.Tensor | None) -> torch.Tensor:
        """``(T, H)`` stream + ``(T, N, H)`` sources -> ``(T, H)`` routed value."""
        if self.force_identity or sources is None or sources.shape[1] == 0:
            return prefix
        v = torch.cat([sources.to(prefix.dtype), prefix.unsqueeze(-2)], dim=-2)
        v_float = v.float()
        variance = v_float.pow(2).mean(-1, keepdim=True)
        k = v_float * torch.rsqrt(variance + self.norm.variance_epsilon)
        score_weight = self.norm.weight.float() * self.proj.weight.squeeze(0).float()
        scores = (k * score_weight).sum(-1)
        probs = scores.softmax(-1)
        routed = torch.matmul(probs.unsqueeze(1), v_float).squeeze(1).to(prefix.dtype)
        if self.read_scale is None:
            return routed  # the reference: replacement
        return prefix + self.read_scale.to(prefix.dtype) * (routed - prefix)


# ---------------------------------------------------------------------------
# DAR: delta sources, additive read (``delta_attn_res``)
# ---------------------------------------------------------------------------


class DeltaRouter(nn.Module):
    """``delta_attn_res`` from ``modeling_qwen3_dar.py`` + identity gate.

    ``K = norm(V)``; ``logits = <query, K>`` with ``query = proj.weight.view(-1)``;
    ``weights = softmax(logits, dim=0)``; ``selected = sum_i weights_i V_i``;
    ``output = partial_block + selected`` -- the read is **added** to the stream.
    The optional learnable null source (a zero vector at init) is prepended,
    exactly as the reference does.

    With ``identity=True`` the returned value is ``prefix + s * selected`` with a
    zero-initialised ``s``: at ``s = 0`` the result is bit-exactly ``prefix`` and
    at ``s = 1`` it is the reference value (``prefix + 1 * selected`` is the same
    arithmetic), so for DAR the identity gate costs nothing but a scalar.
    """

    def __init__(
        self, hidden: int, cfg: DepthConnectionConfig, eps: float = 1e-6, null: bool = False
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.identity = bool(cfg.identity)
        self.force_identity = bool(cfg.force_identity)

        self.norm = WeightedRMSNorm(hidden, eps=eps)
        self.proj = nn.Linear(hidden, 1, bias=False)
        self.null_source = nn.Parameter(torch.zeros(hidden)) if null else None
        if self.identity:
            self.read_scale = nn.Parameter(torch.zeros(1))
        else:
            self.read_scale = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = float(self.cfg.init_std)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=std)
            nn.init.ones_(self.norm.weight)
            if self.null_source is not None:
                self.null_source.zero_()
            if self.read_scale is not None:
                self.read_scale.zero_()

    def forward(self, prefix: torch.Tensor, sources: torch.Tensor | None) -> torch.Tensor:
        if self.force_identity:
            return prefix
        entries = [] if sources is None or sources.shape[1] == 0 else list(sources.unbind(dim=-2))
        if self.null_source is not None:
            entries = [self.null_source.to(prefix.dtype).expand_as(prefix), *entries]
        if not entries:
            return prefix

        V = torch.stack(entries, dim=0)  # (N, T, H)
        K = self.norm(V)
        query = self.proj.weight.view(-1)
        logits = torch.einsum("d, n t d -> n t", query, K)
        weights = logits.softmax(dim=0)
        selected = torch.einsum("n t, n t d -> t d", weights.to(V.dtype), V)
        if self.read_scale is None:
            return prefix + selected.to(prefix.dtype)  # the reference
        return prefix + self.read_scale.to(prefix.dtype) * selected.to(prefix.dtype)


# ---------------------------------------------------------------------------
# DenseFormer: depth-weighted average
# ---------------------------------------------------------------------------


class DepthWeightedAverage(nn.Module):
    """``DepthWeightedAverage`` from ``modeling_qwen3_denseformer.py`` verbatim.

    ``sources`` is ``(T, N, H)`` with the *last* entry = the block output that
    just ran (the official ``InPlaceSetSlice`` order), ``Y = sum_j alpha_j X_j``.
    ``dwa_param="deviation"`` keeps ``alpha = one_hot(last) + delta`` with
    ``delta`` zero-initialised: the forward value is the official init exactly
    and ``dL/d delta_j = <dL/dY, X_j>`` is non-zero, so the dense path is
    trainable (a multiplicative zero gate would freeze it, see the HF file).
    """

    def __init__(self, num_sources: int, cfg: DepthConnectionConfig):
        super().__init__()
        self.num_sources = int(num_sources)
        self.param_mode = cfg.dwa_param
        if self.param_mode == "official":
            self.alpha = nn.Parameter(torch.zeros(self.num_sources))
        else:
            self.alpha_delta = nn.Parameter(torch.zeros(self.num_sources))
        self.force_identity = bool(cfg.force_identity)
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
        """``(T, N, H)`` -> ``(T, H)``; identity when ``alpha = one_hot(last)``."""
        if self.force_identity:
            return sources[..., -1, :].contiguous()
        alpha = self.effective_alpha().to(sources.dtype)
        return torch.einsum("tnd,n->td", sources.float(), alpha.float()).to(sources.dtype)


# ---------------------------------------------------------------------------
# MUDDFormer: multiway dynamic dense connection (static-key form)
# ---------------------------------------------------------------------------


class MultiwayDynamicDense(nn.Module):
    """``MultiwayDynamicDense`` from ``modeling_qwen3_mudd.py`` (single stream).

    The official operator (Sec. 2.2 Eq. 5/6 of arXiv:2502.12170) is::

        A_i    = GELU(RMSNorm(X_i) W1) W2 + a_i          # (T, num_states)
        Xbar_i = sum_j A_{i,j} X_j

    with **no softmax**: ``W2`` plays the role of static, input-independent keys
    and ``a_i`` (the official ``dense_proj2.bias`` / ``dense_bs``) is the static
    prior.  ``mudd_param="deviation"`` keeps ``a = one_hot(last) + delta`` with
    ``delta = 0`` and ``W2 = 0`` at init (the official initialisation), so at
    step 0 ``Xbar_i = X_i`` bit-exactly with a live gradient on both the prior
    and the generated weights.

    ``num_ways``: only the single-stream form (official ``dense_type='l'``) is
    implemented, because Megatron's ``SelfAttention`` takes *one* input tensor
    and its fused ``linear_qkv`` cannot be split into three decoupled
    (query, key, value) streams without replacing the attention module (and
    with it the backbone this port keeps identical).

    ``mudd_scale_dw`` / ``mudd_use_pre_norm`` / ``mudd_use_post_norm`` are the
    official JAX flags (``dynamic_dense_scale_dw``, ``PreDANorm``,
    ``PostDANorm``); all default off, as in the HF file.
    """

    def __init__(self, hidden: int, num_states: int, cfg: DepthConnectionConfig, eps: float = 1e-6):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden
        self.num_states = int(num_states)
        self.num_ways = 1
        self.param_mode = cfg.mudd_param
        self.scale_dw = bool(cfg.mudd_scale_dw)
        self.force_identity = bool(cfg.force_identity)

        out_dim = self.num_ways * self.num_states
        hidden_dim = out_dim
        round_to = int(cfg.mudd_hidden_round or 0)
        if round_to > 0:  # official dynamic_dense_hidden_round = (K // 64 + 1) * 64
            hidden_dim = (hidden_dim // round_to + 1) * round_to
        self.hidden_dim = hidden_dim

        act = str(cfg.mudd_act).lower()
        if act not in ("gelu", "silu", "relu"):
            raise ValueError(f"unsupported mudd_act {cfg.mudd_act!r}")
        self.act_name = act
        self.norm = (
            WeightedRMSNorm(hidden, eps=eps) if cfg.mudd_use_pre_norm else RMSNormNoScale(eps=eps)
        )
        self.w1 = nn.Linear(hidden, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.zero_prior = bool(cfg.mudd_use_pre_norm and cfg.mudd_use_post_norm)
        if self.param_mode == "official":
            self.prior = nn.Parameter(torch.zeros(self.num_ways, self.num_states))
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
        """Official init: ``W2 = 0`` and the prior at its identity target.

        ``W1`` is the HF generic init (``N(0, init_std)``); ``W2`` is then zeroed,
        exactly like the reference's ``reset_parameters``.
        """
        std = float(self.cfg.init_std)
        with torch.no_grad():
            nn.init.normal_(self.w1.weight, mean=0.0, std=std)
            if isinstance(self.norm, WeightedRMSNorm):
                nn.init.ones_(self.norm.weight)
            if self.param_mode == "official":
                target = torch.zeros_like(self.prior)
                if not self.zero_prior:
                    target[:, -1] = 1.0
                self.prior.copy_(target)
                nn.init.zeros_(self.w2.weight)
            elif self.param_mode == "random":
                nn.init.zeros_(self.w2.weight)
            else:
                self.prior_delta.zero_()
                nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(T, H)`` -> ``(T, num_states)`` generated connection weights."""
        if self.force_identity:
            # the identity slice of the DA: A_i = one_hot(last), i.e. Xbar_i = X_i
            out = torch.zeros(*x.shape[:-1], self.num_states, dtype=x.dtype, device=x.device)
            out[..., -1] = 1.0
            return out
        act = (
            F.gelu(self.w1(self.norm(x)))
            if self.act_name == "gelu"
            else (
                F.silu(self.w1(self.norm(x)))
                if self.act_name == "silu"
                else F.relu(self.w1(self.norm(x)))
            )
        )
        dw = self.w2(act)
        if self.scale_dw:  # official dynamic_dense_scale_dw
            dw = dw / math.sqrt(self.hidden_dim)
        dw = dw.view(*x.shape[:-1], self.num_ways, self.num_states)
        return dw + self.effective_prior().to(dw.dtype)

    @staticmethod
    def aggregate(dw: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """``Xbar_c = sum_j dw[t, c, j] * states[t, j]`` for the single stream."""
        out = torch.einsum("tcn,tnd->ctd", dw.float(), states.float()).to(states.dtype)
        return out
