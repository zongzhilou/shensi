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
r"""Qwen3 + Manifold-Constrained Hyper-Connections (mHC).

**Ported from the official implementation on this machine**:
``code/Megatron-LM-FL/megatron/core/transformer/hyper_connection.py``
(NVIDIA copyright header, 2026; megatron-core 0.18.2), plus its call sites
``megatron/core/transformer/transformer_layer.py::HyperConnectionTransformerLayer``,
``transformer_block.py`` (``input_expand`` / ``output_contract`` /
``learned_output_contract``) and ``transformer_config.py``
(``num_residual_streams=4``, ``mhc_sinkhorn_iterations=10``,
``mhc_init_gating_factor=0.01``, ``use_fused_mhc``).  The paper is
Xie et al., *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880
(DeepSeek-AI).  No standalone official repository exists (checked through the
GitHub API: nothing under ``deepseek-ai`` or ``bytedance``); the Megatron file is
the released code, so this port is the reference, not a reconstruction.

What is copied verbatim
-----------------------
``_sinkhorn_iterations`` (row/column renormalisation of ``softmax(logits)`` with
an epsilon floor, \(t_{\\text{max}}\) iterations -- the kernel is not ported, only
the maths), ``_compute_h`` (the coefficient construction), ``native_proj_rms``
(\(r = 1/(\\|x\\|/\\sqrt{nC} + \\epsilon)\), the RMSNorm weight absorbed into
\(\\varphi\)), ``native_h_aggregate``, ``native_h_post_bda``,
``apply_h_res``, ``learned_output_contract``, ``input_expand``,
``output_contract`` and ``HyperConnectionModule._init_weights``.  Not ported:
the CUDA/TileLang kernels, the ``torch.autograd.Function`` Sinkhorn variant
(recompute is an infra optimisation; the plain autograd path is used, which is
the same function and is *not* a numerical approximation), the
``CheckpointManager`` recompute plumbing, TP/sequence-parallel attributes and
dropout (the Qwen3 baseline has none).

The propagation is the paper's Eqs. (3) and (7)-(9)::

    x_{l+1} = H_res^T x_l + H_post^T F(H_pre x_l)                 # Eq. (3), n x C streams

    vec(x)'  = RMSNorm(vec(x_l))                                  # Eq. (7), flattened to nC
    H~_pre   = alpha_pre  * (vec(x)' @ phi_pre)  + b_pre          #   phi_pre  : nC -> n
    H~_post  = alpha_post * (vec(x)' @ phi_post) + b_post         #   phi_post : nC -> n
    H~_res   = alpha_res  * mat(vec(x)' @ phi_res) + b_res        #   phi_res  : nC -> n^2
    H_pre    = sigmoid(H~_pre)                                    # Eq. (8)
    H_post   = 2 sigmoid(H~_post)                                 # Eq. (8)
    H_res    = Sinkhorn-Knopp(H~_res)                             # Eqs. (8)-(9), t_max = 10 here
    x_{l+1}  = H_res^T x_l + H_post^T F(H_pre x_l)                # H_pre x_l = sum_j h_pre_j x_j

with ``Sinkhorn-Knopp(M)``: ``M <- softmax(M) + eps`` then
``M <- M / (M.sum(-1) + eps); M <- M / (M.sum(-2) + eps)`` repeated
``t_max - 1`` times (``_MHC_SINKHORN_EPS = 1e-6``, the paper uses
\(t_{\\text{max}} = 20\), the released code defaults to 10).

Initialisation (Megatron, ``HyperConnectionModule._init_weights``)
------------------------------------------------------------------
* ``mapping_proj`` (the \(\\varphi\) of Eq. 7, ``Linear(nC -> n^2 + 2n, bias=False)``):
  ``xavier_uniform_``;
* ``alpha_pre/post/res``: ``mhc_init_gating_factor`` = 0.01 (the paper's "small
  values", Table 4 of the appendix; float32 scalars);
* static biases ``b_pre, b_post, b_res``: **zeros**.
  With zero biases and \(n=4\) the released init is therefore
  \(H_{pre} = \\sigma(0) + 10^{-6} = 0.500001\) per stream,
  \(H_{post} = 2\\sigma(0) = 1\), and
  \(H_{res} = \\text{Sinkhorn}(0) = \\mathbf{1}\\mathbf{1}^T/n\) -- the *uniform*
  doubly stochastic matrix, not the identity.  Measured deviation from the
  standard PreNorm residual: see ``test_baselines_hc.py``.
* ``input_expand``: the single stream is **replicated** n times;
  ``output_contract``: the n streams are **averaged** (the paper's Algorithm 1
  sums the rows; the released code averages, which is the scale-preserving
  choice and is what makes contraction commute with the identity mapping);
  ``learned_output_contract`` (DSv4): a learnable ``n x nC`` head with
  ``base = 0``, ``scale = 1``, ``pre = sigmoid(mixes * scale + base) + 1e-6``,
  ``y = sum_j pre_j x_j`` -- the scaling is what the paper describes as
  "the RMSNorm weight absorbed in the projection" applied to the readout.

Identity initialisation (``hc_init="identity"``, the default here)
------------------------------------------------------------------
The paper's own claim is that the doubly stochastic constraint restores
*stability* (norm preservation, compositional closure), not that step 0 is the
standard residual.  For the comparison in this directory the lower bound is
required to be **exact**, so the identity mode pins the mappings to the exact
PreNorm-residual point:

===================  ===========================================================
``H_pre``            ``sigma`` read cannot hit 1/n exactly, so the identity mode
                     uses the normalised non-negative read (``hc_read="simplex"``,
                     softmax of zero logits = exactly 1/n) or the unconstrained
                     read (``hc_read="linear"``, ``b_pre = e_k``): both give
                     ``sum_j h_pre_j = 1`` exactly.
``H_post``           ``2 sigma(0) = 1`` exactly (zero bias) -- already exact.
``H_res``            Sinkhorn(zero logits) = uniform ``1/n``, an exact fixed
                     point *provided eps = 0*: with the released
                     ``eps = 1e-6`` the uniform point moves by ~n*eps and the
                     identity is only approximate (~1e-6 relative).  The
                     identity mode therefore forces ``eps = 0``; that is safe
                     because the Sinkhorn start is ``softmax``, whose rows sum to
                     1, so no row or column can sum to zero.
``alpha_*``          exactly 0 (the branch contributes 0 at step 0) while the
                     xavier projection stays a non-zero carrier, so
                     ``d H / d alpha != 0`` and the dynamic branch is reachable
                     (a zero carrier would freeze it forever).
===================  ===========================================================

The identity point is stream-permutation-symmetric, and that has a measured
consequence for what can learn there (full table in ``test_baselines_hc.py``,
section 2b):

* the n streams are bit-identical, so the read is insensitive to its own weights
  (a convex combination of equal vectors) and the mix is insensitive to ``H_res``
  (whose column sums are 1) -- the only coefficient with a non-zero gradient is
  the **write** ``H_post``;
* for the *projected* ``H_res`` that gradient is **exactly zero**, not merely
  small: the mix gradient is ``a_i x_j`` with ``a_i`` equal across streams, i.e.
  the uniform matrix, and the Birkhoff projection (whose tangent space is
  ``{B : B1 = 0, 1^T B = 0}``) annihilates exactly that direction.  Measured:
  ``|dL/dH_res| = 0.00e+00`` at step 0, ``H_res`` logits still at 1e-12 after 30
  steps.  The symmetry is broken by ``H_post`` becoming stream-asymmetric, which
  needs the input-dependent branch (``hc_dynamic=True``, the Megatron default) --
  ``mHC-lite`` (static + Sinkhorn + identity) is an *exactly* stable symmetric
  point (measured stream spread 0.0 after 30 AdamW steps): use it with
  ``hc_dynamic=True`` or ``hc_init="official"`` if the mixing is meant to learn.
  Note this also holds for the released initialisation (``alpha = 0.01``): its
  ``|dL/dH_res|`` measured 7.3e-19 - 2.0e-18, i.e. the mapping the paper's
  Table 1 calls the most valuable one is the slowest to start moving.

The three mixing points are evaluated in the algebraically identical
"relative-to-a-reference-stream" form::

    sum_j w_j x_j        ==  x_0 * (sum_j w_j) + sum_j w_j (x_j - x_0)
    (H^T x)_i            ==  x_0 * (sum_j H_ji) + sum_j H_ji (x_j - x_0)

Both sides are exactly equal in real arithmetic for *any* weights, and the
re-associated form returns ``x_0`` bit-exactly whenever the n streams are
bit-identical and the weight/sum term is exactly 1 -- which is what makes
``torch.equal(HC(0), h + f(h))`` true instead of merely close.  Deviation from
the literal Megatron expression is measured in ``test_baselines_hc.py``.

Block layout (same semantics as ``attn_res_block_size`` elsewhere)
-----------------------------------------------------------------
``attn_res_block_size`` is the length of a *stream chunk*: the n-stream residual
is created at the chunk entry (``input_expand``: replicate) and collapsed at the
chunk exit (``output_contract``), i.e. exactly Megatron's TransformerBlock
boundary with pipeline parallelism.  ``None`` -> stock Qwen3; ``>= num_hidden_layers``
-> one chunk for the whole stack (the Megatron-equivalent end-to-end n-stream
layout); ``1`` -> the streams exist only inside a single decoder layer.

Known limitations
-----------------
* **Decoding with a KV cache is not handled**: the n-stream residual is a
  per-forward activation, not a cache entry, so incremental decoding would
  restart the streams from the current token's embedding.  Training and
  full-prefill (``use_cache=False``) are the supported paths -- the same caveat
  applies to the AR/DAR/GDAR depth state in this directory.
* The fused kernels, recompute plumbing and TP sharding of the reference are
  not ported; parameter counts and FLOPs are therefore identical to Megatron's
  native path, but wall-clock is not.
"""

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


# ---------------------------------------------------------------------------
# Manifold projection: Sinkhorn-Knopp (ported from _sinkhorn_iterations)
# ---------------------------------------------------------------------------


def sinkhorn_knopp_project(
    logits: torch.Tensor, num_iterations: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    """Project ``logits`` onto the Birkhoff polytope (doubly stochastic matrices).

    Line-for-line port of ``_sinkhorn_iterations`` from
    ``megatron/core/transformer/hyper_connection.py``: start from a row-stochastic
    ``softmax`` (not from ``exp`` as in the paper's Eq. 9 -- Megatron normalises
    first, which is the same map up to the row normalisation being applied one
    step earlier), then alternately renormalise rows and columns ``num_iterations``
    times with an ``eps`` floor guarding the divisions.

    ``eps = 0`` is required for the uniform matrix to be an *exact* fixed point
    (the identity initialisation relies on it); it is safe because the softmax
    start makes every row sum exactly 1, so no row or column sum can be zero.
    """
    matrix = logits.softmax(dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(num_iterations) - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def _read_weights(logits: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    """``H_pre`` from its logits.  See ``Qwen3MHCConfig.hc_read``."""
    if mode == "simplex":
        return logits.softmax(dim=-1)
    if mode == "sigmoid":
        return logits.sigmoid() + eps
    if mode == "linear":
        return logits
    raise ValueError(f"unknown hc_read: {mode}")


def _write_weights(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """``H_post`` from its logits.  See ``Qwen3MHCConfig.hc_write``."""
    if mode == "sigmoid2":
        return logits.sigmoid() * 2
    if mode == "linear":
        return logits
    raise ValueError(f"unknown hc_write: {mode}")


def _identity_index(layer_index: int, sublayer: str, num_streams: int) -> int:
    """``k mod n`` of the paper's Eq. 14, with one module per *sublayer*.

    The paper's ``k`` is the layer index; here a decoder layer holds two
    hyper-connection modules (attention and MLP, as in Megatron), so the index is
    ``2 * layer_index (+1 for the MLP)``, which spreads the static reads over the
    streams exactly as the reference does over layers.
    """
    offset = 0 if sublayer == "attn" else 1
    return (2 * layer_index + offset) % num_streams


# ---------------------------------------------------------------------------
# The connection module (ported from HyperConnectionModule)
# ---------------------------------------------------------------------------


class HyperConnection(nn.Module):
    """One mHC connection module: n-stream <-> 1-stream, for one sublayer.

    Interface mirrors the reference (``HyperConnectionModule``): a forward that
    returns ``(aggregated, h_res, h_post, residual)`` so the caller can run the
    sublayer in between, and a ``fuse`` that applies ``H_res``/``H_post`` plus the
    bias-dropout-add of the reference (dropout dropped, the Qwen3 baseline has
    none).  Passing ``stats`` collects the quantities the paper's stability
    analysis uses (``Amax Gain Magnitude`` of a single mapping) from the
    coefficients of the pass that just ran.
    """

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
        # The released eps floor moves the uniform fixed point by ~n*eps, which
        # breaks the bit-exact identity; the identity mode therefore uses 0.
        self.sinkhorn_eps = (
            0.0
            if (self.init_mode == "identity" and self.manifold == "doubly_stochastic")
            else float(config.mhc_sinkhorn_eps)
        )

        num_coeffs = self.n * self.n + 2 * self.n
        self.num_coeffs = num_coeffs
        # Eq. (7): one projection per module, from the flattened nC stream.
        self.mapping_proj = (
            nn.Linear(self.n * self.hidden_size, num_coeffs, bias=False) if self.dynamic else None
        )
        # Eq. (5)/(7): learnable gates on the dynamic part.
        self.alpha_pre = nn.Parameter(torch.full((1,), self.gating_factor))
        self.alpha_post = nn.Parameter(torch.full((1,), self.gating_factor))
        self.alpha_res = nn.Parameter(torch.full((1,), self.gating_factor))
        # Static coefficients b_pre | b_post | b_res.
        self.bias = nn.Parameter(torch.zeros(num_coeffs))

        # Set by ``train/guarantee.py``-style guards to pin the module to the identity slice.
        self._force_identity = False

        self.reset_parameters()

    # -- initialisation -----------------------------------------------------
    @torch.no_grad()
    def reset_parameters(self) -> None:
        r"""Weights of the reference, or the exact identity anchor.

        ``hc_init="official"`` is ``HyperConnectionModule._init_weights``: xavier
        ``mapping_proj``, ``alpha = mhc_init_gating_factor``, zero biases.  One
        deviation is unavoidable there: for an *unprojected* ``H_res``
        (``mhc_manifold="none"``, i.e. plain HC) a zero matrix would delete the
        residual stream, so ``H_res`` is anchored at ``I`` -- the HC paper's
        static \(A_r = I\) from its own Eq. 14.
        """
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

        # ---- identity initialisation (ours; see the file header) -----------
        self.alpha_pre.zero_()
        self.alpha_post.zero_()
        self.alpha_res.zero_()
        self.bias.zero_()

        # H_pre: a convex combination summing to exactly 1.
        pre = self.bias[:n]
        if self.read_mode == "linear":
            # the paper's Eq. 14: A_m = e_k
            pre[_identity_index(self.layer_index, self.sublayer, n)] = 1.0
        elif self.read_mode == "simplex":
            pre.zero_()  # softmax(0) = 1/n exactly
        else:  # "sigmoid": the closest representable point, 1/n - eps, + eps
            target = 1.0 / n - self.compute_h_eps
            pre.fill_(math.log(target / (1.0 - target)))

        # H_post: 2*sigmoid(0) = 1 (sigmoid2) or the coefficient 1 (linear).
        post = self.bias[n : 2 * n]
        post.fill_(0.0 if self.write_mode == "sigmoid2" else 1.0)

        # H_res: I when unprojected (exact, no saturation, live gradients),
        # uniform n x n when projected (Sinkhorn's exact fixed point at eps = 0,
        # which mixes all streams equally but keeps every parameter reachable).
        res = self.bias[2 * n :].view(n, n)
        if self.manifold == "none":
            res.copy_(torch.eye(n, dtype=self.bias.dtype))
        else:
            res.zero_()

    # -- the three mappings (ported from compute_mappings / _compute_h) -----
    def compute_mappings(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``streams``: ``(..., n, C)`` -> ``(H_pre, H_post, H_res)``.

        Eq. (7): the hyper hidden matrix is flattened to ``vec(x) in R^{nC}`` and
        RMS-normalised, then projected to ``n^2 + 2n`` coefficients; Megatron
        reorders the normalisation *after* the matmul and absorbs the RMS weight
        into the projection (no learnable norm weight here), with
        ``r = 1 / (||x|| / sqrt(nC) + eps)``.
        """
        n, hidden = self.n, self.hidden_size
        shape = streams.shape[:-2]
        x = streams.reshape(*shape, n * hidden)
        dtype = x.dtype
        # The paper's kernel table keeps the coefficients in float32 (Eq. 12-13).
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

    # -- applications (ported from native_h_aggregate / apply_h_res / ...) --
    @staticmethod
    def _reference(x: torch.Tensor) -> torch.Tensor:
        """Stream 0, used as the anchor of the re-associated mixing forms."""
        return x[..., 0, :]

    def aggregate(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """``H_pre x``: n-stream -> 1-stream, ``(..., n, C)`` -> ``(..., C)``.

        Computed as ``x_0 * sum_j h_j + sum_j h_j (x_j - x_0)`` -- algebraically
        identical to Megatron's ``native_h_aggregate``
        (``(x * h_pre[..., None]).sum(-2)``, deviation measured in the tests),
        and exactly ``x_0`` when the streams are equal and ``sum_j h_j = 1``.
        """
        x_ref = self._reference(streams)
        w_sum = h_pre.sum(dim=-1, keepdim=True)
        correction = ((streams - x_ref.unsqueeze(-2)) * h_pre.unsqueeze(-1)).sum(dim=-2)
        return x_ref * w_sum + correction

    def apply_h_res(self, h_res: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        """``H_res^T x`` (Megatron ``apply_h_res``), same re-association."""
        x_ref = self._reference(streams)
        col_sum = h_res.sum(dim=-2)  # (..., n): sum over j of H_res[j, i]
        correction = torch.einsum("...ji,...jc->...ic", h_res, streams - x_ref.unsqueeze(-2))
        return x_ref.unsqueeze(-2) * col_sum.unsqueeze(-1) + correction

    @staticmethod
    def apply_h_post(x: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
        """``H_post^T x``: 1-stream -> n-stream (Megatron ``_apply_h_post``)."""
        return h_post.unsqueeze(-1) * x.unsqueeze(-2)

    def forward(
        self, streams: torch.Tensor, stats: list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(aggregated, h_res, h_post, residual)`` (Megatron's contract).

        ``stats`` is an optional sink for the diagnostics of
        :meth:`mapping_stats`, using the coefficients this pass already computed
        (no second Sinkhorn); the caller only passes it under
        ``return_attn_res_stats=True``.
        """
        if self._force_identity:
            # The exact standard-residual slice: read the reference stream, keep
            # the coefficients at their identity values (see train/guarantee.py).
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
        """``H_res^T x + H_post^T F(x)`` -- Megatron's ``fused_h_res_h_post_bda``.

        Megatron also expands a bias and applies dropout here; the Qwen3 baseline
        has neither (``attention_dropout=0``, ``bias=False`` linear layers).
        """
        return self.apply_h_res(h_res, residual) + self.apply_h_post(layer_output, h_post)

    # -- diagnostics --------------------------------------------------------
    def mapping_stats(self, h_pre: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> dict:
        """The paper's per-mapping stability quantities (Fig. 3a)."""
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


# ---------------------------------------------------------------------------
# Decoder layer and backbone
# ---------------------------------------------------------------------------


class Qwen3MHCDecoderLayer(nn.Module):
    """Qwen3 decoder layer whose two sublayers read/write an n-stream residual.

    Same sublayers, norms and order as ``Qwen3DecoderLayer``; the difference is
    that the pre-norm input is the *aggregated* stream and the sublayer output is
    written back into all n streams (Megatron's ``HyperConnectionTransformerLayer``).
    """

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

        # ---- self attention ----
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

        # ---- mlp ----
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
        """Stock Qwen3 (``attn_res_block_size=None``): bit-identical to the baseline."""
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
    """Qwen3 backbone with an n-stream residual governed by mHC mappings.

    Layers are grouped into chunks of ``attn_res_block_size``; a chunk is entered
    by replicating the single stream (``input_expand``) and left by collapsing the
    streams (``output_contract``) -- the Megatron TransformerBlock boundary.
    """

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
            # Megatron ``transformer_block.py``: hc_head_fn = randn then xavier,
            # hc_head_base = zeros, hc_head_scale = ones.
            n, hidden = self.num_streams, config.hidden_size
            self.hc_head_fn = nn.Parameter(torch.randn(n, n * hidden))
            self.hc_head_base = nn.Parameter(torch.zeros(n))
            self.hc_head_scale = nn.Parameter(torch.ones(1))
            nn.init.xavier_uniform_(self.hc_head_fn)
        self.gradient_checkpointing = False
        self.post_init()

    # -- weight init: delegate to the connection module, keep Qwen3 elsewhere --
    def _init_weights(self, module):
        if isinstance(module, HyperConnection):
            # The reference initialises the module itself (``_init_weights``);
            # doing it here too keeps ``post_init`` from overwriting the
            # identity anchors with the generic Linear init.
            module.reset_parameters()
            return
        # Everything else falls through to the transformers default, which also
        # (re)creates the non-persistent RoPE buffers.  Keeping `super()` here is
        # what makes ``from_pretrained`` work: the checkpoint is built on the meta
        # device, so any buffer this method does not touch stays uninitialised.
        super()._init_weights(module)

    # -- stream expand / contract (ported from input_expand / output_contract) --
    def _input_expand(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """1-stream -> n-stream: replicate (Megatron ``input_expand``)."""
        batch_size, seq_len, hidden = hidden_states.shape
        n = self.num_streams
        return (
            hidden_states.unsqueeze(2)
            .expand(batch_size, seq_len, n, hidden)
            .reshape(batch_size, seq_len, n * hidden)
        )

    def _output_contract(self, streams: torch.Tensor) -> torch.Tensor:
        """n-stream -> 1-stream (Megatron ``output_contract`` / ``learned_output_contract``)."""
        batch_size, seq_len, _ = streams.shape
        n, hidden = self.num_streams, self.config.hidden_size
        x = streams.view(batch_size, seq_len, n, hidden)
        x_ref = x[..., 0, :]
        if self.output_contract == "mean":
            # Average of the streams.  Same re-association as the mixing path:
            # mathematically the plain mean (weights 1/n sum to 1), exactly the
            # reference stream when the streams are identical.
            return x_ref + (x - x_ref.unsqueeze(-2)).mean(dim=-2)
        if self.output_contract == "sum":
            # The paper's Algorithm 1 ("sum rows of H^L").
            return x.sum(dim=-2)
        if self.output_contract == "learned":
            # Megatron ``learned_output_contract`` (DSv4), with x' the flattened
            # nC vector; its coefficient scale is head_scale (init 1.0).
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
            # The n-stream residual is a per-forward activation, not a cache entry, so an
            # incremental decode step would restart the streams from the new token and
            # silently compute a different function.  Prefill (a full pass) is correct.
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
                # chunk entry: expand 1 -> n streams (identical copies)
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
                # chunk exit: contract n -> 1 streams
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
    """Qwen3 + Manifold-Constrained Hyper-Connections, causal LM head."""

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
        # Everything else falls through to the transformers default, which also
        # (re)creates the non-persistent RoPE buffers.  Keeping `super()` here is
        # what makes ``from_pretrained`` work: the checkpoint is built on the meta
        # device, so any buffer this method does not touch stays uninitialised.
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
        pass  # already registered (module imported more than once)

# ``register_for_auto_class`` is what makes a *fresh* process able to resolve
# ``model_type = "qwen3_mhc"`` straight from a checkpoint directory: it sets
# ``_auto_class``, which makes ``save_pretrained`` write ``auto_map`` into
# ``config.json`` and copy these modules next to the weights, and it is the flag
# the ``trust_remote_code=True`` path checks.  Together with the
# ``AutoConfig.register`` / ``AutoModelForCausalLM.register`` calls above it
# covers both routes -- imported package and checkpoint-local code -- because
# verl's MegatronWorker does
# ``AutoConfig.from_pretrained(local_path, trust_remote_code=...)`` on a
# checkpoint whose directory this package is not on ``sys.path`` for.
for _cls, _auto in (
    (Qwen3MHCConfig, "AutoConfig"),
    (Qwen3MHCModel, "AutoModel"),
    (Qwen3MHCForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
