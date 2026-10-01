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
"""Qwen3 + Gated Delta Attention Residuals (GDAR).

Connection operator adapted from ``zongzhilou/transformers@shensi``
(``modeling_shensi.py``: ``ShensiAttentionResidual``), keeping that
implementation's maths line for line:

    state               = norm(prefix + delta)                 # *unweighted* RMSNorm
    gates               = gate_proj(state)                     # D -> 3D
    decay, erase, write = sigmoid(gates.reshape(..., 3, -1)).unbind(-2)
    khat                = F.normalize(k_proj(state), dim=-1)
    forgotten           = decay * prefix
    r                   = (khat * erase * forgotten).sum(-1, keepdim=True)
    updated             = forgotten - khat * r + write * delta

    values              = cat([blocks, updated.unsqueeze(-2)], dim=-2)
    reciprocal_std      = rsqrt(values.square().mean(-1) + eps)
    logits              = (values * q_proj(state).unsqueeze(-2)).sum(-1) * reciprocal_std
    routed              = softmax(logits, -1) @ values
    output              = updated + routed

``q_proj`` / ``k_proj`` are plain ``nn.Parameter(D, D)`` applied with
``F.linear``, as in that repository; ``gate_proj`` is a full-rank
``nn.Linear(D, 3D, bias=True)``; the state norm has no learnable weight.

Defaults reproduce that repository (``attn_res_gate_rank=None``,
``attn_res_gate_init="paper"``, ``attn_res_k_rank=None``).  The redo plan's two
implementation fixes are opt-in, see ``configuration_qwen3_gdar.py``:

  * ``attn_res_gate_init="identity"`` -> biases ``(b, -b, b)`` (decay open, erase
    *closed*, write open) so step 0 collapses to the DAR update;
  * ``attn_res_gate_rank=64`` / ``attn_res_k_rank=64`` -> low-rank projections
    (the full-rank gates cost ~45% of an 8B model and ``k_proj`` alone ~15%).

.. warning::
   "Three gates at bias +4 makes GDAR(0) == DAR" is wrong for the *erase* gate:
   ``erase = sigmoid(+4) = 0.98`` would erase the stream.  Erase must start
   closed.  Measured per-step deviation from the DAR update: ~2.4-3.6% at bias 4,
   ~0.1% at bias 8, ~50% with the paper init.

Ablation switches (E2-E6, see ``configuration_qwen3_gdar.py`` for the full table):
``attn_res_gate_channels`` (which gates exist), ``attn_res_gate_init`` +
``attn_res_gate_init_bias`` (where they start), ``attn_res_gate_rank`` /
``attn_res_q_rank`` / ``attn_res_k_rank`` (rank), ``attn_res_block_size`` (source
granularity, i.e. cumulative block snapshots at ``> 1`` vs per-sublayer deltas at
``1``) and ``attn_res_address`` (erase direction: ``"state"`` = the accumulated
stream).  ``Qwen3GDARConfig.gated_ar_preset()`` names the fourth cell of the
source x gate 2x2 (cumulative sources + state address + three gates).  Every switch
keeps the construction guarantee: at initialisation the connection is still
bit-exactly the DAR update.

File structure mirrors ``moonshotai/Kimi-K3:modeling_kimi_linear.py``: separate
configuration file, tensor depth state carried between layers, one output routing
pass before the final norm.
"""

from __future__ import annotations

import math
from typing import Any

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
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from .configuration_qwen3_gdar import GATE_CHANNELS, Qwen3GDARConfig

__all__ = ["Qwen3GDARConfig", "Qwen3GDARDecoderLayer", "Qwen3GDARModel", "Qwen3GDARForCausalLM"]


# ---------------------------------------------------------------------------
# The shensi connection module
# ---------------------------------------------------------------------------


class UnweightedRMSNorm(nn.Module):
    """``ShensiUnweightedRMSNorm``: RMS normalisation with no learnable weight."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


def _make_proj(in_features, out_features, rank, bias=False):
    """``Linear`` when ``rank is None`` (the repo's parameterisation) else low-rank."""
    if rank is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return nn.Sequential(
        nn.Linear(in_features, rank, bias=bias),
        nn.Linear(rank, out_features, bias=bias),
    )


def _apply_proj(x, proj):
    """Apply a plain ``Parameter`` (repo style) or a (low-rank) module, always in fp32.

    The connection computes in fp32 throughout -- it mixes normalised quantities and takes a
    square root of a source covariance -- while the model may be bf16/fp16 under inference.  So
    every projection is done here with explicit fp32 weights instead of calling the module (whose
    weights would still be bf16), and the module boundary casts the result back to the input
    dtype.  Under fp32 training ``.float()`` is a no-op, so this costs nothing there.
    """
    x = x.float()
    if isinstance(proj, torch.Tensor):
        return F.linear(x, proj.float())
    if isinstance(proj, nn.Sequential):
        layers = list(proj)
        for lin in layers:
            bias = None if lin.bias is None else lin.bias.float()
            x = F.linear(x, lin.weight.float(), bias)
        return x
    bias = None if proj.bias is None else proj.bias.float()
    return F.linear(x, proj.weight.float(), bias)


def _proj_last(proj):
    return proj if isinstance(proj, nn.Linear) else proj[-1]


def _proj_bias(proj):
    return proj.bias if isinstance(proj, nn.Linear) else proj[-1].bias


def _proj_weight(proj):
    if isinstance(proj, torch.Tensor):
        return proj
    return proj[1].weight @ proj[0].weight if isinstance(proj, nn.Sequential) else proj.weight


def _init_gate_proj(proj, init, init_bias):
    """Gate bias initialisation (E4).

    ``"paper"`` keeps the repo's ``nn.Linear`` defaults.  ``"zero"`` is the all-zero
    bias, i.e. every gate at ``sigmoid(0) = 0.5`` (the "0.5-init" row).
    ``"identity"`` sets ``(b, -b, b)`` -- decay and write open, erase closed -- and
    ``"uniform"`` sets all three to ``b``, so ``b = 0`` reproduces ``"zero"`` and
    ``b = -20`` is the "0-init" row (``sigmoid(-20) = 2.1e-9``).  The weights are
    randomised in every non-``paper`` case, since a zero weight would make the gate
    input-independent and its gradient vanish.
    """
    if init == "paper":
        return
    bias = proj.bias if isinstance(proj, nn.Linear) else proj[-1].bias
    hidden = bias.numel() // 3
    with torch.no_grad():
        if init == "zero":
            bias.zero_()
        elif init == "identity":
            target = torch.zeros_like(bias)
            for i, sign in enumerate((1.0, -1.0, 1.0)):  # decay open, erase closed, write open
                target[i * hidden : (i + 1) * hidden] = sign * init_bias
            bias.copy_(target)
        elif init == "uniform":
            bias.fill_(init_bias)
        else:
            raise ValueError(f"unknown gate init: {init}")
        linears = [proj] if isinstance(proj, nn.Linear) else list(proj)
        scale = 1.0 / math.sqrt(hidden)
        for module in linears:
            nn.init.uniform_(module.weight, -scale, scale)


def _apply_gate_channels(decay, erase, write, channels):
    """The E3 gate-structure switch: which of the three gates exist.

    ``"dew"`` (the default) returns the gates untouched and is the only path the
    reference takes.  Anything else replaces the gates the ablation removes by their
    identity constant *after* the projection, so every E3 arm carries the same
    parameters and differs only in the gate *structure* (the removed slices get zero
    gradient -- reported, not hidden, by the parameter column of the E3 table).

    * ``"d" | "e" | "w" | "de" | "dw" | "ew"`` -- only the listed gates survive;
      a removed decay is pinned to 1, a removed erase to 0, a removed write to 1.
    * ``"scalar"`` -- the write gate only, averaged over channels: one learned write
      value per token instead of one per channel (the "single scalar gate" row).
    * ``"none"`` -- ``(decay, erase, write) = (1, 0, 1)`` for every input, so the
      stream update *is* the DAR rule; this is the "no gate" row.

    Correctness hinge: every combination is the identity at initialisation.  With
    ``"deviation"`` gates the untouched ones are exactly (1, 0, 1) and the pinned
    ones are those same constants, so ``updated == prefix + delta`` bit-exactly for
    all nine values (``test_ablation_switches.py`` checks each one).
    """
    if channels == "dew":
        return decay, erase, write
    if channels == "none":
        ones = torch.ones_like(decay)
        return ones, torch.zeros_like(erase), ones
    if channels == "scalar":
        write = write.mean(dim=-1, keepdim=True)
        channels = "w"
    if "d" not in channels:
        decay = torch.ones_like(decay)
    if "e" not in channels:
        erase = torch.zeros_like(erase)
    if "w" not in channels:
        write = torch.ones_like(write)
    return decay, erase, write


def _whitening_transform(values, mode, ridge, return_inverse=False):
    """Whitening operator for the *scoring* path (retrieval still uses raw values).

    ``return_inverse`` also returns the inverse operator, which is only needed by the
    ``attn_res_read_mix="whitened"`` ablation (mix in whitened space, map the mixture
    back); the default ``"raw"`` path never touches it.

    One transform per call, pooled over tokens and sources -- per-token whitening
    would cost T x D^3.  ``diag`` is the diagonal estimator used by the whitened
    attention literature, ``full`` builds Sigma^{-1/2} from the pooled covariance.
    """
    # The transform is a *preconditioner* estimated from the sources, not a learnable
    # parameter: it is detached on purpose.  Two reasons -- (i) that is what the
    # whitening literature does (statistics are estimated, e.g. by a running EMA,
    # and the reader is a Gauss-Markov estimator for the *known* Sigma), and
    # (ii) `torch.linalg.eigh` has a NaN backward when eigenvalues are degenerate,
    # which is exactly the regime here (rank <= num_sources covariance + ridge).
    with torch.no_grad():
        S = values.reshape(-1, values.shape[-1]).float().detach()
        if mode == "diag":
            var = S.pow(2).mean(dim=0)
            w = torch.rsqrt(var + ridge)
            return (w, 1.0 / w) if return_inverse else w
        cov = (S.transpose(0, 1) @ S) / S.shape[0]
        d = cov.shape[0]
        cov = cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(cov)
        w = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)
        if not return_inverse:
            return w
        w_inv = evecs @ torch.diag(torch.sqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)
        return w, w_inv


def _softmax1(logits, dim):
    """Numerically stable Softmax_1: ``p_i = exp(z_i) / (1 + sum_j exp(z_j))``.

    The "+1" makes the usual max-subtraction non-neutral, so work in log space::

        s       = logsumexp(z)              # stable
        p_i     = exp(z_i - softplus(s))    # == exp(z_i) / (1 + exp(s))
        null    = 1 - sum_i p_i = sigmoid(-s)

    Never overflows, for either sign of the logits (an ``exp(-M)`` shift would
    overflow when every logit is very negative, which is exactly the regime
    whitened scores can reach).
    """
    s = torch.logsumexp(logits, dim=dim, keepdim=True)
    return torch.exp(logits - F.softplus(s))


def _depth_read(values, query, eps, heads=1, null=False, whiten="off", ridge=1e-3, return_scores=False,
                mix="raw"):
    # dtype-transparent: the connection works in fp32 (it mixes normalised quantities and
    # takes a square root of a source covariance), and returns the caller's dtype.
    out_dtype = values.dtype
    values, query = values.float(), query.float()
    """Depth read: softmax (optionally Softmax_1) over sources, optionally whitened.

    Args:
        values: ``(num_tokens, num_sources, hidden)`` float32, raw retrieval values.
        query:  ``(num_tokens, hidden)`` float32.
    Returns:
        ``(routed, probs)``; ``probs`` is ``(num_tokens, num_sources)`` or
        ``(num_tokens, num_sources, heads)`` when ``heads > 1``.
    """
    if mix not in ("raw", "whitened"):
        raise ValueError(f"mix must be 'raw' or 'whitened', got {mix!r}")
    num_tokens, num_sources, hidden = values.shape
    w_inv = None
    if whiten in ("diag", "full"):
        if mix == "whitened":
            w, w_inv = _whitening_transform(values, whiten, ridge, return_inverse=True)
        else:
            w = _whitening_transform(values, whiten, ridge)
        # the whitening is computed in fp32; align with the operands before use
        w = w.to(values.dtype)
        if w.dim() == 1:
            values_s = values * w
            query_s = query * w
        else:
            w = w.float()
            values_s = values @ w
            query_s = query @ w
    else:
        values_s, query_s = values, query
    # "raw" (ours): score in the whitened space, average the *actual* values (GLS/BLUE).
    # "whitened": average the whitened values and map the mixture back -- the ablation.
    mix_values = values_s if mix == "whitened" else values

    if heads > 1:
        dh = hidden // heads
        vs = values_s.view(num_tokens, num_sources, heads, dh)
        qs = query_s.view(num_tokens, heads, dh)
        recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
        logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip          # (T, N, H)
        probs = _softmax1(logits, dim=1) if null else logits.softmax(dim=1)
        routed = (probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)).sum(dim=1)
        routed = routed.reshape(num_tokens, hidden)
    else:
        recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
        logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip   # (T, N)
        probs = _softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
        routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
    if mix == "whitened" and w_inv is not None:
        # map the mixture back out of the whitened space it was averaged in
        w_inv = w_inv.to(routed.dtype)
        routed = routed * w_inv if w_inv.dim() == 1 else routed @ w_inv
    if return_scores:
        return routed.to(out_dtype), probs
    return routed.to(out_dtype)


class AttentionResidual(nn.Module):
    """``ShensiAttentionResidual``: gated delta rule + depth routing.

    ``forward(prefix, delta, blocks)`` returns ``(output, updated, gates, scores)``:
    the stream becomes ``updated`` and the sublayer input is
    ``output = updated + routed``, as in the reference module.

    ``attn_res_gate_channels`` (E3) selects *which* of the three gates exist: the
    default ``"dew"`` reaches the gates untouched and is the reference, and every
    other value pins the gates it removes to their identity constant after the
    projection (``"scalar"`` keeps the write gate only, averaged over channels;
    ``"none"`` removes the gate and leaves the DAR update).  Because the pinned
    constants *are* the identity values, every combination still satisfies
    ``GDAR(0) == DAR`` bit-exactly -- checked for all nine values in
    ``test_ablation_switches.py``.  See ``_apply_gate_channels``.

    ``attn_res_gate_init`` (E4) is the published parameterisation's initialisation:
    ``(b, -b, b)`` for ``"identity"``, all-zero biases for ``"zero"`` (gates at
    ``sigmoid(0) = 0.5``) and a common bias ``b`` for ``"uniform"`` (``b = 0`` the
    0.5-init row, ``b = -20`` the 0-init row).  ``attn_res_update="objective"`` +
    ``attn_res_gate_param="deviation"`` is the parameterisation this file's identity
    guarantee is built on; the sigmoid ones can only approach it, since
    ``sigmoid(±b)`` is never exactly ``(1, 0, 1)``.

    Two parameterisations of the update are available.

    ``attn_res_update="shensi"`` (default; reproduces the *older* reference rule, **not**
    what the shensi branch does today -- that is ``"objective"``)::

        forgotten = decay * prefix
        r         = sum_d khat_d * erase_d * forgotten_d
        updated   = forgotten - khat * r + write * delta

    ``attn_res_update="objective"`` -- the exact closed-form minimiser of

        J(h') = 1/2 ||h' - m||^2 + (lambda/2) <khat, h'>^2,
        m = decay * prefix + write * delta,   lambda = erase (per-token scalar)

    which is (Proposition 1)::

        updated = m - lambda/(1+lambda) * khat * <khat, m>

    Properties of that form (all checkable numerically, see ``test_theory.py``):

    * ``lambda = 0``  =>  ``updated = decay*prefix + write*delta``, i.e. exactly the
      DAR update when ``decay = write = 1``;
    * ``lambda -> inf`` => ``updated = (I - khat khat^T) m``, the orthogonal
      projection that clears the address direction and **leaves the orthogonal
      complement exactly invariant** (minimum-norm correction, Proposition 2);
    * the objective is strongly convex, so the minimiser is unique.

    ``attn_res_gate_param="deviation"`` replaces the reference's near-identity
    sigmoid gates (gate biases at -20, i.e. ``decay = 1 - 2e-9``) by a
    zero-initialised deviation *from* the DAR update::

        decay = exp(-softplus(r_decay) * s_decay * tau_c)   s init 0 -> decay = 1
        erase = softplus(r_erase) * s_erase                 s init 0 -> erase = 0
        write = 1 + tanh(r_write) * s_write                 s init 0 -> write = 1

    so ``GDAR(0)`` *is* DAR -- bit-exactly, checked with ``torch.equal`` rather than
    to within an epsilon (``test_theory.py`` T2) -- and the gate weights are
    initialised to **zero**, which makes the identity point input-independent and
    hence flat: the optimiser has to *learn to leave it* rather than being perturbed
    off it by a random first step.  The read has its own zero-initialised gate
    (``read_scale``), so the identity holds for the whole module output, not only for
    the ``update`` half.

    A deviation parameterisation only works if each deviation has a **non-vanishing
    gradient at the identity point**.  Writing ``gate = f(r) * s`` with ``s = 0``
    gives ``d gate/d s = f(r)``, so ``f(r)`` must not vanish where we start:
    ``softplus(0) = 0.693`` for decay/erase, and the write head is given a carrier
    bias of ``attn_res_write_carrier_bias`` (default -4, ``tanh(-4) = -0.999``).
    A naive ``write = 1 + tanh(r) * s`` has ``d/ds = tanh(0) = 0`` -- the write
    gate would then never move; that trap is covered by ``test_theory.py``.

    ``attn_res_decay_ladder=C > 1`` gives each channel its own time constant
    ``tau_c``, *learned*: initialised geometrically over ``[1, attn_res_decay_tau_max]``
    and then free to move, which is the cascade model's multi-timescale memory
    (power-law rather than single-exponential forgetting).  This is the one
    capability DAR cannot express: its read applies a single scalar weight per
    source and cannot give channels different horizons.
    """

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__()
        self.norm = UnweightedRMSNorm(config.rms_norm_eps)
        hidden = config.hidden_size
        self.hidden = hidden

        self.gate_param = getattr(config, "attn_res_gate_param", "sigmoid")
        # "reference" is an alias for "shensi": both name the *older* rule.  The
        # shensi branch of transformers implements "objective" today (see the field
        # docstring in configuration_qwen3_gdar.py).
        _rule = getattr(config, "attn_res_update", "shensi")
        self.update_rule = "shensi" if _rule == "reference" else _rule
        self.write_carrier_bias = getattr(config, "attn_res_write_carrier_bias", -4.0)
        self.read_heads = int(getattr(config, "attn_res_read_heads", 1) or 1)
        self.read_null = bool(getattr(config, "attn_res_read_null", False))
        self.read_whiten = getattr(config, "attn_res_read_whiten", "off")
        # design-ablation knobs (defaults reproduce every run so far)
        self.gate_source = getattr(config, "attn_res_gate_source", "state")
        self.decay_positivity = getattr(config, "attn_res_decay_positivity", "free")
        self.lambda_clamp = getattr(config, "attn_res_lambda_clamp", -0.5)
        self.read_mix = getattr(config, "attn_res_read_mix", "raw")
        if self.gate_source not in ("state", "prefix", "delta"):
            raise ValueError(f"attn_res_gate_source must be state|prefix|delta, got {self.gate_source!r}")
        if self.decay_positivity not in ("free", "project"):
            raise ValueError(f"attn_res_decay_positivity must be free|project, got {self.decay_positivity!r}")
        if self.read_mix not in ("raw", "whitened"):
            raise ValueError(f"attn_res_read_mix must be raw|whitened, got {self.read_mix!r}")
        self.read_ridge = float(getattr(config, "attn_res_read_ridge", 1e-3))
        self.address = getattr(config, "attn_res_address", "state")
        # E3: which of the three gates exist.  "dew" is the reference and the only
        # value that reaches the gates untouched (see ``_apply_gate_channels``).
        self.gate_channels = getattr(config, "attn_res_gate_channels", "dew")
        if self.gate_channels not in GATE_CHANNELS:
            raise ValueError(
                f"attn_res_gate_channels={self.gate_channels!r} not in {GATE_CHANNELS}"
            )
        if hidden % self.read_heads:
            raise ValueError(f"attn_res_read_heads={self.read_heads} must divide hidden_size={hidden}")

        self.gate_proj = _make_proj(hidden, 3 * hidden, getattr(config, "attn_res_gate_rank", None), bias=True)
        # q_proj / k_proj are plain parameters in the reference; the low-rank
        # variants exist because each of them costs a full D x D per sublayer.
        self.q_rank = getattr(config, "attn_res_q_rank", None)
        self.k_rank = getattr(config, "attn_res_k_rank", None)
        self.q_proj = self._make_qk(hidden, self.q_rank)
        self.k_proj = self._make_qk(hidden, self.k_rank)

        # Per-channel time constants of the decay gate, *learned*: initialised on a
        # geometric ladder over [1, attn_res_decay_tau_max] (the cascade / power-law
        # prior) and then free to move, so both the range and the shape of the
        # multi-timescale forgetting come from training.  Stored in log space, so tau
        # stays positive -- a negative tau would turn forgetting into amplification.
        # The ladder values are shared by the channels that tile onto them, so each
        # timescale is estimated from hidden_size / ladder channels of gradient.
        # attn_res_decay_ladder < 2 disables the ladder (tau == 1 for every channel).
        # The ladder itself is rebuilt from these two numbers in ``reset_parameters``
        # rather than kept in a non-persistent buffer: buffers are not checkpointed, and
        # ``from_pretrained`` builds the model on the meta device, so a buffer would be
        # uninitialised garbage there (that copy once produced NaN taus).
        self.decay_ladder = int(getattr(config, "attn_res_decay_ladder", 0) or 0)
        self.decay_tau_max = float(getattr(config, "attn_res_decay_tau_max", 100.0))
        self.decay_tau = nn.Parameter(torch.zeros(hidden)) if self.decay_ladder > 1 else None

        # The read is gated by a learned scalar that starts at 0, so at init the *whole*
        # connection -- read included -- is bit-exactly ``prefix + delta``, not just its
        # update part.  The routing then fades in as training moves this gate off zero
        # (its own gradient, <dL/drouted, routed>, does not vanish there).
        self.read_scale = nn.Parameter(torch.zeros(1))

        if self.gate_param == "deviation":
            # Zero weights + zero biases + zero deviation scales => the three gates
            # are *exactly* (1, 0, 1) at init, with no vanishing gradient: each
            # deviation direction has its own learned scale whose gradient is O(1).
            for module in ([self.gate_proj] if isinstance(self.gate_proj, nn.Linear) else list(self.gate_proj)):
                nn.init.zeros_(module.weight)
            self.decay_scale = nn.Parameter(torch.zeros(1))
            self.erase_scale = nn.Parameter(torch.zeros(1))
            self.write_scale = nn.Parameter(torch.zeros(1))

        self.init_name = getattr(config, "attn_res_gate_init", "paper")
        self.identity_bias = getattr(config, "attn_res_gate_init_bias", 4.0)
        self.reset_parameters()

    @staticmethod
    def _make_qk(hidden, rank):
        if rank is None:
            return nn.Parameter(torch.empty(hidden, hidden))
        return nn.Sequential(nn.Linear(hidden, rank, bias=False), nn.Linear(rank, hidden, bias=False))

    def _gate_head(self, state):
        """Raw gate logits, (..., 3, hidden).

        Two fp32 steps through ``_apply_proj`` rather than one step with the *composed* weight:
        that matches the reference implementation's `F.linear(F.linear(...))` order and keeps
        bf16/fp16 inference exact-as-possible (composing in bf16 would also lose precision).
        """
        raw = _apply_proj(state, self.gate_proj)
        return raw.reshape(*state.shape[:-1], 3, -1)

    def _gate_input(self, state, prefix, delta):
        """Which tensor drives the gates (design decision (a)); default is the state."""
        src = self.gate_source
        if src == "prefix" and prefix is not None:
            return self.norm(prefix.float())
        if src == "delta" and delta is not None:
            return self.norm(delta.float())
        return state  # "state", or the fallback where the requested tensor does not exist

    def _gates(self, state):
        """(decay, erase, write), each (..., hidden).

        When ``_force_identity`` is set (see ``train/guarantee.py``) the gates are
        pinned to exactly (1, 0, 1), which turns the whole module into the DAR
        update.  That slice is what makes "GDAR is never worse than DAR" checkable
        during training, not just at initialisation.

        ``attn_res_gate_channels`` (E3) is applied *last*, so it can only pin gates
        to constants; ``"dew"`` -- the default and the reference -- returns them
        untouched through ``_apply_gate_channels``.
        """
        if getattr(self, "_force_identity", False):
            ones = torch.ones_like(state)
            zeros = torch.zeros_like(state)
            return ones, zeros, ones
        raw = self._gate_head(state)
        if self.gate_param == "sigmoid":
            decay, erase, write = torch.sigmoid(raw).unbind(-2)
        else:
            r_decay, r_erase, r_write = raw.unbind(-2)
            # decay = exp(-softplus(r) * scale * tau_c): range (0, 1], exactly 1 at init, and
            # the learned per-channel tau_c gives the cascade geometry when enabled.
            tau = self.decay_tau.exp() if self.decay_tau is not None else 1.0
            # decay <= 1 iff s_decay >= 0.  "project" is the ablation-free setting that makes
            # that true by construction (and keeps identity bit-exact: 0 clamps to 0); "free"
            # is what every run so far used, and the parameter is measured to go negative.
            if self.decay_positivity == "project":
                # Straight-through projection: the forward value is clamped (so decay <= 1 holds
                # by construction and the identity stays bit-exact, since clamp(0) == 0), while the
                # backward pass is the identity -- a plain clamp() has *zero* gradient at the
                # boundary, which would freeze the scale at 0 and make the decay gate inert
                # (decay == 1 forever).
                decay_scale = self.decay_scale + (self.decay_scale.clamp(min=0.0) - self.decay_scale).detach()
            else:
                decay_scale = self.decay_scale
            decay = torch.exp(-F.softplus(r_decay) * decay_scale * tau)
            erase = F.softplus(r_erase) * self.erase_scale
            write = 1.0 + torch.tanh(r_write) * self.write_scale
        return _apply_gate_channels(decay, erase, write, self.gate_channels)

    def reset_parameters(self):
        std = 0.02
        with torch.no_grad():
            for proj in (self.q_proj, self.k_proj):
                if isinstance(proj, nn.Parameter):
                    nn.init.normal_(proj, std=std)
            if self.gate_param == "deviation":
                # order is (decay, erase, write).  Identity comes from the scales
                # being 0; the carriers must be non-zero there (softplus(0)=0.693,
                # tanh(-4)=-0.999) or their scales would have vanishing gradients.
                bias = _proj_bias(self.gate_proj)
                bias.zero_()
                bias[2 * self.hidden : 3 * self.hidden] = self.write_carrier_bias
                for module in ([self.gate_proj] if isinstance(self.gate_proj, nn.Linear) else list(self.gate_proj)):
                    nn.init.zeros_(module.weight)
                    # inner biases are meaningless here (the last one carries the gates); a
                    # random inner bias would also make the two stacks' parameter sets differ
                    if module.bias is not None and module is not _proj_last(self.gate_proj):
                        nn.init.zeros_(module.bias)
                for scale in (self.decay_scale, self.erase_scale, self.write_scale):
                    scale.zero_()
            else:
                _init_gate_proj(self.gate_proj, self.init_name, self.identity_bias)
            if self.decay_tau is not None:
                # back onto the cascade ladder, like every other init here: post_init
                # re-runs this, so the learned value must not survive a re-init
                log_tau = torch.linspace(0.0, 1.0, self.decay_ladder) * math.log(self.decay_tau_max)
                repeats = -(-self.hidden // self.decay_ladder)  # ceil
                ladder = log_tau.repeat(repeats)[: self.hidden]
                self.decay_tau.copy_(ladder.to(self.decay_tau.device, self.decay_tau.dtype))
            self.read_scale.zero_()  # the read is silent at init, see __init__

    def _state(self, prefix, delta):
        return self.norm(prefix.float() + (delta.float() if delta is not None else 0.0))

    def read(self, prefix, blocks, state=None):
        """Route over ``[blocks..., prefix]``; returns ``(prefix + routed, scores)``.

        The routed sum is *added* to the stream (nothing is written here), which is
        what makes ``read`` and ``update`` composable: a sublayer output can be
        written the moment it is produced instead of at the next call.
        """
        prefix = prefix.float()
        stream_dtype = prefix.dtype if not hasattr(prefix, 'dtype') else prefix.dtype
        if blocks is None or blocks.shape[-2] == 0:
            return prefix.to(stream_dtype), None
        if state is None:
            state = self._state(prefix, None)
        values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
        query = _apply_proj(state, self.q_proj)
        routed, scores = _depth_read(
            values,
            query,
            self.norm.eps,
            heads=self.read_heads,
            null=self.read_null,
            whiten=self.read_whiten,
            mix=self.read_mix,
            ridge=self.read_ridge,
            return_scores=True,
        )
        return prefix + self.read_scale * routed, scores

    @torch.no_grad()
    def read_premises(self, prefix, blocks):
        """Statistical premises behind the read-side claims -- measure, then claim.

        * ``key_cond``  : condition number of the source Gram matrix.  Whitening is
          *strictly* better only when the sources are not isotropic, which is exactly
          this number being >> 1 (it is 1 for an orthogonal set of equal norms).
        * ``null_mass`` : mass on the null route, ``1 - sum_i p_i``.  Softmax_1 is
          strictly better only where abstaining is the right answer, i.e. this > 0.
        * ``head_corr`` : mean |correlation| between per-head routing distributions.
          The ensemble variance identity makes multi-head strictly better only when
          the heads decorrelate.
        """
        if blocks is None or blocks.shape[-2] == 0:
            return {"n_sources": 0}
        prefix = prefix.float()
        values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
        query = _apply_proj(self._state(prefix, None), self.q_proj)
        _, probs = _depth_read(
            values, query, self.norm.eps, heads=self.read_heads, null=self.read_null,
            whiten=self.read_whiten, ridge=self.read_ridge, return_scores=True, mix=self.read_mix,
        )
        out = {"n_sources": int(values.shape[-2])}
        # key conditioning on a few tokens (N x N Gram, N <= 2 * layers)
        k = min(values.shape[0], 8)
        gram = values[:k].transpose(1, 2) @ values[:k]
        out["key_cond"] = float(torch.linalg.cond(gram).mean())
        p_sum = probs.sum(dim=1).mean()  # summed over sources, averaged over tokens/heads
        out["null_mass"] = float(1.0 - p_sum)
        if probs.dim() == 3 and probs.shape[-1] > 1:
            w = probs.reshape(-1, probs.shape[-1]).float()
            w = w - w.mean(dim=0, keepdim=True)
            std = w.std(dim=0).clamp_min(1e-8)
            corr = (w / std).transpose(0, 1) @ (w / std) / w.shape[0]
            off = corr - torch.diag(torch.diagonal(corr))
            out["head_corr"] = float(off.abs().sum() / (corr.shape[0] * (corr.shape[0] - 1)))
        return out

    def update(self, prefix, delta, state=None):
        """One gated delta-rule step over the stream; returns ``(updated, gates)``."""
        prefix_f = prefix.float()
        up_dtype = prefix.dtype
        delta_f = delta.float() if delta is not None else None
        if state is None:
            state = self._state(prefix_f, delta_f)

        # One 3H projection of the state drives all three gates of the delta rule.
        decay, erase, write = self._gates(self._gate_input(state, prefix_f, delta_f))

        # Decay the prefix, then either project out the erased direction and write
        # (reference) or solve the depth-memory objective in closed form.
        # Address selection.  "state" reproduces the reference (address from the stream);
        # "delta" derives it from the content being written, which is the DeltaNet-consistent
        # choice: the erase direction is the key of the item being written, so the write is an
        # exact rank-1 overwrite.  "novelty" instead orthogonalises against the retained
        # stream (max written signal-to-interference), i.e. it optimises for an *additive*
        # write and therefore conflicts with the overwrite semantics: measured at initialisation
        # it is not better than "delta", and with a random k_proj the three choices are
        # statistically indistinguishable, so this one can only be settled by an A/B run.
        if self.address == "novelty":
            base = decay * prefix_f
            src = delta_f if delta_f is not None else state
            denom = base.square().sum(dim=-1, keepdim=True).clamp_min(self.norm.eps)
            address_src = src - (src * base).sum(dim=-1, keepdim=True) / denom * base
        elif self.address == "delta" and delta_f is not None:
            address_src = delta_f
        else:
            address_src = state
        khat = F.normalize(_apply_proj(address_src, self.k_proj), dim=-1)
        delta_term = delta_f if delta_f is not None else 0.0
        if self.update_rule == "objective":
            m = decay * prefix_f + write * delta_term
            # J's Hessian is I + lambda * khat khat^T, strictly convex iff lambda > -1,
            # with a pole at lambda = -1.  Nothing in the parameterisation keeps the
            # learned erase positive, so clamp into the feasible region (margin 0.5):
            # outside it the closed form is no longer a minimiser and lambda -> -1
            # would blow up.  -0.5 keeps a usable "anti-erase" band.
            lam = erase.mean(dim=-1, keepdim=True)
            if self.lambda_clamp is not None:
                lam = lam.clamp(min=float(self.lambda_clamp))
            updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)
        else:
            forgotten = decay * prefix_f
            r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
            updated = forgotten - khat * r + write * delta_term
        return updated.to(up_dtype), (decay, erase, write)

    def forward(self, prefix, delta, blocks, output_norm_weight=None, num_blocks=0):
        """Reference-style composed call: read, update, then add the two."""
        out_dtype = prefix.dtype
        prefix_f = prefix.float()
        delta_f = delta.float() if delta is not None else None
        # Wiring note: the read's candidate set is [blocks..., prefix] -- the depth memory
        # *before* this sublayer's write.  That is the AR/DAR convention (a sublayer's input is
        # routed over what was written before it) and it keeps the mixture free of the value it
        # is about to be added to.  The `transformers@shensi` branch arranges its module
        # differently (its write happens before the read, so `updated` is a candidate); the
        # difference is structural, not a switch, and is documented in SHENSI_UPSTREAM_VERIFY.md
        # section 6 with a bit-level cross-check.
        state = self._state(prefix_f, delta_f)
        routed, scores = self.read(prefix_f, blocks, state=state)
        updated, gates = self.update(prefix_f, delta_f, state=state)
        output = updated + (routed - prefix_f)  # routed already contains the stream
        if output_norm_weight is not None:
            reciprocal_std = torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self.norm.eps)
            output = output * reciprocal_std * output_norm_weight.float()
        return output.to(out_dtype), updated.to(out_dtype), gates, scores


class DepthRead(nn.Module):
    """Read-only depth routing used for the final (output) pass.

    No gates and no update: the stream is left untouched and the routed sum is
    added to it, ``output = prefix + softmax(logits) @ [blocks, prefix]``.  Keeping
    this separate matters for two reasons: the output step has no new content to
    write, so the gate/erase machinery would be dead weight (literally dead
    parameters), and the reference's output routing is a pure read as well.
    """

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__()
        self.q_proj = AttentionResidual._make_qk(config.hidden_size, getattr(config, "attn_res_q_rank", None))
        if isinstance(self.q_proj, nn.Parameter):
            nn.init.normal_(self.q_proj, std=0.02)
        self.eps = config.rms_norm_eps
        self.read_heads = int(getattr(config, "attn_res_read_heads", 1) or 1)
        self.read_null = bool(getattr(config, "attn_res_read_null", False))
        self.read_whiten = getattr(config, "attn_res_read_whiten", "off")
        self.read_mix = getattr(config, "attn_res_read_mix", "raw")
        self.read_ridge = float(getattr(config, "attn_res_read_ridge", 1e-3))
        self.read_scale = nn.Parameter(torch.zeros(1))  # silent at init, see AttentionResidual

    def forward(self, prefix_flat, blocks):
        if blocks is None or blocks.shape[-2] == 0:
            return prefix_flat
        values = torch.cat([blocks.float(), prefix_flat.float().unsqueeze(-2)], dim=-2)
        query = _apply_proj(prefix_flat.float(), self.q_proj)
        routed = _depth_read(
            values,
            query,
            self.eps,
            heads=self.read_heads,
            null=self.read_null,
            whiten=self.read_whiten,
            mix=self.read_mix,
            ridge=self.read_ridge,
        )
        return (prefix_flat.float() + self.read_scale * routed).to(prefix_flat.dtype)


def record_router_stats(stats, layer_idx, sublayer, scores=None, gates=None, n_sources=None, premises=None):
    """Sharpness / entropy / gate means for the routing-collapse analysis."""
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": sublayer}
    with torch.no_grad():
        if scores is not None:
            w = scores.detach().float()
            entry["n_sources"] = int(w.shape[-1])
            entry["sharpness"] = float(w.max(dim=-1).values.mean())
            entry["entropy"] = float(-(w * (w + 1e-8).log()).sum(dim=-1).mean())
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
        if gates is not None:
            decay, erase, write = (g.detach().float().mean() for g in gates)
            entry["gate_decay"] = float(decay)
            entry["gate_erase"] = float(erase)
            entry["gate_write"] = float(write)
        if premises:
            entry.update(premises)
    stats.append(entry)


class Qwen3GDARDecoderLayer(nn.Module):
    """Qwen3 decoder layer with Gated Delta Attention Residuals."""

    def __init__(self, config: Qwen3GDARConfig, layer_idx: int):
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
            self.self_attention_attn_res = AttentionResidual(config)
            self.mlp_attn_res = AttentionResidual(config)

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

    def _append_source(self, delta_residual, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1])
        return torch.cat([delta_residual, flat.unsqueeze(1)], dim=1)

    @staticmethod
    def _num_blocks(delta_residual):
        return 0 if delta_residual is None else delta_residual.shape[1]

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
        """Depth-routed layer.

        The connection module is used in two steps: ``read`` routes over the delta
        sources (nothing written), and ``update`` is the gated delta rule.  Doing the
        write immediately after each sublayer is what adapts the reference module to
        a plain residual stream -- the reference defers it to the next call because
        its stream is carried by hyper-connections, which would silently drop the
        last layer's MLP output here.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states.view(-1, hidden_size)
        per_sublayer_sources = self.attn_res_block_size == 1

        if not per_sublayer_sources and self.layer_idx % self.attn_res_block_size == 0:
            # close the previous block: snapshot the stream (the embedding seeds the list)
            delta_residual = self._append_source(delta_residual, prefix_sum)
        elif per_sublayer_sources and (delta_residual is None or delta_residual.shape[1] == 0):
            delta_residual = self._append_source(delta_residual, prefix_sum)

        # ---- attention sublayer: read -> attention -> write ----
        routed, scores = self.self_attention_attn_res.read(prefix_sum, delta_residual)
        # the connection computes in fp32; hand the stream back in the model dtype
        routed = routed.to(hidden_states.dtype)
        attn_out = self._attention(
            self.input_layernorm(routed.view(batch_size, seq_len, hidden_size)),
            attention_mask, position_ids, past_key_values, use_cache, position_embeddings,
        )
        prefix_sum, gates = self.self_attention_attn_res.update(prefix_sum, attn_out.reshape(-1, hidden_size))
        record_router_stats(
            attn_res_stats, self.layer_idx, "attn", scores=scores, gates=gates,
            premises=self.self_attention_attn_res.read_premises(prefix_sum, delta_residual)
            if attn_res_stats is not None else None,
        )
        if per_sublayer_sources:
            delta_residual = self._append_source(delta_residual, attn_out)

        # ---- MLP sublayer: read -> mlp -> write ----
        routed, scores = self.mlp_attn_res.read(prefix_sum, delta_residual)
        # the connection computes in fp32; hand the stream back in the model dtype
        routed = routed.to(hidden_states.dtype)
        # the connection computes in fp32; hand the stream back in the model dtype
        routed = routed.to(hidden_states.dtype)
        mlp_out = self.mlp(self.post_attention_layernorm(routed.view(batch_size, seq_len, hidden_size)))
        prefix_sum, gates = self.mlp_attn_res.update(prefix_sum, mlp_out.reshape(-1, hidden_size))
        record_router_stats(
            attn_res_stats, self.layer_idx, "mlp", scores=scores, gates=gates,
            premises=self.mlp_attn_res.read_premises(prefix_sum, delta_residual)
            if attn_res_stats is not None else None,
        )
        if per_sublayer_sources:
            delta_residual = self._append_source(delta_residual, mlp_out)

        return prefix_sum.view(batch_size, seq_len, hidden_size), delta_residual


class Qwen3GDARModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Gated Delta Attention Residuals."""

    config_class = Qwen3GDARConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3GDARDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.use_attn_residuals = config.attn_res_block_size is not None
        self.output_attn_res = self.use_attn_residuals and config.attn_res_output_route
        if self.output_attn_res:
            self.output_attn_res_module = DepthRead(config)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, AttentionResidual):
            # identity / zero-gate anchors; must run *after* any generic pass (see post_init)
            module.reset_parameters()
            return
        super()._init_weights(module)

    def post_init(self):
        super().post_init()
        # HF's generic _init_weights zeroes Linear biases, which would undo the
        # identity / zero gate initialisation of the connection modules.
        for module in self.modules():
            if isinstance(module, AttentionResidual):
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
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = self.output_attn_res_module(
                hidden_states.view(-1, hidden_size), delta_residual
            ).view(batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3GDARForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 + Gated Delta Attention Residuals, causal LM head."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3GDARConfig

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__(config)
        self.model = Qwen3GDARModel(config)
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
    (AutoConfig.register, ("qwen3_gdar", Qwen3GDARConfig)),
    (AutoModel.register, (Qwen3GDARConfig, Qwen3GDARModel)),
    (AutoModelForCausalLM.register, (Qwen3GDARConfig, Qwen3GDARForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass  # already registered (module imported more than once)

# ``register_for_auto_class`` is what makes a *fresh* process able to resolve
# ``model_type = "qwen3_gdar"`` straight from a checkpoint directory: it sets
# ``_auto_class``, which makes ``save_pretrained`` write ``auto_map`` into
# ``config.json`` and copy these modules next to the weights, and it is the flag
# the ``trust_remote_code=True`` path checks.  Together with the
# ``AutoConfig.register`` / ``AutoModelForCausalLM.register`` calls above it
# covers both routes -- imported package and checkpoint-local code -- because
# verl's MegatronWorker does
# ``AutoConfig.from_pretrained(local_path, trust_remote_code=...)`` on a
# checkpoint whose directory this package is not on ``sys.path`` for.
for _cls, _auto in (
    (Qwen3GDARConfig, "AutoConfig"),
    (Qwen3GDARModel, "AutoModel"),
    (Qwen3GDARForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
