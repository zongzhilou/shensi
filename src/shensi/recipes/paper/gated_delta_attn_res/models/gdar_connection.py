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
"""Gated Delta Attention Residuals (GDAR) connection module -- Megatron-Core port.

Line-for-line port of ``models/modeling_qwen3_gdar.py`` (which itself mirrors
``zongzhilou/transformers@shensi:ShensiAttentionResidual``) with Megatron-Core
tensor conventions (``[s, b, h]``) and Megatron initialisation instead of
HuggingFace ``post_init``.  The maths is unchanged::

    state               = norm(prefix + delta)                 # *unweighted* RMSNorm
    gates               = gate_proj(state)                     # D -> 3D
    decay, erase, write = sigmoid(gates.reshape(..., 3, -1)).unbind(-2)   # "sigmoid"
    # or, for gate_param="deviation" (exact identity at init):
    decay = exp(-softplus(r_decay) * decay_scale * tau_c)      # s=0 -> decay == 1
    erase = softplus(r_erase) * erase_scale                    # s=0 -> erase == 0
    write = 1 + tanh(r_write) * write_scale                    # s=0 -> write == 1
    khat                = F.normalize(k_proj(address_src), dim=-1)
    forgotten           = decay * prefix
    r                   = (khat * erase * forgotten).sum(-1, keepdim=True)
    updated             = forgotten - khat * r + write * delta

Differences from the HF file, all of them deliberate and mechanical:

* Megatron modules (``nn.Module``) instead of HF ``PreTrainedModel`` plumbing;
  ``reset_parameters()`` is called from the owning layer, not ``post_init``.
* The first matrix of a low-rank projection is created with ``bias=False``: in the
  HF file its bias is created, never used (``_gate_head`` uses the *composed*
  weight ``W2 @ W1`` plus the last bias only) and never updated, so it is
  identically zero for the whole run.  Dropping it removes a dead parameter
  without changing a single arithmetic operation.
* ``write_dropout_p``: the HF module has no dropout.  In Megatron the connection
  *replaces* ``bias_dropout_add`` in the residual branch, so, to keep the layer a
  drop-in replacement of a Megatron layer (and to keep ``hidden_dropout``
  semantics), the sublayer output may be dropped out before the write.  It is
  off by default and enabled by the layer with ``config.hidden_dropout``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GDAR_UPDATE_RULES",
    "AttentionResidual",
    "DepthRead",
    "GdarConfig",
    "UnweightedRMSNorm",
]


# ---------------------------------------------------------------------------
# helpers (identical maths to models/modeling_qwen3_gdar.py)
# ---------------------------------------------------------------------------


class UnweightedRMSNorm(nn.Module):
    """``ShensiUnweightedRMSNorm``: RMS normalisation with no learnable weight."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class LowRankLinear(nn.Module):
    """``Sequential(Linear(d_in, r, bias=False), Linear(r, d_out, bias=True))``.

    ``composed_weight()`` reproduces what the HF file feeds to ``F.linear`` for a
    low-rank projection (``proj[1].weight @ proj[0].weight``), including its
    choice of bias (the *last* one).
    """

    def __init__(self, d_in: int, d_out: int, rank: int, down_bias: bool = False, out_bias: bool = True):
        super().__init__()
        # The HF reference builds ``Sequential(Linear(d_in, r, bias=B), Linear(r, d_out, bias=B))``
        # with one flag for both halves, so mirror it exactly: ``down_bias``/``out_bias`` are
        # both True for the gate projection and both False for the query/address projections.
        self.down = nn.Linear(d_in, rank, bias=down_bias)
        self.up = nn.Linear(rank, d_out, bias=out_bias)

    def composed_weight(self) -> torch.Tensor:
        return self.up.weight @ self.down.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # == F.linear(x, composed_weight(), up.bias) up to matmul association;
        # the reference applies the composed weight, this is the equivalent
        # factored form (identical semantics, two small GEMMs instead of a
        # per-step H x 3H matmul recomputation).
        return self.up(self.down(x))


def _make_proj(d_in: int, d_out: int, rank: int | None, bias: bool = False) -> nn.Module:
    """``Linear`` when ``rank is None`` (the reference's parameterisation) else low-rank."""
    if rank is None:
        return nn.Linear(d_in, d_out, bias=bias)
    return LowRankLinear(d_in, d_out, rank, down_bias=bias, out_bias=bias)


def _is_parameter(proj) -> bool:
    return isinstance(proj, nn.Parameter)


def _proj_bias(proj):
    if isinstance(proj, nn.Linear):
        return proj.bias
    if isinstance(proj, LowRankLinear):
        return proj.bias
    return proj[-1].bias


def _linears(proj):
    if isinstance(proj, nn.Linear):
        return [proj]
    if isinstance(proj, LowRankLinear):
        return [proj.down, proj.up]
    return list(proj)


def _apply_proj(x: torch.Tensor, proj) -> torch.Tensor:
    """``F.linear(x, W[, b])`` / ``proj(x)`` with ``x`` cast to the projection's dtype.

    Two mechanical notes:

    * the cast: the reference file keeps every intermediate in fp32 and relies on
      ``autocast`` when the model runs in bf16 (``F.linear`` with an fp32 input and
      a bf16 weight is only legal under autocast, which casts the input down and
      accumulates the GEMM in fp32).  Megatron runs without autocast, so the cast
      is written out explicitly; the arithmetic is autocast's.
    * the low-rank pair applies ``W2(W1 x) + b2`` instead of the reference's
      ``(W2 W1) x + b2``: identical semantics up to matmul association, and it
      avoids rebuilding an H x 3H matrix on every forward.  (The reference's
      composed form also silently drops ``b1``, which stays exactly zero for the
      whole run because nothing depends on it.)
    """
    if isinstance(proj, nn.Parameter):
        return F.linear(x.to(proj.dtype), proj)
    if isinstance(proj, LowRankLinear):
        return proj(x.to(proj.down.weight.dtype))
    return proj(x.to(proj.weight.dtype))


def _init_gate_proj(gate_proj, init: str, identity_bias: float) -> None:
    """Gate bias initialisation.  ``"paper"`` keeps the repository's defaults."""
    if init == "paper":
        return
    bias = _proj_bias(gate_proj)
    hidden = bias.numel() // 3
    with torch.no_grad():
        if init == "zero":
            bias.zero_()
        elif init == "identity":
            target = torch.zeros_like(bias)
            for i, sign in enumerate((1.0, -1.0, 1.0)):  # decay open, erase closed, write open
                target[i * hidden : (i + 1) * hidden] = sign * identity_bias
            bias.copy_(target)
        elif init == "uniform":
            # E4's "0-init" row: all three heads at the same bias, so with the *sigmoid* gate the
            # gates start at sigmoid(b) ~ 0 (b = -20 -> 2.1e-9) -- the failure mode the old draft
            # blamed on the gates.  Mirrors `models/modeling_qwen3_gdar.py::_init_gate_proj`.
            bias.fill_(identity_bias)
        else:
            raise ValueError(f"unknown gate init: {init}")
        scale = 1.0 / math.sqrt(hidden)
        for module in _linears(gate_proj):
            nn.init.uniform_(module.weight, -scale, scale)


def _whitening_transform(
    values: torch.Tensor, mode: str, ridge: float, return_inverse: bool = False
) -> torch.Tensor:
    # return_inverse is only used by the read_mix="whitened" ablation (see
    # GDAR_ABLATION_DESIGN.md); the default path never asks for it.
    """Whitening operator for the *scoring* path (retrieval still uses raw values)."""
    # Detached on purpose: the transform is a preconditioner estimated from the
    # sources, not a learnable parameter (and eigh has a NaN backward on the
    # degenerate spectra this produces).  See the HF file for the full rationale.
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


def _softmax1(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable Softmax_1: ``p_i = exp(z_i) / (1 + sum_j exp(z_j))``."""
    s = torch.logsumexp(logits, dim=dim, keepdim=True)
    return torch.exp(logits - F.softplus(s))


def _depth_read(
    values: torch.Tensor,
    query: torch.Tensor,
    eps: float,
    heads: int = 1,
    null: bool = False,
    whiten: str = "off",
    ridge: float = 1e-3,
    return_scores: bool = False,
    mix: str = "raw",
):
    """Depth read: softmax (optionally Softmax_1) over sources, optionally whitened."""
    if mix not in ("raw", "whitened"):
        raise ValueError(f"mix must be 'raw' or 'whitened', got {mix!r}")
    num_tokens, num_sources, hidden = values.shape
    w_inv = None
    if whiten in ("diag", "full"):
        # The whitening operator is estimated (and inverted) in fp32, and `query @ w` is a
        # matmul: bf16 operands against an fp32 operator raise instead of promoting.  Work in
        # fp32 for the whole whitened path -- which is also what the HF reference does -- and
        # let the caller cast the result back (it already does).
        values, query = values.float(), query.float()
        if mix == "whitened":
            w, w_inv = _whitening_transform(values, whiten, ridge, return_inverse=True)
        else:
            w = _whitening_transform(values, whiten, ridge)
        w = w.to(values.dtype)
        if w.dim() == 1:
            values_s = values * w
            query_s = query * w
        else:
            values_s = values @ w
            query_s = query @ w
    else:
        values_s, query_s = values, query
    # "raw" (ours): score whitened, average the actual values.  "whitened": average the
    # whitened values and map the mixture back -- the ablation of that argument.
    mix_values = values_s if mix == "whitened" else values

    if heads > 1:
        dh = hidden // heads
        vs = values_s.view(num_tokens, num_sources, heads, dh)
        qs = query_s.view(num_tokens, heads, dh)
        recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
        logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip  # (T, N, H)
        probs = _softmax1(logits, dim=1) if null else logits.softmax(dim=1)
        routed = (probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)).sum(dim=1)
        routed = routed.reshape(num_tokens, hidden)
    else:
        recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
        logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip  # (T, N)
        probs = _softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
        routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
    if mix == "whitened" and w_inv is not None:
        routed = routed * w_inv if w_inv.dim() == 1 else routed @ w_inv
    if return_scores:
        return routed, probs
    return routed


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------


@dataclass
class GdarConfig:
    """Knobs of the connection, mirroring ``Qwen3GDARConfig`` field by field."""

    # ``None`` disables the connection entirely (plain Qwen3 behaviour)
    block_size: int | None = None
    output_route: bool = True

    # gates
    gate_rank: int | None = None
    gate_init: str = "paper"
    gate_init_bias: float = 4.0
    gate_param: str = "sigmoid"  # "sigmoid" (reference) | "deviation" (exact identity)
    write_carrier_bias: float = -4.0
    decay_ladder: int = 0
    decay_tau_max: float = 100.0
    # "shensi" (alias "reference") = the *older* reference rule; "objective" = the
    # closed-form minimiser, which is what the transformers@shensi branch implements today.
    update: str = "shensi"  # "shensi"/"reference" | "objective"

    # read side
    read_heads: int = 1
    read_null: bool = False
    read_whiten: str = "off"
    read_ridge: float = 1.0e-3
    address: str = "state"  # "state" | "delta" | "novelty"
    # --- design-ablation knobs (GDAR's own design decisions) -------------------
    # "state" (default): the gates see the normalised (prefix + delta); "prefix"/"delta"
    # give them only one of the two (falling back to the state where it does not exist).
    gate_source: str = "state"  # "state" | "prefix" | "delta"
    # decay <= 1 iff the decay scale is >= 0.  The scale is unconstrained and *measured*
    # to go negative in training (24/24 modules in one E12 run); "project" clamps it at 0
    # in the forward (identity is preserved bit-exactly) so the bound holds by construction.
    decay_positivity: str = "free"  # "free" | "project"
    # Lower clamp on lambda = mean_c(erase) in the objective update; None removes it.
    # -0.5 is our margin inside the strictly convex region (lambda > -1).
    lambda_clamp: float | None = -0.5
    # Space the read averages in: "raw" (ours) or "whitened" (the ablation of the
    # GLS/BLUE argument).  Only meaningful when read_whiten != "off".
    read_mix: str = "raw"

    # projections
    q_rank: int | None = None
    k_rank: int | None = None

    # misc
    init_std: float = 0.02
    #: dropout applied to the sublayer output before the write (the connection
    #: replaces ``bias_dropout_add``).  ``None`` = follow ``config.hidden_dropout``,
    #: which is what keeps the layer a drop-in replacement of a Megatron layer.
    residual_dropout: float | None = None

    def validated(self) -> GdarConfig:
        if self.gate_param not in ("sigmoid", "deviation"):
            raise ValueError(f"gate_param must be 'sigmoid' or 'deviation', got {self.gate_param!r}")
        if self.update == "reference":  # alias for the older rule
            self.update = "shensi"
        if self.update not in ("shensi", "objective"):
            raise ValueError(f"update must be 'shensi' or 'objective', got {self.update!r}")
        if self.address not in ("state", "delta", "novelty"):
            raise ValueError(f"address must be 'state'/'delta'/'novelty', got {self.address!r}")
        if self.read_whiten not in ("off", "diag", "full"):
            raise ValueError(f"read_whiten must be 'off'/'diag'/'full', got {self.read_whiten!r}")
        if self.gate_source not in ("state", "prefix", "delta"):
            raise ValueError(f"gate_source must be 'state'/'prefix'/'delta', got {self.gate_source!r}")
        if self.decay_positivity not in ("free", "project"):
            raise ValueError(f"decay_positivity must be 'free'/'project', got {self.decay_positivity!r}")
        if self.read_mix not in ("raw", "whitened"):
            raise ValueError(f"read_mix must be 'raw'/'whitened', got {self.read_mix!r}")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError(f"block_size must be >= 1 or None, got {self.block_size}")
        return self


# ---------------------------------------------------------------------------
# the connection
# ---------------------------------------------------------------------------


class AttentionResidual(nn.Module):
    """``ShensiAttentionResidual``: gated delta rule + depth routing.

    ``read(prefix, blocks)``     -> ``prefix + read_scale * routed``   (nothing written)
    ``update(prefix, delta)``    -> ``(updated, (decay, erase, write))``

    Splitting the two is what adapts the reference module to a plain residual
    stream: the sublayer output is written the moment it is produced instead of
    being deferred to the next call (which would silently drop the last layer's
    MLP output).
    """

    def __init__(self, hidden: int, cfg: GdarConfig, eps: float = 1.0e-6):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.eps = eps
        self.norm = UnweightedRMSNorm(eps)

        if hidden % cfg.read_heads:
            raise ValueError(f"read_heads={cfg.read_heads} must divide hidden_size={hidden}")

        self.gate_proj = _make_proj(hidden, 3 * hidden, cfg.gate_rank, bias=True)
        self.q_proj = self._make_qk(hidden, cfg.q_rank)
        self.k_proj = self._make_qk(hidden, cfg.k_rank)

        ladder = int(cfg.decay_ladder or 0)
        if ladder > 1:
            log_tau = torch.linspace(0.0, 1.0, ladder) * math.log(float(cfg.decay_tau_max))
            repeats = -(-hidden // ladder)  # ceil
            self.register_buffer("decay_tau_init", log_tau.repeat(repeats)[:hidden], persistent=False)
            self.decay_tau = nn.Parameter(self.decay_tau_init.clone())
        else:
            self.decay_tau = None

        # The read is gated by a learned scalar that starts at 0, so at init the
        # *whole* connection (read included) is bit-exactly ``prefix + delta``.
        self.read_scale = nn.Parameter(torch.zeros(1))

        if cfg.gate_param == "deviation":
            # Zero weights + zero biases + zero deviation scales => the three gates
            # are *exactly* (1, 0, 1) at init, with no vanishing gradient on the
            # three scales.
            self.decay_scale = nn.Parameter(torch.zeros(1))
            self.erase_scale = nn.Parameter(torch.zeros(1))
            self.write_scale = nn.Parameter(torch.zeros(1))
        else:
            self.decay_scale = None
            self.erase_scale = None
            self.write_scale = None

        self.reset_parameters()

    @staticmethod
    def _make_qk(hidden: int, rank: int | None):
        """``nn.Parameter(hidden, hidden)`` (reference) or a low-rank pair."""
        if rank is None:
            return nn.Parameter(torch.empty(hidden, hidden))
        return LowRankLinear(hidden, hidden, rank)

    # -- init ---------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Mirrors ``AttentionResidual.reset_parameters`` / HF ``post_init`` order.

        HF first runs the generic ``_init_weights`` (Linear: N(0, initializer_range),
        bias zeroed; Embedding: N(0, initializer_range)) over every submodule and
        only then re-runs ``reset_parameters`` for the connection modules.  Both
        steps are reproduced here, in that order, inside the caller's RNG fork.
        """
        std = float(self.cfg.init_std)
        with torch.no_grad():
            for module in self._linear_modules():
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.zero_()
            for proj in (self.q_proj, self.k_proj):
                if isinstance(proj, nn.Parameter):
                    nn.init.normal_(proj, std=std)

            # -- reset_parameters() proper --
            if self.cfg.gate_param == "deviation":
                bias = _proj_bias(self.gate_proj)
                bias.zero_()
                bias[2 * self.hidden : 3 * self.hidden] = self.cfg.write_carrier_bias
                for module in _linears(self.gate_proj):
                    nn.init.zeros_(module.weight)
                for scale in (self.decay_scale, self.erase_scale, self.write_scale):
                    scale.zero_()
            else:
                _init_gate_proj(self.gate_proj, self.cfg.gate_init, self.cfg.gate_init_bias)
            if self.decay_tau is not None:
                self.decay_tau.copy_(self.decay_tau_init)
            self.read_scale.zero_()

    def _linear_modules(self):
        modules = list(_linears(self.gate_proj))
        for proj in (self.q_proj, self.k_proj):
            if not isinstance(proj, nn.Parameter):
                modules.extend(_linears(proj))
        return modules

    # -- forward pieces -----------------------------------------------------

    def _gate_head(self, state: torch.Tensor) -> torch.Tensor:
        """Raw gate logits, (..., 3, hidden)."""
        raw = _apply_proj(state, self.gate_proj)
        return raw.reshape(*state.shape[:-1], 3, -1)

    def _gate_input(self, state: torch.Tensor, prefix, delta) -> torch.Tensor:
        # design decision (a): which tensor drives the gates; default is the state
        src = self.cfg.gate_source
        if src == "prefix" and prefix is not None:
            return self.norm(prefix.float())
        if src == "delta" and delta is not None:
            return self.norm(delta.float())
        return state

    def _gates(self, state: torch.Tensor):
        """(decay, erase, write), each (..., hidden)."""
        raw = self._gate_head(state)
        if self.cfg.gate_param == "sigmoid":
            return torch.sigmoid(raw).unbind(-2)

        r_decay, r_erase, r_write = raw.unbind(-2)
        tau = self.decay_tau.exp() if self.decay_tau is not None else 1.0
        # decay <= 1 iff the scale is >= 0: "project" enforces that by construction
        if self.cfg.decay_positivity == "project":
            # straight-through: forward clamped (decay <= 1 by construction, identity bit-exact),
            # backward identity -- a plain clamp has zero gradient at the boundary and would
            # freeze the scale at 0, making the decay gate inert.
            decay_scale = self.decay_scale + (self.decay_scale.clamp(min=0.0) - self.decay_scale).detach()
        else:
            decay_scale = self.decay_scale
        decay = torch.exp(-F.softplus(r_decay) * decay_scale * tau)
        erase = F.softplus(r_erase) * self.erase_scale
        write = 1.0 + torch.tanh(r_write) * self.write_scale
        return decay, erase, write

    def _state(self, prefix: torch.Tensor, delta: torch.Tensor | None) -> torch.Tensor:
        return self.norm(prefix.float() + (delta.float() if delta is not None else 0.0))

    def read(self, prefix: torch.Tensor, blocks: torch.Tensor | None, state: torch.Tensor | None = None):
        """Route over ``[blocks..., prefix]``; returns ``(prefix + routed, scores)``."""
        prefix = prefix.float()
        if blocks is None or blocks.shape[-2] == 0:
            return prefix, None
        if state is None:
            state = self._state(prefix, None)
        values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
        query = _apply_proj(state, self.q_proj)
        routed, scores = _depth_read(
            values,
            query,
            self.eps,
            heads=self.cfg.read_heads,
            null=self.cfg.read_null,
            whiten=self.cfg.read_whiten,
            ridge=self.cfg.read_ridge,
            return_scores=True,
            mix=self.cfg.read_mix,
        )
        return prefix + self.read_scale * routed, scores

    def update(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        state: torch.Tensor | None = None,
    ):
        """One gated delta-rule step over the stream; returns ``(updated, gates)``."""
        prefix_f = prefix.float()
        delta_f = delta.float() if delta is not None else None
        if state is None:
            state = self._state(prefix_f, delta_f)

        decay, erase, write = self._gates(self._gate_input(state, prefix_f, delta_f))

        if self.cfg.address == "novelty":
            base = decay * prefix_f
            src = delta_f if delta_f is not None else state
            denom = base.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
            address_src = src - (src * base).sum(dim=-1, keepdim=True) / denom * base
        elif self.cfg.address == "delta" and delta_f is not None:
            address_src = delta_f
        else:
            address_src = state
        k_proj_state = _apply_proj(address_src, self.k_proj)
        khat = F.normalize(k_proj_state, dim=-1)
        delta_term = delta_f if delta_f is not None else 0.0
        if self.cfg.update == "objective":
            m = decay * prefix_f + write * delta_term
            lam = erase.mean(dim=-1, keepdim=True)
            if self.cfg.lambda_clamp is not None:
                lam = lam.clamp(min=float(self.cfg.lambda_clamp))
            updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)
        else:
            forgotten = decay * prefix_f
            r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
            updated = forgotten - khat * r + write * delta_term
        return updated, (decay, erase, write)


class DepthRead(nn.Module):
    """Read-only depth routing used for the final (output) pass."""

    def __init__(self, hidden: int, cfg: GdarConfig, eps: float = 1.0e-6):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.eps = eps
        self.q_proj = AttentionResidual._make_qk(hidden, cfg.q_rank)
        self.read_scale = nn.Parameter(torch.zeros(1))  # silent at init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = float(self.cfg.init_std)
        with torch.no_grad():
            if isinstance(self.q_proj, nn.Parameter):
                nn.init.normal_(self.q_proj, std=std)
            else:
                for module in (self.q_proj.down, self.q_proj.up):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.zero_()
            self.read_scale.zero_()

    def forward(self, prefix_flat: torch.Tensor, blocks: torch.Tensor | None) -> torch.Tensor:
        if blocks is None or blocks.shape[-2] == 0:
            return prefix_flat
        prefix_f = prefix_flat.float()
        values = torch.cat([blocks.float(), prefix_f.unsqueeze(-2)], dim=-2)
        query = _apply_proj(prefix_f, self.q_proj)
        routed = _depth_read(
            values,
            query,
            self.eps,
            heads=self.cfg.read_heads,
            null=self.cfg.read_null,
            whiten=self.cfg.read_whiten,
            mix=self.cfg.read_mix,
            ridge=self.cfg.read_ridge,
        )
        return (prefix_f + self.read_scale * routed).to(prefix_flat.dtype)


GDAR_UPDATE_RULES = ("shensi", "objective")
