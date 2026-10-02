"""GDAR 连接算子：门控 decay/erase/write、目标函数闭式更新、白化多头读。"""

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


class UnweightedRMSNorm(nn.Module):

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class LowRankLinear(nn.Module):

    def __init__(
        self, d_in: int, d_out: int, rank: int, down_bias: bool = False, out_bias: bool = True
    ):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=down_bias)
        self.up = nn.Linear(rank, d_out, bias=out_bias)

    def composed_weight(self) -> torch.Tensor:
        return self.up.weight @ self.down.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _make_proj(d_in: int, d_out: int, rank: int | None, bias: bool = False) -> nn.Module:
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
    if isinstance(proj, nn.Parameter):
        return F.linear(x.to(proj.dtype), proj)
    if isinstance(proj, LowRankLinear):
        return proj(x.to(proj.down.weight.dtype))
    return proj(x.to(proj.weight.dtype))


def _init_gate_proj(gate_proj, init: str, identity_bias: float) -> None:
    if init == "paper":
        return
    bias = _proj_bias(gate_proj)
    hidden = bias.numel() // 3
    with torch.no_grad():
        if init == "zero":
            bias.zero_()
        elif init == "identity":
            target = torch.zeros_like(bias)
            for i, sign in enumerate((1.0, -1.0, 1.0)):
                target[i * hidden : (i + 1) * hidden] = sign * identity_bias
            bias.copy_(target)
        elif init == "uniform":
            bias.fill_(identity_bias)
        else:
            raise ValueError(f"unknown gate init: {init}")
        scale = 1.0 / math.sqrt(hidden)
        for module in _linears(gate_proj):
            nn.init.uniform_(module.weight, -scale, scale)


def _whitening_transform(
    values: torch.Tensor, mode: str, ridge: float, return_inverse: bool = False
) -> torch.Tensor:
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
    if mix not in ("raw", "whitened"):
        raise ValueError(f"mix must be 'raw' or 'whitened', got {mix!r}")
    num_tokens, num_sources, hidden = values.shape
    w_inv = None
    if whiten == "per_head":
        values, query = values.float(), query.float()
        dh = hidden // heads
        flat_v = values.reshape(-1, heads, dh)
        flat_q = query.reshape(-1, heads, dh)
        with torch.no_grad():
            V = flat_v.detach()
            n = V.shape[0]
            cov = torch.einsum("nhd,nhe->hde", V, V) / n
            scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
            eps_d = torch.finfo(cov.dtype).eps
            ridge_h = max(dh, n) * eps_d * scale
            cov.diagonal(dim1=-2, dim2=-1).add_(ridge_h.unsqueeze(-1))
            evals, evecs = torch.linalg.eigh(cov)
            floor = (evals[..., -1:] * dh * eps_d).clamp_min(torch.finfo(cov.dtype).tiny)
            whiten_h = (
                evecs
                @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor)))
                @ evecs.transpose(-1, -2)
            )
        values_s = torch.einsum("nhd,hde->nhe", flat_v, whiten_h).reshape(
            num_tokens, num_sources, hidden
        )
        query_s = torch.einsum("nhd,hde->nhe", flat_q, whiten_h).reshape(num_tokens, hidden)
        mix_values = values
        if heads > 1:
            vs = values_s.view(num_tokens, num_sources, heads, dh)
            qs = query_s.view(num_tokens, heads, dh)
            recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
            logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip
            probs = _softmax1(logits, dim=1) if null else logits.softmax(dim=1)
            routed = (
                probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)
            ).sum(dim=1)
            routed = routed.reshape(num_tokens, hidden)
        else:
            recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
            logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip
            probs = _softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
            routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
        if return_scores:
            return routed, probs
        return routed
    if whiten in ("diag", "full"):
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
    mix_values = values_s if mix == "whitened" else values

    if heads > 1:
        dh = hidden // heads
        vs = values_s.view(num_tokens, num_sources, heads, dh)
        qs = query_s.view(num_tokens, heads, dh)
        recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
        logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip
        probs = _softmax1(logits, dim=1) if null else logits.softmax(dim=1)
        routed = (probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)).sum(
            dim=1
        )
        routed = routed.reshape(num_tokens, hidden)
    else:
        recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
        logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip
        probs = _softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
        routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
    if mix == "whitened" and w_inv is not None:
        routed = routed * w_inv if w_inv.dim() == 1 else routed @ w_inv
    if return_scores:
        return routed, probs
    return routed


@dataclass
class GdarConfig:

    block_size: int | None = None
    output_route: bool = True

    gate_rank: int | None = None
    gate_init: str = "paper"
    gate_init_bias: float = 4.0
    gate_param: str = "sigmoid"
    write_carrier_bias: float = -4.0
    decay_ladder: int = 0
    decay_tau_max: float = 100.0
    update: str = "shensi"

    read_heads: int = 1
    read_null: bool = False
    read_whiten: str = "off"
    read_ridge: float = 1.0e-3
    address: str = "state"
    gate_source: str = "state"
    decay_positivity: str = "free"
    lambda_clamp: float | None = -0.5
    read_mix: str = "raw"

    q_rank: int | None = None
    k_rank: int | None = None

    init_std: float = 0.02
    residual_dropout: float | None = None

    def validated(self) -> GdarConfig:
        if self.gate_param not in ("sigmoid", "deviation"):
            raise ValueError(
                f"gate_param must be 'sigmoid' or 'deviation', got {self.gate_param!r}"
            )
        if self.update == "reference":
            self.update = "shensi"
        if self.update not in ("shensi", "objective"):
            raise ValueError(f"update must be 'shensi' or 'objective', got {self.update!r}")
        if self.address not in ("state", "delta", "novelty"):
            raise ValueError(f"address must be 'state'/'delta'/'novelty', got {self.address!r}")
        if self.read_whiten not in ("off", "diag", "full", "per_head"):
            raise ValueError(
                f"read_whiten must be 'off'/'diag'/'full'/'per_head', got {self.read_whiten!r}"
            )
        if self.gate_source not in ("state", "prefix", "delta"):
            raise ValueError(
                f"gate_source must be 'state'/'prefix'/'delta', got {self.gate_source!r}"
            )
        if self.decay_positivity not in ("free", "project"):
            raise ValueError(
                f"decay_positivity must be 'free'/'project', got {self.decay_positivity!r}"
            )
        if self.read_mix not in ("raw", "whitened"):
            raise ValueError(f"read_mix must be 'raw'/'whitened', got {self.read_mix!r}")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError(f"block_size must be >= 1 or None, got {self.block_size}")
        return self


class AttentionResidual(nn.Module):

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
            repeats = -(-hidden // ladder)
            self.register_buffer(
                "decay_tau_init", log_tau.repeat(repeats)[:hidden], persistent=False
            )
            self.decay_tau = nn.Parameter(self.decay_tau_init.clone())
        else:
            self.decay_tau = None

        # read_scale 零初始化：init 时整条连接是纯残差流（GDAR(0) == Qwen3 逐位）
        self.read_scale = nn.Parameter(torch.zeros(1))

        # deviation：三门 = 1 + 零初始化偏离量，init 即恒等且梯度不消失
        if cfg.gate_param == "deviation":
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
        if rank is None:
            return nn.Parameter(torch.empty(hidden, hidden))
        proj = LowRankLinear(hidden, hidden, rank)
        proj.up.bias.requires_grad_(False)
        return proj


    def reset_parameters(self) -> None:
        std = float(self.cfg.init_std)
        with torch.no_grad():
            for module in self._linear_modules():
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.zero_()
            for proj in (self.q_proj, self.k_proj):
                if isinstance(proj, nn.Parameter):
                    nn.init.normal_(proj, std=std)

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


    def _gate_head(self, state: torch.Tensor) -> torch.Tensor:
        raw = _apply_proj(state, self.gate_proj)
        return raw.reshape(*state.shape[:-1], 3, -1)

    def _gate_input(self, state: torch.Tensor, prefix, delta) -> torch.Tensor:
        src = self.cfg.gate_source
        if src == "prefix" and prefix is not None:
            return self.norm(prefix.float())
        if src == "delta" and delta is not None:
            return self.norm(delta.float())
        return state

    def _gates(self, state: torch.Tensor):
        raw = self._gate_head(state)
        if self.cfg.gate_param == "sigmoid":
            return torch.sigmoid(raw).unbind(-2)

        r_decay, r_erase, r_write = raw.unbind(-2)
        tau = self.decay_tau.exp() if self.decay_tau is not None else 1.0
        # 直通 clamp：前向保证 decay<=1，反向恒等（普通 clamp 在边界零梯度，会冻死 scale）
        if self.cfg.decay_positivity == "project":
            decay_scale = (
                self.decay_scale + (self.decay_scale.clamp(min=0.0) - self.decay_scale).detach()
            )
        else:
            decay_scale = self.decay_scale
        decay = torch.exp(-F.softplus(r_decay) * decay_scale * tau)
        erase = F.softplus(r_erase) * self.erase_scale
        write = 1.0 + torch.tanh(r_write) * self.write_scale
        return decay, erase, write

    def _state(self, prefix: torch.Tensor, delta: torch.Tensor | None) -> torch.Tensor:
        return self.norm(prefix.float() + (delta.float() if delta is not None else 0.0))

    def read(
        self, prefix: torch.Tensor, blocks: torch.Tensor | None, state: torch.Tensor | None = None
    ):
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

    def __init__(self, hidden: int, cfg: GdarConfig, eps: float = 1.0e-6):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.eps = eps
        self.q_proj = AttentionResidual._make_qk(hidden, cfg.q_rank)
        self.read_scale = nn.Parameter(torch.zeros(1))
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
