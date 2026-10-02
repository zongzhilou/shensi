"""Looma 连接算子（mcore 侧）：块内不动点求解与门控 delta 读写。"""


from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoomaAttentionResidual",
    "LoomaConfig",
    "UnweightedRMSNorm",
    "looma_knobs_from_kwargs",
    "solve_block",
]


def _batch_flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _flatten_state(state) -> tuple[torch.Tensor, list[torch.Size]]:
    state = list(state)
    return torch.cat([_batch_flatten(t) for t in state], dim=-1), [t.shape for t in state]


def _unflatten_state(flat: torch.Tensor, shapes: list[torch.Size]) -> list[torch.Tensor]:
    out, offset = [], 0
    for shape in shapes:
        width = math.prod(shape[1:]) if len(shape) > 1 else 1
        out.append(flat[..., offset : offset + width].reshape(shape))
        offset += width
    return out


def _masked_mix(mask: torch.Tensor, new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.reshape(-1, *([1] * (new.dim() - 1))), new, old)


def solve_block(
    func,
    state: list[torch.Tensor],
    *,
    max_iter: int = 8,
    tol: float = 1.0e-2,
    stop_mode: str = "rel",
    tau: float = 1.0,
    grad_steps: int = 1,
) -> list[torch.Tensor]:
    """块内不动点求解：迭代到相对残差达标或到迭代上限。"""
    if stop_mode not in ("rel", "abs"):
        raise ValueError(f"stop_mode must be 'rel' or 'abs', got {stop_mode!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    with torch.no_grad():
        flat, shapes = _flatten_state(state)
        flat = flat.detach()
        lowest = torch.full((flat.shape[0],), float("inf"), device=flat.device, dtype=flat.dtype)
        lowest_est = flat
        fx = x = flat
        for _ in range(int(max_iter)):
            x = fx
            fx = tau * _flatten_state(func(*_unflatten_state(x, shapes)))[0] + (1.0 - tau) * x
            gx = fx - x
            abs_diff = gx.norm(dim=-1)
            diff = abs_diff / (fx.norm(dim=-1) + 1e-9) if stop_mode == "rel" else abs_diff
            is_lowest = diff < lowest
            lowest_est = _masked_mix(is_lowest, fx, lowest_est)
            lowest = _masked_mix(is_lowest, diff, lowest)
            if diff.max() < tol:
                break

    if not grad_steps:
        return _unflatten_state(lowest_est, shapes)
    stepped = _flatten_state(func(*_unflatten_state(lowest_est, shapes)))[0]
    return _unflatten_state(stepped - (stepped - lowest_est).detach(), shapes)


class UnweightedRMSNorm(nn.Module):

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


@dataclass
class LoomaConfig:

    """连接与求解器的开关集合（块粒度、迭代上限、容差、低秩与读头数等）。"""
    max_iter: int = 8
    tol: float = 1.0e-2
    stop_mode: str = "rel"
    tau: float = 1.0
    grad_steps: int = 1
    rank: int = 64
    read_heads: int = 8
    lambda_clamp: float | None = -0.5

    write_carrier_bias: float = -4.0

    output_route: bool = True


    decay_tau_max: float | None = None
    init_std: float = 0.02

    residual_dropout: float | None = None

    def validated(self) -> "LoomaConfig":
        if self.stop_mode not in ("rel", "abs"):
            raise ValueError(f"stop_mode must be 'rel' or 'abs', got {self.stop_mode!r}")
        if int(self.max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if float(self.tol) <= 0.0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if int(self.rank) < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if int(self.read_heads) < 1:
            raise ValueError(f"read_heads must be >= 1, got {self.read_heads}")
        if int(self.grad_steps) not in (0, 1):
            raise ValueError(f"grad_steps must be 0 or 1, got {self.grad_steps}")
        if self.decay_tau_max is not None and float(self.decay_tau_max) <= 1.0:
            raise ValueError(f"decay_tau_max must be > 1, got {self.decay_tau_max}")
        return self


_FIELDS = tuple(LoomaConfig.__dataclass_fields__.keys())


def looma_knobs_from_kwargs(kwargs: dict, config=None) -> LoomaConfig:
    """把 TransformerConfig 上的连接旋钮解析成 LoomaConfig。"""
    values = {}
    for key, value in kwargs.items():
        if key.startswith("looma_"):
            name = key[len("looma_") :]
            if name not in _FIELDS:
                raise TypeError(f"unknown Looma knob {key!r}; valid knobs: {_FIELDS}")
            values[name] = value
    if config is not None:
        values.setdefault("init_std", float(getattr(config, "init_method_std", 0.02) or 0.02))
        if values.get("decay_tau_max") is None:
            values["decay_tau_max"] = 2.0 * float(config.num_layers)
    return LoomaConfig(**values).validated()


class LoomaAttentionResidual(nn.Module):

    """Looma 连接本体：块内不动点求解 + 门控 delta 读写；初始化即恒等。"""
    def __init__(self, hidden: int, cfg: LoomaConfig, eps: float = 1.0e-6):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.eps = eps
        self.norm = UnweightedRMSNorm(eps)
        self.read_heads = int(cfg.read_heads) if hidden % int(cfg.read_heads) == 0 else 1
        self.ladder_top = math.log(float(cfg.decay_tau_max or 2.0 * 64))
        self.init_std = float(cfg.init_std)

        rank = min(int(cfg.rank), hidden)
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden, rank, bias=False), nn.Linear(rank, 3 * hidden, bias=True)
        )
        self.q_proj = nn.Sequential(
            nn.Linear(hidden, rank, bias=False), nn.Linear(rank, hidden, bias=False)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden, rank, bias=False), nn.Linear(rank, hidden, bias=False)
        )
        self.g_scale = nn.Parameter(torch.zeros(4))
        self.register_buffer(
            "decay_tau_init", torch.linspace(0.0, 1.0, hidden) * self.ladder_top, persistent=False
        )
        self.decay_tau = nn.Parameter(self.decay_tau_init.clone())
        self.reset_parameters()

    def _linears(self):
        return [self.gate_proj[0], self.gate_proj[1], self.q_proj[0], self.q_proj[1],
                self.k_proj[0], self.k_proj[1]]

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for module in self._linears():
                nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
                if module.bias is not None:
                    module.bias.zero_()

            self.g_scale.zero_()
            self.decay_tau.copy_(self.decay_tau_init)



            nn.init.normal_(self.gate_proj[0].weight, mean=0.0, std=self.init_std)
            nn.init.zeros_(self.gate_proj[1].weight)
            nn.init.zeros_(self.gate_proj[1].bias)
            self.gate_proj[1].bias[2 * self.hidden : 3 * self.hidden] = float(
                self.cfg.write_carrier_bias
            )

    def _read(self, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        hidden = values.shape[-1]
        head_dim = hidden // self.read_heads
        v_flat = values.reshape(-1, self.read_heads, head_dim)
        q_flat = query.reshape(-1, self.read_heads, head_dim)
        with torch.no_grad():
            v = v_flat.detach()
            cov = torch.einsum("nhd,nhe->hde", v, v) / v.shape[0]
            scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
            ridge = max(head_dim, v.shape[0]) * torch.finfo(cov.dtype).eps * scale
            cov.diagonal(dim1=-2, dim2=-1).add_(ridge.unsqueeze(-1))
            evals, evecs = torch.linalg.eigh(cov)
            floor = (evals[..., -1:] * head_dim * torch.finfo(cov.dtype).eps).clamp_min(
                torch.finfo(cov.dtype).tiny
            )
            whiten = (
                evecs
                @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor)))
                @ evecs.transpose(-1, -2)
            )
        v_w = torch.einsum("nhd,hde->nhe", v_flat, whiten).view(
            *values.shape[:-1], self.read_heads, head_dim
        )
        q_w = torch.einsum("nhd,hde->nhe", q_flat, whiten).view(
            *query.shape[:-1], self.read_heads, head_dim
        )
        logits = (v_w * q_w.unsqueeze(-3)).sum(dim=-1) * torch.rsqrt(
            v_w.square().mean(dim=-1) + self.eps
        )
        s = torch.logsumexp(logits, dim=-2, keepdim=True)
        scores = torch.exp(logits - F.softplus(s))
        raw = values.view(*values.shape[:-1], self.read_heads, head_dim)
        routed = (scores.unsqueeze(-1) * raw).sum(dim=-3)
        return routed.reshape(*values.shape[:-2], hidden)

    def forward(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        output_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self.norm(prefix + (delta if delta is not None else 0.0))

        r_decay, r_erase, r_write = (
            F.linear(
                F.linear(state, self.gate_proj[0].weight.float()),
                self.gate_proj[1].weight.float(),
                self.gate_proj[1].bias.float(),
            )
            .reshape(*state.shape[:-1], 3, -1)
            .unbind(-2)
        )
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()

        decay_scale = decay_scale + (decay_scale.clamp(min=0.0) - decay_scale).detach()
        decay = torch.exp(-F.softplus(r_decay) * decay_scale * self.decay_tau.exp())
        erase = F.softplus(r_erase) * erase_scale
        write = 1.0 + torch.tanh(r_write) * write_scale

        khat = F.normalize(
            F.linear(
                F.linear(
                    (delta if delta is not None else state), self.k_proj[0].weight.float()
                ),
                self.k_proj[1].weight.float(),
            ),
            dim=-1,
        )
        m = decay * prefix + write * (delta if delta is not None else 0.0)
        lam = erase.mean(dim=-1, keepdim=True)
        if self.cfg.lambda_clamp is not None:
            lam = lam.clamp(min=float(self.cfg.lambda_clamp))
        updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)

        if blocks is not None and blocks.shape[-2]:
            values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
            query = F.linear(
                F.linear(state, self.q_proj[0].weight.float()), self.q_proj[1].weight.float()
            )
            updated = updated + read_scale * self._read(values, query)

        if output_norm_weight is not None:
            reciprocal_std = torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + self.eps)
            updated = updated * reciprocal_std * output_norm_weight.float()

        return updated.to(out_dtype)
