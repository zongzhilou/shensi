"""深度连接算子：ARRouter / DeltaRouter / DepthWeightedAverage / MultiwayDynamicDense。"""

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


class WeightedRMSNorm(nn.Module):

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

    def __init__(self, dim: int = -1, eps: float = 1.0e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=self.dim, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype)


@dataclass
class DepthConnectionConfig:

    variant: str = "dar"
    block_size: int | None = 1
    output_route: bool = True

    ar_reset: str = "keep"

    use_null_source: bool = False

    dwa_param: str = "deviation"
    dwa_dilation: int = 1

    mudd_num_ways: int = 1
    mudd_param: str = "deviation"
    mudd_act: str = "gelu"
    mudd_hidden_round: int = 64
    mudd_scale_dw: bool = False
    mudd_use_post_norm: bool = False
    mudd_use_pre_norm: bool = False

    identity: bool = True
    force_identity: bool = False
    init_std: float = 0.02
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


class ARRouter(nn.Module):

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
        std = float(self.cfg.init_std)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=std)
            nn.init.ones_(self.norm.weight)
            if self.read_scale is not None:
                self.read_scale.zero_()

    def forward(self, prefix: torch.Tensor, sources: torch.Tensor | None) -> torch.Tensor:
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
            return routed
        return prefix + self.read_scale.to(prefix.dtype) * (routed - prefix)


class DeltaRouter(nn.Module):

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

        V = torch.stack(entries, dim=0)
        K = self.norm(V)
        query = self.proj.weight.view(-1)
        logits = torch.einsum("d, n t d -> n t", query, K)
        weights = logits.softmax(dim=0)
        selected = torch.einsum("n t, n t d -> t d", weights.to(V.dtype), V)
        if self.read_scale is None:
            return prefix + selected.to(prefix.dtype)
        return prefix + self.read_scale.to(prefix.dtype) * selected.to(prefix.dtype)


class DepthWeightedAverage(nn.Module):

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
        if self.force_identity:
            return sources[..., -1, :].contiguous()
        alpha = self.effective_alpha().to(sources.dtype)
        return torch.einsum("tnd,n->td", sources.float(), alpha.float()).to(sources.dtype)


class MultiwayDynamicDense(nn.Module):

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
        if round_to > 0:
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
        if self.force_identity:
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
        if self.scale_dw:
            dw = dw / math.sqrt(self.hidden_dim)
        dw = dw.view(*x.shape[:-1], self.num_ways, self.num_states)
        return dw + self.effective_prior().to(dw.dtype)

    @staticmethod
    def aggregate(dw: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("tcn,tnd->ctd", dw.float(), states.float()).to(states.dtype)
        return out
