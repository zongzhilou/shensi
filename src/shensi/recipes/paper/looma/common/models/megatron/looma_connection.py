"""Looma 的深度连接与块内不动点求解器，沿用 Megatron-Core 的 ``[s, b, h]`` 张量约定。

连接在深度轴上做门控 delta 更新，块循环用阻尼迭代解到不动点；供 ``looma_layer`` 与 spec 使用。
"""

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
    """把 ``func`` 迭代到不动点，返回逐 token 的最低残差估计。

    循环在 ``no_grad`` 下跑，激活显存与迭代次数无关。``grad_steps=1`` 在估计点补一次带梯度的
    求值，其值恰为估计本身（``f(z*) - detach(f(z*) - z*)``），于是 ``dL/dθ = ∂f(z*)/∂θ``。

    Args:
        func: 一次迭代的映射，接收并返回状态张量列表。
        state: 初始状态。
        max_iter: 迭代次数上界。
        tol: 残差阈值，达到即停。
        stop_mode: ``"rel"`` 用相对残差，``"abs"`` 用绝对残差。
        tau: 阻尼系数，即 ``x <- tau * f(x) + (1 - tau) * x``。
        grad_steps: ``0`` 表示不补梯度步，``1`` 表示补一步。

    Returns:
        收敛估计（或残差最低的那次迭代）的状态列表，形状与 ``state`` 一致。
    """
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
    """不带可学习权重的 RMS 归一化：状态以未加权的形式进入门控。"""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """连接前向：由 deviation scales 与状态算出门与读，输出本段流。"""
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


@dataclass
class LoomaConfig:
    """连接与块求解器的全部可调旋钮。"""

    max_iter: int = 8
    tol: float = 1.0e-2
    stop_mode: str = "rel"
    tau: float = 1.0
    grad_steps: int = 1
    rank: int = 64
    read_heads: int = 8
    lambda_clamp: float | None = -0.5
    # 写门的载体偏置：写尺度为 ``1 + tanh(r_write) * scale``，偏置决定其初值
    write_carrier_bias: float = -4.0
    # 末层是否在模型收尾归一化前对行与 stream 再做一次读；关闭后末层不建输出连接
    output_route: bool = True
    # decay 阶梯的顶端：``decay_tau = linspace(0, 1, hidden) * log(decay_tau_max)``；
    # 为 None 时由层按 ``2 * num_layers`` 填入
    decay_tau_max: float | None = None
    init_std: float = 0.02
    # 写入前施加在 sublayer 输出上的 dropout；None 表示跟随 ``config.hidden_dropout``
    residual_dropout: float | None = None

    def validated(self) -> "LoomaConfig":
        """逐项校验旋钮取值，返回自身以便链式调用。"""
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
    """收集 ``looma_*`` 旋钮并补齐默认值，返回校验后的配置。

    ``config`` 非空时从它取 ``init_method_std`` 与 ``num_layers``（后者给出 decay 阶梯顶端）。
    """
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
    """块的注意力残差：深度轴上的门控 delta 更新。

    状态先做未加权 RMSNorm，再经低秩门投影得到 decay/erase/write 三路（衰减前缀、放大增量、
    沿 ``khat`` 扣掉一部分自身，系数为均值 erase，可钳制），其后可选地叠加一次白化 Softmax₁ 读。
    """

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
        self.g_scale = nn.Parameter(torch.zeros(4))  # 依次为 decay / erase / write / read
        self.register_buffer(
            "decay_tau_init", torch.linspace(0.0, 1.0, hidden) * self.ladder_top, persistent=False
        )
        self.decay_tau = nn.Parameter(self.decay_tau_init.clone())
        self.reset_parameters()

    def _linears(self):
        return [self.gate_proj[0], self.gate_proj[1], self.q_proj[0], self.q_proj[1],
                self.k_proj[0], self.k_proj[1]]

    def reset_parameters(self) -> None:
        """初始化：先通用初始化，再打上恒等锚点。"""
        with torch.no_grad():
            for module in self._linears():
                nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
                if module.bias is not None:
                    module.bias.zero_()
            # 锚点：门尺度归零、decay 阶梯复位
            self.g_scale.zero_()
            self.decay_tau.copy_(self.decay_tau_init)
            # 只把 W_up 置零（投影输出恰为其偏置，也就是恒等），W_down 必须保持非零：
            # 两侧同时为零会梯度互锁（dL/dW_up ∝ W_down、dL/dW_down ∝ W_up，恒为 0）。
            # 这里为 W_down 重抽一次初值，固定 RNG 消耗的顺序。
            nn.init.normal_(self.gate_proj[0].weight, mean=0.0, std=self.init_std)
            nn.init.zeros_(self.gate_proj[1].weight)
            nn.init.zeros_(self.gate_proj[1].bias)
            self.gate_proj[1].bias[2 * self.hidden : 3 * self.hidden] = float(
                self.cfg.write_carrier_bias
            )

    def _read(self, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """逐头白化后的 Softmax₁ 读；混合作用在原始（未白化）的源上。"""
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
        scores = torch.exp(logits - F.softplus(s))  # Softmax₁：剩余质量归空路由
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
        """把 ``prefix + delta`` 与已贴出的行聚合成本层输出。

        Args:
            prefix: 已累计的深度前缀，``[s, b, h]``。
            delta: 本块流与前缀之差；为 ``None`` 时只对前缀做更新。
            blocks: 已贴出的行，``[s, b, N, h]``；为空时跳过读。
            output_norm_weight: 收尾带权 RMSNorm 的权重；为 ``None`` 时不做收尾。
        """
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
        # decay <= 1 要求尺度非负：前向截断、梯度直通（普通 clamp 在边界梯度为零，会冻死门）
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
