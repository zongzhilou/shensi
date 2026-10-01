"""Looma 建模：每个 decoder layer 是一个 block，block 是其自身映射的不动点。

block 把自身映射迭代到不再移动的点，该点即它发布的边界行，逐 block 叠出深度方向的记忆；
骨干仍是原生 Llama，新增的算子只有不动点求解器与替代残差相加的深度连接 ``LoomaAttentionResidual``。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .configuration_looma import LoomaConfig

__all__ = [
    "LoomaAttention",
    "LoomaAttentionResidual",
    "LoomaConfig",
    "LoomaDecoderLayer",
    "LoomaForCausalLM",
    "LoomaModel",
    "LoomaPreTrainedModel",
    "LoomaUnweightedRMSNorm",
    "layer_step",
    "solve_block",
]


def _batch_flatten(x: torch.Tensor) -> torch.Tensor:
    """``(T, ...) -> (T, prod(...))``：每个 token 一行，与后续维度形状无关。"""
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
    """``mask``（每个 token 一个 bool）为真的位置取 ``new``，其余取 ``old``。"""
    return torch.where(mask.reshape(-1, *([1] * (new.dim() - 1))), new, old)


def solve_block(
    func: Callable[..., list[torch.Tensor]],
    state: list[torch.Tensor],
    *,
    max_iter: int = 8,
    tol: float = 1.0e-2,
    stop_mode: str = "rel",
    tau: float = 1.0,
    grad_steps: int = 1,
) -> list[torch.Tensor]:
    """把 ``func`` 迭代到不动点，返回每个 token 残差最低的状态估计。

    ``func`` 接收状态张量并返回同形状的下一状态；统计量在展平的 ``(T, D)`` 上逐 token 计算，
    阻尼为 ``x <- tau * f(x) + (1 - tau) * x``，``grad_steps=1`` 时反向只经过对估计值的一次求值。
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
            nxt = func(*_unflatten_state(x, shapes))
            fx = tau * _flatten_state(nxt)[0] + (1.0 - tau) * x
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
    # 值等于估计值本身，反向只看到对映射在估计值处的那一次求值
    stepped = _flatten_state(func(*_unflatten_state(lowest_est, shapes)))[0]
    return _unflatten_state(stepped - (stepped - lowest_est).detach(), shapes)


class LoomaUnweightedRMSNorm(nn.Module):
    """无可学习权重的 RMSNorm：状态以未加权形式进入各个门。"""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm 前向用实例上缓存的 eps：引擎会改写 nn.RMSNorm，取 self.norm.eps 会炸。"""
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


def _proj(x: torch.Tensor, seq: nn.Sequential) -> torch.Tensor:
    """在 fp32 下计算低秩对 ``W2(W1 x) + b2``：matmul 不做类型提升，两个因子都显式上转。"""
    down, up = seq[0], seq[1]
    bias = None if up.bias is None else up.bias.float()
    return F.linear(F.linear(x, down.weight.float()), up.weight.float(), bias)


class LoomaAttentionResidual(nn.Module):
    """block 的注意力残差：深度轴上的 gated delta rule。

    零初始化的 deviation scale 与 read 门使模块在初始化处恰好等于 ``prefix + delta``，且每个
    scale 都有非零梯度；erase 方向取自待写入的 delta，更新式是目标函数的闭式极小点。
    """

    def __init__(self, config: LoomaConfig):
        super().__init__()
        hidden = config.hidden_size
        rank = min(int(config.looma_rank), hidden)
        self.hidden = hidden
        # hidden 不能被头数整除时 per-head reshape 会静默出错，退回单头
        self.read_heads = int(config.looma_read_heads) if hidden % int(config.looma_read_heads) == 0 else 1
        self.lambda_clamp = config.looma_lambda_clamp
        self.write_carrier_bias = float(config.looma_write_carrier_bias)
        # 逐 channel 时间常数梯子的上界
        self.ladder_top = math.log(2.0 * config.num_hidden_layers)
        self.norm = LoomaUnweightedRMSNorm(config.rms_norm_eps)
        # eps 另存一份：推理引擎会按数据流把 norm 模块换成自己的实现（没有 .eps），
        # 连接不能依赖那个属性。
        self.eps = float(config.rms_norm_eps)
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden, rank, bias=False), nn.Linear(rank, 3 * hidden, bias=True)
        )
        self.q_proj = nn.Sequential(nn.Linear(hidden, rank, bias=False), nn.Linear(rank, hidden, bias=False))
        self.k_proj = nn.Sequential(nn.Linear(hidden, rank, bias=False), nn.Linear(rank, hidden, bias=False))
        self.init_std = float(getattr(config, "initializer_range", 0.02) or 0.02)
        # 四个 deviation scale：decay / erase / write / read
        self.g_scale = nn.Parameter(torch.zeros(4))
        self.decay_tau = nn.Parameter(torch.linspace(0.0, 1.0, hidden) * self.ladder_top)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化即恒等映射，并统一负责本模块的初始化。

        scale 的载体取非零值，否则对应 scale 的梯度会消失；``W_up`` 置零使门投影的输出恰好
        等于其 bias，``W_down`` 保留随机初始化，以免上下两半的梯度互相锁死。
        """
        with torch.no_grad():
            self.g_scale.zero_()
            self.decay_tau.copy_(torch.linspace(0.0, 1.0, self.hidden) * self.ladder_top)
            nn.init.normal_(self.gate_proj[0].weight, mean=0.0, std=self.init_std)
            nn.init.zeros_(self.gate_proj[1].weight)
            nn.init.zeros_(self.gate_proj[1].bias)
            # 3H 投影的最后一段是 write 门，载体偏置须非零
            self.gate_proj[1].bias[2 * self.hidden :] = self.write_carrier_bias

    def _read(self, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """逐头的 whitened Softmax₁ 读，混合仍在原始 sources 上进行。

        白化矩阵是由 sources 估计出的预条件（已 detach），按头各算一份协方差；它施加在
        ``no_grad`` 之外，query 的梯度得以保留。
        """
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
            whiten = evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor))) @ evecs.transpose(-1, -2)
        v_w = torch.einsum("nhd,hde->nhe", v_flat, whiten).view(*values.shape[:-1], self.read_heads, head_dim)
        q_w = torch.einsum("nhd,hde->nhe", q_flat, whiten).view(*query.shape[:-1], self.read_heads, head_dim)
        logits = (v_w * q_w.unsqueeze(-3)).sum(dim=-1) * torch.rsqrt(v_w.square().mean(dim=-1) + self.eps)
        s = torch.logsumexp(logits, dim=-2, keepdim=True)
        # Softmax_1：未分配出去的权重走 null 路径
        scores = torch.exp(logits - F.softplus(s))
        raw = values.view(*values.shape[:-1], self.read_heads, head_dim)
        # 在 sources 维上混合（加上 head 维后为 -3）
        routed = (scores.unsqueeze(-1) * raw).sum(dim=-3)
        return routed.reshape(*values.shape[:-2], hidden)

    def forward(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        output_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在可读的 sources 上聚合 ``prefix + delta``：已发布的各行，加上本 block 写入前的流。

        发布新行由 block 自己在入口处完成，不在这里。
        """
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self.norm(prefix + (delta if delta is not None else 0.0))

        # 对状态做一次 3H 投影，同时产出 delta rule 的三个门
        r_decay, r_erase, r_write = _proj(state, self.gate_proj).reshape(*state.shape[:-1], 3, -1).unbind(-2)
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()
        # decay <= 1 要求 scale 非负：前向取 clamp，梯度直通（普通 clamp 在边界处梯度为零，会把门冻住）
        decay_scale = decay_scale + (decay_scale.clamp(min=0.0) - decay_scale).detach()
        decay = torch.exp(-F.softplus(r_decay) * decay_scale * self.decay_tau.exp())
        erase = F.softplus(r_erase) * erase_scale
        write = 1.0 + torch.tanh(r_write) * write_scale

        # 先 decay 并写入得到 m，再减去 khat 方向上使目标函数最小的分量
        khat = F.normalize(_proj(delta if delta is not None else state, self.k_proj), dim=-1)
        m = decay * prefix + write * (delta if delta is not None else 0.0)
        lam = erase.mean(dim=-1, keepdim=True)
        if self.lambda_clamp is not None:
            lam = lam.clamp(min=float(self.lambda_clamp))
        updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)

        # bank 为空时（模型流程之外单独调用本模块）没有可读的源，更新结果保持不变
        if blocks.shape[-2]:
            values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
            updated = updated + read_scale * self._read(values, _proj(state, self.q_proj))

        if output_norm_weight is not None:
            reciprocal_std = torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + self.eps)
            updated = updated * reciprocal_std * output_norm_weight.float()

        return updated.to(out_dtype)


class LoomaAttention(LlamaAttention):
    """注意力子层：沿用 LlamaAttention 的投影与 RoPE，块内 K/V 只在首轮投出后冻结。"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        loop_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """block 的后续迭代以首轮投影出的 keys/values 重入的 Llama attention。

        给定 ``loop_kv`` 时只重新投影 query，键值沿用首轮结果：细化后的流重问同一份历史，
        而不是去注意一份已被扰动的历史，掩码也保持首轮建好的键长。
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings

        if loop_kv is None:
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        else:
            # 后续迭代只有 query 在动
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
            key_states, value_states = loop_kv

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights, (key_states, value_states)


def layer_step(
    self: "LoomaDecoderLayer",
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    prefix_sum: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """block 的一次迭代：以注意力残差替代 layernorm 的 ``LlamaDecoderLayer`` 前向。

    携带 block 的状态（流、可读的行 ``residual``、``prefix_sum``）；定义在模块级是因为
    block 就是这一步被求解到不动点，后续迭代复用的 keys/values 经 ``kwargs["loop_kv"]`` 传入并原样传出。
    """
    loop_kv = kwargs.pop("loop_kv", None)

    hidden_states = self.self_attention_attn_res(
        prefix_sum,
        hidden_states - prefix_sum,
        residual,
        output_norm_weight=self.input_layernorm.weight,
    )
    attn_output, _, loop_kv = self.self_attn(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values if loop_kv is None else None,
        use_cache=bool(use_cache) and loop_kv is None,
        loop_kv=loop_kv,
        **kwargs,
    )
    hidden_states = hidden_states + attn_output
    prefix_sum = hidden_states

    hidden_states = self.mlp_attn_res(
        prefix_sum,
        prefix_sum,
        residual,
        output_norm_weight=self.post_attention_layernorm.weight,
    )
    hidden_states = hidden_states + self.mlp(hidden_states)
    prefix_sum = prefix_sum + hidden_states
    return hidden_states, prefix_sum, residual, loop_kv


class LoomaDecoderLayer(LlamaDecoderLayer):
    """一个 block：注意力与 MLP 两个子层各由一个 ``LoomaAttentionResidual`` 残差连接。"""

    def __init__(self, config: LoomaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LoomaAttention(config=config, layer_idx=layer_idx)
        self.self_attention_attn_res = LoomaAttentionResidual(config)
        self.mlp_attn_res = LoomaAttentionResidual(config)
        # block 的求解器设置：迭代上限、相对残差阈值与一步 phantom 梯度
        self.solver_max_iter = int(config.looma_max_iter)
        self.solver_tol = float(config.looma_tol)
        self.solver_stop_mode = str(config.looma_stop_mode)
        self.solver_tau = float(config.looma_tau)
        self.solver_grad_steps = int(config.looma_grad_steps)

    def _solve(self, block_map, state):
        """block 的不动点循环：以 ``looma_max_iter`` 为迭代上限，并按数据相关的统计量提前退出。

        循环的退出条件依赖运行时统计量，刻意不放进编译区域。
        """
        return solve_block(
            block_map,
            state,
            max_iter=self.solver_max_iter,
            tol=self.solver_tol,
            stop_mode=self.solver_stop_mode,
            tau=self.solver_tau,
            grad_steps=self.solver_grad_steps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一个 block：把 ``layer_step`` 求解到不动点，并把传入的流存为本 block 的那一行。"""
        # 本 block 的行就是它收到的流：下方 block 收敛到的和，首个 block 则是输入本身
        residual = torch.cat([residual, prefix_sum.unsqueeze(-2).to(residual.dtype)], dim=-2)

        step = partial(
            layer_step,
            self,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # 首轮迭代产出的 keys/values 与 cache 供循环复用
        hidden_states, prefix_sum, residual, loop_kv = step(
            hidden_states, residual, prefix_sum, past_key_values=past_key_values, use_cache=use_cache
        )

        def block_map(stream, prefix):
            return step(stream, residual, prefix, loop_kv=loop_kv)[:2]

        # 求解本身不带图（phantom 梯度只回传估计值），显存不随迭代次数增长；停机用 batch 统计量
        hidden_states, prefix_sum = self._solve(block_map, [hidden_states, prefix_sum])
        return hidden_states, prefix_sum, residual


class LoomaPreTrainedModel(LlamaPreTrainedModel):
    """Looma 各模型类的基类：固定 ``config_class`` 并接管连接模块的初始化。"""

    # config_class 必须写在各子类自己的类体里：__init_subclass__ 会按 MRO 重新推导它
    config_class = LoomaConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LoomaAttentionResidual):
            # 初始化处恒等（decay 1 / erase 0 / write 1 / read 静默），初始化规则由模块自己掌握
            module.reset_parameters()


class LoomaModel(LlamaModel, LoomaPreTrainedModel):
    """Looma 骨干：逐 block 前向，并可选地在末端再做一次深度读。"""

    # 写在类体内，原因见 LoomaPreTrainedModel
    config_class = LoomaConfig

    def __init__(self, config: LoomaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LoomaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.output_attn_res = LoomaAttentionResidual(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """逐 block 前向：每个 block 发布自己收到的流，末端读可选地聚合后再归一化。"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        # 起始时 bank 为空：每个 block 发布自己收到的流，故第 0 行是输入本身，每个 block 都有行可读
        residual = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])
        prefix_sum = inputs_embeds
        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, prefix_sum, residual = decoder_layer(
                hidden_states,
                residual,
                prefix_sum,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # 末端读在已发布的各行与末 block 收敛到的流上聚合
        if self.config.looma_output_route:
            hidden_states = self.output_attn_res(prefix_sum, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class LoomaForCausalLM(LlamaForCausalLM, LoomaPreTrainedModel):
    """Looma 的因果语言模型：``LlamaForCausalLM`` 骨架加 Looma 骨干。"""

    # 写在类体内，原因见 LoomaPreTrainedModel
    config_class = LoomaConfig

    def __init__(self, config: LoomaConfig):
        super().__init__(config)
        self.model = LoomaModel(config)
        self.post_init()


try:
    AutoConfig.register("looma", LoomaConfig)
except ValueError:
    pass

try:
    AutoModel.register(LoomaConfig, LoomaModel)
    AutoModelForCausalLM.register(LoomaConfig, LoomaForCausalLM)
except ValueError:
    pass

# 各注册到自己的 auto 名字下：config 类若注册成模型名，加载拿到的会是 config 而不是模型
for _cls, _auto in (
    (LoomaConfig, "AutoConfig"),
    (LoomaModel, "AutoModel"),
    (LoomaForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
