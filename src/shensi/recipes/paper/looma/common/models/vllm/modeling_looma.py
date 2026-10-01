"""Looma 在 vLLM 上的原生实现：骨干直接复用 vLLM 的 Llama 组件，只新写深度连接与不动点求解。

供 ``register_model`` 登记进 ``ModelRegistry`` 后由引擎加载：q/k/v 与输出投影、RoPE、paged
attention 与 KV cache、SwiGLU、RMSNorm 权重都用原生件，连接与求解器复用
``..megatron.looma_connection``（纯 torch 依赖），serve 与训练/导出因此是同一段算子。块内 KV
冻结复用首轮结果，只有 K/V 走 paged cache；逐 token 的深度状态每次 forward 重建。不支持 PP > 1。
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, default_weight_loader

from ..megatron.looma_connection import (
    LoomaAttentionResidual,
    LoomaConfig as LoomaConnConfig,
    solve_block,
)

__all__ = ["LoomaAttention", "LoomaDecoderLayer", "LoomaForCausalLM", "LoomaModel", "conn_config"]

_CONNECTION_MARK = "_attn_res."


def conn_config(config) -> LoomaConnConfig:
    """HF config 上的 ``looma_*`` 旋钮 → 连接/求解器配置（同名去前缀）。"""
    return LoomaConnConfig(
        max_iter=int(getattr(config, "looma_max_iter", 8)),
        tol=float(getattr(config, "looma_tol", 1e-2)),
        stop_mode=str(getattr(config, "looma_stop_mode", "rel")),
        tau=float(getattr(config, "looma_tau", 1.0)),
        grad_steps=0,  # serve 不建图：不做 phantom 梯度步
        rank=int(getattr(config, "looma_rank", 64)),
        read_heads=int(getattr(config, "looma_read_heads", 8)),
        lambda_clamp=getattr(config, "looma_lambda_clamp", -0.5),
        write_carrier_bias=float(getattr(config, "looma_write_carrier_bias", -4.0)),
        output_route=bool(getattr(config, "looma_output_route", True)),
        decay_tau_max=2.0 * float(config.num_hidden_layers),
    ).validated()


class LoomaAttention(LlamaAttention):
    """Llama 的注意力，加一条"后续迭代复用首轮 K/V"的路（只重写 forward，构造继承）。"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        frozen_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """``frozen_kv`` 为空时投出 q/k/v 并把它作为块的 K/V 返回，否则只更新 query。"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if frozen_kv is None:
            # 首轮迭代：正常投 q/k/v 并转 RoPE，这份 K/V 就是这个块的历史
            q, k = self.rotary_emb(positions, q, k)
            frozen_kv = (k, v)
        else:
            # 后续迭代：只有 query 动（k 已转过 RoPE，不能再转一次）
            k, v = frozen_kv
            q, _ = self.rotary_emb(positions, q, None)
        # K/V 冻结：同一份 K/V 重写进同一槽位是幂等的
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output, frozen_kv


class LoomaDecoderLayer(LlamaDecoderLayer):
    """一个 block：把 ``_step`` 解到不动点。

    构造继承 ``LlamaDecoderLayer``（注意力实现类、量化、cache 配置、bias 开关
    都由它决定），只把一个实例的类换成 ``LoomaAttention``——差别只在 forward
    需要一条"K/V 冻结"的路。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.self_attn.__class__ = LoomaAttention
        cfg = conn_config(config)
        eps = config.rms_norm_eps
        self.self_attention_attn_res = LoomaAttentionResidual(config.hidden_size, cfg, eps=eps)
        self.mlp_attn_res = LoomaAttentionResidual(config.hidden_size, cfg, eps=eps)
        self.solver = dict(
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            stop_mode=cfg.stop_mode,
            tau=cfg.tau,
            grad_steps=cfg.grad_steps,
        )

    def _step(self, stream, prefix_sum, rows, positions, frozen_kv):
        """一次迭代：连接 → 注意力 → 连接 → MLP。"""
        routed = self.self_attention_attn_res(
            prefix_sum,
            stream - prefix_sum,
            rows,
            output_norm_weight=self.input_layernorm.weight,
        )
        attn_out, frozen_kv = self.self_attn(routed, positions, frozen_kv)
        stream = routed + attn_out
        prefix_sum = stream

        routed = self.mlp_attn_res(
            prefix_sum,
            prefix_sum,
            rows,
            output_norm_weight=self.post_attention_layernorm.weight,
        )
        stream = routed + self.mlp(routed)
        prefix_sum = prefix_sum + stream
        return stream, prefix_sum, frozen_kv

    def forward(self, stream, prefix_sum, rows, positions):
        """一个 block：把 ``prefix_sum`` 追加进行银行，再把 ``_step`` 解到不动点。"""
        flat = prefix_sum.unsqueeze(1) if prefix_sum.dim() == 2 else prefix_sum
        rows = flat if rows is None else torch.cat([rows, flat], dim=1)

        stream, prefix_sum, frozen_kv = self._step(stream, prefix_sum, rows, positions, None)

        def block_map(next_stream, next_prefix):
            return self._step(next_stream, next_prefix, rows, positions, frozen_kv)[:2]

        stream, prefix_sum = solve_block(block_map, [stream, prefix_sum], **self.solver)
        return stream, prefix_sum, rows


class LoomaModel(LlamaModel):
    """骨干：建立过程整个继承 ``LlamaModel``（只换 decoder 层的类），forward 换成逐块循环。"""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LoomaDecoderLayer,
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        config = vllm_config.model_config.hf_config
        self.output_attn_res = (
            LoomaAttentionResidual(config.hidden_size, conn_config(config), eps=config.rms_norm_eps)
            if getattr(config, "looma_output_route", True)
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor:
        """与基类同一骨架，但逐块传递 (stream, prefix_sum, rows) 三件套深度状态。"""
        _ = extra_layer_kwargs
        if get_pp_group().world_size > 1:
            raise NotImplementedError(
                "Looma 的 vLLM 实现只做单 stage：深度状态是 (stream, prefix, 行银行) 三件套，"
                "行银行宽度逐层增长，跨 stage 的 p2p 契约没有实现（mcore 侧用"
                "variable_seq_lengths 的动态形状做掉了，vLLM 侧没做）。TP 不受影响。"
            )
        assert intermediate_tensors is None
        stream = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        prefix_sum = stream
        rows: torch.Tensor | None = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            stream, prefix_sum, rows = layer(stream, prefix_sum, rows, positions)
        if self.output_attn_res is not None:
            stream = self.output_attn_res(prefix_sum, stream, rows)
        return self.norm(stream)


class LoomaForCausalLM(LlamaForCausalLM):
    """``LlamaForCausalLM`` 的外壳：只把骨干换成 ``LoomaModel``（走 ``_init_model`` 扩展点）。"""

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LoomaDecoderLayer,
    ) -> LoomaModel:
        """忽略 ``layer_type``，钉死 ``LoomaDecoderLayer``，其余建立过程全继承。"""
        _ = layer_type
        return LoomaModel(vllm_config=vllm_config, prefix=prefix, layer_type=LoomaDecoderLayer)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """连接张量按原名字直连，骨干走父类的映射表。

        父类映射按子串改名，连接里也有 ``gate_proj`` / ``q_proj``，
        混进去会被改坏，所以先按名字分流。
        """
        by_name = dict(self.named_parameters())
        backbone: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        for name, tensor in weights:
            if _CONNECTION_MARK not in name:
                backbone.append((name, tensor))
                continue
            param = by_name.get(name)
            if param is None:
                continue
            default_weight_loader(param, tensor)
            loaded.add(name)
        loaded |= AutoWeightsLoader(self).load_weights(backbone, mapper=self.hf_to_vllm_mapper)
        return loaded
