"""Looma 的 vLLM 原生实现（rollout 与评测用）。"""


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
    return LoomaConnConfig(
        max_iter=int(getattr(config, "looma_max_iter", 8)),
        tol=float(getattr(config, "looma_tol", 1e-2)),
        stop_mode=str(getattr(config, "looma_stop_mode", "rel")),
        tau=float(getattr(config, "looma_tau", 1.0)),
        grad_steps=0,
        rank=int(getattr(config, "looma_rank", 64)),
        read_heads=int(getattr(config, "looma_read_heads", 8)),
        lambda_clamp=getattr(config, "looma_lambda_clamp", -0.5),
        write_carrier_bias=float(getattr(config, "looma_write_carrier_bias", -4.0)),
        output_route=bool(getattr(config, "looma_output_route", True)),
        decay_tau_max=2.0 * float(config.num_hidden_layers),
    ).validated()


class LoomaAttention(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        frozen_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if frozen_kv is None:

            q, k = self.rotary_emb(positions, q, k)
            frozen_kv = (k, v)
        else:

            k, v = frozen_kv
            q, _ = self.rotary_emb(positions, q, None)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output, frozen_kv


class LoomaDecoderLayer(LlamaDecoderLayer):

    """Looma 的 vLLM 原生解码层。"""
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
        flat = prefix_sum.unsqueeze(1) if prefix_sum.dim() == 2 else prefix_sum
        rows = flat if rows is None else torch.cat([rows, flat], dim=1)

        stream, prefix_sum, frozen_kv = self._step(stream, prefix_sum, rows, positions, None)

        def block_map(next_stream, next_prefix):
            return self._step(next_stream, next_prefix, rows, positions, frozen_kv)[:2]

        stream, prefix_sum = solve_block(block_map, [stream, prefix_sum], **self.solver)
        return stream, prefix_sum, rows


class LoomaModel(LlamaModel):

    """Looma 的 vLLM 原生主干。"""
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

    """Looma 的 vLLM 原生因果语言模型（rollout 与评测用）。"""
    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LoomaDecoderLayer,
    ) -> LoomaModel:
        _ = layer_type
        return LoomaModel(vllm_config=vllm_config, prefix=prefix, layer_type=LoomaDecoderLayer)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
