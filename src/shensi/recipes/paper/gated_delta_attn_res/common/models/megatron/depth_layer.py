"""深度连接的 mcore 层：把深度状态打包进 hidden_states 宽度。"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params

from .depth_connection import (
    ARRouter,
    DeltaRouter,
    DepthConnectionConfig,
    DepthWeightedAverage,
    MultiwayDynamicDense,
    WeightedRMSNorm,
)

__all__ = ["DepthTransformerLayer", "build_depth_submodules", "depth_knobs_from_kwargs"]

_DEPTH_FIELDS = DepthConnectionConfig.field_names()


def depth_knobs_from_kwargs(
    kwargs: dict, config: TransformerConfig | None = None
) -> DepthConnectionConfig:
    values = {}
    for key, value in kwargs.items():
        if key.startswith("depth_"):
            name = key[len("depth_") :]
            if name not in _DEPTH_FIELDS:
                raise TypeError(f"unknown depth knob {key!r}; valid knobs: {_DEPTH_FIELDS}")
            values[name] = value
    if config is not None:
        values.setdefault("init_std", float(getattr(config, "init_method_std", 0.02)))
    return DepthConnectionConfig(**values).validated()


def build_depth_submodules(config: TransformerConfig) -> TransformerLayerSubmodules:
    return get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
    )


def _warn_if_block_never_closes(
    block_size, num_layers, layer_number, variant: str = "connection"
) -> None:
    if block_size and block_size >= num_layers and layer_number == num_layers:
        print(
            f"[depth-connection:{variant}] 警告：block_size={block_size} >= num_layers={num_layers}，"
            "连接永不闭合（惰性）。离线/小规模检查请用层数大于 block_size 的配置。",
            flush=True,
        )


@contextlib.contextmanager
def isolated_rng(seed: int):
    seed = int(seed) % (2**31 - 1)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


class DepthTransformerLayer(TransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules | None = None,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        pg_collection=None,
        vp_stage: int | None = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        cfg = depth_knobs_from_kwargs(kwargs, config)
        if submodules is None:
            submodules = build_depth_submodules(config)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        if config.pipeline_model_parallel_size > 1:
            config.variable_seq_lengths = True
        if config.fp32_residual_connection:
            raise NotImplementedError(
                f"depth connection {cfg.variant!r} does not support fp32_residual_connection."
            )
        if config.recompute_granularity == "full":
            raise NotImplementedError(
                f"depth connection {cfg.variant!r} does not support recompute_granularity='full'."
            )

        config.hetereogenous_dist_checkpoint = True

        self.depth_cfg = cfg
        self.variant = cfg.variant
        self.hidden_size = config.hidden_size
        self.block_size = None if cfg.block_size is None else int(cfg.block_size)
        self.enabled = self.block_size is not None
        self.period = self.block_size or 1
        _warn_if_block_never_closes(self.block_size, config.num_layers, self.layer_number)
        self.per_sublayer_sources = self.variant in ("ar", "dar") and self.period == 1
        self.is_last_layer = self.layer_number == config.num_layers
        self.output_route = bool(cfg.output_route)
        self.write_dropout = (
            float(self.hidden_dropout)
            if cfg.residual_dropout is None
            else float(cfg.residual_dropout)
        )

        base_seed = int(getattr(config, "seed", 0)) + 104729 * self.layer_number
        with isolated_rng(base_seed):
            self._build_connections()

        if config.sequence_parallel:
            for module in self._connection_modules():
                for param in module.parameters():
                    param.sequence_parallel = True


    def _build_connections(self) -> None:
        cfg = self.depth_cfg
        h = self.hidden_size
        eps = self.config.layernorm_epsilon
        self.self_attention_attn_res = None
        self.mlp_attn_res = None
        self.output_attn_res = None
        self.block_dwa = None
        self.dense_conn = None
        self.dense_post_norm = None
        if not self.enabled:
            return
        if self.variant in ("ar", "dar"):
            cls = ARRouter if self.variant == "ar" else DeltaRouter
            extra = {} if self.variant == "ar" else {"null": bool(cfg.use_null_source)}
            self.self_attention_attn_res = cls(h, cfg, eps=eps, **extra)
            self.mlp_attn_res = cls(h, cfg, eps=eps, **extra)
            if self.is_last_layer and self.output_route:
                self.output_attn_res = cls(h, cfg, eps=eps, **extra)
        elif self.variant == "denseformer":
            if self.is_dense_event():
                self.block_dwa = DepthWeightedAverage(self.num_sources_at_event(), cfg)
        elif self.variant == "mudd":
            if self.is_dense_event():
                self.dense_conn = MultiwayDynamicDense(h, self.num_sources_at_event(), cfg, eps=eps)
                if cfg.mudd_use_post_norm:
                    self.dense_post_norm = WeightedRMSNorm(h, eps=eps)
                    with torch.no_grad():
                        self.dense_post_norm.weight.fill_(0.001)
        else:  # pragma: no cover - validated in the config
            raise ValueError(self.variant)

    def _connection_modules(self):
        return [
            m
            for m in (
                self.self_attention_attn_res,
                self.mlp_attn_res,
                self.output_attn_res,
                self.block_dwa,
                self.dense_conn,
                self.dense_post_norm,
            )
            if m is not None
        ]

    def is_dense_event(self) -> bool:
        if not self.enabled:
            return False
        if self.layer_number % self.period == 0:
            return True
        return self.output_route and self.is_last_layer

    def num_sources_at_event(self) -> int:
        extra = (
            1
            if (self.output_route and self.is_last_layer and self.layer_number % self.period != 0)
            else 0
        )
        m = self.layer_number // self.period + extra
        values = m + 1
        k = max(1, int(self.depth_cfg.dwa_dilation))
        return values if k == 1 else len(range(m % k, values, k))


    def _unpack(self, hidden_states: Tensor):
        h = self.hidden_size
        width = hidden_states.shape[-1]
        if width == h:
            return hidden_states.reshape(-1, h), None
        flat = hidden_states.reshape(-1, width)
        num_sources = (width - h) // h
        return flat[..., :h], flat[..., h:].reshape(flat.shape[0], num_sources, h)

    # 深度状态打包进 hidden_states 的宽度：块内宽度 (1+N)*H，出块再收拢
    def _pack(self, prefix: Tensor, sources: Tensor | None, shape) -> Tensor:
        prefix = prefix.reshape(-1, self.hidden_size)
        if sources is None:
            return prefix.reshape(shape[0], shape[1], self.hidden_size)
        flat = torch.cat([prefix, sources.reshape(prefix.shape[0], -1)], dim=-1)
        return flat.reshape(shape[0], shape[1], flat.shape[-1])

    @staticmethod
    def _append(sources: Tensor | None, source: Tensor) -> Tensor:
        flat = source.reshape(-1, source.shape[-1])
        if sources is None:
            return flat.unsqueeze(1)
        return torch.cat([sources, flat.unsqueeze(1)], dim=1)

    def _owning_output(self, hidden_states: Tensor) -> Tensor:
        if self.config.pipeline_model_parallel_size > 1 and hidden_states._base is not None:
            return hidden_states.clone()
        return hidden_states

    def _write_dropout(self, bda_fn, x: Tensor) -> Tensor:
        p = self.write_dropout
        if not self.training or p <= 0.0:
            return x
        if bda_fn is None:
            return F.dropout(x, p=p, training=True)
        bda = bda_fn(self.training, self.config.bias_dropout_fusion)
        return bda((x, None), torch.zeros_like(x), p)


    def _attention(
        self,
        sublayer_input: Tensor,
        out_dtype,
        attention_mask,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        rotary_pos_cos_sin,
        attention_bias,
        inference_context,
        packed_seq_params,
        sequence_len_offset,
        shape,
    ) -> Tensor:
        ln_out = apply_module(self.input_layernorm)(
            sublayer_input.to(out_dtype).reshape(shape[0], shape[1], self.hidden_size)
        )
        attn_out = self.self_attention(
            ln_out,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        if isinstance(attn_out, tuple):
            output, bias = attn_out
            attn_out = output + bias if bias is not None else output
        return self._write_dropout(self.self_attn_bda, attn_out).reshape(-1, self.hidden_size)

    def _mlp(self, sublayer_input: Tensor, out_dtype, padding_mask, shape) -> Tensor:
        ln_out = apply_module(self.pre_mlp_layernorm)(
            sublayer_input.to(out_dtype).reshape(shape[0], shape[1], self.hidden_size)
        )
        mlp_out = apply_module(self.mlp)(ln_out, padding_mask=padding_mask)
        if isinstance(mlp_out, tuple):
            output, bias = mlp_out
            mlp_out = output + bias if bias is not None else output
        return self._write_dropout(self.mlp_bda, mlp_out).reshape(-1, self.hidden_size)

    def _plain_block(
        self,
        prefix,
        out_dtype,
        attention_mask,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        rotary_pos_cos_sin,
        attention_bias,
        inference_context,
        packed_seq_params,
        sequence_len_offset,
        padding_mask,
        shape,
    ) -> Tensor:
        attn_out = self._attention(
            prefix,
            out_dtype,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            shape,
        )
        prefix = (prefix + attn_out).to(out_dtype)
        mlp_out = self._mlp(prefix, out_dtype, padding_mask, shape)
        return (prefix + mlp_out).to(out_dtype)


    def _forward_snapshot(
        self,
        shape,
        prefix,
        sources,
        out_dtype,
        attention_mask,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        rotary_pos_cos_sin,
        attention_bias,
        inference_context,
        packed_seq_params,
        sequence_len_offset,
        padding_mask,
    ):
        ar = self.variant == "ar"
        block_closed = (self.layer_number - 1) % self.period == 0

        attn_in = self.self_attention_attn_res(prefix, self._sublayer_sources(sources))
        if block_closed and (ar or not self.per_sublayer_sources):
            sources = self._append(sources, prefix)
            if ar and self.depth_cfg.ar_reset == "zero":
                prefix = None
        attn_out = self._attention(
            attn_in,
            out_dtype,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            shape,
        )
        prefix = attn_out if prefix is None else (prefix + attn_out).to(out_dtype)
        if self.per_sublayer_sources:
            if not ar and (sources is None or sources.shape[1] == 0):
                sources = self._append(sources, prefix - attn_out)
            sources = self._append(sources, attn_out)

        mlp_in = self.mlp_attn_res(prefix, self._sublayer_sources(sources))
        mlp_out = self._mlp(mlp_in, out_dtype, padding_mask, shape)
        prefix = mlp_out if prefix is None else (prefix + mlp_out).to(out_dtype)
        if self.per_sublayer_sources:
            sources = self._append(sources, mlp_out)

        if self.is_last_layer and self.output_route and self.output_attn_res is not None:
            prefix = self.output_attn_res(prefix, self._sublayer_sources(sources))
            return prefix.reshape(shape[0], shape[1], self.hidden_size), sources
        return self._pack(prefix, sources, shape), sources

    def _sublayer_sources(self, sources: Tensor | None):
        if sources is None or sources.shape[1] == 0:
            return None
        if self.variant != "dar" or self.per_sublayer_sources:
            return sources
        if sources.shape[1] < 2:
            return None
        return sources[:, 1:, :] - sources[:, :-1, :]

    def _forward_dense(
        self,
        shape,
        prefix,
        sources,
        out_dtype,
        attention_mask,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        rotary_pos_cos_sin,
        attention_bias,
        inference_context,
        packed_seq_params,
        sequence_len_offset,
        padding_mask,
    ):
        block_out = self._plain_block(
            prefix,
            out_dtype,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_mask,
            shape,
        )
        if sources is None:
            sources = self._append(sources, prefix)
        event = self.is_dense_event()
        if event:
            values = torch.cat([sources, block_out.unsqueeze(1)], dim=1)
            if self.depth_cfg.dwa_dilation > 1 and values.shape[1] > 1:
                k = int(self.depth_cfg.dwa_dilation)
                idx = list(range((values.shape[1] - 1) % k, values.shape[1], k))
                values = values[:, idx, :]
            if self.variant == "denseformer":
                routed = self.block_dwa(values).reshape(shape[0], shape[1], self.hidden_size)
            else:
                dw = self.dense_conn(block_out.reshape(-1, self.hidden_size))
                routed = MultiwayDynamicDense.aggregate(dw, values)[0]
                if self.dense_post_norm is not None:
                    routed = block_out + self.dense_post_norm(routed).reshape(block_out.shape)
                routed = routed.reshape(shape[0], shape[1], self.hidden_size)
            sources = self._append(sources, block_out)
            if self.is_last_layer and self.output_route:
                return routed, sources
            return self._pack(routed, sources, shape), sources
        if (
            self.is_last_layer and self.output_route
        ):  # pragma: no cover - the last layer is an event
            return block_out, sources
        return self._pack(block_out, sources, shape), sources


    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset: Tensor | None = None,
        padding_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        mhc_recompute_manager=None,
        **kwargs,
    ):
        inference_context = deprecate_inference_params(
            inference_context, kwargs.get("inference_params")
        )
        if inference_context is not None:
            raise NotImplementedError(
                "depth-connection layers only implement the training/eval forward path."
            )
        if not self.enabled:
            hidden_states = super().forward(
                hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                input_ids=input_ids,
                **kwargs,
            )
            return hidden_states

        shape = hidden_states.shape[:2]
        out_dtype = hidden_states.dtype
        prefix, sources = self._unpack(hidden_states)
        args = (
            shape,
            prefix,
            sources,
            out_dtype,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            padding_mask,
        )
        if self.variant in ("ar", "dar"):
            hidden_states, sources = self._forward_snapshot(*args)
        else:
            hidden_states, sources = self._forward_dense(*args)
        return self._owning_output(hidden_states), context
