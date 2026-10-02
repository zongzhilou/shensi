"""GDAR 的 mcore Transformer 层：在子层间接入连接与块内不动点迭代。"""


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

from .gdar_connection import AttentionResidual, DepthRead, GdarConfig

__all__ = ["GdarTransformerLayer", "build_gdar_submodules", "gdar_knobs_from_kwargs"]

_GDAR_FIELDS = tuple(GdarConfig.__dataclass_fields__.keys())


def gdar_knobs_from_kwargs(kwargs: dict, config: TransformerConfig | None = None) -> GdarConfig:
    """把 TransformerConfig 上的连接旋钮解析成 GdarConfig。"""
    values = {}
    for key, value in kwargs.items():
        if key.startswith("gdar_"):
            name = key[len("gdar_") :]
            if name not in _GDAR_FIELDS:
                raise TypeError(f"unknown GDAR knob {key!r}; valid knobs: {_GDAR_FIELDS}")
            values[name] = value
    if config is not None:
        values.setdefault("init_std", float(getattr(config, "init_method_std", 0.02)))
    return GdarConfig(**values).validated()


def build_gdar_submodules(config: TransformerConfig) -> TransformerLayerSubmodules:
    """由层规格与配置生成子模块集合（连接与读数模块）。"""
    return get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
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
def _isolated_rng(seed: int):
    seed = int(seed) % (2**31 - 1)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


class GdarTransformerLayer(TransformerLayer):

    """GDAR 的 Transformer 层：按块在子层间插入连接，并做块内不动点迭代。"""
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
        cfg = gdar_knobs_from_kwargs(kwargs, config)
        if submodules is None:
            submodules = build_gdar_submodules(config)
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
            raise NotImplementedError("GDAR does not support fp32_residual_connection.")
        if config.recompute_granularity == "full":
            raise NotImplementedError("GDAR does not support recompute_granularity='full'.")

        config.hetereogenous_dist_checkpoint = True

        self.hidden_size = config.hidden_size
        self.block_size = cfg.block_size
        _warn_if_block_never_closes(self.block_size, config.num_layers, self.layer_number)
        self.per_sublayer_sources = self.block_size == 1
        self.is_last_layer = self.layer_number == config.num_layers
        self.write_dropout = (
            float(self.hidden_dropout)
            if cfg.residual_dropout is None
            else float(cfg.residual_dropout)
        )

        base_seed = int(getattr(config, "seed", 0)) + 104729 * self.layer_number
        with _isolated_rng(base_seed):
            self.self_attention_attn_res = AttentionResidual(
                self.hidden_size, cfg, eps=config.layernorm_epsilon
            )
            self.mlp_attn_res = AttentionResidual(
                self.hidden_size, cfg, eps=config.layernorm_epsilon
            )
            if self.is_last_layer:
                self.output_attn_res = DepthRead(
                    self.hidden_size, cfg, eps=config.layernorm_epsilon
                )
        self.gdar_cfg = cfg

        if config.sequence_parallel:
            for module in (self.self_attention_attn_res, self.mlp_attn_res):
                for param in module.parameters():
                    param.sequence_parallel = True


    def _unpack(self, hidden_states: Tensor):
        h = self.hidden_size
        width = hidden_states.shape[-1]
        if width == h:
            return hidden_states.reshape(-1, h), None
        flat = hidden_states.reshape(-1, width)
        num_blocks = (width - h) // h
        return flat[:, :h], flat[:, h:].reshape(flat.shape[0], num_blocks, h)

    def _pack(self, prefix: Tensor, blocks: Tensor | None, shape) -> Tensor:
        if blocks is None:
            return prefix.reshape(shape[0], shape[1], self.hidden_size)
        flat = torch.cat([prefix, blocks.reshape(prefix.shape[0], -1)], dim=-1)
        return flat.reshape(shape[0], shape[1], flat.shape[-1])

    @staticmethod
    def _append(blocks: Tensor | None, source: Tensor) -> Tensor:
        flat = source.reshape(-1, source.shape[-1])
        if blocks is None:
            return flat.unsqueeze(1)
        return torch.cat([blocks, flat.unsqueeze(1)], dim=1)

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

    def _attention_sublayer(
        self,
        shape,
        prefix,
        blocks,
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
    ):
        routed, _ = self.self_attention_attn_res.read(prefix, blocks)
        ln_out = apply_module(self.input_layernorm)(
            routed.to(out_dtype).reshape(shape[0], shape[1], self.hidden_size)
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
        written = self._write_dropout(self.self_attn_bda, attn_out)
        updated, gates = self.self_attention_attn_res.update(
            prefix, written.reshape(-1, self.hidden_size)
        )
        return updated.to(out_dtype), gates, attn_out

    def _mlp_sublayer(self, shape, prefix, blocks, out_dtype, padding_mask):
        routed, _ = self.mlp_attn_res.read(prefix, blocks)
        ln_out = apply_module(self.pre_mlp_layernorm)(
            routed.to(out_dtype).reshape(shape[0], shape[1], self.hidden_size)
        )
        mlp_out = apply_module(self.mlp)(ln_out, padding_mask=padding_mask)
        if isinstance(mlp_out, tuple):
            output, bias = mlp_out
            mlp_out = output + bias if bias is not None else output
        written = self._write_dropout(self.mlp_bda, mlp_out)
        updated, gates = self.mlp_attn_res.update(prefix, written.reshape(-1, self.hidden_size))
        return updated.to(out_dtype), gates, mlp_out


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
            raise NotImplementedError("GDAR layers only implement the training/eval forward path.")

        shape = hidden_states.shape[:2]
        out_dtype = hidden_states.dtype
        prefix, blocks = self._unpack(hidden_states)

        if not self.per_sublayer_sources and (self.layer_number - 1) % self.block_size == 0:
            blocks = self._append(blocks, prefix)
        elif self.per_sublayer_sources and blocks is None:
            blocks = self._append(blocks, prefix)

        prefix, _, attn_out = self._attention_sublayer(
            shape,
            prefix,
            blocks,
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
        )
        if self.per_sublayer_sources:
            blocks = self._append(blocks, attn_out)

        prefix, _, mlp_out = self._mlp_sublayer(shape, prefix, blocks, out_dtype, padding_mask)
        if self.per_sublayer_sources:
            blocks = self._append(blocks, mlp_out)

        if self.is_last_layer:
            prefix = self.output_attn_res(prefix, blocks)
            return self._owning_output(
                prefix.reshape(shape[0], shape[1], self.hidden_size)
            ), context

        return self._owning_output(self._pack(prefix, blocks, shape)), context
