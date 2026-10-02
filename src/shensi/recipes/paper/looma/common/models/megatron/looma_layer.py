"""Looma 的 mcore Transformer 层：块结构与迭代求解的接线。"""


from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.attention import apply_rotary_pos_emb
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params

from .looma_connection import LoomaAttentionResidual, LoomaConfig, looma_knobs_from_kwargs, solve_block

__all__ = ["LoomaTransformerLayer", "build_looma_submodules", "looma_knobs_from_kwargs"]


def build_looma_submodules(config: TransformerConfig) -> TransformerLayerSubmodules:
    """由层规格与配置生成子模块集合。"""
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


@contextlib.contextmanager
def _isolated_rng(seed: int):
    seed = int(seed) % (2**31 - 1)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _stand_in_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=16,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
    )




PLACEHOLDER_SUBMODULES = build_looma_submodules(_stand_in_config())


class LoomaTransformerLayer(TransformerLayer):

    """Looma 的 Transformer 层：按块组织子层并接线连接与求解器。"""
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
        cfg = looma_knobs_from_kwargs(kwargs, config)
        if (
            submodules is None
            or callable(submodules)
            or submodules is PLACEHOLDER_SUBMODULES
        ):


            submodules = build_looma_submodules(config)
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



        self.is_plain_layer = bool(is_mtp_layer)
        if config.pipeline_model_parallel_size > 1:




            config.variable_seq_lengths = True




        config.hetereogenous_dist_checkpoint = True

        self.hidden_size = config.hidden_size
        self.looma_cfg: LoomaConfig = cfg
        self.is_last_layer = self.layer_number == config.num_layers
        self.write_dropout = (
            float(self.hidden_dropout) if cfg.residual_dropout is None else float(cfg.residual_dropout)
        )

        self.attention_output_gate = bool(self.config.attention_output_gate)
        if self.config.fused_single_qkv_rope:


            print(
                "[depth-connection:looma] 注意：fused_single_qkv_rope 在块循环里不适用（后续迭代"
                "只移动 query），本层走拆分路径（数值等价，只是少了那次融合）。",
                flush=True,
            )



        self.recompute = bool(config.recompute_granularity) and not self.is_plain_layer


        self.fp32_residual = bool(config.fp32_residual_connection)


        base_seed = int(getattr(config, "seed", 0)) + 104729 * self.layer_number
        with _isolated_rng(base_seed):
            self.self_attention_attn_res = LoomaAttentionResidual(
                self.hidden_size, cfg, eps=config.layernorm_epsilon
            )
            self.mlp_attn_res = LoomaAttentionResidual(
                self.hidden_size, cfg, eps=config.layernorm_epsilon
            )
            if self.is_last_layer and cfg.output_route:
                self.output_attn_res = LoomaAttentionResidual(
                    self.hidden_size, cfg, eps=config.layernorm_epsilon
                )

        if config.sequence_parallel:


            for module in (self.self_attention_attn_res, self.mlp_attn_res):
                for param in module.parameters():
                    param.sequence_parallel = True

    def _unpack(self, hidden_states: Tensor):
        h = self.hidden_size
        width = hidden_states.shape[-1]
        flat = hidden_states.reshape(-1, width)
        if self.fp32_residual and flat.dtype != torch.float32:
            flat = flat.float()
        if width == h:
            return flat, flat, None
        stream, prefix = flat[:, :h], flat[:, h : 2 * h]
        rows = flat[:, 2 * h :].reshape(flat.shape[0], (width - 2 * h) // h, h)
        return stream, prefix, rows

    def _pack(self, stream: Tensor, prefix: Tensor, rows: Tensor | None, shape) -> Tensor:
        parts = [stream, prefix] if rows is None else [stream, prefix, rows.reshape(stream.shape[0], -1)]
        flat = torch.cat(parts, dim=-1)
        return flat.reshape(shape[0], shape[1], flat.shape[-1])

    @staticmethod
    def _append(rows: Tensor | None, source: Tensor) -> Tensor:
        flat = source.reshape(-1, source.shape[-1])
        if rows is None:
            return flat.unsqueeze(1)
        return torch.cat([rows, flat.unsqueeze(1)], dim=1)

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
        hidden_states: Tensor,
        frozen_kv: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor | None,
        rotary_pos_emb,
        attention_bias: Tensor | None,
        packed_seq_params,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn = self.self_attention
        gate = None
        if self.attention_output_gate:

            query, gate, key, value = attn.get_query_key_value_tensors(
                hidden_states, split_qkv=True, output_gate=True
            )
        else:
            query, key, value = attn.get_query_key_value_tensors(hidden_states, split_qkv=True)

        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        cu_seqlens_q = cu_seqlens_kv = None
        rope_max_seqlen = None
        if rotary_pos_emb is not None and thd:
            q_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
            kv_padded = getattr(packed_seq_params, "cu_seqlens_kv_padded", None)
            cu_seqlens_q = q_padded if q_padded is not None else packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = kv_padded if kv_padded is not None else packed_seq_params.cu_seqlens_kv
            max_q, max_kv = packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv
            rope_max_seqlen = max(max_q, max_kv) if (max_q is not None and max_kv is not None) else None
        if thd:
            query, key, value = query.squeeze(1), key.squeeze(1), value.squeeze(1)

        if frozen_kv is None and rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
                    mscale=attn._yarn_concentration_factor, max_seqlen=rope_max_seqlen,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
                    mscale=attn._yarn_concentration_factor, max_seqlen=rope_max_seqlen,
                )
            frozen_kv = (key, value)
        elif frozen_kv is not None:

            key, value = frozen_kv
            if rotary_pos_emb is not None and rotary_pos_emb[0] is not None:
                query = apply_rotary_pos_emb(
                    query, rotary_pos_emb[0], config=self.config, cu_seqlens=cu_seqlens_q,
                    mscale=attn._yarn_concentration_factor, max_seqlen=rope_max_seqlen,
                )

        core_attn_out = apply_module(attn.core_attention)(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        if thd:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        if gate is not None:

            core_attn_out = attn._apply_output_gate(core_attn_out, gate)
        output, bias = attn.forward_post_core_attn(core_attn_out)
        if bias is not None:
            output = output + bias
        return output, frozen_kv

    def _block_step(
        self,
        stream: Tensor,
        prefix: Tensor,
        rows: Tensor | None,
        out_dtype,
        frozen_kv,
        attention_mask,
        rotary_pos_emb,
        attention_bias,
        packed_seq_params,
        shape,
        padding_mask,
    ):


        routed = self.self_attention_attn_res(
            prefix, stream - prefix, rows, output_norm_weight=self.input_layernorm.weight
        )
        attn_out, frozen_kv = self._attention(

            routed.to(self._sublayer_dtype()).reshape(shape[0], shape[1], self.hidden_size),
            frozen_kv,
            attention_mask,
            rotary_pos_emb,
            attention_bias,
            packed_seq_params,
        )
        written = self._write_dropout(self.self_attn_bda, attn_out)
        stream = routed + written.reshape(-1, self.hidden_size)
        prefix = stream

        routed = self.mlp_attn_res(
            prefix, prefix, rows, output_norm_weight=self.pre_mlp_layernorm.weight
        )
        mlp_out = apply_module(self.mlp)(
            routed.to(self._sublayer_dtype()).reshape(shape[0], shape[1], self.hidden_size),
            padding_mask=padding_mask,
        )
        if isinstance(mlp_out, tuple):
            output, bias = mlp_out
            mlp_out = output + bias if bias is not None else output
        written = self._write_dropout(self.mlp_bda, mlp_out)
        stream = routed + written.reshape(-1, self.hidden_size)
        prefix = prefix + stream
        return stream, prefix, frozen_kv

    def _sublayer_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _step(self, stream: Tensor, prefix: Tensor, **common):
        if not (self.training and self.recompute):
            return self._block_step(stream, prefix, **common)
        return checkpoint(self._block_step, stream, prefix, use_reentrant=False, **common)

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
        if self.is_plain_layer:

            return super().forward(
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

        inference_context = deprecate_inference_params(
            inference_context, kwargs.get("inference_params")
        )
        if inference_context is not None:
            raise NotImplementedError("Looma layers only implement the training/eval forward path.")
        if rotary_pos_cos is not None or rotary_pos_sin is not None or rotary_pos_cos_sin is not None:
            raise NotImplementedError("Looma does not implement the flash-decode RoPE path.")

        shape = hidden_states.shape[:2]
        out_dtype = hidden_states.dtype
        stream, prefix, rows = self._unpack(hidden_states)



        rows = self._append(rows, prefix)

        common = dict(
            rows=rows,
            out_dtype=out_dtype,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            shape=shape,
            padding_mask=padding_mask,
        )


        stream, prefix, frozen_kv = self._step(stream, prefix, frozen_kv=None, **common)

        def block_map(next_stream, next_prefix):
            out = self._step(next_stream, next_prefix, frozen_kv=frozen_kv, **common)
            return out[0], out[1]

        cfg = self.looma_cfg
        stream, prefix = solve_block(
            block_map,
            [stream, prefix],
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            stop_mode=cfg.stop_mode,
            tau=cfg.tau,
            grad_steps=cfg.grad_steps,
        )

        if self.is_last_layer:

            out = self.output_attn_res(prefix, stream, rows) if cfg.output_route else stream
            return self._owning_output(out.reshape(shape[0], shape[1], self.hidden_size)), context

        return self._owning_output(self._pack(stream, prefix, rows, shape)), context
