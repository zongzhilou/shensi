# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One Megatron-Core decoder layer for the snapshot-list depth connections.

``ar`` / ``dar`` / ``denseformer`` / ``mudd`` all share the *same* cross-layer
state channel as ``gdar_layer.py``: the state is packed into ``hidden_states``,
whose last dimension grows by ``H`` per appended source::

    packed = [ prefix | source_1 | ... | source_N ]        # [s, b, (1 + N) * H]

The layer unpacks ``prefix = packed[..., :H]`` and
``sources = packed[..., H:].view(T, N, H)``; the last layer returns width ``H``
again (an output read/DA/DWA is run inside it), so the block's final layernorm,
the LM head, MTP and the checkpoint format are untouched.

Two consequences, identical to GDAR: ``pipeline_model_parallel_size > 1`` needs
megatron's *dynamic* p2p shape path (the width grows with depth, so a p2p send
would otherwise be sized from ``config.hidden_size``) -- the layer switches
``config.variable_seq_lengths`` on instead of raising, see ``__init__``; as are
``fp32_residual_connection`` and ``recompute_granularity='full'``.
Context parallelism is fine *for this layer* (everything is per-token), but on a
box without Transformer Engine the stock attention rejects
``context_parallel_size > 1`` before this layer is reached
(``megatron/core/transformer/dot_product_attention.py:62``); TP > 1 works the way
``HyperConnectionModule`` does it (the connection's non-TP-aware parameters are
flagged ``sequence_parallel`` so their gradients are all-reduced).

Source bookkeeping per variant (1-based ``layer_number``, ``P = block_size``):

``ar`` (replacement read, cumulative snapshots)
    at ``(layer_number - 1) % P == 0`` the incoming stream is snapshotted into
    the source list; the accumulator then either is zeroed (reference, Kimi) or
    keeps the stream (identity anchor, ``ar_reset="keep"``).  The attn read
    happens *before* the snapshot (so it sees the previous sources only), the MLP
    read after it, exactly like the reference.

``dar`` (additive read, deltas)
    per-sublayer mode (``P == 1``) snapshots every sublayer output (the first
    entry is the running ``prefix - sublayer_out``, the reference's own
    arithmetic); block mode (``P > 1``) snapshots the stream at
    ``(layer_number - 1) % P == 0`` and the *sources* are the differences between
    consecutive snapshots -- the reference's ``partial_block - block_start``.
    The stream always accumulates (there is no reset in DAR).

``denseformer`` / ``mudd`` (dense connections, block-periodic packing)
    the *source list holds one snapshot per DWA/DA event*, not one per block:
    with ``P`` the event period the list has ``ceil(L / P)`` entries (plus the
    embedding seed), so the packed width is ``(1 + ceil(L / P)) * H`` instead of
    the reference's ``(1 + L) * H``.  For ``P = 1`` -- the published
    DenseFormer / MUDDFormer, and what the tiny runs use -- the two coincide.
    The event happens at ``layer_number % P == 0`` (and always at the last layer
    when ``output_route``); the DWA/DA at event ``m`` averages the ``m`` previous
    event snapshots plus the block it just ran, i.e. ``m + 1`` weights, so the
    published ``d(d+3)/2`` scalar count holds with ``d`` = number of events.

Initialisation of every extra module runs inside a **forked RNG**
(``_isolated_rng``, seed = ``config.seed + 104729 * layer_number``), so the
backbone is initialised bit-for-bit exactly like the equivalent plain Qwen3 --
that is what turns ``X(0) == Qwen3`` into a ``torch.equal`` instead of a
tolerance test.  The dropout of a sublayer output is taken over through the
layer's *own* ``bias_dropout_add`` callable (``_write_dropout``), because with
``bias_dropout_fusion=True`` (the default) that callable is ``@jit_fuser``
compiled and its philox consumption differs from an eager ``F.dropout``.
"""

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


def depth_knobs_from_kwargs(kwargs: dict, config: TransformerConfig | None = None) -> DepthConnectionConfig:
    """Pick ``depth_*`` knobs out of a kwargs dict (i.e. out of the spec's ``params``)."""
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
    """Exactly the submodule spec the plain (``--spec``-less) local builder gets.

    Same call, same arguments, same order as
    ``flagscale.train.megatron.gpt_builders._get_transformer_layer_spec`` for a
    dense local model, so the backbone modules are built -- and initialised, RNG
    draw by RNG draw -- identically.  (Identical to ``build_gdar_submodules``.)
    """
    return get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        config.experimental_attention_variant,
        normalization=config.normalization,
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
    )


@contextlib.contextmanager
def isolated_rng(seed: int):
    """Run a block of initialisation without advancing the global RNG stream."""
    seed = int(seed) % (2**31 - 1)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


class DepthTransformerLayer(TransformerLayer):
    """A decoder layer whose residual path is a depth connection.

    The sublayers (attention, MLP, norms, RoPE) are the stock ones; only the
    residual write and the sublayer *input* change.  ``variant`` selects which
    connection is used -- see the module docstring for the bookkeeping.
    """

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
        dualpipev_stage: int | None = None,
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
            dualpipev_stage=dualpipev_stage,
        )

        if config.pipeline_model_parallel_size > 1:
            # Same as GdarTransformerLayer (gdar_layer.py): the packed activation at a stage
            # boundary is (1 + N) * H wide, which megatron's fixed-shape p2p path cannot
            # express (it sizes the buffer from `schedules.get_tensor_shapes` ->
            # config.hidden_size) but its dynamic path can: with
            # `config.variable_seq_lengths` set, `p2p_communication._communicate` sizes the
            # receive buffer from the shape that `_communicate_shapes` sent -- the sender's
            # real `tensor.size()` (p2p_communication.py:342).  The flag has no CLI argument
            # (arguments.py::_add_network_size_args exclude list), so it is set here.
            config.variable_seq_lengths = True
        if config.fp32_residual_connection:
            raise NotImplementedError(f"depth connection {cfg.variant!r} does not support fp32_residual_connection.")
        if config.recompute_granularity == "full":
            raise NotImplementedError(
                f"depth connection {cfg.variant!r} does not support recompute_granularity='full'."
            )

        # Not layer-homogeneous: `output_attn_res` is created on the last layer only, and the
        # DenseFormer/MUDD modules only on their event layers.  megatron's default
        # dist-checkpoint mapping puts all layers of a block into one group with the layer
        # axis prepended, so a tensor that only *some* layers provide covers a subset of that
        # axis and `dist_checkpointing` rejects the pattern (Invalid access pattern for
        # ShardedTensor(key='decoder.layers.output_attn_res.q_proj', ...)).  The per-layer key
        # layout is what `hetereogenous_dist_checkpoint` selects (transformer_block.py:1025);
        # it has no CLI argument (arguments.py::_add_network_size_args exclude list) and is
        # only read by `TransformerBlock.sharded_state_dict`, so `ckpt_format: torch` is
        # unaffected.
        config.hetereogenous_dist_checkpoint = True

        self.depth_cfg = cfg
        self.variant = cfg.variant
        self.hidden_size = config.hidden_size
        self.block_size = None if cfg.block_size is None else int(cfg.block_size)
        self.enabled = self.block_size is not None
        self.period = self.block_size or 1
        self.per_sublayer_sources = self.variant in ("ar", "dar") and self.period == 1
        self.is_last_layer = self.layer_number == config.num_layers
        self.output_route = bool(cfg.output_route)
        self.write_dropout = (
            float(self.hidden_dropout) if cfg.residual_dropout is None else float(cfg.residual_dropout)
        )

        base_seed = int(getattr(config, "seed", 0)) + 104729 * self.layer_number
        with isolated_rng(base_seed):
            self._build_connections()

        if config.sequence_parallel:
            # Mirrors HyperConnectionModule: plain (non-TP-aware) layers, so their
            # gradients have to be all-reduced across the TP group.
            for module in self._connection_modules():
                for param in module.parameters():
                    param.sequence_parallel = True

    # -- construction -------------------------------------------------------

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
        """A DWA/DA event at this layer (reference ``_is_dwa_layer``)."""
        if not self.enabled:
            return False
        if self.layer_number % self.period == 0:
            return True
        return self.output_route and self.is_last_layer

    def num_sources_at_event(self) -> int:
        """Number of sources the DWA/DA at this layer reads.

        At event ``m`` (1-based) the DWA/DA reads the ``m - 1`` previous event
        snapshots, the embedding seed and the block it just ran, i.e. ``m + 1``
        values; with ``dilation = k`` only every k-th of them is kept (official
        ``DWAModules(dilation=k)``).  For ``period = 1`` (the published
        DenseFormer) this is ``layer_number + 1``, the reference's count.
        """
        extra = 1 if (self.output_route and self.is_last_layer and self.layer_number % self.period != 0) else 0
        m = self.layer_number // self.period + extra
        values = m + 1
        k = max(1, int(self.depth_cfg.dwa_dilation))
        return values if k == 1 else len(range(m % k, values, k))

    # -- packing helpers (the GDAR mechanism) -------------------------------

    def _unpack(self, hidden_states: Tensor):
        """``[s, b, W]`` -> ``(prefix [T, H], sources [T, N, H] | None)``."""
        h = self.hidden_size
        width = hidden_states.shape[-1]
        if width == h:
            return hidden_states.reshape(-1, h), None
        flat = hidden_states.reshape(-1, width)
        num_sources = (width - h) // h
        return flat[..., :h], flat[..., h:].reshape(flat.shape[0], num_sources, h)

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
        """Make the layer output a tensor that owns its storage, under pipeline parallel.

        ``schedules.deallocate_output_tensor`` asserts ``out._base is None`` and FlagScale
        hardcodes ``deallocate_pipeline_outputs=True``
        (``flagscale/train/megatron/training/argument_utils.py:306``), so it is applied to
        every stage output in the pipelined schedules
        (``forward_backward_pipelining_without_interleaving``,
        ``pipeline_parallel/schedules.py:2334``).  This layer's output is a ``cat`` +
        ``reshape`` result, i.e. a view, so it has to be materialised -- but only when
        pp > 1 and only when it really is a view.
        """
        if self.config.pipeline_model_parallel_size > 1 and hidden_states._base is not None:
            return hidden_states.clone()
        return hidden_states

    def _write_dropout(self, bda_fn, x: Tensor) -> Tensor:
        """The dropout ``bias_dropout_add`` would have applied to the sublayer output.

        The connection replaces the residual add, so it takes over that dropout as
        well -- through the *same callable* the layer would have used
        (``bda_fn(training, fused)``), because with ``bias_dropout_fusion=True``
        (the default) it is ``@jit_fuser``-compiled and consumes philox
        differently from an eager ``F.dropout``.  A zero residual makes it return
        ``0 + dropout(x) == dropout(x)`` exactly.
        """
        p = self.write_dropout
        if not self.training or p <= 0.0:
            return x
        if bda_fn is None:
            return F.dropout(x, p=p, training=True)
        bda = bda_fn(self.training, self.config.bias_dropout_fusion)
        return bda((x, None), torch.zeros_like(x), p)

    # -- sublayers ----------------------------------------------------------

    def _attention(self, sublayer_input: Tensor, out_dtype, attention_mask, rotary_pos_emb,
                   rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin, attention_bias,
                   inference_context, packed_seq_params, sequence_len_offset, shape) -> Tensor:
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

    def _plain_block(self, prefix, out_dtype, attention_mask, rotary_pos_emb, rotary_pos_cos,
                     rotary_pos_sin, rotary_pos_cos_sin, attention_bias, inference_context,
                     packed_seq_params, sequence_len_offset, padding_mask, shape) -> Tensor:
        """The stock Qwen3 residual block (``X_i = B_i(Y_{i-1})``)."""
        attn_out = self._attention(
            prefix, out_dtype, attention_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
            rotary_pos_cos_sin, attention_bias, inference_context, packed_seq_params,
            sequence_len_offset, shape,
        )
        prefix = (prefix + attn_out).to(out_dtype)
        mlp_out = self._mlp(prefix, out_dtype, padding_mask, shape)
        return (prefix + mlp_out).to(out_dtype)

    # -- variant flows ------------------------------------------------------

    def _forward_snapshot(self, shape, prefix, sources, out_dtype, attention_mask, rotary_pos_emb,
                          rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin, attention_bias,
                          inference_context, packed_seq_params, sequence_len_offset, padding_mask):
        """AR / DAR: two routed reads per layer, one write per sublayer."""
        ar = self.variant == "ar"
        block_closed = (self.layer_number - 1) % self.period == 0

        # ---- attention sublayer ------------------------------------------
        attn_in = self.self_attention_attn_res(prefix, self._sublayer_sources(sources))
        # Block-boundary snapshot: needed by BOTH variants.  AR closes a block with it; DAR in
        # block mode (period > 1, `per_sublayer_sources` False) has no other append path, so
        # gating this on `ar` alone left DAR's read an inert identity with unreachable
        # parameters (0/75 connection tensors moved in 20 steps -- caught by the memory-delta
        # fingerprint, not by any crash).  See MOE_30B_A3B.md 5.4.
        if block_closed and (ar or not self.per_sublayer_sources):
            sources = self._append(sources, prefix)
            if ar and self.depth_cfg.ar_reset == "zero":
                prefix = None  # the Kimi reference: the block restarts from its own writes
        attn_out = self._attention(
            attn_in, out_dtype, attention_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
            rotary_pos_cos_sin, attention_bias, inference_context, packed_seq_params,
            sequence_len_offset, shape,
        )
        prefix = attn_out if prefix is None else (prefix + attn_out).to(out_dtype)
        if self.per_sublayer_sources:
            if not ar and (sources is None or sources.shape[1] == 0):
                # the reference's first entry: `prefix_sum - hidden_states`
                sources = self._append(sources, prefix - attn_out)
            sources = self._append(sources, attn_out)

        # ---- MLP sublayer ------------------------------------------------
        mlp_in = self.mlp_attn_res(prefix, self._sublayer_sources(sources))
        mlp_out = self._mlp(mlp_in, out_dtype, padding_mask, shape)
        prefix = mlp_out if prefix is None else (prefix + mlp_out).to(out_dtype)
        if self.per_sublayer_sources:
            sources = self._append(sources, mlp_out)

        if self.is_last_layer and self.output_route and self.output_attn_res is not None:
            # Output routing is a pure read and happens before the block's final
            # layernorm, exactly as in the reference; returning width H keeps
            # everything downstream (final norm, lm_head, MTP) untouched.
            prefix = self.output_attn_res(prefix, self._sublayer_sources(sources))
            return prefix.reshape(shape[0], shape[1], self.hidden_size), sources
        return self._pack(prefix, sources, shape), sources

    def _sublayer_sources(self, sources: Tensor | None):
        """DAR block mode: the sources are the deltas between consecutive snapshots."""
        if sources is None or sources.shape[1] == 0:
            return None
        if self.variant != "dar" or self.per_sublayer_sources:
            return sources
        if sources.shape[1] < 2:
            return None
        return sources[:, 1:, :] - sources[:, :-1, :]

    def _forward_dense(self, shape, prefix, sources, out_dtype, attention_mask, rotary_pos_emb,
                       rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin, attention_bias,
                       inference_context, packed_seq_params, sequence_len_offset, padding_mask):
        """DenseFormer / MUDD: the stock block, then a DWA/DA over the event states."""
        block_out = self._plain_block(
            prefix, out_dtype, attention_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
            rotary_pos_cos_sin, attention_bias, inference_context, packed_seq_params,
            sequence_len_offset, padding_mask, shape,
        )
        if sources is None:
            # The first layer of a chunk seeds the state list with its own input
            # (the embedding), exactly as ``Qwen3*Model.forward`` does with
            # ``depth_states = hidden_states.unsqueeze(1)`` before the layer loop.
            sources = self._append(sources, prefix)
        event = self.is_dense_event()
        if event:
            # sources = the seed + the previous event snapshots + this block (in
            # that order, the last entry being the block we just ran -- the
            # reference's InPlaceSetSlice order and the identity target of the
            # DWA/DA).
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
        # no event: the stream just moves on to the block output
        if self.is_last_layer and self.output_route:  # pragma: no cover - the last layer is an event
            return block_out, sources
        return self._pack(block_out, sources, shape), sources

    # -- forward ------------------------------------------------------------

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
            raise NotImplementedError("depth-connection layers only implement the training/eval forward path.")
        if not self.enabled:
            # plain Qwen3 layer, byte for byte the stock forward
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
            shape, prefix, sources, out_dtype, attention_mask, rotary_pos_emb, rotary_pos_cos,
            rotary_pos_sin, rotary_pos_cos_sin, attention_bias, inference_context,
            packed_seq_params, sequence_len_offset, padding_mask,
        )
        if self.variant in ("ar", "dar"):
            hidden_states, sources = self._forward_snapshot(*args)
        else:
            hidden_states, sources = self._forward_dense(*args)
        return self._owning_output(hidden_states), context
