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
"""Megatron-Core transformer layer with Gated Delta Attention Residuals.

How the depth state crosses layer boundaries
--------------------------------------------

``models/modeling_qwen3_gdar.py`` carries two things between decoder layers: the
residual stream ``prefix`` (``[T, H]``) and the list of depth sources ``blocks``
(``[T, N, H]``).  Megatron's ``TransformerBlock`` only threads one tensor through
its layer loop, so the state has to be packed.  Three official channels exist:

1. ``hidden_states`` (this implementation).  The layer receives/returns
   ``[s, b, (1 + N) * H]``::

       packed = [prefix | block_1 | ... | block_N]  # width grows by H per write

   and unpacks to ``prefix = packed[..., :H]``,
   ``blocks = packed[..., H:].reshape(T, N, H)``.  The last layer runs the output
   routing and returns width ``H`` again, so the block's final layernorm sees
   exactly what it sees for a plain Qwen3.

2. ``context`` (the tensor ``TransformerBlock`` already threads through its layer
   loop for cross-attention).  Rejected: it is *produced inside* the layer by
   ``cross_attention``, it exists for encoder-decoder attention rather than for
   residual state, and a stage's ``context`` starts as ``None`` -- the seed (the
   embedding) would need a pp-aware hack to be injected at all.

3. the ``*_hyper_connection`` slots of 0.18.2.  Rejected as a *container*: those
   slots are bound to ``HyperConnectionModule``'s API (it returns
   ``(aggregated, h_res, h_post, residual)`` and the block expects
   ``enable_hyper_connections``, ``input_expand``/``output_contract`` and the
   ``hc_head_fn`` contraction), i.e. they implement mHC's doubly-stochastic
   *mixing* of n streams, not "a growing list of snapshots".  Reusing the slots
   means inheriting that maths (and its parameters) instead of reusing a
   container.  GDAR's connection is a *replacement* for the residual op, not a
   mixing matrix over copies of the stream.

Consequences of (1): ``pipeline_model_parallel_size > 1`` needs megatron's *dynamic*
p2p shape path, because the tensor width grows across layers and a p2p send would
otherwise be sized from ``config.hidden_size``; the layer switches
``config.variable_seq_lengths`` on for that (see ``__init__``), as do the paths that
assume ``hidden_states`` keeps width ``H`` (``fp32_residual_connection``,
``recompute_granularity='full'``).  Context parallelism is fine *for this layer*
(everything here is per-token), but on a box without Transformer Engine the stock
attention rejects ``context_parallel_size > 1`` before this layer is ever reached
(``dot_product_attention.DotProductAttention.__init__``,
``megatron/core/transformer/dot_product_attention.py:62``).  TP > 1 works the way
``HyperConnectionModule`` does it: the connection's non-TP-aware parameters are marked
``sequence_parallel`` so that their gradients are reduced.

Initialisation of the extra modules runs inside a **forked RNG**, so it does not
shift the global RNG stream: for the same ``seed`` the backbone is initialised
bit-for-bit exactly like the equivalent plain Qwen3 model.  That is what turns the
``GDAR(0) == Qwen3`` identity check into a ``torch.equal`` instead of a tolerance
test.
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

from .gdar_connection import AttentionResidual, DepthRead, GdarConfig

__all__ = ["GdarTransformerLayer", "build_gdar_submodules", "gdar_knobs_from_kwargs"]

_GDAR_FIELDS = tuple(GdarConfig.__dataclass_fields__.keys())


def gdar_knobs_from_kwargs(kwargs: dict, config: TransformerConfig | None = None) -> GdarConfig:
    """Pick ``gdar_*`` knobs out of a kwargs dict (i.e. out of the spec's ``params``)."""
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
    """Exactly the submodule spec the plain (``--spec``-less) local builder gets.

    Same call, same arguments, same order as
    ``flagscale.train.megatron.gpt_builders._get_transformer_layer_spec`` for a
    dense local model, so the backbone modules are built -- and initialised, RNG
    draw by RNG draw -- identically.
    """
    return get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,  # 0.20 的第 5 位是 fp8 槽；本配方走稠密 local 路径
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
    )


def _warn_if_block_never_closes(
    block_size, num_layers, layer_number, variant: str = "connection"
) -> None:
    """``block_size >= num_layers`` 时整模型只有一个永不闭合的块：连接是惰性的（离线检查里
    表现为 27 个张量 ``grad is None``）。在最后一层打印一次，避免静默得到一个"没接线"的 run。
    """
    if block_size and block_size >= num_layers and layer_number == num_layers:
        print(
            f"[depth-connection:{variant}] 警告：block_size={block_size} >= num_layers={num_layers}，"
            "连接永不闭合（惰性）。离线/小规模检查请用层数大于 block_size 的配置。",
            flush=True,
        )


@contextlib.contextmanager
def _isolated_rng(seed: int):
    """Run a block of initialisation without advancing the global RNG stream."""
    seed = int(seed) % (2**31 - 1)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


class GdarTransformerLayer(TransformerLayer):
    """One decoder layer whose residual connections are GDAR connections.

    Identical to ``TransformerLayer`` up to the residual path: instead of
    ``bias_dropout_add``, each sublayer output is written into the residual stream
    by a gated delta rule, and the sublayer *input* is a depth-routed read over the
    stream's snapshots.
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
            # The state is packed into hidden_states, so the activation at a stage boundary
            # is (1 + N) * H wide -- *not* config.hidden_size (see the module docstring).
            # Megatron's fixed-shape p2p path cannot express that: it sizes the receive
            # buffer from `schedules.get_tensor_shapes`, i.e. from config.hidden_size
            # (FlagScale's hyper-connection hook there, schedules.py:2093, only substitutes a
            # *constant* n-stream width).  Its dynamic path can: `p2p_communication._communicate`
            # takes the shape from `_communicate_shapes`, which sends the sender's real
            # `tensor.size()`, whenever `config.variable_seq_lengths` is set
            # (p2p_communication.py:342).  That flag has no CLI argument (it is in the
            # `exclude` list of arguments.py::_add_network_size_args), so pp > 1 turns it on
            # here, at model build, on the shared TransformerConfig the scheduler reads.
            # Nothing else in the training path consumes it except the MoE allgather token
            # dispatcher, which rejects it (transformer_config.py:2763) -- use alltoall
            # there.  Verified: flagscale_runs/parallel exp/pp2_fixed, 20 steps, 2x GPU.
            config.variable_seq_lengths = True
        if config.fp32_residual_connection:
            raise NotImplementedError("GDAR does not support fp32_residual_connection.")
        if config.recompute_granularity == "full":
            raise NotImplementedError("GDAR does not support recompute_granularity='full'.")

        # `output_attn_res` exists on the *last* layer only (and a DWA/DA module only on the
        # event layers of the dense variants), so this model is not layer-homogeneous.  With
        # `ckpt_format: torch_dist` megatron's default layer mapping would put every layer's
        # tensors into one checkpoint group with the layer axis prepended
        # (transformer_block.py:1063-1070, `sharded_prefix = layer_prefix`), and a tensor
        # that only one layer provides covers one slice of that axis and leaves the rest
        # empty -- which `dist_checkpointing` rejects:
        #   CheckpointingException: Invalid sharding pattern validation. Errors: Invalid
        #   access pattern for ShardedTensor(key='decoder.layers.output_attn_res.q_proj',
        #   global_shape=(8, 256, 256), global_offset=(7, 0, 0), ...)
        # (`hetereogenous_dist_checkpoint` = per-layer keys, transformer_block.py:1025, is
        # megatron's own switch for exactly this: "whether to use heterogenous layers in
        # distributed checkpoint").  It is in the `exclude` list of
        # arguments.py::_add_network_size_args, i.e. it has no CLI argument, so it is set
        # here; it is read only by `TransformerBlock.sharded_state_dict`, so the `torch`
        # format is unaffected.  With it on, `torch` *and* `torch_dist` save/resume work
        # (flagscale_runs/parallel exp/{dense8_gdar,torchdist,torchdist_resume}).
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

        # The extra modules are built (and initialised) under a forked RNG so the
        # backbone above keeps its exact RNG draws.  The seed is a pure function of
        # (config.seed, layer_number), so two runs -- or a run and the offline
        # comparison -- produce the same connection weights.
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
            # Mirrors HyperConnectionModule: these are plain (non-TP-aware) layers,
            # so their gradients have to be all-reduced across the TP group.
            for module in (self.self_attention_attn_res, self.mlp_attn_res):
                for param in module.parameters():
                    param.sequence_parallel = True

    # -- packing helpers ----------------------------------------------------

    def _unpack(self, hidden_states: Tensor):
        """``[s, b, W]`` -> ``(prefix [T, H], blocks [T, N, H] | None)``."""
        h = self.hidden_size
        width = hidden_states.shape[-1]
        if width == h:
            return hidden_states.reshape(-1, h), None
        flat = hidden_states.reshape(-1, width)
        num_blocks = (width - h) // h
        return flat[:, :h], flat[:, h:].reshape(flat.shape[0], num_blocks, h)

    def _pack(self, prefix: Tensor, blocks: Tensor | None, shape) -> Tensor:
        """``(prefix, blocks)`` -> ``[s, b, (1 + N) * H]``."""
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
        """Make the layer output a tensor that owns its storage, under pipeline parallel.

        ``schedules.deallocate_output_tensor`` asserts ``out._base is None`` -- "freeing a
        view is counter-productive" -- and FlagScale's
        ``core_transformer_config_from_args`` hardcodes ``deallocate_pipeline_outputs=True``
        (``flagscale/train/megatron/training/argument_utils.py:306``), so every *stage
        output* is deallocated in the pipelined schedules
        (``forward_backward_pipelining_without_interleaving``,
        ``pipeline_parallel/schedules.py:2334``).  This layer's output is a ``cat`` +
        ``reshape`` result, i.e. a view (``_base`` is the flattened tensor), so it has to be
        materialised.  Only done when pp > 1 (the deallocation path) and only when the
        tensor really is a view, so the pp = 1 runs keep their exact tensors.
        """
        if self.config.pipeline_model_parallel_size > 1 and hidden_states._base is not None:
            return hidden_states.clone()
        return hidden_states

    def _write_dropout(self, bda_fn, x: Tensor) -> Tensor:
        """The dropout that ``bias_dropout_add`` would have applied to ``x``.

        The connection *replaces* the residual add, so it takes over the dropout of
        the sublayer output as well -- and it has to take it over *through the very
        same callable* the layer would have used (``bda_fn( training, fused )``,
        exactly as ``TransformerLayer._forward_attention`` builds it), because with
        ``bias_dropout_fusion=True`` -- the default, and what this port is tested
        with -- the bda is ``@jit_fuser``-compiled and its philox consumption
        differs from an eager ``F.dropout``.  Handing it a zero residual makes it
        return ``0 + dropout(x) == dropout(x)`` exactly.
        """
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
        # The connection computes in fp32 (as the reference does); the sublayer sees
        # the model dtype again, exactly like the HF file's RMSNorm(bf16) would.
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
        # Round back to the model dtype after every write, exactly like the
        # ``bias_dropout_add`` this replaces (a single bf16 add of the sublayer
        # output into the stream).  Keeping the stream in fp32 between writes
        # would differ from the plain model by one ulp now and then.
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
            raise NotImplementedError("GDAR layers only implement the training/eval forward path.")

        shape = hidden_states.shape[:2]
        out_dtype = hidden_states.dtype
        prefix, blocks = self._unpack(hidden_states)

        # Close the previous source block.  With block_size == 1 every sublayer
        # write is a source; otherwise the stream is snapshotted every
        # ``block_size`` layers, seeded by the embedding (the reference's rule).
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
            # The output routing is a pure read (no gates, hence no dead
            # parameters) and must happen before the block's final layernorm,
            # exactly as in the reference.  Returning width H keeps everything
            # downstream (final norm, lm_head, MTP) untouched.
            prefix = self.output_attn_res(prefix, blocks)
            return self._owning_output(
                prefix.reshape(shape[0], shape[1], self.hidden_size)
            ), context

        return self._owning_output(self._pack(prefix, blocks, shape)), context
