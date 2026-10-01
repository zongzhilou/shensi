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
"""HC / mHC inside FlagScale-Megatron by **reusing the official 0.18.2 module**.

The official mechanism lives in ``megatron/core/transformer/hyper_connection.py``
(``HyperConnectionModule``: the ``n^2 + 2n`` projection, ``H_pre``/``H_post``,
the Sinkhorn-Knopp doubly-stochastic projection of ``H_res``, ``input_expand`` /
``output_contract`` / ``learned_output_contract``), is driven by the official
``TransformerConfig`` fields (``num_residual_streams``,
``mhc_sinkhorn_iterations``, ``mhc_init_gating_factor``, ``use_fused_mhc``) and is
slotted in through ``TransformerLayerSubmodules.self_attention_hyper_connection``
/ ``mlp_hyper_connection`` + ``HyperConnectionTransformerLayer``.

Nothing here re-implements any of that.  Two things are added:

1. a **spec assembly layer**: FlagScale's ``_get_transformer_layer_spec`` never
   passes ``enable_hyper_connection`` to ``get_gpt_layer_local_spec``, so even
   with ``--enable-hyper-connections`` the layers would come back without the
   hyper-connection slots.  ``HcTransformerLayer`` builds the *official*
   submodules with the official flag on (same builder, same arguments as the
   plain path, plus ``enable_hyper_connection=True``) and lets
   ``HyperConnectionTransformerLayer`` wire them, and it runs the official
   block-level ``input_expand`` / output contraction at the chunk boundaries --
   the two calls ``TransformerBlock`` makes when
   ``config.enable_hyper_connections`` is on.  Doing it here (instead of setting
   the flag) is what keeps the *identity anchor* possible: the block always
   contracts with the learned head (``torch.randn`` init), which is not the
   standard residual, while the anchor needs ``output_contract`` (the exact mean).
2. ``HcHyperConnection``: a thin subclass that adds the two switches the identity
   anchor needs -- a softmax read (an exact convex combination, ``1/n`` at zero
   logits, where the official ``sigmoid + 1e-6`` cannot be exactly ``1/n``) and
   ``sinkhorn_eps = 0`` (Sinkhorn's uniform fixed point is only exact at
   ``eps = 0``; the released default of ``1e-6`` moves it by ``~n * eps``).
   Everything else -- the projection, the Sinkhorn iterations, ``H_post``, the
   residual mix, the expansion -- is the official code path.

With the anchor in place, ``HC(0)`` / ``mHC(0)`` is the plain Megatron residual
``x + f(x)`` **bit-exactly**: the streams start as ``n`` replicas of ``x``, the
read is ``1/n`` (exact for ``n`` a power of two), ``H_post = 2*sigmoid(0) = 1``,
``H_res`` is the uniform doubly-stochastic matrix (``Sinkhorn(0)`` at ``eps = 0``,
whose action on identical streams is exactly ``x``), and the contraction is the
exact mean.  The published initialisation (``mhc_init_gating_factor = 0.01``,
zero biases, xavier projection, the learned head) is *not* the standard residual;
the deviation is measured in ``flagscale_runs/depth_check.py``.

``_force_identity`` has the same semantics as in ``modeling_qwen3_hc.py`` /
``modeling_qwen3_mhc.py``: the connection returns the identity coefficients
(stream 0, ``H_res = I``, ``H_post = 1``) regardless of the learned weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    learned_output_contract,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayerSubmodules,
)

from .depth_layer import isolated_rng

__all__ = ["HcHyperConnection", "HcTransformerLayer", "HcConfig", "build_hc_submodules", "hc_knobs_from_kwargs"]


@dataclass
class HcConfig:
    """Knobs of the HC/mHC spec (all of them optional; ``None`` = official default)."""

    #: ``"hc"`` (no manifold projection, the unprojected HC of arXiv:2409.19606)
    #: or ``"mhc"`` (Sinkhorn-Knopp ``H_res``, the released module).
    family: str = "mhc"
    #: number of residual streams ``n`` (official ``num_residual_streams``).
    num_streams: int | None = None
    #: official ``mhc_sinkhorn_iterations``.
    sinkhorn_iterations: int | None = None
    #: official ``mhc_init_gating_factor`` (the ``alpha`` of Eq. 5).
    gating_factor: float | None = None
    #: ``"official"`` = the released initialisation (this is what "published"
    #: means); ``"identity"`` = the exact plain-residual anchor.
    init: str = "identity"
    #: ``"simplex"`` (softmax read; exact ``1/n``) or ``"sigmoid"`` (the official
    #: ``sigmoid + 1e-6``, what the released code does).
    read: str = "simplex"
    #: ``"mean"`` (official ``output_contract``) or ``"learned"`` (official
    #: ``learned_output_contract`` with the block's ``hc_head_*`` init).
    contract: str = "mean"
    #: chunk size in layers: the n-stream is expanded at the chunk entry and
    #: contracted at the chunk exit.  ``None`` = the whole stack, i.e. the
    #: official layout (one n-stream across the decoder).
    chunk_size: int | None = None
    #: run-time guard, same semantics as ``_force_identity`` in the HF files.
    force_identity: bool = False

    def validated(self) -> HcConfig:
        if self.family not in ("hc", "mhc"):
            raise ValueError(f"family must be 'hc' or 'mhc', got {self.family!r}")
        if self.init not in ("identity", "official"):
            raise ValueError(f"init must be 'identity' or 'official', got {self.init!r}")
        if self.read not in ("simplex", "sigmoid"):
            raise ValueError(f"read must be 'simplex' or 'sigmoid', got {self.read!r}")
        if self.contract not in ("mean", "learned"):
            raise ValueError(f"contract must be 'mean' or 'learned', got {self.contract!r}")
        return self

    @property
    def manifold(self) -> str:
        return "doubly_stochastic" if self.family == "mhc" else "none"

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__.keys())


def hc_knobs_from_kwargs(kwargs: dict, config: TransformerConfig | None = None) -> HcConfig:
    """Pick ``hc_*`` knobs out of a kwargs dict (i.e. out of the spec's ``params``)."""
    values = {}
    for key, value in kwargs.items():
        if key.startswith("hc_"):
            name = key[len("hc_") :]
            if name not in HcConfig.field_names():
                raise TypeError(f"unknown HC knob {key!r}; valid knobs: {HcConfig.field_names()}")
            values[name] = value
    return HcConfig(**values).validated()


def build_hc_submodules(config: TransformerConfig) -> TransformerLayerSubmodules:
    """The plain local submodules with the official hyper-connection slots opened.

    Same call and same arguments as ``build_depth_submodules`` /
    ``flagscale.train.megatron.gpt_builders._get_transformer_layer_spec``; the
    only difference is ``enable_hyper_connection=True``, which swaps
    ``self_attention_hyper_connection`` / ``mlp_hyper_connection`` from
    ``IdentityOp`` to ``HcHyperConnection`` (the official module, subclassed).
    ``HyperConnectionTransformerLayer`` asserts both slots are present.
    """
    submodules = get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        config.experimental_attention_variant,
        normalization=config.normalization,
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
        enable_hyper_connection=True,
    )
    # The official builder puts the official class in both slots; swap in the
    # subclass that carries the identity anchor (everything it does is inherited).
    submodules.self_attention_hyper_connection = HcHyperConnection
    submodules.mlp_hyper_connection = HcHyperConnection
    return submodules


class HcHyperConnection(HyperConnectionModule):
    """``HyperConnectionModule`` + the two switches the identity anchor needs.

    See the module docstring: the softmax read and ``sinkhorn_eps = 0`` are the
    only differences from the released code path, both active only when
    ``init == "identity"``.  The knobs arrive through the ``TransformerConfig``
    (the official module reads the official fields), set by ``HcTransformerLayer``
    from the spec params.
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        self.read_mode = getattr(config, "hc_read_mode", "simplex")
        self.identity = getattr(config, "hc_identity", False)
        self.force_identity = getattr(config, "hc_force_identity", False)
        self.manifold = getattr(config, "hc_manifold", "doubly_stochastic")
        super().__init__(config, layer_number)

        if self.manifold == "none":
            # HC has no manifold projection and its released init leaves the H_res
            # logits at zero: an unprojected zero mixing matrix deletes the residual
            # stream outright, so anchor it at I -- the HC paper's static A_r = I
            # (Eq. 14), and exactly what ``modeling_qwen3_hc.py`` does for this case.
            # mHC keeps the zero logits, whose Sinkhorn fixed point is the uniform
            # doubly-stochastic matrix (a convex combination of the streams).
            with torch.no_grad():
                self.bias[2 * self.n :].view(self.n, self.n).copy_(
                    torch.eye(self.n, dtype=self.bias.dtype)
                )
        if self.identity:
            with torch.no_grad():
                self.alpha_pre.zero_()
                self.alpha_post.zero_()
                self.alpha_res.zero_()
                self.bias.zero_()
                if self.manifold == "none":
                    self.bias[2 * self.n :].view(self.n, self.n).copy_(
                        torch.eye(self.n, dtype=self.bias.dtype)
                    )
            # Sinkhorn's uniform fixed point is exact only at eps = 0 (the released
            # 1e-6 moves it by ~n * eps); safe because softmax rows already sum to 1.
            self.sinkhorn_eps = 0.0

    def _compute_h(self, proj: Tensor, r: Tensor):
        """Official ``_compute_h``, with the read optionally a softmax.

        ``h_pre = softmax(logits)`` is a convex combination by construction and is
        exactly ``1/n`` when the ``n`` logits coincide (the identity anchor);
        ``sigmoid(logits) + 1e-6`` (the official read) can never hit ``1/n``
        exactly.  ``H_post`` and the ``H_res`` logits are untouched, and the
        Sinkhorn projection itself is still the official one (``compute_mappings``
        calls ``self._sinkhorn_op`` afterwards).
        """
        if not (self.identity and self.read_mode == "simplex"):
            return super()._compute_h(proj, r)
        alpha_ = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )
        h = r * proj * alpha_ + self.bias
        h_pre = h[..., : self.n].softmax(dim=-1)
        h_post = h[..., self.n : 2 * self.n].sigmoid() * 2
        h_res = h[..., 2 * self.n :]
        return h_pre, h_post, h_res

    def forward(self, hidden_states: Tensor, mhc_recompute_manager=None):
        if self.force_identity:
            # The exact standard-residual slice (``_force_identity`` semantics):
            # read stream 0, keep H_res = I and H_post = 1.
            s, b, _ = hidden_states.shape
            n, C = self.n, self.hidden_size
            streams = hidden_states.view(s, b, n, C)
            eye = torch.eye(n, dtype=hidden_states.dtype, device=hidden_states.device)
            eye = eye.expand(s, b, n, n).contiguous()
            ones = torch.ones(s, b, n, dtype=hidden_states.dtype, device=hidden_states.device)
            return streams[..., 0, :].contiguous(), eye, ones, hidden_states
        return super().forward(hidden_states, mhc_recompute_manager=mhc_recompute_manager)


class HcTransformerLayer(HyperConnectionTransformerLayer):
    """The official mHC layer + the block-level expand/contract it relies on.

    ``TransformerBlock`` runs ``input_expand`` at the entry and
    ``learned_output_contract`` at the exit only when
    ``config.enable_hyper_connections`` is set.  This layer does the same two
    calls itself (the official static methods), so the connection works through
    ``--spec`` alone and the contraction can be the exact mean for the identity
    anchor.
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
        name: str | None = None,
        dualpipev_stage: int | None = None,
        **kwargs,
    ):
        cfg = hc_knobs_from_kwargs(kwargs, config)

        # The official module reads the official TransformerConfig fields; the spec
        # params (when given) write them here so the task YAML stays identical to
        # the plain-Qwen3 one apart from --spec.
        if cfg.num_streams is not None:
            config.num_residual_streams = int(cfg.num_streams)
        if cfg.sinkhorn_iterations is not None:
            config.mhc_sinkhorn_iterations = int(cfg.sinkhorn_iterations)
        if cfg.gating_factor is not None:
            config.mhc_init_gating_factor = float(cfg.gating_factor)
        config.hc_read_mode = cfg.read
        config.hc_identity = cfg.init == "identity"
        config.hc_force_identity = bool(cfg.force_identity)
        config.hc_manifold = cfg.manifold

        if submodules is None:
            submodules = build_hc_submodules(config)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            name=name,
            dualpipev_stage=dualpipev_stage,
        )

        if config.pipeline_model_parallel_size > 1:
            raise NotImplementedError(
                "HC/mHC expands the stream to num_residual_streams * H from the chunk entry to its "
                "exit; pipeline parallel would need a matching p2p shape (the official path handles "
                "it through config.enable_hyper_connections, which this spec cannot set). Use "
                "pipeline_model_parallel_size=1."
            )
        if config.recompute_granularity == "full":
            raise NotImplementedError("HC/mHC does not support recompute_granularity='full'.")

        self.hc_cfg = cfg
        self.num_streams = int(config.num_residual_streams)
        self.hidden_size = config.hidden_size
        self.chunk_size = int(cfg.chunk_size) if cfg.chunk_size else int(config.num_layers)
        self.is_chunk_entry = (self.layer_number - 1) % self.chunk_size == 0
        self.is_chunk_exit = self.layer_number % self.chunk_size == 0 or self.layer_number == config.num_layers
        self.contract_mode = cfg.contract
        self.head_fn = None
        self.head_base = None
        self.head_scale = None
        if self.is_chunk_exit and self.contract_mode == "learned":
            # Mirror TransformerBlock.__init__ exactly (same shapes, same init),
            # inside the forked RNG so the backbone keeps its RNG draws.
            with isolated_rng(int(getattr(config, "seed", 0)) + 104729 * self.layer_number + 7):
                n, C = self.num_streams, self.hidden_size
                self.head_fn = torch.nn.Parameter(torch.randn(n, C * n))
                self.head_base = torch.nn.Parameter(torch.zeros(n))
                self.head_scale = torch.nn.Parameter(torch.ones(1))
                torch.nn.init.xavier_uniform_(self.head_fn)

    @property
    def connection_modules(self):
        return (self.self_attention_hyper_connection, self.mlp_hyper_connection)

    def forward(self, hidden_states, *args, **kwargs):
        """Chunk entry: expand 1-stream -> n-stream; chunk exit: contract back."""
        C = self.hidden_size
        if self.is_chunk_entry and hidden_states.shape[-1] == C:
            hidden_states = HyperConnectionModule.input_expand(hidden_states, self.num_streams)
        output, context = super().forward(hidden_states, *args, **kwargs)
        if self.is_chunk_exit and output.shape[-1] == self.num_streams * C:
            if self.contract_mode == "learned":
                output = learned_output_contract(
                    output,
                    self.head_fn,
                    self.head_base,
                    self.head_scale,
                    self.num_streams,
                    self.config.layernorm_epsilon,
                )
            else:
                output = HyperConnectionModule.output_contract(output, self.num_streams)
        return output, context
