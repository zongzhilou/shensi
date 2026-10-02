"""HC（超连接）的 mcore 层实现。"""


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

__all__ = [
    "HcHyperConnection",
    "HcTransformerLayer",
    "HcConfig",
    "build_hc_submodules",
    "hc_knobs_from_kwargs",
]


@dataclass
class HcConfig:

    """HC（超连接）的开关集合。"""
    family: str = "mhc"
    num_streams: int | None = None
    sinkhorn_iterations: int | None = None
    gating_factor: float | None = None
    init: str = "identity"
    read: str = "simplex"
    contract: str = "mean"
    chunk_size: int | None = None
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
    values = {}
    for key, value in kwargs.items():
        if key.startswith("hc_"):
            name = key[len("hc_") :]
            if name not in HcConfig.field_names():
                raise TypeError(f"unknown HC knob {key!r}; valid knobs: {HcConfig.field_names()}")
            values[name] = value
    return HcConfig(**values).validated()


def build_hc_submodules(config: TransformerConfig) -> TransformerLayerSubmodules:
    submodules = get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
    )
    submodules.self_attention_hyper_connection = HcHyperConnection
    submodules.mlp_hyper_connection = HcHyperConnection
    return submodules


class HcHyperConnection(HyperConnectionModule):

    """超连接本体：读 / 写权重经 Sinkhorn 约束。"""
    def __init__(self, config: TransformerConfig, layer_number: int):
        self.read_mode = getattr(config, "hc_read_mode", "simplex")
        self.identity = getattr(config, "hc_identity", False)
        self.force_identity = getattr(config, "hc_force_identity", False)
        self.manifold = getattr(config, "hc_manifold", "doubly_stochastic")
        super().__init__(config, layer_number)

        if self.manifold == "none":
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
            self.sinkhorn_eps = 0.0

    def _compute_h(self, proj: Tensor, r: Tensor):
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
            s, b, _ = hidden_states.shape
            n, C = self.n, self.hidden_size
            streams = hidden_states.view(s, b, n, C)
            eye = torch.eye(n, dtype=hidden_states.dtype, device=hidden_states.device)
            eye = eye.expand(s, b, n, n).contiguous()
            ones = torch.ones(s, b, n, dtype=hidden_states.dtype, device=hidden_states.device)
            return streams[..., 0, :].contiguous(), eye, ones, hidden_states
        return super().forward(hidden_states, mhc_recompute_manager=mhc_recompute_manager)


class HcTransformerLayer(HyperConnectionTransformerLayer):

    """HC 的 Transformer 层封装。"""
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
        **kwargs,
    ):
        cfg = hc_knobs_from_kwargs(kwargs, config)

        if cfg.num_streams is not None:
            config.mhc_num_residual_streams = int(cfg.num_streams)
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
            name=name,
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
        self.num_streams = int(config.mhc_num_residual_streams)
        self.hidden_size = config.hidden_size
        self.chunk_size = int(cfg.chunk_size) if cfg.chunk_size else int(config.num_layers)
        self.is_chunk_entry = (self.layer_number - 1) % self.chunk_size == 0
        self.is_chunk_exit = (
            self.layer_number % self.chunk_size == 0 or self.layer_number == config.num_layers
        )
        self.contract_mode = cfg.contract
        self.head_fn = None
        self.head_base = None
        self.head_scale = None
        if self.is_chunk_exit and self.contract_mode == "learned":
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
