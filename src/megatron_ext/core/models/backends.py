# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial
from typing import Optional, Protocol, cast

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import (
    GroupedMLPSubmodules,
    InferenceGroupedMLP,
    SequentialMLP,
)
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder, WrappedTorchNorm
from megatron.core.typed_torch import not_none
from megatron.core.utils import is_te_min_version
try:
    import apex
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    FusedLayerNorm = None
    HAVE_APEX = False
    LNImpl = WrappedTorchNorm
from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEDotProductAttention,
    TELinear,
    TENorm,
)
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.utils import is_te_min_version


class BackendSpecProvider(Protocol):
    @abstractmethod
    def column_parallel_linear(self) -> type:
        ...
    @abstractmethod
    def row_parallel_linear(self) -> type:
        ...
    @abstractmethod
    def fuse_layernorm_and_linear(self) -> bool:
        ...
    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        ...
    @abstractmethod
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        ...
    @abstractmethod
    def core_attention(self) -> type:
        ...
    @abstractmethod
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        ...
    @abstractmethod
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        ...


class LocalSpecProvider(BackendSpecProvider):

    def column_parallel_linear(self) -> type:
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type[RowParallelLinear]:
        return RowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        return None

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        if rms_norm:
            global LNImpl
            LNImpl = WrappedTorchNorm
        return LNImpl

    def core_attention(self) -> type:
        return DotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        return None


class InferenceSpecProvider(BackendSpecProvider):

    def linear(self) -> type:
        return TELinear

    def column_parallel_linear(self) -> type:
        return InferenceColumnParallelLinear

    def row_parallel_linear(self) -> type[InferenceRowParallelLinear]:
        return InferenceRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        return True

    def column_parallel_layer_norm_linear(self) -> type[InferenceLayerNormColumnParallelLinear]:
        return InferenceLayerNormColumnParallelLinear

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        if for_qk and not is_te_min_version("1.9.0"):
            return not_none(FusedLayerNorm)
        return TENorm

    def core_attention(self) -> type[TEDotProductAttention]:
        return TEDotProductAttention

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        return cast(TEActivationFunctionBuilder, TEActivationOp)

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
                activation_func=self.activation_func(),
            ),
        )


def require(requirement: str, requested_by: str = "This backend", instead: str = "") -> None:
    if requirement != "transformer_engine":
        raise ValueError(f"no availability check is known for {requirement!r}")
    from megatron.core.extensions.transformer_engine import HAVE_TE
    if not HAVE_TE:
        message = f"Transformer Engine is not installed, and {requested_by} needs it."
        raise ImportError(f"{message} {instead}".strip())


def select_cross_entropy(
    cross_entropy_loss_fusion: bool = False,
    cross_entropy_fusion_impl: str = "native",
    cuda_graph_impl: Optional[str] = None,
) -> CrossEntropyTarget:
    if cross_entropy_fusion_impl not in ("native", "te"):
        raise ValueError(
            f"unknown cross_entropy_fusion_impl={cross_entropy_fusion_impl!r}. "
            "Valid choices: native, te"
        )
    if not cross_entropy_loss_fusion:
        return unfused_cross_entropy
    if cross_entropy_fusion_impl == "native":
        return _fused_ce
    capturable = cuda_graph_impl == "full_iteration"
    if capturable and not is_te_min_version("2.7.0"):
        from megatron.core.utils import get_te_version
        raise AssertionError(
            "CUDA graph compatible cross entropy requires TransformerEngine >= 2.7.0, but "
            f"found version {get_te_version()}. Please upgrade TransformerEngine or set "
            "cuda_graph_impl to a value other than 'full_iteration'."
        )
    return partial(te_cross_entropy, cuda_graph_capturable=capturable)


def backend_slot(backend: BackendSpecProvider, name: str, default: Callable[[], object], **kwargs):
    method = getattr(backend, name, None)
    if method is None:
        return default()
    if getattr(method, "__func__", None) is getattr(BackendSpecProvider, name, None):
        return default()
    return method(**kwargs)


def get_backend(
    transformer_impl: Literal["local", "transformer_engine", "inference_optimized"],
    *,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    use_te_op_fuser: bool = False,
    cross_entropy_loss_fusion: bool = False,
    cross_entropy_fusion_impl: str = "native",
    cuda_graph_impl: str | None = None,
) -> BackendSpecProvider:
    if transformer_impl == "transformer_engine":
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        base: BackendSpecProvider = TESpecProvider(
            use_te_op_fuser=use_te_op_fuser,
            cross_entropy_loss_fusion=cross_entropy_loss_fusion,
            cross_entropy_fusion_impl=cross_entropy_fusion_impl,
            cuda_graph_impl=cuda_graph_impl,
        )
    elif transformer_impl == "inference_optimized":
        base = InferenceSpecProvider(
            cross_entropy_loss_fusion=cross_entropy_loss_fusion,
            cross_entropy_fusion_impl=cross_entropy_fusion_impl,
            cuda_graph_impl=cuda_graph_impl,
        )
    elif transformer_impl == "local":
        base = LocalSpecProvider(
            cross_entropy_loss_fusion=cross_entropy_loss_fusion,
            cross_entropy_fusion_impl=cross_entropy_fusion_impl,
            cuda_graph_impl=cuda_graph_impl,
        )
    else:
        raise ValueError(
            f"unknown transformer_impl='{transformer_impl}'. "
            "Valid choices: local, transformer_engine, inference_optimized"
        )
    if not use_kitchen:
        return base
    from megatron.core.extensions.kitchen import HAVE_KITCHEN, KitchenSpecProvider
    if not HAVE_KITCHEN:
        raise ImportError(
            "Kitchen is not installed, and this backend needs it. The public stub would "
            "otherwise build a model out of mocks that fails somewhere unrelated."
        )
    return KitchenSpecProvider(
        fallback=base,
        use_kitchen_attention=use_kitchen_attention,
        kitchen_attention_backend=kitchen_attention_backend,
    )


def get_backend_from_config(
    config: object, *, transformer_impl: Optional[str] = None
) -> BackendSpecProvider:
    impl = transformer_impl or getattr(config, "transformer_impl", None)
    if impl is None:
        raise AttributeError(
            "config has no transformer_impl, and this needs to know which backend to build. "
            "Pass transformer_impl= explicitly."
        )
    return get_backend(
        impl,
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
        use_te_op_fuser=getattr(config, "use_transformer_engine_op_fuser", False),
        cross_entropy_loss_fusion=getattr(config, "cross_entropy_loss_fusion", False),
        cross_entropy_fusion_impl=getattr(config, "cross_entropy_fusion_impl", "native"),
        cuda_graph_impl=getattr(config, "cuda_graph_impl", None),
    )
