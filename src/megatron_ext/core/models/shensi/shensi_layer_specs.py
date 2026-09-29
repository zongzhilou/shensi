# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


import functools
from typing import Callable, List, Optional
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _get_backend_spec_provider,
    get_dsv4_hybrid_module_spec_for_backend,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron_ext.core.transformer.shensi.attn_res import ShensiAttentionResidual
from megatron_ext.core.transformer.shensi.hyper_connection import ShensiHyperConnection
from megatron_ext.core.transformer.shensi.moe import (
    ShensiHashMLP,
    ShensiMoELayer,
    ShensiMoESubmodules,
)
from megatron_ext.core.transformer.shensi.transformer_layer import (
    ShensiTransformerLayer,
    ShensiTransformerLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

SHENSI_SPEC_OVERRIDES = {
    "module": (
        ShensiTransformerLayer,
        "DeepSeekTransformerLayer -> ShensiTransformerLayer：AttnRes 块栈 + mHC 收紧到层内，"
        "层返回值保持 (output, context)",
    ),
    "self_attention_hyper_connection": (
        ShensiHyperConnection,
        "HyperConnectionModule（n->n Sinkhorn 流混合）-> 流路由 mHC（固定流 + top-k 路由流、"
        "覆盖式写回、MLP 侧因果卷积正交支路），详见 hyper_connection.py",
    ),
    "mlp_hyper_connection": (
        ShensiHyperConnection,
        "同上，is_mlp=True（带 hc_conv_kernels 的 causal depthwise conv + Gram-Schmidt）",
    ),
    "mlp": (
        ShensiMoELayer,
        "MoELayer -> ShensiMoELayer：低秩路由专家用 mcore 的 moe_latent_size 承载，"
        "只补 rank 空间 routed_expert_norm（覆盖 combine()）",
    ),
    "mlp(hash 层)": (
        ShensiHashMLP,
        "mcore 的 hash-MoE（tid2eid 查表）-> dense SwiGLU MLP + deepemb(input_ids) 门控；"
        "hash 层不是 MoE，故 moe_n_hash_layers 在 mcore 侧只用于把 input_ids 透传到层",
    ),
    "self_attention_attn_res": (
        ShensiAttentionResidual,
        "新增 slot：AttnRes（attention 侧）",
    ),
    "mlp_attn_res": (ShensiAttentionResidual, "新增 slot：AttnRes（MLP 侧）"),
    "layer_norm(block)": (
        None,
        "置 None（而非 identity）让上游 has_final_layernorm_in_this_stage() 判假，"
        "从而不触发上游 learned_output_contract；末层归一化由 ShensiModel.output_norm 承担"
        "（发生在 output_attn_res + hc_head 收缩之后，收缩前是 n 流张量）",
    ),
}


def _hash_mlp_builder(submodules: MLPSubmodules, ffn_hidden_size: int) -> Callable:
    def build(config, pg_collection=None, is_mtp_layer=False, name=None, **_ignored):
        return ShensiHashMLP(
            config, submodules=submodules, ffn_hidden_size=ffn_hidden_size, name=name
        )

    build.__name__ = "shensi_hash_mlp_builder"
    return build


def _moe_mlp_builder(experts_spec, norm_spec, layer_number: int) -> Callable:
    submodules = ShensiMoESubmodules(experts=experts_spec, routed_expert_norm=norm_spec)
    return functools.partial(
        ShensiMoELayer, submodules=submodules, layer_number=layer_number
    )


def get_shensi_layer_spec(
    use_te: bool,
    config,
    layer_number: int = 1,
    is_hash_layer: bool = False,
    build_engram: bool = False,
) -> ModuleSpec:
    if build_engram:
        raise NotImplementedError(
            "Shensi 不用 engram（ShensiConfig 里没有 engram 相关字段）"
        )
    backend = _get_backend_spec_provider(config=config)
    attention_spec = get_dsv4_hybrid_module_spec_for_backend(
        config=config, backend=backend
    )
    rms_norm = config.normalization == "RMSNorm"
    mlp_submodules = MLPSubmodules(
        linear_fc1=backend.column_parallel_linear(),
        linear_fc2=backend.row_parallel_linear(),
        activation_func=backend.activation_func(),
    )
    if is_hash_layer:
        mlp = _hash_mlp_builder(mlp_submodules, hmac_intermediate(config))
    else:
        mlp = _moe_mlp_builder(
            experts_spec=backend.grouped_mlp_modules(bool(config.moe_grouped_gemm)),
            norm_spec=backend.layer_norm(rms_norm=True, for_qk=False),
            layer_number=layer_number,
        )
    submodules = ShensiTransformerLayerSubmodules(
        input_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
        self_attention=attention_spec,
        self_attention_hyper_connection=ModuleSpec(
            module=ShensiHyperConnection, params={"is_mlp": False}
        ),
        pre_mlp_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
        mlp=mlp,
        mlp_hyper_connection=ModuleSpec(
            module=ShensiHyperConnection, params={"is_mlp": True}
        ),
        self_attention_attn_res=ModuleSpec(module=ShensiAttentionResidual),
        mlp_attn_res=ModuleSpec(module=ShensiAttentionResidual),
    )
    return ModuleSpec(module=ShensiTransformerLayer, submodules=submodules)


def get_shensi_mtp_layer_spec(
    config, use_transformer_engine: bool = True
) -> ModuleSpec:
    n_hash = int(config.moe_n_hash_layers)
    num_layers = int(config.num_layers)
    last_layer_is_hash = (num_layers - 1) < n_hash
    return get_shensi_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        layer_number=num_layers + 1,
        is_hash_layer=last_layer_is_hash,
    )


def hmac_intermediate(config) -> int:
    return int(config.routed_expert_hidden_size)


def layer_ids_from_layout(layout, *, vp_stage=None, pp_rank=None) -> list[int]:
    from megatron.core.transformer.enums import LayerType

    return list(
        layout.get_layer_id_list(
            layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
        )
    )


def local_layer_ids(
    config, *, vp_stage=None, pp_rank=None, dualpipev_stage=None
) -> list[int]:
    if getattr(config, "pipeline_model_parallel_layout", None) is not None:
        return layer_ids_from_layout(
            config.pipeline_model_parallel_layout, vp_stage=vp_stage, pp_rank=pp_rank
        )
    offset = get_transformer_layer_offset(
        config, vp_stage=vp_stage, pp_rank=pp_rank, dualpipev_stage=dualpipev_stage
    )
    num_layers_to_build = get_num_layers_to_build(
        config, vp_stage=vp_stage, pp_rank=pp_rank, dualpipev_stage=dualpipev_stage
    )
    return list(range(offset, offset + num_layers_to_build))


def get_shensi_decoder_block_spec(
    config,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
    dualpipev_stage: Optional[int] = None,
    use_moe: Optional[bool] = True,
):
    n_hash = int(config.moe_n_hash_layers)
    layer_specs: List[ModuleSpec] = []
    for layer_idx in range(config.num_layers):
        layer_specs.append(
            get_shensi_layer_spec(
                use_te=use_transformer_engine,
                config=config,
                layer_number=layer_idx + 1,
                is_hash_layer=layer_idx < n_hash,
            )
        )
    local_layer_specs = [
        layer_specs[layer_id]
        for layer_id in local_layer_ids(
            config, vp_stage=vp_stage, pp_rank=pp_rank, dualpipev_stage=dualpipev_stage
        )
    ]
    return TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=None)


def _spec_name(obj) -> str:
    if isinstance(obj, ModuleSpec):
        inner = obj.module
        if isinstance(inner, type):
            return inner.__name__
        func = getattr(inner, "func", None)
        if func is not None:
            return f"partial({getattr(func, '__name__', func)})"
        return f"ModuleSpec({type(inner).__name__})"
    if isinstance(obj, type):
        return obj.__name__
    func = getattr(obj, "func", None)
    if func is not None:
        return f"partial({getattr(func, '__name__', func)})"
    name = getattr(obj, "__name__", None)
    if isinstance(name, str):
        return name
    return type(obj).__name__


def describe_block_spec(block_spec) -> list:
    rows = []
    for idx, layer_spec in enumerate(block_spec.layer_specs):
        sub = layer_spec.submodules
        rows.append(
            (
                idx,
                _spec_name(sub.self_attention),
                _spec_name(sub.mlp),
                _spec_name(layer_spec.module),
                _spec_name(getattr(sub, "self_attention_hyper_connection", None)),
                _spec_name(getattr(sub, "mlp_hyper_connection", None)),
                _spec_name(getattr(sub, "self_attention_attn_res", None)),
                _spec_name(getattr(sub, "mlp_attn_res", None)),
            )
        )
    return rows


def build_shensi_transformer_layer(
    config, layer_number: int = 1, is_hash_layer: bool = False
):
    spec = get_shensi_layer_spec(
        use_te=True,
        config=config,
        layer_number=layer_number,
        is_hash_layer=is_hash_layer,
    )
    return build_module(spec, config=config, layer_number=layer_number)


ShensiTransformerLayerSubmodules = ShensiTransformerLayerSubmodules
