"""verl 的 Megatron 后端桥：登记 Looma 架构与权重映射（导入即注册）。"""

from __future__ import annotations

from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

from shensi.recipes.paper.looma.common.models.megatron.looma_spec import make_looma_spec

__all__ = ["BRIDGES", "LoomaBridge", "MODEL_TYPE", "build_layer_spec", "layer_spec_knobs"]

MODEL_TYPE = "looma"
LM_CLASS = "LoomaForCausalLM"

_CONNECTION_SUFFIXES: tuple[str, ...] = (
    "gate_proj.0.weight",
    "gate_proj.1.weight",
    "gate_proj.1.bias",
    "q_proj.0.weight",
    "q_proj.1.weight",
    "k_proj.0.weight",
    "k_proj.1.weight",
    "g_scale",
    "decay_tau",
)
_PER_LAYER_MODULES: tuple[str, ...] = ("self_attention_attn_res", "mlp_attn_res")
_OUTPUT_MODULE = "output_attn_res"


def layer_spec_knobs(hf_config: Any) -> dict:
    """解析层规格上的连接旋钮。"""
    knobs = {
        key: getattr(hf_config, key)
        for key in dir(hf_config)
        if key.startswith("looma_") and key != "looma_decay_tau_max"
    }
    knobs["looma_decay_tau_max"] = 2.0 * float(hf_config.num_hidden_layers)
    return knobs


def build_layer_spec(hf_config: Any) -> ModuleSpec:
    """由变体与旋钮构建层规格对象。"""
    return make_looma_spec(**layer_spec_knobs(hf_config))


class LoomaBridge(LlamaBridge):
    """verl 的 Megatron 后端桥：把 Looma 架构与其权重表注册进 Bridge。"""

    MODEL_CONFIG_CLASS = None

    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        provider.transformer_layer_spec = build_layer_spec(self.hf_config or hf_pretrained)
        return provider

    def hf_config_to_provider_kwargs(self, hf_config: Any) -> dict[str, Any]:
        kwargs = super().hf_config_to_provider_kwargs(hf_config)
        kwargs.update(
            persist_layer_norm=False,
            masked_softmax_fusion=False,
            apply_rope_fusion=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
        )
        return kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        base = super().mapping_registry()
        num_layers = int(self.hf_config.num_hidden_layers)
        entries: list = []
        for layer in range(num_layers):
            for module in _PER_LAYER_MODULES:
                for suffix in _CONNECTION_SUFFIXES:
                    entries.append(
                        ReplicatedMapping(
                            megatron_param=f"decoder.layers.{layer}.{module}.{suffix}",
                            hf_param=f"model.layers.{layer}.{module}.{suffix}",
                        )
                    )
        if getattr(self.hf_config, "looma_output_route", True):
            for suffix in _CONNECTION_SUFFIXES:
                entries.append(
                    ReplicatedMapping(
                        megatron_param=f"decoder.layers.{num_layers - 1}.{_OUTPUT_MODULE}.{suffix}",
                        hf_param=f"model.{_OUTPUT_MODULE}.{suffix}",
                    )
                )
        return MegatronMappingRegistry(*base.mappings, *entries)


def _register() -> type:
    cls = type(
        "LoomaForCausalLMBridge",
        (LoomaBridge,),
        {
            "__doc__": (
                "Bridge for ``looma`` (Llama backbone + the block's depth connection): "
                "layer spec from ``models/megatron/looma_spec.py``, weights from the "
                "identical-name connection rows plus mbridge's Llama table."
            ),
        },
    )
    MegatronModelBridge.register_bridge(source=LM_CLASS, target=GPTModel, model_type=MODEL_TYPE)(
        cls
    )
    return cls


# 导入即注册：verl 的 worker 按 VERL_USE_EXTERNAL_MODULES 加载本模块时就完成登记。
BRIDGES: dict[str, type] = {MODEL_TYPE: _register()}
