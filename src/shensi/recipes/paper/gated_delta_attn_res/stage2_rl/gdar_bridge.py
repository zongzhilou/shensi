"""verl 的 Megatron 后端桥：登记 GDAR 架构与权重映射（导入即注册）。"""

from __future__ import annotations

import warnings
from typing import ClassVar

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.core.models.gpt.gpt_model import GPTModel

from .convert import SynthesisPolicy, build_table
from .variants import VARIANTS, Variant, build_layer_spec, variant_for_model_type

__all__ = [
    "MEGATRON_ONLY_PREFIX",
    "DepthBridge",
    "MegatronOnlyMapping",
    "REGISTERED",
]


MEGATRON_ONLY_PREFIX = "<megatron-only>/"


class MegatronOnlyMapping(MegatronParamMapping[torch.Tensor]):
    """只存在于 mcore 侧的权重映射（合成或缺省）。"""

    def __init__(self, megatron_param: str, value: float, note: str = "") -> None:
        super().__init__(
            megatron_param=megatron_param,
            hf_param=f"{MEGATRON_ONLY_PREFIX}{megatron_param}",
        )
        self.value = float(value)
        self.note = note
        self.allow_hf_name_mismatch = True

    def hf_to_megatron(self, hf_weights, megatron_module) -> torch.Tensor:
        leaf = self.megatron_param.rsplit(".", 1)[-1]
        target = getattr(megatron_module, leaf, None)
        if not isinstance(target, torch.Tensor):
            raise RuntimeError(
                f"{self.megatron_param}: expected {type(megatron_module).__name__}.{leaf} to be a "
                f"tensor, found {type(target).__name__}; the table row and the Megatron module disagree"
            )
        return torch.full_like(target, self.value)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        return {}


class DepthBridge(Qwen3Bridge):
    """verl 的 Megatron 后端桥：把 GDAR 架构与其权重表注册进 Bridge。"""

    VARIANT_KEY: ClassVar[str] = ""

    MODEL_CONFIG_CLASS = None

    @property
    def cfg(self):
        for candidate in (
            getattr(self, "hf_config", None),
            getattr(getattr(self, "hf_pretrained", None), "config", None),
            getattr(self, "hf_pretrained", None),
        ):
            if candidate is not None and hasattr(candidate, "model_type"):
                return candidate
        raise RuntimeError(
            f"{type(self).__name__}: no HF config on the bridge -- provider_bridge()/mapping_registry() "
            f"ran before the checkpoint's config reached it"
        )

    @property
    def variant(self) -> Variant:
        variant = variant_for_model_type(getattr(self.cfg, "model_type", "") or "")
        if variant is None or variant.name != self.VARIANT_KEY:
            raise ValueError(
                f"{type(self).__name__} is registered for model_type={self.VARIANT_KEY!r} but was "
                f"handed {getattr(self.cfg, 'model_type', None)!r}; the dispatch table and the "
                f"checkpoint disagree"
            )
        return variant

    @property
    def num_layers(self) -> int:
        return int(self.cfg.num_hidden_layers)

    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        spec, _resolved, dropped = build_layer_spec(self.cfg, self.variant)
        if dropped:
            warnings.warn(
                f"{self.variant.model_type}: these HF config fields look like connection knobs but "
                f"have no Megatron counterpart and are ignored: {sorted(dropped)}",
                RuntimeWarning,
                stacklevel=2,
            )
        provider.transformer_layer_spec = spec
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        cfg = self.cfg
        policy = SynthesisPolicy()
        full = build_table(self.variant.name, cfg, self.num_layers, policy=policy)
        backbone = {
            pair.mcore
            for pair in build_table(self.variant.name, cfg, self.num_layers, connection=False).pairs
        }
        entries: list[MegatronParamMapping] = []
        for pair in full.pairs:
            if pair.kind == "synth":
                entries.append(
                    MegatronOnlyMapping(pair.mcore, pair.synthesized_value(policy), pair.note)
                )
            elif pair.kind == "qkv":
                entries.append(
                    QKVMapping(megatron_param=pair.mcore, q=pair.hf[0], k=pair.hf[1], v=pair.hf[2])
                )
            elif pair.kind == "fc1":
                entries.append(
                    GatedMLPMapping(megatron_param=pair.mcore, gate=pair.hf[0], up=pair.hf[1])
                )
            elif pair.mcore in backbone:
                entries.append(AutoMapping(megatron_param=pair.mcore, hf_param=pair.hf[0]))
            else:
                entries.append(ReplicatedMapping(megatron_param=pair.mcore, hf_param=pair.hf[0]))
        return MegatronMappingRegistry(*entries)

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        if isinstance(hf_param, str) and hf_param.startswith(MEGATRON_ONLY_PREFIX):
            return torch.zeros(1)
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)


def _register(variant: Variant) -> type:
    cls = type(
        f"{variant.lm_class}Bridge",
        (DepthBridge,),
        {
            "VARIANT_KEY": variant.name,
            "__doc__": (
                f"Bridge for ``{variant.model_type}`` (depth connection {variant.name!r}): "
                f"Qwen3 backbone + {variant.name}'s connection layer, weights from "
                f"``stage2_rl.convert.tables``."
            ),
        },
    )
    MegatronModelBridge.register_bridge(
        source=variant.lm_class,
        target=GPTModel,
        model_type=variant.model_type,
    )(cls)
    return cls


# 导入即注册：verl 的 worker 按 VERL_USE_EXTERNAL_MODULES 加载本模块时就完成登记。
REGISTERED: dict[str, type] = {
    variant.model_type: _register(variant) for variant in VARIANTS.values()
}
