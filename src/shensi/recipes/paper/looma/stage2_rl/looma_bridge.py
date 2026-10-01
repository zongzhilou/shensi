"""verl 的 Megatron 后端与 Looma 检查点之间的桥（导入本模块即注册）。

verl 建模型走 ``AutoBridge.from_hf_pretrained`` → ``to_megatron_provider`` →
``provide_distributed_model`` → ``load_hf_weights`` 这条链，训练中再由 ``export_hf_weights``
同步给 rollout 引擎。本模块用 megatron-bridge 自己的扩展点补上三处：按 ``model_type``
分发到本桥、从 HF 配置的 ``looma_*`` 旋钮生成层规格、在 Llama 权重表上追加连接张量的条目。
基类取 ``LlamaBridge``，因为骨干就是 Llama，只有残差位置换成了连接。
"""

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
    """把 HF 配置的 ``looma_*`` 旋钮读成 :func:`make_looma_spec` 的 kwargs。

    键名必须带 ``looma_`` 前缀（``_DEFAULT`` 与层都只认这个前缀），去掉前缀会让旋钮静默退回
    默认值；``looma_decay_tau_max`` 是时间常数阶梯的顶端，由层数按 ``2 * num_hidden_layers``
    推出来，不取配置里的值。
    """
    knobs = {
        key: getattr(hf_config, key)
        for key in dir(hf_config)
        if key.startswith("looma_") and key != "looma_decay_tau_max"
    }
    knobs["looma_decay_tau_max"] = 2.0 * float(hf_config.num_hidden_layers)
    return knobs


def build_layer_spec(hf_config: Any) -> ModuleSpec:
    """从 HF 配置建 Looma 的层规格（与配方训练器同一份 ``make_looma_spec``）。"""
    return make_looma_spec(**layer_spec_knobs(hf_config))


class LoomaBridge(LlamaBridge):
    """Llama 骨干 + Looma 的 block 层，注册在 ``model_type = "looma"``。"""

    MODEL_CONFIG_CLASS = None

    def provider_bridge(self, hf_pretrained):
        """建 Llama 的 provider，并把层规格换成 Looma 的 block。"""
        provider = super().provider_bridge(hf_pretrained)
        provider.transformer_layer_spec = build_layer_spec(self.hf_config or hf_pretrained)
        return provider

    def hf_config_to_provider_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """在父类的 provider kwargs 上关掉全部 TE 专属开关。

        训练档一律不开持久化 norm 与各类融合，桥这条路必须钉住同样的数值口径，否则 RL 里训出
        的模型与 PT/SFT 训出来的会差在这些开关上。
        """
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
        """在父类的 Llama 表上追加连接张量的条目。

        连接两侧名字逐字相同、作用在完整 hidden 宽度上且每个 rank 持有同一份，所以逐条登记为
        :class:`ReplicatedMapping`，而不是让 :class:`AutoMapping` 去猜布局。返回的是重建的注册
        表：构造时会自动补上派生条目，对已有名字幂等。
        """
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
            # 输出路由两侧的位置不同：mcore 把它放在末层里（`decoder.layers.{末}.output_attn_res`），
            # HF 参考放在模型上（`model.output_attn_res`）。
            for suffix in _CONNECTION_SUFFIXES:
                entries.append(
                    ReplicatedMapping(
                        megatron_param=f"decoder.layers.{num_layers - 1}.{_OUTPUT_MODULE}.{suffix}",
                        hf_param=f"model.{_OUTPUT_MODULE}.{suffix}",
                    )
                )
        return MegatronMappingRegistry(*base.mappings, *entries)


def _register() -> type:
    """建出 Looma 的桥类、注册到 ``register_bridge``，并返回该类。"""
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


BRIDGES: dict[str, type] = {MODEL_TYPE: _register()}
