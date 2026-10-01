"""verl 的 Megatron 后端 ↔ 本配方的 depth-connection 检查点。

verl 建模型只走一条链（``MegatronWorker._init_hf_config_and_tf_config``）：:

    AutoBridge.from_hf_pretrained(local_path, trust_remote_code=True)   # 按 model_type 分发
      -> bridge.to_megatron_provider(load_weights=False)                # 建 provider
      -> provider.provide_distributed_model(...)                        # 建 GPTModel
      -> bridge.load_hf_weights(module, local_path)                     # 灌权重
    ...训练中... bridge.export_hf_weights(module)                        # 同步给 rollout 引擎

本模块补上其中三处，用的都是上游自己的扩展点（不改 site-packages 里任何文件）：

1. **分发**：``config.json`` 的 ``auto_map`` 会把架构解析成类名（``Qwen3GDARForCausalLM``），
   mbridge 拿它在自己的注册表里查桥。本模块导入即注册七个变体
   （``MegatronModelBridge.register_bridge``）。导入即开关，verl 侧用
   ``VERL_USE_EXTERNAL_MODULES=...,<本模块>`` 让每个进程都走一遍。
2. **层规格**：我们的 decoder layer 就是连接本身（``models/megatron/gdar_layer.py`` 等），
   不是 stock Qwen3。provider 上承载层规格的字段是 ``transformer_layer_spec``，规格由
   :func:`...stage2_rl.variants.build_layer_spec` 从 HF 配置的连接旋钮生成——和配方自己的
   训练器（``--model-algo``）走同一份代码，所以 RL 里训的模型就是 PT/SFT 训的那个。
3. **权重表**：连接张量在 mbridge 的 Qwen3 表里没有条目，而且本配方的规格建的是 *local*
   子模块（``input_layernorm``/``pre_mlp_layernorm`` 各自独立），与 mbridge 表假定的
   TransformerEngine 融合布局不同。所以注册表由 ``convert.tables`` 生成——那份经过审计的
   名称表——而不是手写：改一个连接旋钮（低秩、ladder、deviation 标量、输出路由……）不可能
   悄悄漏掉这里的一行。

``export_hf_weights`` 是 rollout 引擎取权重的那条路，所以表里的 ``synth`` 行（HF 侧根本没有
的张量）必须让两侧**语义恒等**：它们在模型里要么恒为零（低秩 ``q/k`` 的 ``up.bias``，HF 的
``_make_qk`` 根本不建这个偏置），要么被钉在策略常量上（AR/DAR 的 ``read_scale``：1.0 就是
HF 的算子）。零值那些在 ``gdar_connection.AttentionResidual._make_qk`` 里
``requires_grad_(False)``，所以训练不会把它们推离零——导出时丢弃它们不会让被服务的模型和
被训练的模型分叉。

刻意不做的事
------------
* 不往 site-packages 写文件、不改上游源码；注册用 ``register_bridge``，"导入本模块"就是开关。
* 不动 provider 的并行度。``pp > 1`` 是支持的：depth/gdar 层会自己设
  ``config.variable_seq_lengths``，让 (1+N)·H 宽的打包激活走动态 p2p 过 stage 边界
  （``depth_layer.py``）。只有 ``fp32_residual_connection`` 和
  ``recompute_granularity='full'`` 会在层里被拒——那两处按 ``hidden_states`` 宽度恒为 H 假设。
"""

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


#: ``hf_param`` 前缀：Megatron 独有行的假名字。它不可能出现在任何检查点里（这是故意的）：
#: :meth:`DepthBridge.maybe_modify_loaded_hf_weight` 认这个前缀，直接给 mapping 一个占位
#: 张量，而不是去 state_dict 里查。
MEGATRON_ONLY_PREFIX = "<megatron-only>/"


class MegatronOnlyMapping(MegatronParamMapping[torch.Tensor]):
    """只有 Megatron 有的张量（表的 ``synth`` 行）。

    导入时**不读检查点**：值是 :attr:`value`——``SynthesisPolicy`` 里那个让 Megatron 模块
    复现 HF 模块算术的常量（HF 根本没有的低秩 ``q/k`` ``up.bias`` 是 ``0.0``，AR/DAR 的读
    门是 ``1.0``，因为参考实现的算子**就是**门为 1 的那个算子）。

    导出时不产生任何条目（``{}``）：HF 没有它的位置。
    """

    def __init__(self, megatron_param: str, value: float, note: str = "") -> None:
        super().__init__(
            megatron_param=megatron_param,
            hf_param=f"{MEGATRON_ONLY_PREFIX}{megatron_param}",
        )
        #: 写进参数里的常量
        self.value = float(value)
        #: 表里这一行为什么存在（审计/调试用）
        self.note = note
        # 上面那个名字故意不在任何检查点里；不设这个开关，task 构建会把这一行当成"缺 HF 权重"
        # 报错——那正是我们要避免的假警报。
        self.allow_hf_name_mismatch = True

    def hf_to_megatron(self, hf_weights, megatron_module) -> torch.Tensor:
        """模块自己的参数，填 :attr:`value`（``hf_weights`` 是占位，不看）。"""
        leaf = self.megatron_param.rsplit(".", 1)[-1]
        target = getattr(megatron_module, leaf, None)
        if not isinstance(target, torch.Tensor):
            raise RuntimeError(
                f"{self.megatron_param}: expected {type(megatron_module).__name__}.{leaf} to be a "
                f"tensor, found {type(target).__name__}; the table row and the Megatron module disagree"
            )
        return torch.full_like(target, self.value)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        """没有可导出的东西：参考实现里不存在这个张量。"""
        return {}


class DepthBridge(Qwen3Bridge):
    """Qwen3 骨干 + 一层 depth 连接；每个变体一个具体子类（见文件末尾的注册）。

    ``VARIANT_KEY`` 说明这个类服务哪个 :class:`~...stage2_rl.variants.Variant`，于是类本身
    就能和自己拿到的检查点对账（配置与注册表不一致时报错，而不是建出另一种连接）。
    """

    VARIANT_KEY: ClassVar[str] = ""

    #: 标准的 GPT model config 表达不了连接：层规格**就是**连接，而且它是从 spec 模块的
    #: ``params`` 里读旋钮的（FlagScale ``--spec`` 的同一套机制）。置 ``None`` 就是明说这件事，
    #: 顺便免掉 ``provider_bridge`` 那条"这个族还没迁移"的 FutureWarning。
    MODEL_CONFIG_CLASS = None

    # ---- 检查点自述 --------------------------------------------------------

    @property
    def cfg(self):
        """HF 配置，按上游填充它的两种方式之一取。

        ``provider_bridge`` 跑在 conversion 侧之前（``build_conversion_tasks`` 才会把
        ``self.hf_config`` 填上），所以这里按"哪份在就取哪份"的顺序读：桥上的 ``hf_config``、
        预训练包装对象的 ``.config``、再退到包装对象本身（``register_bridge_implementation``
        在只有 config 时把 config 直接放在 ``hf_pretrained`` 上）。
        """
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

    # ---- 1) 层规格 --------------------------------------------------------

    def provider_bridge(self, hf_pretrained):
        """Qwen3 的 provider，加一条：层规格换成变体自己的连接层。"""
        provider = super().provider_bridge(hf_pretrained)
        spec, _resolved, dropped = build_layer_spec(self.cfg, self.variant)
        if dropped:
            # 这些字段看起来是连接旋钮、但 Megatron 侧没有对应物。说出来，不要静默吞掉：
            # 静默意味着 Megatron 模型与它的 HF 双胞胎悄悄不再是同一个东西。
            warnings.warn(
                f"{self.variant.model_type}: these HF config fields look like connection knobs but "
                f"have no Megatron counterpart and are ignored: {sorted(dropped)}",
                RuntimeWarning,
                stacklevel=2,
            )
        provider.transformer_layer_spec = spec
        return provider

    # ---- 2) 权重表 --------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        """审计表里每一行，翻译成 mbridge 的 mapping。

        用**生成**的表而不是手写的字典：表是转换器/审计共用的那一份，改一个连接旋钮不可能
        让这里落后。行的 ``kind`` 决定用哪种 mapping：

        ``qkv``    三个 HF 张量 -> 一个融合 ``linear_qkv``（张量并行感知）
        ``fc1``    gate/up -> ``linear_fc1``
        ``vocab``  embedding / head（vocab padding 由 mbridge 负责）
        ``copy``   骨干张量 -> :class:`AutoMapping`（它从模块本身判断列/行/复制并行）；
                   连接张量 -> :class:`ReplicatedMapping`（连接作用在完整 hidden 宽度上，
                   每个 rank 都持有同一份）
        ``synth``  Megatron 独有 -> :class:`MegatronOnlyMapping`
        """
        cfg = self.cfg
        policy = SynthesisPolicy()
        full = build_table(self.variant.name, cfg, self.num_layers, policy=policy)
        # 骨干行与连接行的分界由表自己给出：连 ``connection=False`` 的表就是 stock Qwen3 部分。
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

    # ---- 3) 加载钩子 ------------------------------------------------------

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        """Megatron 独有行不查检查点，直接给占位。

        加载器要求这里返回一个张量；mapping 会忽略它的值与形状，自己填模块参数（形状取参数
        的），所以一元素占位就够——而且故意不长得像任何东西，免得有人以为它有含义。
        """
        if isinstance(hf_param, str) and hf_param.startswith(MEGATRON_ONLY_PREFIX):
            return torch.zeros(1)
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)


def _register(variant: Variant) -> type:
    """建一个变体的桥类并注册（注册会改类的 ``SOURCE_NAME``/``MODEL_TYPE``，所以每个变体
    必须有自己的类）。"""
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


#: ``{model_type: bridge class}``；导入本模块即完成注册
REGISTERED: dict[str, type] = {
    variant.model_type: _register(variant) for variant in VARIANTS.values()
}
