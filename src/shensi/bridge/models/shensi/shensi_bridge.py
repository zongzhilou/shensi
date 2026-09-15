# Copyright (c) 2026 Zongzhi Lou. All rights reserved.
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

"""Shensi 的 Megatron Bridge：provider 覆盖 + HF↔Megatron 参数映射。

整体只写差异：provider 由 :class:`~megatron.bridge.models.deepseek.deepseek_v4_bridge.
DeepSeekV4Bridge` 派生，注意力 / 压缩器 / Indexer / MoE 的字段沿用 DSv4 的语义；映射表沿用
上游的改名风格（``AutoMapping`` 自动判定 TP 切分、自建模块用 ``ReplicatedMapping``、专家用
上游的 ``FusedGatedExpertMapping`` / ``FusedExpertMapping``）。

三类差异：

1. **HF 配置缺字段**：Shensi 的 config 删掉了 ``hc_sinkhorn_iters`` / ``n_shared_experts``，
   而父类直接读它们；压缩路径的 rope 参数也没有 ``factor``。这些在调父类前补齐。
2. **provider 字段**：Shensi 没有共享专家、路由器没有 expert bias、hash 层不是上游的哈希路由
   MoE（是 dense MLP × deepemb），且输出分组投影的取值与 mcore 默认不同。
3. **映射表**：Shensi 独有的模块（超连接、AttentionResidual、hash MLP、出口收缩）在 HF 侧
   同名同形，按名字一一对应即可。
"""

from __future__ import annotations

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel

from ....core.models.gpt.shensi_spec import get_shensi_spec
from .shensi_provider import ShensiModelProvider

__all__ = ["ShensiBridge"]


def _has_apex_fused_grad() -> bool:
    """Apex 的融合权重梯度扩展是否可用（mcore 的 ``gradient_accumulation_fusion`` 依赖它）。"""
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401
    except ImportError:
        return False
    return True


class _ReplicatedOptional(ReplicatedMapping):
    """复制的映射，同时允许 HF 侧没有这个键。

    压缩器只在被压缩的层上存在、Indexer 只在 CSA 层上存在，而映射表按通配符一次性登记，
    因此这些族必须容忍缺键（上游同名的私有类也是这个做法）。
    """

    def __init__(self, megatron_param: str, hf_param: str) -> None:
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self.allow_hf_name_mismatch = True


@MegatronModelBridge.register_bridge(
    source="ShensiForCausalLM",
    target=GPTModel,
    provider=ShensiModelProvider,
    model_type="shensi",
)
class ShensiBridge(DeepSeekV4Bridge):
    """Shensi v1：DSv4 的 provider 与映射 + Shensi 差异。"""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        """由 HF config 生成 provider。"""
        hf_config = hf_pretrained.config
        # 父类直接读这两个字段，而 Shensi 的 config 里没有它们（modular 把继承来的属性删了）。
        if not hasattr(hf_config, "hc_sinkhorn_iters"):
            hf_config.hc_sinkhorn_iters = 20
        if not hasattr(hf_config, "n_shared_experts"):
            hf_config.n_shared_experts = 0
        # 压缩路径的 rope：Shensi 用 rope_type="default"（不做 yarn 缩放），而父类会直接索引
        # factor 等键；补上单位缩放使两者语义一致。
        rope = getattr(hf_config, "rope_parameters", None)
        compress = rope.get("compress") if isinstance(rope, dict) else None
        if isinstance(compress, dict):
            compress.setdefault("factor", 1.0)
            compress.setdefault("beta_fast", 32)
            compress.setdefault("beta_slow", 1)
            compress.setdefault(
                "original_max_position_embeddings",
                getattr(hf_config, "max_position_embeddings", 4096),
            )

        provider = super().provider_bridge(hf_pretrained)

        # Shensi 没有共享专家：mcore 以 None 表示，且 latent MoE 下必须关掉 overlap ——
        # MoELayer 对该组合直接断言，与是否真有共享专家无关。
        provider.moe_shared_expert_intermediate_size = None
        provider.moe_shared_expert_overlap = False
        # 路由器没有 expert bias（HF 侧显式删掉了 e_score_correction_bias）。
        provider.moe_router_enable_expert_bias = False
        # Shensi 的 hash 层是 dense MLP × deepemb，不是上游的哈希路由 MoE。
        provider.moe_n_hash_layers = 0

        # 层与块：Shensi 自己的 spec 工厂（注意力等仍取上游）。
        provider.transformer_layer_spec = get_shensi_spec
        provider.attn_res_block_layer_types = list(hf_config.attn_res_block_layer_types)
        provider.mlp_layer_types = list(hf_config.mlp_layer_types)

        # 专家在降维空间里计算：对应 mcore 的原生 latent MoE（在升维前做归一化）。
        latent = getattr(hf_config, "routed_expert_hidden_size", None)
        if latent:
            provider.moe_latent_size = int(latent)
            provider.moe_use_norm_before_up_proj = True

        # 输出分组投影：mcore 默认 8 / 1024，Shensi 取值不同，必须显式映射。
        provider.output_projection_groups = int(hf_config.o_groups)
        provider.output_projection_lora_rank = int(hf_config.o_lora_rank)

        # Shensi 专有的超连接参数：mcore 无对应字段，供 core 侧模块读取。
        for name in ("hc_active_streams", "hc_fixed_streams", "hc_conv_kernels"):
            if hasattr(hf_config, name):
                setattr(provider, name, getattr(hf_config, name))

        # 上游为 DSv4 指定 flex/HybridEP 派发（配方级性能选项）；未装 HybridEP 时退回默认。
        from megatron.core.transformer.moe.token_dispatcher import hybrid_ep_dispatch

        if hybrid_ep_dispatch is None:
            provider.moe_token_dispatcher_type = "allgather"
            provider.moe_flex_dispatcher_backend = None
        if not _has_apex_fused_grad():
            provider.gradient_accumulation_fusion = False
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """构建 Shensi 的映射表。

        分四组：全局改名对、层内改名对（上游层）、按上游选择的映射类（输出分组投影 / sink /
        压缩器 / Indexer）、以及自建模块（超连接 / AttentionResidual / hash MLP，名字与 HF
        逐字一致所以直接对应）；专家用上游的 fused 映射（HF 是合并的 3-D 张量）。
        """
        mappings = []

        # 全局：embedding / 输出层。
        mappings += [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
        ]
        # 全局自建模块：出口收缩（挂在块的 final_layernorm 槽位上）。
        mappings += [
            ReplicatedMapping(f"decoder.final_layernorm.{tail}", f"model.{hf_tail}")
            for tail, hf_tail in (
                ("norm.weight", "norm.weight"),
                ("hc_head.hc_fn", "hc_head.hc_fn"),
                ("hc_head.hc_base", "hc_head.hc_base"),
                ("hc_head.hc_scale", "hc_head.hc_scale"),
                ("output_attn_res.gate_proj.weight", "output_attn_res.gate_proj.weight"),
                ("output_attn_res.gate_proj.bias", "output_attn_res.gate_proj.bias"),
                ("output_attn_res.k_proj", "output_attn_res.k_proj"),
                ("output_attn_res.q_proj", "output_attn_res.q_proj"),
            )
        ]

        # 层内改名对（上游层：注意力 / 压缩器 / Indexer / MoE 外壳）。
        layer_auto = (
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
            ("self_attention.q_layernorm.weight", "self_attn.q_a_norm.weight"),
            ("self_attention.kv_layernorm.weight", "self_attn.kv_norm.weight"),
            ("self_attention.linear_q_down_proj.weight", "self_attn.q_a_proj.weight"),
            ("self_attention.linear_q_up_proj.weight", "self_attn.q_b_proj.weight"),
            ("self_attention.linear_kv_proj.weight", "self_attn.kv_proj.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_b_proj.weight"),
            ("mlp.fc1_latent_proj.weight", "mlp.routed_expert_down_proj.weight"),
            ("mlp.fc2_norm.weight", "mlp.routed_expert_norm.weight"),
            ("mlp.fc2_latent_proj.weight", "mlp.routed_expert_up_proj.weight"),
            ("mlp.router.weight", "mlp.gate.weight"),
        )
        mappings += [
            AutoMapping(f"decoder.layers.*.{megatron_tail}", f"model.layers.*.{hf_tail}")
            for megatron_tail, hf_tail in layer_auto
        ]

        # 层内：按上游对同类参数的选择。
        mappings += [
            # linear_o_group_proj 是裸 Parameter，且每个 TP rank 都持有全部 o_groups → 复制。
            ReplicatedMapping(
                "decoder.layers.*.self_attention.linear_o_group_proj",
                "model.layers.*.self_attn.o_a_proj.weight",
            ),
            # sink 按头分片。
            ColumnParallelMapping(
                "decoder.layers.*.self_attention.core_attention.attn_sink",
                "model.layers.*.self_attn.sinks",
            ),
            # 压缩器 / Indexer：只有部分层才有 → 容忍 HF 缺键。
            ColumnParallelMapping(
                "decoder.layers.*.self_attention.core_attention.compressor.linear_wkv.weight",
                "model.layers.*.self_attn.compressor.kv_proj.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.linear_wgate.weight",
                "model.layers.*.self_attn.compressor.gate_proj.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.ape",
                "model.layers.*.self_attn.compressor.position_bias",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.norm.weight",
                "model.layers.*.self_attn.compressor.kv_norm.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.linear_wkv.weight",
                "model.layers.*.self_attn.compressor.indexer.kv_proj.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.linear_wgate.weight",
                "model.layers.*.self_attn.compressor.indexer.gate_proj.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.ape",
                "model.layers.*.self_attn.compressor.indexer.position_bias",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.norm.weight",
                "model.layers.*.self_attn.compressor.indexer.kv_norm.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight",
                "model.layers.*.self_attn.compressor.indexer.q_b_proj.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight",
                "model.layers.*.self_attn.compressor.indexer.scorer.weights_proj.weight",
            ),
        ]

        # 层内自建模块：超连接 / AttentionResidual / hash MLP（名字与 HF 逐字一致）。
        layer_replicated = (
            ("self_attention_attn_res.gate_proj.weight",),
            ("self_attention_attn_res.gate_proj.bias",),
            ("self_attention_attn_res.k_proj",),
            ("self_attention_attn_res.q_proj",),
            ("mlp_attn_res.gate_proj.weight",),
            ("mlp_attn_res.gate_proj.bias",),
            ("mlp_attn_res.k_proj",),
            ("mlp_attn_res.q_proj",),
            ("self_attention_hyper_connection.pre_fn", "attn_hc.pre_fn"),
            ("self_attention_hyper_connection.pre_base", "attn_hc.pre_base"),
            ("self_attention_hyper_connection.pre_scale", "attn_hc.pre_scale"),
            ("self_attention_hyper_connection.route_fn", "attn_hc.route_fn"),
            ("self_attention_hyper_connection.route_base", "attn_hc.route_base"),
            ("self_attention_hyper_connection.route_scale", "attn_hc.route_scale"),
            ("self_attention_hyper_connection.route_norm.weight", "attn_hc.route_norm.weight"),
            ("self_attention_hyper_connection.route_norm.bias", "attn_hc.route_norm.bias"),
            ("self_attention_hyper_connection.post_fn", "attn_hc.post_fn"),
            ("self_attention_hyper_connection.post_base", "attn_hc.post_base"),
            ("self_attention_hyper_connection.post_scale", "attn_hc.post_scale"),
            ("mlp_hyper_connection.pre_fn", "ffn_hc.pre_fn"),
            ("mlp_hyper_connection.pre_base", "ffn_hc.pre_base"),
            ("mlp_hyper_connection.pre_scale", "ffn_hc.pre_scale"),
            ("mlp_hyper_connection.route_fn", "ffn_hc.route_fn"),
            ("mlp_hyper_connection.route_base", "ffn_hc.route_base"),
            ("mlp_hyper_connection.route_scale", "ffn_hc.route_scale"),
            ("mlp_hyper_connection.route_norm.weight", "ffn_hc.route_norm.weight"),
            ("mlp_hyper_connection.route_norm.bias", "ffn_hc.route_norm.bias"),
            ("mlp_hyper_connection.post_fn", "ffn_hc.post_fn"),
            ("mlp_hyper_connection.post_base", "ffn_hc.post_base"),
            ("mlp_hyper_connection.post_scale", "ffn_hc.post_scale"),
            ("mlp_hyper_connection.temporal_convs.0.weight", "ffn_hc.temporal_convs.0.weight"),
            ("mlp.gate_proj.weight",),
            ("mlp.up_proj.weight",),
            ("mlp.down_proj.weight",),
            ("mlp.deepemb.weight",),
        )
        mappings += [
            ReplicatedMapping(
                f"decoder.layers.*.{pair[0]}",
                f"model.layers.*.{pair[-1]}",
            )
            for pair in layer_replicated
        ]

        # 专家：HF 是合并的 3-D 张量（gate_up_proj / down_proj），mcore 是逐专家权重。
        mappings += [
            FusedGatedExpertMapping(
                "decoder.layers.*.mlp.experts.linear_fc1.weight*",
                "model.layers.*.mlp.experts.gate_up_proj",
            ),
            FusedExpertMapping(
                "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                "model.layers.*.mlp.experts.down_proj",
            ),
        ]

        return MegatronMappingRegistry(*mappings)
