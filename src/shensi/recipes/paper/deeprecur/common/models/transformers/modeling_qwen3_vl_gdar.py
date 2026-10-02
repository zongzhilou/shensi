"""Qwen3-VL + GDAR：Text 与 Vision 两塔换成单流 attn-res（无 HC）。

复用 ``gated_delta_attn_res`` 配方的连接算子（``AttentionResidual`` / ``DepthRead`` /
``record_router_stats``），两塔共用同一份实现。层内两段式（与 ``Qwen3GDARDecoderLayer`` 同一套）：

    routed = attn_res.read(state, bank)          # 从「行银行 + 当前流」白化读
    sublayer_out = attn(norm(routed)) / mlp(norm(routed))
    state, gates = attn_res.update(state, sublayer_out)   # 门控 delta 更新

块边界（两塔各自 ``floor(i × depth / n)`` 处）把该层输入追加成 bank 的一行；塔的收口在
最后一块做一次 ``output_attn_res``（``DepthRead``，vision 的那次产出即 vision_final）。

对外契约与上游一致：vision 塔仍出 ``BaseModelOutputWithDeepstackFeatures``、text 塔仍
hidden → hidden —— 顶层 ``Qwen3VLModel`` 的接线（mrope / deepstack / merger /
masked_scatter）原样沿用，DeepRecur 的交织顶层后续再叠。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (
    AttentionResidual,
    DepthRead,
    record_router_stats,
)

from .configuration_qwen3_vl_gdar import Qwen3VLGdarConfig

__all__ = [
    "Qwen3VLGdarConfig",
    "Qwen3VLGdarDeepRecurForConditionalGeneration",
    "Qwen3VLGdarDeepRecurModel",
    "Qwen3VLGdarForConditionalGeneration",
    "Qwen3VLGdarModel",
    "Qwen3VLGdarTextModel",
    "Qwen3VLGdarVisionModel",
    "block_boundaries",
]


def block_boundaries(depth: int, n_blocks: int) -> list[int]:
    """``floor(i × depth / n)``：两塔都得到 n 块，块内层数尽量均匀，余数归末块。"""
    if n_blocks < 1 or n_blocks > depth:
        raise SystemExit(
            f"[deeprecur·gdar] recur_blocks={n_blocks} 必须落在 1..{depth}（该塔层数）"
        )
    return [(i * depth) // n_blocks for i in range(n_blocks)]


class _AttnResCarrier:
    """把「顶层 attn_res_* 旋钮 + 该塔的 hidden_size / rms_norm_eps」合给 ``AttentionResidual``。

    ``AttentionResidual`` 只按 ``getattr(config, ...)`` 读旋钮，所以把 None 的旋钮跳过、
    让算子自己的默认值生效即可。
    """

    def __init__(self, knobs, hidden_size: int, rms_norm_eps: float):
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        for name in dir(knobs):
            if not name.startswith("attn_res_"):
                continue
            value = getattr(knobs, name)
            if value is not None:
                setattr(self, name, value)


def _append_source(bank: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor:
    """把 ``tensor`` 展平成一行追加进银行（bank 为空时新建）。"""
    flat = tensor.reshape(-1, tensor.shape[-1])
    if bank is None:
        return flat.unsqueeze(1)
    return torch.cat([bank, flat.unsqueeze(1)], dim=1)


class _GdarTowerMixin:
    """两塔共用：AR 模块的初始化口径（构造与 post_init 后都按算子自己的 reset 定盘）。"""

    def _init_weights(self, module):
        if isinstance(module, AttentionResidual):
            module.reset_parameters()
            return
        super()._init_weights(module)

    def post_init(self):
        super().post_init()
        for module in self.modules():
            if isinstance(module, AttentionResidual):
                module.reset_parameters()


class Qwen3VLGdarTextDecoderLayer(Qwen3VLTextDecoderLayer):
    """Qwen3-VL text 层 + 单流 GDAR 路径（两段式：attn 段、mlp 段各一次 read/update）。"""

    def __init__(self, config, layer_idx: int, carrier: _AttnResCarrier, *, is_block_start: bool, per_sublayer: bool):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.is_block_start = is_block_start
        self.per_sublayer = per_sublayer
        self.self_attention_attn_res = AttentionResidual(carrier)
        self.mlp_attn_res = AttentionResidual(carrier)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        delta_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        state = hidden_states.reshape(-1, hidden_size)

        if (self.per_sublayer and (delta_residual is None or delta_residual.shape[-2] == 0)) or (
            not self.per_sublayer and self.is_block_start
        ):
            delta_residual = _append_source(delta_residual, state)

        routed, scores = self.self_attention_attn_res.read(state, delta_residual)
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(routed.view(batch_size, seq_len, hidden_size)),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        state, gates = self.self_attention_attn_res.update(state, attn_out.reshape(-1, hidden_size))
        record_router_stats(attn_res_stats, self.layer_idx, "attn", scores=scores, gates=gates)
        if self.per_sublayer:
            delta_residual = _append_source(delta_residual, attn_out)

        routed, scores = self.mlp_attn_res.read(state, delta_residual)
        mlp_out = self.mlp(
            self.post_attention_layernorm(routed.view(batch_size, seq_len, hidden_size))
        )
        state, gates = self.mlp_attn_res.update(state, mlp_out.reshape(-1, hidden_size))
        record_router_stats(attn_res_stats, self.layer_idx, "mlp", scores=scores, gates=gates)
        if self.per_sublayer:
            delta_residual = _append_source(delta_residual, mlp_out)

        return state.view(batch_size, seq_len, hidden_size), delta_residual


class Qwen3VLGdarVisionBlock(Qwen3VLVisionBlock):
    """Qwen3-VL vision block + 单流 GDAR 路径（patch 序列本就如 [P, D] 展平）。"""

    def __init__(self, config, layer_idx: int, carrier: _AttnResCarrier, *, is_block_start: bool, per_sublayer: bool):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.is_block_start = is_block_start
        self.per_sublayer = per_sublayer
        self.self_attention_attn_res = AttentionResidual(carrier)
        self.mlp_attn_res = AttentionResidual(carrier)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        delta_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = hidden_states

        if (self.per_sublayer and (delta_residual is None or delta_residual.shape[-2] == 0)) or (
            not self.per_sublayer and self.is_block_start
        ):
            delta_residual = _append_source(delta_residual, state)

        routed, scores = self.self_attention_attn_res.read(state, delta_residual)
        attn_out = self.attn(
            self.norm1(routed.to(hidden_states.dtype)),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        state, gates = self.self_attention_attn_res.update(state, attn_out)
        record_router_stats(attn_res_stats, self.layer_idx, "attn", scores=scores, gates=gates)
        if self.per_sublayer:
            delta_residual = _append_source(delta_residual, attn_out)

        routed, scores = self.mlp_attn_res.read(state, delta_residual)
        mlp_out = self.mlp(self.norm2(routed.to(hidden_states.dtype)))
        state, gates = self.mlp_attn_res.update(state, mlp_out)
        record_router_stats(attn_res_stats, self.layer_idx, "mlp", scores=scores, gates=gates)
        if self.per_sublayer:
            delta_residual = _append_source(delta_residual, mlp_out)

        return state, delta_residual


class Qwen3VLGdarTextModel(_GdarTowerMixin, Qwen3VLTextModel):
    """text 塔：层换成 GDAR 版；收口在最后一块做一次 ``output_attn_res``（深度读 + norm）。"""

    def __init__(self, config, knobs=None):
        super().__init__(config)
        knobs = knobs if knobs is not None else config
        self.gdar_enabled = getattr(knobs, "recur_blocks", None) is not None
        if self.gdar_enabled:
            carrier = _AttnResCarrier(knobs, config.hidden_size, config.rms_norm_eps)
            depth = config.num_hidden_layers
            boundaries = block_boundaries(depth, int(knobs.recur_blocks))
            starts = set(boundaries)
            per_sublayer = len(boundaries) == depth
            self.layers = nn.ModuleList(
                [
                    Qwen3VLGdarTextDecoderLayer(
                        config, layer_idx, carrier, is_block_start=layer_idx in starts, per_sublayer=per_sublayer
                    )
                    for layer_idx in range(depth)
                ]
            )
            self.output_attn_res = bool(getattr(knobs, "attn_res_output_route", True))
            if self.output_attn_res:
                self.output_attn_res_module = DepthRead(carrier)
            self.block_boundaries = list(boundaries)
            self.tower_depth = depth
            self.post_init()

    def text_prepare(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Cache | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.LongTensor | None, torch.Tensor]:
        """语言侧的公共准备：4 维 mrope 位置、因果 mask、position_embeddings（交织顶层也用它）。"""
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        return position_embeddings, text_position_ids, causal_mask

    def run_layers(
        self,
        hidden_states: torch.Tensor,
        delta_residual: torch.Tensor,
        *,
        layer_range: range,
        position_embeddings,
        attention_mask,
        position_ids,
        past_key_values: Cache | None,
        attn_res_stats: list | None = None,
        post_layer=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """跑指定的一段层（DeepRecur 的每个块调一次；非交织路径整塔一次）。

        ``post_layer(hidden_states, layer_idx)`` 在每层之后调用（deepstack 注入走它）。
        """
        for layer_idx in layer_range:
            hidden_states, delta_residual = self.layers[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                delta_residual=delta_residual,
                attn_res_stats=attn_res_stats,
                **kwargs,
            )
            if post_layer is not None:
                hidden_states = post_layer(hidden_states, layer_idx)
        return hidden_states, delta_residual

    def finalize(self, hidden_states: torch.Tensor, delta_residual: torch.Tensor) -> torch.Tensor:
        """收口：output_attn_res 深度读 + 最终 norm。"""
        if getattr(self, "output_attn_res", False):
            hidden_states = self.output_attn_res_module(
                hidden_states.reshape(-1, hidden_states.shape[-1]), delta_residual
            ).view_as(hidden_states)
        return self.norm(hidden_states)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if not getattr(self, "gdar_enabled", False):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeddings, text_position_ids, causal_mask = self.text_prepare(
            inputs_embeds, attention_mask, position_ids, past_key_values
        )
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        delta_residual = inputs_embeds.new_zeros(batch_size * seq_len, 0, hidden_size)

        def _inject_deepstack(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                return self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            return hidden_states

        hidden_states, delta_residual = self.run_layers(
            inputs_embeds,
            delta_residual,
            layer_range=range(self.tower_depth),
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            attn_res_stats=attn_res_stats,
            post_layer=_inject_deepstack,
            **kwargs,
        )
        hidden_states = self.finalize(hidden_states, delta_residual)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3VLGdarVisionModel(_GdarTowerMixin, Qwen3VLVisionModel):
    """vision 塔：block 换成 GDAR 版；收口（``output_attn_res``）在 merger 之前做一次。

    vision 塔跑到收口那一步的产物即 **vision_final**（DeepRecur 的 reinject 用它的
    merger 投影把最新视觉覆盖进语言）。
    """

    def __init__(self, config, knobs=None):
        super().__init__(config)
        knobs = knobs if knobs is not None else config
        self.gdar_enabled = getattr(knobs, "recur_blocks", None) is not None
        if self.gdar_enabled:
            carrier = _AttnResCarrier(knobs, config.hidden_size, 1e-6)
            depth = config.depth
            boundaries = block_boundaries(depth, int(knobs.recur_blocks))
            starts = set(boundaries)
            per_sublayer = len(boundaries) == depth
            self.blocks = nn.ModuleList(
                [
                    Qwen3VLGdarVisionBlock(
                        config, layer_idx, carrier, is_block_start=layer_idx in starts, per_sublayer=per_sublayer
                    )
                    for layer_idx in range(depth)
                ]
            )
            self.output_attn_res = bool(getattr(knobs, "attn_res_output_route", True))
            if self.output_attn_res:
                self.output_attn_res_module = DepthRead(carrier)
            self.block_boundaries = list(boundaries)
            self.tower_depth = depth
            self.post_init()

    def vision_prepare(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, int]:
        """视觉侧的公共准备：patch_embed + 位置嵌入 + RoPE + 打包注意力元数据。

        返回 ``(hidden, position_embeddings, cu_seqlens, max_seqlen)``；交织顶层逐块跑 vision
        时也用它（patch 顺序 = 上游契约）。
        """
        from transformers.vision_utils import (
            get_vision_attention_seqlens,
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden = hidden + pos_embeds.to(hidden.dtype)
        position_embeddings = self.rotary_pos_emb(hidden, position_ids)
        return hidden, position_embeddings, cu_seqlens, max_seqlen

    def run_blocks(
        self,
        hidden_states: torch.Tensor,
        delta_residual: torch.Tensor,
        *,
        block_range: range,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings,
        attn_res_stats: list | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """跑指定的一段 block（DeepRecur 的每个块调一次；非交织路径整塔一次）。"""
        for layer_idx in block_range:
            hidden_states, delta_residual = self.blocks[layer_idx](
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                delta_residual=delta_residual,
                attn_res_stats=attn_res_stats,
                **kwargs,
            )
        return hidden_states, delta_residual

    def finalize(self, hidden_states: torch.Tensor, delta_residual: torch.Tensor) -> torch.Tensor:
        """收口：output_attn_res 深度读（该产物即 vision_final，尚未过 merger）。"""
        if getattr(self, "output_attn_res", False):
            hidden_states = self.output_attn_res_module(hidden_states, delta_residual)
        return hidden_states

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, attn_res_stats: list | None = None, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        if not getattr(self, "gdar_enabled", False):
            return super().forward(hidden_states, grid_thw, **kwargs)

        hidden_states, position_embeddings, cu_seqlens, max_seqlen = self.vision_prepare(
            hidden_states, grid_thw, **kwargs
        )
        seq_len, hidden_size = hidden_states.size()
        delta_residual = hidden_states.new_zeros(seq_len, 0, hidden_size)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states, delta_residual = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                delta_residual=delta_residual,
                attn_res_stats=attn_res_stats,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.finalize(hidden_states, delta_residual)
        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )


class Qwen3VLGdarModel(Qwen3VLModel):
    """顶层：两塔换成 GDAR 版，其余（mrope / deepstack / merger / masked_scatter）全部沿用上游。"""

    config_class = Qwen3VLGdarConfig

    def __init__(self, config: Qwen3VLGdarConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLGdarVisionModel(config.vision_config, knobs=config)
        self.language_model = Qwen3VLGdarTextModel(config.text_config, knobs=config)
        self.rope_deltas = None
        self.post_init()


class Qwen3VLGdarForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """LM 包装：只把 ``self.model`` 换成 GDAR 顶层（lm_head / tied weights 沿用上游）。"""

    config_class = Qwen3VLGdarConfig

    def __init__(self, config: Qwen3VLGdarConfig):
        super().__init__(config)
        self.model = Qwen3VLGdarModel(config)
        self.post_init()


def _image_spans(input_ids: torch.Tensor, token_id: int) -> list[list[tuple[int, int]]]:
    """每行的图像占位 run：``[(start, length)]``（run 顺序 = clip 打包顺序）。"""
    spans: list[list[tuple[int, int]]] = []
    for row in input_ids.tolist():
        row_spans, index = [], 0
        while index < len(row):
            if row[index] == token_id:
                end = index
                while end < len(row) and row[end] == token_id:
                    end += 1
                row_spans.append((index, end - index))
                index = end
            else:
                index += 1
        spans.append(row_spans)
    return spans


class _FeedbackAttention(nn.Module):
    """feedback（语言→视觉）的回注注意力：每个 patch 只看自己 clip 的语言 token 段。

    逐 clip 调用，「只读本 clip」就是它的语义（不需要 mask 矩阵）；输出落在视觉空间，
    回注时再乘一个 tanh 标量门（初值 0 = 回注关闭起步）。
    """

    def __init__(self, vision_dim: int, text_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads
        self.q_proj = nn.Linear(vision_dim, vision_dim)
        self.k_proj = nn.Linear(text_dim, vision_dim)
        self.v_proj = nn.Linear(text_dim, vision_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(queries).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(keys).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(keys).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)


class Qwen3VLGdarDeepRecurModel(Qwen3VLModel):
    """DeepRecur 顶层：两塔按共享块数同步推进（冻结设计）。

    每块：vision chunk →（末块先 ``output_attn_res`` 定型为 vision_final）→ merger 投影 →
    **reinject**（把最新视觉硬覆盖进语言的图像位）→ language chunk → **feedback**
    （语言回注给下一块的视觉）。没有循环、没有软证据检索；块数即递归次数。
    """

    config_class = Qwen3VLGdarConfig

    def __init__(self, config: Qwen3VLGdarConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        if getattr(config, "recur_blocks", None) is None:
            raise SystemExit("[deeprecur] DeepRecur 顶层要求 recur_blocks 非空（GDAR 两塔必须开启）")
        if list(getattr(config.vision_config, "deepstack_visual_indexes", []) or []):
            raise SystemExit(
                "[deeprecur] DeepRecur 与 deepstack 多点注入未接线："
                "请把 vision_config.deepstack_visual_indexes 置空"
            )
        self.visual = Qwen3VLGdarVisionModel(config.vision_config, knobs=config)
        self.language_model = Qwen3VLGdarTextModel(config.text_config, knobs=config)
        n_vision_blocks = len(self.visual.block_boundaries)
        n_text_blocks = len(self.language_model.block_boundaries)
        if n_vision_blocks != n_text_blocks:
            raise SystemExit(
                "[deeprecur] 两塔块数必须一致（reinject/feedback 依赖逐块对冲）："
                f"vision {n_vision_blocks} != text {n_text_blocks}；"
                "选一个两塔层数的公约块数（2B=7 / 4B=6 / 8B=9）"
            )
        self.rope_deltas = None
        self.do_reinject = bool(getattr(config, "recur_reinject", True))
        self.do_feedback = bool(getattr(config, "recur_feedback", True))
        self.feedback = _FeedbackAttention(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            int(getattr(config, "recur_feedback_heads", 8)),
        )
        self.feedback_gate = nn.Parameter(torch.zeros(1))
        self.post_init()

    def _feedback_step(
        self,
        vis_state: torch.Tensor,
        lang_state: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """语言→视觉回注：逐 clip 让 patch 读自己 clip 的语言 token 段，再乘门写回视觉流。"""
        spans = _image_spans(input_ids, self.config.image_token_id)
        counts = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_unit
        ).tolist()
        update = torch.zeros_like(vis_state)
        patch_cursor, clip_cursor = 0, 0
        for row_index, row_spans in enumerate(spans):
            for start, length in row_spans:
                n_patches = counts[clip_cursor]
                patch_index = torch.arange(
                    patch_cursor, patch_cursor + n_patches, device=vis_state.device
                )
                out = self.feedback(
                    vis_state[patch_cursor : patch_cursor + n_patches],
                    lang_state[row_index, start : start + length],
                )
                update = torch.index_add(update, 0, patch_index, out)
                patch_cursor += n_patches
                clip_cursor += 1
        return vis_state + torch.tanh(self.feedback_gate) * update

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLModelOutputWithPast:
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise SystemExit("[deeprecur] DeepRecur 顶层只支持图像输入")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is None or image_grid_thw is None:
            # 纯文本（引擎的 profiling 与文本 batch 都走这条）：不走视觉/重入，语言塔整段前向
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            return Qwen3VLModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                rope_deltas=self.rope_deltas,
            )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        pos_emb, text_position_ids, causal_mask = self.language_model.text_prepare(
            inputs_embeds, attention_mask, position_ids, past_key_values
        )
        vis_state, vis_pos_emb, cu_seqlens, max_seqlen = self.visual.vision_prepare(
            pixel_values, image_grid_thw, **kwargs
        )
        vis_bank = vis_state.new_zeros(vis_state.shape[0], 0, vis_state.shape[-1])
        lang_state = inputs_embeds
        lang_bank = inputs_embeds.new_zeros(
            inputs_embeds.shape[0] * inputs_embeds.shape[1], 0, inputs_embeds.shape[-1]
        )

        v_bounds = self.visual.block_boundaries
        t_bounds = self.language_model.block_boundaries
        v_depth = self.visual.tower_depth
        t_depth = self.language_model.tower_depth
        n_blocks = len(v_bounds)

        for block_index in range(n_blocks):
            vis_state, vis_bank = self.visual.run_blocks(
                vis_state,
                vis_bank,
                block_range=range(
                    v_bounds[block_index],
                    v_bounds[block_index + 1] if block_index + 1 < n_blocks else v_depth,
                ),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=vis_pos_emb,
                attn_res_stats=attn_res_stats,
                **kwargs,
            )
            # 末块先定型：vision_final（全塔唯一一次 output_attn_res）
            vision_out = (
                self.visual.finalize(vis_state, vis_bank)
                if block_index == n_blocks - 1
                else vis_state
            )
            # reinject：把最新视觉硬覆盖进语言的图像位
            if self.do_reinject:
                visual = self.visual.merger(vision_out)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=lang_state, image_features=visual
                )
                lang_state = lang_state.masked_scatter(image_mask, visual.to(lang_state.dtype))
            lang_state, lang_bank = self.language_model.run_layers(
                lang_state,
                lang_bank,
                layer_range=range(
                    t_bounds[block_index],
                    t_bounds[block_index + 1] if block_index + 1 < n_blocks else t_depth,
                ),
                position_embeddings=pos_emb,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                attn_res_stats=attn_res_stats,
                **kwargs,
            )
            # feedback：语言回注到视觉（回边只在还有下一块视觉时才需要）
            if self.do_feedback and block_index < n_blocks - 1:
                vis_state = self._feedback_step(vis_state, lang_state, input_ids, image_grid_thw)

        hidden_states = self.language_model.finalize(lang_state, lang_bank)

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLGdarDeepRecurForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """LM 包装：``self.model`` 换成 DeepRecur 顶层。"""

    config_class = Qwen3VLGdarConfig

    def __init__(self, config: Qwen3VLGdarConfig):
        super().__init__(config)
        self.model = Qwen3VLGdarDeepRecurModel(config)
        self.post_init()
