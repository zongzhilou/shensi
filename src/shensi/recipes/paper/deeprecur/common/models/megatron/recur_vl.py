"""mcore 侧的 DGAR/DeepRecur 接入（**能复用就复用**）。

| 件 | 做法 | 复用 |
|---|---|---|
| **视觉塔 GDAR** | 取 mbridge 的 ViT 层规格（``get_vit_layer_with_transformer_engine_spec()``），只把层类换成 ``gdar_layer.GdarTransformerLayer``（状态打包在 hidden 里，drop-in 替换） | ViT 注意力 / 并行 / 初始化全原生 |
| **文本塔 GDAR** | 层类同样换 GDAR，**子模块默认沿用 TE** → 视觉/文本两处 spec 替换都在 provider 内 | 文本层其余件全原生 |
| **块计划** | 两塔共享块数：复用 transformers 镜像的 ``block_boundaries``（单一出处） | — |
| **交织容器** | ``deeprecur_container_class()``：逐块 ``vision chunk → reinject → language chunk → feedback``，复用 mbridge 的塔组装 / ``_preprocess``/``_postprocess`` / merger / 检查点 | 只在需要处新增（块循环 / 硬覆盖 / 软回注） |

容器读"流"的约定：mcore 的 GDAR 把状态**打包进 hidden 的宽度**（宽度 = h + 行数×h），
前 h 列是当前流——reinject/feedback/merger 都只读写这段，行银行（历史）不动。

边界（显式报错，不静默）：容器要求单 PP 段、非 packed 输入、空 deepstack、只支持图像。

自查（gloo world=1；构造与容器前瞻在 CUDA tiny 上跑，不需要数据）：

    python -m shensi.recipes.paper.deeprecur.common.models.megatron.recur_vl --check
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch
from torch import nn

__all__ = [
    "build_tiny_deeprecur_vl",
    "deeprecur_container_class",
    "deeprecur_provider_class",
    "gdar_vision_spec_active",
    "image_spans",
    "recur_block_plan",
    "text_gdar_layer_spec",
    "vision_gdar_layer_spec",
]


def image_spans(input_ids: torch.Tensor, token_id: int) -> list[list[tuple[int, int]]]:
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


def vision_gdar_layer_spec(**gdar_knobs):
    """视觉塔的 GDAR 层规格：ViT 子模块 + GDAR 层类（drop-in）。

    用法（与 mbridge 的视觉塔组装组合）::

        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
        # mbridge 内部：vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
        # 本函数即等价替换：
        vision_spec = vision_gdar_layer_spec(gdar_block_size=3)
    """
    from megatron.core.models.gpt.gpt_layer_specs import (  # noqa: F401
        get_gpt_layer_local_submodules,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec  # noqa: F401

    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer import (
        GdarTransformerLayer,
    )

    # ViT 的层规格来自 mbridge 的用法（同款调用），只换 module
    try:
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec,
        )

        vit_spec = get_vit_layer_with_transformer_engine_spec()
        submodules = vit_spec.submodules
    except Exception:  # pragma: no cover - 上游路径变动时回退到文本子模块（仍可构造，仅提示）
        print(
            "[deeprecur·megatron] 取不到 ViT 层规格，回退到通用子模块（视觉塔接线需核）",
            file=sys.stderr,
        )
        submodules = None

    return ModuleSpec(module=GdarTransformerLayer, submodules=submodules, params=dict(gdar_knobs))


def text_gdar_layer_spec(submodules=None, params=None):
    """文本塔的 GDAR 层规格：层类换 GDAR，**子模块默认沿用 TE**（与 mbridge 原生路径一致）。

    ``submodules=None`` 时自动取 ``get_gpt_layer_with_transformer_engine_spec().submodules``；
    取不到就退回 local（会打印提示，不静默）。``params`` 透传 ``gdar_*`` 旋钮（如
    ``{"gdar_block_size": 2}``，与容器的块计划对齐）。
    """
    from megatron.core.transformer.spec_utils import ModuleSpec

    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer import (
        GdarTransformerLayer,
    )

    if submodules is None:
        try:
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_with_transformer_engine_spec,
            )

            submodules = get_gpt_layer_with_transformer_engine_spec(
                num_experts=None, moe_grouped_gemm=False
            ).submodules
        except Exception:  # pragma: no cover - TE 不可用时退回 local
            print(
                "[deeprecur·megatron] 取不到 TE 文本子模块，退回 local 子模块",
                file=sys.stderr,
            )
            submodules = None
    return ModuleSpec(module=GdarTransformerLayer, submodules=submodules, params=dict(params or {}))


@contextlib.contextmanager
def gdar_vision_spec_active(**gdar_knobs):
    """构造期把 mbridge 的视觉层规格换成 GDAR（ViT 子模块与 Qwen3VLSelfAttention 全保留）。

    mbridge 在 ``Qwen3VLModel.__init__`` 里硬编码 ``get_vit_layer_with_transformer_engine_spec()``，
    所以用构造期补丁替换该函数；退出时恢复。
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as vl_model_module

    original = vl_model_module.get_vit_layer_with_transformer_engine_spec
    gdar_spec = vision_gdar_layer_spec(**gdar_knobs)
    vl_model_module.get_vit_layer_with_transformer_engine_spec = lambda *a, **kw: gdar_spec
    try:
        yield gdar_spec
    finally:
        vl_model_module.get_vit_layer_with_transformer_engine_spec = original


def deeprecur_provider_class(
    gdar_vision_block_size: int = 3,
    gdar_text_block_size: int | None = None,
    recur_blocks: int | None = None,
    do_reinject: bool = True,
    do_feedback: bool = True,
):
    """两塔都换 GDAR 的 provider（继承 mbridge ``Qwen3VLModelProvider``，只重写 ``provide``）。

    ``gdar_vision_block_size`` / ``gdar_text_block_size`` 是两塔的 GDAR 块粒度（provider 是
    dataclass，不能挂新字段，所以走工厂参数）；给 ``recur_blocks`` 时构建**DeepRecur 交织容器**
    （否则是两塔 GDAR 的原生 VL 模型）。
    """
    from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider

    class DeepRecurQwen3VLProvider(Qwen3VLModelProvider):
        """Qwen3-VL 的 GDAR 版：视觉塔走构造期 spec 补丁，文本塔走我们的 spec。"""

        def provide(self, pre_process=None, post_process=None, vp_stage=None):
            from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

            # 单进程默认两个塔都建（视觉塔的挂载条件是 pre_process and add_encoder）
            pre_process = True if pre_process is None else pre_process
            post_process = True if post_process is None else post_process
            model_cls = Qwen3VLModel if recur_blocks is None else deeprecur_container_class()
            extra = (
                {}
                if recur_blocks is None
                else dict(
                    recur_blocks=int(recur_blocks),
                    do_reinject=do_reinject,
                    do_feedback=do_feedback,
                )
            )
            text_params = {"gdar_block_size": gdar_text_block_size} if gdar_text_block_size else None
            with gdar_vision_spec_active(gdar_block_size=gdar_vision_block_size):
                model = model_cls(
                    language_transformer_config=self,
                    language_transformer_layer_spec=text_gdar_layer_spec(params=text_params),
                    vision_transformer_config=self.vision_config,
                    pre_process=pre_process,
                    post_process=post_process,
                    pg_collection=self._pg_collection,
                    add_encoder=self.add_encoder,
                    add_decoder=self.add_decoder,
                    **extra,
                )
            if (
                self.freeze_language_model
                or self.freeze_vision_model
                or self.freeze_vision_projection
            ):
                model.freeze(
                    freeze_language_model=self.freeze_language_model,
                    freeze_vision_model=self.freeze_vision_model,
                    freeze_vision_projection=self.freeze_vision_projection,
                )
            return model

    return DeepRecurQwen3VLProvider


def build_tiny_deeprecur_vl(container: bool = False):
    """用两塔 GDAR 的 provider 真搭一个 tiny Qwen3-VL（``container=True`` 时是交织容器）。

    构造级验证；失败原因原样抛出。tiny 词表小，所以 image_token_id 换成 7（在词表内）。
    """
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    provider_cls = deeprecur_provider_class(
        gdar_vision_block_size=2,
        gdar_text_block_size=2,
        recur_blocks=2 if container else None,
    )
    vision = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        out_hidden_size=128,
        num_position_embeddings=256,
        deepstack_visual_indexes=[],
    )
    provider = provider_cls(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        vocab_size=1024,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        params_dtype=torch.float32,
        bf16=False,
        # 初始化口径由 finalize() 落成可调用对象（mbridge 标准流程）
        init_method_std=0.02,
        gradient_accumulation_fusion=False,  # tiny 自查无 APEX 扩展
        image_token_id=7,  # tiny 词表 1024：占位 id 要落在词表内
        vision_config=vision,
    )
    provider.finalize()
    model = provider.provide()
    return model, {"provider": provider_cls.__name__, "vision_depth": 2, "num_layers": 2}


def recur_block_plan(vision_depth: int, text_layers: int, recur_blocks: int) -> dict:
    """两塔共享块数的切块计划（复用 transformers 镜像的单一出处）。"""
    from shensi.recipes.paper.deeprecur.common.models.transformers.modeling_qwen3_vl_gdar import (
        block_boundaries,
    )

    return {
        "recur_blocks": recur_blocks,
        "vision": block_boundaries(vision_depth, recur_blocks),
        "text": block_boundaries(text_layers, recur_blocks),
    }


class FeedbackAttention(nn.Module):
    """feedback（语言→视觉）回注注意力：每个 patch 只看自己 clip 的语言 token 段。

    与 transformers 镜像的 ``_FeedbackAttention`` 同构（mcore 侧独立实现，避免跨镜像私有导入）。
    """

    def __init__(self, vision_dim: int, text_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads
        self.q_proj = nn.Linear(vision_dim, vision_dim)
        self.k_proj = nn.Linear(text_dim, vision_dim)
        self.v_proj = nn.Linear(text_dim, vision_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        q = self.q_proj(queries).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(keys).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(keys).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)


def deeprecur_container_class():
    """Mcore 的 DeepRecur 交织容器：逐块 vision chunk → reinject → language chunk → feedback。

    复用（全原生）：mbridge 的视觉塔（patch_embed/位置/rotary/打包）、文本塔的
    ``_preprocess``/``_postprocess``/decoder 层、merger/并行/检查点；两塔的层已是 GDAR
    （状态打包在 hidden 里，所以**按层切块天然连续**）。
    新增：块循环、reinject（硬覆盖语言的图像位）、feedback（软门控回注视觉流）。

    边界（显式报错，不静默）：单 PP 段、非 packed 输入、空 deepstack、只支持图像。
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

    class DeepRecurVLModel(Qwen3VLModel):
        """两塔 GDAR + 逐块交织的 DeepRecur 容器。"""

        def __init__(
            self,
            *args,
            recur_blocks: int,
            do_reinject: bool = True,
            do_feedback: bool = True,
            feedback_heads: int = 4,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if recur_blocks is None or int(recur_blocks) < 1:
                raise SystemExit("[deeprecur·megatron] 容器需要 recur_blocks（两塔共享的递归块数）")
            if list(getattr(self.vision_transformer_config, "deepstack_visual_indexes", []) or []):
                raise SystemExit(
                    "[deeprecur·megatron] 容器与 deepstack 未接线：vision 的 deepstack_visual_indexes 必须为空"
                )
            vision_layers = len(self.vision_model.decoder.layers)
            text_layers = len(self.language_model.decoder.layers)
            self._plan = recur_block_plan(vision_layers, text_layers, int(recur_blocks))
            if len(self._plan["vision"]) != len(self._plan["text"]):
                raise SystemExit("[deeprecur·megatron] 两塔块数不一致（改 recur_blocks）")
            self.do_reinject = bool(do_reinject)
            self.do_feedback = bool(do_feedback)
            self.image_token_id = int(getattr(self.config, "image_token_id", 151655))
            self.feedback = FeedbackAttention(
                int(self.vision_transformer_config.hidden_size),
                int(self.config.hidden_size),
                int(feedback_heads),
            )
            self.feedback_gate = nn.Parameter(torch.zeros(1))

        # ---- 视觉：准备 / 分块 / 合并（逐行镜像 Qwen3VLVisionModel.forward，只是可切块）----
        def _vision_prepare(self, pixel_values, grid_thw):
            from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import (
                _vision_forward_packed_attention_setup,
            )

            hidden = self.vision_model.patch_embed(pixel_values)
            hidden = hidden + self.vision_model.fast_pos_embed_interpolate(grid_thw)
            seq_len = hidden.size(0)
            rotary = self.vision_model.rot_pos_emb(grid_thw).reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)
            hidden = hidden[:, None]
            packed, mask = _vision_forward_packed_attention_setup(
                use_cuda_graph_padding=False,
                hidden_states=hidden,
                original_seq_len=seq_len,
                seq_len=seq_len,
                grid_thw=grid_thw,
                build_packed_seq_params=self.vision_model.build_packed_seq_params,
            )
            return hidden, rotary, packed, mask

        def _vision_chunk(self, hidden, layer_range, rotary, packed, mask):
            for layer in self.vision_model.decoder.layers[layer_range]:
                hidden, _ = layer(
                    hidden, attention_mask=mask, rotary_pos_emb=rotary, packed_seq_params=packed
                )
            return hidden

        # ---- 状态与流：mcore 的 GDAR 把状态打包进 hidden 的宽度（前 h 列 = 当前流）----
        @staticmethod
        def _stream_prefix(hidden, h):
            """取「当前流」：GDAR 打包后 hidden 宽度 = h + 行数×h，前 h 列就是流本身。"""
            width = hidden.shape[-1]
            if width == h:
                return hidden
            flat = hidden.reshape(-1, width)
            return flat[:, :h].reshape(*hidden.shape[:-1], h)

        def _vision_merge(self, hidden_vision, grid_thw):
            stream = self._stream_prefix(hidden_vision, self.vision_model.config.hidden_size)
            merged = self.vision_model.merger(stream)
            split_sizes = (
                grid_thw.prod(-1) // self.vision_model.spatial_merge_unit
            ).tolist()
            return torch.cat(torch.split(merged, split_sizes, dim=0), dim=0)

        # ---- 语言：reinject / 分块 ----
        def _reinject(self, lang_state, visual, input_ids):
            """把最新视觉硬覆盖进语言的图像位（只写「当前流」；行银行是历史，不动）。"""
            h = int(self.config.hidden_size)
            stream = self._stream_prefix(lang_state, h)
            mask = (input_ids == self.image_token_id).transpose(0, 1).unsqueeze(-1)  # [S, B, 1]
            n_tokens = int(mask.sum())
            if n_tokens != visual.shape[0]:
                raise SystemExit(
                    f"[deeprecur·megatron] 视觉 soft token 数（{visual.shape[0]}）与占位（{n_tokens}）不一致"
                )
            stream = stream.masked_scatter(mask.to(stream.device), visual.to(stream.dtype))
            if lang_state.shape[-1] == h:
                return stream
            flat = lang_state.reshape(-1, lang_state.shape[-1])
            merged = torch.cat([stream.reshape(-1, h), flat[:, h:]], dim=-1)
            return merged.reshape(*lang_state.shape[:-1], lang_state.shape[-1])

        def _language_chunk(self, hidden, layer_range, attention_mask, rotary_pos_emb):
            for layer in self.language_model.decoder.layers[layer_range]:
                hidden, _ = layer(hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
            return hidden

        def _feedback_step(self, vision_hidden, lang_hidden, input_ids, grid_thw):
            spans = image_spans(input_ids, self.image_token_id)
            counts = (
                grid_thw.prod(-1) // self.vision_model.spatial_merge_unit
            ).tolist()
            v_h = int(self.vision_model.config.hidden_size)
            vis_stream = self._stream_prefix(vision_hidden, v_h)
            lang_stream = self._stream_prefix(lang_hidden, self.config.hidden_size)
            update = torch.zeros_like(vis_stream)
            patch_cursor, clip_cursor = 0, 0
            for row, row_spans in enumerate(spans):
                for start, length in row_spans:
                    n = counts[clip_cursor]
                    idx = torch.arange(patch_cursor, patch_cursor + n, device=vis_stream.device)
                    out = self.feedback(
                        vis_stream[patch_cursor : patch_cursor + n, 0],
                        lang_stream[start : start + length, row],
                    )
                    update = torch.index_add(update, 0, idx, out.unsqueeze(1))
                    patch_cursor += n
                    clip_cursor += 1
            updated_stream = vis_stream + torch.tanh(self.feedback_gate) * update.to(vis_stream.dtype)
            if vision_hidden.shape[-1] == v_h:
                return updated_stream
            # 写回打包 hidden：只替换前 h 列（银行行是跨块累积状态，不动）
            flat = vision_hidden.reshape(-1, vision_hidden.shape[-1])
            merged = torch.cat([updated_stream.reshape(-1, v_h), flat[:, v_h:]], dim=-1)
            return merged.reshape(*vision_hidden.shape[:-1], vision_hidden.shape[-1])

        def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            labels: torch.Tensor = None,
            pixel_values: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            packed_seq_params: object | None = None,
            pixel_values_videos: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
            loss_mask: torch.Tensor = None,
            padding_mask: torch.Tensor = None,
            **kwargs,
        ):
            if pixel_values_videos is not None or video_grid_thw is not None:
                raise SystemExit("[deeprecur·megatron] 容器只支持图像输入")
            if packed_seq_params is not None:
                raise SystemExit("[deeprecur·megatron] 容器暂不支持 packed 输入（THD）")
            if pixel_values is None or image_grid_thw is None:
                raise SystemExit("[deeprecur·megatron] 容器需要图像输入（pixel_values + image_grid_thw）")
            if not self.pre_process and not self.add_encoder:
                raise SystemExit("[deeprecur·megatron] 容器要求单 PP 段（pre_process + add_encoder）")

            # 视觉准备（一次）+ 文本嵌入（一次）
            vis_hidden, vis_rotary, vis_packed, vis_mask = self._vision_prepare(
                pixel_values, image_grid_thw
            )
            lang_state = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # 与 mbridge 原生路径同口径（mrope 位置不走词嵌入表）
            )
            state = None
            rotary_pos_emb = None
            rotary_pos_cos = rotary_pos_sin = None
            n_blocks = len(self._plan["vision"])
            for block in range(n_blocks):
                v_a = self._plan["vision"][block]
                v_b = (
                    self._plan["vision"][block + 1]
                    if block + 1 < n_blocks
                    else len(self.vision_model.decoder.layers)
                )
                vis_hidden = self._vision_chunk(
                    vis_hidden, slice(v_a, v_b), vis_rotary, vis_packed, vis_mask
                )
                visual = self._vision_merge(vis_hidden, image_grid_thw)
                if self.do_reinject:
                    lang_state = self._reinject(lang_state, visual, input_ids)

                if block == 0:
                    pre = self.language_model._preprocess(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        decoder_input=lang_state,
                        inference_context=None,
                        packed_seq_params=None,
                        padding_mask=padding_mask,
                    )
                    lang_state, rotary_pos_emb = pre[0], pre[1]
                    rotary_pos_cos, rotary_pos_sin = pre[2], pre[3]
                    if len(pre) > 5:
                        padding_mask = pre[5]

                t_a = self._plan["text"][block]
                t_b = (
                    self._plan["text"][block + 1]
                    if block + 1 < n_blocks
                    else len(self.language_model.decoder.layers)
                )
                lang_state = self._language_chunk(
                    lang_state, slice(t_a, t_b), attention_mask, rotary_pos_emb
                )
                if self.do_feedback and block + 1 < n_blocks:
                    vis_hidden = self._feedback_step(
                        vis_hidden, lang_state, input_ids, image_grid_thw
                    )
                state = lang_state

            return self.language_model._postprocess(
                hidden_states=state,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                mtp_in_postprocess=False,
                loss_mask=loss_mask,
                decoder_input=lang_state,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **kwargs,
            )

    return DeepRecurVLModel


def check() -> int:
    """CPU 自查：层类别名进规格、视觉规格可构造、GDAR 层能前向、块计划与几何一致。"""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    from megatron.core.parallel_state import initialize_model_parallel

    initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)  # mcore 的 TP RNG 追踪器要先种上（层构造需要）

    from megatron.core.transformer.transformer_config import TransformerConfig

    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer import (
        GdarTransformerLayer,
    )

    checks: list[str] = []

    # 1) 视觉规格：层类换成 GDAR，子模块保留（ViT 件）
    vision_spec = vision_gdar_layer_spec(gdar_block_size=3)
    assert vision_spec.module is GdarTransformerLayer, "视觉规格的层类没换到 GDAR"
    checks.append(f"视觉规格：module={vision_spec.module.__name__}（子模块来自 ViT 规格）")

    # 2) 文本规格：复用 gdar 工厂
    text_spec = text_gdar_layer_spec()
    assert text_spec.module is GdarTransformerLayer
    checks.append("文本规格：复用 make_gdar_spec()（gated_delta_attn_res 的 mcore 件）")

    # 3) 块计划与 2B/4B/8B 几何一致（n 块 = n 个起点；floor(i×depth/n) 分配）
    plan = recur_block_plan(24, 28, 7)
    assert plan["vision"] == [0, 3, 6, 10, 13, 17, 20], plan["vision"]
    assert plan["text"] == [0, 4, 8, 12, 16, 20, 24], plan["text"]
    assert len(recur_block_plan(27, 36, 9)["vision"]) == 9
    assert len(recur_block_plan(24, 36, 6)["text"]) == 6
    checks.append(
        "块计划：2B=7（vision 24→[0,3,6,10,13,17,20] 块内 3/4 交替 · text 28→[0,4,…,24] 每块 4）"
        " · 8B=9（vision 27→步长 3） · 4B=6"
    )

    # 4) GDAR 层能构造并前向（文本子模块，tiny）：中间层把状态打包进 hidden，末层收敛回 H
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        gated_linear_unit=True,
        init_method_std=0.02,
        add_bias_linear=False,
        transformer_impl="local",
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = GdarTransformerLayer(config, submodules=None, layer_number=1, gdar_block_size=2)
    layer = layer.to(device)
    n_conn = sum(
        1 for n, _ in layer.named_parameters() if "attn_res" in n or "output_attn_res" in n
    )
    hidden = torch.randn(8, 2, 128, device=device)
    out, _ = layer(hidden)
    packed_width = int(out.shape[-1])
    assert packed_width % 128 == 0 and packed_width >= 128, f"打包宽度不对：{packed_width}"
    layer2 = GdarTransformerLayer(config, submodules=None, layer_number=2, gdar_block_size=2)
    layer2 = layer2.to(device)
    out2, _ = layer2(out)
    assert out2.shape == (8, 2, 128) and torch.isfinite(out2).all(), tuple(out2.shape)
    checks.append(
        f"GDAR 层前向/续层通过（连接参数 {n_conn} 组；中间层打包 {packed_width}=…×128，末层收敛回 128；"
        f"device={device}）"
    )

    # 5) 两塔 GDAR 的 tiny VL 模型真构造（mbridge provider + 视觉/文本两处 spec 替换）
    model, info = build_tiny_deeprecur_vl()

    def _gdrt_layers(module):
        return [
            type(m).__name__
            for m in module.modules()
            if type(m).__name__.endswith("TransformerLayer")
        ]

    vision_classes = _gdrt_layers(model.vision_model) if hasattr(model, "vision_model") else []
    text_classes = _gdrt_layers(model.language_model)
    assert vision_classes and all(c == "GdarTransformerLayer" for c in vision_classes), vision_classes
    assert text_classes and all(c == "GdarTransformerLayer" for c in text_classes), text_classes
    checks.append(
        f"tiny VL 真构造通过：视觉塔 {len(vision_classes)} 层 + 文本塔 {len(text_classes)} 层"
        f"全部是 GDAR，provider={info['provider']}"
    )

    # 6) 容器前向（tiny 交织）：出 logits；reinject / feedback 开关都真实改变输出
    container, info2 = build_tiny_deeprecur_vl(container=True)
    container = container.to(device).eval()
    grid = torch.tensor([[1, 4, 4]], device=device)  # 1 图 16 patch → 4 个 soft token
    patch_dim = 3 * 2 * 16 * 16
    pixel_values = torch.randn(int(grid.prod()), patch_dim, device=device)
    n_soft = int(grid.prod()) // int(container.vision_model.spatial_merge_unit)
    input_ids = torch.tensor([[1, 2] + [7] * n_soft + [3, 4]], device=device)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, device=device).view(1, 1, seq).expand(3, 1, seq)

    def _run():
        with torch.no_grad():
            out = container(
                input_ids=input_ids,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=grid,
            )
        return out.float()

    logits = _run()
    assert torch.isfinite(logits).all() and logits.dim() == 3, tuple(logits.shape)
    container.do_reinject = False
    logits_no_reinject = _run()
    container.do_reinject = True
    with torch.no_grad():
        container.feedback_gate.fill_(1.0)
    logits_feedback_on = _run()
    container.do_feedback = False
    logits_feedback_off = _run()
    container.do_feedback = True
    assert not torch.allclose(logits, logits_no_reinject), "reinject 开关没有改变输出"
    assert not torch.allclose(logits_feedback_on, logits_feedback_off), "feedback 开关没有改变输出"
    checks.append(
        f"容器前向（{info2['provider']}）：tiny 交织出 logits {tuple(logits.shape)}；"
        "reinject / feedback 开关都真实改变输出"
    )

    print("[deeprecur·megatron] mcore 侧自查：")
    for line in checks:
        print(f"  ✓ {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mcore 侧的视觉 GDAR / DeepRecur 接入件")
    parser.add_argument("--check", action="store_true", help="CPU 自查")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
