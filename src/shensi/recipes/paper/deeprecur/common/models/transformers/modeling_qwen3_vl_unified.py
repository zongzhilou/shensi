"""qwen3_vl_unified：Qwen3-VL text 塔 + Gemma 4 式 **encoder-free 视觉（没有 ViT）**。

视觉路径逐件对齐 transformers 的 ``Gemma4UnifiedVisionEmbedder``：

    raw_patches → LN₁ → Dense → LN₂ → +因子化 2D 位置 → LN₃ → 无缩放 RMSNorm → Linear

"raw_patches" 是**合并后**的 48×48×3 原始像素块（图像处理器 patch 16 × 3×3 merge 产出，
每个 block 恰好一个 soft token）。视觉侧的全部参数就是上面这几层——没有注意力、没有 ViT。

占位对齐沿用 HF 的 ``masked_scatter`` 契约：每图的 soft token 数必须等于它在 ``input_ids``
里的占位 span 长度，不匹配显式报错（不静默截断）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from .configuration import Qwen3VLUnifiedConfig

__all__ = [
    "Qwen3VLUnifiedForConditionalGeneration",
    "Qwen3VLUnifiedModel",
    "Qwen3VLUnifiedProcessor",
    "Qwen3VLUnifiedVisionEmbedder",
    "build_unified",
    "build_unified_processor",
]


class _UnscaledRMSNorm(nn.Module):
    """Gemma4Unified 的 ``RmsNorm(with_scale=False)``：只归一化、不带学习缩放。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(origin)


class Qwen3VLUnifiedVisionEmbedder(nn.Module):
    """单个大 matmul 取代整个 ViT：合并后的原始 patch 直接投影进 LLM 空间。"""

    def __init__(self, vision_config, text_config):
        super().__init__()
        patch_dim = vision_config.model_patch_size**2 * 3  # 48×48×3 = 6912
        mm = vision_config.mm_embed_dim
        if mm != vision_config.output_proj_dims:
            raise SystemExit(
                "[deeprecur·unified] mm_embed_dim 必须等于 output_proj_dims"
                f"（对齐 Gemma 4 的接线），拿到 {mm} / {vision_config.output_proj_dims}"
            )
        self.patch_dim = patch_dim
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm)
        self.patch_ln2 = nn.LayerNorm(mm)
        #: 因子化 2D 位置：按 (x, y) 坐标各查一轴再相加
        self.pos_embedding = nn.Parameter(torch.zeros(vision_config.mm_posemb_size, 2, mm))
        self.pos_norm = nn.LayerNorm(mm)
        self.embedding_pre_projection_norm = _UnscaledRMSNorm(
            vision_config.output_proj_dims, eps=vision_config.rms_norm_eps
        )
        self.embedding_projection = nn.Linear(
            vision_config.output_proj_dims, text_config.hidden_size, bias=False
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``pixel_values [B, N, patch_dim]`` + ``image_position_ids [B, N, 2]``（padding=(-1,-1)）。

        返回 ``(hidden [B, N, text_hidden], valid [B, N])``。
        """
        if (target_dtype := self.patch_dense.weight.dtype).is_floating_point:
            pixel_values = pixel_values.to(target_dtype)
        x = self.patch_ln1(pixel_values)
        x = self.patch_dense(x)
        x = self.patch_ln2(x)

        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(x.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos = (self.pos_embedding[clamped, axes] * valid).sum(-2)
        x = self.pos_norm(x + pos)

        x = self.embedding_pre_projection_norm(x)
        x = self.embedding_projection(x)
        return x, (image_position_ids != -1).all(-1)


class Qwen3VLUnifiedModel(PreTrainedModel):
    """encoder-free 的 Qwen3-VL 顶层：文本塔 + 单 matmul 视觉（无视觉塔、无 deepstack）。"""

    config_class = Qwen3VLUnifiedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLUnifiedConfig):
        super().__init__(config)
        self.text_model = Qwen3VLTextModel(config.text_config)
        self.vision_embedder = Qwen3VLUnifiedVisionEmbedder(config.vision_config, config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_image_features(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor, return_dict: bool = True
    ):
        """跑 encoder-free 嵌入器并按图切分（每图只留有效 patch，返回 per-image 列表）。"""
        if pixel_values.dim() != 3:
            raise SystemExit(
                f"[deeprecur·unified] pixel_values 形状应为 [B, N, patch_dim]，拿到 {tuple(pixel_values.shape)}"
            )
        hidden, valid = self.vision_embedder(pixel_values, image_position_ids)
        features = [hidden[row][valid[row]] for row in range(hidden.shape[0])]
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=features)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if input_ids is None:
                raise SystemExit("[deeprecur·unified] 图像输入需要 input_ids（占位 span 做对齐）")
            features = self.get_image_features(
                pixel_values, image_position_ids, return_dict=True
            ).pooler_output
            image_features = torch.cat(features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_id
            n_tokens = int(image_mask.sum())
            if n_tokens != image_features.shape[0]:
                raise SystemExit(
                    f"[deeprecur·unified] 视觉 soft token 数（{image_features.shape[0]}）与占位 token 数"
                    f"（{n_tokens}）不一致：检查图像处理器的 patch/合并口径与占位 span"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).to(inputs_embeds.device), image_features
            )

        outputs = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


class Qwen3VLUnifiedForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """LM 包装：编码器-自由的视觉嵌入器 + Qwen3-VL 文本塔 + 词表头（tied）。"""

    config_class = Qwen3VLUnifiedConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: Qwen3VLUnifiedConfig):
        super().__init__(config)
        self.model = Qwen3VLUnifiedModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )
        return _causal_lm_output(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values
        )


def _causal_lm_output(**fields):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    return CausalLMOutputWithPast(**fields)


class Qwen3VLUnifiedProcessor:
    """unified 的处理器：Qwen tokenizer（文本/模板）+ Gemma4Unified 图像处理器（patch/预算）。

    ``__call__(text, images)`` 把文本里的每个 ``<|image_pad|>`` 展开成对应图像的 soft token
    数（= 该图有效 patch 数），并产出 ``pixel_values`` / ``image_position_ids``。
    """

    #: collator 用的识别标记（与 Qwen/Gemma 原生处理器区分）
    species = "qwen3_vl_unified"

    def __init__(self, tokenizer, image_processor, max_soft_tokens: int):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_soft_tokens = int(max_soft_tokens)
        self.image_token = "<|image_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def save_pretrained(self, save_directory, **kwargs) -> str:
        """落盘：tokenizer（含 VL 模板）+ Gemma 图像处理器 + 预算元数据。"""
        import json
        from pathlib import Path

        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(out)
        if self.tokenizer.chat_template:
            (out / "chat_template.jinja").write_text(self.tokenizer.chat_template, encoding="utf-8")
        self.image_processor.save_pretrained(out)
        (out / "unified_processor.json").write_text(
            json.dumps(
                {"species": self.species, "max_soft_tokens": self.max_soft_tokens},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(out)

    def _expand(self, text: str, counts: list[int]) -> str:
        """把第 i 个 ``<|image_pad|>`` 展开成 ``counts[i]`` 个占位 token。"""
        parts = text.split(self.image_token)
        if len(parts) - 1 != len(counts):
            raise SystemExit(
                f"[deeprecur·unified] 文本里 {len(parts) - 1} 个图像占位与 {len(counts)} 张图不匹配"
            )
        out = parts[0]
        for count, tail in zip(counts, parts[1:]):
            out += self.image_token * count + tail
        return out

    def __call__(self, text: list[str] | str, images=None, return_tensors: str = "pt", max_length: int | None = None, **kwargs):
        """``images`` 两种给法：扁平 list（单段文本配多图）或 list-of-lists（每段文本一组图）。"""
        texts = [text] if isinstance(text, str) else list(text)
        tokenize_kwargs = {"truncation": True, "max_length": max_length} if max_length else {}
        if not images:
            encoded = self.tokenizer(texts, padding=True, return_tensors=return_tensors, **tokenize_kwargs)
            return dict(encoded)
        groups = images if isinstance(images[0], (list, tuple)) else [list(images)]
        if len(groups) != len(texts):
            raise SystemExit(
                f"[deeprecur·unified] {len(texts)} 段文本与 {len(groups)} 组图像不匹配"
                "（多段文本请用 list-of-lists 传图）"
            )
        flat = [image for group in groups for image in group]
        out = self.image_processor(images=flat, return_tensors=return_tensors)
        pixel_values = out["pixel_values"]
        position_ids = out["image_position_ids"]
        counts = [int(c) for c in (position_ids != -1).all(-1).sum(-1).tolist()]
        expanded, cursor = [], 0
        for one_text, group in zip(texts, groups):
            expanded.append(self._expand(one_text, counts[cursor : cursor + len(group)]))
            cursor += len(group)
        encoded = self.tokenizer(expanded, padding=True, return_tensors=return_tensors)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixel_values,
            "image_position_ids": position_ids,
        }


def build_unified_processor(tokenizer_dir: str, max_soft_tokens: int) -> Qwen3VLUnifiedProcessor:
    """组装 unified 处理器（离线可用：只吃本地 tokenizer 目录）。

    vendored 的是纯文本 Qwen3 tokenizer——它的默认模板不认 ``{"type": "image"}`` 部件，
    所以要挂上配方统一的 VL 模板（与 native 档 tiny_processor 用的是同一份）。
    """
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
        Gemma4UnifiedImageProcessor,
    )

    from ...model import VL_CHAT_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.chat_template = VL_CHAT_TEMPLATE
    image_processor = Gemma4UnifiedImageProcessor(max_soft_tokens=int(max_soft_tokens))
    return Qwen3VLUnifiedProcessor(tokenizer, image_processor, max_soft_tokens)


def build_unified(
    base: str | None = None,
    *,
    tokenizer_dir: str | None = None,
    budget: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    tiny_config: dict | None = None,
):
    """组装 encoder-free 模型（+ 处理器）。

    - ``tiny_config``：``{"text": {...}, "vision": {...}}``，随机初始化（冒烟档，不下载权重）
    - ``base``：HF 模型名/目录 → **取其文本塔与词表头权重**（去掉整个 ViT），
      encoder-free 嵌入器按初始分布新建——这就是"以 Qwen3-VL 暂代"的正确口径。
    """
    if tiny_config is not None:
        config = Qwen3VLUnifiedConfig(
            text_config=dict(tiny_config["text"]), vision_config=dict(tiny_config["vision"])
        )
        model = Qwen3VLUnifiedForConditionalGeneration(config).to(dtype)
        processor = (
            build_unified_processor(tokenizer_dir, budget) if tokenizer_dir and budget else None
        )
        return model, processor
    if base is None:
        raise SystemExit("[deeprecur·unified] build_unified 需要 base（模型名/目录）或 tiny_config")

    from transformers import Qwen3VLConfig

    source = Qwen3VLConfig.from_pretrained(base)
    vision_cfg = Qwen3VLUnifiedConfig().vision_config
    config = Qwen3VLUnifiedConfig(
        text_config=source.text_config.to_dict(),
        vision_config=vision_cfg.to_dict(),
        image_token_id=source.image_token_id,
    )
    model = Qwen3VLUnifiedForConditionalGeneration(config).to(dtype)
    remapped, report = _remap_qwen3vl_text_weights(base, model)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    processor = (
        build_unified_processor(tokenizer_dir or base, budget) if budget is not None else None
    )
    print(
        f"[deeprecur·unified] 以 {base} 的文本塔暂代：搬运 {report['moved']} 个张量"
        f"（跳过视觉塔 {report['dropped']} 个）；嵌入器新建；缺失 {len(missing)} / 意外 {len(unexpected)}"
    )
    return model, processor


def _remap_qwen3vl_text_weights(base: str, model) -> tuple[dict, dict]:
    """把 Qwen3-VL 检查点的 ``model.language_model.*`` + ``lm_head`` 映射到本模型。"""
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    root = Path(base)
    if not root.is_dir():
        root = Path(snapshot_download(base))
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]

    moved, dropped = {}, 0
    for shard in shards:
        for key, tensor in load_file(root / shard).items():
            if key.startswith("model.visual."):
                dropped += 1
                continue
            if key.startswith("model.language_model."):
                moved["model.text_model." + key[len("model.language_model.") :]] = tensor
            elif key.startswith("lm_head."):
                moved[key] = tensor
    return moved, {"moved": len(moved), "dropped": dropped}
