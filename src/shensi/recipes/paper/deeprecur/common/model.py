"""占位模型：Qwen3-VL 的装载、tiny 随机几何、processor 与论文式冻结方案。

论文（arXiv 2406.04334, DeepStack）的模型是 CLIP-large-336 + MLP projector + Vicuna；
本配方在其训练配方（数据 / 超参 / 冻结口径）不变的前提下，模型先以 Qwen3-VL 暂代：

- 论文的 "projector" 对应 Qwen3-VL 视觉塔出口的 ``model.visual.merger``（PatchMerger MLP）
  与 ``model.visual.deepstack_merger_list.*``（多分辨率注入件，Qwen 自己的机制，与论文方法无关）；
- 论文的 "vision encoder" 对应 ``model.visual.*``（含 patch_embed / blocks / pos_embed）；
- 论文的 "LLM" 对应 ``model.language_model.*`` 与 ``lm_head.*``。
"""

from __future__ import annotations

import os

import torch
from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

from .paths import MODEL_ENV, TOKENIZER_DIR, env_paths

#: 论文口径的模型分组（参数名前缀 → 组名），冻结与分组 LR 都按它算。
#: 特定前缀（projector）必须排在 vision 前面——``model.visual.merger.`` 也以
#: ``model.visual.`` 开头，顺序错了会把 projector 误归进视觉塔。
#: unified（encoder-free）没有视觉塔：可训的 projector 位由 ``model.vision_embedder.`` 承担。
GROUP_PREFIXES = (
    ("projector", "model.vision_embedder."),
    ("projector", "model.visual.merger."),
    ("projector", "model.visual.deepstack_merger_list."),
    ("vision", "model.visual."),
    ("llm", "model.language_model."),
    ("llm", "model.text_model."),
    ("llm", "lm_head."),
)

#: DeepRecur 自研件（AR 模块 / feedback）——名字出现在层内部，按**子串**识别，先于前缀判定。
#: 这些是三个臂里"要训练的新东西"，PT 档不随两塔一起冻结。
RECUR_MARKERS = (
    ".self_attention_attn_res.",
    ".mlp_attn_res.",
    ".output_attn_res_module.",
    ".feedback.",
    ".feedback_gate",
)


def param_group(name: str) -> str:
    """参数名 → 组名（vision / projector / llm / recur）。"""
    if any(marker in name for marker in RECUR_MARKERS):
        return "recur"
    for group, prefix in GROUP_PREFIXES:
        if name.startswith(prefix):
            return group
    return "other"


def placeholder_name(cfg: dict) -> str:
    """占位模型名：环境变量 > 配置。"""
    return os.environ.get(MODEL_ENV) or cfg["model"]["placeholder"]["name"]


def load_model(cfg: dict):
    """按配置装载模型：三个臂共用同一套 stage 脚本。

    - ``native``（**deepstack 臂**）：上游 Qwen3-VL 原样；继承权重但把 projector /
      deepstack 权重**重随机**（公平口径：不继承 Qwen 自家数据训出的对齐件）。
    - ``unified``（**encoder-free 臂**）：删掉整个 ViT，文本塔继承、单 matmul 嵌入器新建。
    - ``gdar`` / ``deeprecur``（**我们的臂**）：继承两塔权重，GDAR / DeepRecur 新件新建。
    """
    from .models.transformers import build_unified
    from .models.variants import UNIFIED

    model_cfg = cfg["model"]
    variant = model_cfg.get("variant", "native")
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    attn = model_cfg.get("attn_implementation", "sdpa")
    if variant in ("gdar", "deeprecur"):
        return _load_recur_arm(cfg, model_cfg, variant, dtype, attn)
    if model_cfg.get("tiny", False):
        if variant == "unified":
            model, _ = build_unified(
                tiny_config=tiny_unified_config(model_cfg), dtype=dtype
            )
            return model
        # 构造器 post-init 已随机初始化；构造参数不收 dtype，装完再转
        return Qwen3VLForConditionalGeneration(tiny_config(model_cfg)).to(dtype)
    load = model_cfg.get("placeholder", {}).get("load")
    if variant == "unified":
        model, _ = build_unified(
            load or placeholder_name(cfg),
            budget=model_cfg.get("visual_token_budget") or UNIFIED.visual_token_budget,
            dtype=dtype,
            attn_implementation=attn,
        )
        return model
    model = Qwen3VLForConditionalGeneration.from_pretrained(load or placeholder_name(cfg), dtype=dtype, attn_implementation=attn)
    counts = refresh_alignment_weights(model)
    print(
        "[deeprecur] 公平口径（deepstack 臂）：projector/deepstack 权重不继承、已重随机"
        f"（{ {k: v for k, v in counts.items()} }）"
    )
    return model


def _load_recur_arm(cfg: dict, model_cfg: dict, variant: str, dtype, attn: str):
    """GDAR / DeepRecur 臂：继承 Qwen3-VL 两塔权重（新件随机），deepstack 不参与。"""
    from .models.transformers import (
        Qwen3VLGdarConfig,
        Qwen3VLGdarDeepRecurForConditionalGeneration,
        Qwen3VLGdarForConditionalGeneration,
    )

    recur_blocks = model_cfg.get("recur_blocks")
    if recur_blocks is None:
        raise SystemExit(
            "[deeprecur] variant=gdar/deeprecur 需要 model.recur_blocks"
            "（两塔共享的递归块数；8B 口径 = 9）"
        )
    load = model_cfg.get("placeholder", {}).get("load")
    if model_cfg.get("tiny", False):
        base = tiny_config(model_cfg)
        text, vision, inherited = base.text_config.to_dict(), base.vision_config.to_dict(), None
    else:
        base = Qwen3VLConfig.from_pretrained(load or placeholder_name(cfg))
        text = base.text_config.to_dict()
        vision = {**base.vision_config.to_dict(), "deepstack_visual_indexes": []}
        inherited = load or placeholder_name(cfg)
    config = Qwen3VLGdarConfig(
        text_config=text,
        vision_config=vision,
        recur_blocks=int(recur_blocks),
        image_token_id=int(getattr(base, "image_token_id", 151655)),
    )
    cls = (
        Qwen3VLGdarDeepRecurForConditionalGeneration
        if variant == "deeprecur"
        else Qwen3VLGdarForConditionalGeneration
    )
    model = cls(config)
    if model_cfg.get("tiny", False):
        model = model.to(dtype)
    if inherited:
        state, dropped = _base_tower_state(inherited)
        missing, unexpected = model.load_state_dict(state, strict=False)
        refreshed = refresh_alignment_weights(model)
        print(
            f"[deeprecur] {variant} 臂：从 {inherited} 继承两塔 {len(state)} 个张量"
            f"（跳过 deepstack {dropped} 个；新件缺失 {len(missing)} / 意外 {len(unexpected)}）；"
            f"对齐件按公平口径重随机（{sum(refreshed.values())} 个 merger）"
        )
    return model


def _base_tower_state(base: str) -> tuple[dict, int]:
    """取 Qwen3-VL 检查点里可继承的两塔 + 词表头权重（deepstack 件剔除）。"""
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    root = Path(base)
    if not root.is_dir():
        root = Path(snapshot_download(base))
    index = root / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state, dropped = {}, 0
    for shard in shards:
        for key, tensor in load_file(root / shard).items():
            if key.startswith("model.visual.deepstack_merger_list."):
                dropped += 1
                continue
            if key.startswith(("model.visual.", "model.language_model.", "lm_head.")):
                state[key] = tensor
    return state, dropped


def refresh_alignment_weights(model) -> dict:
    """公平口径：把继承来的 projector / deepstack 权重重随机（按模型的初始化口径）。

    两个臂对等的前提：unified 臂的视觉对齐件是**从零学**的，所以 deepstack 臂也不许吃
    Qwen 用自家数据训好的 ``visual.merger`` 与 ``visual.deepstack_merger_list``。
    """
    prefixes = ("model.visual.merger", "model.visual.deepstack_merger_list")
    counts = {prefix: 0 for prefix in prefixes}
    for name, module in model.named_modules():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                if next(module.parameters(recurse=False), None) is not None:
                    model._init_weights(module)
                    counts[prefix] += 1
                break
    return counts


def tiny_unified_config(model_cfg: dict) -> dict:
    """unified（encoder-free）的 tiny 几何：文本用小档，视觉侧只有单 matmul 的嵌入器。"""
    text = tiny_config(model_cfg).text_config.to_dict()
    return {
        "text": text,
        "vision": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "mm_embed_dim": 64,
            "output_proj_dims": 64,
            "mm_posemb_size": 64,
        },
    }


def tiny_config(model_cfg: dict) -> Qwen3VLConfig:
    """Tiny 随机几何：词表与 vendored Qwen3-0.6B 对齐（special-token id 落在范围内）。"""
    t = model_cfg.get("tiny_geometry", {})
    text = {
        "vocab_size": int(t.get("vocab_size", 151669)),
        "hidden_size": int(t.get("hidden_size", 256)),
        "intermediate_size": int(t.get("intermediate_size", 768)),
        "num_hidden_layers": int(t.get("num_hidden_layers", 4)),
        "num_attention_heads": int(t.get("num_attention_heads", 8)),
        "num_key_value_heads": int(t.get("num_key_value_heads", 2)),
        "head_dim": int(t.get("head_dim", 32)),
        "max_position_embeddings": int(t.get("max_position_embeddings", 4096)),
    }
    vision = {
        "depth": int(t.get("vision_depth", 4)),
        "hidden_size": int(t.get("vision_hidden_size", 256)),
        "intermediate_size": int(t.get("vision_intermediate_size", 512)),
        "num_heads": int(t.get("vision_num_heads", 8)),
        # 钉住 patch 口径：Qwen3-VL 视觉塔是 16px patch，图像处理器必须同值（见 tiny_processor）
        "patch_size": int(t.get("vision_patch_size", 16)),
        # merger 的输出维度要跟 text hidden 对齐（默认是 8B 的 3584，tiny 不设就对不上）
        "out_hidden_size": text["hidden_size"],
        # tiny 塔只有几层，Qwen 默认的 deepstack 注入点 (8,16,24) 会越界，显式关掉
        "deepstack_visual_indexes": [],
    }
    return Qwen3VLConfig(text_config=text, vision_config=vision)


def load_processor(cfg: dict):
    """占位模型的 processor：unified 档用 encoder-free 的处理器（Qwen tokenizer + Gemma 图像侧）。"""
    from .models.variants import UNIFIED

    if cfg["model"].get("variant", "native") == "unified":
        from .models.transformers import build_unified_processor

        budget = cfg["model"].get("visual_token_budget") or UNIFIED.visual_token_budget
        tokenizer_dir = cfg["model"].get("placeholder", {}).get("load") or str(TOKENIZER_DIR)
        return build_unified_processor(tokenizer_dir, budget)
    if cfg["model"].get("tiny", False):
        return tiny_processor()
    load = cfg["model"].get("placeholder", {}).get("load")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(load or placeholder_name(cfg))


def tiny_processor() -> Qwen3VLProcessor:
    """离线拼一份 tiny 档 processor：vendored Qwen3 tokenizer + Qwen 图像处理器 + VL 模板。"""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return Qwen3VLProcessor(
        # patch 口径与 tiny 视觉塔一致（Qwen3-VL 是 16px，Qwen2 系默认 14）
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=VL_CHAT_TEMPLATE,
    )


def apply_freeze(model, freeze: list[str]) -> dict:
    """论文式冻结：``freeze`` 列出要冻的组（vision / projector / llm），其余可训。

    PT 档 ``[vision, llm]``（只训 projector）；SFT 档 ``[vision]``（LLM+projector 可训，
    视觉编码器冻结）；DeepStack-V/HD 变体 ``[]``（视觉编码器也训，配 ``vision_lr`` 分组学习率）。
    """
    frozen = set(freeze)
    if frozen - {"vision", "projector", "llm"}:
        raise SystemExit(f"[deeprecur] freeze 只认 vision/projector/llm：{sorted(frozen)}")
    for name, param in model.named_parameters():
        param.requires_grad = param_group(name) not in frozen
    return trainable_report(model)


def trainable_report(model) -> dict:
    """可训/冻结参数计数与分组明细（--dry-run 和训练前都会打）。"""
    stats: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        bucket = stats.setdefault(param_group(name), {"train": 0, "frozen": 0})
        bucket["train" if param.requires_grad else "frozen"] += param.numel()

    def _m(n: int) -> str:
        return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train == 0:
        raise SystemExit("[deeprecur] 没有任何可训练参数（freeze 配置把所有组都冻了）")
    for group, bucket in sorted(stats.items()):
        print(
            f"[deeprecur] {group}: 可训 {_m(bucket['train'])} / 冻结 {_m(bucket['frozen'])}"
        )
    print(f"[deeprecur] 合计可训 {n_train / 1e6:.1f}M 参数")
    return stats


def split_param_groups(model, base_lr: float, vision_lr: float | None) -> list[dict]:
    """按组拆 optimizer param groups：DeepStack-V/HD 的视觉编码器用独立学习率。"""
    if vision_lr is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]
    groups = {
        "vision": {"params": [], "lr": vision_lr},
        "other": {"params": [], "lr": base_lr},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups["vision" if param_group(name) == "vision" else "other"]["params"].append(param)
    return [g for g in groups.values() if g["params"]]


def data_root_note() -> str:
    """冒烟/文档用的产物根提示。"""
    return str(env_paths()["data"])


VL_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{ '<|im_start|>system\\n' + messages[0]['content'] | trim + '<|im_end|>\\n' }}"
    "{%- set __rest = messages[1:] %}"
    "{%- else %}{%- set __rest = messages %}{%- endif %}"
    "{%- for message in __rest %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{%- if message['content'] is string %}{{ message['content'] | trim }}"
    "{%- else %}"
    "{%- for part in message['content'] %}"
    "{%- if part['type'] == 'image' %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{%- elif part['type'] == 'text' %}{{ part['text'] | trim }}{%- endif %}"
    "{%- endfor %}{%- endif %}"
    "{{ '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)
