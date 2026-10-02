#!/usr/bin/env python3
"""造三臂（unified / gdar / deeprecur）的 tiny 检查点：随机权重 + 真处理器 + remote-code 垫片。

产物 = 标准 HF 目录（config 带 model_type + auto_map）+ 预算化/原生处理器 + **垫片文件**
（``configuration_*.py`` / ``modeling_*.py``，内容只是从已安装的 shensi 包转发导入）——
这样 ``trust_remote_code=True`` 的加载器（transformers / vLLM 的 Transformers 后端）能在
ckpt 目录里解析到我们包内的真实实现。用途：不训练就验证各臂在引擎里的推理链。

python -m shensi.recipes.paper.deeprecur.common.models.vllm.tiny_checkpoint --arm deeprecur --out /tmp/q3vl_recur_tiny
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..variants import UNIFIED

TINY_TEXT = {
    "vocab_size": 151669,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 4096,
}
TINY_VISION = {
    "depth": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 2,
    "patch_size": 16,
    "out_hidden_size": 128,
    "num_position_embeddings": 256,
    "deepstack_visual_indexes": [],
}
TINY_UNIFIED_VISION = {
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "mm_embed_dim": 64,
    "output_proj_dims": 64,
    "mm_posemb_size": 64,
}

ARMS = ("unified", "gdar", "deeprecur")

_GDAR_SHIMS = {
    "configuration_qwen3_vl_gdar.py": (
        "from shensi.recipes.paper.deeprecur.common.models.transformers."
        "configuration_qwen3_vl_gdar import Qwen3VLGdarConfig  # noqa: F401\n"
    ),
    "modeling_qwen3_vl_gdar.py": (
        "from shensi.recipes.paper.deeprecur.common.models.transformers.modeling_qwen3_vl_gdar "
        "import (  # noqa: F401\n"
        "    Qwen3VLGdarDeepRecurForConditionalGeneration,\n"
        "    Qwen3VLGdarDeepRecurModel,\n"
        "    Qwen3VLGdarForConditionalGeneration,\n"
        "    Qwen3VLGdarModel,\n"
        ")\n"
    ),
}
_UNIFIED_SHIMS = {
    "configuration_qwen3_vl_unified.py": (
        "from shensi.recipes.paper.deeprecur.common.models.transformers.configuration "
        "import Qwen3VLUnifiedConfig  # noqa: F401\n"
    ),
    "modeling_qwen3_vl_unified.py": (
        "from shensi.recipes.paper.deeprecur.common.models.transformers."
        "modeling_qwen3_vl_unified import (  # noqa: F401\n"
        "    Qwen3VLUnifiedForConditionalGeneration,\n"
        "    Qwen3VLUnifiedModel,\n"
        ")\n"
    ),
    "processing_qwen3_vl_unified.py": (
        "from shensi.recipes.paper.deeprecur.common.models.transformers."
        "modeling_qwen3_vl_unified import Qwen3VLUnifiedProcessor  # noqa: F401\n"
    ),
}


def _write_shims(out: Path, arm: str) -> list[str]:
    shims = _UNIFIED_SHIMS if arm == "unified" else _GDAR_SHIMS
    for name, body in shims.items():
        (out / name).write_text(body, encoding="utf-8")
    return sorted(shims)


def build(
    out: Path,
    *,
    arm: str = "unified",
    budget: int = UNIFIED.visual_token_budget,
    recur_blocks: int = 2,
    seed: int = 0,
) -> dict:
    """在 ``out`` 造 tiny ckpt（指定臂；含处理器与 remote-code 垫片），返回报告。"""
    import shutil

    import torch

    from ...paths import TOKENIZER_DIR

    if arm not in ARMS:
        raise SystemExit(f"[deeprecur·vllm] 臂只认 {ARMS}：{arm}")

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    torch.manual_seed(seed)

    if arm == "unified":
        from ..transformers.modeling_qwen3_vl_unified import (
            build_unified,
            build_unified_processor,
        )

        model = build_unified(
            tiny_config={"text": dict(TINY_TEXT), "vision": dict(TINY_UNIFIED_VISION)},
            dtype=torch.float32,
        )[0]
        processor = build_unified_processor(str(TOKENIZER_DIR), budget)
        vision_params = sum(p.numel() for p in model.model.vision_embedder.parameters())
    else:
        from ...model import tiny_processor
        from ..transformers.modeling_qwen3_vl_gdar import (
            Qwen3VLGdarConfig,
            Qwen3VLGdarDeepRecurForConditionalGeneration,
            Qwen3VLGdarForConditionalGeneration,
        )

        config = Qwen3VLGdarConfig(
            text_config=dict(TINY_TEXT), vision_config=dict(TINY_VISION), recur_blocks=recur_blocks
        )
        cls = (
            Qwen3VLGdarDeepRecurForConditionalGeneration
            if arm == "deeprecur"
            else Qwen3VLGdarForConditionalGeneration
        )
        model = cls(config).to(torch.float32)
        # auto_map 按本臂改写（引擎/transformers 会按它解析模型类）
        model.config.auto_map = dict(getattr(model.config, "auto_map", {}) or {})
        model.config.auto_map.update(
            {
                "AutoModel": f"modeling_qwen3_vl_gdar.{cls.__name__.replace('ForConditionalGeneration', 'Model')}",
                "AutoModelForCausalLM": f"modeling_qwen3_vl_gdar.{cls.__name__}",
                "AutoModelForConditionalGeneration": f"modeling_qwen3_vl_gdar.{cls.__name__}",
            }
        )
        processor = tiny_processor()
        vision_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if "self_attention_attn_res" in n or "mlp_attn_res" in n
        )

    model.save_pretrained(out)
    processor.save_pretrained(out)
    shims = _write_shims(out, arm)

    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "out": str(out),
        "arm": arm,
        "architectures": list(getattr(model.config, "architectures", []) or []),
        "params": n_params,
        "arm_params": vision_params,
        "shims": shims,
        "files": sorted(p.name for p in out.iterdir()),
    }
    print(
        f"[deeprecur·vllm] tiny ckpt（arm={arm}）→ {out}（{n_params / 1e6:.1f}M 参数；"
        f"本臂自研件 {vision_params / 1e3:.0f}K；垫片 {len(shims)} 个）"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="deeprecur 三臂的 tiny 检查点")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", default="unified", choices=list(ARMS))
    parser.add_argument("--budget", type=int, default=UNIFIED.visual_token_budget)
    parser.add_argument("--recur-blocks", type=int, default=2)
    args = parser.parse_args()
    build(
        Path(args.out), arm=args.arm, budget=args.budget, recur_blocks=args.recur_blocks
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
