"""路径与文件系统约定（looma 同款：复用 shensi 基座，只加本配方的东西）。

tokenizer **保持 shensi 的 DSV4F 默认**（`$SHENSI_TOKENIZER`，默认
`$SHENSI_FS/models/DeepSeek-V4-Flash-0731`）——配方只在它之上做**增量**特殊 token 扩展
（见 vl_tokens.py），基座词表一字不动。

processor / image_processor 用 Kimi K3 的（moonshotai/Kimi-K3 发布件，trust_remote_code）；
默认先找本地镜像 `$SHENSI_FS/models/Kimi-K3`，没有就落到 hub id。
"""

from __future__ import annotations

import os
from pathlib import Path

from shensi.recipes.shensi.common import common as base

RECIPE = Path(__file__).resolve().parent.parent

# 环境变量名（都有 shensi 侧的默认值，见 env_paths）
PROCESSOR_ENV = "SHENSI_VL_PROCESSOR"  # Kimi K3 processor / image_processor 来源
VISION_ENV = "SHENSI_VL_VISION"  # 视觉塔（HF ViT 目录或 hub id）
LLM_ENV = "SHENSI_VL_LLM"  # 语言骨干（HF 目录；默认与 shensi RL 的 model.path 同源）
VL_TOK_ENV = "SHENSI_VL_TOKENIZER"  # 扩展后 tokenizer 的落点（base 同 DSV4F + 原语/图像 token）

KIMI_K3_HUB = "moonshotai/Kimi-K3"


def _safe_is_dir(p: str | Path) -> bool:
    """存在性探测不因权限炸（/root/work 在非 root 下 stat 会 EACCES）。"""
    try:
        return Path(p).is_dir()
    except OSError:
        return False


def env_paths() -> dict:
    """返回路径表：shensi 的默认值 + 本配方的 processor / vision / 扩展 tokenizer / 数据子目录。"""
    paths = base.env_paths()
    fs_root = Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
    # tokenizer：与 shensi 完全一致（DSV4F）；扩展副本另存，不改原目录
    paths["vl_tokenizer"] = os.environ.get(VL_TOK_ENV) or str(
        fs_root / "shensi/models/shensi-vl-tok"
    )
    # Kimi K3 processor / image_processor：本地镜像优先，其次 hub id
    local_proc = os.environ.get(PROCESSOR_ENV) or str(fs_root / "models/Kimi-K3")
    paths["processor"] = local_proc if _safe_is_dir(local_proc) else KIMI_K3_HUB
    # 视觉塔：论文的 DeepSeek-ViT 是 in-house 未开源；开源替代默认给 Kimi ViT 权重的落地目录
    # （与 processor 同口径 14×14 patch），冒烟档可缺省（modeling_vl 会随机初始化小 ViT）。
    paths["vision"] = os.environ.get(VISION_ENV) or str(fs_root / "models/Kimi-K3-ViT")
    # 语言骨干：与 shensi 的 verl model.path 同一份 HF 目录
    paths["llm"] = os.environ.get(LLM_ENV) or str(fs_root / "models/DeepSeek-V4-Flash-0731")
    paths["vl_images"] = fs_root / "datasets/llm/images"  # 图像落地根（按数据集名分目录）
    paths["data"] = Path(paths["data"]) / "shensi_vl"
    paths["runs"] = Path(paths["runs"]) / "shensi_vl"
    paths["ckpt"] = Path(paths["ckpt"]) / "shensi_vl"
    return paths


def stage_dirs(stage: str) -> Path:
    """定位 stage 目录：``<recipe>/<stage>``，或分组下的一层（``stage2_rl/`` …）。"""
    for cand in (RECIPE / stage, *sorted(RECIPE.glob(f"*/{stage}"))):
        if cand.is_dir():
            return cand
    raise SystemExit(f"[shensi_vl] 找不到 stage 目录：{stage}")
