"""路径约定：配方根、自带 tokenizer、stage 目录定位与产物目录。"""

from __future__ import annotations

import os
from pathlib import Path

from shensi.recipes.shensi.common import common as base

RECIPE = Path(__file__).resolve().parent.parent

TOKENIZER_ENV = "SHENSI_LOOMA_TOKENIZER"
TOKENIZER_DIR = RECIPE / "common" / "tokenizer" / "MiniCPM5-2B"


def env_paths() -> dict:
    """返回路径表：在 shensi 默认值基础上换成自带的 tokenizer 与配方专属的产物目录。"""
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["data"] = str(Path(paths["data"]) / "looma")
    paths["runs"] = str(Path(paths["runs"]) / "looma")
    paths["ckpt"] = str(Path(paths["ckpt"]) / "looma")
    return paths


def stage_dirs(stage: str) -> Path:
    """定位 stage 目录（含分组下的一层，如 stage0_pretrain/、stage2_rl/）。"""
    for cand in (RECIPE / stage, *sorted(RECIPE.glob(f"*/{stage}"))):
        if cand.is_dir():
            return cand
    raise SystemExit(f"[looma] 找不到 stage 目录：{stage}")
