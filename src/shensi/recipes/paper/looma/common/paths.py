"""路径与文件系统约定。"""

from __future__ import annotations

import os
from pathlib import Path

from shensi.recipes.shensi.common import common as base

RECIPE = Path(__file__).resolve().parent.parent

TOKENIZER_ENV = "SHENSI_LOOMA_TOKENIZER"
TOKENIZER_DIR = RECIPE / "common" / "tokenizer" / "MiniCPM5-2B"


def env_paths() -> dict:
    """返回路径表：shensi 的默认值加上本配方的 tokenizer、data / runs / ckpt 子目录。

    三处都挂在 ``looma/`` 下，本配方的东西不散到 shensi 的公共目录里。
    """
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["data"] = str(Path(paths["data"]) / "looma")
    paths["runs"] = str(Path(paths["runs"]) / "looma")
    paths["ckpt"] = str(Path(paths["ckpt"]) / "looma")
    return paths


def stage_dirs(stage: str) -> Path:
    """定位 stage 目录：``<recipe>/<stage>``，或分组下的一层（``stage0_pretrain/``、``stage2_rl/`` …）。"""
    for cand in (RECIPE / stage, *sorted(RECIPE.glob(f"*/{stage}"))):
        if cand.is_dir():
            return cand
    raise SystemExit(f"[looma] 找不到 stage 目录：{stage}")
