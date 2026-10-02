"""路径约定：产物挂在本配方名下，分词器自带（全链路离线可用）。"""

from __future__ import annotations

import os
from pathlib import Path

from shensi.recipes.shensi.common import common as base

RECIPE = Path(__file__).resolve().parent.parent

#: 占位模型目录覆盖（指向本地已下载/已导出的模型目录）
MODEL_ENV = "SHENSI_DEEPRECUR_MODEL"

#: 冒烟分词器目录覆盖
TOKENIZER_ENV = "SHENSI_DEEPRECUR_TOKENIZER"
TOKENIZER_DIR = RECIPE / "common" / "tokenizer" / "Qwen3-0.6B"


def env_paths() -> dict:
    """路径表：shensi 默认值之上，把 data/runs/ckpt 收进本配方子目录。"""
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["data"] = str(Path(paths["data"]) / "deeprecur")
    paths["runs"] = str(Path(paths["runs"]) / "deeprecur")
    paths["ckpt"] = str(Path(paths["ckpt"]) / "deeprecur")
    return paths


def stage_dirs(stage: str) -> Path:
    """定位 stage 目录。"""
    cand = RECIPE / stage
    if cand.is_dir():
        return cand
    raise SystemExit(f"[deeprecur] 找不到 stage 目录：{cand}")
