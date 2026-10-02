"""路径与文件系统约定（deeprecur 配方）。"""

from __future__ import annotations

import os
from pathlib import Path

from shensi.recipes.shensi.common import common as base

RECIPE = Path(__file__).resolve().parent.parent

#: 占位模型的本地目录覆盖（指向一个已下载/已导出的 Qwen3-VL 目录，避免训练机上拉网）
MODEL_ENV = "SHENSI_DEEPRECUR_MODEL"

#: 冒烟档的分词器（与 Qwen3VLConfig 默认 special-token id 对齐的 Qwen3 词表，全链路离线）
TOKENIZER_ENV = "SHENSI_DEEPRECUR_TOKENIZER"
TOKENIZER_DIR = RECIPE / "common" / "tokenizer" / "Qwen3-0.6B"


def env_paths() -> dict:
    """返回路径表：shensi 的默认值加上本配方的 tokenizer 与 data / runs / ckpt 子目录。

    三处产物都挂在 ``deeprecur/`` 下，本配方的东西不散到 shensi 的公共目录里。
    """
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["data"] = str(Path(paths["data"]) / "deeprecur")
    paths["runs"] = str(Path(paths["runs"]) / "deeprecur")
    paths["ckpt"] = str(Path(paths["ckpt"]) / "deeprecur")
    return paths


def stage_dirs(stage: str) -> Path:
    """定位 stage 目录：``<recipe>/<stage>``。"""
    cand = RECIPE / stage
    if cand.is_dir():
        return cand
    raise SystemExit(f"[deeprecur] 找不到 stage 目录：{cand}")
