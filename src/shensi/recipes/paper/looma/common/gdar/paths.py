"""GDAR 配方的路径与文件系统约定。"""

from __future__ import annotations

import os
from pathlib import Path

import shensi.recipes.paper.gated_delta_attn_res as _recipe_pkg
from shensi.recipes.shensi.common import common as base

RECIPE = Path(_recipe_pkg.__file__).resolve().parent

TOKENIZER_ENV = "SHENSI_GDAR_TOKENIZER"
TOKENIZER_DIR = RECIPE / "tokenizer" / "Qwen3-0.6B"

_SHARED_GEOMS = RECIPE / "stage0_pretrain/stage1_pretrain/config/geoms"


def stage_dirs(stage: str) -> tuple[Path, Path]:
    from .algos import MODEL_ALGOS

    for cand in (RECIPE / stage, RECIPE / "stage0_pretrain" / stage, RECIPE / "stage2_rl" / stage):
        if cand.is_dir():
            return cand, cand / "config"
    known = sorted(
        {*MODEL_ALGOS}
        | {"stage1_pretrain", "stage2_midtrain", "stage1_sft", "stage2_*", "stage3_opd"}
    )
    raise SystemExit(f"[gdar] 找不到 stage 目录：{stage}（已知 stage：{known}）")


def env_paths() -> dict:
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["runs"] = str(Path(paths["runs"]) / "gated_delta_attn_res")
    paths["data"] = Path(paths["data"]) / "gated_delta_attn_res"
    return paths
