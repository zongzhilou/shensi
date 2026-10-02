"""预训练与中训练的训练入口（stable / decay / 两个中训练档共用）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """预训练 / 中训练入口（按 stage 名装配配置并启动）。"""
    ap = argparse.ArgumentParser(description=f"{stage}（mcore 预训练 / 中训练）")
    common.add_common_train_args(ap)
    args = ap.parse_args(argv)
    return common.train_from_args(stage, args)
