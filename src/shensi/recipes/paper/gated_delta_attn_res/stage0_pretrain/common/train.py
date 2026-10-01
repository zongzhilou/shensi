"""预训练段的训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage}（mcore 预训练 / 中训练）")
    common.add_common_train_args(ap)
    args = ap.parse_args(argv)
    return common.train_from_args(stage, args)
