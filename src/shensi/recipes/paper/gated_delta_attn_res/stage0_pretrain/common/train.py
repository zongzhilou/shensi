"""预训练段共用的训练入口（PT-1 stable / PT-2 decay / 中训练两段同一套参数）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage}（mcore 预训练 / 中训练）")
    common.add_common_train_args(ap)
    args = ap.parse_args(argv)
    return common.train_from_args(stage, args)
