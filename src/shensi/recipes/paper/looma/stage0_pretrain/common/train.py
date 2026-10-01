"""预训练段共用的训练入口（PT-1 stable / PT-2 decay / 中训练两段同一套参数）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.looma.common import add_common_train_args, train_from_args

__all__ = ["train_main"]


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """训练入口：公共参数 + 组配置 + 起训。"""
    parser = argparse.ArgumentParser(description=f"{stage}（mcore 预训练 / 中训练）")
    add_common_train_args(parser)
    return train_from_args(stage, parser.parse_args(argv))
