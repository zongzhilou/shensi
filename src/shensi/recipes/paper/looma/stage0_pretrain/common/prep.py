"""预训练段共用的语料准备：配比 json → Megatron bin/idx。"""

from __future__ import annotations

from pathlib import Path

from shensi.recipes.paper.looma.common.prep import prep_main as _prep_main

__all__ = ["prep_main"]


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """语料准备入口；参数与配置见 ``common/prep.py``。"""
    return _prep_main(stage, here, argv)
