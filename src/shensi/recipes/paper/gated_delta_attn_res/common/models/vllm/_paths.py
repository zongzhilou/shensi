"""vLLM 侧脚本共用的路径：把 src/ 与配方根放上 sys.path。"""


from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[7]

RECIPE = Path(__file__).resolve().parents[3]

DEFAULT_TOKENIZER = RECIPE / "common" / "tokenizer" / "Qwen3-0.6B"


def ensure_src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
