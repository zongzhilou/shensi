"""Paths shared by the vLLM-side rollout utilities.

``SRC`` is the checkout's ``src/`` directory -- putting it on ``sys.path`` makes
``import shensi...`` work in any interpreter that has the dependencies, which is what
the engine's worker processes need (they never inherit our driver's ``sys.path``).
``RECIPE`` is the recipe root that carries the vendored Qwen3 tokenizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: <repo>/src
SRC = Path(__file__).resolve().parents[6]
#: <repo>/src/shensi/recipes/paper/gated_delta_attn_res
RECIPE = Path(__file__).resolve().parents[2]
#: the Qwen3 tokenizer vendored by this recipe (a real tokenizer, no network needed)
DEFAULT_TOKENIZER = RECIPE / "tokenizer" / "Qwen3-0.6B"


def ensure_src_on_path() -> None:
    """Make ``import shensi...`` work from this interpreter (idempotent)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
