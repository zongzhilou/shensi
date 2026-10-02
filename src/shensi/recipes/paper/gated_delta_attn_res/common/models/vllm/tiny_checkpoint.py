"""Build a *tiny, random-weight* checkpoint of any of the 7 variants.

The point is a checkpoint that is small enough to smoke-test an inference engine on
a laptop GPU but that still has the connection switched on and is self-describing
(``config.json`` carries the variant's own ``model_type`` + ``auto_map``, and
``save_pretrained`` copies the two ``.py`` files next to the weights).  Nothing here
depends on ``models`` being importable by the consumer -- that is the whole point of
``auto_map``: a fresh process with only ``transformers`` + ``trust_remote_code=True``
can load it.

Usage
-----
    from .tiny_checkpoint import build
    info = build("gdar", "/tmp/rollout_smoke/gdar")
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from ._paths import DEFAULT_TOKENIZER, ensure_src_on_path

#: the HF reference package that carries the seven ``configuration_*``/``modeling_*`` pairs
HF_PACKAGE = "shensi.recipes.paper.gated_delta_attn_res.common.models.transformers"

ensure_src_on_path()

from .variants import BY_KEY, SHAPES, Variant, base_of  # noqa: E402

#: A real Qwen3 tokenizer, already on disk (no network needed).  A converted
#: FlagScale checkpoint would carry exactly this directory next to its weights.
DEFAULT_TOKENIZER = DEFAULT_TOKENIZER


def build(
    variant_key: str,
    out_dir: str | Path,
    *,
    shape: str = "tiny",
    tokenizer_dir: str | Path | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    """Materialise a random-weight checkpoint; returns a small report dict.

    ``shape`` picks the backbone geometry (``.variants.SHAPES``): ``tiny``
    (2 layers, hidden 64) for the fast smoke test, ``0.6b`` (28 layers, hidden 1024,
    16 heads) for a run at the width the real models have.
    """
    import torch
    from transformers import AutoTokenizer

    variant: Variant = BY_KEY[variant_key]
    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tok_dir = Path(tokenizer_dir) if tokenizer_dir else DEFAULT_TOKENIZER
    if not tok_dir.is_dir():
        raise FileNotFoundError(
            f"tokenizer directory {tok_dir} not found; pass tokenizer_dir=... "
            f"(a Qwen3 tokenizer directory is needed so the tiny model can be prompted with text)"
        )

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    vocab_size = len(tokenizer)

    # 1. tokenizer first: the model's save_pretrained then writes config.json /
    #    generation_config.json / the two remote-code .py files on top of it.
    tokenizer.save_pretrained(out)

    # 2. import *this repo's* implementation -- build-time only.  The saved
    #    checkpoint does not need it (auto_map + the copied .py files cover that).
    import importlib

    cfg_mod = importlib.import_module(f"{HF_PACKAGE}.{variant.config_file}")
    model_mod = importlib.import_module(f"{HF_PACKAGE}.{variant.model_file}")
    config_cls = getattr(cfg_mod, variant.config_class)
    model_cls = getattr(model_mod, variant.model_class)

    torch.manual_seed(seed)
    config = config_cls(**base_of(shape, vocab_size), **variant.tiny_knobs)
    # The engine reads the stop tokens off the *model* config (vLLM: eos_token_id /
    # pad_token_id on the HF config), and the tiny backbone leaves them unset, so a
    # generation run would have no stopping criterion at all.
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = model_cls(config)
    n_params = sum(p.numel() for p in model.parameters())

    model.save_pretrained(out, safe_serialization=True)

    # 3. generation_config: keep the tokenizer's stop tokens, add nothing else.
    gen = (
        json.loads((out / "generation_config.json").read_text())
        if (out / "generation_config.json").exists()
        else {}
    )
    gen.setdefault("bos_token_id", tokenizer.bos_token_id or tokenizer.eos_token_id)
    gen.setdefault("eos_token_id", tokenizer.eos_token_id)
    gen.setdefault("pad_token_id", tokenizer.pad_token_id or tokenizer.eos_token_id)
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2, sort_keys=True) + "\n")

    saved = json.loads((out / "config.json").read_text())
    report = {
        "variant": variant.key,
        "shape": shape,
        "out_dir": str(out),
        "architecture": saved.get("architectures"),
        "model_type": saved.get("model_type"),
        "auto_map": saved.get("auto_map"),
        "remote_code_files": sorted(p.name for p in out.glob("*.py")),
        "parameters": n_params,
        "vocab_size": vocab_size,
        "tiny_knobs": variant.tiny_knobs,
        "files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variant", choices=sorted(BY_KEY))
    ap.add_argument("out_dir")
    ap.add_argument(
        "--shape",
        default="tiny",
        choices=sorted(SHAPES),
        help="backbone geometry: tiny (2x64) or 0.6b (28x1024)",
    )
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    rep = build(
        args.variant,
        args.out_dir,
        shape=args.shape,
        tokenizer_dir=args.tokenizer_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    for k, v in rep.items():
        print(f"{k:20s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
