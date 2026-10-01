"""Produce the HF half of the audit: real checkpoints, saved to disk.

Runs under the **transformers-5** environment (``code/.venv``), because that is
the only interpreter in this repository that can import
``models/modeling_qwen3_*.py`` (see ``VERL_REGISTRATION.md`` §1, blocking B: the
modeling files need ``merge_with_config_defaults`` and a dataclass
``PretrainedConfig``, both of which arrive with transformers v5).  The
Megatron/mbridge environment has 4.57.6 and cannot import them.

So the two halves are split by an intermediate file instead of by a fake model:
this script writes ``<out>/<profile>.pt`` containing the *real* HF
``state_dict()`` -- built by the real modeling code, with ``torch.manual_seed``
fixed so the run is reproducible -- and :mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.audit` loads it
and runs the conversion against a real Megatron model.

Usage::

    code/.venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.hf_reference --out /tmp/hf_ref

Exit code 0 means every profile was written.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from .profiles import PROFILES, Profile

#: variant -> (config class name, model class name); the class names are the ones
#: the HF files declare, and the audit checks them again on the Megatron side.
CLASSES = {
    "ar": ("Qwen3ARConfig", "Qwen3ARForCausalLM"),
    "dar": ("Qwen3DARConfig", "Qwen3DARForCausalLM"),
    "gdar": ("Qwen3GDARConfig", "Qwen3GDARForCausalLM"),
    "denseformer": ("Qwen3DenseFormerConfig", "Qwen3DenseFormerForCausalLM"),
    "hc": ("Qwen3HCConfig", "Qwen3HCForCausalLM"),
    "mhc": ("Qwen3MHCConfig", "Qwen3MHCForCausalLM"),
    "mudd": ("Qwen3MUDDConfig", "Qwen3MUDDForCausalLM"),
}

__all__ = ["build_hf_state_dict", "main", "CLASSES"]

SEED = 0

#: length of the forward-probe sequence saved with every checkpoint
PROBE_SEQ = 16
#: seed for the probe's *input*; independent of the model-init seed, so the two can
#: be varied separately
PROBE_SEED = 1234


def build_hf_state_dict(profile: Profile, seed: int = SEED) -> dict[str, torch.Tensor]:
    """Instantiate the HF model for ``profile`` and return its ``state_dict()``."""
    config_class, model_class = CLASSES[profile.variant]
    module = importlib.import_module(f"models.modeling_qwen3_{profile.variant}")
    config_module = importlib.import_module(f"models.configuration_qwen3_{profile.variant}")
    config = getattr(config_module, config_class)(**profile.config_kwargs())
    torch.manual_seed(seed)
    model = getattr(module, model_class)(config)
    model.eval()
    return model


def _state_dict(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def build_probe(model, profile: Profile) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed input and the HF model's logits for it.

    This is what makes the audit more than a name check.  The tensor round trip
    proves the *values* landed in the right places; it cannot see a permutation
    that is its own inverse, a wrong axis, or a Python-level arithmetic difference
    between the two implementations.  A forward pass on a fixed input can, and it
    is the closest thing to "the converted model *is* the reference model" that is
    available without a rollout engine (``VERL_REGISTRATION.md`` §7).
    """
    generator = torch.Generator().manual_seed(PROBE_SEED)
    vocab = int(profile.config_kwargs()["vocab_size"])
    input_ids = torch.randint(0, vocab, (1, PROBE_SEQ), generator=generator)
    with torch.no_grad():
        logits = model(input_ids).logits
    return input_ids, logits.float().detach().clone()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="directory for <profile>.pt files")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these profile names")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for profile in PROFILES:
        if args.only and profile.name not in args.only:
            continue
        try:
            model = build_hf_state_dict(profile, args.seed)
            state = _state_dict(model)
        except Exception as exc:  # noqa: BLE001 - one broken profile must not hide the rest
            print(f"[hf_reference] {profile.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            index[profile.name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        payload = {"state_dict": state, "variant": profile.variant, "knobs": profile.knobs}
        try:
            input_ids, logits = build_probe(model, profile)
            payload |= {"input_ids": input_ids, "logits": logits, "probe_seq": PROBE_SEQ}
        except Exception as exc:  # noqa: BLE001 - the tensor half is still usable
            print(f"[hf_reference] {profile.name}: forward probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        total = sum(t.numel() for t in state.values())
        path = args.out / f"{profile.name}.pt"
        torch.save(payload, path)
        index[profile.name] = {
            "ok": True,
            "tensors": len(state),
            "scalars": total,
            "file": path.name,
            "default_dtype": str(next(iter(state.values())).dtype),
            "probe": "logits" in payload,
        }
        print(f"[hf_reference] {profile.name:18s} {len(state):3d} tensors  {total:>10,} scalars  -> {path.name}")

    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    failed = [name for name, entry in index.items() if not entry["ok"]]
    if failed:
        print(f"[hf_reference] FAILED for {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
