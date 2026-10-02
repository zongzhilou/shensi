"""HF 参考权重构建：从配置造出参考 state_dict 供对拍。"""


from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from .profiles import PROFILES, Profile



CLASSES = {
    "ar": ("Qwen3ARConfig", "Qwen3ARForCausalLM"),
    "dar": ("Qwen3DARConfig", "Qwen3DARForCausalLM"),
    "gdar": ("Qwen3GDARConfig", "Qwen3GDARForCausalLM"),
    "denseformer": ("Qwen3DenseFormerConfig", "Qwen3DenseFormerForCausalLM"),
    "realformer": ("Qwen3RealFormerConfig", "Qwen3RealFormerForCausalLM"),
    "hc": ("Qwen3HCConfig", "Qwen3HCForCausalLM"),
    "mhc": ("Qwen3MHCConfig", "Qwen3MHCForCausalLM"),
    "mudd": ("Qwen3MUDDConfig", "Qwen3MUDDForCausalLM"),
}

__all__ = ["build_hf_state_dict", "main", "CLASSES"]

SEED = 0


PROBE_SEQ = 16


PROBE_SEED = 1234


def build_hf_state_dict(profile: Profile, seed: int = SEED) -> dict[str, torch.Tensor]:
    """按配置造出 HF 参考权重（对拍用）。"""
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
    """造一个小探测模型（形状检查用）。"""
    generator = torch.Generator().manual_seed(PROBE_SEED)
    vocab = int(profile.config_kwargs()["vocab_size"])
    input_ids = torch.randint(0, vocab, (1, PROBE_SEQ), generator=generator)
    with torch.no_grad():
        logits = model(input_ids).logits
    return input_ids, logits.float().detach().clone()


def main(argv: list[str] | None = None) -> int:
    """参考权重构建入口。"""
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
