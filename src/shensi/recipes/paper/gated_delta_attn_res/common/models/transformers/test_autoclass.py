"""AutoConfig / AutoModel 分发单测（八个变体）。"""


from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import torch





REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from transformers import AutoConfig  # noqa: E402


VARIANTS: dict[str, tuple[str, str, str, dict]] = {
    "ar": (
        "Qwen3ARConfig",
        "Qwen3ARModel",
        "Qwen3ARForCausalLM",
        dict(attn_res_block_size=1, attn_res_output_route=False),
    ),
    "dar": (
        "Qwen3DARConfig",
        "Qwen3DARModel",
        "Qwen3DARForCausalLM",
        dict(attn_res_block_size=1, attn_res_use_null_source=True),
    ),
    "gdar": (
        "Qwen3GDARConfig",
        "Qwen3GDARModel",
        "Qwen3GDARForCausalLM",
        dict(
            attn_res_block_size=1,
            attn_res_gate_rank=16,
            attn_res_q_rank=16,
            attn_res_k_rank=16,
            attn_res_gate_param="deviation",
            attn_res_update="objective",
            attn_res_address="delta",
            attn_res_decay_ladder=8,
            attn_res_read_heads=2,
            attn_res_read_null=True,
            attn_res_read_whiten="diag",
        ),
    ),
    "hc": (
        "Qwen3HCConfig",
        "Qwen3HCModel",
        "Qwen3HCForCausalLM",
        dict(attn_res_block_size=1, hc_num_streams=2),
    ),
    "mhc": (
        "Qwen3MHCConfig",
        "Qwen3MHCModel",
        "Qwen3MHCForCausalLM",
        dict(attn_res_block_size=1, mhc_sinkhorn_iterations=5),
    ),


    "mudd": (
        "Qwen3MUDDConfig",
        "Qwen3MUDDModel",
        "Qwen3MUDDForCausalLM",
        dict(attn_res_block_size=1, mudd_num_ways=1),
    ),
    "denseformer": (
        "Qwen3DenseFormerConfig",
        "Qwen3DenseFormerModel",
        "Qwen3DenseFormerForCausalLM",
        dict(attn_res_block_size=1, attn_res_dwa_dilation=2),
    ),
}

BASE = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    max_position_embeddings=64,
)

TOKENS = 8


PACKAGE = "shensi.recipes.paper.gated_delta_attn_res.common.models.transformers"



FRESH_CODE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoConfig, AutoModelForCausalLM

ckpt, probe = sys.argv[1], sys.argv[2]
out = {}
cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
out["config_class"] = type(cfg).__name__
out["model_type"] = cfg.model_type
out["architectures"] = list(cfg.architectures or [])
out["extras"] = {k: getattr(cfg, k) for k in json.load(open(probe + "/extras.json"))}
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
out["model_class"] = type(model).__name__
ids = torch.load(probe + "/ids.pt", weights_only=True)
before = torch.load(probe + "/logits.pt", weights_only=True)
with torch.no_grad():
    after = model.eval()(ids).logits
out["logits_equal"] = bool(torch.equal(before, after))
print("__RESULT__" + json.dumps(out))
"""


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")
    return ok



SRC = Path(__file__).resolve().parents[7]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _import_models():
    try:
        import importlib  # noqa: PLC0415

        return importlib.import_module(f"{PACKAGE}"), None
    except Exception as exc:
        return None, exc


def _import_config_by_path(mod: str):
    path = Path(__file__).resolve().parent / f"configuration_qwen3_{mod}.py"
    spec = importlib.util.spec_from_file_location(f"_cfg_{mod}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    print("=" * 104)
    print("AutoClass / checkpoint conformance of the depth-connection variants")
    print("=" * 104)

    models, import_error = _import_models()
    have_models = models is not None
    if not have_models:
        print(f"\n  transformers {_tf_version()}: the modeling modules are not importable")
        print(f"    ({type(import_error).__name__}: {str(import_error).splitlines()[0]})")
        print("    -> A2/A3/A5 and the modeling half of A4 are SKIPPED; running the config half.")
        print("    Run this file with .venv/bin/python for the full battery.")

    results: list[bool] = []
    root = Path(tempfile.mkdtemp(prefix="autoclass_"))
    seen_model_types: dict[str, str] = {}

    for name, (cfg_name, model_name, lm_name, knobs) in VARIANTS.items():
        model_type = f"qwen3_{name}"
        print(f"\n{'=' * 104}\n{model_type}  ({cfg_name} / {lm_name})\n{'=' * 104}")


        cfg_cls = (
            getattr(models, cfg_name)
            if have_models
            else _import_config_by_path(name).__dict__[cfg_name]
        )
        results.append(
            check(
                "config declares its own model_type",
                cfg_cls.model_type == model_type,
                f"model_type = {cfg_cls.model_type!r}",
            )
        )
        duplicate = seen_model_types.get(cfg_cls.model_type)
        seen_model_types[cfg_cls.model_type] = name
        results.append(
            check(
                "model_type unique across variants",
                duplicate is None,
                f"model_type = {cfg_cls.model_type!r}, other = {duplicate}",
            )
        )
        results.append(
            check(
                "model_type is not 'qwen3'",
                cfg_cls.model_type != "qwen3",
                "would resolve to stock Qwen3Config otherwise",
            )
        )
        results.append(
            check(
                "auto_map is declared",
                set(cfg_cls.auto_map) == {"AutoConfig", "AutoModel", "AutoModelForCausalLM"},
                f"{sorted(cfg_cls.auto_map)}",
            )
        )

        cfg = cfg_cls(**BASE, **knobs)


        as_dict = cfg.to_dict()
        missing = [k for k in knobs if as_dict.get(k) != knobs[k]]
        results.append(
            check(
                "to_dict keeps every knob",
                not missing,
                f"{len(knobs)} non-default knobs, missing/wrong: {missing}",
            )
        )
        results.append(
            check(
                "to_dict keeps auto_map",
                "auto_map" in as_dict,
                f"auto_map = {as_dict.get('auto_map') is not None}",
            )
        )

        if not have_models:
            continue


        ckpt = root / name / "ckpt"
        probe = root / name / "probe"
        ckpt.mkdir(parents=True)
        probe.mkdir(parents=True)
        torch.manual_seed(0)
        model = getattr(models, lm_name)(cfg).eval()
        ids = torch.randint(0, BASE["vocab_size"], (1, TOKENS))
        with torch.no_grad():
            logits = model(ids).logits
        model.save_pretrained(ckpt)
        torch.save(ids, probe / "ids.pt")
        torch.save(logits, probe / "logits.pt")
        json.dump(sorted(knobs), open(probe / "extras.json", "w"))


        config_json = json.loads((ckpt / "config.json").read_text())
        results.append(
            check(
                "config.json model_type",
                config_json.get("model_type") == model_type,
                f"{config_json.get('model_type')!r}",
            )
        )
        results.append(
            check(
                "config.json architectures",
                config_json.get("architectures") == [lm_name],
                f"{config_json.get('architectures')}",
            )
        )
        auto_map = config_json.get("auto_map", {})
        results.append(
            check(
                "config.json auto_map complete",
                set(auto_map) == {"AutoConfig", "AutoModel", "AutoModelForCausalLM"},
                f"{sorted(auto_map)}",
            )
        )
        copied = sorted(p for p in os.listdir(ckpt) if p.endswith(".py"))
        results.append(
            check(
                "modules copied beside weights",
                copied == [f"configuration_qwen3_{name}.py", f"modeling_qwen3_{name}.py"],
                f"{copied}",
            )
        )
        results.append(
            check(
                "auto_map points at those files",
                all(f"{v.split('.')[0]}.py" in copied for v in auto_map.values()),
                f"{list(auto_map.values())}",
            )
        )


        reloaded = models.__dict__[lm_name].from_pretrained(ckpt).eval()
        with torch.no_grad():
            reloaded_logits = reloaded(ids).logits
        results.append(
            check(
                "reload is the same class",
                type(reloaded).__name__ == lm_name,
                f"{type(reloaded).__name__}",
            )
        )
        results.append(
            check(
                "logits bit-identical (torch.equal)",
                bool(torch.equal(logits, reloaded_logits)),
                f"max|d| = {(logits - reloaded_logits).abs().max():.3e}",
            )
        )


        auto_cfg = AutoConfig.from_pretrained(ckpt)
        results.append(
            check(
                "AutoConfig resolves the variant",
                type(auto_cfg).__name__ == cfg_name,
                f"{type(auto_cfg).__name__} (not Qwen3Config)",
            )
        )


        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [sys.executable, "-c", FRESH_CODE, str(ckpt), str(probe)],
            capture_output=True,
            text=True,
            cwd=str(root),
            env=env,
            timeout=900,
        )
        payload = next(
            (
                ln[len("__RESULT__") :]
                for ln in proc.stdout.splitlines()
                if ln.startswith("__RESULT__")
            ),
            None,
        )
        if payload is None:
            results.append(
                check(
                    "fresh process (package not imported)",
                    False,
                    f"rc={proc.returncode} {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}",
                )
            )
        else:
            fresh = json.loads(payload)
            ok = (
                fresh["config_class"] == cfg_name
                and fresh["model_class"] == lm_name
                and fresh["logits_equal"]
            )
            results.append(
                check(
                    "fresh process (package not imported)",
                    ok,
                    f"{fresh['config_class']} + {fresh['model_class']}, logits_equal={fresh['logits_equal']}",
                )
            )
            results.append(
                check(
                    "fresh process rebuilt every knob",
                    fresh["extras"] == {k: knobs[k] for k in knobs},
                    f"{fresh['extras']}",
                )
            )


        old = root / name / "old"
        shutil.copytree(ckpt, old)
        legacy = json.loads((old / "config.json").read_text())
        legacy["model_type"] = "qwen3"
        legacy.pop("auto_map", None)
        (old / "config.json").write_text(json.dumps(legacy, indent=2))
        legacy_cfg = AutoConfig.from_pretrained(old)
        results.append(
            check(
                "legacy 'qwen3' still parses",
                type(legacy_cfg).__name__ == "Qwen3Config",
                f"{type(legacy_cfg).__name__} (unchanged pre-fix behaviour)",
            )
        )



        legacy_explicit = models.__dict__[lm_name].from_pretrained(old, config=cfg).eval()
        with torch.no_grad():
            legacy_logits = legacy_explicit(ids).logits
        results.append(
            check(
                "legacy loads with explicit config",
                bool(torch.equal(logits, legacy_logits)),
                f"max|d| = {(logits - legacy_logits).abs().max():.3e}",
            )
        )

    passed = sum(results)
    print(f"\n{'=' * 104}")
    if not have_models:
        print(
            f"{passed}/{len(results)} config-level checks passed  (model checks skipped: transformers {_tf_version()})"
        )
    else:
        print(f"{passed}/{len(results)} checks passed")
    print(f"artifacts under {root}")
    print("=" * 104)
    return 0 if all(results) else 1


def _tf_version() -> str:
    import transformers

    return transformers.__version__


if __name__ == "__main__":
    raise SystemExit(main())
