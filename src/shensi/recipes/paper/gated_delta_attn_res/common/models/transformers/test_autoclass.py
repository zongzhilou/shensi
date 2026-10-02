"""Checkpoint conformance for the seven depth-connection variants (AutoClass path).

The blocker this file guards against: every variant used to declare
``model_type = "qwen3"``, so a checkpoint written by ``Qwen3GDARForCausalLM``
resolved -- in a process that had not imported this package -- to *stock* Qwen3,
and the first ``attn_res_*`` attribute access died with
``AttributeError: 'Qwen3Config' object has no attribute ...``.  That is exactly
the lookup verl's ``MegatronWorker`` does
(``AutoConfig.from_pretrained(local_path, trust_remote_code=...)``) and the one
the vLLM rollout does, so it blocked the RL path at the very first step.

Each variant now has its own ``model_type``, registers itself with
``AutoConfig``/``AutoModelForCausalLM`` on import, and marks itself as
checkpoint-local code so ``save_pretrained`` writes ``auto_map`` and copies the
two modules next to the weights.

Checks, per variant
-------------------
A1  ``config.json``: the variant's own ``model_type``, a ``Qwen3<X>ForCausalLM``
    architecture, a complete ``auto_map``, and both ``.py`` files beside the
    weights (that is what makes the directory self-describing).
A2  round trip: ``save_pretrained`` -> ``AutoModelForCausalLM.from_pretrained``
    **with no ``config=`` argument** -> logits ``torch.equal`` to the model that
    was saved.  Every variant's knobs are moved off their defaults first, so a
    field that fails to serialise cannot pass by accident.
A3  ``AutoConfig.from_pretrained(path)`` resolves to the variant's config class
    rather than ``Qwen3Config``.
A4  **fresh interpreter** with this package *not* importable, i.e. the real
    deployment shape: ``AutoConfig.from_pretrained(path, trust_remote_code=True)``
    and ``AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)``
    both resolve, and the reloaded logits are ``torch.equal`` to the saved ones.
A5  backward compatibility: a checkpoint whose ``config.json`` still says
    ``model_type = "qwen3"`` keeps working -- ``AutoConfig`` still yields stock
    ``Qwen3Config`` (nothing that used to load is broken) and the model still
    loads bit-exactly when the variant config is passed explicitly.

Run:  .venv/bin/python models/transformers/test_autoclass.py
      .venv/bin/python models/transformers/test_autoclass.py   # transformers 4.x 下自动只跑 config 半边

The last form is useful because the two environments differ: ``.venv`` has
transformers 5 (the modeling modules need it) while the verl environment has
transformers 4.57, where only the *configuration* modules are importable.  The
script detects that and reports the model checks as skipped rather than failing.
"""

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

#: 本包目录本身（configuration_/modeling_ 就住在这里）。**不能**把上一层（models/）放上
#: sys.path：那会让 `import transformers` 解析到 models/transformers/ ——本包的名字与真库
#: 撞名，随后 `from transformers import AutoConfig` 直接 circular import。所以插入的是本包
#: 目录，让这七个 `modeling_qwen3_*` 以顶层模块名可导入（原包的用法），真库照常解析。
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from transformers import AutoConfig  # noqa: E402

#: variant -> (config class, base model class, causal-LM class, non-default knobs)
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
    # ``mudd_num_ways`` is documented as 4 (qkvr) or 1 (single stream); 4 is the
    # default, so 1 is the only non-default value that builds.
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

#: this package's import path (the seven pairs live here)
PACKAGE = "shensi.recipes.paper.gated_delta_attn_res.common.models.transformers"

# Run in a fresh interpreter: this package is not imported anywhere, so the only way the
# config can resolve is through ``auto_map`` in the checkpoint directory.
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


#: <repo>/src —— checkout 未安装（没有 editable install）时也能 ``import shensi...``。
SRC = Path(__file__).resolve().parents[7]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _import_models():
    """Import this package; returns ``(module, None)`` or ``(None, exception)``."""
    try:
        import importlib  # noqa: PLC0415

        return importlib.import_module(f"{PACKAGE}"), None
    except Exception as exc:  # transformers 4.x cannot import the modeling half
        return None, exc


def _import_config_by_path(mod: str):
    """Load ``configuration_qwen3_<mod>.py`` straight from the file.

    This is the transformers-4-compatible half: the file imports nothing from
    this package, which is what lets the verl environment read a checkpoint's
    config without the modeling modules being importable.
    """
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

        # ---- config module: importable standalone, right name, right extras ----
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
        # v5 serialises dataclass fields natively; v4 only sees instance attributes,
        # so this is where the ``to_dict`` override has to prove itself.
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

        # ---------------------------- build + save ----------------------------
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

        # ------------------------------- A1 ----------------------------------
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

        # ------------------------------- A2 ----------------------------------
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

        # ------------------------------- A3 ----------------------------------
        auto_cfg = AutoConfig.from_pretrained(ckpt)
        results.append(
            check(
                "AutoConfig resolves the variant",
                type(auto_cfg).__name__ == cfg_name,
                f"{type(auto_cfg).__name__} (not Qwen3Config)",
            )
        )

        # ------------------------------- A4 ----------------------------------
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

        # ------------------------------- A5 ----------------------------------
        old = root / name / "old"
        shutil.copytree(ckpt, old)
        legacy = json.loads((old / "config.json").read_text())
        legacy["model_type"] = "qwen3"  # what every variant used to write
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
        # ``config=`` is the documented escape hatch for a checkpoint whose
        # ``config.json`` predates the per-variant ``model_type``; no
        # ``ignore_mismatched_sizes`` here, so a renamed parameter cannot hide.
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
