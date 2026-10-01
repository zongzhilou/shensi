"""Register the 7 depth-routed Qwen3 variants with the inference engine (vLLM).

What "registering" has to mean here
-----------------------------------
vLLM dispatches on ``config.json["architectures"][0]`` through a plain dict,
``vllm.model_executor.models.registry.ModelRegistry.models``.  For a new architecture
the *supported* extension point is

    ModelRegistry.register_model("<Architecture>", <nn.Module subclass or "mod:Cls">)

plus, for anything that must be registered where no user code runs (engine worker
processes), the ``vllm.general_plugins`` entry point -- vLLM's own plugin discovery,
which is what this module installs on request.  Neither touches the installed vLLM
package: the first writes into a dict, the second is standard Python packaging
metadata.

Two honest caveats, both verified rather than assumed (see ../README.md（本目录）与包内 code/ROLLOUT_ENV.md):

1. ``ModelRegistry.register_model`` alone is **not enough** to make a generation run
   work: vLLM v1 builds the model inside a separate ``EngineCore`` process, whose
   registry is empty.  ``rollout/sitecustomize.py`` (or the ``.pth`` written by
   ``install``) closes that gap -- exactly the trap already documented for Ray on the
   training side in ``stage2_rl 侧的注册（gdar_package 的 code/verl_plugin）``.
2. vLLM has no *native* kernel for our connection module (the depth routing is
   Qwen3 + an extra per-token read/update chain), and writing one is a separate job.
   So the class registered here delegates execution to the transformers
   implementation rather than reimplementing attention on vLLM's paged cache.  What
   that does and does not exercise is spelled out in ``../README.md（本目录）与包内 code/ROLLOUT_ENV.md``.

Usage
-----
    # as a library
    from .register_model import register_all
    report = register_all()

    # from the shell (also installs the entry point + worker-process hook)
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model register --check
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model install
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model show
"""

from __future__ import annotations

import importlib

import os
import sys
from pathlib import Path

from ._paths import RECIPE, SRC, ensure_src_on_path

ensure_src_on_path()

from .variants import VARIANTS, Variant  # noqa: E402

ENTRY_POINT_GROUP = "vllm.general_plugins"
ENTRY_POINT_NAME = "shensi_depth_rollout"
DIST_NAME = "shensi-depth-rollout"

#: Our own subclass of vLLM's generic HF backend.  It is that same backend, with the
#: connection subtrees held out of the engine's module rewrite (see
#: ``rollout/vllm_bridge.py`` for the two measured failures that make that necessary).
BRIDGE_IMPL = (
    "shensi.recipes.paper.gated_delta_attn_res.models.vllm.vllm_bridge:DepthTransformersForCausalLM"
)

#: vLLM's stock generic "execute an arbitrary HF model" implementation.  Kept as an
#: option (``--impl``): it is the unmodified engine behaviour, and the comparison is
#: the evidence for why the bridge exists.
TRANSFORMERS_IMPL = "vllm.model_executor.models.transformers:TransformersForCausalLM"


# --------------------------------------------------------------------------- #
# impl resolution
# --------------------------------------------------------------------------- #
def _import_attr(spec: str):
    if ":" not in spec:
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def resolve_impl(spec: str | None = None) -> tuple[object, str]:
    """Return ``(impl, how)``.

    The default is the bridge, imported eagerly so a broken import fails here rather
    than inside an engine worker process.  ``--impl`` overrides it, e.g.
    ``--impl vllm.model_executor.models.transformers:TransformersForCausalLM`` for the
    stock engine behaviour, or ``--impl my_pkg.my_model:Qwen3GDARNative`` once a native
    implementation exists.
    """
    if spec:
        if ":" in spec:
            return _import_attr(spec), f"explicit ({spec})"
        return spec, f"explicit ({spec})"
    try:
        return _import_attr(BRIDGE_IMPL), BRIDGE_IMPL
    except Exception as exc:  # pragma: no cover - depends on installed vLLM
        print(
            f"[shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model] {BRIDGE_IMPL} unusable ({type(exc).__name__}: {exc}); "
            f"falling back to the stock backend {TRANSFORMERS_IMPL}",
            file=sys.stderr,
        )
        return _import_attr(TRANSFORMERS_IMPL), f"{TRANSFORMERS_IMPL} (fallback)"


# --------------------------------------------------------------------------- #
# the registration itself
# --------------------------------------------------------------------------- #
def _engine_registry():
    from vllm.model_executor.models.registry import ModelRegistry

    return ModelRegistry


def register_all(
    *,
    impl: str | None = None,
    variants: tuple[Variant, ...] = VARIANTS,
    verbose: bool = True,
) -> dict:
    """Register every variant's architecture with the engine.  Idempotent.

    Returns a report dict: ``{"impl", "registered", "already", "failed"}``.
    """
    registry = _engine_registry()
    resolved, how = resolve_impl(impl)

    report: dict = {"impl": how, "registered": [], "already": [], "failed": {}}
    for variant in variants:
        arch = variant.architecture
        if arch in registry.models:
            report["already"].append(arch)
            continue
        try:
            registry.register_model(arch, resolved)
            report["registered"].append(arch)
        except Exception as exc:  # pragma: no cover
            report["failed"][arch] = f"{type(exc).__name__}: {exc}"

    if verbose:
        print(
            f"[shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model] impl={how} "
            f"registered={len(report['registered'])} already={len(report['already'])} "
            f"failed={len(report['failed'])}",
            flush=True,
        )
        for arch, err in report["failed"].items():
            print(
                f"[shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model]   FAILED {arch}: {err}",
                file=sys.stderr,
                flush=True,
            )
    return report


def is_registered() -> dict[str, bool]:
    """Which architectures the *current* process's registry already knows."""
    registry = _engine_registry()
    return {v.architecture: (v.architecture in registry.models) for v in VARIANTS}


def _registry_models(registry) -> dict:
    """vLLM keeps the dispatch table in ``registry.models``; tolerate a move."""
    for attr in ("models", "_models"):
        table = getattr(registry, attr, None)
        if isinstance(table, dict):
            return table
    return {}


def describe_resolution(names: list[str]) -> dict[str, str]:
    """What the engine's dispatch table says about each architecture name.

    This is the cheapest available proof that registration reached the engine: it
    reads the same table the engine reads when it loads a model.  The entry is
    rendered down to the name of the class the engine will build, because vLLM's own
    ``repr`` (``_RegisteredModel(interfaces=_ModelInfo(...))``) is unreadable and its
    shape differs between a lazily-registered and a materialised entry.
    """
    registry = _engine_registry()
    table = _registry_models(registry)
    out: dict[str, str] = {}
    for name in names:
        entry = table.get(name)
        if entry is None:
            out[name] = "MISSING"
            continue
        cls = getattr(getattr(entry, "interfaces", None), "architecture", None)
        out[name] = (
            f"{type(entry).__name__}({cls})" if cls else f"{type(entry).__name__}({entry!r})"[:160]
        )
    return out


def register_plugin() -> None:
    """The ``vllm.general_plugins`` entry point body.

    vLLM calls this once per process (driver and each engine worker) before models
    are built, so no ``sitecustomize`` gymnastics are needed when this entry point is
    installed.
    """
    register_all(
        impl=os.environ.get("ROLLOUT_PLUGIN_IMPL") or None,
        verbose=os.environ.get("ROLLOUT_PLUGIN_VERBOSE", "1") not in {"0", ""},
    )


# --------------------------------------------------------------------------- #
# packaging-side install: entry point + worker-process hook
# --------------------------------------------------------------------------- #
def _site_packages() -> Path:
    import site

    for p in site.getsitepackages() + [site.getusersitepackages()]:
        if p and Path(p).name == "site-packages" and Path(p).is_dir():
            return Path(p)
    raise RuntimeError("could not locate site-packages")


def install(*, write_pth: bool = True, write_entry_point: bool = True) -> dict:
    """Install the plugin *without* touching any file vLLM ships.

    Writes only inside the venv (``.../site-packages``):

    * ``shensi_depth_rollout.pth``  -- puts ``<repo>/src`` on ``sys.path`` so the
      ``rollout`` package is importable from any interpreter in this venv;
    * ``shensi_depth_rollout.dist-info/`` -- metadata + ``entry_points.txt`` declaring
      the ``vllm.general_plugins`` entry point, so vLLM discovers it by itself;
    * ``shensi_rollout_sitecustomize.pth`` -- opt-in (``ROLLOUT_PLUGIN_AUTOLOAD=1``) hook for
      engines whose workers do not inherit plugin metadata cleanly.
    """
    sp = _site_packages()
    written: list[str] = []

    if write_pth:
        pth = sp / "shensi_depth_rollout.pth"
        pth.write_text(f"{SRC}\n")
        written.append(str(pth))

        hook = sp / "shensi_rollout_sitecustomize.pth"
        hook.write_text(f"from . import sitecustomize\n")
        written.append(str(hook))

    if write_entry_point:
        dist = sp / f"{DIST_NAME.replace('-', '_')}-0.1.0.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {DIST_NAME}\n"
            "Version: 0.1.0\n"
            "Summary: Register the depth-routed Qwen3 variants with vLLM\n"
        )
        (dist / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        (dist / "entry_points.txt").write_text(
            f"[{ENTRY_POINT_GROUP}]\n{ENTRY_POINT_NAME} = shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model:register_plugin\n"
        )
        (dist / "RECORD").write_text("")
        written.append(str(dist))
    return {"site_packages": str(sp), "written": written}


def uninstall() -> dict:
    import shutil

    sp = _site_packages()
    removed: list[str] = []
    for name in ("shensi_depth_rollout.pth", "shensi_rollout_sitecustomize.pth"):
        p = sp / name
        if p.exists():
            p.unlink()
            removed.append(str(p))
    dist = sp / f"{DIST_NAME.replace('-', '_')}-0.1.0.dist-info"
    if dist.exists():
        shutil.rmtree(dist)
        removed.append(str(dist))
    return {"site_packages": str(sp), "removed": removed}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Register the depth-routed variants with vLLM.")
    ap.add_argument(
        "command",
        nargs="?",
        default="register",
        choices=["register", "show", "install", "uninstall", "variants"],
    )
    ap.add_argument(
        "--impl", default=None, help="override the registered implementation, 'module:Class'"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="after registering, re-read the registry (proves it took)",
    )
    args = ap.parse_args(argv)

    if args.command == "variants":
        for v in VARIANTS:
            print(f"{v.key:12s} arch={v.architecture:28s} model_type={v.model_type}")
        return 0

    if args.command == "install":
        print(json.dumps(install(), indent=2))
        return 0
    if args.command == "uninstall":
        print(json.dumps(uninstall(), indent=2))
        return 0

    import vllm

    print(f"vllm {vllm.__version__}  python {sys.version.split()[0]}")
    rep = register_all(impl=args.impl)
    if args.check:
        print(json.dumps(describe_resolution([v.architecture for v in VARIANTS]), indent=2))
    return 1 if rep["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
