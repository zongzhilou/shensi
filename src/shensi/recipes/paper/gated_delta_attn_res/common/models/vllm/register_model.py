"""vLLM 插件登记：安装 / 卸载 / 查询本配方的原生模型实现。"""


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




BRIDGE_IMPL = (
    "shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.vllm_bridge:DepthTransformersForCausalLM"
)




TRANSFORMERS_IMPL = "vllm.model_executor.models.transformers:TransformersForCausalLM"





def _import_attr(spec: str):
    if ":" not in spec:
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def resolve_impl(spec: str | None = None) -> tuple[object, str]:
    """解析该模型走哪份原生实现（插件 / 内置 / remote-code）。"""
    if spec:
        if ":" in spec:
            return _import_attr(spec), f"explicit ({spec})"
        return spec, f"explicit ({spec})"
    try:
        return _import_attr(BRIDGE_IMPL), BRIDGE_IMPL
    except Exception as exc:  # pragma: no cover - depends on installed vLLM
        print(
            f"[shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.register_model] {BRIDGE_IMPL} unusable ({type(exc).__name__}: {exc}); "
            f"falling back to the stock backend {TRANSFORMERS_IMPL}",
            file=sys.stderr,
        )
        return _import_attr(TRANSFORMERS_IMPL), f"{TRANSFORMERS_IMPL} (fallback)"





def _engine_registry():
    from vllm.model_executor.models.registry import ModelRegistry

    return ModelRegistry


def register_all(
    *,
    impl: str | None = None,
    variants: tuple[Variant, ...] = VARIANTS,
    verbose: bool = True,
) -> dict:
    """把本配方的原生实现登记进 vLLM 的模型注册表。"""
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
            f"[shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.register_model] impl={how} "
            f"registered={len(report['registered'])} already={len(report['already'])} "
            f"failed={len(report['failed'])}",
            flush=True,
        )
        for arch, err in report["failed"].items():
            print(
                f"[shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.register_model]   FAILED {arch}: {err}",
                file=sys.stderr,
                flush=True,
            )
    return report


def is_registered() -> dict[str, bool]:
    """查询原生实现是否已登记。"""
    registry = _engine_registry()
    return {v.architecture: (v.architecture in registry.models) for v in VARIANTS}


def _registry_models(registry) -> dict:
    for attr in ("models", "_models"):
        table = getattr(registry, attr, None)
        if isinstance(table, dict):
            return table
    return {}


def describe_resolution(names: list[str]) -> dict[str, str]:
    """打印解析结果（每个模型走哪份实现）。"""
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
    """把插件入口点写进 site-packages（vLLM worker 都能加载）。"""
    register_all(
        impl=os.environ.get("ROLLOUT_PLUGIN_IMPL") or None,
        verbose=os.environ.get("ROLLOUT_PLUGIN_VERBOSE", "1") not in {"0", ""},
    )





def _site_packages() -> Path:
    import site

    for p in site.getsitepackages() + [site.getusersitepackages()]:
        if p and Path(p).name == "site-packages" and Path(p).is_dir():
            return Path(p)
    raise RuntimeError("could not locate site-packages")


def install(*, write_pth: bool = True, write_entry_point: bool = True) -> dict:
    """安装插件（写入口点）。"""
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
            "Wheel-Version: 1.0\nGenerator: shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.register_model\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        (dist / "entry_points.txt").write_text(
            f"[{ENTRY_POINT_GROUP}]\n{ENTRY_POINT_NAME} = shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.register_model:register_plugin\n"
        )
        (dist / "RECORD").write_text("")
        written.append(str(dist))
    return {"site_packages": str(sp), "written": written}


def uninstall() -> dict:
    """卸载插件。"""
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
