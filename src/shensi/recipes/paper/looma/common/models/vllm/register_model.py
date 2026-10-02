#!/usr/bin/env python3
"""vLLM 插件登记：安装 / 卸载 / 查询本配方的原生模型实现。"""



from __future__ import annotations

import argparse
import os
import sys

from .variants import VARIANTS

ENTRY_POINT_GROUP = "vllm.general_plugins"
ENTRY_POINT_NAME = "shensi_looma_rollout"
DIST_NAME = "shensi-looma-rollout"

MODULE = "shensi.recipes.paper.looma.common.models.vllm.register_model"
NATIVE_IMPL = "shensi.recipes.paper.looma.common.models.vllm.modeling_looma:LoomaForCausalLM"


def _src_dir() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[6])


def register_all(*, impl: str | None = None, verbose: bool = True) -> dict:
    """把本配方的原生实现登记进 vLLM 的模型注册表。"""
    from vllm.model_executor.models.registry import ModelRegistry

    spec = impl or os.environ.get("LOOMA_ROLLOUT_IMPL") or NATIVE_IMPL
    module_name, _, cls_name = spec.partition(":")
    report: dict = {"impl": spec, "registered": [], "already": [], "failed": {}}
    if _src_dir() not in sys.path:
        sys.path.insert(0, _src_dir())
    for variant in VARIANTS.values():
        arch = variant.architecture
        if arch in ModelRegistry.models:
            report["already"].append(arch)
            continue
        try:
            ModelRegistry.register_model(arch, spec)
            report["registered"].append(arch)
        except Exception as exc:  # noqa: BLE001
            report["failed"][arch] = f"{type(exc).__name__}: {exc}"
    if verbose:
        print(f"[looma·vllm] 登记实现：{module_name}:{cls_name}")
        print(f"  新登记 {report['registered']}｜已存在 {report['already']}｜失败 {report['failed']}")
    return report


def describe() -> dict:
    """打印解析结果（每个模型走哪份实现）。"""
    from vllm.model_executor.models.registry import ModelRegistry

    out = {}
    for variant in VARIANTS.values():
        entry = ModelRegistry.models.get(variant.architecture)
        if entry is None:
            out[variant.architecture] = None
            continue
        impl = getattr(entry, "model", entry)
        out[variant.architecture] = f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', impl)}"
    return out


def register_plugin() -> None:
    """把插件入口点写进 site-packages（vLLM worker 都能加载）。"""
    register_all(verbose=os.environ.get("LOOMA_PLUGIN_VERBOSE", "1") not in {"0", ""})


def install() -> int:
    """安装插件（写入口点）。"""
    import site
    from pathlib import Path

    target = next((Path(p) for p in site.getsitepackages() if Path(p).is_dir()), None)
    if target is None:  # pragma: no cover
        raise SystemExit("[looma·vllm] 找不到 site-packages")
    (target / f"{DIST_NAME.replace('-', '_')}.pth").write_text(_src_dir() + "\n", encoding="utf-8")

    dist = target / f"{DIST_NAME.replace('-', '_')}-0.1.0.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DIST_NAME}\nVersion: 0.1.0\n", encoding="utf-8"
    )
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shensi-looma\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        f"[{ENTRY_POINT_GROUP}]\n{ENTRY_POINT_NAME} = {MODULE}:register_plugin\n", encoding="utf-8"
    )
    print(f"[looma·vllm] 已写 {target}（{ENTRY_POINT_GROUP}: {ENTRY_POINT_NAME}）")
    return 0


def uninstall() -> int:
    """卸载插件。"""
    import site
    from pathlib import Path

    target = next((Path(p) for p in site.getsitepackages() if Path(p).is_dir()), None)
    if target is None:  # pragma: no cover
        return 0
    removed = 0
    for path in (target / f"{DIST_NAME.replace('-', '_')}.pth",):
        if path.exists():
            path.unlink()
            removed += 1
    dist = target / f"{DIST_NAME.replace('-', '_')}-0.1.0.dist-info"
    if dist.is_dir():
        for f in dist.iterdir():
            f.unlink()
        dist.rmdir()
        removed += 1
    print(f"[looma·vllm] 已移除 {removed} 项")
    return 0


def main(argv: list[str] | None = None) -> int:
    """登记工具的入口。"""
    ap = argparse.ArgumentParser(description="Looma 在 vLLM 里的登记")
    ap.add_argument("action", nargs="?", default="register",
                    choices=["register", "show", "install", "uninstall", "variants"])
    ap.add_argument("--impl", default=None, help=f"覆盖实现（默认 {NATIVE_IMPL}）")
    args = ap.parse_args(argv)
    if args.action == "register":
        report = register_all(impl=args.impl)
        return 0 if not report["failed"] else 1
    if args.action == "show":
        for arch, impl in describe().items():
            print(f"  {arch:20s} -> {impl or '(未登记，会走 auto_map 远程代码)'}")
        return 0
    if args.action == "variants":
        for key, v in VARIANTS.items():
            print(f"  {key:10s} arch={v.architecture:20s} model_type={v.model_type} "
                  f"tiny_knobs={v.tiny_knobs}")
        return 0
    if args.action == "install":
        return install()
    return uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
