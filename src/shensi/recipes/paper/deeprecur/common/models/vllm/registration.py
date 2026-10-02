"""把三臂的自研架构登记进 vLLM 的模型注册表（原生主干 + 桥）。

- **native** 不用登记：vLLM 自带 ``Qwen3VLForConditionalGeneration`` 原生实现；
- **unified / gdar / deeprecur** 登记到 ``RecurTransformersForCausalLM``（Transformers 后端 + 自研件遮罩）。

用法：
    python -m shensi.recipes.paper.deeprecur.common.models.vllm.registration register
    python -m shensi.recipes.paper.deeprecur.common.models.vllm.registration show
"""

from __future__ import annotations

import argparse
import importlib
import sys

from .bridge import ARCHITECTURES

BRIDGE_IMPL = (
    "shensi.recipes.paper.deeprecur.common.models.vllm.bridge:RecurTransformersForCausalLM"
)


def _import_attr(spec: str):
    module_path, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_path), attr)


def register_all(*, verbose: bool = True) -> dict:
    """把三臂架构登记进 vLLM 注册表（已在表里的跳过）。"""
    from vllm.model_executor.models.registry import ModelRegistry

    report: dict = {"impl": BRIDGE_IMPL, "registered": [], "already": [], "failed": {}}
    try:
        impl = _import_attr(BRIDGE_IMPL)
    except Exception as exc:
        report["failed"]["<bridge>"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"[deeprecur·vllm] 桥不可用（{exc}）", file=sys.stderr)
        return report
    for arch in ARCHITECTURES:
        if arch in ModelRegistry.models:
            report["already"].append(arch)
            continue
        try:
            ModelRegistry.register_model(arch, impl)
            report["registered"].append(arch)
        except Exception as exc:  # pragma: no cover
            report["failed"][arch] = f"{type(exc).__name__}: {exc}"
    if verbose:
        print(
            f"[deeprecur·vllm] 登记：registered={len(report['registered'])} "
            f"already={len(report['already'])} failed={len(report['failed'])}"
        )
        for arch, err in report["failed"].items():
            print(f"[deeprecur·vllm]   FAILED {arch}: {err}", file=sys.stderr)
    return report


def describe() -> dict[str, str]:
    """查每个架构现在解析到哪份实现。"""
    from vllm.model_executor.models.registry import ModelRegistry

    table = getattr(ModelRegistry, "models", None) or getattr(ModelRegistry, "_models", {})
    out: dict[str, str] = {}
    for arch in (*ARCHITECTURES, "Qwen3VLForConditionalGeneration"):
        entry = table.get(arch)
        if entry is None:
            out[arch] = "MISSING"
            continue
        cls = getattr(getattr(entry, "interfaces", None), "architecture", None)
        out[arch] = f"{type(entry).__name__}({cls})" if cls else str(entry)[:140]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 deeprecur 三臂登记进 vLLM 注册表")
    parser.add_argument("command", nargs="?", default="register", choices=["register", "show"])
    args = parser.parse_args(argv)
    import vllm

    print(f"[deeprecur·vllm] vllm {vllm.__version__}  python {sys.version.split()[0]}")
    if args.command == "show":
        for arch, how in describe().items():
            print(f"  {arch:<42} {how}")
        return 0
    report = register_all()
    for arch, how in describe().items():
        print(f"  {arch:<42} {how}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
