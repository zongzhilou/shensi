#!/usr/bin/env python3
"""按 ``benchmarks.py`` 的表核一遍数据集：名字在不在注册表、HF 双胞胎在不在、子集对不对得上。

**要用评测 venv 的 python 跑**（它才有 EvalScope 与 datasets）：

    .venv_eval/bin/python check_datasets.py                    # 只核元数据（快，几乎不下载）
    .venv_eval/bin/python check_datasets.py --suite main --load  # 真拉下来，报每个子集的行数
    .venv_eval/bin/python check_datasets.py --datasets gsm8k --defaults

三件事：

1. ``--registry``（默认开）：名字必须是 EvalScope 注册表里的键，否则 ``eval.py`` 会直接报未知数据集。
2. ``--defaults``：打印每个项在注册表里的默认 ``dataset_id`` / 切分 / 子集列表（HF 覆写就是对着它写的）。
3. ``--load``：按 ``hf_id``（没有就用默认 id）真正装载一次，报行数——这是"这张表能不能用"的硬判据。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_benchmarks():
    """按路径装载 ``benchmarks.py``：评测 venv 里没有 shensi 包，走包导入会失败。"""
    spec = importlib.util.spec_from_file_location("looma_benchmarks", HERE / "benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 解析注解时要能在 sys.modules 里找到本模块
    spec.loader.exec_module(module)
    return module


benchmarks = _load_benchmarks()


def registered_names() -> set[str]:
    """EvalScope 注册表里的数据集名。"""
    from evalscope.api.registry import BENCHMARK_REGISTRY

    return set(BENCHMARK_REGISTRY)


def check_registry(names: tuple[str, ...], known: set[str]) -> list[str]:
    """名字必须在注册表里；返回没对上的。"""
    missing = [name for name in names if name not in known]
    for name in names:
        print(f"[注册表] {name}: {'✓' if name in known else '✗ 不在注册表'}")
    return missing


def show_defaults(names: tuple[str, ...]) -> None:
    """打印注册表里的默认口径（HF 覆写就是对着这些字段写的）。"""
    from evalscope.api.registry import BENCHMARK_REGISTRY

    for name in names:
        cls = BENCHMARK_REGISTRY.get(name)
        if cls is None:
            continue
        inst = cls()
        subsets = getattr(inst, "subset_list", None)
        if isinstance(subsets, (list, tuple)) and len(subsets) > 4:
            subsets = f"<{len(subsets)} 项: {list(subsets)[:3]} …>"
        print(
            f"[默认口径] {name}: id={getattr(inst, 'dataset_id', None)} "
            f"split={getattr(inst, 'eval_split', None)} subset={getattr(inst, 'default_subset', None)} "
            f"subsets={subsets} few_shot={getattr(inst, 'few_shot_num', None)}"
        )


def check_hf(names: tuple[str, ...]) -> list[str]:
    """有 ``hf_id`` 的项：仓库存在 + 需要的 config 名都在。"""
    from datasets import get_dataset_config_names
    from huggingface_hub import HfApi

    api = HfApi()
    bad: list[str] = []
    for name in names:
        item = benchmarks.BY_NAME.get(name)
        if item is None or not item.hf_id:
            continue
        try:
            api.repo_info(item.hf_id, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"[HF] {name} → {item.hf_id}: ✗ 仓库不可达（{type(exc).__name__}）")
            bad.append(name)
            continue
        try:
            configs = get_dataset_config_names(item.hf_id)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[HF] {name} → {item.hf_id}: ⚠ config 名单取不到（{type(exc).__name__}: {str(exc)[:60]}）"
            )
            continue
        need = list(item.hf_subset_list)
        if not need:
            print(
                f"[HF] {name} → {item.hf_id}: ✓ configs={configs[:4]}{' …' if len(configs) > 4 else ''}"
            )
            continue
        missing = [sub for sub in need if sub not in configs]
        print(f"[HF] {name} → {item.hf_id}: {'✓' if not missing else f'✗ 缺 {missing}'}")
        if missing:
            bad.append(name)
    return bad


def load_rows(names: tuple[str, ...], default_subset: str = "main") -> list[str]:
    """真正装载（按 ``hf_id`` 优先），报行数；返回装载失败的项。"""
    from evalscope.api.dataset.hub import load_dataset_from_hub

    bad: list[str] = []
    for name in names:
        item = benchmarks.BY_NAME.get(name) or benchmarks.Benchmark(name, name)
        dataset_id = item.hf_id or name
        subsets = list(item.hf_subset_list)
        if not subsets:
            subsets = [_registry_default(name, default_subset)]
        for subset in subsets:
            try:
                data = load_dataset_from_hub(
                    data_id_or_path=dataset_id,
                    split=None,
                    subset=subset,
                    data_source="huggingface" if item.hf_id else "modelscope",
                    trust_remote=True,
                )
                rows = (
                    {key: len(value) for key, value in data.items()}
                    if hasattr(data, "items")
                    else len(data)
                )
                print(f"[装载] {name} ← {dataset_id} / {subset}: ✓ {rows}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[装载] {name} ← {dataset_id} / {subset}: ✗ {type(exc).__name__}: {str(exc)[:120]}"
                )
                bad.append(name)
    return bad


def _registry_default(name: str, fallback: str) -> str:
    """取注册表里该项的默认子集名。"""
    try:
        from evalscope.api.registry import BENCHMARK_REGISTRY

        cls = BENCHMARK_REGISTRY.get(name)
        if cls is not None:
            inst = cls()
            subsets = getattr(inst, "subset_list", None) or []
            if len(subsets) == 1:
                return str(subsets[0])
            default = getattr(inst, "default_subset", None)
            if default:
                return str(default)
    except Exception:  # noqa: BLE001
        pass
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser(
        description="核 benchmarks.py 的数据集表（注册表 / HF 双胞胎 / 装载）"
    )
    ap.add_argument("--suite", default=None, help="集合名（mini/main/long/agent/all）")
    ap.add_argument("--datasets", nargs="*", default=None, help="显式数据集名")
    ap.add_argument("--registry", action="store_true", default=True, help="核注册表（默认开）")
    ap.add_argument("--defaults", action="store_true", help="打印注册表默认口径")
    ap.add_argument("--load", action="store_true", help="真装载一次（要下载数据）")
    args = ap.parse_args()

    names = benchmarks.resolve(args.datasets, [args.suite] if args.suite else ["all"])
    print(f"[检查] {len(names)} 项：{list(names)}\n")

    bad = check_registry(names, registered_names()) if args.registry else []
    if args.defaults:
        print()
        show_defaults(names)
    print()
    bad += check_hf(names)
    if args.load:
        print()
        bad += load_rows(names)

    if bad:
        print(f"\n[检查] 有问题的项：{sorted(set(bad))}")
        return 1
    print("\n[检查] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
