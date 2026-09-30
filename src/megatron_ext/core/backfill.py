"""把 FL fork 缺、而第三方（Megatron-Bridge / verl）要用的 mcore 符号补齐。

上游按 mcore main 的 API 写，fork 里少几个名字，导入期就会炸。我们不改上游文件，
改成在目标模块被导入之后把缺的名字挂上去（`shensi.runtime` 调 `apply()`）。

- 搬名字（COPY）：名字在别的模块里有，原样搬过来；
- 改名（ALIAS）：main 里改了名，指回 fork 里那一版；
- 直接定义（DEFINE）：fork 里根本没有的能力，按「不支持」处理。
只补缺的，绝不覆盖已有的。
"""

from __future__ import annotations

import importlib
import sys

COPY: dict[str, tuple[str, tuple[str, ...]]] = {
    # 「名字从哪来」都是我们这边的模块：fork 的原文件不动
    "megatron.core.utils": (
        "megatron_ext.core._rank_utils",
        ("set_default_log_ranks", "get_default_log_ranks", "warn_single_rank"),
    ),
    # Bridge 的 provider 要 get_backend；fork 的 backends.py 里有三个 provider 但没这个函数
    "megatron.core.models.backends": ("megatron_ext.core.models.backends", ("get_backend",)),
}

ALIAS: dict[str, tuple[tuple[str, str], ...]] = {
    "megatron.core.distributed.fsdp.mcore_fsdp_adapter": (
        # main 把 fork 的 FullyShardedDataParallel 改名 V1 并加了 V2；别名只是让第三方的
        # isinstance 解包认得 fork 的那一版，等 fork 自己长出来就自动失效
        ("FullyShardedDataParallelV1", "FullyShardedDataParallel"),
        ("FullyShardedDataParallelV2", "FullyShardedDataParallel"),
    ),
}


def _unsupported_grouped_mxfp8(*_args, **_kwargs):
    raise NotImplementedError("grouped mxfp8 is not available in this Megatron Core build")


DEFINE: dict[str, dict[str, object]] = {
    "megatron.core.fp8_utils": {
        # 这两个是 mcore main 的 grouped mxfp8 原语；fork 没有 → 第三方按「不支持」走回退分支
        "is_grouped_mxfp8tensor": lambda *_a, **_k: False,
        "get_grouped_quantized_members": _unsupported_grouped_mxfp8,
    },
}

TARGETS: tuple[str, ...] = tuple(sorted(set(COPY) | set(ALIAS) | set(DEFINE)))


def apply() -> list[str]:
    """补一次（只看已经在 sys.modules 里的目标模块）；返回这次真补上的符号。"""
    done: list[str] = []
    for target_name in TARGETS:
        target = sys.modules.get(target_name)
        if target is None:
            continue
        source_name, names = COPY.get(target_name, (None, ()))
        source = importlib.import_module(source_name) if source_name else None
        pairs = [(name, getattr(source, name, None) if source else None) for name in names]
        pairs += [(new, getattr(target, old, None)) for new, old in ALIAS.get(target_name, ())]
        pairs += list(DEFINE.get(target_name, {}).items())
        for name, value in pairs:
            if hasattr(target, name) or value is None:
                continue
            setattr(target, name, value)
            done.append(f"{target_name}.{name}")
    return done
