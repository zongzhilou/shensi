"""把 mcore main 有、FL fork 没有的符号补进对应模块的命名空间。

Bridge@a393057 与 verl 是按 mcore main 的 API 写的，两处坑都在**导入期**炸：

1. `megatron.bridge.models.megatron_mimo.*` 里 `from megatron.core.utils import set_default_log_ranks`
   —— FL fork 的 `utils.py` / `_rank_utils.py` 还没有这个名字（我们那份 `_rank_utils.py` 补齐了实现）。
2. Bridge `models/conversion/utils.py::unwrap_model` 里
   `from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1, ...V2`
   —— main 把 fork 的 `FullyShardedDataParallel` 改名成 `V1` 并新增了 `V2`。

上游文件不动（增量原则），改成在目标模块导入之后把缺的名字挂上去。触发方式见 `shensi/__init__.py`
的导入后钩子：每导入一个目标模块就调一次 `apply()`；只补缺的，绝不覆盖已有的（等 fork 自己长出来
这些名字时，这里自动失效）。
"""

from __future__ import annotations

import importlib
import sys

# 目标模块 → (符号来源模块, 要原样搬过来的名字)
COPY: dict[str, tuple[str, tuple[str, ...]]] = {
    "megatron.core.utils": (
        "megatron.core._rank_utils",
        ("set_default_log_ranks", "get_default_log_ranks", "warn_single_rank"),
    ),
}

# 目标模块 → ((新名字, 模块里已有的名字), ...)：main 里改了名的符号
ALIAS: dict[str, tuple[tuple[str, str], ...]] = {
    "megatron.core.distributed.fsdp.mcore_fsdp_adapter": (
        # main 只是把它改名成 V1；V2 是 main 新增的一版实现，fork 里没有，这里也指向同一个类——
        # 这个别名的用处只是让 Bridge 的 isinstance 解包认得 fork 的那一版（我们不用 V2；
        # 等 fork 真有了 V2，下面的 hasattr 判断会让别名自动失效）。
        ("FullyShardedDataParallelV1", "FullyShardedDataParallel"),
        ("FullyShardedDataParallelV2", "FullyShardedDataParallel"),
    ),
}

TARGETS: tuple[str, ...] = tuple(sorted(set(COPY) | set(ALIAS)))


def apply() -> list[str]:
    """补一次（只补已经在 sys.modules 里的目标模块）；返回这次真补上的符号。"""
    done: list[str] = []
    for target_name in TARGETS:
        target = sys.modules.get(target_name)
        if target is None:
            continue
        source_name, names = COPY.get(target_name, (None, ()))
        source = importlib.import_module(source_name) if source_name else None
        pairs = [(name, getattr(source, name, None) if source else None) for name in names]
        # 别名：右值是"模块里已有的名字"，取它的值
        pairs += [(new, getattr(target, old, None)) for new, old in ALIAS.get(target_name, ())]
        for name, value in pairs:
            if hasattr(target, name) or value is None:
                continue
            setattr(target, name, value)
            done.append(f"{target_name}.{name}")
    return done
