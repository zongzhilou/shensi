"""纯权重检查点的标记读写：`pure_weight.json` 跟着 ckpt 走，保存与加载两侧共用。

`no_save_optim: true` 的档存出的检查点里没有优化器状态；拿它续跑会撞上游的 KeyError，正确
口径是显式给 `--no-load-optim`。这个标记就是让那条口径可发现：保存侧落文件（probes.py 的
`install_pure_weight_marker`），加载侧（args.py 的 `warn_pure_weight_load`）据此把提示打全。
"""

from __future__ import annotations

import json
import os

PURE_WEIGHT_MARKER = "pure_weight.json"


def marker_path(load_path: str | os.PathLike | None) -> str | None:
    """找标记文件：`<load>/pure_weight.json` 或 `<load>/../pure_weight.json` 都在范围内。

    前者覆盖 `--load <save>`，后者覆盖 `--load <save>/iter_0000007` 这种指到具体迭代的写法。
    """
    if not load_path:
        return None
    base = str(load_path).rstrip("/\\")
    candidates = (
        os.path.join(base, PURE_WEIGHT_MARKER),
        os.path.join(os.path.dirname(base), PURE_WEIGHT_MARKER),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def is_pure_weight(load_path: str | os.PathLike | None) -> str | None:
    """是纯权重检查点就返回标记路径；内容对不上（顶层键里声称有 optimizer）返回 None。

    宁可不认：标记与实况不符时退回上游自己的报错，不让提示把加载口径带偏。
    """
    path = marker_path(load_path)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            info = json.load(handle)
    except Exception:  # noqa: BLE001 - 标记损坏一律当没有
        return None
    keys = set(info.get("top_level_keys") or [])
    if info.get("pure_weight") and "optimizer" not in keys:
        return path
    return None


def write_marker(
    save_dir: str | os.PathLike | None,
    *,
    no_save_optim: bool,
    no_save_rng: bool = False,
    iteration: int | None = None,
    reason: str = "no_save_optim",
) -> str | None:
    """按本次存盘的实况维护标记：权重存盘写下顶层键，非权重存盘清掉过期标记。

    顶层键由存盘开关直接决定（`--no-save-optim` 的 ckpt 不可能带 optimizer），所以标记天然
    与实况一致；写临时文件再 `os.replace`，避免别的 rank 读到写了一半的标记。
    """
    if not save_dir:
        return None
    marker = os.path.join(str(save_dir), PURE_WEIGHT_MARKER)
    if not no_save_optim:
        if os.path.isfile(marker):
            os.remove(marker)
        return None
    os.makedirs(str(save_dir), exist_ok=True)
    payload = {
        "pure_weight": True,
        "top_level_keys": ["model"] + ([] if no_save_rng else ["rng_state"]),
        "reason": reason,
    }
    if iteration is not None:
        payload["iteration"] = int(iteration)
    tmp = marker + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, marker)
    return marker
