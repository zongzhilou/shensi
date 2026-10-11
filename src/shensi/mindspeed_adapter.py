#!/usr/bin/env python3
"""MindSpeed 的确定性接入：包装 `MindSpeedPatchesManager.apply_patches`，把
"全量遍历、中途失败留部分应用"换成"预检分类 → 只应用可用的 → 快照自检 → 响亮失败"。

MindSpeed 的补丁机制：所有特性把补丁点注册进 `MindSpeedPatchesManager.patches_info`
（默认线 400+ 个），`apply_patches()` 无差别遍历——目标在本 mcore 上不存在时中途抛错，
**之前已应用的补丁留在 megatron 里**（实测把 config 的 `vocab_size` 弄丢、训练起不来）。
本包装把三件事 determin 化：

1. **预检**：只读解析每个补丁目标（不落 dummy、不 setattr——MindSpeed 自己的
   `parse_path` 对缺属性会顺手 `setattr(None)` 污染 megatron 命名空间，不能复用）。
   缺位的不应用、点名告警；`create_dummy=True` 的缺位同样不落地。
2. **应用失败即抛**：带完整报告（已应用/失败点）响亮失败，不留半套状态；回滚用上游
   自带的 `MindSpeedPatchesManager.remove_patches()`。
3. **快照自检**：首个补丁应用**前**抓 `TransformerConfig`（及 Bridge 的
   `ShensiTransformerConfig`，可导入时）的字段名基线，应用后差分——**被删字段即报**
   （增字段是正常适配；删/改名是事故）。不依赖具体字段名单，跨 mcore 分支成立
   （如新 mcore 本就没有 `vocab_size`，只有 `hash_moe_vocab_size`）。

本仓不 pin mindspeed，也不在这里决定用不用 MindSpeed（机器环境的安装选择，安装顺序见
ASCEND_ADAPTATION）；这里只保证"一旦有人按官方文档 import mindspeed.megatron_adaptor，
得到的是确定性路径而不是全量或全崩"。
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# 这些补丁点缺位视为致命（在应用任何补丁之前就抛）。按分支配对的结果填：配得好时应该为空。
MINDSPEED_REQUIRED_PATCHES: tuple[str, ...] = ()

# 任何 mcore 线都该有的古老字段：连它们都没了就不是"适配"是"换骨"，直接报。
MINDSPEED_STABLE_CONFIG_FIELDS: tuple[str, ...] = (
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "ffn_hidden_size",
)


def _config_field_names(config_cls) -> frozenset[str]:
    import dataclasses

    try:
        return frozenset(f.name for f in dataclasses.fields(config_cls))
    except TypeError:  # 被换成了普通类：按属性名核
        return frozenset(dir(config_cls))


def _config_classes() -> list[tuple[str, type]]:
    """要盯的 config 类：mcore 的 TransformerConfig，加 Bridge 的 ShensiTransformerConfig（可导入时）。"""
    out: list[tuple[str, type]] = []
    try:
        mod = importlib.import_module("megatron.core.transformer.transformer_config")
        out.append(("megatron.core.TransformerConfig", mod.TransformerConfig))
    except ImportError:
        pass
    try:
        mod = importlib.import_module("megatron.bridge.models.shensi.configuration_shensi")
        out.append(("bridge.ShensiTransformerConfig", mod.ShensiTransformerConfig))
    except ImportError:
        pass
    return out


def _mindspeed_config_selfcheck(config_cls) -> list[str]:
    """返回 config_cls 上丢失的稳定字段名；空列表 = 通过。"""
    names = _config_field_names(config_cls)
    return [name for name in MINDSPEED_STABLE_CONFIG_FIELDS if name not in names]


def _resolve_target(mod_name: str, func_name: str | None):
    """只读解析 `mod_name(.func_name)`；缺位返回 None。不 import 不存在的模块、不 setattr。"""
    parts = mod_name.split(".")
    mod = None
    rest: list[str] = []
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
            rest = parts[i:]
            break
        except ImportError:
            continue
    if mod is None:
        return None
    obj = mod
    for part in rest:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    if func_name is not None and not hasattr(obj, func_name):
        return None
    return obj


def _mindspeed_preflight(patches_info: dict) -> tuple[list[str], list[str]]:
    """逐点解析补丁目标，返回 (可用名单, 缺位名单)。只读，不改 megatron 命名空间。"""
    ok: list[str] = []
    missing: list[str] = []
    for name, patch in list(patches_info.items()):
        if _resolve_target(patch.orig_module_name, patch.orig_func_name) is None:
            missing.append(name)
        else:
            ok.append(name)
    return ok, missing


def _removed_config_fields(baseline: dict[str, frozenset[str]]) -> list[str]:
    """对照基线差分出被补丁删掉/弄丢的字段（含稳定字段名单兜底）。"""
    removed: list[str] = []
    for name, cls in _config_classes():
        b = baseline.get(name)
        if b is not None:
            removed += [f"{name}.{f}" for f in b - _config_field_names(cls)]
        removed += [f"{name}.{f}" for f in _mindspeed_config_selfcheck(cls)]
    return sorted(set(removed))


def _harden_mindspeed_apply_patches() -> None:
    try:
        from mindspeed.patch_utils import MindSpeedPatchesManager
    except ImportError:
        return  # 没装 MindSpeed：无需任何处理
    if getattr(MindSpeedPatchesManager.apply_patches, "_shensi_guarded", False):
        return
    original = MindSpeedPatchesManager.__dict__["apply_patches"].__func__
    state: dict = {"baseline": None}

    def _take_baseline() -> None:
        if state["baseline"] is None:
            state["baseline"] = {name: _config_field_names(cls) for name, cls in _config_classes()}

    def apply_patches(cls):
        ok, missing = _mindspeed_preflight(cls.patches_info)
        if missing:
            strict_missing = sorted(set(missing) & set(MINDSPEED_REQUIRED_PATCHES))
            if strict_missing:
                raise RuntimeError(
                    "mindspeed 必需补丁点在本 mcore 上缺位（分支没配对）："
                    f"{strict_missing}；请换 MindSpeed 分支或补 MINDSPEED_REQUIRED_PATCHES 之外的适配"
                )
            logger.warning(
                "shensi runtime: mindspeed %d/%d 个补丁点在本 mcore 上缺位，已跳过：%s",
                len(missing),
                len(cls.patches_info),
                ", ".join(sorted(missing)),
            )
        # 基线在**任何本阶段补丁落地之前**抓（pre 阶段 megatron 没导入时为空表，主阶段会补上）
        _take_baseline()
        applied: list[str] = []
        for name in ok:
            try:
                cls.patches_info[name].apply_patch()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "mindspeed 补丁应用失败（已应用 "
                    f"{len(applied)}/{len(ok)}：{applied[:8]}…；当前失败点 {name}：{exc}）——"
                    "不要带半套补丁继续跑，先按报告修分支配对或回滚 "
                    "MindSpeedPatchesManager.remove_patches()"
                ) from exc
            applied.append(name)
        # 自检：被删字段 = 适配事故，当场报而不是训练时报（增字段是正常适配，不算）
        removed = _removed_config_fields(state["baseline"])
        if removed:
            raise RuntimeError(
                f"mindspeed 补丁把 config 的字段弄丢了：{sorted(set(removed))}（正是"
                '"部分应用"的症状）；用 MindSpeedPatchesManager.remove_patches() 回滚后重配'
            )
        logger.info(
            "shensi runtime: mindspeed 补丁确定性应用完成：%d 应用 / %d 缺位跳过 / 自检通过",
            len(applied),
            len(missing),
        )

    apply_patches._shensi_guarded = True
    MindSpeedPatchesManager.apply_patches = classmethod(apply_patches)
    MindSpeedPatchesManager._shensi_original_apply_patches = original
