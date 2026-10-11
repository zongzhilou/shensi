#!/usr/bin/env python3
"""MindSpeed 确定性接入的闸门：预检分类、缺位跳过、失败即抛、自检、严格模式、幂等。

用**真的** `mindspeed.patch_utils`（本机 /tmp/mindspeed_r014 浅克隆或 pip 装的包）和
**真的** mcore（Bridge 钉的 3rdparty/Megatron-LM）测：假补丁注册进真 manager，
逐项核对我们包装后的语义。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import mindspeed.patch_utils as pu  # noqa: E402,F401  - 导入即验证 mindspeed 可达
from mindspeed.patch_utils import MindSpeedPatchesManager  # noqa: E402

from shensi.mindspeed_adapter import (  # noqa: E402
    _harden_mindspeed_apply_patches,
    _mindspeed_config_selfcheck,
    _mindspeed_preflight,
    _removed_config_fields,
)

FAILS: list[str] = []


def check(name: str, ok, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def target_attr():
    return "untouched"


# 一个真实存在、可被安全替换又可还原的 megatron 目标
TARGET = "megatron.core.utils.divide"


def fresh_manager(monkeypatched: dict) -> None:
    MindSpeedPatchesManager.patches_info.clear()
    for name, (orig_name, func, dummy) in monkeypatched.items():
        MindSpeedPatchesManager.register_patch(orig_name, func, create_dummy=dummy)


def main() -> int:
    print("== mindspeed 确定性接入 ==")
    with tempfile.TemporaryDirectory():
        import megatron.core.utils as mcu

        sentinel = mcu.divide
        baseline_selfcheck = _mindspeed_config_selfcheck(
            __import__("megatron.core.transformer.transformer_config", fromlist=["TransformerConfig"]).TransformerConfig
        )
        check("自检：真 mcore 的 TransformerConfig 通过", baseline_selfcheck == [], f"missing={baseline_selfcheck}")

        # 伪造一个丢字段的 config 类，验证自检能抓到
        class PoisonConfig:  # noqa: D401 - 仅测自检
            pass

        lost = _mindspeed_config_selfcheck(PoisonConfig)
        from shensi.mindspeed_adapter import MINDSPEED_STABLE_CONFIG_FIELDS

        check("自检：丢字段被抓住",
              set(lost) == set(MINDSPEED_STABLE_CONFIG_FIELDS), f"lost={len(lost)}")

        # 快照差分：基线里有、补丁后没了的字段被点名（增字段不算事故）
        real_classes = __import__(
            "shensi.mindspeed_adapter", fromlist=["_config_classes"]
        )._config_classes()
        baseline = {name: _mindspeed_config_selfcheck and __import__(
            "shensi.mindspeed_adapter", fromlist=["_config_field_names"]
        )._config_field_names(cls) for name, cls in real_classes}

        class StrippedConfig(PoisonConfig):
            ffn_hidden_size = 0

        import shensi.mindspeed_adapter as adapter

        orig_classes = adapter._config_classes
        adapter._config_classes = lambda: [("fake.Config", StrippedConfig)]
        try:
            removed = _removed_config_fields(baseline)
        finally:
            adapter._config_classes = orig_classes
        check("快照差分：删字段被点名", any("fake.Config" in r for r in removed), f"removed={removed[:3]}")

        # 预检分类：存在 / 缺属性 / 缺模块
        patches = {
            "ok": (TARGET, lambda: "patched", False),
            "attr": ("megatron.core.utils.no_such_attr_xyz", lambda: None, False),
            "module": ("megatron.core.no_such_module_xyz.func", lambda: None, False),
            "dummy": ("megatron.core.utils.also_missing_xyz", lambda: None, True),
        }
        fresh_manager(patches)
        import megatron.core.utils as mcu0

        ok, missing = _mindspeed_preflight(MindSpeedPatchesManager.patches_info)
        # patches_info 以**目标全名**为键
        check("预检分类正确", ok == [TARGET] and sorted(missing) == sorted(
            {"megatron.core.utils.no_such_attr_xyz", "megatron.core.utils.also_missing_xyz",
             "megatron.core.no_such_module_xyz.func"}),
              f"ok={ok} missing={sorted(missing)}")
        check("预检只读：缺位属性没有被 setattr 污染",
              not hasattr(mcu0, "no_such_attr_xyz") and not hasattr(mcu0, "also_missing_xyz"))

        # 包装后应用：只有 ok 被应用；dummy 不落地；失败点不抛（缺位=跳过）
        _harden_mindspeed_apply_patches()
        check("包装生效（标记在）", getattr(MindSpeedPatchesManager.apply_patches, "_shensi_guarded", False))
        MindSpeedPatchesManager.apply_patches()
        import megatron.core.utils as mcu2

        check("ok 补丁已应用", mcu2.divide() == "patched")
        check("缺位属性没被安装（含 create_dummy）", not hasattr(mcu2, "no_such_attr_xyz")
              and not hasattr(mcu2, "also_missing_xyz"))
        check("保留还原钩子", hasattr(MindSpeedPatchesManager, "_shensi_original_apply_patches"))
        MindSpeedPatchesManager.remove_patches()
        check("上游 remove_patches 回滚有效", mcu.divide is sentinel or mcu2.divide is sentinel)

        # 应用失败即抛：patch 函数抛错发生在**调用时**，应用期故障用可注入的 Patch.apply_patch 模拟
        fresh_manager({"bad": (TARGET, lambda: None, False)})
        orig_apply = pu.Patch.apply_patch

        def boom(self):
            raise ZeroDivisionError("boom")

        pu.Patch.apply_patch = boom
        try:
            MindSpeedPatchesManager.apply_patches()
            check("应用失败响亮抛出", False)
        except RuntimeError as exc:
            check("应用失败响亮抛出", "boom" in str(exc) and "0/1" in str(exc), str(exc)[:70])
        finally:
            pu.Patch.apply_patch = orig_apply
            MindSpeedPatchesManager.remove_patches()

        # 严格模式：必需点缺位 → 什么都不应用就抛
        import shensi.mindspeed_adapter as adapter

        old_required = adapter.MINDSPEED_REQUIRED_PATCHES
        adapter.MINDSPEED_REQUIRED_PATCHES = ("megatron.core.utils.no_such_attr_xyz",)
        try:
            fresh_manager({
                "ok": (TARGET, lambda: "patched", False),
                "req": ("megatron.core.utils.no_such_attr_xyz", lambda: None, False),
            })
            try:
                MindSpeedPatchesManager.apply_patches()
                check("严格模式：必需点缺位先抛", False)
            except RuntimeError as exc:
                applied = mcu.divide is not sentinel
                check("严格模式：必需点缺位先抛", "no_such_attr_xyz" in str(exc) and not applied,
                      f"early-fail={not applied}")
        finally:
            adapter.MINDSPEED_REQUIRED_PATCHES = old_required
            MindSpeedPatchesManager.remove_patches()

        # 幂等：重复安装不叠加（classmethod 每次访问产生新 bound 方法，比对底层函数）
        first = MindSpeedPatchesManager.__dict__["apply_patches"].__func__
        _harden_mindspeed_apply_patches()
        check("重复安装幂等", MindSpeedPatchesManager.__dict__["apply_patches"].__func__ is first)

        MindSpeedPatchesManager.patches_info.clear()

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} 项未过：{FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
