#!/usr/bin/env python3
"""GDAR 配方 · stage2_rl 预检（四臂共用）：配置 → verl 命令、奖励模块、数据、依赖。

与 shensi 配方 RL stage 的判据一致：RL 不跑 tiny 训练（verl 起真训练的成本不在
"链路还活着"的检查范围里），预检每项 ✓ 才算 PASS。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common
from shensi.recipes.shensi import rl

ARMS = ("stage2_math", "stage2_code", "stage2_agent", "stage2_writing")


def preflight_arm(arm: str) -> bool:
    here = common.RECIPE / "stage2_rl" / arm
    checks: list[tuple[str, bool, str]] = []

    # ① 配置能读且能映射成 verl 命令（不执行）
    try:
        cfg = common.base.resolve_cfg(rl._load_with_base(here / "config/default.yaml"))
        cmd = common.build_verl_command(
            cfg, arm, Path("/tmp/rl_data_placeholder"), here / "reward.py"
        )
        checks.append(("config → verl CLI", True, f"{len(cmd)} 段命令"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("config → verl CLI", False, str(exc)[:120]))
        for name, ok, note in checks:
            print(f"  [{'✓' if ok else '✗'}] {name}：{note}")
        return False

    # ② 奖励模块可导入且有 compute_score
    try:
        spec = importlib.util.spec_from_file_location(f"{arm}_reward", here / "reward.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        has_fn = callable(getattr(mod, "compute_score", None))
        checks.append(("reward.compute_score", has_fn, "可导入" if has_fn else "缺 compute_score"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("reward.compute_score", False, str(exc)[:120]))

    # ③ verl 可导入
    try:
        importlib.import_module("verl.trainer.main_ppo")
        checks.append(("verl 可导入", True, "import verl.trainer.main_ppo"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("verl 可导入", False, str(exc)[:120]))

    # ④ 数据（parquet 在就 ✓，不在只提示）
    data_dir = common.env_paths()["data"] / arm
    has_parquet = (data_dir / "train.parquet").is_file()
    checks.append(
        (
            "RL 数据 parquet",
            has_parquet,
            str(data_dir) if has_parquet else "未准备（先 data_prep.py）",
        )
    )

    for name, ok, note in checks:
        print(f"  [{'✓' if ok else ('·' if name.endswith('parquet') else '✗')}] {name}：{note}")
    return all(ok for name, ok, _ in checks if not name.endswith("parquet"))


def main() -> int:
    ok_all = True
    for arm in ARMS:
        print(f"[test_train:{arm}]")
        ok_all &= preflight_arm(arm)
    print(f"[test_train:stage2_rl] {'PASS' if ok_all else 'FAIL'}（数据项只提示不拦）")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
