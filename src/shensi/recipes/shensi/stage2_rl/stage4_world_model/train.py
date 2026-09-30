#!/usr/bin/env python3
# stage5 的三段训练：CPT（环境知识）→ SFT（下一状态）→ RL（模拟保真度）。
# 三段都复用现成训练器，本文件只负责把本阶段的档与数据接到那些训练器上。

import argparse
import subprocess
import sys
from itertools import chain
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import common, rl

_HERE = Path(__file__).resolve().parent
_RECIPES = _HERE.parents[2]

STAGE = "stage2_world_model"


def load_cfg(profile: str) -> dict:
    return common.resolve_cfg(rl._load_with_base(_HERE / f"config/{profile}.yaml"))


def step_cpt(cfg, args, paths, data_dir: Path, dry: bool) -> int:
    """① 环境知识：继续预训练，档与入口都用 stage0_pretrain/stage2_midtrain 的。"""
    prof = str(cfg["cpt"].get("profile") or "default")
    overrides = [
        f"experiment.exp_dir={paths['runs'] / STAGE / 'cpt'}",
        f"train.system.checkpoint.save={paths['ckpt'] / STAGE / 'cpt'}",
        "experiment.exp_name=shensi_stage2_world_model_cpt",
        *args.override,
    ]
    built = common.build_config(
        "stage2_midtrain", prof, overrides, data_dir / "cpt_bins", tokens=cfg["cpt"].get("tokens")
    )
    return common.run(built, "stage2_midtrain", f"world_model_cpt_{prof}", dry)


def step_sft(cfg, args, paths, data_dir: Path, dry: bool) -> int:
    """② 下一状态预测：stage1_sft 原样跑，只是数据换成世界模型的（动作 → 观测）。"""
    jsonl = data_dir / "sft/sft_train.jsonl"
    cmd = [
        sys.executable,
        str(_RECIPES / "stage1_sft/train.py"),
        "--profile",
        str(cfg["sft"].get("profile") or "debug"),
        "--data-jsonl",
        str(jsonl),
        "--set",
        "experiment.exp_name=shensi_stage2_world_model_sft",
        "--set",
        f"experiment.exp_dir={paths['runs'] / STAGE / 'sft'}",
        "--set",
        f"train.system.checkpoint.save={paths['ckpt'] / STAGE / 'sft'}",
        *chain.from_iterable(("--set", o) for o in args.override),
    ]
    if dry:
        cmd.append("--dry-run")
    print("[world_model] SFT 命令：\n  " + " \\\n    ".join(cmd))
    return 0 if dry else subprocess.call(cmd)


def step_rl(cfg, args, data_dir: Path, dry: bool) -> int:
    """③ 模拟保真度：verl 跑 GRPO，奖励是 reward.py 里的判分（AgentWorldBench 五维）。"""
    argv = [
        "--config",
        str(_HERE / "config/rl" / f"{cfg['rl'].get('profile') or 'default'}.yaml"),
        "--data-dir",
        str(data_dir),
    ]
    if dry:
        argv.append("--dry-run")
    for o in args.override:
        argv += ["--set", o]
    return rl.launch(STAGE, argv=argv, here=_HERE, reward=_HERE / "reward.py")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Shensi stage2_rl/stage4_world_model（世界模型：CPT → SFT → RL）"
    )
    ap.add_argument("--step", default="all", choices=("cpt", "sft", "rl", "all"))
    ap.add_argument(
        "--profile", default="default", help="config/<名字>.yaml（三段各自的上游档写在里面）"
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help=f"data_prep 的产物目录，默认 $SHENSI_FS/shensi/data/{STAGE}",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号覆盖，透传给各段"
    )
    args = ap.parse_args(argv)

    paths = common.env_paths()
    data_dir = Path(args.data_dir or paths["data"] / STAGE)
    cfg = load_cfg(args.profile)
    steps = ("cpt", "sft", "rl") if args.step == "all" else (args.step,)
    print(f"[world_model] 数据目录 {data_dir}；步骤 {list(steps)}（profile={args.profile}）")
    rc = 0
    for step in steps:
        print(f"\n[world_model] === {step} ===")
        if step == "cpt":
            rc = step_cpt(cfg, args, paths, data_dir, args.dry_run)
        elif step == "sft":
            rc = step_sft(cfg, args, paths, data_dir, args.dry_run)
        else:
            rc = step_rl(cfg, args, data_dir, args.dry_run)
        if rc:
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
