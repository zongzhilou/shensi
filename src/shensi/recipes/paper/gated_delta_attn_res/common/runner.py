"""GDAR 的训练运行时：run 目录、启动、冒烟与早停看门狗计划。"""

from __future__ import annotations

from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res.common.train import launcher

from .config import smoke_config

EARLY_STOP_DEFAULTS: dict[str, dict] = {
    "default": {"metric": "lm loss value", "mode": "min", "patience": 20, "grace": 1200.0},
    "stage2_rl": {"metric": "val/reward", "mode": "max", "patience": 8, "grace": 3600.0},
}


def early_stop_plan(stage: str, cfg: dict, patience: int | None = None) -> dict:
    plan = dict(EARLY_STOP_DEFAULTS.get(stage, EARLY_STOP_DEFAULTS["default"]))
    plan.update(
        {
            k: v
            for k, v in ((cfg.get("experiment") or {}).get("early_stop") or {}).items()
            if v is not None
        }
    )
    if patience is not None:
        plan["patience"] = int(patience)
    plan.setdefault("poll", 30.0)
    return plan


def write_run_dir(cfg: dict) -> Path:
    run_dir = Path(cfg["experiment"]["exp_dir"])
    run_dir = launcher.write_run_dir(cfg, run_dir)
    print(f"[gdar] 配置与命令已写入 {run_dir}（config.yaml / run.sh）")
    return run_dir


def run(cfg: dict, dry_run: bool, watch: dict | None = None) -> int:
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir, dry_run=dry_run, watch=watch)


def smoke(stage: str, profile: str = "tiny", override: list[str] | None = None) -> int:
    cfg = smoke_config(stage, profile, override)
    print(f"[gdar] 冒烟档：{stage} / tiny 几何 / mock 数据 / 5 步（配置见 config/{profile}.yaml）")
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir)
