"""训练运行时：写 run 目录、启动训练、冒烟档与早停看门狗计划。"""

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
    """给出该 stage 的早停计划（指标、方向、耐心与宽限期），可用参数覆盖。"""
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
    """把 config.yaml 与 run.sh 写进运行目录并返回该目录。"""
    run_dir = Path(cfg["experiment"]["exp_dir"])
    run_dir = launcher.write_run_dir(cfg, run_dir)
    print(f"[gdar] 配置与命令已写入 {run_dir}（config.yaml / run.sh）")
    return run_dir


def run(cfg: dict, dry_run: bool, watch: dict | None = None) -> int:
    """写运行目录并启动训练（同会话带早停看门狗）。"""
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir, dry_run=dry_run, watch=watch)


def smoke(stage: str, profile: str = "tiny", override: list[str] | None = None) -> int:
    """按冒烟档配置启动 tiny 规模训练。"""
    cfg = smoke_config(stage, profile, override)
    print(f"[gdar] 冒烟档：{stage} / tiny 几何 / mock 数据 / 5 步（配置见 config/{profile}.yaml）")
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir)
