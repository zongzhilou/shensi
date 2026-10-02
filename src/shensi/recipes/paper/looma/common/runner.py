"""训练运行时：命令组装、run 目录、启动、冒烟与早停看门狗。"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from shensi.recipes.shensi.common.train import launcher as base_launcher

from .config import smoke_config
from .paths import RECIPE

ENTRY = RECIPE / "common" / "train" / "train_looma.py"

EARLY_STOP_DEFAULTS: dict[str, dict] = {
    "default": {"metric": "lm loss value", "mode": "min", "patience": 20, "grace": 1200.0},
    "stage2_rl": {"metric": "val/reward", "mode": "max", "patience": 8, "grace": 3600.0},
}


def early_stop_plan(stage: str, cfg: dict, patience: int | None = None) -> dict:
    """给出该 stage 的早停计划（指标、方向、耐心与宽限期），可用参数覆盖。"""
    plan = dict(EARLY_STOP_DEFAULTS.get(stage, EARLY_STOP_DEFAULTS["default"]))
    configured = (cfg.get("experiment") or {}).get("early_stop") or {}
    plan.update({k: v for k, v in configured.items() if v is not None})
    if patience is not None:
        plan["patience"] = int(patience)
    plan.setdefault("poll", 30.0)
    return plan


def build_command(cfg: dict, override: list[str] | None = None) -> list[str]:
    """按配置组装 torchrun 命令（含并行度与派生开关）。"""
    runner = (cfg.get("experiment") or {}).get("runner") or {}
    nproc = int(runner.get("nproc_per_node") or 0) or base_launcher._visible_devices()
    if int(runner.get("nnodes") or 1) != 1:
        raise SystemExit("[looma] 只支持单机（nnodes=1）")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(ENTRY),
    ]
    cmd += base_launcher.flatten_train_section(cfg["train"])
    model = (cfg.get("train") or {}).get("model") or {}
    heads, groups = model.get("num_attention_heads"), model.get("num_query_groups")
    if heads and groups and int(groups) < int(heads) and "--group-query-attention" not in cmd:
        cmd.append("--group-query-attention")
    if model.get("qk_layernorm"):
        raise SystemExit("[looma] 骨干是 Llama，没有 qk norm")
    return cmd + list(override or [])


def write_run_dir(cfg: dict, run_dir: Path | None = None) -> Path:
    """把 config.yaml 与 run.sh 写进运行目录并返回该目录。"""
    base_launcher.apply_defaults(cfg)
    run_dir = run_dir or Path(cfg["experiment"]["exp_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    cmd = build_command(cfg)
    (run_dir / "run.sh").write_text("#!/bin/bash\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    return run_dir


def spawn(
    cmd: list[str], env: dict, run_dir: Path, log_path: Path, watch: dict | None = None
) -> tuple[int, dict | None]:
    """前台启动一个命令并在同一会话里看护早停。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    watchdog = report_path = None
    if watch:
        report_path = log_path.parent / "early_stop.json"
        if report_path.exists():
            report_path.unlink()  # 上一轮的早停报告不能冒充这一轮的
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        watchdog_py = RECIPE.parent.parent / "shensi" / "common" / "early_stop.py"
        if not watchdog_py.is_file():
            raise SystemExit(f"[looma] 找不到早停脚本：{watchdog_py}（--no-early-stop 可跳过）")
        wcmd = [
            sys.executable,
            str(watchdog_py),
            "--log",
            str(log_path),
            "--metric",
            str(watch.get("metric") or "lm loss value"),
            "--mode",
            str(watch.get("mode") or "min"),
            "--patience",
            str(int(watch.get("patience") or 20)),
            "--min-delta",
            str(float(watch.get("min_delta") or 1e-4)),
            "--grace",
            str(float(watch.get("grace") or 900.0)),
            "--poll",
            str(float(watch.get("poll") or 30.0)),
            "--training-pgid",
            str(proc.pid),
            "--report",
            str(report_path),
        ]
        watchdog = subprocess.Popen(
            wcmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(1.0)
        if watchdog.poll() is not None:
            out = watchdog.stdout.read() if watchdog.stdout else ""
            raise SystemExit(f"[looma] 早停看门狗起不来：\n{out}")
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
    rc = proc.wait()
    if watchdog is not None:
        try:
            out, _ = watchdog.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            watchdog.kill()
            out = ""
        for line in (out or "").splitlines():
            print("[early_stop] " + line)
    report = None
    if report_path is not None and report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = None
    return rc, report


def launch(cfg: dict, dry_run: bool = False, watch: dict | None = None) -> int:
    """写运行目录后启动训练；``dry_run`` 时只打印命令。"""
    run_dir = write_run_dir(cfg)
    log_path = Path(cfg["experiment"]["exp_dir"]) / "logs" / "host_0_localhost.output"
    cmd = build_command(cfg)
    env = base_launcher.build_env(cfg)
    print("[looma] 命令：\n  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"[looma] 日志：{log_path}")
    if dry_run:
        return 0
    rc, report = spawn(cmd, env, run_dir, log_path, watch=watch)
    if report is not None:
        print(f"[looma] 早停生效：{report.get('why')}（metric={report.get('metric')}）——按成功处理")
        return 0
    return rc


def run(cfg: dict, dry_run: bool, watch: dict | None = None) -> int:
    """写运行目录并启动训练（同会话带早停看门狗）。"""
    return launch(cfg, dry_run=dry_run, watch=watch)


def smoke(stage: str, profile: str = "tiny", override: list[str] | None = None) -> int:
    """按冒烟档配置启动 tiny 规模训练。"""
    cfg = smoke_config(stage, profile, override)
    ckpt = Path(cfg["train"]["system"]["checkpoint"]["save"])
    if ckpt.is_dir():
        shutil.rmtree(ckpt)
        print(f"[looma] 冒烟：已清理上次的检查点（{ckpt}）")
    print(f"[looma] 冒烟：{stage} / {profile} 几何 / mock 数据")
    return launch(cfg)
