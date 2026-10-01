"""本配方自己的 launcher：与 `shensi.recipes.shensi.train.launcher` 同一套语义，
只是训练入口换成 GDAR 的 `train_gdar.py`。

- 摊平（`train.{system,model,data}` → mcore CLI）、`apply_defaults`、`build_env`
  直接复用 shensi 那份（一个出处）；
- `build_command` / `write_run_dir` / `launch` 在这里重写，因为入口路径不同——
  `--spec` 指到本配方 `models/` 的层规格（见 config 里 `train.model.spec`）。
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from shensi.recipes.shensi.train import launcher as base

#: 本配方的训练入口（区别于 shensi 配方的 train_shensi.py）
ENTRY = Path(__file__).resolve().parent / "train_gdar.py"


def entry_path() -> Path:
    return ENTRY


def build_command(cfg: dict, override: list[str] | None = None) -> list[str]:
    """Torchrun 命令；`--nnodes 1` 时直接用 `--standalone`。"""
    runner = (cfg.get("experiment") or {}).get("runner") or {}
    nproc = int(runner.get("nproc_per_node") or 0) or base._visible_devices()
    if int(runner.get("nnodes") or 1) != 1:
        raise SystemExit("[gdar] 本 launcher 只跑单机（nnodes=1）；多机请自行起 torchrun")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(entry_path()),
    ]
    cmd += base.flatten_train_section(cfg["train"])
    cmd += list(override or [])
    return cmd


def write_run_dir(cfg: dict, run_dir: Path) -> Path:
    """把最终配置与将要执行的命令落到 exp_dir（可复现、可手工照抄）。"""
    base.apply_defaults(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    cmd = build_command(cfg)
    (run_dir / "run.sh").write_text("#!/bin/bash\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    return run_dir


def launch(cfg: dict, run_dir: Path, dry_run: bool = False) -> int:
    """跑一次训练；输出同时进 stdout 与 `<exp_dir>/logs/host_0_localhost.output`。"""
    base.apply_defaults(cfg)
    log_path = Path(cfg["experiment"]["exp_dir"]) / "logs/host_0_localhost.output"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg)
    env = base.build_env(cfg)
    print("[gdar] 命令：\n  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"[gdar] 日志：{log_path}")
    if dry_run:
        return 0
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
        return proc.wait()
