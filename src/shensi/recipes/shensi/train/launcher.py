"""本地单机 launcher：配置 yaml 的 `train.{system,model,data}` → mcore CLI → torchrun。

取代 FlagScale 的 runner（单机 1~8 卡用不上它的容错/调度）：
- 摊平语义与 FlagScale 一致（`_`→`-`、嵌套字典不带前缀、`no_xxx: true` 表示关掉、
  list 摊成 `--key v1 v2 ...`），所以既有 yaml 不用改；
- 产物仍是 `<exp_dir>/config.yaml` 与 `<exp_dir>/logs/host_0_localhost.output`，
  `early_stop.py`、`common.watch()` 照旧能用；
- 跑在前台：返回码就是训练进程的返回码（没有"提交即返回"的异步语义）。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ENTRY = Path(__file__).resolve().parent / "train_shensi.py"
IGNORE_KEYS = ("log_dir", "details_dir", "scripts_dir", "pids_dir", "straggler_dir")


def flatten_train_section(cfg: dict) -> list[str]:
    """`train.{system,model,data}` → mcore 的 CLI 参数（后者覆盖前者同名字段）。"""
    merged: dict = {}
    for section in ("system", "model", "data"):
        merged.update(cfg.get(section) or {})
    return _flatten(merged)


def _flatten(node: dict) -> list[str]:
    out: list[str] = []
    for key, value in node.items():
        if key in IGNORE_KEYS:
            continue
        key = str(key).replace("_", "-")
        if isinstance(value, dict):
            out.extend(_flatten(value))
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            out.append(f"--{key}")
            out.extend(str(v) for v in value)
        elif isinstance(value, bool):
            if value:
                out.append(f"--{key}")
        elif value is None:
            continue
        else:
            out.append(f"--{key}")
            out.append(str(value))
    return out


def entry_path() -> Path:
    """训练入口文件（本包内的 `train_shensi.py`）。"""
    return ENTRY


def apply_defaults(cfg: dict) -> dict:
    """补默认值（幂等）：`train.system.checkpoint.save` 没写就落 `<exp_dir>/ckpt`。

    FlagScale 当年由 runner 替我们填这一项，现在由 launcher 填——不填的话上游
    `args.save is None`，`save_interval` 再小也一个检查点都不存。
    """
    ckpt = cfg["train"].setdefault("system", {}).setdefault("checkpoint", {})
    ckpt.setdefault("save", str(Path(cfg["experiment"]["exp_dir"]) / "ckpt"))
    return cfg


def build_command(cfg: dict, override: list[str] | None = None) -> list[str]:
    """Torchrun 命令；`--nnodes 1` 时直接用 `--standalone`（静态 rendezvous）。"""
    runner = (cfg.get("experiment") or {}).get("runner") or {}
    nproc = int(runner.get("nproc_per_node") or 0) or _visible_devices()
    if int(runner.get("nnodes") or 1) != 1:
        raise SystemExit("[recipe] 本 launcher 只跑单机（nnodes=1）；多机请自行起 torchrun")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(entry_path()),
    ]
    cmd += flatten_train_section(cfg["train"])
    cmd += list(override or [])
    return cmd


def _visible_devices() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw:
        return len([d for d in raw.split(",") if d.strip()])
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def build_env(cfg: dict) -> dict:
    """训练进程的环境：配置里的 `experiment.envs` + 把本 venv 的 bin 放到 PATH 最前。

    mcore 的 `core/datasets/Makefile` 硬编码了 `python3 -m pybind11 --includes`：首次起训会在包目录里
    现编一次 dataset helpers，PATH 最前面不是本 venv 的 python3 就会编不过（缺 pybind11）。
    """
    env = dict(os.environ)
    env.update(
        {str(k): str(v) for k, v in ((cfg.get("experiment") or {}).get("envs") or {}).items()}
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if Path("/usr/local/cuda/bin/nvcc").exists():
        # flashinfer 在 SM120 上要靠 JIT 补稀疏 MLA 内核
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
    # SM120 上 FlagGems 的 flagos 后端在 te_general_grouped_gemm 上会段错误（RL 那条线同样处理）
    env.setdefault("TE_FL_PREFER", "vendor")
    return env


def write_run_dir(cfg: dict, run_dir: Path) -> Path:
    """把最终配置与将要执行的命令落到 exp_dir（可复现、可手工照抄）。"""
    apply_defaults(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    cmd = build_command(cfg)
    (run_dir / "run.sh").write_text("#!/bin/bash\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    return run_dir


def launch(cfg: dict, run_dir: Path, dry_run: bool = False) -> int:
    """跑一次训练；输出同时进 stdout 与 `<exp_dir>/logs/host_0_localhost.output`。"""
    apply_defaults(cfg)
    log_path = Path(cfg["experiment"]["exp_dir"]) / "logs/host_0_localhost.output"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg)
    env = build_env(cfg)
    print("[recipe] 命令：\n  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"[recipe] 日志：{log_path}")
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


def wait_for_finish(exp_dir, poll: int = 15) -> None:
    """等本机这次 run 收尾（我们跑在前台，这里只是给串接多阶段的旧接口兜底）。"""
    out = Path(exp_dir) / "logs/host_0_localhost.output"
    while True:
        if out.is_file():
            text = out.read_text(encoding="utf-8", errors="ignore")
            if "after training is done" in text or "Traceback" in text:
                print(
                    "[recipe] 本次 run 已收尾"
                    if "after" in text
                    else f"[recipe] 有 Traceback：{out}"
                )
                return
        time.sleep(poll)
