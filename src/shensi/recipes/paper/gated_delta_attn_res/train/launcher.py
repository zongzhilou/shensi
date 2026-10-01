"""训练启动器：组装 torchrun 命令、写 run 目录、前台跑训练并同会话起早停看门狗。"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from shensi.recipes.shensi.train import launcher as base

ENTRY = Path(__file__).resolve().parent / "train_gdar.py"


def entry_path() -> Path:
    return ENTRY


def build_command(cfg: dict, override: list[str] | None = None) -> list[str]:
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
    if "--muon-scalar-optimizer" in cmd:
        i = cmd.index("--muon-scalar-optimizer")
        # mcore 的 choices 只认 adam/lion，换成本配方自己的参数名，解析后落到 OptimizerConfig
        cmd[i] = "--gdar-scalar-optimizer"
    model = (cfg.get("train") or {}).get("model") or {}
    heads, groups = model.get("num_attention_heads"), model.get("num_query_groups")
    if (
        heads
        and groups
        and int(groups) < int(heads)
        and "--group-query-attention" not in cmd
        and "group_query_attention" not in cmd
    ):
        # mcore 的 --num-query-groups 只在 --group-query-attention 在场时生效，否则静默变 MHA
        cmd.append("--group-query-attention")
    system = (cfg.get("train") or {}).get("system") or {}
    explicit_qk = bool(model.get("qk_layernorm")) or bool(system.get("qk_layernorm"))
    if not explicit_qk and "--qk-layernorm" not in cmd and "qk_layernorm" not in cmd:
        # Qwen3 的 q/k 归一化是架构的一部分（HF 参考实现里有这两组权重）
        cmd.append("--qk-layernorm")
    cmd += list(override or [])
    return cmd


def write_run_dir(cfg: dict, run_dir: Path) -> Path:
    base.apply_defaults(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    cmd = build_command(cfg)
    (run_dir / "run.sh").write_text("#!/bin/bash\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    return run_dir


def spawn_with_watchdog(
    cmd: list[str],
    env: dict,
    run_dir: Path,
    log_path: Path,
    *,
    watch: dict | None = None,
) -> tuple[int, dict | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    watchdog = None
    report_path = None
    if watch:
        report_path = log_path.parent / "early_stop.json"
        trainer = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        early_stop_py = Path(__file__).resolve().parents[3] / "shensi" / "early_stop.py"
        if not early_stop_py.is_file():
            raise SystemExit(
                f"[gdar] 找不到早停看门狗脚本：{early_stop_py}（--no-early-stop 可跳过）"
            )

        def _num(key: str, default: float) -> float:
            v = watch.get(key)
            return default if v is None else float(v)

        wcmd = [
            sys.executable,
            str(early_stop_py),
            "--log",
            str(log_path),
            "--metric",
            str(watch.get("metric") or "validation loss"),
            "--mode",
            str(watch.get("mode") or "min"),
            "--patience",
            str(int(_num("patience", 20))),
            "--min-delta",
            str(_num("min_delta", 1e-4)),
            "--grace",
            str(_num("grace", 900.0)),
            "--poll",
            str(_num("poll", 30.0)),
            "--training-pgid",
            str(trainer.pid),
            "--report",
            str(report_path),
        ]
        if watch.get("max_wait") is not None:
            wcmd += ["--max-wait", str(float(watch["max_wait"]))]
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
            early_out = watchdog.stdout.read() if watchdog.stdout else ""
            print(
                "[early_stop] 看门狗启动失败：\n  "
                + (early_out or "(无输出)。").replace("\n", "\n  ")
            )
            raise SystemExit("[gdar] 早停看门狗起不来（--no-early-stop 可跳过）")
    else:
        trainer = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    assert trainer.stdout is not None
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        for line in trainer.stdout:
            log.write(line)
            sys.stdout.write(line)
    rc = trainer.wait()
    if watchdog is not None:
        try:
            wd_out, _ = watchdog.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            watchdog.kill()
            wd_out = ""
        for line in (wd_out or "").splitlines():
            print("[early_stop] " + line)
    report = None
    if report_path is not None and report_path.is_file():
        import json

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  报告坏了不算数
            report = None
    return rc, report


def launch(cfg: dict, run_dir: Path, dry_run: bool = False, watch: dict | None = None) -> int:
    base.apply_defaults(cfg)
    log_path = Path(cfg["experiment"]["exp_dir"]) / "logs/host_0_localhost.output"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg)
    env = base.build_env(cfg)
    print("[gdar] 命令：\n  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"[gdar] 日志：{log_path}")
    if watch:
        print(
            f"[gdar] 早停看门狗：metric={watch.get('metric')} mode={watch.get('mode')} "
            f"patience={watch.get('patience')}（--no-early-stop 可关）"
        )
    if dry_run:
        return 0
    rc, report = spawn_with_watchdog(cmd, env, run_dir, log_path, watch=watch)
    if report is not None:
        print(
            f"[gdar] 早停生效：{report.get('why')}（metric={report.get('metric')} "
            f"best={report.get('best')}）——按成功处理"
        )
        return 0
    return rc
