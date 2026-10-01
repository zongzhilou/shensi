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
import time
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
    # mcore 里 `--num-query-groups` **只在 `--group-query-attention` 在场时才生效**
    # （`training/argument_utils.py`：`if args.group_query_attention: num_query_groups=args...
    # else: num_query_groups=None` → `TransformerConfig` 退回 `num_attention_heads`，也就是
    # 静默变成 MHA）。本配方的配置声明的是 GQA（`num_query_groups < num_attention_heads`），
    # 所以这里按声明派生这个开关——不派生的话每次 run 实际训的都是另一种注意力几何。
    model = (cfg.get("train") or {}).get("model") or {}
    heads, groups = model.get("num_attention_heads"), model.get("num_query_groups")
    if (
        heads
        and groups
        and int(groups) < int(heads)
        and "--group-query-attention" not in cmd
        and "group_query_attention" not in cmd  # 配置里显式关掉时不越权
    ):
        cmd.append("--group-query-attention")
    # Qwen3 的 q_norm/k_norm 是架构的一部分（HF 参考实现里就有这两组权重），而 mcore 侧
    # 要 `--qk-layernorm` 才会建它们——不派生的话训出来的不是 Qwen3（实测参数计数差 512 =
    # 4 层 × q/k × 64）。配置里显式写了 `qk_layernorm: true` 就不重复加。
    system = (cfg.get("train") or {}).get("system") or {}
    explicit_qk = bool(model.get("qk_layernorm")) or bool(system.get("qk_layernorm"))
    if not explicit_qk and "--qk-layernorm" not in cmd and "qk_layernorm" not in cmd:
        cmd.append("--qk-layernorm")
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


def spawn_with_watchdog(
    cmd: list[str],
    env: dict,
    run_dir: Path,
    log_path: Path,
    *,
    watch: dict | None = None,
) -> tuple[int, dict | None]:
    """前台跑一条命令 + 可选看门狗；返回 ``(rc, 早停报告或 None)``。

    看门狗（``shensi.recipes.shensi.early_stop``）与训练**同会话组**起：训练用
    ``start_new_session=True`` 自成一个进程组，看门狗拿到它的 pgid 后（a）超耐心就只给
    这个进程组发 SIGTERM、写 ``--report``；（b）训练自己收官时看门狗看到 pgid 消失即退出。
    **早停算成功**：训练返回码非 0 但报告在，调用方按早停处理（不再当失败）。
    """
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
        # <recipe>/train/launcher.py → parents[3] = recipes/（early_stop.py 在 recipes/shensi/ 下）
        early_stop_py = Path(__file__).resolve().parents[3] / "shensi" / "early_stop.py"
        if not early_stop_py.is_file():
            raise SystemExit(
                f"[gdar] 找不到早停看门狗脚本：{early_stop_py}（--no-early-stop 可跳过）"
            )

        # 数值参数只在"没给"（None）时才取默认：0 是合法值（如 grace=0 表示不设启动宽限），
        # `x or default` 会把 0 吃掉——踩过一次。
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
        # 活性检查：看门狗若秒退（路径/参数错），它的 stderr 会被管道吞到最后——
        # 早停现在是默认契约，起不来必须立刻大声失败，而不是静默跑到底。
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
        except subprocess.TimeoutExpired:  # 看门狗卡住了：不强留
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
    """跑一次训练；输出同时进 stdout 与 `<exp_dir>/logs/host_0_localhost.output`。

    ``watch`` 非空时同会话组起早停看门狗（默认由 ``common.early_stop_plan`` 给，
    见各 stage 的 ``--no-early-stop``）。早停视为成功。
    """
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
