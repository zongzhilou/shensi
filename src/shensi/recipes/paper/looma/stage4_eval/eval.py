#!/usr/bin/env python3
"""评测入口：起 vLLM 端点、跑 OpenCompass、可选跑 harness 并汇总。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parent

if str(RECIPE.parents[3]) not in sys.path:
    sys.path.insert(0, str(RECIPE.parents[3]))

from shensi.recipes.paper.looma.common import apply_overrides, load_yaml  # noqa: E402
from shensi.recipes.paper.looma.stage4_eval import benchmarks  # noqa: E402
from shensi.recipes.shensi.common import common, harness  # noqa: E402

_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def resolve_env(value):
    """解析运行环境变量（root / FS / 端点等）。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    return value


def run_dir_of(cfg: dict, stage: str) -> Path:
    """本次评测的产物目录。"""
    fs = Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
    runs = Path(os.environ.get("SHENSI_RUNS", str(fs / "shensi" / "runs")))
    return runs / "looma" / stage / str(cfg.get("experiment", {}).get("profile", "default"))


def build_vllm_command(serving: dict) -> list[str]:
    """组装 ``vllm serve`` 的启动命令。"""
    model_path = str(serving["model_path"])
    name = str(serving.get("served_model_name") or "looma")
    if not Path(model_path).exists():
        raise SystemExit(
            f"[eval] 模型目录不存在：{model_path}（先用 common/train/export_hf.py 导出）"
        )
    cmd = ["vllm", "serve", model_path, "--served-model-name", name]
    flags = (
        ("host", "--host"),
        ("port", "--port"),
        ("tensor_parallel_size", "--tensor-parallel-size"),
        ("data_parallel_size", "--data-parallel-size"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
        ("kv_cache_dtype", "--kv-cache-dtype"),
        ("max_model_len", "--max-model-len"),
        ("dtype", "--dtype"),
    )
    for key, flag in flags:
        value = serving.get(key)
        if value is not None and value != "auto":
            cmd += [flag, str(value)]
    if serving.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    if serving.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    cmd += [str(item) for item in (serving.get("extra_args") or [])]
    return cmd


def cap_max_model_len(serving: dict) -> None:
    """按模型几何与显存上限收窄 max_model_len。"""
    want = serving.get("max_model_len")
    cfg_path = Path(str(serving["model_path"])) / "config.json"
    if not want or not cfg_path.is_file():
        return
    try:
        limit = int(
            json.loads(cfg_path.read_text(encoding="utf-8")).get("max_position_embeddings") or 0
        )
    except (ValueError, OSError):
        return
    if limit and int(want) > limit:
        print(f"[eval] max_model_len {want} > 模型上限 {limit}，压到 {limit}")
        serving["max_model_len"] = limit


def wait_healthy(base_url: str, timeout_s: float = 1200.0, interval_s: float = 5.0) -> bool:
    """探活直到端点可用或超时。"""
    base = base_url.rstrip("/")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(base + path, timeout=10) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
        time.sleep(interval_s)
    return False


def opencompass_datasets(cfg: dict, picked: tuple[str, ...], explicit: bool) -> str:
    """选出本次要跑的 OpenCompass 数据集。"""
    if explicit:
        return str((cfg.get("opencompass") or {}).get("datasets") or "leaderboard")
    if picked:
        chosen = ",".join(str(prefix) for prefix in benchmarks.oc_entries(picked))
    else:
        chosen = str((cfg.get("opencompass") or {}).get("datasets") or "leaderboard")
    if chosen in benchmarks.OC_SETS:
        chosen = ",".join(str(prefix) for prefix in benchmarks.oc_entries((chosen,)))
    return chosen


def harness_commands(cfg: dict, names: tuple[str, ...]) -> list[list[str]]:
    """组装 harness（工具类基准）的命令。"""
    bench = harness.harness_cfg(cfg) or {}
    return [
        harness.command(cfg, task=name, extra=list(bench.get("extra_args") or [])) for name in names
    ]


def harness_env_of(cfg: dict) -> dict:
    endpoint = cfg.get("endpoint") or {}
    env = dict(
        harness.harness_env(
            cfg,
            base_url=str(endpoint.get("base_url", "")),
            model=str(endpoint.get("model", "")),
        )
    )
    env.setdefault("DEEPSEEK_API_KEY", "dummy")
    return env


def load_config(profile: str, overrides: list[str] | None = None) -> dict:
    """读评测配置。"""
    path = HERE / "config" / f"{profile}.yaml"
    if not path.is_file():
        raise SystemExit(f"[eval] 没有这个档：{path}")
    cfg = resolve_env(load_yaml(path) or {})
    cfg.pop("defaults", None)
    apply_overrides(cfg, overrides)
    cfg.setdefault("experiment", {})["profile"] = profile
    return cfg


def main() -> int:
    """评测入口：起端点、跑 OpenCompass、可选跑 harness 并汇总。"""
    ap = argparse.ArgumentParser(description="Looma 基准评测（vLLM 端点 + OpenCompass + harness）")
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml")
    ap.add_argument("--config", default=None, help="配置文件路径（与 --profile 等价）")
    ap.add_argument(
        "--suite",
        default="leaderboard",
        choices=["leaderboard", "minicpm5", "mini", "long", "agent", "all"],
        help="评测集合：leaderboard（自带集合）/ minicpm5（口径主力）/ mini（冒烟）/ long / agent / all",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个数据集最多评多少条（写进 reader_cfg.test_range）",
    )
    ap.add_argument("--model-path", default=None, help="覆盖 serving.model_path")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    ap.add_argument("--no-serve", action="store_true", help="端点已起好，直接打")
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号键覆写，可多次"
    )
    args = ap.parse_args()
    profile = args.profile if not args.config else Path(args.config).stem
    cfg = load_config(profile, args.override)
    if args.model_path:
        cfg.setdefault("serving", {})["model_path"] = args.model_path

    picked = benchmarks.resolve([args.suite]) if args.suite != "leaderboard" else ()
    agent_names = benchmarks.subset(picked, benchmarks.AGENT) if picked else ()
    open_names = benchmarks.subset(picked, benchmarks.OPEN) if picked else ()
    oc_names = tuple(name for name in open_names if benchmarks.oc_name(name))
    skipped = [name for name in open_names if not benchmarks.oc_name(name)]
    if skipped:
        print(f"[eval] 这些口径项没有 OpenCompass 数据集，跳过：{skipped}（要跑得另配数据集）")
    run_oc = bool(oc_names) or not picked
    if args.limit:
        cfg.setdefault("opencompass", {})["limit"] = int(args.limit)
    explicit_ds = any(item.split("=", 1)[0] == "opencompass.datasets" for item in args.override)
    cfg.setdefault("opencompass", {})["datasets"] = opencompass_datasets(cfg, oc_names, explicit_ds)

    out_dir = run_dir_of(cfg, "stage4_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE))
    import opencompass_eval  # noqa: E402

    serving = cfg.get("serving") or {}
    vllm_cmd = build_vllm_command(serving)
    oc_conf = opencompass_eval.build_config(cfg, out_dir)
    oc_cmd = opencompass_eval.build_command(cfg, oc_conf, out_dir / "opencompass")
    harness_cmds = harness_commands(cfg, agent_names)

    print(f"[eval] 档：{profile}  端点：{cfg['endpoint']['base_url']}")
    print(f"[eval] 集合：{args.suite}  OpenCompass datasets={cfg['opencompass']['datasets']}")
    if agent_names:
        print(f"[eval] 工具集（harness）：{list(agent_names)}")
    print("[eval] vLLM      ：\n  " + " ".join(vllm_cmd))
    if run_oc:
        print("[eval] OpenCompass：\n  " + " ".join(oc_cmd))
    for cmd in harness_cmds:
        print("[eval] harness   ：\n  " + " ".join(cmd))
    if args.dry_run:
        return 0

    proc = None
    if serving.get("start", True) and not args.no_serve:
        cap_max_model_len(serving)
        vllm_cmd = build_vllm_command(serving)
        proc = subprocess.Popen(vllm_cmd, env=common.subprocess_env(strip_proxy=True))
        if not wait_healthy(str(cfg["endpoint"]["base_url"])):
            proc.terminate()
            raise SystemExit(f"[eval] 端点没起来：{cfg['endpoint']['base_url']}")

    result: dict = {
        "profile": profile,
        "suite": args.suite,
        "endpoint": cfg["endpoint"]["base_url"],
    }
    try:
        if run_oc:
            result["opencompass"] = opencompass_eval.run(cfg, out_dir, dry_run=False)
        if harness_cmds:
            env = common.subprocess_env()
            env.update(harness_env_of(cfg))
            rcs = [subprocess.call(cmd, env=env) for cmd in harness_cmds]
            result["harness"] = {"rc": rcs, "datasets": list(agent_names)}
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=60)

    report = out_dir / "summary.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval] summary 写入 {report}")
    print(f"[eval] OpenCompass 产物：{out_dir / 'opencompass'}（predictions / results / summary）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
