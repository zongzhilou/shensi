#!/usr/bin/env python3
"""基准评测：起 vLLM 端点 → OpenCompass 跑 LLM 基准 → 可选 agent 类走 harness。

    python eval.py --dry-run                    # 只打印命令（vllm serve / opencompass / dsh）
    python eval.py --config tiny --suite mini   # 离线小档：tiny 检查点 + 冒烟子集
    python eval.py --suite minicpm5             # MiniCPM5-2B 口径的主力项
    python eval.py                              # 默认：OpenCompass 自带 leaderboard 集合
    python eval.py --suite agent                # 工具类基准（交给 deepseek-harness）

口径表见 ``benchmarks.py``（口径名 ↔ OpenCompass 数据集模块）；OpenCompass 装在独立 venv
（`.venv-opencompass`），由 ``bash setup_env.sh`` 准备。
"""

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
from shensi.recipes.paper.looma.stage5_eval import benchmarks  # noqa: E402
from shensi.recipes.shensi.common import common, harness  # noqa: E402

_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def resolve_env(value):
    """展开配置里的 ``${oc.env:变量,默认值}``（字典/列表逐项递归）。

    本文件用的是朴素 YAML 装载器，插值得自己做；不展开的话，路径会以字面量传进子进程。
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    return value


def run_dir_of(cfg: dict, stage: str) -> Path:
    """产物目录：``<runs>/looma/<stage>/<profile>/``。"""
    fs = Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
    runs = Path(os.environ.get("SHENSI_RUNS", str(fs / "shensi" / "runs")))
    return runs / "looma" / stage / str(cfg.get("experiment", {}).get("profile", "default"))


def build_vllm_command(serving: dict) -> list[str]:
    """按 ``serving`` 段拼 ``vllm serve``。"""
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
    """给的 ``max_model_len`` 超过检查点的 ``max_position_embeddings`` 时压回上限，免得 vLLM 拒启。"""
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
    """轮询 OpenAI 兼容端点直到就绪。"""
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
    """定 OpenCompass 的 ``datasets`` 取值：显式 ``--set`` → 命令行集合 → 配置值 → leaderboard。

    口径名（或集合名）在这里翻成 OpenCompass 的模块前缀；已经是前缀 / ``leaderboard`` /
    ``all`` 的原样传下去。
    """
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
    """Agent 类基准交给 harness（deepseek-harness）跑，端点用同一个；一个数据集一条命令。"""
    bench = harness.harness_cfg(cfg) or {}
    return [
        harness.command(cfg, task=name, extra=list(bench.get("extra_args") or [])) for name in names
    ]


def harness_env_of(cfg: dict) -> dict:
    """Harness 要的环境：dsh home + 我们的端点。"""
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
    """读 ``config/<profile>.yaml``。"""
    path = HERE / "config" / f"{profile}.yaml"
    if not path.is_file():
        raise SystemExit(f"[eval] 没有这个档：{path}")
    cfg = resolve_env(load_yaml(path) or {})
    cfg.pop("defaults", None)
    apply_overrides(cfg, overrides)
    cfg.setdefault("experiment", {})["profile"] = profile
    return cfg


def main() -> int:
    """评测入口：起 vLLM 端点 → OpenCompass 基准 → 可选 harness。"""
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
        "--limit", type=int, default=None, help="调试：OpenCompass 走 --debug（少量样本）"
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
    run_oc = bool(open_names) or not picked
    if args.limit:
        cfg.setdefault("opencompass", {})["limit"] = int(args.limit)
    explicit_ds = any(item.split("=", 1)[0] == "opencompass.datasets" for item in args.override)
    cfg.setdefault("opencompass", {})["datasets"] = opencompass_datasets(
        cfg, open_names, explicit_ds
    )

    out_dir = run_dir_of(cfg, "stage5_eval")
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
