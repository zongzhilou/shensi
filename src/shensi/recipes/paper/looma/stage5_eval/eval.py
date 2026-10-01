#!/usr/bin/env python3
"""基准评测：起 vLLM 端点 → EvalScope 跑 MiniCPM5-2B 口径的评测集 → 可选 agent 类走 harness。

    python eval.py --dry-run                       # 只打印三条命令（vllm serve / evalscope / dsh）
    python eval.py --config tiny --limit 2         # 离线小档：tiny 检查点 + 每集少量样本
    python eval.py --suite main                    # 主表（通用/数学/指令/代码）
    python eval.py --suite agent                   # 工具类基准（交给 deepseek-harness）

评测集与口径见 ``benchmarks.py``；机器人的环境（独立评测 venv、EvalScope、dsh profile）由
``setup_env.sh`` 准备。端点只起一次，EvalScope 与 harness 打的是同一个 OpenAI 兼容端点。
"""

from __future__ import annotations

import argparse
import json
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


def eval_venv() -> Path:
    """独立评测 venv（setup_env.sh 建的那份）。"""
    import os

    root = Path(os.environ.get("SHENSI_ROOT", "/root/work/shensi"))
    return Path(os.environ.get("SHENSI_EVAL_VENV", str(root / ".venv_eval")))


def run_dir_of(cfg: dict, stage: str) -> Path:
    """产物目录：``<runs>/stage5_eval/<profile>/``。"""
    import os

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
        if value is not None:
            cmd += [flag, str(value)]
    if serving.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    if serving.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    cmd += [str(item) for item in (serving.get("extra_args") or [])]
    return cmd


def wait_healthy(base_url: str, timeout_s: float = 900.0, interval_s: float = 5.0) -> bool:
    """等端点就绪：先探 vLLM 的 ``/health``，再探 OpenAI 面的 ``/v1/models``。"""
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


_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def resolve_env(value):
    """展开配置里的 ``${oc.env:变量,默认值}``（字典/列表逐项递归）。

    本文件用的是朴素 YAML 装载器，插值得自己做；不展开的话，路径会以字面量传进子进程。
    """
    import os

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    return value


def dataset_root(cfg: dict) -> Path | None:
    """EvalScope 的数据集根（缓存与本地数据都放这儿）。"""
    import os

    raw = (cfg.get("evalscope") or {}).get("dataset_dir")
    if raw:
        return Path(str(raw))
    fs = Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
    return fs / "shensi" / "data" / "looma" / "stage5_eval" / "datasets"


def merged_dataset_args(cfg: dict, names: tuple[str, ...]) -> dict:
    """合并 ``--dataset-args``：先放 HF 双胞胎覆写，再让配置里的同名字段覆盖它。

    只在 ``dataset_hub: huggingface`` 时注入双胞胎——默认 id 是 ModelScope 的项，换到 HF 得先换成
    HF 上的同名仓库，否则整集拉不下来。
    """
    bench = cfg.get("evalscope") or {}
    args: dict = {}
    if str(bench.get("dataset_hub") or "").lower() == "huggingface":
        args.update(benchmarks.hf_dataset_args(names))
    for name, extra in (bench.get("dataset_args") or {}).items():
        args.setdefault(name, {}).update(extra or {})
    return args


def build_evalscope_command(
    cfg: dict, names: tuple[str, ...], out_dir: Path, limit: int | None
) -> list[str]:
    """按 ``evalscope`` 段拼 EvalScope 的命令（打的是同一个端点）。"""
    bench = cfg.get("evalscope") or {}
    endpoint = cfg.get("endpoint") or {}
    cmd = [
        str(eval_venv() / "bin" / "evalscope"),
        "eval",
        "--model",
        str(endpoint.get("model") or "looma"),
        "--api-url",
        f"{str(endpoint['base_url']).rstrip('/')}/v1",
        "--api-key",
        str(endpoint.get("api_key") or "dummy"),
        "--datasets",
        *names,
        "--work-dir",
        str(out_dir / "evalscope"),
    ]
    cmd += ["--eval-type", "openai_api"]
    root = dataset_root(cfg)
    if root is not None:
        cmd += ["--dataset-dir", str(root)]
    if bench.get("dataset_hub"):
        cmd += ["--dataset-hub", str(bench["dataset_hub"])]
    if bench.get("eval_batch_size") is not None:
        cmd += ["--eval-batch-size", str(bench["eval_batch_size"])]
    if bench.get("generation_config"):
        cmd += ["--generation-config", json.dumps(bench["generation_config"], ensure_ascii=False)]
    args = merged_dataset_args(cfg, names)
    if args:
        cmd += ["--dataset-args", json.dumps(args, ensure_ascii=False)]
    if bench.get("timeout") is not None:
        cmd += ["--timeout", str(bench["timeout"])]
    if limit:
        cmd += ["--limit", str(limit)]
    if bench.get("no_timestamp", True):
        cmd.append("--no-timestamp")
    cmd += [str(item) for item in (bench.get("extra_args") or [])]
    return cmd


def build_harness_commands(cfg: dict, names: tuple[str, ...]) -> list[list[str]]:
    """Agent 类基准交给 harness（deepseek-harness）跑，端点用同一个；一个数据集一条命令。"""
    bench = harness.harness_cfg(cfg) or {}
    return [
        harness.command(cfg, task=name, extra=list(bench.get("extra_args") or [])) for name in names
    ]


def harness_env_of(cfg: dict) -> dict:
    """Harness 要的环境：dsh home + 我们的端点。"""
    env = dict(
        harness.harness_env(
            cfg,
            base_url=str((cfg.get("endpoint") or {}).get("base_url", "")),
            model=str((cfg.get("endpoint") or {}).get("model", "")),
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
    ap = argparse.ArgumentParser(description="Looma 基准评测（vLLM 端点 + EvalScope + harness）")
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml")
    ap.add_argument("--config", default=None, help="配置文件路径（与 --profile 等价）")
    ap.add_argument("--suite", default="main", choices=["mini", "main", "long", "agent", "all"])
    ap.add_argument("--datasets", nargs="*", default=None, help="显式数据集名（覆盖 --suite）")
    ap.add_argument("--limit", type=int, default=None, help="每个数据集最多评多少条")
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

    names = benchmarks.resolve(args.datasets, None if args.datasets else [args.suite])
    open_names = benchmarks.subset(names, benchmarks.OPEN)
    agent_names = benchmarks.subset(names, benchmarks.AGENT)
    out_dir = run_dir_of(cfg, "stage5_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    serving = cfg.get("serving") or {}
    vllm_cmd = build_vllm_command(serving)
    evalscope_cmd = build_evalscope_command(cfg, open_names, out_dir, args.limit)
    harness_cmds = build_harness_commands(cfg, agent_names)

    print(f"[eval] 档：{profile}  端点：{cfg['endpoint']['base_url']}")
    print(f"[eval] 开放集（EvalScope）：{list(open_names)}")
    if agent_names:
        print(f"[eval] 工具集（harness）：{list(agent_names)}")
    no_hf = benchmarks.without_hf(open_names)
    if no_hf and str((cfg.get("evalscope") or {}).get("dataset_hub", "")).lower() == "huggingface":
        print(
            f"[eval] 提示：{list(no_hf)} 在 HF 上没有可用仓库，会拉不下来；改用 --dataset-hub modelscope 或先预热缓存"
        )
    print("[eval] vLLM     ：\n  " + " ".join(vllm_cmd))
    if open_names:
        print("[eval] EvalScope：\n  " + " ".join(evalscope_cmd))
    for cmd in harness_cmds:
        print("[eval] harness  ：\n  " + " ".join(cmd))
    if args.dry_run:
        return 0

    if not (eval_venv() / "bin" / "evalscope").is_file():
        raise SystemExit(f"[eval] 评测 venv 不完整：{eval_venv()}（先跑 bash setup_env.sh）")

    proc = None
    if serving.get("start", True) and not args.no_serve:
        proc = subprocess.Popen(vllm_cmd, env=common.subprocess_env(strip_proxy=True))
        if not wait_healthy(str(cfg["endpoint"]["base_url"])):
            proc.terminate()
            raise SystemExit(f"[eval] 端点没起来：{cfg['endpoint']['base_url']}")

    result: dict = {"profile": profile, "endpoint": cfg["endpoint"]["base_url"], "benchmarks": {}}
    try:
        if open_names:
            # 保留代理：EvalScope 要连数据集 hub（`strip_proxy` 是给单机引擎初始化用的，这里不能带）。
            rc = subprocess.call(evalscope_cmd, env=common.subprocess_env())
            result["benchmarks"]["evalscope"] = {
                "rc": rc,
                "datasets": list(open_names),
                "work_dir": str(out_dir / "evalscope"),
            }
        if harness_cmds:
            env = common.subprocess_env()
            env.update(harness_env_of(cfg))
            rcs = [subprocess.call(cmd, env=env) for cmd in harness_cmds]
            result["benchmarks"]["harness"] = {"rc": rcs, "datasets": list(agent_names)}
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=60)

    result["reference"] = {
        item.name: item.reference
        for item in benchmarks.BENCHMARKS
        if item.name in names and item.reference is not None
    }
    report = out_dir / "summary.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval] summary 写入 {report}")
    print(f"[eval] EvalScope 报告：{out_dir / 'evalscope'}（reports/ 下按数据集分目录）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
