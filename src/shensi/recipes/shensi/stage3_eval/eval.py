#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent  # stage3_eval/

from shensi import runtime  # noqa: E402
from shensi.recipes.shensi.common import common, harness# noqa: E402

# ---------------- vLLM 服务 ----------------


def build_vllm_command(serving: dict) -> list[str]:
    """按 serving 段拼 `vllm serve`（Shensi 侧靠 vllm 里的 shensi 模型实现）。"""
    cmd = ["vllm", "serve", str(serving["model_path"])]
    flags = [
        ("served_model_name", "--served-model-name", False),
        ("host", "--host", False),
        ("port", "--port", False),
        ("tensor_parallel_size", "--tensor-parallel-size", False),
        ("data_parallel_size", "--data-parallel-size", False),
        ("gpu_memory_utilization", "--gpu-memory-utilization", False),
        ("kv_cache_dtype", "--kv-cache-dtype", False),
        ("max_model_len", "--max-model-len", False),
        ("reasoning_parser", "--reasoning-parser", False),
        ("tool_call_parser", "--tool-call-parser", False),
        ("speculative_config", "--speculative-config", False),
    ]
    for key, flag, _ in flags:
        val = serving.get(key)
        if val not in (None, "", "auto" if key == "kv_cache_dtype" else None):
            cmd += [flag, str(val)]
    if serving.get("enable_expert_parallel"):
        cmd.append("--enable-expert-parallel")
    cmd += [str(x) for x in (serving.get("extra_args") or [])]
    return cmd


def ensure_model_path(serving: dict) -> None:
    """`serving.model_path` 得是真实存在的目录：占位路径要当场报错，别等 vLLM 起不来。"""
    path = Path(str(serving["model_path"]))
    if path.is_dir():
        return
    raise SystemExit(
        f"[eval] serving.model_path 不存在：{path}\n"
        "  本机跑：--model-path <本机 HF 目录>（如 $SHENSI_FS/shensi/models/sft-hf），"
        "或用 --profile tiny_local（本机 tiny 产物）"
    )


def cap_max_model_len(serving: dict) -> None:
    """Profile 里的 max_model_len 大于模型的 max_position_embeddings 时压回模型上限，
    免得 vLLM 直接拒绝启动（tiny ckpt 常见）。"""
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


def wait_healthy(base_url: str, timeout: int = 1200) -> bool:
    """轮询 /v1/models 直到服务起来（OpenAI 兼容端点的通用探活）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=10):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return False


# ---------------- NeMo Gym（instruct / agentic 基准） ----------------


def agent_overrides(cfg: dict) -> list[str]:
    """Agent 段 → Gym 的 HarnessAgent 覆盖项：外部 harness（默认 dsh）在沙箱里解 Gym 的任务。

    接线统一在 `harness.py`（stage2_rl 的 agentic 段用同一份：同一个 harness、同一个端点约定），
    这里只把 Gym 自己的字段补上——字段名对应 Gym 的
    responses_api_agents/harness_agent/app.py::HarnessAgentConfig。
    """
    ag = cfg.get("agent") or {}
    if not (cfg.get("harness") or {}).get("name"):
        return []
    root = "policy_model.responses_api_agents.harness_agent"
    base_url = f"{cfg['endpoint']['base_url']}/v1"
    out = harness.gym_agent_overrides(
        cfg, base_url=base_url, model=cfg.get("serving", {}).get("served_model_name")
    )
    out.append(f"++{root}.sandbox_model_base_url={base_url}")
    if ag.get("command"):
        out.append(f"++{root}.agent_kwargs=" + json.dumps(ag["command"], ensure_ascii=False))
    if ag.get("sandbox_image"):
        out.append(f"++{root}.sandbox_image={ag['sandbox_image']}")
    if ag.get("setup_commands"):
        out.append(
            f"++{root}.setup_commands=" + json.dumps(list(ag["setup_commands"]), ensure_ascii=False)
        )
    return out


def gym_command(
    gym: dict, stage: str, bench: dict, base_url: str, agent_over: list[str] | None = None
) -> list[str]:
    name = bench["name"] if isinstance(bench, dict) else str(bench)
    over = list(gym.get("common_overrides") or []) + list(
        (bench or {}).get("overrides", []) if isinstance(bench, dict) else []
    )
    over += list(agent_over or []) + list(gym.get("extra_overrides") or [])
    cmd = list(str(gym.get("command", "uv run gym")).split()) + ["eval", stage, "--benchmark", name]
    if stage == "run":
        cmd += ["--model-type", str(gym.get("model_type", "vllm_model")), "--model-url", base_url]
        if gym.get("concurrency"):
            cmd += ["--concurrency", str(gym["concurrency"])]
        if gym.get("limit"):
            cmd += ["--limit", str(gym["limit"])]
        if gym.get("num_repeats"):
            cmd += ["--num-repeats", str(gym["num_repeats"])]
    cmd += over
    return cmd


def run_gym(cfg: dict, out_dir: Path, dry_run: bool) -> dict:
    gym = cfg["gym"]
    ep = cfg["endpoint"]
    summary: dict = {}
    agent_over = agent_overrides(cfg)
    for bench in gym["benchmarks"]:
        for stage in ("prepare", "run"):
            cmd = gym_command(gym, stage, bench, ep["base_url"], agent_over)
            print(f"[eval][gym] {' '.join(cmd)}")
            if dry_run:
                continue
            env = dict(os.environ, GYM_API_KEY=str(gym.get("api_key", "dummy")))
            rc = subprocess.call(cmd, cwd=str(gym["workdir"]), env=env)
            if rc != 0:
                raise SystemExit(f"[eval] gym {stage} 失败（{bench}）：退出码 {rc}")
    return summary


def collect_summary(res_dir: Path) -> dict:
    """把每个 benchmark 的 *_aggregate_metrics.json 收成一份 summary.json（Nemotron 同口径）。"""
    out = {}
    for p in sorted(res_dir.rglob("*_aggregate_metrics.json")):
        try:
            out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            out[p.parent.name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ---------------- local 套件（不依赖 Gym：能力集 + 长上下文） ----------------


def ask(
    base_url: str, model: str, prompt: str | list[dict], max_tokens: int, temperature: float,
    api_key: str = "",
) -> str:
    """打一次 chat 端点；`prompt` 给字符串就是单轮，给 list[dict] 就整段消息发过去（MRCR 用）。"""
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions", data=body, headers=hdr
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _sft_prep():
    import importlib.util

    f = HERE.parent / "stage1_sft/data_prep.py"
    spec = importlib.util.spec_from_file_location("shensi_sft_prep", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_local_prompts(cfg: dict) -> list[dict]:
    """能力集 + 长上下文集（埋一个只出现一次的编号再问，Nemotron/V4 的长上下文口径）。"""
    loc = cfg["local"]
    prep = _sft_prep()
    rows: list[dict] = []
    for name in loc["capability_sets"]:
        files = [
            f
            for f in sorted(Path(loc["root"], name).glob("**/*"))
            if f.suffix in (".parquet", ".jsonl", ".json")
        ]
        k = 0
        for row in prep.iter_rows(files, None):
            msgs = prep.to_messages(row)
            if not msgs:
                continue
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ans = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
            if user and ans:
                rows.append({"capability": name, "prompt": user, "ground_truth": ans})
                k += 1
            if k >= loc["per_set"]:
                break
        print(f"[eval][local] {name}: {k} 条")
    # 长文套件：优先用 stage3_longctx 构建的 MRCR 类多针集（`build_longctx.py --step mrcr` 产出的
    # mrcr_eval.jsonl）——它和训练段用的是同一批针，判分要求按出现顺序全对；没有就退回「拼长文 +
    # 埋一个针」的本地抽样（口径弱得多，只够冒烟）。
    src = Path(loc["longctx_root"])
    mrcr = src / "mrcr_eval.jsonl" if src.is_dir() else None
    if mrcr is not None and mrcr.is_file():
        picked = 0
        for row in prep.iter_rows([mrcr], None):
            if not isinstance(row.get("prompt"), str) or not row.get("ground_truth"):
                continue
            rows.append(
                {
                    "capability": str(row.get("capability") or "mrcr"),
                    "prompt": row["prompt"],
                    "ground_truth": row["ground_truth"],
                }
            )
            picked += 1
            if picked >= loc["longctx_num"]:
                break
        print(f"[eval][local] MRCR 类多针检索：{picked} 条（来自 {mrcr}）")
        return rows
    files = (
        [f for f in sorted(src.glob("**/*")) if f.suffix in (".parquet", ".jsonl", ".json")]
        if src.is_dir()
        else []
    )
    if files:
        target, buf, made = int(loc["longctx_chars"]), [], 0
        for row in prep.iter_rows(files, None):
            text = next(
                (
                    row[k]
                    for k in ("text", "content", "raw_content", "document")
                    if isinstance(row.get(k), str) and len(row[k]) > 2000
                ),
                None,
            )
            if text is None:
                continue
            buf.append(text)
            joined = "\n\n".join(buf)
            if len(joined) < target:
                continue
            fact = f"编号 {made + 1}-{abs(hash(joined[:64])) % 10**6}"
            body = f"{joined[: target // 2]}\n\n【记录】本次发布的内部编号是 {fact}，只在本段出现一次。\n\n{joined[-target // 2 :]}"
            rows.append(
                {
                    "capability": f"longctx-{target // 1000}k",
                    "prompt": f"{body}\n\n问题：上面那段材料里提到的内部编号是什么？只回答编号本身。",
                    "ground_truth": fact,
                }
            )
            buf, made = [], made + 1
            if made >= loc["longctx_num"]:
                break
        print(f"[eval][local] longctx-{target // 1000}k：{made} 条（本地抽样，非 MRCR）")
    return rows


def run_local(cfg: dict, out_dir: Path, limit: int | None, dry_run: bool) -> dict:
    rows = build_local_prompts(cfg)
    if limit:
        rows = rows[:limit]
    if dry_run:
        print(f"[eval][local] 将跑 {len(rows)} 条（dry-run 不发请求）")
        return {}
    if not rows:
        raise SystemExit(
            "[eval][local] 一条 prompt 都没造出来：检查 local.root / capability_sets 指向的目录"
        )
    sys.path.insert(0, str(HERE.parent / "stage2_rl"))
    from reward import compute_score  # noqa: E402

    ep = cfg["endpoint"]
    api_key = os.environ.get(ep.get("api_key_env") or "", "")
    per: dict[str, list] = {}
    detail = []
    for i, r in enumerate(rows):
        try:
            ans = ask(
                ep["base_url"],
                ep["model"],
                r["prompt"],
                int(ep.get("max_tokens", 1024)),
                float(ep.get("temperature", 0.0)),
                api_key,
            )
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"[eval] 第 {i} 条请求失败（{ep['base_url']} 起了吗？）：{exc}"
            ) from None
        score = float(compute_score(r["capability"], ans, r["ground_truth"]))
        per.setdefault(r["capability"], []).append(score)
        detail.append({"capability": r["capability"], "score": score, "answer": ans[:4000]})
        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(rows)}", flush=True)
    card = {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in sorted(per.items())}
    overall = sum(x["score"] for x in detail) / max(len(detail), 1)
    print("[eval][local] 分数：")
    for k, v in card.items():
        print(f"  {k:<46} n={v['n']:<4} 均分 {v['mean']:.3f}")
    print(f"  {'总体':<46} n={len(detail):<4} 均分 {overall:.3f}")
    return {"card": card, "overall": overall, "detail": detail}


# ---------------- MRCR（官方开放集 openai/mrcr） ----------------


def run_mrcr(cfg: dict, out_dir: Path, limit: int | None, dry_run: bool) -> dict:
    """官方 MRCR：从 `openai/mrcr` 取样本，按官方判分（前缀哈希 + SequenceMatcher 比值）。

    配置段 `mrcr:`（见 config/default.yaml）：`needles` / `bins` / `per_bin` / `tokenizer` /
    `min_tokens` / `max_tokens`（题面 token 范围）/ `max_answer_tokens` / `temperature`。
    """
    sys.path.insert(0, str(HERE))
    import mrcr_official

    mc = cfg.get("mrcr") or {}
    needles = tuple(int(n) for n in (mc.get("needles") or [2]))
    rows = mrcr_official.load_samples(
        needles=needles,
        per_bin=int(mc.get("per_bin", 1)),
        limit=limit,
        min_tokens=mc.get("min_tokens"),
        max_tokens=mc.get("max_tokens"),
        tokenizer=mc.get("tokenizer"),
    )
    if not rows:
        print("[eval][mrcr] 没取到样本（针数/桶/token 范围不匹配？）")
        return {"card": {}, "overall": 0.0, "detail": []}
    if dry_run:
        for r in rows:
            print(
                f"[eval][mrcr] 将跑：针={r['n_needles']} tokens≈{r['n_tokens']} "
                f"桶={r['bin']} 消息数={len(r['messages'])}"
            )
        print(f"[eval][mrcr] 共 {len(rows)} 条（dry-run 不发请求）")
        return {}

    ep = cfg["endpoint"]
    api_key = os.environ.get(ep.get("api_key_env") or "", "")
    per: dict[str, list] = {}
    detail = []
    for i, r in enumerate(rows):
        try:
            ans = ask(
                ep["base_url"],
                ep["model"],
                r["messages"],
                int(mc.get("max_answer_tokens", 4096)),
                float(mc.get("temperature", 0.0)),
                api_key,
            )
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"[eval][mrcr] 第 {i} 条请求失败（{ep['base_url']} 起了吗？）：{exc}"
            ) from None
        score = mrcr_official.grade(ans, r["answer"], r["prefix"])
        key = f"mrcr-{r['n_needles']}needle-{r['bin']}"
        per.setdefault(key, []).append(score)
        detail.append(
            {
                "key": key,
                "n_needles": r["n_needles"],
                "bin": r["bin"],
                "n_tokens": r["n_tokens"],
                "score": score,
                "answer": ans[:4000],
            }
        )
        print(f"  [{i + 1}/{len(rows)}] {key} → {score:.3f}", flush=True)
    card = {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in sorted(per.items())}
    overall = sum(x["score"] for x in detail) / max(len(detail), 1)
    print("[eval][mrcr] 分数（官方判分口径）：")
    for k, v in card.items():
        print(f"  {k:<40} n={v['n']:<4} 均分 {v['mean']:.3f}")
    print(f"  {'总体':<40} n={len(detail):<4} 均分 {overall:.3f}")
    return {"card": card, "overall": overall, "detail": detail}


# ---------------- 入口 ----------------


def run_harness(cfg: dict, out_dir: Path, dry_run: bool) -> dict:
    """不经 Gym，直接让 harness（默认 dsh）跑基准——端点仍是本机的 `vllm serve`。

    具体行为由 `$DSH_HOME` 下的 profile 决定（`harness.py` 只负责把端点、模型名、
    DSH_HOME 接上）；一个 benchmark 起一次命令，stdout 收进 summary。
    """
    ben = cfg.get("bench") or {}
    names = [b["name"] if isinstance(b, dict) else str(b) for b in (ben.get("benchmarks") or [])]
    env = harness.harness_env(
        cfg,
        base_url=f"{cfg['endpoint']['base_url']}/v1",
        model=cfg.get("serving", {}).get("served_model_name"),
    )
    out: dict = {"harness": harness.harness_name(cfg), "env": sorted(env), "benchmarks": {}}
    for name in names:
        cmd = harness.command(cfg, task=name, extra=list(ben.get("extra_args") or []))
        if dry_run:
            print("[eval][harness] " + " ".join(cmd))
            out["benchmarks"][name] = None
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
        print(f"[eval][harness] {name}: rc={proc.returncode}")
        out["benchmarks"][name] = {
            "rc": proc.returncode,
            "tail": (proc.stdout or proc.stderr).strip()[-2000:],
        }
        (out_dir / f"harness_{name}.log").write_text(
            (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Shensi stage3_eval：vLLM 服务 + harness（默认 dsh；Gym 是它的宿主之一）+ local 套件"
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml（tiny = 5 条冒烟）")
    ap.add_argument("--suite", default=None, choices=("gym", "local", "harness", "mrcr", "all"))
    ap.add_argument("--base-url", default=None, help="覆盖 endpoint.base_url")
    ap.add_argument("--model", default=None, help="覆盖 endpoint.model")
    ap.add_argument("--model-path", default=None, help="覆盖 serving.model_path（起服务用）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="结果目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印 vLLM 与 gym 的命令")
    ap.add_argument("--no-serve", action="store_true", help="端点已经起好了，本脚本不起 vllm")
    args = ap.parse_args()
    runtime.setup()

    paths = common.env_paths()
    cfg = common.resolve_cfg(
        common.load_yaml(Path(args.config) if args.config else HERE / f"config/{args.profile}.yaml")
    )
    if args.base_url:
        cfg["endpoint"]["base_url"] = args.base_url
    if args.model:
        cfg["endpoint"]["model"] = args.model
    if args.model_path:
        cfg["serving"]["model_path"] = args.model_path
    if args.limit is None and cfg.get("limit"):
        args.limit = int(cfg["limit"])
    suite = args.suite or cfg.get("suite", "all")
    out_dir = Path(args.out or cfg.get("output_dir") or paths["runs"] / "stage3_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_cmd = build_vllm_command(cfg["serving"])
    print("[eval] vLLM：\n  " + " \\\n    ".join(vllm_cmd))
    if args.dry_run:
        for name, fn in (
            ("gym", lambda: run_gym(cfg, out_dir, True)),
            ("harness", lambda: run_harness(cfg, out_dir, True)),
            ("local", lambda: run_local(cfg, out_dir, args.limit, True)),
        ):
            if suite in (name, "all"):
                fn()
        if suite == "mrcr" or (suite == "all" and (cfg.get("mrcr") or {}).get("enabled")):
            run_mrcr(cfg, out_dir, args.limit, True)
        return 0

    proc = None
    if cfg["serving"].get("start", True) and not args.no_serve:
        ensure_model_path(cfg["serving"])
        cap_max_model_len(cfg["serving"])
        vllm_cmd = build_vllm_command(cfg["serving"])
        # vLLM 要的环境（PATH 放本 venv 让 flashinfer 能 JIT、CUDA_HOME、去代理）见 common.subprocess_env
        proc = subprocess.Popen(vllm_cmd, env=common.subprocess_env(strip_proxy=True))
        if not wait_healthy(cfg["endpoint"]["base_url"]):
            raise SystemExit(f"[eval] 端点没起来：{cfg['endpoint']['base_url']}")

    result: dict = {}
    try:
        if suite in ("local", "all"):
            result["local"] = run_local(cfg, out_dir, args.limit, False)
        # 官方 MRCR 默认只在显式点名时跑（`--suite mrcr`，或档里把 `mrcr.enabled` 打开）：
        # 它要下开放集、并且要长上下文模型才有意义
        if suite == "mrcr" or (suite == "all" and (cfg.get("mrcr") or {}).get("enabled")):
            result["mrcr"] = run_mrcr(cfg, out_dir, args.limit, False)
        if suite in ("gym", "all"):
            run_gym(cfg, out_dir, False)
            result["gym"] = collect_summary(out_dir)
        if suite in ("harness", "all"):
            result["harness"] = run_harness(cfg, out_dir, False)
    finally:
        if proc is not None:
            proc.terminate()

    p = out_dir / "summary.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval] summary 写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
