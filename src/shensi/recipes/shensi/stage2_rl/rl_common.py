import argparse
import json
import sys
from pathlib import Path

from shensi import activate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage0_pretrain"))

import importlib.util  # noqa: E402


def _pretrain_common():
    """按文件路径加载 stage0_pretrain/common.py：本模块名与它不同，但显式点明更稳。"""
    f = Path(__file__).resolve().parents[1] / "stage0_pretrain/common.py"
    spec = importlib.util.spec_from_file_location("shensi_stage0_common", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pretrain_common = _pretrain_common()

# yaml 里的键 → verl CLI 的覆盖键（verl 只认自己的配置路径，这里显式映射，避免猜）
CLI_MAP = {
    "model.path": "actor_rollout_ref.model.path",
    "algorithm.adv_estimator": "algorithm.adv_estimator",
    "algorithm.kl_coef": "algorithm.kl_ctrl.kl_coef",
    "rollout.name": "actor_rollout_ref.rollout.name",
    "rollout.n": "actor_rollout_ref.rollout.n",
    "rollout.temperature": "actor_rollout_ref.rollout.temperature",
    "rollout.top_p": "actor_rollout_ref.rollout.top_p",
    "rollout.max_model_len": "actor_rollout_ref.rollout.max_model_len",
    "rollout.gpu_memory_utilization": "actor_rollout_ref.rollout.gpu_memory_utilization",
    "rollout.log_prob_micro_batch_size_per_gpu": "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "rollout.multi_turn.enable": "actor_rollout_ref.rollout.multi_turn.enable",
    "rollout.multi_turn.format": "actor_rollout_ref.rollout.multi_turn.format",
    "rollout.multi_turn.max_assistant_turns": "actor_rollout_ref.rollout.multi_turn.max_assistant_turns",
    "rollout.multi_turn.max_user_turns": "actor_rollout_ref.rollout.multi_turn.max_user_turns",
    "rollout.multi_turn.max_parallel_calls": "actor_rollout_ref.rollout.multi_turn.max_parallel_calls",
    "rollout.multi_turn.max_tool_response_length": "actor_rollout_ref.rollout.multi_turn.max_tool_response_length",
    "rollout.multi_turn.tool_response_truncate_side": "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side",
    "rollout.multi_turn.tool_config_path": "actor_rollout_ref.rollout.multi_turn.tool_config_path",
    "rollout.multi_turn.interaction_config_path": "actor_rollout_ref.rollout.multi_turn.interaction_config_path",
    "rollout.agent.default_agent_loop": "actor_rollout_ref.rollout.agent.default_agent_loop",
    "rollout.agent.num_workers": "actor_rollout_ref.rollout.agent.num_workers",
    "rollout.agent.agent_loop_config_path": "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    "actor.optim.lr": "actor_rollout_ref.actor.optim.lr",
    "actor.optim.lr_warmup_steps": "actor_rollout_ref.actor.optim.lr_warmup_steps",
    "actor.optim.weight_decay": "actor_rollout_ref.actor.optim.weight_decay",
    "actor.optim.betas": "actor_rollout_ref.actor.optim.betas",
    "actor.optim.clip_grad": "actor_rollout_ref.actor.optim.clip_grad",
    "actor.ppo_mini_batch_size": "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor.ppo_micro_batch_size_per_gpu": "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor.use_kl_loss": "actor_rollout_ref.actor.use_kl_loss",
    "model.enable_gradient_checkpointing": "actor_rollout_ref.model.enable_gradient_checkpointing",
    "actor.megatron.tensor_model_parallel_size": "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
    "actor.megatron.pipeline_model_parallel_size": "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size",
    "actor.megatron.use_mbridge": "actor_rollout_ref.actor.megatron.use_mbridge",
    "actor.megatron.use_distributed_optimizer": "actor_rollout_ref.actor.megatron.use_distributed_optimizer",
    "actor.megatron.use_remove_padding": "actor_rollout_ref.actor.megatron.use_remove_padding",
    "ref.megatron.tensor_model_parallel_size": "actor_rollout_ref.ref.megatron.tensor_model_parallel_size",
    "ref.megatron.pipeline_model_parallel_size": "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size",
    "ref.megatron.use_mbridge": "actor_rollout_ref.ref.megatron.use_mbridge",
    "ref.megatron.use_remove_padding": "actor_rollout_ref.ref.megatron.use_remove_padding",
    "ref.log_prob_micro_batch_size_per_gpu": "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    "rollout.tensor_model_parallel_size": "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "rollout.max_num_seqs": "actor_rollout_ref.rollout.max_num_seqs",
    "rollout.max_num_batched_tokens": "actor_rollout_ref.rollout.max_num_batched_tokens",
    "rollout.enforce_eager": "actor_rollout_ref.rollout.enforce_eager",
    "reward.reward_manager.name": "reward.reward_manager.name",
    "data.filter_overlong_prompts": "data.filter_overlong_prompts",
    "data.truncation": "data.truncation",
    "data.dataloader_num_workers": "data.dataloader_num_workers",
    "actor.clip_ratio_low": "actor_rollout_ref.actor.clip_ratio_low",
    "actor.clip_ratio_high": "actor_rollout_ref.actor.clip_ratio_high",
    "data.max_prompt_length": "data.max_prompt_length",
    "data.max_response_length": "data.max_response_length",
    "data.train_batch_size": "data.train_batch_size",
    "data.val_batch_size": "data.val_batch_size",
    "data.need_tools_kwargs": "data.need_tools_kwargs",
    "trainer.total_epochs": "trainer.total_epochs",
    "trainer.val_before_train": "trainer.val_before_train",
    "trainer.test_freq": "trainer.test_freq",
    "trainer.n_gpus_per_node": "trainer.n_gpus_per_node",
    "trainer.nnodes": "trainer.nnodes",
    "trainer.logger": "trainer.logger",
    "trainer.project_name": "trainer.project_name",
    "trainer.experiment_name": "trainer.experiment_name",
}

PROMPT_KEYS = ("prompt", "question", "problem", "instruction", "query")
ANSWER_KEYS = ("answer", "ground_truth", "target", "solution", "label", "response", "completion")
# 奖励规格类字段：没有直给答案时，把这些整段塞进 ground_truth，交给奖励函数解释
SPEC_KEYS = ("verifier", "rubric", "tests", "test_cases", "expected", "expected_markers", "pass_rate")


def _load_with_base(path: Path, seen: tuple = ()) -> dict:
    """读训练配置；带 `base:`（文件或目录）就以基座为底叠覆盖，支持多级继承。"""
    path = path.resolve()
    if path in seen:
        raise SystemExit(f"[rl] 配置的 base 链出现环：{path}")
    raw = dict(pretrain_common.load_yaml(path))
    base = raw.pop("base", None)
    if not base:
        return raw
    target = (path.parent / base).resolve()
    if target.is_dir():
        target = target / "default.yaml"
    return pretrain_common._deep_merge(_load_with_base(target, seen + (path,)), raw)


def flatten(prefix: str, obj, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out.append((prefix, obj))


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = pretrain_common._coerce(val)
    return cfg


# 这三个路径是 verl 按运行时 cwd 解析的：相对路径统一固定成 stage 目录下的绝对路径
PATH_KEYS = (
    "rollout.multi_turn.tool_config_path",
    "rollout.multi_turn.interaction_config_path",
    "rollout.agent.agent_loop_config_path",
)


def resolve_paths(cfg: dict, here: Path) -> dict:
    for key in PATH_KEYS:
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
        if isinstance(node, dict) and isinstance(node.get(parts[-1]), str) and node[parts[-1]]:
            if not Path(node[parts[-1]]).is_absolute():
                node[parts[-1]] = str((here / node[parts[-1]]).resolve())
    return cfg


def build_command(cfg: dict, stage: str, data_dir: Path, reward: Path) -> list[str]:
    """yaml 配置 → `python -m verl.trainer.main_ppo --config-name ... <覆盖项>`。"""
    pairs: list = []
    flatten("", cfg, pairs)
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "ppo_megatron_trainer",
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/val.parquet",
        f"reward.custom_reward_function.path={reward}",
        "reward.custom_reward_function.name=compute_score",
    ]
    unknown = []
    for key, val in pairs:
        if key.startswith("rollout.engine_kwargs."):  # verl 的 engine kwargs 不在配置结构里，得用 + 加
            extra = key[len("rollout.engine_kwargs.") :]
            cmd.append(f"+actor_rollout_ref.rollout.engine_kwargs.{extra}={val}")
            continue
        cli = CLI_MAP.get(key)
        if cli is None:
            unknown.append(key)
            continue
        cmd.append(f"{cli}={val}")
    if unknown:
        raise SystemExit(f"[{stage}] config 里有没映射到 verl CLI 的键：{unknown}")
    return cmd


def launch(
    stage: str,
    argv: list[str] | None = None,
    here: Path | None = None,
    reward: Path | None = None,
) -> int:
    """RL 各子 stage 的 train.py 共用入口：解析参数 → 拼命令 → 打印/执行。

    `here` / `reward` 可换：世界模型那一段（stage2_rl/stage4_world_model）也走 verl，但配置与奖励是自己的。
    """
    here = Path(here) if here else Path(__file__).resolve().parent / stage
    reward = Path(reward) if reward else Path(__file__).resolve().parent / "reward.py"
    ap = argparse.ArgumentParser(description=f"Shensi {stage} 启动器（verl GRPO + Megatron actor）")
    ap.add_argument("--config", default=None, help="默认 config/default.yaml")
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml")
    ap.add_argument("--data-dir", default=None, help="data_prep 产物目录（含 train.parquet / val.parquet）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    ap.add_argument("--set", dest="override", action="append", default=[], help="点号覆盖，可多次")
    args = ap.parse_args(argv)

    paths = pretrain_common.env_paths()
    cfg = pretrain_common.resolve_cfg(
        _load_with_base(Path(args.config) if args.config else here / f"config/{args.profile}.yaml")
    )
    apply_overrides(cfg, args.override)
    resolve_paths(cfg, here)
    data_dir = Path(args.data_dir or paths["data"] / stage)
    cmd = build_command(cfg, stage, data_dir, reward)
    cmd.append(f"hydra.run.dir={paths['runs'] / stage}")
    print(f"[{stage}] 命令：\n  " + " \\\n    ".join(cmd))
    if args.dry_run:
        return 0
    import os
    import subprocess as sp

    env = dict(os.environ)
    # 与工作区里验证过的调用同形：verl 与 mcore 都从工作区 fork 树取（装的是 @main 那版，缺 mbridge）
    env["PYTHONPATH"] = str(paths["mcore"])  # verl 走安装版（官方），mcore 用工作区树
    # ray/vLLM 在带代理的单机环境下会在引擎初始化阶段失败（本机踩过），子进程一律去掉代理
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # 上游在 use_distributed_optimizer=False 时没有 flat param buffer，load_megatron_model_to_gpu 漏判空；
    # 补丁放在配方里，由 .pth 让每个子进程启动时自动应用
    import tempfile

    import verl_patch

    verl_patch.inject(env, Path(tempfile.gettempdir()) / "shensi_verl_patch")
    return sp.call(cmd, env=activate(env))


# ---------------- 语料：各种形状 → verl 的 RL schema ----------------


def to_rl_row(row: dict, source: str) -> dict | None:
    rcp = row.get("responses_create_params")
    if isinstance(rcp, str):
        try:
            rcp = json.loads(rcp)
        except Exception:  # noqa: BLE001
            rcp = None
    if isinstance(rcp, dict) and isinstance(rcp.get("input"), list):
        row = {**row, "prompt": rcp["input"]}
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        prompt = [{"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")} for m in prompt]
    else:
        msg = row.get("messages") or row.get("conversations")
        if isinstance(msg, list) and msg and isinstance(msg[0], dict):
            prompt = [
                {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
                for m in msg
                if str(m.get("role")) != "assistant"
            ]
        else:
            text = next((row[k] for k in PROMPT_KEYS if isinstance(row.get(k), str)), None)
            if not text:
                return None
            prompt = [{"role": "user", "content": text}]
    if not prompt:
        return None

    answer = next((row[k] for k in ANSWER_KEYS if row.get(k) is not None), None)
    spec = next((row[k] for k in SPEC_KEYS if row.get(k) is not None), None)
    if answer is None and spec is None:
        return None
    if answer is None:
        answer = json.dumps(spec, ensure_ascii=False) if not isinstance(spec, str) else spec
    verifier = row.get("verifier")
    agent_ref = row.get("agent_ref")
    style = (
        (verifier.get("type") if isinstance(verifier, dict) else None)
        or row.get("verifier_type")
        or row.get("reward_type")
        or "rule"
    )
    return {
        "prompt": prompt,
        "data_source": source,
        "agent": agent_ref if isinstance(agent_ref, (dict, str)) else None,
        "verifier": verifier if isinstance(verifier, (dict, str)) else None,
        "reward_model": {"style": str(style), "ground_truth": str(answer)},
        "extra_info": {"source": source, "split": "train"},
    }


def iter_rows(files: list[Path], limit: int | None):
    n = 0
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield row
                    n += 1
                    if limit and n >= limit:
                        return
        else:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
                        n += 1
                        if limit and n >= limit:
                            return


def files_of(root: Path, d: dict) -> list[Path]:
    base = root / d["name"]
    if not base.is_dir():
        return []
    out = []
    for f in sorted(base.glob("**/*")):
        if f.is_dir() or f.suffix not in (".parquet", ".jsonl", ".json"):
            continue
        rel = str(f.relative_to(base))
        if d.get("config") and d["config"] not in rel:
            continue
        out.append(f)
    return out


def prepare(stage: str, argv: list[str] | None = None) -> int:
    """三个 RL 子 stage 的 data_prep.py 共用入口。"""
    here = Path(__file__).resolve().parent / stage
    ap = argparse.ArgumentParser(description=f"Shensi {stage} 语料准备")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--blend", default=None, help="换一份配比 json（默认 config/data_prep/data_blend_raw.json）")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--root", default=None, help="RL 语料根，默认 $SHENSI_FS/datasets/llm/post-training")
    ap.add_argument("--out", default=None, help=f"产物目录，默认 $SHENSI_FS/shensi/data/{stage}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--max-chars", type=int, default=None, help="丢掉 prompt 超过这个字符数的样本（调试档用）")
    args = ap.parse_args(argv)

    paths = pretrain_common.env_paths()
    root = Path(args.root or paths["post"])
    out = Path(args.out or paths["data"] / stage)
    spec = pretrain_common.load_blend_spec(
        Path(args.blend) if args.blend else here / "config/data_prep/data_blend_raw.json"
    )
    datasets = [d for d in spec["datasets"] if not args.only or args.only in d["name"]]

    if args.discover:
        print(f"[{stage}] 语料根：{root}")
        for d in datasets:
            files = files_of(root, d)
            print(f"  {'✅' if files else '❌'} {d['name']:<48} 文件 {len(files):<4} weight={d.get('weight')}")
        return 0
    if not args.prepare:
        ap.error("至少给一个：--discover / --prepare")

    rows: list[dict] = []
    for d in datasets:
        files = files_of(root, d)
        if not files:
            msg = f"{d['name']}: 在 {root} 下没找到文件"
            if args.skip_missing:
                print(f"[{stage}] 跳过（--skip-missing）：{msg}")
                continue
            raise SystemExit(f"{msg}，先跑 --discover 看看")
        n = 0
        for row in iter_rows(files, args.limit):
            rl = to_rl_row(row, d["name"])
            if rl:
                rows.append(rl)
                n += 1
        print(f"[{stage}] {d['name']}: {n} 条")
    if not rows:
        raise SystemExit(f"[{stage}] 一条都没解析出来：看看 --discover 打出来的字段名")

    if args.max_chars:
        kept = [r for r in rows if len(str(r["prompt"])) <= args.max_chars]
        if len(kept) < 2:  # 至少留两条，val 拆分才有东西
            kept = sorted(rows, key=lambda r: len(str(r["prompt"])))[:2]
        print(f"[{stage}] --max-chars {args.max_chars}：{len(rows)} → {len(kept)} 条")
        rows = kept

    import pyarrow as pa
    import pyarrow.parquet as pq

    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i
    n_val = max(1, int(len(rows) * args.val_ratio))
    table = pa.Table.from_pylist(rows)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(n_val, len(rows) - n_val), out / "train.parquet")
    pq.write_table(table.slice(0, n_val), out / "val.parquet")
    print(f"[{stage}] 写出 {out}/train.parquet（{len(rows) - n_val} 行）+ val.parquet（{n_val} 行）")
    return 0
