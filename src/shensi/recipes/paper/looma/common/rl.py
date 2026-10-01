"""RL 阶段的启动：配置到 verl 命令行、外部 harness 接线、前台运行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.shensi.common import common as base
from shensi.recipes.shensi.common import rl as shensi_rl

from .paths import env_paths
from .runner import spawn

BRIDGE_MODULE = "shensi.recipes.paper.looma.stage2_rl.looma_bridge"
_AGENT_MODEL_TYPES = ("looma",)
#: 配置里这两段的子键要整块传给 verl（见 `build_verl_command`）。
_TRANSFORMER_CFG = ".megatron.override_transformer_config."

#: verl 命令行里本配方需要的、shensi 映射表之外的键。
_VERL_CLI_EXTRA: dict[str, str] = {
    # 本配方的检查点自带建模代码（`auto_map` + 两个文件），transformers 侧必须放行
    "model.trust_remote_code": "actor_rollout_ref.model.trust_remote_code",
    "algorithm.norm_adv_by_std_in_grpo": "algorithm.norm_adv_by_std_in_grpo",
    "algorithm.filter_groups.enable": "algorithm.filter_groups.enable",
    "algorithm.filter_groups.metric": "algorithm.filter_groups.metric",
    "algorithm.filter_groups.max_num_gen_batches": "algorithm.filter_groups.max_num_gen_batches",
    "algorithm.use_kl_in_reward": "algorithm.use_kl_in_reward",
    "actor.clip_ratio_low": "actor_rollout_ref.actor.clip_ratio_low",
    "actor.clip_ratio_high": "actor_rollout_ref.actor.clip_ratio_high",
    "actor.clip_ratio_c": "actor_rollout_ref.actor.clip_ratio_c",
    "actor.loss_agg_mode": "actor_rollout_ref.actor.loss_agg_mode",
    "actor.strategy": "actor_rollout_ref.actor.strategy",
    "actor.entropy_coeff": "actor_rollout_ref.actor.entropy_coeff",
    "actor.use_dynamic_bsz": "actor_rollout_ref.actor.use_dynamic_bsz",
    "critic.optim.lr": "critic.optim.lr",
    "critic.model.path": "critic.model.path",
    "critic.ppo_micro_batch_size_per_gpu": "critic.ppo_micro_batch_size_per_gpu",
    "critic.megatron.tensor_model_parallel_size": "critic.megatron.tensor_model_parallel_size",
    "critic.megatron.pipeline_model_parallel_size": "critic.megatron.pipeline_model_parallel_size",
    "critic.megatron.use_mbridge": "critic.megatron.use_mbridge",
    "rollout.agent.default_agent_loop": "actor_rollout_ref.rollout.agent.default_agent_loop",
    "rollout.multi_turn.tool_config_path": "actor_rollout_ref.rollout.multi_turn.tool_config_path",
}


def model_type_of(path: str | Path | None) -> str:
    """读检查点目录（或 config.json）里的 ``model_type``，读不到返回空串。"""
    if not path:
        return ""
    target = Path(path)
    target = target / "config.json" if target.is_dir() else target
    try:
        return str(json.loads(target.read_text(encoding="utf-8")).get("model_type") or "")
    except Exception:
        return ""


def agent_harness(model_path: str | Path | None) -> dict:
    """模型属于多轮工具族时返回 harness 段，否则返回空字典。"""
    if not model_type_of(model_path).startswith(_AGENT_MODEL_TYPES):
        return {}
    from shensi.recipes.shensi.common import harness

    return {
        "name": harness.HARNESS_DEFAULT,
        "home": harness.dsh_home({}),
        "command": harness.DEFAULT_COMMAND,
        "install": harness.SDK_PACKAGE,
    }


def agent_overrides(model_path: str | Path | None, *, tool_config: str | None = None) -> list[str]:
    """接上 harness 时需要额外打开的 rollout 旋钮。"""
    if not agent_harness(model_path):
        return []
    overrides = ["rollout.multi_turn.enable=true", "rollout.agent.default_agent_loop=tool_agent"]
    if tool_config:
        overrides.append(f"rollout.multi_turn.tool_config_path={tool_config}")
    return overrides


def build_verl_command(cfg: dict, data_dir: Path, reward: Path) -> list[str]:
    """把 RL 配置摊平成 verl 命令行；未映射的键直接报错，不静默丢弃。"""
    pairs: list = []
    shensi_rl.flatten("", cfg, pairs)
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "ppo_megatron_trainer",
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/val.parquet",
        f"reward.custom_reward_function.path={reward}",
        "reward.custom_reward_function.name=compute_score",
    ]
    config_only = set(getattr(shensi_rl, "CONFIG_ONLY_SECTIONS", ())) | {"early_stop"}
    table = {**shensi_rl.CLI_MAP, **_VERL_CLI_EXTRA}
    for key, value in pairs:
        if key.split(".", 1)[0] in config_only:
            continue
        if key.startswith("rollout.engine_kwargs."):
            suffix = key[len("rollout.engine_kwargs.") :]
            cmd.append(f"+actor_rollout_ref.rollout.engine_kwargs.{suffix}={value}")
            continue
        # 这两段的子键要用 `++`（有则覆写、无则追加）：verl 把 override_transformer_config 声明成空
        # dict，hydra 的 struct 里没有这些子键，普通覆写会被拒（整块传 dict 又过不了覆写语法）；
        # 而 ref 侧用 `oc.select` 继承 actor 的那份，所以 `+` 在 ref 上会报"已存在"。
        side, sep, _child = key.partition(_TRANSFORMER_CFG)
        if sep and side in ("actor", "ref"):
            cmd.append(f"++actor_rollout_ref.{key}={value}")
            continue
        cli = table.get(key)
        if cli is None:
            raise SystemExit(f"[looma] config 里有没映射到 verl CLI 的键：{key}")
        cmd.append(f"{cli}={value}")
    return cmd


def run_verl(stage: str, here: Path, reward: Path, argv: list[str], watch: dict | None) -> int:
    """起一个 RL 臂：解析配置、构造命令、前台运行（带早停看门狗）。"""
    parser = argparse.ArgumentParser(
        description=f"Looma {stage} 启动器（verl GRPO + Megatron actor）"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="override", action="append", default=[])
    args = parser.parse_args(argv)

    paths = env_paths()
    cfg_path = Path(args.config) if args.config else Path(here) / f"config/{args.profile}.yaml"
    cfg = base.resolve_cfg(shensi_rl._load_with_base(cfg_path))
    shensi_rl.apply_overrides(cfg, args.override)
    shensi_rl.resolve_paths(cfg, here)
    data_dir = Path(args.data_dir or Path(paths["data"]) / stage)
    cmd = build_verl_command(cfg, data_dir, reward)
    cmd.append(f"hydra.run.dir={Path(paths['runs']) / stage}")
    print(f"[looma] {stage} 命令：\n  " + " \\\n    ".join(cmd))
    if args.dry_run:
        return 0
    env = base.subprocess_env(strip_proxy=True)
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("VERL_USE_EXTERNAL_MODULES", f"shensi.runtime,{BRIDGE_MODULE}")
    env.setdefault("VERL_PLATFORM", "nvidia_noipc")
    env.setdefault("VLLM_PLUGINS", "")
    run_dir = Path(paths["runs"]) / stage
    log_path = run_dir / "logs" / "host_0_localhost.output"
    run_dir.mkdir(parents=True, exist_ok=True)
    rc, report = spawn(cmd, env, run_dir, log_path, watch=watch)
    if report is not None:
        print(f"[looma] {stage} 早停生效：{report.get('why')}——按成功处理")
        return 0
    return rc
