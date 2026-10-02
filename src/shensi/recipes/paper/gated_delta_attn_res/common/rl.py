"""RL 启动：verl 命令行映射、agent harness 接线与进程拉起。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res.common.train import launcher
from shensi.recipes.shensi.common import common as base

from .paths import env_paths

_VERL_CLI_EXTRA: dict[str, str] = {
    "algorithm.norm_adv_by_std_in_grpo": "algorithm.norm_adv_by_std_in_grpo",
    "algorithm.filter_groups.enable": "algorithm.filter_groups.enable",
    "algorithm.filter_groups.metric": "algorithm.filter_groups.metric",
    "algorithm.filter_groups.max_num_gen_batches": "algorithm.filter_groups.max_num_gen_batches",
    "algorithm.use_kl_in_reward": "algorithm.use_kl_in_reward",
    "actor.optim.use_layer_wise_distributed_optimizer": (
        "actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer"
    ),
    "actor.optim.use_layer_wise_param_layout": (
        "actor_rollout_ref.actor.optim.use_layer_wise_param_layout"
    ),
    "actor.clip_ratio_low": "actor_rollout_ref.actor.clip_ratio_low",
    "actor.clip_ratio_high": "actor_rollout_ref.actor.clip_ratio_high",
    "actor.clip_ratio_c": "actor_rollout_ref.actor.clip_ratio_c",
    "actor.loss_agg_mode": "actor_rollout_ref.actor.loss_agg_mode",
    "actor.policy_loss.loss_mode": "actor_rollout_ref.actor.policy_loss.loss_mode",
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


def build_verl_command(cfg: dict, stage: str, data_dir: Path, reward: Path) -> list[str]:
    """把本配方的 RL 配置翻译成 verl 的命令行：逐键映射，没映射到 CLI 的键直接报错（不静默丢配置）。"""
    from shensi.recipes.shensi.common import rl as shensi_rl

    pairs: list = []
    shensi_rl.flatten("", cfg, pairs)
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
    table = {**shensi_rl.CLI_MAP, **_VERL_CLI_EXTRA}
    config_only = set(getattr(shensi_rl, "CONFIG_ONLY_SECTIONS", ())) | {"early_stop"}
    unknown = []
    for key, val in pairs:
        if key.split(".", 1)[0] in config_only:
            continue
        if key.startswith("rollout.engine_kwargs."):
            cmd.append(
                f"+actor_rollout_ref.rollout.engine_kwargs.{key[len('rollout.engine_kwargs.') :]}={val}"
            )
            continue
        cli = table.get(key)
        if cli is None and (
            key.startswith("critic.")
            or key.startswith("distillation.")
            or key.startswith("actor.megatron.override_transformer_config.")
            or key.startswith("ref.megatron.override_transformer_config.")
        ):
            cli = "actor_rollout_ref." + key if key.startswith(("actor.", "ref.")) else key
        if cli is None:
            unknown.append(key)
            continue
        cmd.append(f"{cli}={val}")
    if unknown:
        raise SystemExit(f"[{stage}] config 里有没映射到 verl CLI 的键：{unknown}")
    return cmd


def run_verl(stage: str, here: Path, reward: Path, argv: list[str], watch: dict | None) -> int:
    """RL 的公共启动后半程：组装命令、把 bridge 挂进 ``VERL_USE_EXTERNAL_MODULES``、拉起进程并在同会话看护早停。"""
    import argparse as _ap

    from shensi.recipes.shensi.common import rl as shensi_rl

    ap = _ap.ArgumentParser(description=f"GDAR {stage} 启动器（verl GRPO + Megatron actor）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args(argv)

    paths = env_paths()
    cfg_path = Path(args.config) if args.config else here / f"config/{args.profile}.yaml"
    cfg = base.resolve_cfg(shensi_rl._load_with_base(cfg_path))
    shensi_rl.apply_overrides(cfg, args.override)
    shensi_rl.resolve_paths(cfg, here)
    data_dir = Path(args.data_dir or Path(paths["data"]) / stage)
    cmd = build_verl_command(cfg, stage, data_dir, reward)
    cmd.append(f"hydra.run.dir={Path(paths['runs']) / stage}")
    print(f"[{stage}] 命令：\n  " + " \\\n    ".join(cmd))
    if args.dry_run:
        return 0
    env = base.subprocess_env(strip_proxy=True)
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault(
        "VERL_USE_EXTERNAL_MODULES",
        "shensi.runtime,shensi.recipes.paper.gated_delta_attn_res.stage2_rl.gdar_bridge",
    )
    env.setdefault("VERL_PLATFORM", "nvidia_noipc")
    env.setdefault("VLLM_PLUGINS", "")
    log_path = Path(paths["runs"]) / stage / "logs" / "host_0_localhost.output"
    run_dir = Path(paths["runs"]) / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    rc, report = launcher.spawn_with_watchdog(cmd, env, run_dir, log_path, watch=watch)
    if report is not None:
        print(
            f"[{stage}] 早停生效：{report.get('why')}（metric={report.get('metric')}）——按成功处理"
        )
        return 0
    return rc


_AGENT_FAMILY_PREFIX = ("qwen3_gdar",)


def model_type_of(path: str | Path | None) -> str:
    """读 HF 目录 config.json 里的 model_type；读不到返回空串。"""
    if not path:
        return ""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    try:
        import json

        return str(json.loads(p.read_text(encoding="utf-8")).get("model_type") or "")
    except Exception:  # noqa: BLE001  缺配置/非 HF 目录都不该打断启动
        return ""


def agent_harness(model_path: str | Path | None) -> dict:
    """模型属于本配方的 agent 家族时，返回 harness 接线信息；否则返回空。"""
    if not model_type_of(model_path).startswith(_AGENT_FAMILY_PREFIX):
        return {}
    from shensi.recipes.shensi.common import harness

    return {
        "name": harness.HARNESS_DEFAULT,
        "home": harness.dsh_home({}),
        "command": harness.DEFAULT_COMMAND,
        "install": harness.SDK_PACKAGE,
    }


def agent_overrides(model_path: str | Path | None, *, tool_config: str | None = None) -> list[str]:
    """按需给 agent 家族模型追加多轮 rollout 相关覆写。"""
    if not agent_harness(model_path):
        return []
    over = [
        "rollout.multi_turn.enable=true",
        "rollout.agent.default_agent_loop=tool_agent",
    ]
    if tool_config:
        over.append(f"rollout.multi_turn.tool_config_path={tool_config}")
    return over
