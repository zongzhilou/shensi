"""本配方的 common.py 模块。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res.train import launcher
from shensi.recipes.shensi import common as base

RECIPE = Path(__file__).resolve().parent

TOKENIZER_ENV = "SHENSI_GDAR_TOKENIZER"
TOKENIZER_DIR = RECIPE / "tokenizer" / "Qwen3-0.6B"

_MODELS = "shensi.recipes.paper.gated_delta_attn_res.models.megatron"
MODEL_ALGOS: dict[str, tuple[str, str] | None] = {
    "base": None,
    "qwen3_ar": (_MODELS, "ar_layer_spec"),
    "qwen3_ar_block4": (_MODELS, "ar_layer_spec_block4"),
    "qwen3_dar": (_MODELS, "dar_layer_spec"),
    "qwen3_dar_block4": (_MODELS, "dar_layer_spec_block4"),
    "qwen3_denseformer": (_MODELS, "denseformer_layer_spec"),
    "qwen3_realformer": (_MODELS, "realformer_layer_spec"),
    "qwen3_realformer_identity": (_MODELS, "realformer_layer_spec_identity"),
    "qwen3_realformer_reference": (_MODELS, "realformer_layer_spec_reference"),
    "qwen3_realformer_mean": (_MODELS, "realformer_layer_spec_mean"),
    "qwen3_mudd": (_MODELS, "mudd_layer_spec"),
    "qwen3_hc": (_MODELS, "hc_layer_spec"),
    "qwen3_mhc": (_MODELS, "mhc_layer_spec"),
    "qwen3_gdar_paper": (_MODELS, "gdar_layer_spec_paper"),
    "qwen3_gdar": (_MODELS, "gdar_layer_spec"),
    "qwen3_gdar_theory": (_MODELS, "gdar_layer_spec_theory"),
    "qwen3_gdar_upstream": (
        _MODELS,
        "gdar_layer_spec_upstream",
    ),
    "qwen3_gdar_fullrank": (_MODELS, "gdar_layer_spec_fullrank"),
    "qwen3_gdar_block2": (_MODELS, "gdar_layer_spec_block2"),
    "qwen3_gdar_block4": (_MODELS, "gdar_layer_spec_block4"),
    "qwen3_gdar_block8": (_MODELS, "gdar_layer_spec_block8"),
    "qwen3_gdar_block16": (_MODELS, "gdar_layer_spec_block16"),
    "qwen3_gdar_r16": (_MODELS, "gdar_layer_spec_block4_r16"),
    "qwen3_gdar_noladder": (_MODELS, "gdar_layer_spec_paper_noladder"),
    "qwen3_gdar_no_output_route": (_MODELS, "gdar_layer_spec_no_output_route"),
    "qwen3_gated_ar": (_MODELS, "gated_ar_layer_spec"),
    "a1a_gate_prefix": (_MODELS, "gdar_layer_spec_gate_prefix"),
    "a1b_gate_delta": (_MODELS, "gdar_layer_spec_gate_delta"),
    "a3_decay_projected": (_MODELS, "gdar_layer_spec_decay_projected"),
    "a4_lambda_free": (_MODELS, "gdar_layer_spec_lambda_free"),
    "a6_reference": (_MODELS, "gdar_layer_spec_reference"),
    "a9_half_init": (_MODELS, "gdar_half_init_layer_spec"),
    "a9_uniform_init": (_MODELS, "gdar_uniform_init_layer_spec"),
    "e3_scalar_gate": (_MODELS, "gated_ar_layer_spec_scalar"),
    "e3_no_gate": (_MODELS, "gated_ar_layer_spec_no_gate"),
    "e3_decay_only": (_MODELS, "gated_ar_layer_spec_decay_only"),
    "e3_erase_only": (_MODELS, "gated_ar_layer_spec_erase_only"),
    "e3_write_only": (_MODELS, "gated_ar_layer_spec_write_only"),
}

DEFAULT_ALGO = "qwen3_gdar_paper"

_MODELS_ABL = _MODELS + ".ablation_spec"
MODEL_ALGOS.update(
    {
        "qwen3_gdar_main": (_MODELS, "gdar_layer_spec_paper"),
        "qwen3_gdar_main_sublayer": (_MODELS, "gdar_layer_spec_paper_sublayer"),
        "qwen3_gdar_main_b2": (_MODELS, "gdar_layer_spec_paper_b2"),
        "qwen3_gdar_main_b8": (_MODELS, "gdar_layer_spec_paper_b8"),
        "qwen3_gdar_main_b16": (_MODELS, "gdar_layer_spec_paper_b16"),
        "qwen3_gdar_main_rank16": (_MODELS, "gdar_layer_spec_paper_rank16"),
        "qwen3_gdar_main_rankfull": (_MODELS, "gdar_layer_spec_paper_rankfull"),
        "qwen3_gdar_main_decay_free": (_MODELS, "gdar_layer_spec_paper_decay_free"),
        "qwen3_gdar_main_lambda_free": (_MODELS, "gdar_layer_spec_paper_lambda_free"),
        "qwen3_gdar_main_ladder0": (_MODELS, "gdar_layer_spec_paper_ladder0"),
        "qwen3_gdar_main_gate_prefix": (_MODELS, "gdar_layer_spec_paper_gate_prefix"),
        "qwen3_gdar_main_gate_delta": (_MODELS, "gdar_layer_spec_paper_gate_delta"),
        "qwen3_gdar_main_address_state": (_MODELS, "gdar_layer_spec_paper_address_state"),
        "qwen3_gdar_main_address_novelty": (_MODELS, "gdar_layer_spec_paper_address_novelty"),
        "qwen3_gdar_main_update_reference": (_MODELS, "gdar_layer_spec_paper_update_reference"),
        "qwen3_gdar_main_heads1": (_MODELS, "gdar_layer_spec_paper_heads1"),
        "qwen3_gdar_main_null_off": (_MODELS, "gdar_layer_spec_paper_null_off"),
        "qwen3_gdar_main_whiten_diag": (_MODELS, "gdar_layer_spec_paper_whiten_diag"),
        "qwen3_gdar_main_whiten_off": (_MODELS, "gdar_layer_spec_paper_whiten_off"),
        "qwen3_gdar_main_mix_whitened": (_MODELS, "gdar_layer_spec_paper_mix_whitened"),
        "qwen3_gdar_main_no_output_route": (_MODELS, "gdar_layer_spec_paper_no_output_route"),
        "qwen3_gdar_main_carrier0": (_MODELS, "gdar_layer_spec_paper_carrier0"),
        "qwen3_gdar_main_init_paper": (_MODELS, "gdar_layer_spec_paper_init_paper"),
        "qwen3_gdar_main_init_uniform": (_MODELS, "gdar_layer_spec_paper_init_uniform"),
        "qwen3_gdar_main_init_half": (_MODELS, "gdar_layer_spec_paper_init_half"),
        "qwen3_gdar_main_gates_d": (_MODELS_ABL, "gdar_paper_gates_d"),
        "qwen3_gdar_main_gates_e": (_MODELS_ABL, "gdar_paper_gates_e"),
        "qwen3_gdar_main_gates_w": (_MODELS_ABL, "gdar_paper_gates_w"),
        "qwen3_gdar_main_gates_de": (_MODELS_ABL, "gdar_paper_gates_de"),
        "qwen3_gdar_main_gates_dw": (_MODELS_ABL, "gdar_paper_gates_dw"),
        "qwen3_gdar_main_gates_ew": (_MODELS_ABL, "gdar_paper_gates_ew"),
        "qwen3_gdar_main_gates_scalar": (_MODELS_ABL, "gdar_paper_gates_scalar"),
        "qwen3_gdar_main_gates_none": (_MODELS_ABL, "gdar_paper_gates_none"),
        "qwen3_mhc_lite": (_MODELS, "mhc_layer_spec_lite"),
    }
)


def stage_dirs(stage: str) -> tuple[Path, Path]:
    for cand in (RECIPE / stage, RECIPE / "stage0_pretrain" / stage, RECIPE / "stage2_rl" / stage):
        if cand.is_dir():
            return cand, cand / "config"
    known = sorted(
        {*MODEL_ALGOS}
        | {"stage1_pretrain", "stage2_midtrain", "stage1_sft", "stage2_*", "stage3_opd"}
    )
    raise SystemExit(f"[gdar] 找不到 stage 目录：{stage}（已知 stage：{known}）")


def env_paths() -> dict:
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["runs"] = str(Path(paths["runs"]) / "gated_delta_attn_res")
    paths["data"] = Path(paths["data"]) / "gated_delta_attn_res"
    return paths


def apply_model_algo(cfg: dict, algo: str | None) -> str | None:
    if algo is None:
        return None
    if algo not in MODEL_ALGOS:
        raise SystemExit(
            f"[gdar] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}（base = plain Qwen3）"
        )
    entry = MODEL_ALGOS[algo]
    model = cfg.setdefault("train", {}).setdefault("model", {})
    if entry is None:
        model.pop("spec", None)
    else:
        module, obj = entry
        model["spec"] = [module, obj]
    return algo


def load_yaml(path: Path) -> dict:
    return base.load_yaml(path)


def resolve_cfg(cfg: dict) -> dict:
    return base.resolve_cfg(cfg)


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    base._set_dotted(cfg, dotted, value)  # noqa: SLF001  配方内部的点号覆写


def _coerce(val: str):
    return base._coerce(val)


def load_blend(data_dir: Path):
    return base.load_blend(data_dir)


def load_blend_spec(path: Path) -> dict:
    return base.load_blend_spec(path)


_SHARED_GEOMS = Path(__file__).resolve().parent / "stage0_pretrain/stage1_pretrain/config/geoms"


def add_common_train_args(ap) -> None:
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml")
    ap.add_argument("--config", default=None, help="配置文件路径（与 --profile 等价）")
    ap.add_argument(
        "--model-algo",
        default=None,
        help=f"模型算法（不给则用 {DEFAULT_ALGO}；profile 自带 spec 时用 profile 的）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不启动")
    ap.add_argument("--smoke", action="store_true", help="跑 tiny 档 5 步（mock 数据）")
    ap.add_argument(
        "--tokens", type=lambda v: int(float(v)), default=None, help="token 预算（认 1e9）"
    )
    ap.add_argument("--data-dir", default=None, help="预处理产物目录（含 blend.json）")
    ap.add_argument("--load", default=None, help="接续的 ckpt 目录")
    ap.add_argument("--set", dest="override", action="append", default=[], help="点号键覆写")
    ap.add_argument("--early-stop", type=int, default=None, help="早停耐心（默认按配置）")
    ap.add_argument("--no-early-stop", action="store_true", help="关掉早停看门狗")


def train_from_args(
    stage: str,
    args,
    *,
    data_dir: Path | None = None,
    overrides: list[str] | None = None,
    smoke_overrides: list[str] | None = None,
) -> int:
    args.profile = profile_from_args(getattr(args, "config", None), args.profile, stage)
    if getattr(args, "smoke", False):
        return smoke(stage, "tiny", smoke_overrides or [])
    algo = apply_algo_or_die(args.model_algo)
    paths = env_paths()
    cfg = build_config(
        stage,
        args.profile,
        [*(overrides or []), *args.override],
        Path(data_dir or paths["data"] / stage),
        tokens=args.tokens,
        model_algo=algo,
        load_ckpt=args.load,
    )
    watch = None if args.no_early_stop else early_stop_plan(stage, cfg, args.early_stop)
    return run(cfg, args.dry_run, watch=watch)


def profile_from_args(config: str | None, profile: str, stage: str) -> str:
    """把 --config 归一成档名。

    认三种写法：`x`、`x.yaml`、`config/子目录/x.yaml`；`config/` 与调用方自己会拼的
    `data_prep/` 前缀都去掉，其余子目录（`geoms/`、`ablations/`）保留。
    """
    if not config:
        return profile
    p = Path(config)
    parts = list(p.with_suffix("").parts) if p.suffix in (".yaml", ".yml") else list(p.parts)
    if "config" in parts:
        parts = parts[parts.index("config") + 1 :]
    if parts and parts[0] == "data_prep":
        parts = parts[1:]
    return "/".join(parts) if parts else profile


def dataprep_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[gdar] 没有这个 data_prep 配置：{p}")
    cfg = base.load_yaml(p) or {}
    cfg.pop("defaults", None)
    return cfg


def dataprep_config_for(here: Path, config: str | None, blend: str | None = None) -> dict:
    """data_prep 的配置：`config/data_prep/<档名>.yaml` 优先；给混合名时直接当 blend 用。"""
    name = profile_from_args(config, "default", "")
    cdir = Path(here) / "config/data_prep"
    cfg_file = cdir / f"{name}.yaml"
    if cfg_file.is_file():
        cfg = dataprep_config(cfg_file)
    else:
        base = Path(config).name if config else ""
        if base.endswith(".json"):
            base = base[: -len(".json")]
        stem = base[len("data_blend_") :] if base.startswith("data_blend_") else base
        if not stem or not (cdir / f"data_blend_{stem}.json").is_file():
            raise SystemExit(
                f"[gdar] 没有这个 data_prep 配置：{cfg_file}"
                "（也可以给 config/data_prep/ 下的混合名，如 --config hybrid）"
            )
        cfg = {"blend": f"data_blend_{stem}.json"}
    if blend:
        cfg["blend"] = blend
    return cfg


def build_config(
    stage: str,
    profile: str,
    override: list[str],
    data_dir: Path,
    tokens: int | None = None,
    model_algo: str | None = None,
    load_ckpt: str | None = None,
) -> dict:
    sdir, cdir = stage_dirs(stage)
    cfg = _stage_cfg(cdir)
    if profile not in ("default", "", None):
        prof = cdir / f"{profile}.yaml"
        if not prof.is_file() and profile.startswith("geoms/"):
            # geoms/* 是 PT/Mid/SFT 共享的几何档，只有一份，落在 stage1_pretrain 下
            prof = _SHARED_GEOMS / f"{profile[len('geoms/') :]}.yaml"
        if not prof.is_file():
            raise SystemExit(f"[gdar] {stage} 没有这个 profile：{prof}")
        cfg = base._deep_merge(cfg, base.load_yaml(prof))
    paths = env_paths()
    cfg.setdefault("experiment", {})
    cfg["experiment"]["exp_dir"] = str(Path(paths["runs"]) / stage / profile)
    if load_ckpt:
        cfg["experiment"]["load"] = load_ckpt
    train = cfg.setdefault("train", {})
    model = train.setdefault("model", {})
    model["tokenizer_model"] = model.get("tokenizer_model") or paths["tokenizer"]
    data = train.setdefault("data", {})
    data["tokenizer_model"] = data.get("tokenizer_model") or paths["tokenizer"]
    tok = data.setdefault("tokenizer", {})
    tok.setdefault("tokenizer_type", "HuggingFaceTokenizer")
    tok.setdefault("tokenizer_model", paths["tokenizer"])
    ckpt = train.setdefault("system", {}).setdefault("checkpoint", {})
    ckpt["save"] = str(Path(paths["ckpt"]) / "gated_delta_attn_res" / stage / profile)
    if load_ckpt:
        ckpt["load"] = load_ckpt
    blend = load_blend(data_dir)
    if blend:
        data["data_path"] = blend
    # 优先级：--set 的 spec > --model-algo > profile 自带 spec > 默认算法
    if model_algo:
        apply_model_algo(cfg, model_algo)
    elif not (cfg.get("train", {}).get("model", {}) or {}).get("spec"):
        apply_model_algo(cfg, DEFAULT_ALGO)
    for item in override or []:
        key, _, val = item.partition("=")
        _set_dotted(cfg, key, _coerce(val))
    cfg = resolve_cfg(cfg)
    train = cfg["train"]
    model = train["model"]
    if tokens:
        gb = int(model.get("global_batch_size") or 0)
        seq = int(model.get("seq_length") or 0)
        if gb > 0 and seq > 0:
            iters = max(1, int(tokens) // (gb * seq))
            model["train_iters"] = iters
            print(
                f"[gdar] token 预算 {tokens / 1e9:.1f}B → train_iters={iters}"
                f"（gb={gb} × seq={seq}）"
            )
    if not train["system"]["checkpoint"].get("save"):
        raise SystemExit("[gdar] 没有 checkpoint.save")
    return cfg


def _stage_cfg(cdir: Path) -> dict:
    raw = load_yaml(cdir / "default.yaml")
    raw.pop("defaults", None)
    base_name = raw.pop("base", None)
    return base._deep_merge(_stage_cfg((cdir / base_name).resolve()) if base_name else {}, raw)


def apply_algo_or_die(algo: str | None) -> str | None:
    if algo is not None and algo not in MODEL_ALGOS:
        raise SystemExit(
            f"[gdar] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}（base = plain Qwen3）"
        )
    return algo


EARLY_STOP_DEFAULTS: dict[str, dict] = {
    "default": {"metric": "lm loss value", "mode": "min", "patience": 20, "grace": 1200.0},
    "stage2_rl": {"metric": "val/reward", "mode": "max", "patience": 8, "grace": 3600.0},
}


def early_stop_plan(stage: str, cfg: dict, patience: int | None = None) -> dict:
    plan = dict(EARLY_STOP_DEFAULTS.get(stage, EARLY_STOP_DEFAULTS["default"]))
    plan.update(
        {
            k: v
            for k, v in ((cfg.get("experiment") or {}).get("early_stop") or {}).items()
            if v is not None
        }
    )
    if patience is not None:
        plan["patience"] = int(patience)
    plan.setdefault("poll", 30.0)
    return plan


def write_run_dir(cfg: dict) -> Path:
    run_dir = Path(cfg["experiment"]["exp_dir"])
    run_dir = launcher.write_run_dir(cfg, run_dir)
    print(f"[gdar] 配置与命令已写入 {run_dir}（config.yaml / run.sh）")
    return run_dir


def run(cfg: dict, dry_run: bool, watch: dict | None = None) -> int:
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir, dry_run=dry_run, watch=watch)


def smoke_config(stage: str, profile: str = "tiny", override: list[str] | None = None) -> dict:
    _, cdir = stage_dirs(stage)
    path = cdir / f"{profile}.yaml"
    if not path.is_file():
        raise SystemExit(f"[gdar] 没有这个冒烟档：{path}")
    cfg = _stage_cfg(cdir)
    prof = base.load_yaml(path)
    prof.pop("defaults", None)
    cfg = base._deep_merge(cfg, prof)
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault(
        "exp_dir", str(Path(env_paths()["runs"]) / f"smoke_{stage}_{profile}")
    )
    cfg.setdefault("train", {}).setdefault("system", {}).setdefault("checkpoint", {})["save"] = str(
        Path(env_paths()["ckpt"]) / "gated_delta_attn_res" / stage / profile
    )
    for item in override or []:
        key, _, val = item.partition("=")
        _set_dotted(cfg, key, _coerce(val))
    return resolve_cfg(cfg)


_VERL_CLI_EXTRA: dict[str, str] = {
    "algorithm.norm_adv_by_std_in_grpo": "algorithm.norm_adv_by_std_in_grpo",
    "algorithm.filter_groups.enable": "algorithm.filter_groups.enable",
    "algorithm.filter_groups.metric": "algorithm.filter_groups.metric",
    "algorithm.filter_groups.max_num_gen_batches": "algorithm.filter_groups.max_num_gen_batches",
    "algorithm.use_kl_in_reward": "algorithm.use_kl_in_reward",
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
    from shensi.recipes.shensi import rl as shensi_rl

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
    import argparse as _ap

    from shensi.recipes.shensi import rl as shensi_rl

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


def smoke(stage: str, profile: str = "tiny", override: list[str] | None = None) -> int:
    cfg = smoke_config(stage, profile, override)
    print(f"[gdar] 冒烟档：{stage} / tiny 几何 / mock 数据 / 5 步（配置见 config/{profile}.yaml）")
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir)


def watch(cfg: dict, patience: int, **kwargs) -> int:
    return base.watch(cfg, patience, **kwargs)


_AGENT_FAMILY_PREFIX = ("qwen3_gdar",)


def model_type_of(path: str | Path | None) -> str:
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
    if not model_type_of(model_path).startswith(_AGENT_FAMILY_PREFIX):
        return {}
    from shensi.recipes.shensi import harness

    return {
        "name": harness.HARNESS_DEFAULT,
        "home": harness.dsh_home({}),
        "command": harness.DEFAULT_COMMAND,
        "install": harness.SDK_PACKAGE,
    }


def agent_overrides(model_path: str | Path | None, *, tool_config: str | None = None) -> list[str]:
    if not agent_harness(model_path):
        return []
    over = [
        "rollout.multi_turn.enable=true",
        "rollout.agent.default_agent_loop=tool_agent",
    ]
    if tool_config:
        over.append(f"rollout.multi_turn.tool_config_path={tool_config}")
    return over
