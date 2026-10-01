"""GDAR 配方共用：stage 定位、路径、配置合并、模型算法选择与启动。

组织与 shensi 主配方（`shensi.recipes.shensi`）一致，按训练环节拆 stage：

    stage0_pretrain/stage1_pretrain   预训练（stable → decay 两段）
    stage0_pretrain/stage2_midtrain   中训练（Mid-1 能力强化 → Mid-2 长文档）
    stage1_sft                        SFT（SFT-1 deep-thinking → SFT-2 agent）
    stage2_rl/stage2_{math,code,agent,writing}   四方向 RL teacher（并行）
    stage3_opd                        OPD 蒸馏回发布基座

与主配方的两点差别：

1. **模型算法可选**：`--model-algo qwen3_gdar_paper / qwen3_ar / qwen3_dar / …`
   从 `MODEL_ALGOS` 注册表把 `train.model.spec` 指到本配方 `models/` 的层规格
   （算法即 spec 预设；`base` = 无 spec 的 plain Qwen3 对照）。所有 stage 用同一
   注册表，所以 PT→SFT→RL→OPD 全链路的算法选择是一个开关。
2. **tokenizer 默认 Qwen3**（`tokenizer/Qwen3-0.6B`，`$SHENSI_GDAR_TOKENIZER` 可覆盖）。

合并/覆写/解析这些纯函数直接复用 shensi common（一个出处，不复制第二份）。
"""

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

# ---------------------------------------------------------------------------
# 模型算法注册表：名字 -> (spec 模块, spec 对象)；None = plain Qwen3（base 对照臂）。
# 全部指向本配方 models/ 的层规格（mcore `--spec` 官方扩展点），算法即 spec。
# ---------------------------------------------------------------------------
_MODELS = "shensi.recipes.paper.gated_delta_attn_res.models.megatron"
MODEL_ALGOS: dict[str, tuple[str, str] | None] = {
    # --- 对照与连接模块矩阵 ---
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
    # --- GDAR 本体（论文主行打星）---
    "qwen3_gdar_paper": (_MODELS, "gdar_layer_spec_paper"),  # ★ 论文主行
    "qwen3_gdar": (_MODELS, "gdar_layer_spec"),  # 忠实参考版（shensi 更新式）
    "qwen3_gdar_theory": (_MODELS, "gdar_layer_spec_theory"),
    "qwen3_gdar_upstream": (
        _MODELS,
        "gdar_layer_spec_upstream",
    ),  # 与上游 shensi 分支逐位对齐（逐头白化）
    "qwen3_gdar_fullrank": (_MODELS, "gdar_layer_spec_fullrank"),
    "qwen3_gdar_block2": (_MODELS, "gdar_layer_spec_block2"),
    "qwen3_gdar_block4": (_MODELS, "gdar_layer_spec_block4"),
    "qwen3_gdar_block8": (_MODELS, "gdar_layer_spec_block8"),
    "qwen3_gdar_block16": (_MODELS, "gdar_layer_spec_block16"),
    "qwen3_gdar_r16": (_MODELS, "gdar_layer_spec_block4_r16"),  # 参数匹配版
    "qwen3_gdar_noladder": (_MODELS, "gdar_layer_spec_paper_noladder"),
    "qwen3_gdar_no_output_route": (_MODELS, "gdar_layer_spec_no_output_route"),
    "qwen3_gated_ar": (_MODELS, "gated_ar_layer_spec"),
    # --- 设计消融（每行只改一处，与主行直接可比）---
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

#: 各 stage 的默认算法（没给 --model-algo 时用）
#: 正式训练的默认 = 论文主行（main：theory+正性投影+r64，block 形态 B=4）。
DEFAULT_ALGO = "qwen3_gdar_paper"

# ---------------------------------------------------------------------------
# 设计矩阵（EXPERIMENT_MATRIX.json）的 GDAR 主臂族：一个 --model-algo 名字对一行。
# 名字规则：`qwen3_gdar_main*` = 论文主配置（main）与它的 block 形态/秩/单旋钮行；
# `qwen3_gdar_main_gates_*` = 门结构族（A8）。每个预设在 gdar_spec / ablation_spec 里
# 都只与 main 差一处，所以任意两行都能直接 diff。
# ---------------------------------------------------------------------------
_MODELS_ABL = _MODELS + ".ablation_spec"
MODEL_ALGOS.update(
    {
        "qwen3_gdar_main": (_MODELS, "gdar_layer_spec_paper"),  # 主行（B=4，theory+project+r64）
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
    """定位 stage 目录：直接子目录 / stage0_pretrain 下 / stage2_rl 的方向臂。"""
    for cand in (RECIPE / stage, RECIPE / "stage0_pretrain" / stage, RECIPE / "stage2_rl" / stage):
        if cand.is_dir():
            return cand, cand / "config"
    known = sorted(
        {*MODEL_ALGOS}
        | {"stage1_pretrain", "stage2_midtrain", "stage1_sft", "stage2_*", "stage3_opd"}
    )
    raise SystemExit(f"[gdar] 找不到 stage 目录：{stage}（已知 stage：{known}）")


def env_paths() -> dict:
    """Shensi 的路径表 + 本配方的覆盖项（tokenizer / runs / data 子目录）。"""
    paths = base.env_paths()
    paths["tokenizer"] = os.environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["runs"] = str(Path(paths["runs"]) / "gated_delta_attn_res")
    # data_prep 产物目录：data/gated_delta_attn_res/<stage>（避免和主配方产物撞名）
    paths["data"] = Path(paths["data"]) / "gated_delta_attn_res"
    return paths


def apply_model_algo(cfg: dict, algo: str | None) -> str | None:
    """把 `--model-algo` 落成 `train.model.spec`（算法即 spec 预设）。

    注册表里值为 None 的（base）会**移除** spec；显式 `--set train.model.spec=...`
    在 algo 之后应用（common.build_config 的顺序），仍可强制自定义。
    """
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


#: 跨 stage 共享的几何档目录（`--profile geoms/qwen3_8b` 从任何 stage 都解析到这里）
_SHARED_GEOMS = Path(__file__).resolve().parent / "stage0_pretrain/stage1_pretrain/config/geoms"


def add_common_train_args(ap) -> None:
    """各训练 stage 共用的参数（stage 自己的开关由 stage 的 `common/train.py` 追加）。"""
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
    """公共核：`--smoke` / 组配置 / 起训（含早停看门狗）。

    `data_dir` 不给就用 `<FS>/shensi/data/gated_delta_attn_res/<stage>`；`overrides` 是
    stage 侧额外注入的点号覆写（如 SFT 的 `train.data.data_path=...`）。
    """
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
    """把 `--config <路径|名字>` 折成 profile 名：`config/decay.yaml` 与 `decay` 等价。

    没给 `--config` 时原样返回 `--profile`。
    """
    if not config:
        return profile
    name = Path(config).name
    if name.endswith((".yaml", ".yml")):
        name = name.rsplit(".", 1)[0]
    return name


def dataprep_config(path: str | Path | None) -> dict:
    """读 `config/data_prep/<name>.yaml`（键：blend / limit / workers / only / data_dir）。"""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[gdar] 没有这个 data_prep 配置：{p}")
    cfg = base.load_yaml(p) or {}
    cfg.pop("defaults", None)
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
    """读 `<stage>/config/{default,<profile>}.yaml` 并解析成可直接起训的配置。

    `model_algo`（默认 `DEFAULT_ALGO`，但 profile 自带 spec 时以 profile 为准）落成
    `train.model.spec`；`load_ckpt` 供
    decay / mid / SFT / RL 等接续段加载上一段 ckpt（`experiment.load` + `--load`）。
    """
    sdir, cdir = stage_dirs(stage)
    cfg = _stage_cfg(cdir)
    if profile not in ("default", "", None):
        prof = cdir / f"{profile}.yaml"
        if not prof.is_file() and profile.startswith("geoms/"):
            # `geoms/*` 是 PT / Mid / SFT 共享的几何档（只维护一份，落在 stage1_pretrain 的
            # config 下）；从任一 stage 都指得到，省得三处各抄一份会漂移的几何。
            prof = _SHARED_GEOMS / f"{profile[len('geoms/') :]}.yaml"
        if not prof.is_file():
            raise SystemExit(f"[gdar] {stage} 没有这个 profile：{prof}")
        cfg = base._deep_merge(cfg, base.load_yaml(prof))
    paths = env_paths()
    cfg.setdefault("experiment", {})
    # 产物路径按 stage 强制赋值（不用 setdefault）：default.yaml 走 `base:` 继承链时，
    # 上一段的 exp_dir/ckpt 会跟着继承进来，必须在这里按本 stage 覆盖掉。
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
    # 算法优先级：--set train.model.spec=... > --model-algo > profile 自带的 spec > 默认算法。
    # profile 自带 spec 的档（ablations/*.yaml、ar.yaml、dar.yaml）不能被默认算法静默盖掉。
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
    raw.pop("defaults", None)  # 冒烟档里的 hydra 残留键（launcher 不读它）
    base_name = raw.pop("base", None)
    return base._deep_merge(_stage_cfg((cdir / base_name).resolve()) if base_name else {}, raw)


def apply_algo_or_die(algo: str | None) -> str | None:
    """校验 --model-algo 的名字，**原样返回**（None 保持 None）。

    默认算法的兜底在 build_config 里做：显式给了才覆盖 profile 自带的 spec。
    """
    if algo is not None and algo not in MODEL_ALGOS:
        raise SystemExit(
            f"[gdar] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}（base = plain Qwen3）"
        )
    return algo


#: 各 stage 的早停默认（patience 以"评估次数"计；grace 是启动宽限秒数）。
#: 训练步数/轮次配置一律给"无限大"，收敛与收尾交给这里——早停算成功，不算失败。
EARLY_STOP_DEFAULTS: dict[str, dict] = {
    # metric 用 "lm loss value"（不是 "validation loss"）：日志是
    # "validation loss at iteration 5 | lm loss value: 1.06E+01"
    # 看门狗取"指标名后第一个数"，跟 "validation loss" 会抓到 iteration 号（踩过）。
    "default": {"metric": "lm loss value", "mode": "min", "patience": 20, "grace": 1200.0},
    "stage2_rl": {"metric": "val/reward", "mode": "max", "patience": 8, "grace": 3600.0},
}


def early_stop_plan(stage: str, cfg: dict, patience: int | None = None) -> dict:
    """看门狗计划：stage 默认 + 配置 ``experiment.early_stop`` 覆盖 + CLI patience 覆盖。"""
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
    """起一次训练：写 run 目录 → torchrun → 前台等返回码（默认带早停看门狗）。

    ``watch`` 由调用方用 :func:`early_stop_plan` 生成；早停生效时返回 0（算成功）。
    """
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir, dry_run=dry_run, watch=watch)


def smoke_config(stage: str, profile: str = "tiny", override: list[str] | None = None) -> dict:
    """读 `<stage>/config/{profile}.yaml`（mock 数据档）并解析成可直接起训的配置。"""
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
    # 检查点也按 stage/profile 各存一份——与 build_config 同一条规则。此前冒烟档用了 profile
    # 里的 save（配方 ckpt 根/iter_*），几个 stage 的冒烟互相覆盖同名迭代，导出的 ckpt 会和它
    # 自己的 exp 目录 config 对不上。
    cfg.setdefault("train", {}).setdefault("system", {}).setdefault("checkpoint", {})["save"] = str(
        Path(env_paths()["ckpt"]) / "gated_delta_attn_res" / stage / profile
    )
    for item in override or []:
        key, _, val = item.partition("=")
        _set_dotted(cfg, key, _coerce(val))
    # 冒烟档自带 spec（tiny.yaml 写明冒的就是哪条路），不走 --model-algo 覆盖
    return resolve_cfg(cfg)


#: 设计矩阵需要的 verl 旋钮在 shensi 映射表之外的部分（DAPO/Dr.GRP O/ critic 等）。
#: 与 shensi 的映射合并后由本配方的 ``build_verl_command`` 使用（不改共享模块）。
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
    # 策略损失本体（verl 的 policy_loss 注册表：vanilla / gspo / cispo / sapo / dppo_tv / dppo_kl /
    # dro / geo_mean / clip_cov / kl_cov …）—— GSPO 在这里，**不在** advantage 估计量里。
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
    """与 ``shensi.recipes.shensi.rl.build_command`` 同构，映射表叠加 ``_VERL_CLI_EXTRA``。"""
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
    # 本仓库私有配置段（harness 接线、早停计划等）由本配方自己消费，不进 verl CLI
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
            or key.startswith("distillation.")  # verl 原生的 on-policy distillation 段（任意字典）
            or key.startswith("actor.megatron.override_transformer_config.")
            or key.startswith("ref.megatron.override_transformer_config.")
        ):
            # 任意 dict 段：verl CLI 里与 yaml 同构，直接直传（critic.*、
            # override_transformer_config.* 那种没有固定键名的段）
            cli = "actor_rollout_ref." + key if key.startswith(("actor.", "ref.")) else key
        if cli is None:
            unknown.append(key)
            continue
        cmd.append(f"{cli}={val}")
    if unknown:
        raise SystemExit(f"[{stage}] config 里有没映射到 verl CLI 的键：{unknown}")
    return cmd


def run_verl(stage: str, here: Path, reward: Path, argv: list[str], watch: dict | None) -> int:
    """RL 各臂的启动：命令与环境和 ``shensi.recipes.shensi.rl.launch`` 同一套
    （同一份 yaml → verl CLI 映射、同一份进程环境），只是换成我们的前台启动器 + 早停看门狗。
    """
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
    # 每个 import verl 的进程（driver / ray worker / vLLM server）都会加载这些模块：
    # shensi.runtime 登记第三方要的东西，gdar_bridge 登记本配方的 model_type -> 桥
    # （导入即注册；不注册的话 verl 建模型时会以 "architecture not yet supported" 报错，不会静默建错）。
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
    """冒烟：tiny 几何 + mock 数据 + 5 步（不碰真实语料）。"""
    cfg = smoke_config(stage, profile, override)
    print(f"[gdar] 冒烟档：{stage} / tiny 几何 / mock 数据 / 5 步（配置见 config/{profile}.yaml）")
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir)


def watch(cfg: dict, patience: int, **kwargs) -> int:
    """早停看门狗（复用 shensi 的 early_stop.py）。"""
    return base.watch(cfg, patience, **kwargs)


# ---------------------------------------------------------------------------
# RL / OPD 的 rollout 驱动：多轮工具环境由本框架的外部 harness 执行
# （``shensi.recipes.shensi.harness``，默认 dsh）。是否启用由**起点的模型类型**决定：
# 读 ckpt（目录或 config.json）里的 ``model_type``，命中即接上；其余档保持默认。
# ---------------------------------------------------------------------------
_AGENT_FAMILY_PREFIX = ("qwen3_gdar",)


def model_type_of(path: str | Path | None) -> str:
    """读一个 ckpt 目录（或 config.json）的 ``model_type``；读不到返回空串。"""
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
    """外部 harness 段（``harness:``）；不在族内返回空 dict。"""
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
    """接上 harness 时要打开的 rollout 旋钮（多轮 + 工具/环境交给 harness）。"""
    if not agent_harness(model_path):
        return []
    over = [
        "rollout.multi_turn.enable=true",
        "rollout.agent.default_agent_loop=tool_agent",
    ]
    if tool_config:
        over.append(f"rollout.multi_turn.tool_config_path={tool_config}")
    return over
