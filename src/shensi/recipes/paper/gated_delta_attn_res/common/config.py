"""配置组装：YAML 深合并、profile 解析、token 预算换算与训练入口参数。"""

from __future__ import annotations

from pathlib import Path

from shensi.recipes.shensi.common import common as base

from .algos import DEFAULT_ALGO, apply_algo_or_die, apply_model_algo
from .paths import _SHARED_GEOMS, env_paths, stage_dirs


def load_yaml(path: Path) -> dict:
    """读 YAML 成字典；文件不存在时返回空表。"""
    return base.load_yaml(path)


def resolve_cfg(cfg: dict) -> dict:
    """把配置里的相对路径与 ``${oc.env:…}`` 落成可直接使用的值。"""
    return base.resolve_cfg(cfg)


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    base._set_dotted(cfg, dotted, value)  # noqa: SLF001  配方内部的点号覆写


def _coerce(val: str):
    return base._coerce(val)


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """应用 ``点号键=值`` 覆写列表（``--set`` 的公共实现）。"""
    for item in items or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key, _coerce(value))
    return cfg


def load_blend(data_dir: Path):
    """读取数据目录里的 blend.json（语料清单）；没有就返回空。"""
    return base.load_blend(data_dir)


def load_blend_spec(path: Path) -> dict:
    """读取配比 JSON（数据集清单、权重与分级）。"""
    return base.load_blend_spec(path)


def add_common_train_args(ap) -> None:
    """把各 stage 共用的训练参数挂进 argparse。"""
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
    """训练入口的公共后半程：解析 profile、装配配置、默认开早停并启动。"""
    from .runner import early_stop_plan, run, smoke

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
    """把 ``--config`` 归一成档名：接受 ``x``、``x.yaml``、``config/子目录/x.yaml``；去掉 ``config/`` 前缀与调用方自行拼接的 ``data_prep/``，其余子目录保留。"""
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
    """读取 data_prep 配置；没给路径时返回空表。"""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[gdar] 没有这个 data_prep 配置：{p}")
    cfg = base.load_yaml(p) or {}
    cfg.pop("defaults", None)
    return cfg


def dataprep_config_for(here: Path, config: str | None, blend: str | None = None) -> dict:
    """解析语料准备配置：优先 ``config/data_prep/<档名>.yaml``；给的是配比名（data_blend_<名>.json）时直接当 blend 用。"""
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
    """把 stage 默认档、profile 叠加与命令行覆写合成一份完整训练配置：按 stage/profile 固定产物目录，补 tokenizer 与数据路径，再按 ``--set`` > ``--model-algo`` > profile 自带 > 默认算法 的顺序定层规格。"""
    sdir, cdir = stage_dirs(stage)
    cfg = _stage_cfg(cdir)
    if profile not in ("default", "", None):
        prof = cdir / f"{profile}.yaml"
        if not prof.is_file() and profile.startswith("geoms/"):
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


def smoke_config(stage: str, profile: str = "tiny", override: list[str] | None = None) -> dict:
    """组装冒烟档配置（tiny 几何、mock 数据），产物目录与正式档同规则。"""
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
