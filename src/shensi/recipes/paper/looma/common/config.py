"""配置组装：YAML 深合并、profile 解析与训练入口参数。"""

from __future__ import annotations

from pathlib import Path

from shensi.recipes.shensi.common import common as base

from .algos import DEFAULT_ALGO, apply_algo_or_die, apply_model_algo
from .paths import env_paths, stage_dirs

_YAML_SUFFIXES = (".yaml", ".yml")


def load_yaml(path: Path) -> dict:
    """读 YAML 成字典；文件不存在时返回空表。"""
    return base.load_yaml(path)


def resolve_cfg(cfg: dict) -> dict:
    """把配置里的相对路径与 ``${oc.env:…}`` 落成可直接使用的值。"""
    return base.resolve_cfg(cfg)


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    base._set_dotted(cfg, dotted, value)


def _coerce(value: str):
    return base._coerce(value)


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """应用 ``点号键=值`` 覆写列表（``--set`` 的公共实现）。"""
    for item in items or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key, _coerce(value))
    return cfg


def _stage_cfg(cdir: Path) -> dict:
    raw = load_yaml(cdir / "default.yaml")
    raw.pop("defaults", None)
    parent_name = raw.pop("base", None)
    parent = _stage_cfg((cdir / parent_name).resolve()) if parent_name else {}
    return base._deep_merge(parent, raw)


def merge_profile(stage: str, profile: str) -> dict:
    """合并 ``config/<profile>.yaml`` 与它的 ``base:`` 继承链。"""
    cfg_dir = stage_dirs(stage) / "config"
    profile_path = cfg_dir / f"{profile}{'.yaml' if not profile.endswith(_YAML_SUFFIXES) else ''}"
    if not profile_path.is_file():
        raise SystemExit(f"[looma] {stage} 没有这个 profile：{profile_path}")
    overlay = load_yaml(profile_path)
    overlay.pop("defaults", None)
    return base._deep_merge(_stage_cfg(cfg_dir), overlay)


def _bind_paths(cfg: dict, stage: str, profile: str, ckpt_profile: str | None = None) -> Path:
    paths = env_paths()
    cfg["experiment"]["exp_dir"] = str(Path(paths["runs"]) / stage / profile)
    ckpt = Path(paths["ckpt"]) / stage / (ckpt_profile or profile)
    cfg.setdefault("train", {}).setdefault("system", {}).setdefault("checkpoint", {})["save"] = str(
        ckpt
    )
    return ckpt


def build_config(
    stage: str,
    profile: str,
    override: list[str],
    data_dir: Path,
    tokens: int | None = None,
    model_algo: str | None = None,
    load_ckpt: str | None = None,
) -> dict:
    """把 stage 默认档、profile 叠加与命令行覆写合成一份完整训练配置。"""
    cfg = merge_profile(stage, profile)
    cfg.setdefault("experiment", {})
    ckpt = _bind_paths(cfg, stage, profile)
    paths = env_paths()
    train = cfg.setdefault("train", {})
    for section in ("model", "data"):
        train.setdefault(section, {})["tokenizer_model"] = paths["tokenizer"]
    tok = train["data"].setdefault("tokenizer", {})
    tok["tokenizer_model"] = paths["tokenizer"]
    tok.setdefault("tokenizer_type", "HuggingFaceTokenizer")
    blend = base.load_blend(data_dir)
    if blend:
        train["data"]["data_path"] = blend
    if load_ckpt:
        cfg["experiment"]["load"] = load_ckpt
        train["system"]["checkpoint"]["load"] = load_ckpt
    if model_algo:
        apply_model_algo(cfg, model_algo)
    elif not train["model"].get("spec"):
        apply_model_algo(cfg, DEFAULT_ALGO)
    apply_overrides(cfg, override)
    cfg = resolve_cfg(cfg)
    model = cfg["train"]["model"]
    if tokens:
        batch, seq = int(model.get("global_batch_size") or 0), int(model.get("seq_length") or 0)
        if batch > 0 and seq > 0:
            iters = max(1, int(tokens) // (batch * seq))
            model["train_iters"] = iters
            print(
                f"[looma] token 预算 {tokens / 1e9:.1f}B → train_iters={iters}（batch={batch} × seq={seq}）"
            )
    if not cfg["train"]["system"]["checkpoint"].get("save"):
        raise SystemExit("[looma] 配置里没有 checkpoint.save")
    _ = ckpt
    return cfg


def smoke_config(stage: str, profile: str = "tiny", override: list[str] | None = None) -> dict:
    """组装冒烟档配置（tiny 几何、mock 数据）。"""
    cfg = merge_profile(stage, profile)
    cfg.setdefault("experiment", {})
    paths = env_paths()
    cfg["experiment"]["exp_dir"] = str(Path(paths["runs"]) / f"smoke_{stage}_{profile}")
    _bind_paths(cfg, stage, profile)
    if not cfg["train"]["model"].get("spec"):
        apply_model_algo(cfg, DEFAULT_ALGO)
    for item in override or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key, _coerce(value))
    return resolve_cfg(cfg)


def profile_from_args(config: str | None, profile: str, stage: str) -> str:
    """把 ``--config <路径或名字>`` 归一成档名。"""
    if not config:
        return profile
    name = Path(config).name
    if name.endswith((".yaml", ".yml")):
        name = name.rsplit(".", 1)[0]
    return name


def load_blend_spec(path: Path) -> dict:
    """读取配比 JSON（数据集清单与权重）。"""
    return base.load_blend_spec(path)


def dataprep_config(path: str | Path | None) -> dict:
    """读取 data_prep 配置；没给路径时返回空表。"""
    if not path:
        return {}
    target = Path(path)
    if not target.is_file():
        raise SystemExit(f"[looma] 没有这个 data_prep 配置：{target}")
    cfg = load_yaml(target) or {}
    cfg.pop("defaults", None)
    return cfg


def add_common_train_args(ap) -> None:
    """把各 stage 共用的训练参数挂进 argparse。"""
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml")
    ap.add_argument("--config", default=None, help="配置文件路径（与 --profile 等价）")
    ap.add_argument("--model-algo", default=None, help=f"模型算法（不给则用 {DEFAULT_ALGO}）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不启动")
    ap.add_argument("--smoke", action="store_true", help="跑 tiny 档（mock 数据）")
    ap.add_argument(
        "--tokens", type=lambda v: int(float(v)), default=None, help="token 预算（认 1e9）"
    )
    ap.add_argument("--data-dir", default=None, help="预处理产物目录（含 blend.json）")
    ap.add_argument("--load", default=None, help="接续的检查点目录")
    ap.add_argument("--set", dest="override", action="append", default=[], help="点号键覆写")
    ap.add_argument("--early-stop", type=int, default=None, help="早停耐心")
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
    from .runner import early_stop_plan, run
    from .runner import smoke as smoke_run

    profile = profile_from_args(getattr(args, "config", None), args.profile, stage)
    if getattr(args, "smoke", False):
        return smoke_run(stage, "tiny", [*(smoke_overrides or []), *args.override])
    algo = apply_algo_or_die(args.model_algo)
    paths = env_paths()
    cfg = build_config(
        stage,
        profile,
        [*(overrides or []), *args.override],
        Path(data_dir or Path(paths["data"]) / stage),
        tokens=args.tokens,
        model_algo=algo,
        load_ckpt=args.load,
    )
    watch = None if args.no_early_stop else early_stop_plan(stage, cfg, args.early_stop)
    return run(cfg, args.dry_run, watch=watch)
