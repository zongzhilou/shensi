"""配置读取与合并：stage 定位、profile 叠加、点号覆写、token 预算换算。

约定与 looma 一致：``config/default.yaml`` 是 stage 的完整底档，``config/<profile>.yaml``
只写增量（合并时盖在 default 之上）；``--set 点号键=值`` 最后应用。
"""

from __future__ import annotations

from pathlib import Path

from shensi.recipes.shensi.common import common as base

from .paths import env_paths, stage_dirs

_YAML_SUFFIXES = (".yaml", ".yml")


def load_yaml(path: Path) -> dict:
    """读 YAML 成字典（不解析插值，合并后统一 resolve）。"""
    return base.load_yaml(path)


def resolve_cfg(cfg: dict) -> dict:
    """把 ``${oc.env:…}`` 一类插值落成可直接用的值。"""
    return base.resolve_cfg(cfg)


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """应用 ``点号键=值`` 覆写列表（``--set`` 的公共实现）。"""
    for item in items or []:
        key, _, value = item.partition("=")
        base._set_dotted(cfg, key, base._coerce(value))
    return cfg


def merge_profile(stage: str, profile: str) -> dict:
    """合并 ``config/<profile>.yaml``（增量）到 ``config/default.yaml``（底档）之上。

    档的查找顺序（``geoms/`` 是跨 stage 共享的规模几何档）：
    ``<stage>/config/<profile>.yaml`` → ``<stage>/config/geoms/<名字>.yaml``
    → ``<recipe>/common/config/geoms/<名字>.yaml``。
    """
    cfg_dir = stage_dirs(stage) / "config"
    name = profile if profile.endswith(_YAML_SUFFIXES) else f"{profile}.yaml"
    recipe_root = stage_dirs(stage).parent
    candidates = [
        cfg_dir / name,
        cfg_dir / "geoms" / Path(name).name,
        recipe_root / "common" / "config" / "geoms" / Path(name).name,
    ]
    profile_path = next((path for path in candidates if path.is_file()), None)
    if profile_path is None:
        raise SystemExit(
            f"[deeprecur] {stage} 没有这个 profile：{profile}"
            f"（找过：{', '.join(str(p) for p in candidates)}）"
        )
    overlay = load_yaml(profile_path)
    overlay.pop("defaults", None)
    bottom = load_yaml(cfg_dir / "default.yaml")
    bottom.pop("defaults", None)
    return base._deep_merge(bottom, overlay)


def _bind_paths(cfg: dict, stage: str, profile: str) -> Path:
    """按 stage 与 profile 强制赋值 exp_dir 与 ckpt 目录，返回 ckpt 目录。"""
    paths = env_paths()
    cfg.setdefault("experiment", {})["exp_dir"] = str(Path(paths["runs"]) / stage / profile)
    ckpt = Path(paths["ckpt"]) / stage / profile
    cfg.setdefault("train", {})["save_dir"] = str(ckpt)
    return ckpt


def build_config(
    stage: str,
    profile: str,
    override: list[str] | None,
    data_dir: Path,
    tokens: int | None = None,
    load_ckpt: str | None = None,
) -> dict:
    """组装一次训练的完整配置。

    参数:
      stage: stage 目录名（stage0_pt / stage1_sft）。
      profile: ``config/<profile>.yaml`` 的名字；``default`` 只用底档。
      override: ``点号键=值`` 覆写列表，最后应用。
      data_dir: 数据产物目录，含 ``<stage>_train.jsonl`` 时注入 ``data.jsonl``。
      tokens: token 预算，按 ``global_batch_size × assumed_seq_length`` 估 ``max_steps``
        （多模态序列长不定，这里是工程换算，论文口径默认 1 epoch）。
      load_ckpt: 接续的起点检查点（HF 格式目录）。
    """
    cfg = merge_profile(stage, profile)
    _bind_paths(cfg, stage, profile)
    paths = env_paths()
    cfg.setdefault("data", {})["tokenizer"] = paths["tokenizer"]
    if load_ckpt:
        cfg["model"]["placeholder"]["load"] = load_ckpt
    train_jsonl = Path(data_dir) / f"{stage}_train.jsonl"
    val_jsonl = Path(data_dir) / f"{stage}_val.jsonl"
    if train_jsonl.is_file():
        cfg["data"]["jsonl"] = str(train_jsonl)
        cfg["data"]["val_jsonl"] = str(val_jsonl) if val_jsonl.is_file() else None
    apply_overrides(cfg, override)
    cfg = resolve_cfg(cfg)
    if tokens:
        gbs = int(cfg["train"].get("global_batch_size") or 0)
        seq = int(cfg["train"].get("assumed_seq_length") or 0)
        if gbs > 0 and seq > 0:
            steps = max(1, int(tokens) // (gbs * seq))
            cfg["train"]["max_steps"] = steps
            print(
                f"[deeprecur] token 预算 {tokens / 1e9:.1f}B → max_steps≈{steps}"
                f"（batch={gbs} × 估序长 {seq}）"
            )
    return cfg


def smoke_config(stage: str, override: list[str] | None = None) -> dict:
    """组装冒烟配置：tiny 随机 Qwen3-VL + 合成数据 + 少量步数，全程离线。"""
    cfg = merge_profile(stage, "tiny")
    paths = env_paths()
    cfg["experiment"]["exp_dir"] = str(Path(paths["runs"]) / f"smoke_{stage}")
    cfg["train"]["save_dir"] = str(Path(paths["runs"]) / f"smoke_{stage}" / "ckpt")
    cfg["data"]["tokenizer"] = paths["tokenizer"]
    apply_overrides(cfg, override)
    return resolve_cfg(cfg)


def profile_from_args(config: str | None, profile: str) -> str:
    """把 ``--config <路径|名字>`` 折成 profile 名：``config/decay.yaml`` 与 ``decay`` 等价。"""
    if not config:
        return profile
    name = Path(config).name
    if name.endswith(_YAML_SUFFIXES):
        name = name.rsplit(".", 1)[0]
    return name


def dataprep_config(path: str | Path | None) -> dict:
    """读 ``config/data_prep/<name>.yaml``（键：blend / limit / val_frac / only / data_dir）。"""
    if not path:
        return {}
    target = Path(path)
    if not target.is_file():
        raise SystemExit(f"[deeprecur] 没有这个 data_prep 配置：{target}")
    cfg = load_yaml(target) or {}
    cfg.pop("defaults", None)
    return cfg


def add_common_train_args(ap) -> None:
    """各训练 stage 共用的命令行参数。"""
    ap.add_argument("--profile", default="default", help="config/<名字>.yaml")
    ap.add_argument("--config", default=None, help="配置文件路径（与 --profile 等价）")
    ap.add_argument("--dry-run", action="store_true", help="打印训练计划，不启动")
    ap.add_argument("--smoke", action="store_true", help="tiny 随机模型 + 合成数据，全程离线")
    ap.add_argument("--tokens", type=lambda v: int(float(v)), default=None, help="token 预算（认 1e9）")
    ap.add_argument("--data-dir", default=None, help="数据产物目录（含 <stage>_train.jsonl）")
    ap.add_argument("--load", default=None, help="接续的检查点目录（HF 格式）")
    ap.add_argument("--set", dest="override", action="append", default=[], help="点号键覆写")
    ap.add_argument("--early-stop", type=int, default=None, help="早停耐心（eval loss）")
    ap.add_argument("--no-early-stop", action="store_true", help="关掉早停")
