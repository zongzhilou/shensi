"""GDAR 配方共用：路径、配置合并与启动。

与 `shensi.recipes.shensi.common` 同一套约定，差别有三：

1. stage 目录是本配方的 `pretrain/`（论文配方只有一个训练 stage）；
2. tokenizer 默认 **Qwen3**（`tokenizer/Qwen3-0.6B`，即论文口径"全实验统一 Qwen3
   tokenizer"），可用 `$SHENSI_GDAR_TOKENIZER` 覆盖——不是 shensi 配方的 DeepSeek-V4；
3. 训练入口是本配方 `train/train_gdar.py`（`train.launcher` 在本包内）。

合并/覆写/解析这些纯函数直接复用 shensi common（`_deep_merge` / `_coerce` /
`resolve_cfg` / blend 装载），不复制第二份。
"""

from __future__ import annotations

from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res.train import launcher
from shensi.recipes.shensi import common as base

RECIPE = Path(__file__).resolve().parent
STAGE = "pretrain"
STAGE_DIR = RECIPE / STAGE
CONFIG = STAGE_DIR / "config"

TOKENIZER_ENV = "SHENSI_GDAR_TOKENIZER"
TOKENIZER_DIR = RECIPE / "tokenizer" / "Qwen3-0.6B"


def env_paths() -> dict:
    """Shensi 的路径表 + 本配方的覆盖项（tokenizer / runs / data 子目录）。"""
    paths = base.env_paths()
    paths["tokenizer"] = __import__("os").environ.get(TOKENIZER_ENV) or str(TOKENIZER_DIR)
    paths["runs"] = str(Path(paths["runs"]) / "gated_delta_attn_res")
    # data_prep 产物目录：data/gated_delta_attn_res（不叫 data/pretrain，避免和主配方的
    # stage0_pretrain/stage1_pretrain 产物撞名）
    paths["data"] = str(Path(paths["data"]) / "gated_delta_attn_res")
    return paths


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


def build_config(
    profile: str, override: list[str], data_dir: Path, tokens: int | None = None
) -> dict:
    """读 `pretrain/config/{default,<profile>}.yaml` 并解析成一个可直接起训的配置。"""
    cfg = base.load_yaml(CONFIG / "default.yaml")
    if profile not in ("default", "", None):
        prof = CONFIG / f"{profile}.yaml"
        if not prof.is_file():
            raise SystemExit(f"[gdar] 没有这个 profile：{prof}")
        cfg = base._deep_merge(cfg, base.load_yaml(prof))
    paths = env_paths()
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("exp_dir", str(Path(paths["runs"]) / STAGE / profile))
    train = cfg.setdefault("train", {})
    model = train.setdefault("model", {})
    model["tokenizer_model"] = model.get("tokenizer_model") or paths["tokenizer"]
    data = train.setdefault("data", {})
    data["tokenizer_model"] = data.get("tokenizer_model") or paths["tokenizer"]
    tok = data.setdefault("tokenizer", {})
    tok.setdefault("tokenizer_type", "HuggingFaceTokenizer")
    tok.setdefault("tokenizer_model", paths["tokenizer"])
    ckpt = train.setdefault("system", {}).setdefault("checkpoint", {})
    ckpt.setdefault("save", str(Path(paths["ckpt"]) / "gated_delta_attn_res" / profile))
    blend = load_blend(data_dir)
    if blend:
        data["data_path"] = blend
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
                f"[gdar] token 预算 {tokens / 1e9:.1f}B → train_iters={iters}（gb={gb} × seq={seq}）"
            )
    if not train["system"]["checkpoint"].get("save"):
        raise SystemExit("[gdar] 没有 checkpoint.save")
    return cfg


def write_run_dir(cfg: dict) -> Path:
    run_dir = Path(cfg["experiment"]["exp_dir"])
    run_dir = launcher.write_run_dir(cfg, run_dir)
    print(f"[gdar] 配置与命令已写入 {run_dir}（config.yaml / run.sh）")
    return run_dir


def run(cfg: dict, dry_run: bool) -> int:
    """起一次训练：写 run 目录 → torchrun → 前台等返回码。"""
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir, dry_run=dry_run)


def smoke_config(profile: str = "tiny", override: list[str] | None = None) -> dict:
    """读 `pretrain/config/{profile}.yaml`（mock 数据档）并解析成可直接起训的配置。"""
    path = CONFIG / f"{profile}.yaml"
    if not path.is_file():
        raise SystemExit(f"[gdar] 没有这个冒烟档：{path}")
    cfg = load_yaml(path)
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("exp_dir", str(Path(env_paths()["runs"]) / f"smoke_{profile}"))
    for item in override or []:
        key, _, val = item.partition("=")
        _set_dotted(cfg, key, _coerce(val))
    return resolve_cfg(cfg)


def smoke(profile: str = "tiny", override: list[str] | None = None) -> int:
    """冒烟：tiny 几何 + mock 数据 + 5 步（不碰真实语料，几何见 config/tiny.yaml）。"""
    cfg = smoke_config(profile, override)
    print("[gdar] 冒烟档：tiny 几何 / mock 数据 / 5 步（配置见 pretrain/config/tiny.yaml）")
    run_dir = write_run_dir(cfg)
    return launcher.launch(cfg, run_dir)


def watch(cfg: dict, patience: int, **kwargs) -> int:
    """早停看门狗（复用 shensi 的 early_stop.py）。"""
    return base.watch(cfg, patience, **kwargs)
