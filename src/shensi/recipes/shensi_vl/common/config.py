"""配置档合并（对齐 shensi/common/common.py 的语义，但服务本配方的 HF 侧训练循环）。

约定与 shensi 一致：`config/default.yaml`（可带 `base:` 指向本 stage 或别的 config 目录）→
`config/<profile>.yaml` → `--set k=v`（点号键，显式覆写永远最后生效）；合并完再统一解析 `${...}`。
"""

from __future__ import annotations

from pathlib import Path

from shensi.recipes.shensi.common import common as base
from shensi.recipes.shensi_vl.common import paths


def load_yaml(path: Path) -> dict:
    return base.load_yaml(Path(path))


def _deep_merge(base_d: dict, over: dict) -> dict:
    return base._deep_merge(base_d, over)


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _load_cfg_chain(path: Path, seen: tuple = ()) -> dict:
    """读一份配置并解析它自己的 `base:` 链（base 可指文件或目录→其 default.yaml）。"""
    path = Path(path).resolve()
    if path in seen:
        raise SystemExit(f"[shensi_vl] 配置 base 链出现环：{path}")
    raw = load_yaml(path)
    base_ref = raw.pop("base", None)
    if not base_ref:
        return raw
    target = (path.parent / base_ref).resolve()
    src = target / "default.yaml" if target.is_dir() else target
    return _deep_merge(_load_cfg_chain(src, seen + (path,)), raw)


def build_config(stage: str, profile: str | None, override: list[str]) -> dict:
    """合并 stage 配置：default（base 链）→ profile（含其 base 链）→ --set；注入路径后解析插值。"""
    here = paths.stage_dirs(stage) / "config"
    cfg = _load_cfg_chain(here / "default.yaml")
    if profile and profile not in ("default",):
        prof = Path(profile)
        if prof.suffix != ".yaml":
            prof = here / f"{profile}.yaml"
        elif not prof.is_absolute():
            prof = Path(paths.stage_dirs(stage)) / prof
        if not prof.is_file():
            raise SystemExit(f"[shensi_vl] 没有这个 profile：{prof}")
        cfg = _deep_merge(cfg, _load_cfg_chain(prof))
    for item in override or []:
        key, _, val = item.partition("=")
        _set_dotted(cfg, key, base._coerce(val))

    p = paths.env_paths()
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault(
        "exp_dir", str(Path(p["runs"]) / stage / (profile or "default"))
    )
    train = cfg.setdefault("train", {})
    model = train.setdefault("model", {})
    model.setdefault("tokenizer_dir", p["vl_tokenizer"])  # 扩展版（base 同 DSV4F）
    model.setdefault("llm_path", p["llm"])
    model.setdefault("vision_path", p["vision"])
    model.setdefault("processor_path", p["processor"])
    return base.resolve_cfg(cfg)


def write_run_dir(cfg: dict, stage: str, profile: str) -> Path:
    """把最终配置落到 exp_dir/config.yaml（run.sh 由训练循环不打——HF 侧启动是单进程 python）。"""
    import json

    run_dir = Path(cfg["experiment"]["exp_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(f"[shensi_vl] 配置已写入 {run_dir / 'config.yaml'}")
    return run_dir
