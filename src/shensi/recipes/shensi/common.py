"""预训练各 stage 共用：路径、配置、FlagScale 命令与语料准备。

`env_paths()` 认 `SHENSI_ROOT`（shensi 检出根）与 `SHENSI_FS`（filestorage）；
上游库取 `$SHENSI_ROOT/3rdparty/common/**`，平铺布局（老工作区）也认。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记/补齐第三方要的东西

STAGE0 = Path(__file__).resolve().parent
RECIPES = STAGE0.parent
MCORE = "Megatron-LM-FL"


def open_text(path):
    """按扩展名选解压再打开（zstandard 的地方也用上下文管理器）。"""
    if str(path).endswith((".zst", ".zstd")):
        import zstandard

        return zstandard.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def count_lines(path) -> int:
    """非空行数。"""
    with open_text(path) as fh:
        return sum(1 for line in fh if line.strip())


def env_paths() -> dict:
    root = Path(os.environ.get("SHENSI_ROOT", "/root/work/shensi"))
    fs = Path(os.environ.get("SHENSI_FS", "/root/work/filestorage"))
    # 上游库在 shensi 检出的 3rdparty/common/ 下（子模块）；平铺布局（老工作区）也认
    third = root / "3rdparty" / "common"
    if not (third / "FlagScale").is_dir():
        third = root
    return {
        "root": root,
        "flagscale": third / "FlagScale",
        "mcore": third / "Megatron-LM-FL",
        "tokenizer": os.environ.get("SHENSI_TOKENIZER", str(fs / "models/DeepSeek-V4-Flash-0731")),
        "pre": fs / "datasets/llm/pre-training",
        "post": fs / "datasets/llm/post-training",
        "data": fs / "shensi/data",
        "ckpt": fs / "shensi/ckpt",
        "logs": fs / "shensi/logs",
        "runs": fs / "shensi/runs",
    }


def stage_dirs(stage: str) -> tuple[Path, Path]:
    d = STAGE0 / stage
    if not d.is_dir():  # stage1_sft / stage2_rl / stage3_eval 与 stage0_pretrain 平级
        d = RECIPES / stage
    if not d.is_dir():
        raise SystemExit(f"找不到 stage 目录：{stage}")
    return d, d / "config"


def load_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf

    # 不在这里解析 ${...}：要先合并 default 与 profile，profile 覆盖的 experiment.* 才能
    # 影响到 train 段里的插值（例 save_interval: ${experiment.save_steps}）。合并后统一 resolve。
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def resolve_cfg(cfg: dict) -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def load_blend(data_dir: Path) -> list | None:
    p = Path(data_dir) / "blend.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_config(
    stage: str, profile: str, override: list[str], data_dir: Path, tokens: int | None = None
) -> dict:
    sdir, cdir = stage_dirs(stage)
    cfg = _stage_cfg(cdir)
    if profile not in ("default", "", None):
        prof = cdir / f"{profile}.yaml"
        if not prof.is_file():
            raise SystemExit(f"没有这个 profile：{prof}")
        cfg = _deep_merge(cfg, load_yaml(prof))
    paths = env_paths()
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("exp_dir", str(paths["runs"] / stage / profile))
    train = cfg.setdefault("train", {})
    model = train.setdefault("model", {})
    model["tokenizer_model"] = model.get("tokenizer_model") or paths["tokenizer"]
    data = train.setdefault("data", {})
    data["tokenizer_model"] = paths["tokenizer"]
    ckpt = train.setdefault("system", {}).setdefault("checkpoint", {})
    ckpt.setdefault("save", str(paths["ckpt"] / stage / profile))
    blend = load_blend(data_dir)
    if blend:
        data["data_path"] = blend
    for item in override or []:
        key, _, val = item.partition("=")
        _set_dotted(cfg, key, _coerce(val))
    cfg = resolve_cfg(cfg)  # 合并 + 覆写完成后再解析 ${...}
    train = cfg["train"]
    model = train["model"]
    if tokens:
        gb = int(model.get("global_batch_size") or train.get("global_batch_size") or 0)
        seq = int(model.get("seq_length") or 0)
        if gb > 0 and seq > 0:
            iters = max(1, int(tokens) // (gb * seq))
            model["train_iters"] = iters
            print(
                f"[recipe] token 预算 {tokens / 1e9:.1f}B → train_iters={iters}（gb={gb} × seq={seq}）"
            )
    if not train["system"]["checkpoint"].get("save"):
        raise SystemExit(
            "[recipe] 没有 checkpoint.save：检查 config 里 checkpoint 是否写在 train.system 下"
        )
    return cfg


# 带 base: 的档以基座为底按 (name, config) 叠加；覆盖项没写 config 时按 name 兜底，合并后校验 weight
def load_blend_spec(path: Path) -> dict:
    raw = load_yaml(path)
    base = raw.pop("base", None)
    if not base:
        return raw
    base_spec = load_blend_spec((path.parent / base).resolve())
    over = {f"{d['name']}|{d.get('config', '')}": d for d in raw.get("datasets", [])}
    # 覆盖项没写 config 时按 name 兜底（同一数据集只有一个配置时才允许，避免悄悄盖错东西）
    by_name: dict = {}
    for d in base_spec["datasets"]:
        by_name.setdefault(d["name"], []).append(d)
    merged = []
    for d in base_spec["datasets"]:
        hit = over.pop(f"{d['name']}|{d.get('config', '')}", None)
        merged.append({**d, **hit} if hit else d)
    rest = list(over.values())
    for d in list(rest):
        if d.get("config", "") == "" and len(by_name.get(d["name"], [])) == 1:
            base_entry = by_name[d["name"]][0]
            merged[base_spec["datasets"].index(base_entry)] = {**base_entry, **d}
            rest.remove(d)
    merged += rest
    out = {**base_spec, **{k: v for k, v in raw.items() if k != "datasets"}, "datasets": merged}
    # 校验：除了显式 metadata-only 的，其它条目都得有 weight（否则就是 (name, config) 没对上 base）
    bad = [
        f"{d['name']}|{d.get('config', '')}"
        for d in out["datasets"]
        if d.get("weight") is None and d.get("mode") != "metadata-only"
    ]
    if bad:
        raise SystemExit(
            f"[recipe] 配比里这些条目没有 weight（多半是 (name, config) 跟 base 对不上）：{bad}"
        )
    return out


def _stage_cfg(cdir: Path) -> dict:
    raw = load_yaml(cdir / "default.yaml")
    base = raw.pop("base", None)
    return _deep_merge(_stage_cfg((cdir / base).resolve()) if base else {}, raw)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _coerce(val: str):
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    return val


def write_run_dir(cfg: dict, stage: str, profile: str) -> Path:
    from omegaconf import OmegaConf

    run_dir = Path(cfg["experiment"]["exp_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    return run_dir


def flagscale_cmd(run_dir: Path, stage_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "flagscale.run",
        "--config-path",
        str(run_dir),
        "--config-name",
        "config",
    ]


def wait_for_run(run_dir, poll: int = 15) -> None:
    """FlagScale 是「提交即返回」；串接多阶段时用 --wait 等本机这次 run 真跑完。"""
    out = Path(run_dir) / "logs/host_0_localhost.output"
    while True:
        if out.is_file():
            text = out.read_text(encoding="utf-8", errors="ignore")
            if "after training is done" in text:
                print("[recipe] 本次 run 已收尾")
                return
            if "Traceback" in text:
                print(f"[recipe] 本次 run 有 Traceback，见 {out}")
                return
        time.sleep(poll)


def run(cfg: dict, stage: str, profile: str, dry_run: bool, wait: bool = False) -> int:
    paths = env_paths()
    run_dir = write_run_dir(cfg, stage, profile)
    cmd = flagscale_cmd(run_dir, run_dir)
    print(f"[recipe] 配置已写入 {run_dir / 'config.yaml'}")
    print(f"[recipe] cwd={paths['flagscale']}")
    print("[recipe] 命令：\n  " + " ".join(cmd))
    if dry_run:
        return 0
    env = dict(os.environ)
    code = subprocess.call(cmd, cwd=paths["root"], env=env)
    if wait and code == 0:
        wait_for_run(run_dir)
    return code


# 盯 exp_dir 日志里的指标，超耐心就调 early_stop.py 收尾
def watch(
    cfg: dict,
    patience: int,
    metric: str = "validation loss",
    mode: str = "min",
    min_delta: float = 1e-4,
    max_wait: float = 0.0,
) -> int:
    import subprocess as sp

    log = Path(cfg["experiment"]["exp_dir"]) / "logs/host_0_localhost.output"
    cmd = [
        sys.executable,
        str(RECIPES / "early_stop.py"),
        "--log",
        str(log),
        "--metric",
        metric,
        "--mode",
        mode,
        "--patience",
        str(patience),
        "--min-delta",
        str(min_delta),
    ]
    if max_wait:
        cmd += ["--max-wait", str(max_wait)]
    print("[recipe] 早停看门狗：", " ".join(cmd[-8:]))
    return sp.call(cmd)


def smoke(tiny_scale: str = "tiny") -> int:
    paths = env_paths()
    cmd = [
        sys.executable,
        "-m",
        "flagscale.run",
        "--config-path",
        str(Path(__file__).resolve().parents[2] / "flagscale_ext/examples/shensi/conf"),
        "--config-name",
        "train",
        f"train={tiny_scale}",
    ]
    return subprocess.call(cmd, cwd=paths["root"], env=dict(os.environ))


TEXT_KEYS = ("text", "content", "raw_content", "document", "code", "response")


def _iter_records(files: list[Path], limit: int | None, text_key: str | None, min_chars: int = 0):
    n = 0
    for f in files:
        name = f.name.lower()
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(f).iter_batches(batch_size=512):
                cols = batch.schema.names
                key = text_key or next((k for k in TEXT_KEYS if k in cols), None)
                if key is None:
                    raise SystemExit(f"{f}: 找不到文本列（列={cols}），请在 blend 里指定 text_key")
                for row in batch.column(cols.index(key)).to_pylist():
                    if row and len(row) >= min_chars:
                        yield row
                        n += 1
                        if limit and n >= limit:
                            return
        elif name.endswith((".jsonl", ".json", ".jsonl.zst", ".json.zst")):
            with open_text(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = text_key or next((k for k in TEXT_KEYS if k in obj), None)
                    if key is None:
                        raise SystemExit(
                            f"{f}: 找不到文本列（键={list(obj)}），请在 blend 里指定 text_key"
                        )
                    text = obj[key]
                    if text and len(text) >= min_chars:
                        yield text
                        n += 1
                        if limit and n >= limit:
                            return
        else:
            print(f"  [跳过] 未识别的格式：{f.name}")


def _dataset_files(root: Path, spec: dict) -> list[Path]:
    d = root / spec["name"]
    files: list[Path] = []
    for sub in sorted(d.glob("**/*")):
        if sub.is_dir() or sub.suffix not in (".parquet", ".jsonl", ".json", ".zst"):
            continue
        rel = str(sub.relative_to(d))
        if spec.get("config") and spec["config"] not in rel:
            continue
        if spec.get("split") and spec["split"] not in rel:
            continue
        files.append(sub)
    return files


def discover(root: Path, blend_spec: dict, out: Path | None = None) -> dict:
    report = {}
    seen = set()
    skipped = []
    for spec in blend_spec["datasets"]:
        files = _dataset_files(root, spec)
        seen.add(spec["name"])
        mode = spec.get("mode")
        info = {"files": len(files), "weight": spec.get("weight")}
        if mode == "codev3":
            done = _codev3_jsonl(spec, out).is_file() if out else False
            info["状态"] = (
                "已落地（--codev3 产物在 out 里）" if done else "待落地：跑 data_prep.py --codev3"
            )
            if not done:
                skipped.append(spec["name"])
        elif mode == "metadata-only":
            info["状态"] = "仅元数据（需先按 metadata 回捞文本）→ 不进 blend"
            skipped.append(spec["name"])
        if files:
            try:
                import pyarrow.parquet as pq

                if files[0].suffix == ".parquet":
                    info["columns"] = pq.ParquetFile(files[0]).schema_arrow.names
                else:
                    with open(files[0], encoding="utf-8", errors="ignore") as fh:
                        info["columns"] = list(json.loads(fh.readline()).keys())
            except Exception as exc:  # noqa: BLE001
                info["columns"] = f"<读取失败 {type(exc).__name__}: {exc}>"
            if mode in ("metadata-only", "codev3"):
                info["抽样条数"] = "—（只有元数据，没有文本列可抽）"
            else:
                sample = list(
                    _iter_records(files[:1], 64, spec.get("text_key"), spec.get("min_chars", 0))
                )
                chars = sum(len(s) for s in sample)
                info["抽样条数"] = len(sample)
                info["样本文本 平均字符"] = int(chars / max(len(sample), 1))
        report[spec["name"]] = info
        mark = (
            "❌"
            if mode == "metadata-only"
            or (mode == "codev3" and not info.get("状态", "").startswith("已落地"))
            else "✅"
        )
        print(
            f"  {mark} {spec['name']:<50} 文件 {info['files']:<5} {info.get('columns')}"
            + (f"  [{info['状态']}]" if "状态" in info else "")
        )
    if skipped:
        print(f"  ── 排除（仅元数据/待落地）：{skipped}")
    extra = [p.name for p in root.glob("*") if p.is_dir() and p.name not in seen]
    if extra:
        print(f"  [提示] 目录里还有没进 blend 的：{extra}")
    return report


# 不用 mcore 的 tools/preprocess_data.py：它关掉 --split-sentences 时 sentence_lens 是 None 会 TypeError
def build_indexed(jsonl: Path, prefix: Path, tokenizer: str, batch_docs: int = 2000) -> int:
    import numpy as np
    from megatron.core.datasets import indexed_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    eod = tok.eos_token_id if tok.eos_token_id is not None else 0
    builder = indexed_dataset.IndexedDatasetBuilder(
        f"{prefix}_text_document.bin",
        dtype=indexed_dataset.DType.optimal_dtype(len(tok)),
    )
    n = 0

    def flush(texts: list[str]) -> None:
        nonlocal n
        if not texts:
            return
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            arr = np.array([*ids, eod], dtype=np.int32)
            builder.add_document(arr, [len(arr)])
            n += 1
        texts.clear()

    buf: list[str] = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                buf.append(json.loads(line)["text"])
            if len(buf) >= batch_docs:
                flush(buf)
                print(f"  … 已编码 {n} 篇", flush=True)
    flush(buf)
    builder.finalize(f"{prefix}_text_document.idx")
    return n


# 缺 weight 直接报错：mcore 的 blend 权重是相对值，没权重等于没进配比
def _blend_entry(spec: dict, jsonl: Path, out: Path, tokenizer: str, n_hint: int | None = None):
    if "weight" not in spec:
        raise SystemExit(f"[data_prep] {spec['name']}: blend 里缺 weight —— 要进配比就得给权重")
    n = n_hint if n_hint is not None else count_lines(jsonl)
    if n == 0:
        print(
            f"[data_prep] {spec['name']}: 0 条（min_chars={spec.get('min_chars')}？）→ 不进 blend"
        )
        return None
    prefix = out / jsonl.stem
    doc = prefix.with_name(prefix.name + "_text_document")
    if not (Path(str(doc) + ".bin").is_file() and Path(str(doc) + ".idx").is_file()):
        print(f"[data_prep] {spec['name']}: {n} 篇 → 编码成 {doc.name}.bin/.idx ...")
        n_docs = build_indexed(jsonl, prefix, tokenizer)
        print(f"[data_prep] {spec['name']}: 编码完成，{n_docs} 篇")
    else:
        print(f"[data_prep] {spec['name']}: 已有 .bin/.idx，跳过（{n} 篇）")
    return [str(spec["weight"]), str(doc)]


def _codev3_jsonl(spec: dict, out: Path) -> Path:
    tag = spec["name"] + ("__" + spec["config"] if spec.get("config") else "")
    return out / f"{tag}.jsonl"


# metadata-only 一律跳过；mode: codev3 的条目看 --codev3 有没有落地产物
def prepare(
    blend_spec: dict,
    root: Path,
    out: Path,
    tokenizer: str,
    limit: int | None,
    workers: int,
    include_metadata_only: bool = False,
    only: str | None = None,
    skip_missing: bool = False,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    blend: list = []
    for spec in blend_spec["datasets"]:
        if only and only not in spec["name"]:
            continue
        if spec.get("mode") == "codev3":
            jsonl = _codev3_jsonl(spec, out)
            if jsonl.is_file():
                entry = _blend_entry(spec, jsonl, out, tokenizer)
                if entry:
                    blend += entry
                continue
            msg = (
                f"{spec['name']}: 只有元数据且还没落地 —— 先跑 `data_prep.py --codev3`"
                "（按 v1/v2 元数据分类后回 GitHub 取文本，见 stage0_pretrain/codev3.py）"
            )
            if include_metadata_only:
                raise SystemExit(msg)
            print(f"[data_prep] 跳过：{msg}")
            continue
        if spec.get("mode") == "metadata-only":
            msg = (
                f"{spec['name']}: 只有元数据（列见 --discover），需要先按 metadata 回捞文本再预训练"
            )
            if include_metadata_only:
                raise SystemExit(msg)
            print(f"[data_prep] 跳过：{msg}")
            continue
        files = _dataset_files(root, spec)
        if not files:
            msg = f"{spec['name']}: 在 {root} 下没找到文件"
            if skip_missing:
                print(f"[data_prep] 跳过（--skip-missing）：{msg}")
                continue
            raise SystemExit(f"{msg}，先跑 --discover 看看")
        tag = spec["name"] + ("__" + spec["config"] if spec.get("config") else "")
        jsonl = out / f"{tag}.jsonl"
        n = 0
        with open(jsonl, "w", encoding="utf-8") as fh:
            for text in _iter_records(files, limit, spec.get("text_key"), spec.get("min_chars", 0)):
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
        entry = _blend_entry(spec, jsonl, out, tokenizer, n_hint=n)
        if entry:
            blend += entry
    with open(out / "blend.json", "w", encoding="utf-8") as f:
        json.dump(blend, f, ensure_ascii=False, indent=1)
    print(f"[data_prep] blend.json 已写入 {out / 'blend.json'}")
    return out / "blend.json"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--root", default=None, help="语料根目录，默认 $SHENSI_FS/datasets/llm/pre-training"
    )
    ap.add_argument("--out", default=None, help="预处理产物目录，默认 $SHENSI_FS/shensi/data")
    ap.add_argument("--tokenizer", default=None, help="tokenizer 目录，默认 $SHENSI_TOKENIZER")
    ap.add_argument("--limit", type=int, default=None, help="每个数据集最多取多少条（冒烟用）")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", default=None, help="只处理名字里含该子串的数据集（冒烟用）")
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="语料还没齐时跳过缺的数据集（默认是遇到缺的就停下报错）",
    )
