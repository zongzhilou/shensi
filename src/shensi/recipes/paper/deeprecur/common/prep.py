"""语料准备：配比（blend）→ 采集（HF hub / 本地）→ messages jsonl。

配比条目字段：

    name    数据集名，同时是落位目录名（`<FS>/datasets/llm/{pre,post}-training/<name>`）
    path    数据源；`hf://org/dataset` 走 hub 采集，null 表示需手动放置
    subset  HF 数据集内的子目录/配置名（可选）
    kind    转换器：llava_pretrain | llava_instruct | vl_qa | sharegpt
    count   该数据集条数（登记用；采集后按此截断）
    fields  转换器参数（如 llava_instruct 的 image_root、vl_qa 的字段别名与 task_prompt）

产物为一行一个样本的 messages jsonl：

    {"id": "...", "images": ["images/x.jpg"],
     "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
                  {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import dataprep_config, profile_from_args
from .paths import env_paths

#: 原始数据落位根（按 pre/post 两级区分训练前后段）
HF_ROOT = {"stage0_pt": "datasets/llm/pre-training", "stage1_sft": "datasets/llm/post-training"}

_VQA_FIELD_ALIASES = {
    "question": ("question", "problem", "query"),
    "answer": ("answer", "answers", "response", "solution"),
}


def _fs_root() -> Path:
    """文件系统根：`<FS>/shensi/data/...` 往上两级。"""
    return Path(env_paths()["data"]).parents[2]


def dataset_dir(entry: dict, stage: str) -> Path:
    """数据集落位目录。"""
    return _fs_root() / HF_ROOT.get(stage, HF_ROOT["stage1_sft"]) / entry["name"]


def ensure_raw(entry: dict, stage: str, offline: bool) -> Path:
    """定位原始数据；本地缺失且允许联网时从 `path` 采集。"""
    directory = dataset_dir(entry, stage)
    if directory.is_dir() and any(directory.iterdir()):
        return directory
    source = entry.get("path")
    if offline or not source:
        raise SystemExit(
            f"[deeprecur] 数据集 {entry['name']} 本地缺失：{directory}"
            f"（path={source or '未配置'}；离线模式不拉网，请先放置数据）"
        )
    from huggingface_hub import snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(source).removeprefix("hf://"),
        repo_type="dataset",
        local_dir=str(directory),
    )
    return directory


def _convert_conversation(turns: list[dict]) -> list[dict]:
    """LLaVA 对话格式（from/value，`<image>` 前缀）→ messages；相邻同角色合并。"""
    messages: list[dict] = []
    for turn in turns:
        role = "user" if turn["from"] in ("human", "user") else "assistant"
        content: list[dict] = []
        value = str(turn["value"])
        while value.startswith("<image>"):
            content.append({"type": "image"})
            value = value[len("<image>") :].lstrip("\n")
        content.append({"type": "text", "text": value.strip()})
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": role, "content": content})
    return messages


def _rows_llava_pretrain(directory: Path, fields: dict) -> list[dict]:
    """LCS-558k：parquet（id / 内嵌图像 / conversations）→ 单轮 caption 对话。"""
    import pandas as pd

    rows: list[dict] = []
    images_dir = directory / "images"
    images_dir.mkdir(exist_ok=True)
    for path in sorted(directory.glob("**/*.parquet")):
        frame = pd.read_parquet(path)
        for _, record in frame.iterrows():
            image_cell = record.get("image")
            if image_cell is None:
                continue
            image_id = str(record.get("id") or f"img{len(rows):08d}")
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.is_file():
                image_path.write_bytes(image_cell["bytes"])
            turns = list(record["conversations"])
            if len(turns) < 2:
                continue
            user_text = str(turns[0]["value"]).replace("<image>\n", "").strip()
            rows.append(
                {
                    "id": image_id,
                    "images": [str(image_path)],
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": str(turns[1]["value"]).strip()}],
                        },
                    ],
                }
            )
    return rows


def _rows_llava_instruct(directory: Path, fields: dict) -> list[dict]:
    """LLaVA-Instruct-150K：json 数组，`image` 是相对 `fields.image_root` 的文件名。"""
    image_root = fields.get("image_root")
    if not image_root:
        raise SystemExit("[deeprecur] llava_instruct 条目缺 fields.image_root（图像目录）")
    root = Path(image_root)
    if not root.is_absolute():
        root = _fs_root() / HF_ROOT["stage1_sft"] / root
    rows: list[dict] = []
    for path in sorted(directory.glob("**/*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            if not record.get("image"):
                continue
            rows.append(
                {
                    "id": f"instruct-{len(rows):08d}",
                    "images": [str(root / record["image"])],
                    "messages": _convert_conversation(record["conversations"]),
                }
            )
    return rows


def _first_field(record, aliases: tuple[str, ...]):
    for alias in aliases:
        if record.get(alias) is not None:
            return record[alias]
    return None


def _rows_vl_qa(directory: Path, fields: dict) -> list[dict]:
    """图像内嵌的学术 VQA 集（parquet）；字段别名与 `task_prompt` 由 `fields` 给。"""
    import pandas as pd

    aliases = {
        role: tuple(fields.get(role) or ()) + _VQA_FIELD_ALIASES[role]
        for role in ("question", "answer")
    }
    prompt = fields.get("task_prompt")
    rows: list[dict] = []
    images_dir = directory / "images"
    images_dir.mkdir(exist_ok=True)
    for path in sorted(directory.glob("**/*.parquet")):
        frame = pd.read_parquet(path)
        for _, record in frame.iterrows():
            image_cell = record.get("image")
            if image_cell is None:
                continue
            raw = image_cell["bytes"] if isinstance(image_cell, dict) else image_cell
            image_id = f"{path.stem}-{len(rows):08d}"
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.is_file():
                image_path.write_bytes(raw)
            question = _first_field(record, aliases["question"])
            answer = _first_field(record, aliases["answer"])
            if question is None or answer is None:
                continue
            if isinstance(answer, list):
                answer = answer[0] if answer else None
            question = str(question)
            if prompt:
                question = f"{question} {prompt}"
            rows.append(
                {
                    "id": image_id,
                    "images": [str(image_path)],
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": question}],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
                    ],
                }
            )
    return rows


def _rows_sharegpt(directory: Path, fields: dict) -> list[dict]:
    """纯文本对话集（ShareGPT 一类：json 数组或 `{"data": [...]}`），无图像。"""
    rows: list[dict] = []
    for path in sorted(directory.glob("**/*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("data") or records.get("conversations") or []
        for record in records:
            turns = record.get("conversations") if isinstance(record, dict) else None
            if not turns:
                continue
            rows.append(
                {
                    "id": f"sharegpt-{len(rows):08d}",
                    "images": [],
                    "messages": _convert_conversation(turns),
                }
            )
    return rows


_CONVERTERS = {
    "llava_pretrain": _rows_llava_pretrain,
    "llava_instruct": _rows_llava_instruct,
    "vl_qa": _rows_vl_qa,
    "sharegpt": _rows_sharegpt,
}


def load_spec(blend_path: Path) -> dict:
    """读配比 json。"""
    return json.loads(Path(blend_path).read_text(encoding="utf-8"))


def prepare(
    *,
    stage: str,
    blend_path: Path,
    sample: int | None,
    valid_ratio: float,
    only: str | None,
    offline: bool,
    out_dir: Path | None,
) -> int:
    """按配比采集并转换，写 `<stage>_train.jsonl` / `<stage>_val.jsonl`，返回总条数。"""
    spec = load_spec(blend_path)
    entries = [item for item in spec.get("datasets", []) if not only or only in item["name"]]
    if not entries:
        raise SystemExit(f"[deeprecur] 配比里没有可用的数据集：{blend_path}")
    rng = random.Random(42)
    rows: list[dict] = []
    for entry in entries:
        converter = _CONVERTERS.get(entry.get("kind"))
        if converter is None:
            raise SystemExit(
                f"[deeprecur] 数据集 {entry['name']} 的 kind 未实现：{entry.get('kind')}"
                f"（已实现：{sorted(_CONVERTERS)}）"
            )
        directory = ensure_raw(entry, stage, offline)
        subset = converter(directory, entry.get("fields") or {})
        rng.shuffle(subset)
        count = entry.get("count")
        if count:
            subset = subset[:count]
        if not subset:
            raise SystemExit(f"[deeprecur] 数据集 {entry['name']} 转换出 0 条，检查 kind/fields")
        print(f"[deeprecur] {entry['name']}: {len(subset)} 条（配比登记 {count or '—'}）")
        rows.extend(subset)
    rng.shuffle(rows)
    if sample:
        rows = rows[:sample]
    n_val = max(1, int(len(rows) * valid_ratio))
    target = Path(out_dir or env_paths()["data"] / stage)
    target.mkdir(parents=True, exist_ok=True)
    for name, subset in (
        (f"{stage}_val.jsonl", rows[:n_val]),
        (f"{stage}_train.jsonl", rows[n_val:]),
    ):
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in subset) + "\n"
        (target / name).write_text(text, encoding="utf-8")
    print(f"[deeprecur] {target}/{stage}_train.jsonl（{len(rows) - n_val}）+ val（{n_val}）")
    return len(rows)


def discover(stage: str, blend_path: Path) -> None:
    """报告配比里每个数据集的落位与登记条数。"""
    spec = load_spec(blend_path)
    print(f"[deeprecur] 配比 {blend_path.name}：{spec.get('_note', '')}")
    for entry in spec.get("datasets", []):
        directory = dataset_dir(entry, stage)
        state = "本地已有" if directory.is_dir() and any(directory.iterdir()) else "本地缺失"
        print(
            f"  {entry['name']:<24} 登记 {entry.get('count') or '—':>7}  "
            f"path={entry.get('path') or '—':<38} {state}  {directory}"
        )


def prep_main(stage: str, here: Path, default_blend: str, argv: list[str] | None = None) -> int:
    """命令行入口（stage 的 data_prep.py 调它）。"""
    parser = argparse.ArgumentParser(description=f"{stage} 语料准备（messages jsonl）")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--discover", action="store_true", help="只报告数据集落位，不转换")
    parser.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    parser.add_argument("--blend", default=None, help="配比 json（默认取 data_prep 配置）")
    parser.add_argument("--sample", type=int, default=None, help="每数据集抽样上限")
    parser.add_argument("--valid-ratio", type=float, default=None)
    parser.add_argument("--only", default=None, help="只处理名字含该子串的数据集")
    parser.add_argument("--offline", action="store_true", help="只用本地已有数据，不拉 HF")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if not args.prepare and not args.discover:
        parser.print_help()
        return 1
    name = profile_from_args(args.config, "default")
    cfg = dataprep_config(here / "config" / "data_prep" / f"{name}.yaml")
    blend = args.blend or cfg.get("blend_path") or default_blend
    blend_path = Path(blend)
    if not blend_path.is_absolute():
        blend_path = here / "config" / "data_prep" / blend_path
    if args.discover:
        discover(stage, blend_path)
        return 0
    prepare(
        stage=stage,
        blend_path=blend_path,
        sample=args.sample if args.sample is not None else cfg.get("sample"),
        valid_ratio=(
            args.valid_ratio if args.valid_ratio is not None else (cfg.get("valid_ratio") or 0.02)
        ),
        only=args.only if args.only is not None else cfg.get("only"),
        offline=args.offline,
        out_dir=Path(args.out_dir) if args.out_dir else (
            Path(cfg["output_dir"]) if cfg.get("output_dir") else None
        ),
    )
    return 0
