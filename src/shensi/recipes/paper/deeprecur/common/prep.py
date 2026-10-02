"""数据准备公共核：blend 配比 → 采集（HF hub 或本地）→ messages jsonl。

配比 json（``config/data_prep/data_blend_*.json``）每个条目：

```json
{"name": "LLaVA-Pretrain", "count": 558390, "hf": "liuhaotian/LLaVA-Pretrain",
 "kind": "llava_pretrain", "image_root": null, "fields": {}}
```

- ``kind``：``llava_pretrain``（LCS-558k，parquet 内嵌图像）｜``llava_instruct``
  （LLaVA-Instruct-150K 的 json，图像在 COCO/VG 本地目录）｜``vl_qa``
  （图像内嵌的学术 VQA 集，字段别名由 ``fields`` 给）｜``sharegpt``（纯文本对话）。
- 采集顺序：已有本地目录（``$SHENSI_FS/datasets/llm/{pre,post}-training/<name>``）
  → HF hub snapshot（``--offline`` 时不拉）→ 都没有就显式报错，不静默跳过。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.config import dataprep_config, profile_from_args
from shensi.recipes.paper.deeprecur.common.paths import env_paths, stage_dirs

#: 已下载 HF 数据集的落位根（按 shensi 的 pre/post 约定挂原始数据）
HF_ROOT_PRE = "datasets/llm/pre-training"
HF_ROOT_POST = "datasets/llm/post-training"


def _hf_root(kind_default: str) -> Path:
    paths = env_paths()
    fs = Path(paths["data"]).parents[2]  # <FS>/shensi/data/deeprecur → <FS>
    return fs / kind_default


def dataset_dir(entry: dict, stage: str) -> Path:
    """原始数据目录：``<FS>/datasets/llm/{pre,post}-training/<name>``。"""
    post = stage != "stage0_pt"
    return _hf_root(HF_ROOT_POST if post else HF_ROOT_PRE) / entry["name"]


def ensure_raw(entry: dict, stage: str, offline: bool) -> Path:
    """定位或拉取原始数据，返回目录。"""
    directory = dataset_dir(entry, stage)
    if directory.is_dir() and any(directory.iterdir()):
        return directory
    if offline or not entry.get("hf"):
        raise SystemExit(
            f"[deeprecur] 数据集 {entry['name']} 本地缺失：{directory}"
            f"（hf={entry.get('hf') or '未配置'}；离线模式不拉网，请先放置数据）"
        )
    directory.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=entry["hf"], repo_type="dataset", local_dir=str(directory)
    )
    return directory


def _rows_llava_pretrain(directory: Path, fields: dict) -> list[dict]:
    """LCS-558k：parquet（id / image 内嵌 / conversations）→ 单轮 caption 对话。"""
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
    """LLaVA-Instruct-150K：json 数组，image 字段是相对 ``image_root`` 的文件名。"""
    image_root = fields.get("image_root")
    if not image_root:
        raise SystemExit("[deeprecur] llava_instruct 条目缺 fields.image_root（COCO/VG 图像目录）")
    root = Path(image_root)
    if not root.is_absolute():
        root = _hf_root(HF_ROOT_POST) / root  # 相对路径按 post-training 根解析
    rows: list[dict] = []
    for path in sorted(directory.glob("**/*.json")):
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            rel = record.get("image")
            if not rel:
                continue
            rows.append(
                {
                    "id": f"instruct-{len(rows):08d}",
                    "images": [str(root / rel)],
                    "messages": _convert_conversation(record["conversations"]),
                }
            )
    return rows


def _convert_conversation(turns: list[dict]) -> list[dict]:
    """LLaVA 对话格式（from/value，"<image>\\n" 前缀）→ messages。"""
    messages = []
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


def _rows_sharegpt(directory: Path, fields: dict) -> list[dict]:
    """纯文本对话集（ShareGPT 一类：json 数组，conversations from human/gpt），无图像。"""
    rows: list[dict] = []
    for path in sorted(directory.glob("**/*.json")):
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
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


_VQA_FIELD_ALIASES = {
    "question": ["question", "problem", "query", "conversations"],
    "answer": ["answer", "answers", "response", "solution"],
}


def _rows_vl_qa(directory: Path, fields: dict) -> list[dict]:
    """图像内嵌的学术 VQA 集（lmms-lab / HuggingFaceM4 一系的 parquet）。

    ``fields.task_prompt`` 是论文 Table 9 的任务提示词，按 DeepStack 的做法拼在问题后。
    """
    import pandas as pd

    aliases = {
        role: fields.get(role, []) + _VQA_FIELD_ALIASES[role] for role in ("question", "answer")
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
                            "content": [{"type": "image"}, {"type": "text", "text": str(question)}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": str(answer)}],
                        },
                    ],
                }
            )
    return rows


def _first_field(record, aliases: list[str]):
    for alias in aliases:
        if alias in record and record[alias] is not None:
            return record[alias]
    return None


_CONVERTERS = {
    "llava_pretrain": _rows_llava_pretrain,
    "llava_instruct": _rows_llava_instruct,
    "vl_qa": _rows_vl_qa,
    "sharegpt": _rows_sharegpt,
}


def prepare(
    *,
    stage: str,
    blend: str,
    limit: int | None,
    val_frac: float,
    only: str | None,
    offline: bool,
    out_dir: Path | None,
) -> int:
    """按配比采集并转换，写 ``<stage>_train.jsonl`` / ``<stage>_val.jsonl``，返回总条数。"""
    blend_path = stage_dirs(stage) / "config" / "data_prep" / blend
    spec = json.loads(Path(blend_path).read_text(encoding="utf-8"))
    entries = [
        item
        for item in spec.get("datasets", [])
        if not only or only in item["name"]
    ]
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
        count = entry.get("count")
        directory = ensure_raw(entry, stage, offline)
        subset = converter(directory, entry.get("fields") or {})
        rng.shuffle(subset)
        if count:
            subset = subset[:count]
        if not subset:
            raise SystemExit(f"[deeprecur] 数据集 {entry['name']} 转换出 0 条，检查 kind/fields")
        print(f"[deeprecur] {entry['name']}: {len(subset)} 条（论文口径 {count or '—'}）")
        rows.extend(subset)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    n_val = max(1, int(len(rows) * val_frac))
    target = Path(out_dir or env_paths()["data"] / stage)
    target.mkdir(parents=True, exist_ok=True)
    for name, subset in ((f"{stage}_val.jsonl", rows[:n_val]), (f"{stage}_train.jsonl", rows[n_val:])):
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in subset) + "\n"
        (target / name).write_text(text, encoding="utf-8")
    print(f"[deeprecur] {target}/{stage}_train.jsonl（{len(rows) - n_val}）+ val（{n_val}）")
    return len(rows)


def discover(stage: str, blend: str) -> None:
    """报告配比里每个数据集的本地落位与论文口径条数。"""
    blend_path = stage_dirs(stage) / "config" / "data_prep" / blend
    spec = json.loads(Path(blend_path).read_text(encoding="utf-8"))
    print(f"[deeprecur] 配比 {blend}：{spec.get('note', '')}")
    for entry in spec.get("datasets", []):
        directory = dataset_dir(entry, stage)
        state = "本地已有" if directory.is_dir() and any(directory.iterdir()) else "本地缺失"
        print(
            f"  {entry['name']:<24} 论文 {entry.get('count') or '—':>7}  "
            f"hf={entry.get('hf') or '—':<38} {state}  {directory}"
        )


def prep_main(stage: str, here: Path, default_blend: str, argv: list[str] | None = None) -> int:
    """命令行入口（stage 的 data_prep.py 调它）。"""
    parser = argparse.ArgumentParser(description=f"{stage} 语料准备（messages jsonl）")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--discover", action="store_true", help="只报告数据集落位，不转换")
    parser.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    parser.add_argument("--blend", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--offline", action="store_true", help="只用本地已有数据，不拉 HF")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if not args.prepare and not args.discover:
        parser.print_help()
        return 1
    name = profile_from_args(args.config, "default")
    cfg = dataprep_config(here / "config" / "data_prep" / f"{name}.yaml")
    blend = args.blend or cfg.get("blend") or default_blend
    if args.discover:
        discover(stage, blend)
        return 0
    prepare(
        stage=stage,
        blend=blend,
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        val_frac=args.val_frac if args.val_frac is not None else (cfg.get("val_frac") or 0.02),
        only=args.only if args.only is not None else cfg.get("only"),
        offline=args.offline,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    return 0
