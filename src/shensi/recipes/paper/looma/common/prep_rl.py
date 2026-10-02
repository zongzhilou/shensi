"""RL 语料准备：prompts 转成 verl 需要的 train / val parquet。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shensi.recipes.paper.looma.common import (
    dataprep_config,
    env_paths,
    load_blend_spec,
    profile_from_args,
    stage_dirs,
)

PROMPT_KEYS = ("prompt", "question", "problem", "instruction", "messages")

ANSWER_KEYS = ("answer", "ground_truth", "solution", "label", "target")


def _source_dirs(root: Path, blend_spec: dict | None) -> list[Path]:
    if not blend_spec:
        return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out = []
    for spec in blend_spec.get("datasets", []):
        name, sub = spec["name"], spec.get("config")
        directory = root / name / sub if sub else root / name
        if not directory.is_dir():
            directory = root / name
        if directory.is_dir():
            out.append(directory)
    return out


def prepare(
    *,
    stage: str,
    blend: str = "data_blend_raw.json",
    limit: int | None = None,
    val_frac: float = 0.02,
    root: Path | None = None,
    out: Path | None = None,
) -> int:
    """把 prompts 转成 verl 需要的 train / val parquet。"""
    paths = env_paths()
    root = Path(root or paths["post"])
    out = Path(out or (Path(paths["data"]) / stage))
    blend_path = stage_dirs(stage) / "config" / "data_prep" / blend
    blend_spec = load_blend_spec(blend_path) if blend_path.is_file() else None
    rows: list[dict] = []
    for directory in _source_dirs(root, blend_spec):
        for path in sorted(directory.glob("**/*")):
            if path.suffix not in (".jsonl", ".json"):
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt = next((row[key] for key in PROMPT_KEYS if key in row), None)
                    if prompt:
                        answer = next((row[key] for key in ANSWER_KEYS if row.get(key)), "")
                        rows.append({"prompt": prompt, "ground_truth": str(answer)})
                    if limit and len(rows) >= limit:
                        break
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise SystemExit(f"[looma] 没有可用 prompt：{root}（配比 {blend}）")
    with_answer = sum(1 for row in rows if row["ground_truth"])
    print(f"[looma] 带答案的 prompt：{with_answer}/{len(rows)}（规则奖励只对带答案的那部分有效）")
    n_val = max(1, int(len(rows) * val_frac))
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("val.parquet", rows[:n_val]), ("train.parquet", rows[n_val:])):
        pq.write_table(
            pa.table(
                {
                    "prompt": [[{"role": "user", "content": row["prompt"]}] for row in subset],
                    "data_source": [stage] * len(subset),
                    "ability": [stage.split("_")[-1]] * len(subset),
                    "reward_model": [
                        {"style": "rule", "ground_truth": row["ground_truth"]} for row in subset
                    ],
                    "extra_info": [{"prompt": row["prompt"]} for row in subset],
                }
            ),
            out / name,
        )
    print(f"[looma] {out}/train.parquet（{len(rows) - n_val}）+ val.parquet（{n_val}）")
    return len(rows)


def smoke_prompts(stage: str) -> Path:
    """合成一批冒烟用 prompts。"""
    out = Path(env_paths()["runs"]) / "smoke_stage2_rl" / stage / "smoke_prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for index in range(1, 9):
            a, b = index * 7, index * 3
            handle.write(
                json.dumps(
                    {
                        "problem": f"What is {a} + {b}? Put the final answer in \\boxed{{}}.",
                        "answer": str(a + b),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[looma] 冒烟用 prompt jsonl：{out}（8 条带答案）")
    return out


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{stage} 的 RL 数据准备")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    parser.add_argument("--blend", default=None, help="配比 json（相对 config/data_prep/）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--root", default=None, help="prompts 根目录（--smoke 时忽略）")
    parser.add_argument("--smoke", action="store_true", help="用合成的算术 prompts 自检")
    args = parser.parse_args(argv)
    if not args.prepare:
        parser.print_help()
        return 1
    name = profile_from_args(args.config, "default", stage)
    cfg = dataprep_config(here / "config" / "data_prep" / f"{name}.yaml")
    root = Path(args.root) if args.root else None
    blend = args.blend or cfg.get("blend") or "data_blend_raw.json"
    if args.smoke:
        root = smoke_prompts(stage).parent.parent
        blend = "__smoke__"
    prepare(
        stage=stage,
        blend=blend,
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        val_frac=args.val_frac if args.val_frac is not None else (cfg.get("val_frac") or 0.02),
        root=root,
    )
    return 0
