"""监督微调的训练入口，含冒烟用的合成对话生成。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.looma.common import add_common_train_args, env_paths, train_from_args

__all__ = ["smoke_jsonl", "train_main"]


def smoke_jsonl() -> Path:
    """生成冒烟用的合成 messages jsonl。"""
    out = Path(env_paths()["runs"]) / "smoke_stage1_sft" / "smoke_sft.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for index in range(16):
            messages = {
                "messages": [
                    {"role": "user", "content": f"What is {index} plus {index}?"},
                    {"role": "assistant", "content": f"The answer is {2 * index}."},
                ]
            }
            handle.write(json.dumps(messages, ensure_ascii=False) + "\n")
    print(f"[looma] 冒烟用合成 messages jsonl：{out}（16 条）")
    return out


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """SFT 训练入口：默认读预处理的 jsonl，``--smoke`` 走合成数据。"""
    parser = argparse.ArgumentParser(description=f"{stage}（mcore SFT）")
    add_common_train_args(parser)
    parser.add_argument("--data-jsonl", default=None, help="messages jsonl")
    args = parser.parse_args(argv)
    if args.smoke:
        jsonl = smoke_jsonl()
        return train_from_args(
            stage,
            args,
            overrides=[f"train.data.data_path={jsonl}"],
            smoke_overrides=[f"train.data.data_path={jsonl}"],
        )
    paths = env_paths()
    jsonl = Path(args.data_jsonl or Path(paths["data"]) / stage / "sft_train.jsonl")
    if not jsonl.is_file() and not args.dry_run:
        raise SystemExit(f"没找到 {jsonl}，先跑 data_prep.py --prepare")
    return train_from_args(stage, args, overrides=[f"train.data.data_path={jsonl}"])
