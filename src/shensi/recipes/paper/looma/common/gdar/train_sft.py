"""监督微调段的训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.looma.common import gdar as common


def smoke_jsonl() -> Path:
    out = Path(common.env_paths()["runs"]) / "smoke_stage1_sft" / "smoke_sft.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(16):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"第 {i} 题：1+1 等于几？"},
                    {
                        "role": "assistant",
                        "reasoning_content": "先数一遍，再回答。",
                        "content": "2",
                    },
                ]
            }
        )
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"[gdar] 冒烟用合成 messages jsonl：{out}（{len(rows)} 条）")
    return out


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage}（监督微调：SFT-1 / SFT-2 / SFT-3）")
    common.add_common_train_args(ap)
    ap.add_argument(
        "--data-jsonl",
        default=None,
        help="messages jsonl（默认 <FS>/shensi/data/<stage>/sft_train.jsonl）",
    )
    args = ap.parse_args(argv)
    if args.smoke:
        return common.smoke(stage, "tiny", [f"train.data.data_path={smoke_jsonl()}"])
    paths = common.env_paths()
    jsonl = Path(args.data_jsonl or paths["data"] / stage / "sft_train.jsonl")
    if not jsonl.is_file() and not args.dry_run:
        raise SystemExit(f"没找到 {jsonl}，先跑 data_prep.py --prepare")
    return common.train_from_args(stage, args, overrides=[f"train.data.data_path={jsonl}"])
