#!/usr/bin/env python3
"""OPD 数据：学生（统一模型 F）在数据池上**自采样**轨迹（on-policy 的"on-policy"就在这）。

论文 §2.5.4：学生基于**自己生成的轨迹**去学专家的输出分布（不是学专家的轨迹）。
本脚本用 F 的 ckpt 对任务池 rollout（温度采样），保留轨迹文本（不再用奖励过滤——
蒸馏目标是分布对齐，不是只学好样本；论文对轨迹也没有做正确性过滤），
按统一 schema 落成 opd_train.jsonl（thinking/response 来自生成文本的拆分）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shensi import runtime  # noqa: F401,E402
from shensi.recipes.shensi_vl.common import paths as vl_paths  # noqa: E402
from shensi.recipes.shensi_vl.common.train import train_loop  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import grpo  # noqa: E402
from shensi.recipes.shensi_vl.stage3_rft.data_prep import split_thinking  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        yield from fh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage4_opd 数据（学生自采样）")
    ap.add_argument("--pool", action="append", default=[], help="任务池 jsonl（默认 stage1_sft 两份）")
    ap.add_argument("--model", default=None, help="学生 ckpt（默认 stage3_rft final）")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    out = Path(args.out or paths["data"] / "stage4_opd")
    out.mkdir(parents=True, exist_ok=True)
    pools = args.pool or [
        str(Path(paths["data"]) / "stage1_sft" / f)
        for f in ("box_tasks.jsonl", "point_tasks.jsonl")
    ]
    rows = []
    for p in pools:
        if Path(p).is_file():
            rows += [json.loads(line) for line in _read_jsonl(p) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("[opd] 任务池为空")

    cfg = {"train": {"model": {"tokenizer_dir": args.tokenizer or paths["vl_tokenizer"],
                               "llm_path": args.model or str(Path(paths["ckpt"]) / "stage3_rft/final"),
                               "vision_path": args.model or str(Path(paths["ckpt"]) / "stage3_rft/final"),
                               "processor_path": paths["processor"]}}}
    tok = train_loop.build_tokenizer(cfg)
    student = train_loop.make_model(cfg, 0)
    from shensi.recipes.shensi_vl.common import processors as proc_mod

    iproc = proc_mod.load_image_processor(paths["processor"])
    n = 0
    with open(out / "opd_train.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            prompt = grpo.encode_prompt(tok, iproc, row, trigger=True)
            pb = grpo.to_device(grpo.batch_prompts(tok, [prompt]), student.device)
            gen = student.generate(
                pb["input_ids"], pb["attention_mask"],
                pb["pixel_values"], pb["image_positions"],
                max_new_tokens=1024, temperature=args.temperature,
            )
            text = tok.decode(gen[0], skip_special_tokens=False)
            thinking, response = split_thinking(text)
            fh.write(
                json.dumps(
                    {
                        "task": row.get("task"),
                        "image": row.get("image"),
                        "question": row["question"],
                        "thinking": thinking,
                        "response": response,
                        "source": "opd_student",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
            if n % 100 == 0:
                print(f"[opd] {n}/{len(rows)}", flush=True)
    print(f"[opd] 学生自采样 {n} 条 → {out / 'opd_train.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
