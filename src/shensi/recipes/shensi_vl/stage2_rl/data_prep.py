#!/usr/bin/env python3
"""RL 任务池准备 + N-rollout 难度分层（论文 §2.5.2 的 RL Data 一节）。

流程：
  1. 读 stage1_sft 产出的任务池（box_tasks.jsonl / point_tasks.jsonl，带 spec/ground_truth）；
  2. 用对应专家 ckpt（F_TwG / F_TwP）对每题 rollout N 次，按奖励判对错：
     Easy-Level（全对）/ Normal-Level（部分对）/ Hard-Level（全错）；
  3. RL 只喂 Normal-Level（GRPO 有梯度信号；Easy 无信息、Hard 无梯度）。
  `--rollouts 0` 跳过分层（直接全量，冒烟用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shensi import runtime  # noqa: F401,E402
from shensi.recipes.shensi_vl.common import paths as vl_paths  # noqa: E402
from shensi.recipes.shensi_vl.common import processors as proc_mod  # noqa: E402
from shensi.recipes.shensi_vl.common.train import train_loop  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import grpo  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import reward as R

FAMILY_POOL = {"grounding": "box_tasks.jsonl", "pointing": "point_tasks.jsonl"}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        yield from fh


def difficulty_split(rows: list[dict], *, model_dir: str, tokenizer_dir: str,
                     processor_path: str, n: int, temperature: float = 1.0,
                     limit: int | None = None) -> tuple[list, list, list]:
    """返回 (easy, normal, hard)。判定：correct 数 k ∈ {0..N}。"""
    cfg = {
        "train": {
            "model": {
                "tokenizer_dir": tokenizer_dir, "llm_path": model_dir,
                "vision_path": model_dir, "processor_path": processor_path,
            }
        }
    }
    tok = train_loop.build_tokenizer(cfg)
    model = train_loop.make_model(cfg, 0)  # image_token_id 只在训练嵌入时用；这里仅生成
    iproc = proc_mod.load_image_processor(processor_path)
    easy, normal, hard = [], [], []
    for i, row in enumerate(rows):
        if limit and i >= limit:
            break
        prompt = grpo.encode_prompt(tok, iproc, row, trigger=True)
        pb = grpo.to_device(grpo.batch_prompts(tok, [prompt]), model.device)
        gen = model.generate(
            pb["input_ids"].repeat(n, 1), pb["attention_mask"].repeat(n, 1),
            torch.cat([pb["pixel_values"]] * n, dim=0) if pb["pixel_values"] is not None else None,
            pb["image_positions"].repeat(n, 1),
            max_new_tokens=1024, temperature=temperature,
        )
        texts = tok.batch_decode(gen, skip_special_tokens=False)
        gt = row.get("spec") or row.get("ground_truth") or {}
        rs = [R.compute_score(row.get("task", "count"), s, gt, {}) for s in texts]
        k = sum(1 for x in rs if x > 0.5)
        bucket = easy if k == n else (normal if k > 0 else hard)
        bucket.append({**row, "rollout_correct": k, "rollout_n": n})
        if (i + 1) % 50 == 0:
            print(f"[difficulty] {i + 1}: easy={len(easy)} normal={len(normal)} hard={len(hard)}", flush=True)
    return easy, normal, hard


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage2_rl 任务池")
    ap.add_argument("--family", choices=tuple(FAMILY_POOL), required=True)
    ap.add_argument("--pool", default=None, help="任务池 jsonl（默认 stage1_sft 产物）")
    ap.add_argument("--model", default=None, help="rollout 用的 ckpt（默认 stage1_sft 对应专家 final）")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--rollouts", type=int, default=8, help="难度分层的 N；0=跳过分层全量直出")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    pool = Path(
        args.pool or Path(paths["data"]) / "stage1_sft" / FAMILY_POOL[args.family]
    )
    rows = [json.loads(line) for line in _read_jsonl(pool) if line.strip()]
    print(f"[stage2_rl] 任务池 {pool}：{len(rows)} 条")
    if args.rollouts and args.rollouts > 0:
        model_dir = args.model or str(
            Path(paths["ckpt"]) / "stage1_sft" / ("box" if args.family == "grounding" else "point") / "final"
        )
        tokenizer_dir = args.tokenizer or paths["vl_tokenizer"]
        easy, normal, hard = difficulty_split(
            rows, model_dir=model_dir, tokenizer_dir=tokenizer_dir,
            processor_path=paths["processor"], n=args.rollouts, limit=args.limit,
        )
        print(f"[stage2_rl] 难度分层（N={args.rollouts}）：easy={len(easy)} normal={len(normal)} hard={len(hard)}；"
              "RL 只取 Normal-Level（论文口径）")
        rows = normal
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out or paths["data"] / "stage2_rl" / args.family)
    out.mkdir(parents=True, exist_ok=True)
    import random

    rng = random.Random(23)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    with open(out / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in rows[n_val:]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "val.jsonl", "w", encoding="utf-8") as fh:
        for r in rows[:n_val]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stage2_rl] {args.family}: train {len(rows) - n_val} + val {n_val} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
