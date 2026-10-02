#!/usr/bin/env python3
"""Unified RFT 数据（论文 §2.5.3）。

两个专家 E_TwG / E_TwP 在（更大的）数据池上 rollout 生成 RFT 数据：
  - 按难度分层保留 **全部 Normal-Level** + 随机 **5% Easy-Level**（防过于简单场景的灾难性遗忘）；
  - 保留的题取奖励最高的**正确** rollout 作为监督目标（思考中的原语无需再验证——奖励已经判过）；
  - Hard-Level 全错、无监督信号，不进 RFT。
产物与 stage1 同 schema（image/question/thinking/response），train.py 从**基座预训练模型**
重新 SFT（超参与冷启动 SFT 相同，只换数据配比 → 统一模型 F）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shensi import runtime  # noqa: F401,E402
from shensi.recipes.shensi_vl.common import paths as vl_paths  # noqa: E402
from shensi.recipes.shensi_vl.common import processors as proc_mod  # noqa: E402
from shensi.recipes.shensi_vl.common.train import train_loop  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import grpo  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import reward as R

POOLS = {
    "grounding": ("box_tasks.jsonl", "box"),
    "pointing": ("point_tasks.jsonl", "point"),
}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        yield from fh


def split_thinking(text: str) -> tuple[str, str]:
    """生成文本 → (thinking, response)：按 </think> 切；没有就整体当 response。"""
    parts = text.split("</think>", 1)
    if len(parts) == 2:
        thinking = parts[0].replace("<think>", "").strip()
        return thinking, parts[1].strip()
    return "", text.strip()


def rollout_pool(rows: list[dict], *, model_dir: str, tokenizer_dir: str, processor_path: str,
                 n: int, limit: int | None, easy_frac: float, seed: int) -> list[dict]:
    """专家 rollout → RFT 样本（schema 同 stage1）。"""
    cfg = {"train": {"model": {"tokenizer_dir": tokenizer_dir, "llm_path": model_dir,
                               "vision_path": model_dir, "processor_path": processor_path}}}
    tok = train_loop.build_tokenizer(cfg)
    model = train_loop.make_model(cfg, 0)
    iproc = proc_mod.load_image_processor(processor_path)
    rng = random.Random(seed)
    kept = []
    buckets = {"normal": [], "easy": []}
    for i, row in enumerate(rows):
        if limit and i >= limit:
            break
        prompt = grpo.encode_prompt(tok, iproc, row, trigger=True)
        pb = grpo.to_device(grpo.batch_prompts(tok, [prompt]), model.device)
        gen = model.generate(
            pb["input_ids"].repeat(n, 1), pb["attention_mask"].repeat(n, 1),
            torch.cat([pb["pixel_values"]] * n, dim=0) if pb["pixel_values"] is not None else None,
            pb["image_positions"].repeat(n, 1),
            max_new_tokens=1024, temperature=1.0,
        )
        texts = tok.batch_decode(gen, skip_special_tokens=False)
        gt = row.get("spec") or row.get("ground_truth") or {}
        scored = [(R.compute_score(row.get("task", "count"), s, gt, {}), s) for s in texts]
        k = sum(1 for s, _ in scored if s > 0.5)
        if k == 0:
            continue  # Hard-Level：无监督信号，不进 RFT
        best_score, best = max(scored, key=lambda x: x[0])
        thinking, response = split_thinking(best)
        out = {
            "task": row.get("task"),
            "image": row.get("image"),
            "question": row["question"],
            "thinking": thinking,
            "response": response,
            "source": f"rft_{row.get('source', '')}",
        }
        (buckets["easy"] if k == n else buckets["normal"]).append(out)
        if (i + 1) % 50 == 0:
            print(f"[rft] {i + 1}: normal={len(buckets['normal'])} easy={len(buckets['easy'])}", flush=True)
    n_easy = int(len(buckets["easy"]) * easy_frac)
    kept = buckets["normal"] + rng.sample(buckets["easy"], n_easy)
    print(f"[rft] 保留 Normal {len(buckets['normal'])} + Easy {n_easy}/{len(buckets['easy'])}（5% 口径）")
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage3_rft 数据")
    ap.add_argument("--pool", action="append", default=[],
                    help="任务池 jsonl（可多次：grounding/pointing 各一）")
    ap.add_argument("--model-grounding", default=None, help="E_TwG 目录")
    ap.add_argument("--model-pointing", default=None, help="E_TwP 目录")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--rollouts", type=int, default=0, help="0=读现成 rft jsonl 不重跑 rollout")
    ap.add_argument("--ready", default=None, help="已生成好的 RFT jsonl（直接混合用）")
    ap.add_argument("--general", default=None, help="通用多模态 jsonl（70% 部分）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--easy-frac", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    out = Path(args.out or paths["data"] / "stage3_rft")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rft_rows: list[dict] = []
    if args.ready:
        rft_rows = [json.loads(line) for line in _read_jsonl(args.ready) if line.strip()]
    elif args.rollouts > 0:
        tokenizer_dir = args.tokenizer or paths["vl_tokenizer"]
        for fam, pool_file in POOLS.values():
            pool = Path(paths["data"]) / "stage1_sft" / pool_file
            if not pool.is_file():
                print(f"[rft] 跳过 {fam}：{pool} 不存在")
                continue
            rows = [json.loads(line) for line in _read_jsonl(pool) if line.strip()]
            model_dir = (args.model_grounding if fam == "grounding" else args.model_pointing) or str(
                Path(paths["ckpt"]) / "stage2_rl" / ("stage1_grounding" if fam == "grounding" else "stage2_pointing") / "final"
            )
            rft_rows += rollout_pool(
                rows, model_dir=model_dir, tokenizer_dir=tokenizer_dir,
                processor_path=paths["processor"], n=args.rollouts,
                limit=args.limit, easy_frac=args.easy_frac, seed=args.seed,
            )
    else:
        raise SystemExit("[rft] 给 --ready 或 --rollouts N")
    if not rft_rows:
        raise SystemExit("[rft] 0 条 RFT 样本")

    # 70% 通用 + 30% RFT（与冷启动 SFT 同配比；超参不变、只换数据 —— 论文 §2.5.3）
    general = []
    if args.general and Path(args.general).is_file():
        general = [json.loads(line) for line in _read_jsonl(args.general) if line.strip()]
        rng.shuffle(general)
    k = min(len(general), int(len(rft_rows) * 7 / 3))
    mixed = rft_rows + general[:k]
    rng.shuffle(mixed)
    fp = out / "rft_train.jsonl"
    with open(fp, "w", encoding="utf-8") as fh:
        for r in mixed:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "rft_val.jsonl", "w", encoding="utf-8") as fh:
        for r in mixed[: max(2, len(mixed) // 200)]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rft] {len(mixed)} 条（RFT {len(rft_rows)} + 通用 {min(k, len(general))}）→ {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
