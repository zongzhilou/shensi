#!/usr/bin/env python3
"""LR 三点标定台：×0.5 / ×1 / ×2（默认 1.35e-4 / 2.7e-4 / 5.4e-4）各跑 N 步，出「末段 loss」表。

为什么必须标定：2–3B / ~57B-token 档的公开 LR 带是 2–2.7e-4（Cerebras-2.7B 2e-4、phi-1.5 2e-4、
DeepSeek-V3 2.2e-4、GLM-4.5 2.5e-4、DeepSeek-V4-Flash 2.7e-4），但它们的**步批**是 1–64M token，
我们用 1.05M；再加上 AdaMuon + GrokFastAdamW 没有公开的 η(B) 公式（Moonlight 只发了 η(C)/B(C) 拟合）
——所以「峰值 LR」是整套配方里唯一必须上机扫的参数。三点扫完取 loss 最低档；三档差 <0.01 nats 说明
对 LR 不敏感，取中间档即可（并把结论回写 default.yaml 的注释）。

用法：
    python -m shensi.recipes.shensi.common.train.pilot_lr --dry-run          # 只打印命令
    python -m shensi.recipes.shensi.common.train.pilot_lr --steps 1000       # 3 点 × 1000 步
    python -m shensi.recipes.shensi.common.train.pilot_lr --lrs 2.7e-4,5.4e-4
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from shensi.recipes.shensi.common import common
from shensi.recipes.shensi.common.train import launcher
from shensi.recipes.shensi.common.train.loss_gate import expected_loss, parse_log

STAGE = "stage0_pretrain/stage1_pretrain"
ANCHOR_LR = 2.7e-4  # DeepSeek-V4-Flash 的峰值（default.yaml 的现值）


def build_point(lr: float, steps: int, profile: str, root: Path) -> tuple[dict, Path]:
    tag = f"lr{lr:.1e}".replace("+", "")
    cfg = common.build_config(
        STAGE,
        profile,
        [
            f"train.model.optimizer.lr_scheduler.lr={lr}",
            f"train.model.optimizer.lr_scheduler.min_lr={lr / 10.0}",
            f"train.model.train_iters={steps}",
            "experiment.save_steps=1000000",  # 标定期间不存
            f"experiment.exp_name=shensi_pilot_{tag}",
            f"experiment.exp_dir={root / tag}",
            "experiment.runner.nproc_per_node=8",
            "experiment.runner.nnodes=1",
        ],
        Path("/tmp/shensi_pilot_data"),
    )
    return cfg, root / tag


def tail_loss(log: Path, fraction: float = 0.25) -> dict:
    series = parse_log(log)
    if not series:
        return {"status": "no-loss-in-log"}
    losses = [v for _, v in series]
    tail = losses[-max(3, int(len(losses) * fraction)) :]
    return {
        "status": "ok",
        "points": len(series),
        "tokens": round(series[-1][0], 1),
        "tail_median": round(statistics.median(tail), 4),
        "tail_min": round(min(tail), 4),
        "last": round(losses[-1], 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi LR 三点标定台")
    ap.add_argument(
        "--steps", type=int, default=1000, help="每点步数（1000 步 × 1.05M token/步 ≈ 1.05B）"
    )
    ap.add_argument(
        "--lrs",
        default=",".join(f"{ANCHOR_LR * m:.3g}" for m in (0.5, 1.0, 2.0)),
        help="逗号分隔的峰值 LR（默认锚点的 ×0.5/×1/×2）",
    )
    ap.add_argument(
        "--profile", default="scale8", help="跑哪一档（默认 scale8，继承 default 的布局与批）"
    )
    ap.add_argument("--active-params", type=float, default=0.7570e9)
    ap.add_argument("--root", default="/tmp/shensi_pilot", help="产物根目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印每点命令（本机可跑）")
    args = ap.parse_args(argv)

    root = Path(args.root)
    lrs = [float(x) for x in args.lrs.split(",") if x]
    results: dict[str, dict] = {}
    for lr in lrs:
        tag = f"lr{lr:.1e}".replace("+", "")
        cfg, run_dir = build_point(lr, args.steps, args.profile, root)
        # 预期值（law + MoE 折扣，唯一语料按 pilot 全量=新鲜 token）
        tokens = 1.05e6 * args.steps
        exp = expected_loss(tokens, n_params=args.active_params)
        if args.dry_run:
            cmd = launcher.build_command(cfg)
            print(
                f"\n=== [{tag}] {args.steps} 步 ≈ {tokens / 1e9:.2f}B token｜law+MoE 折扣的预期 "
                f"{exp:.3f}（pilot 是冷启动、LR 未退火，观测天然偏高，只看三档相对高低）"
            )
            print("    " + " ".join(str(c) for c in cmd))
            results[tag] = {"status": "planned", "lr": lr, "expected": round(exp, 3)}
            continue
        launcher.write_run_dir(cfg, run_dir)
        print(f"\n=== [{tag}] {args.steps} 步 → {run_dir}")
        rc = launcher.launch(cfg, run_dir, dry_run=False)
        log = next((run_dir / "logs").glob("*.output"), None)
        res = tail_loss(log) if rc == 0 and log else {"status": f"launch rc={rc}"}
        res.update({"lr": lr, "expected": round(exp, 3), "profile": args.profile})
        results[tag] = res
        print(f"    {res}")

    out = root / "pilot_lr.json"
    root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 汇总（取 tail_median 最低的一档；三档差 <0.01 → 取中间档）===")
    print(f"{'point':>10} {'LR':>9} {'tail_median':>12} {'预期':>8} {'偏差':>8}")
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    for tag, res in results.items():
        if res.get("status") == "ok":
            print(
                f"{tag:>10} {res['lr']:>9.3g} {res['tail_median']:>12.4f} "
                f"{res['expected']:>8.3f} {res['tail_median'] - res['expected']:>+8.3f}"
            )
        else:
            print(f"{tag:>10} {res.get('lr', lrs[0]):>9.3g} {res.get('status'):>12}")
    if ok:
        best = min(ok.items(), key=lambda kv: kv[1]["tail_median"])
        print(
            f"\n选择：{best[0]}（LR {best[1]['lr']:.3g}）；把 default.yaml 的 lr/min_lr 回写成它 + 1/10。"
        )
    print(f"\n记账：{out}")
    return 0 if len(ok) == len(results) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
