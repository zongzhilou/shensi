#!/usr/bin/env python3
"""吞吐标定台：一条命令把候选配置点各跑 N 步，打「token/s、每卡 TFLOP/s、MFU」表。

真机上跑（要 mcore + TE-NPU + 打好的数据）；本机先 `--dry-run` 看它要跑什么。

四个点（`--points` 可换）：

| 点 | 布局 | 批 | 说明 |
|---|---|---|---|
| `base` | TP1×EP8×DP1 | seq64 × mbs1 × 累积8 | 实测基线（对照，14.9 s/步 ≈ 34 token/s） |
| `thru_fp32` | TP2×EP4×DP1 | seq4096 × mbs8 × 累积16 | 大微批摊销发射（fp32-keep 开 = 现状口径） |
| `thru_bf16` | 同 thru_fp32 | 同左 | `config/thru.yaml`：再把 mHC/AttnRes 换 bf16（fp32 累加） |
| `ep1` | TP2×EP1×DP4 | seq4096 × mbs4 × 累积4 | 专家不切卡：免每层 all-to-all（要 precision-aware） |

用法：
    python -m shensi.recipes.shensi.common.train.bench_throughput --dry-run     # 只打印要跑的命令
    python -m shensi.recipes.shensi.common.train.bench_throughput --steps 20    # 真跑并出表
    python -m shensi.recipes.shensi.common.train.bench_throughput --steps 20 --points thru,bf16
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from shensi.recipes.shensi.common import common
from shensi.recipes.shensi.common.train import launcher

STAGE = "stage0_pretrain/stage1_pretrain"
PEAK_TFLOPS_PER_CARD = 376.0  # 910B2 bf16

# 点定义：profile + 覆写（--set 形式，顺序即优先级）
POINTS: dict[str, tuple[str, list[str], str]] = {
    "base": (
        "default",
        [
            "train.system.tensor_model_parallel_size=1",
            "train.system.expert_model_parallel_size=8",
            "train.model.seq_length=64",
            "train.model.micro_batch_size=1",
            "train.model.global_batch_size=8",
        ],
        "基线：TP1×EP8×DP1、seq64×mbs1×累积8",
    ),
    "thru_fp32": (
        "default",  # 默认档 = hc_fp32_keep 开着（fp32 快照口径）→ 本点是"大微批 + fp32"的对照
        [
            "train.system.tensor_model_parallel_size=2",
            "train.system.expert_model_parallel_size=4",
            "train.system.sequence_parallel=true",
            "train.model.seq_length=4096",
            "train.model.micro_batch_size=8",
            "train.model.global_batch_size=128",
        ],
        "大微批 + fp32-keep 开：TP2×EP4×DP1、seq4096×mbs8×累积16",
    ),
    "thru_bf16": (
        "thru",  # thru 档 = 同上 + no_shensi_hc_fp32_keep
        ["train.model.global_batch_size=128"],
        "大微批 + fp32-keep 关（mHC/AttnRes 走 bf16，fp32 累加）",
    ),
    "ep1": (
        "thru",
        [
            "train.system.expert_model_parallel_size=1",
            "train.model.micro_batch_size=4",
            "train.model.global_batch_size=16",
        ],
        "TP2×EP1×DP4：专家不切卡，免 all-to-all",
    ),
}

_ITER_MS = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")


def build_point(tag: str, steps: int, root: Path) -> tuple[dict, Path]:
    profile, overrides, _ = POINTS[tag]
    cfg = common.build_config(
        STAGE,
        profile,
        [
            *overrides,
            f"train.model.train_iters={steps}",
            "experiment.save_steps=1000000",  # 标定期间不存
            f"experiment.exp_name=shensi_bench_{tag}",
            f"experiment.exp_dir={root / tag}",
            "experiment.runner.nproc_per_node=8",
            "experiment.runner.nnodes=1",
        ],
        Path("/tmp/shensi_bench_data"),
    )
    return cfg, root / tag


def step_stats(cfg: dict) -> tuple[int, int]:
    argv = launcher.flatten_train_section(cfg["train"])
    get = lambda k, d=0: int(float(argv[argv.index(k) + 1])) if k in argv else d
    seq = get("--seq-length", 4096)
    gbs = get("--global-batch-size", 1)
    return gbs * seq, seq


def parse_throughput(log: Path, tokens_per_step: int, active_params: float) -> dict:
    ms = [float(m) for m in _ITER_MS.findall(log.read_text(encoding="utf-8", errors="ignore"))]
    if not ms:
        return {"status": "no-timing-in-log"}
    tail = ms[-max(3, len(ms) // 4) :]
    per_step_s = statistics.median(tail) / 1000.0
    tok_s = tokens_per_step / per_step_s
    tflops_card = tok_s * 6.0 * active_params / 8.0 / 1e12
    return {
        "status": "ok",
        "iters_logged": len(ms),
        "median_ms": round(statistics.median(tail), 1),
        "tokens_per_step": tokens_per_step,
        "tokens_per_s": round(tok_s, 1),
        "tflops_per_card": round(tflops_card, 2),
        "mfu": round(tflops_card / PEAK_TFLOPS_PER_CARD, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi 8 卡吞吐标定台")
    ap.add_argument("--steps", type=int, default=20, help="每个点跑多少步")
    ap.add_argument("--points", default="base,thru_fp32,thru_bf16,ep1", help="跑哪些点（逗号分隔）")
    ap.add_argument(
        "--active-params", type=float, default=4.235e9, help="每 token 激活参数（9B 档）"
    )
    ap.add_argument("--root", default="/tmp/shensi_bench", help="标定产物根目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印每点要跑的命令（本机可跑）")
    args = ap.parse_args(argv)

    root = Path(args.root)
    results: dict[str, dict] = {}
    tags = [t for t in args.points.split(",") if t]
    unknown = [t for t in tags if t not in POINTS]
    if unknown:
        raise SystemExit(f"未知的标定点 {unknown}；可用：{sorted(POINTS)}")

    for tag in tags:
        cfg, run_dir = build_point(tag, args.steps, root)
        tokens_per_step, seq = step_stats(cfg)
        plan = f"{POINTS[tag][2]}｜{tokens_per_step} token/步 × {args.steps} 步"
        if args.dry_run:
            cmd = launcher.build_command(cfg)
            print(f"\n=== [{tag}] {plan}")
            print("    " + " ".join(str(c) for c in cmd))
            results[tag] = {"status": "planned", "tokens_per_step": tokens_per_step}
            continue
        launcher.write_run_dir(cfg, run_dir)
        print(f"\n=== [{tag}] {plan} → {run_dir}")
        rc = launcher.launch(cfg, run_dir, dry_run=False)
        log = next((run_dir / "logs").glob("*.output"), None)
        res = (
            parse_throughput(log, tokens_per_step, args.active_params)
            if rc == 0 and log
            else {"status": f"launch rc={rc}"}
        )
        res["profile"], res["overrides"] = POINTS[tag][0], POINTS[tag][1]
        results[tag] = res
        print(f"    {res}")

    out = root / "bench_throughput.json"
    root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 汇总 ===")
    print(f"{'point':6s} {'token/s':>10s} {'TFLOP/s/卡':>11s} {'MFU':>7s}  说明")
    for tag, res in results.items():
        if res.get("status") == "ok":
            print(
                f"{tag:6s} {res['tokens_per_s']:>10.1f} {res['tflops_per_card']:>11.2f}"
                f" {res['mfu']:>7.2%}  {POINTS[tag][2]}"
            )
        else:
            print(f"{tag:6s} {res.get('status'):>10s} {'':>11s} {'':>7s}  {POINTS[tag][2]}")
    print(f"\n记账：{out}")
    return 0 if all(r.get("status") in ("ok", "planned") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
