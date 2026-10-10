#!/usr/bin/env python3
"""按 stage 的**语料加工单**：NeMo Curator（重筛）+ DataDesigner（合成）+ teacher 蒸馏 → 落进语料根。

分工：本模块只描述并驱动**加工**这一步（筛选/去重/合成/蒸馏），产物按 blend 条目名落成
parquet/jsonl；扣 token 预算、入库前轻筛与打包仍是 `data_prep.py --prepare` 的活（见 common.py
的 `--tokens/--quality/--dedup`）。**加工完的语料根是 data_prep 的唯一输入**，train 再直接吃
data_prep 的 blend。

用法（本机可跑的是 --plan；--run 要 nemo_curator / DataDesigner / LLM 端点）：

    python -m shensi.recipes.shensi.common.curate --plan                  # 打印全部 stage 的加工单
    python -m shensi.recipes.shensi.common.curate --plan --scale week_2b
    python -m shensi.recipes.shensi.common.curate --run pretrain          # 真跑（逐条执行）

两个 scale（token 预算按"一周训完"倒排；吞吐见 README 的「吞吐」一节）：

| scale | 适用 | PT | 中训练 | 长文 | SFT | 一周能不能训完 |
|---|---|---|---|---|---|---|
| `week_2b` | **≤2B 模型（推荐）** | 45B（≈Chinchilla@2B = 40B） | 3B | 2B | 1B | 能（~5–6 天 @20–30% MFU） |
| `week_9b` | 9B 不动 | 12B（1.3 t/p，欠训） | 1B | — | 0.3B | 能跑通，但远不到发布（Chinchilla@9B = 183B） |

8 卡的物理上限：一周 ≈ 10–20B token@9B、45–150B token@0.6–2B（按 20–30% MFU）——所以"一周 +
发布可测"在算术上要求模型 ≤2B；9B 一周只能给 1–2 t/p。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess

# 加工单：stage → (源, 工具链, 目标形态, 落点, 各 scale 的 token 预算)
# 工具链里的 Curator 步名对应 NeMo Curator 的 TextCurationPipeline 阶段；DataDesigner 给"配方摘要"
RECIPES: dict[str, dict] = {
    "pretrain": {
        "sources": "blend 的 32 源（网页/代码/数学/多语）",
        "curator": [
            "LanguageIdentification(zh,en,code)",
            "HeuristicFilter(gopher 子集：长度/字母比/重复行)",
            "ClassifierFilter(fasttext 质量分)",
            "ExactDuplicates + FuzzyDuplicates(MinHash/LSH)",
            "SemanticDuplicates(可选，embedding 近邻)",
            "PIIRedaction",
        ],
        "synth": "DataDesigner 配方 textbook：（数学/代码/推理）× 教材式改写 + LLM-as-judge 过滤",
        "distill": "stage3_opd teacher 链路产 token（掺 1/3 上沿）",
        "output": "<语料根>/<源名>__<config>.parquet（blend 条目名的直落）",
        "budget": {"week_2b": "45B", "week_9b": "12B"},
        "hours": {"week_2b": "4.7 天 @30%MFU", "week_9b": "5.8 天"},
    },
    "midtrain": {
        "sources": "同 PT + repo 级代码 slice + 长文档 slice",
        "curator": [
            "ExactDuplicates(跨 stage 增量，防 PT 重复计入)",
            "LengthFilter(min=4k chars, 长文档)",
            "ClassifierFilter(代码/数学/科学域)",
        ],
        "synth": "DataDesigner 配方 reasoning：链式推理题 + 校验器过滤",
        "distill": "teacher 蒸馏（DSA 引入段用短数据）",
        "output": "同 PT（另加 blend 条目名 __mid）",
        "budget": {"week_2b": "3B", "week_9b": "1B"},
        "hours": {"week_2b": "7.5 小时", "week_9b": "12 小时"},
    },
    "longctx": {
        "sources": "长文档 slice（build_longctx.py 的合成 + 自然长文）",
        "curator": ["LengthFilter(min=32k chars)", "ExactDuplicates", "PIIRedaction"],
        "synth": "多针检索合成（build_longctx.py 现有口径）",
        "distill": "—",
        "output": "<语料根>/longctx__*.jsonl",
        "budget": {"week_2b": "2B", "week_9b": "—"},
        "hours": {"week_2b": "5 小时", "week_9b": "—"},
    },
    "sft": {
        "sources": "指令数据（多域）+ 合成的 instruction-response",
        "curator": [
            "ExactDuplicates + FuzzyDuplicates",
            "HeuristicFilter(轮次/长度/模板合规)",
            "PIIRedaction",
            "长度分桶（SFT 的 THD 打包口径见 stage1_sft/data_prep.py）",
        ],
        "synth": "DataDesigner 配方 instruct：多域指令 + 响应；judge 过滤",
        "distill": "teacher 蒸馏响应（占比 1/3 上沿）",
        "output": "messages jsonl / parquet（stage1_sft/data_prep.py 的三种口径直吃）",
        "budget": {"week_2b": "1B", "week_9b": "0.3B"},
        "hours": {"week_2b": "2.5 小时", "week_9b": "3.4 小时"},
    },
    "eval": {
        "sources": "各源固定留出（一旦定下不再变）",
        "curator": ["ExactDuplicates(与训练集隔离)", "holdout 标记"],
        "synth": "—",
        "distill": "—",
        "output": "<语料根>/holdout/*.jsonl（只给 stage3_eval / valid-test 口径）",
        "budget": {"week_2b": "0.1B", "week_9b": "0.1B"},
        "hours": {"week_2b": "—", "week_9b": "—"},
    },
}


def _tools_available() -> dict[str, str | None]:
    return {
        "nemo_curator": shutil.which("nemo_curator"),
        "data_designer": shutil.which("data-designer") or shutil.which("data_designer"),
    }


def plan(scale: str, stages: list[str]) -> None:
    tools = _tools_available()
    print(f"=== 语料加工单（scale={scale}）｜工具：", end="")
    print(", ".join(f"{k}={'有' if v else '缺'}" for k, v in tools.items()))
    for stage in stages:
        r = RECIPES[stage]
        print(f"\n--- {stage}｜预算 {r['budget'][scale]}｜训练耗时 {r['hours'][scale]}")
        print(f"    源    : {r['sources']}")
        print(f"    Curator: {' → '.join(r['curator'])}")
        print(f"    合成  : {r['synth']}")
        print(f"    蒸馏  : {r['distill']}")
        print(f"    落点  : {r['output']}")
    print(
        "\n下一步：加工产物落进 $SHENSI_FS/datasets/llm/pre-training（按 blend 条目名）→ "
        "`python data_prep.py --config config/data_prep/formal.yaml --prepare` → `train.py --tokens <预算>`"
    )


def run(scale: str, stage: str) -> int:
    tools = _tools_available()
    missing = [k for k, v in tools.items() if not v]
    if missing:
        raise SystemExit(
            f"[curate] 缺工具 {missing}：NeMo Curator 与 DataDesigner 装好后再跑 --run"
            "（Curator 的 GPU 档要 CUDA/RAPIDS；NPU 机器用它的 CPU 路径，或在 CUDA 机器上加工完搬语料）。"
            "先用 --plan 看加工单。"
        )
    r = RECIPES[stage]
    # 工具在但具体流程参数要按语料根/blend 展开：这里给出可执行的骨架，逐条补参数即可
    for step in r["curator"]:
        print(f"[curate] {stage}: {step}（参数按 blend 源展开）")
    subprocess.run(["nemo_curator", "--help"], check=False)
    print(f"[curate] {stage}: 加工单已列出；DataDesigner 配方见 --plan 的「合成」行")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi 语料加工单（NeMo 工具链驱动）")
    ap.add_argument("--scale", choices=("week_2b", "week_9b"), default="week_2b")
    ap.add_argument("--plan", action="store_true", help="打印加工单（本机可跑）")
    ap.add_argument("--run", metavar="STAGE", default=None, help="真跑某个 stage 的加工（要工具）")
    ap.add_argument(
        "--stages", default="pretrain,midtrain,longctx,sft,eval", help="--plan 覆盖哪些 stage"
    )
    args = ap.parse_args(argv)
    if args.run:
        if args.run not in RECIPES:
            raise SystemExit(f"未知 stage {args.run}；可用：{sorted(RECIPES)}")
        return run(args.scale, args.run)
    plan(args.scale, [s for s in args.stages.split(",") if s in RECIPES])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
