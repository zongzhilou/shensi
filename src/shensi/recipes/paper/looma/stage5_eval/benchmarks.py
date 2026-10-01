"""评测口径表：与 MiniCPM5-2B 公布的口径对齐，并映射到 OpenCompass 的数据集。

每一项给四样东西：口径里的名字（也是本表的键）、展示名、OpenCompass 数据集模块前缀（`oc`，
按前缀在安装包的 ``configs/datasets/**`` 里解析，每项取排序后的第一个；None = OpenCompass
没有这一项，交给 harness 或另配）、以及参考分数（MiniCPM5-2B 公布值，用于同口径对照）。

    --set opencompass.datasets=leaderboard   # OpenCompass 自带 leaderboard 集合（默认）
    --set opencompass.datasets=minicpm5      # 本表的主力口径集合
    --set opencompass.datasets=mmlu_pro,gsm8k  # 单项（直接给 OpenCompass 名字前缀也行）
"""

from __future__ import annotations

from dataclasses import dataclass

OPEN = "open"
AGENT = "agent"


@dataclass(frozen=True)
class Bench:
    """一个口径项：口径名 + 展示名 + OpenCompass 模块前缀 + 参考分数 + 谁跑。"""

    name: str
    label: str
    oc: str | None = None
    kind: str = OPEN
    reference: float | None = None
    note: str = ""


BENCHMARKS: tuple[Bench, ...] = (
    # --- 通用知识与推理 ---
    Bench("mmlu_pro", "MMLU-Pro", oc="mmlu_pro.mmlu_pro_0shot_cot_gen", reference=70.8),
    Bench("mmlu_redux", "MMLU-Redux", reference=84.7, note="OpenCompass 无此项，需另配数据集"),
    Bench("super_gpqa", "SuperGPQA", oc="supergpqa.supergpqa_gen", reference=40.8),
    Bench("gpqa", "GPQA-Diamond", oc="gpqa.gpqa_0shot_nocot_gen", reference=48.59, note="取 gpqa 的 0-shot 生成档"),
    Bench("mmlu", "MMLU", oc="mmlu.mmlu_gen", reference=59.8, note="参考值为第三方 5-shot"),
    Bench("ceval", "C-Eval", oc="ceval.ceval_gen"),
    Bench("cmmlu", "CMMLU", oc="cmmlu.cmmlu_gen"),
    Bench("bbh", "BBH", oc="bbh.bbh_gen"),
    # --- 数学 ---
    Bench("math_500", "MATH-500", oc="math.math_500_gen", reference=94.6),
    Bench("aime24", "AIME 2024", oc="aime2024.aime2024_gen"),
    Bench(
        "aime25",
        "AIME 2025",
        oc="aime2025.aime2025_cascade_eval_gen",
        reference=86.5,
        note="参考值为公布口径的 AIME 平均；OpenCompass 侧只有 cascade/llmjudge 档",
    ),
    Bench("gsm8k", "GSM8K", oc="gsm8k.gsm8k_gen", reference=82.1, note="参考值为第三方 3-shot CoT"),
    # --- 指令跟随 ---
    Bench("ifeval", "IFEval", oc="IFEval.IFEval_gen", reference=86.7),
    Bench("ifbench", "IFBench", oc="IFBench.IFBench_gen", reference=50.67),
    # --- 代码 ---
    Bench("live_code_bench", "LiveCodeBench", oc="livecodebench.livecodebench_gen", reference=69.1, note="参考值对应 v6 版本"),
    Bench("scicode", "SciCode", oc="scicode.scicode_gen", reference=26.3, note="参考值为 wbg 子项"),
    Bench("humaneval", "HumanEval", oc="humaneval.humaneval_gen", reference=81.1, note="参考值为第三方 pass@1"),
    Bench("mbpp", "MBPP", oc="mbpp.mbpp_gen"),
    Bench("bigcodebench", "BigCodeBench", oc="bigcodebench.bigcodebench_gen"),
    # --- 长上下文 ---
    Bench("longbench", "LongBench", oc="longbench.longbench", note="聚合档；OpenCompass 无 v2"),
    Bench("ruler", "RULER", oc="ruler.ruler_128k_gen", note="128K 档"),
    Bench("needlebench", "NeedleBench", oc="needlebench.atc.atc", note="OpenCompass 的 Needle 家族"),
    # --- agent / 工具类（OpenCompass 里没有，走 harness）---
    Bench("swe_bench_verified", "SWE-Bench-Verified", kind=AGENT, reference=46.4),
    Bench("terminal_bench_v2_1", "Terminal-Bench v2.1", kind=AGENT, reference=8.6),
    Bench("bfcl_v4", "BFCL v4", kind=AGENT, reference=55.4),
    Bench("gaia", "GAIA Text-103", kind=AGENT, reference=79.29),
    Bench("tau2_bench", "τ²-Bench Telecom", kind=AGENT, reference=97.1),
)

BY_NAME: dict[str, Bench] = {item.name: item for item in BENCHMARKS}

#: 口径集合：minicpm5 = 公布口径的主力项；mini 是冒烟子集；long / agent 各自成集。
OC_SETS: dict[str, tuple[str, ...]] = {
    "mini": ("mmlu_pro", "math_500", "ifeval", "humaneval"),
    "minicpm5": (
        "mmlu_pro", "super_gpqa", "gpqa", "mmlu", "ceval", "cmmlu", "bbh",
        "math_500", "aime24", "aime25", "gsm8k", "ifeval", "ifbench",
        "live_code_bench", "scicode", "humaneval", "mbpp",
    ),
    "long": ("longbench", "ruler", "needlebench"),
    "agent": tuple(item.name for item in BENCHMARKS if item.kind == AGENT),
    "all": tuple(item.name for item in BENCHMARKS),
}


def oc_name(name: str) -> str | None:
    """取该口径项的 OpenCompass 模块前缀（None = OpenCompass 没有这项）。"""
    item = BY_NAME.get(name)
    if item is not None:
        return item.oc
    return name  # 允许直接写 OpenCompass 的名字前缀


def resolve(names: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """把集合名与显式名字合并成一个去重的口径清单。"""
    picked: list[str] = []
    for name in list(names or []):
        expanded = OC_SETS.get(name, (name,))
        for item in expanded:
            if item not in picked:
                picked.append(item)
    return tuple(picked)


def oc_entries(entries: tuple[str, ...] | list[str]) -> list[str]:
    """把口径清单（或集合名）转成 OpenCompass 模块前缀清单，缺项直接报错不静默丢。"""
    expanded = resolve(entries)
    missing = [name for name in expanded if not oc_name(name)]
    if missing:
        raise SystemExit(
            f"[looma·eval] 这些口径项在 OpenCompass 里没有对应数据集：{missing}"
            "（agent 类请用 --suite agent 走 harness）"
        )
    return [str(oc_name(name)) for name in expanded]


def compare_references(card: dict[str, float]) -> dict[str, dict]:
    """把 OpenCompass 的汇总（{数据集名: 分数}）对上口径表，给出参考分与差值。

    OpenCompass 的 summary 里 ``dataset`` 是数据集的 abbr（如 ``mmlu``），不一定等于我们表里的
    模块前缀，所以两遍匹配：先按口径名精确对，再把剩下的按家族名（模块前缀的第一段，如
    ``aime2024`` / ``IFEval``）对；一个数据集只认领一次。
    """
    out: dict[str, dict] = {}
    claimed: set[str] = set()
    for exact in (True, False):
        for bench in BENCHMARKS:
            if not bench.oc or bench.name in out:
                continue
            family = str(bench.oc).split(".")[0].lower()
            for dataset, score in card.items():
                if dataset in claimed:
                    continue
                low = dataset.lower()
                hit = low == bench.name.lower() if exact else (low == family or low.startswith(family + "_"))
                if not hit:
                    continue
                out[bench.name] = {
                    "label": bench.label,
                    "dataset": dataset,
                    "score": score,
                    "reference": bench.reference,
                }
                claimed.add(dataset)
                break
    return out


def subset(entries: tuple[str, ...] | list[str], kind: str) -> tuple[str, ...]:
    """按 ``kind`` 过滤（``open`` 走 OpenCompass，``agent`` 走 harness）。"""
    return tuple(name for name in resolve(entries) if BY_NAME.get(name, Bench(name, name)).kind == kind)
