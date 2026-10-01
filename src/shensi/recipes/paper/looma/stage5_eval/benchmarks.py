"""评测集清单：与 MiniCPM5-2B 公布的口径对齐。

每一项给四样东西：EvalScope 注册表里的数据集名（``--datasets`` 直接用）、口径里的名字、参考分数
（MiniCPM5-2B 公布值，用于同口径对照；没有公开值就留 None）、以及 ``hf_id``：

* ``kind``：``open`` 只打模型端点，EvalScope 直接跑；``agent`` 要工具/环境（仓库、浏览器、终端），
  交给 harness（deepseek-harness）跑，端点用同一个。
* ``hf_id``：EvalScope 的默认数据集 id 只在 ModelScope 上时给的 **Hugging Face 双胞胎**。用
  ``--dataset-hub huggingface`` 跑时就把它当作 ``dataset_id`` 传给 EvalScope；``hf_subset_list``
  非空时连子集名一起换（双胞胎的 config 名与默认口径不同）。留 None 表示没有可用的双胞胎
  （脚本型数据集被 datasets 4.x 拒收，或只有 ModelScope 镜像），那种项要走 modelscope hub。

这里的名字都在本机装的 EvalScope 1.12 注册表里逐一核过；``hf_id`` 的双胞胎也逐个探过仓库、config
名单与切分口径（见 ``check_datasets.py``，它能把整张表再验一遍）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPEN = "open"
AGENT = "agent"


@dataclass(frozen=True)
class Benchmark:
    """一个评测项：EvalScope 数据集名 + 口径名 + 参考分数 + HF 双胞胎 + 由谁跑。"""

    name: str
    label: str
    kind: str = OPEN
    reference: float | None = None
    hf_id: str | None = None
    hf_subset_list: tuple[str, ...] = field(default=())
    note: str = ""


#: MiniCPM5-2B 公开口径的评测项（reference 为模型卡公布值，None = 未公布）
BENCHMARKS: tuple[Benchmark, ...] = (
    # --- 通用知识与推理 ---
    Benchmark("mmlu_pro", "MMLU-Pro", reference=70.8),
    Benchmark(
        "mmlu_redux",
        "MMLU-Redux",
        reference=84.7,
        hf_id="edinburgh-dawg/mmlu-redux-2.0",
        note="双胞胎 57 科目与默认口径同名",
    ),
    Benchmark("super_gpqa", "SuperGPQA", reference=40.8),
    Benchmark(
        "gpqa_diamond",
        "GPQA-Diamond",
        reference=48.59,
        hf_id="Idavidrein/gpqa",
        hf_subset_list=("gpqa_diamond",),
        note="双胞胎是 gated 仓库，需 HF_TOKEN 并接受条款",
    ),
    Benchmark("mmlu", "MMLU", reference=59.8, note="参考值为第三方 5-shot"),
    Benchmark(
        "ceval",
        "C-Eval",
        hf_id="ceval/ceval-exam",
        note="双胞胎 52 科目与默认口径同名（同源打包）",
    ),
    Benchmark(
        "cmmlu",
        "CMMLU",
        note="默认 id 只有 ModelScope 镜像，HF 双胞胎是脚本型（datasets 4.x 拒收）",
    ),
    Benchmark("bbh", "BBH", hf_id="lukaemon/bbh", note="双胞胎 27 任务与默认口径同名"),
    # --- 数学 ---
    Benchmark("math_500", "MATH-500", reference=94.6, hf_id="HuggingFaceH4/MATH-500"),
    Benchmark("aime24", "AIME 2024", hf_id="Maxwell-Jia/AIME_2024"),
    Benchmark(
        "aime25",
        "AIME 2025",
        reference=86.5,
        hf_id="opencompass/AIME2025",
        hf_subset_list=("AIME2025-I", "AIME2025-II"),
        note="参考值为公布口径的 AIME 平均（25/26 同口径）",
    ),
    Benchmark("aime26", "AIME 2026", note="默认 id 只有 ModelScope 镜像，无 HF 双胞胎"),
    Benchmark(
        "gsm8k",
        "GSM8K",
        reference=82.1,
        hf_id="openai/gsm8k",
        note="参考值为第三方 3-shot CoT",
    ),
    # --- 指令跟随 ---
    Benchmark("ifeval", "IFEval", reference=86.7, hf_id="google/IFEval"),
    Benchmark("ifbench", "IFBench", reference=50.67),
    # --- 代码 ---
    Benchmark(
        "live_code_bench",
        "LiveCodeBench v6",
        note="默认 id 是官方 parquet 镜像；HF 双胞胎是脚本型（datasets 4.x 拒收）",
    ),
    Benchmark(
        "scicode", "SciCode", reference=26.3, hf_id="SciCode1/SciCode", note="参考值为 wbg 子项"
    ),
    Benchmark(
        "humaneval",
        "HumanEval",
        reference=81.1,
        hf_id="openai/openai_humaneval",
        note="参考值为第三方 pass@1",
    ),
    Benchmark("mbpp", "MBPP"),
    Benchmark(
        "bigcodebench", "BigCodeBench", hf_id="bigcode/bigcodebench", note="默认切分是 v0.1.4"
    ),
    # --- 长上下文 ---
    Benchmark(
        "longbench_v2",
        "LongBench v2",
        hf_id="zai-org/LongBench-v2",
        note="双胞胎换了组织名（zai-org）；长文档，需要长上下文档",
    ),
    Benchmark("needle_haystack", "Needle in a Haystack", note="默认 id 只有 ModelScope 镜像"),
    Benchmark("longmemeval", "LongMemEval", note="长程记忆，EvalScope 注册表内选项"),
    # --- agent / 工具类（走 harness）---
    Benchmark(
        "swe_bench_verified",
        "SWE-Bench-Verified",
        kind=AGENT,
        reference=46.4,
        note="另可用 swe_bench_verified_agentic 走 agentic 口径",
    ),
    Benchmark("terminal_bench_v2_1", "Terminal-Bench v2.1", kind=AGENT, reference=8.6),
    Benchmark("bfcl_v4", "BFCL v4", kind=AGENT, reference=55.4),
    Benchmark(
        "gaia",
        "GAIA Text-103",
        kind=AGENT,
        reference=79.29,
        note="gated 仓库，需 HF_TOKEN 并接受条款",
    ),
    Benchmark(
        "tau2_bench",
        "τ²-Bench Telecom",
        kind=AGENT,
        reference=97.1,
        note="默认子集是 airline/retail，Telecom 子集按发布口径另配",
    ),
)

BY_NAME: dict[str, Benchmark] = {item.name: item for item in BENCHMARKS}

#: 默认 id 本身就在 HF 上的项：不必覆写，但要让"能不能从 HF 拿到"只有一个判据。
HF_DEFAULTS: dict[str, str] = {
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "mmlu": "cais/mmlu",
    "super_gpqa": "m-a-p/SuperGPQA",
    "ifbench": "allenai/IFBench_test",
    "mbpp": "google-research-datasets/mbpp",
}


def hf_usable(name: str) -> bool:
    """该项在 ``--dataset-hub huggingface`` 下能不能拿到（默认 id 就在 HF 上，或有双胞胎）。"""
    item = BY_NAME.get(name)
    return bool((item.hf_id if item else None) or HF_DEFAULTS.get(name))

#: 常用集合：主表（能力覆盖）、精简（冒烟/回归）、长上下文、agent（工具类）
SETS: dict[str, tuple[str, ...]] = {
    "mini": ("mmlu_pro", "math_500", "ifeval", "humaneval"),
    "main": (
        "mmlu_pro",
        "mmlu_redux",
        "super_gpqa",
        "gpqa_diamond",
        "mmlu",
        "ceval",
        "cmmlu",
        "bbh",
        "math_500",
        "aime24",
        "aime25",
        "aime26",
        "gsm8k",
        "ifeval",
        "ifbench",
        "live_code_bench",
        "scicode",
        "humaneval",
        "mbpp",
    ),
    "long": ("needle_haystack", "longbench_v2", "longmemeval"),
    "agent": tuple(item.name for item in BENCHMARKS if item.kind == AGENT),
    "all": tuple(item.name for item in BENCHMARKS),
}


def resolve(
    names: tuple[str, ...] | list[str] | None, sets: tuple[str, ...] | list[str] | None
) -> tuple[str, ...]:
    """把集合名与显式数据集名合并成一个去重的数据集清单。

    参数:
      names: 显式数据集名（EvalScope 的名字，或集合名）。
      sets: 集合名（``mini`` / ``main`` / ``long`` / ``agent`` / ``all``）。
    """
    picked: list[str] = []
    for name in list(sets or []) + list(names or []):
        expanded = SETS.get(name, (name,))
        for item in expanded:
            if item not in picked:
                picked.append(item)
    if not picked:
        picked = list(SETS["mini"])
    return tuple(picked)


def subset(names: tuple[str, ...] | list[str], kind: str) -> tuple[str, ...]:
    """按 ``kind`` 过滤（``open`` 走 EvalScope，``agent`` 走 harness）。"""
    return tuple(name for name in names if BY_NAME.get(name, Benchmark(name, name)).kind == kind)


def hf_dataset_args(names: tuple[str, ...] | list[str]) -> dict:
    """取这些评测项的 HF 双胞胎覆写（没有双胞胎的跳过）。

    返回值直接并进 EvalScope 的 ``--dataset-args``：``{数据集名: {dataset_id, subset_list}}``。
    """
    out: dict[str, dict] = {}
    for name in names:
        item = BY_NAME.get(name)
        if item is None or not item.hf_id:
            continue
        arg: dict = {"dataset_id": item.hf_id}
        if item.hf_subset_list:
            arg["subset_list"] = list(item.hf_subset_list)
        out[name] = arg
    return out


def without_hf(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """挑出在 ``--dataset-hub huggingface`` 下拿不到数据的评测项（默认 id 与双胞胎都不在 HF）。"""
    return tuple(name for name in names if BY_NAME.get(name) is not None and not hf_usable(name))
