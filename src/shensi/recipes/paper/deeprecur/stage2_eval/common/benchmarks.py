"""评测基准清单：**与 DeepStack 论文（arXiv 2406.04334）逐表对应**。

本阶段用 **lmms-eval** 作 harness；套件与基准如下。

| 套件 | 论文出处 | 基准 |
|---|---|---|
| ``main`` | Table 1（主表）与 Table 11 | VQAv2, GQA, TextVQA, DocVQA, InfoVQA, SEED, POPE, MMMU, MM-Vet |
| ``text`` | Table 2（文本向） | ChartQA, DocVQA, InfoVQA, MultiDocVQA, TextVQA |
| ``video`` | Table 3（零样本视频 QA） | EgoSchema, NextQA, MSVD, ActivityNet |
| ``ablation`` | Tables 4–8（消融） | GQA, POPE, SEED, TextVQA, DocVQA, ChartQA, InfoVQA |

**task 名已对真实 harness 核对**（2026-10-02，两代口径）：

- **论文年代（lmms-eval 0.1.2）**：任务名是**裸名**——``vqav2 / textvqa / docvqa / infovqa /
  chartqa / mmmu / pope / seedbench / multidocvqa / gqa / mmvet``（与论文表格逐字一致）；
  该代**没有视频 QA 任务**。
- **当前（lmms-eval 0.7.3）**：多数带 split 后缀——``vqav2_val_lite``、``textvqa_val``、
  ``docvqa_val``、``infovqa_val``、``mmmu_val``、``multidocvqa_val``；``gqa / pope /
  seedbench / chartqa / mmvet`` 同名；视频族为 ``egoschema / nextqa_mc_test / activitynetqa``。
- **MSVD 两代都没有**（论文 Table 3 的 4 项里有它）：需要自注册 task 或用等价替代并在论文里注明
  ——``resolve_tasks`` 会对它显式报错并给出候选，不静默跳过。

所以每个基准给**版本感知的候选名**（先当前代、后论文代），``eval.py --resolve-tasks`` 会在装了
harness 的机器上逐一解析，对不上就显式报出。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 论文用的 harness（原文 "LLMs-Eval"）与**已核实的两代口径**
HARNESS = "lmms-eval"
HARNESS_PIP = "lmms-eval"
HARNESS_VERIFIED = {
    "date": "2026-10-02",
    "paper_era": "0.1.2",  # 裸名；无视频 QA
    "current": "0.7.3",  # split 后缀；视频族在
    "method": "0.7.3 按 task YAML 的 `task:` 清单核对；0.1.2 按 tasks/ 目录名核对",
}

#: 两代 harness 都没有的论文基准（显式登记，解析会失败并给指引）
KNOWN_MISSING_IN_HARNESS = {
    "MSVD": "lmms-eval 0.1.2 与 0.7.3 均无 MSVD 任务；需自注册 MSVD-QA 或换等价基准并在论文注明",
}


@dataclass(frozen=True)
class Benchmark:
    """一个论文基准：论文里的名字 + lmms-eval 候选 task 名（当前代→论文代）+ 口径备注。"""

    paper_name: str
    tasks: tuple[str, ...]
    note: str = ""
    #: 论文用 ``‡`` 标了"报验证集"
    validation_split: bool = False


_MAIN = (
    Benchmark(
        "VQAv2",
        ("vqav2_val_lite", "vqav2_val", "vqav2"),
        note="‡ 验证集；VQA 官方答案列表判分（0.7.3=_val_lite / 0.1.2=裸名）",
        validation_split=True,
    ),
    Benchmark("GQA", ("gqa",), note="两代同名", validation_split=True),
    Benchmark(
        "TextVQA",
        ("textvqa_val", "textvqa"),
        note="‡；答案列表判分（0.7.3=_val / 0.1.2=裸名）",
        validation_split=True,
    ),
    Benchmark(
        "DocVQA",
        ("docvqa_val", "docvqa"),
        note="‡；ANLS 判分（0.7.3=_val / 0.1.2=裸名）",
        validation_split=True,
    ),
    Benchmark(
        "InfoVQA",
        ("infovqa_val", "infovqa"),
        note="‡；ANLS 判分（0.7.3=_val / 0.1.2=裸名）",
        validation_split=True,
    ),
    Benchmark("SEED", ("seedbench", "seedbench_lite"), note="论文列名 'SEED (all)'（0.1.2 另有 seedbench_2）"),
    Benchmark("POPE", ("pope",), note="论文列名 'POPE (all)'：三划分聚合；两代同名"),
    Benchmark("MMMU", ("mmmu_val", "mmmu"), note="‡（0.7.3=_val / 0.1.2=裸名）", validation_split=True),
    Benchmark("MM-Vet", ("mmvet",), note="GPT-4 判分（judge 走 lmms-eval 的 API 配置）；两代同名"),
)

_TEXT = (
    Benchmark("ChartQA", ("chartqa", "chartqa_lite"), note="‡；relaxed 准确率；两代同名（0.1.2=chartqa）"),
    Benchmark("DocVQA", ("docvqa_val", "docvqa"), note="‡"),
    Benchmark("InfoVQA", ("infovqa_val", "infovqa"), note="‡"),
    Benchmark("MultiDocVQA", ("multidocvqa_val", "multidocvqa"), note="0.7.3=_val / 0.1.2=裸名"),
    Benchmark("TextVQA", ("textvqa_val", "textvqa"), note="‡"),
)

_VIDEO = (
    Benchmark("EgoSchema", ("egoschema", "egoschema_subset"), note="零样本；**0.7.3 才有视频族**（0.1.2 无）"),
    Benchmark("NextQA", ("nextqa_mc_test",), note="零样本；0.7.3 名"),
    Benchmark("MSVD", ("msvd_qa", "msvd"), note=KNOWN_MISSING_IN_HARNESS["MSVD"]),
    Benchmark("ActivityNet", ("activitynetqa",), note="零样本；0.7.3 名"),
)

_ABLATION = (
    Benchmark("GQA", ("gqa",)),
    Benchmark("POPE", ("pope",)),
    Benchmark("SEED", ("seedbench", "seedbench_lite")),
    Benchmark("TextVQA", ("textvqa_val", "textvqa")),
    Benchmark("DocVQA", ("docvqa_val", "docvqa")),
    Benchmark("ChartQA", ("chartqa", "chartqa_lite")),
    Benchmark("InfoVQA", ("infovqa_val", "infovqa")),
)

#: 套件 → 论文基准列表（顺序按论文表格列序）
SUITES: dict[str, tuple[Benchmark, ...]] = {
    "main": _MAIN,
    "text": _TEXT,
    "video": _VIDEO,
    "ablation": _ABLATION,
}

#: 训练里出现过的数据集（论文 Table 1 的 ``*`` 标注：训练图在训练中见过）
TRAIN_OBSERVED = frozenset({"TextVQA", "DocVQA", "InfoVQA", "ChartQA", "GQA", "POPE", "SEED"})


def suite_benchmarks(suite: str) -> tuple[Benchmark, ...]:
    """取套件（找不到显式报错）。"""
    if suite not in SUITES:
        raise SystemExit(f"[deeprecur·eval] 不认识这个套件：{suite}（可选：{sorted(SUITES)}）")
    return SUITES[suite]


def task_names(suite: str, *, first_only: bool = True) -> list[str]:
    """套件的 lmms-eval task 列表（``first_only`` 取每基准的首选名）。"""
    names: list[str] = []
    seen: set[str] = set()
    for bench in suite_benchmarks(suite):
        name = bench.tasks[0] if first_only else bench.tasks[0]
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def resolve_tasks(suite: str) -> dict[str, str]:
    """**在装了 lmms-eval 的机器上**把候选名解析到真实 task 名（对不上显式报错）。

    返回 {论文基准名: lmms-eval task 名}。
    """
    try:
        from lmms_eval.tasks import get_task_dict  # noqa: F401  仅确认 harness 在
    except Exception as exc:
        raise SystemExit(
            f"[deeprecur·eval] 没有 lmms-eval：{exc}\n"
            f"  先装：pip install {HARNESS_PIP}（版本对齐论文时的实现；数据由 harness 自下）"
        ) from exc

    resolved: dict[str, str] = {}
    failures: list[str] = []
    for bench in suite_benchmarks(suite):
        for candidate in bench.tasks:
            try:
                get_task_dict([candidate])
                resolved[bench.paper_name] = candidate
                break
            except Exception:
                continue
        else:
            failures.append(f"{bench.paper_name}（候选：{list(bench.tasks)}）")
    if failures:
        hints = [
            f"{name}：{KNOWN_MISSING_IN_HARNESS[name]}"
            for name in KNOWN_MISSING_IN_HARNESS
            if any(name in item for item in failures)
        ]
        extra = ("\n  已知缺口——" + "；".join(hints)) if hints else ""
        raise SystemExit(
            "[deeprecur·eval] 这些论文基准在本机 lmms-eval 里对不上 task 名："
            + "；".join(failures)
            + "\n  用 `lmms-eval --tasks list` 查真实名后更新 benchmarks.py（不静默跳过）"
            + f"\n  已核实口径：论文年代={HARNESS_VERIFIED['paper_era']}（裸名）/ 当前={HARNESS_VERIFIED['current']}"
            + extra
        )
    return resolved


def paper_table() -> str:
    """打印套件表（README 与 CLI 共用）。"""
    lines = ["论文基准套件（DeepStack arXiv 2406.04334）："]
    for suite, benches in SUITES.items():
        names = ", ".join(b.paper_name for b in benches)
        lines.append(f"  {suite:<9} {names}")
    lines.append(f"  harness：{HARNESS}")
    return "\n".join(lines)


def main() -> int:
    print(paper_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
