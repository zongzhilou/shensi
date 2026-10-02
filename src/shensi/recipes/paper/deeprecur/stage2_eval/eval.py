#!/usr/bin/env python3
"""评测入口：基准套件 → harness 执行 → 汇总成表格形态的 summary.json。

默认 harness 为 lmms-eval（`--harness opencompass` 走可选后端）；被评模型由 vLLM 起成
OpenAI 兼容端点（native 走引擎原生实现，unified/gdar/deeprecur 走桥，见
``common/models/vllm/{registration,bridge}.py``）。

用法：
    python eval.py --suite main --dry-run                 # 只看将执行的命令
    python eval.py --suite main --base-url http://127.0.0.1:8000/v1 --arm deeprecur
    python eval.py --resolve-tasks --suite main           # 在有 lmms-eval 的机器上核对 task 名
    python eval.py --selftest                             # 离线自检（不需要 harness/数据/网络）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.stage2_eval.common.benchmarks import (
    HARNESS,
    HARNESS_PIP,
    SUITES,
    TRAIN_OBSERVED,
    paper_table,
    suite_benchmarks,
    task_names,
)

STAGE = "stage2_eval"


def build_command(
    *,
    tasks: list[str],
    output_path: Path,
    base_url: str,
    model_version: str,
    batch_size: int,
    limit: int | None,
    extra: list[str] | None = None,
) -> list[str]:
    """组装 lmms-eval 命令（openai_compatible 后端接 vLLM 端点）。"""
    binary = os.environ.get("LMMS_EVAL_BIN", "lmms-eval")
    command = [
        binary,
        "--model",
        "openai_compatible",
        "--model_args",
        f"model_version={model_version},base_url={base_url},api_key=EMPTY",
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]
    if limit:
        command += ["--limit", str(limit)]
    command += list(extra or [])
    return command


def dry_run(suite: str, output_path: Path, base_url: str, model_version: str, batch_size: int, limit: int | None) -> int:
    """打印将执行的命令与套件对照（不要求装 harness）。"""
    tasks = task_names(suite)
    print(paper_table())
    print(f"[deeprecur·eval] 套件 {suite} 将跑 {len(tasks)} 个 task：{tasks}")
    print("[deeprecur·eval] lmms-eval 命令：")
    print("  " + " ".join(build_command(
        tasks=tasks, output_path=output_path, base_url=base_url,
        model_version=model_version, batch_size=batch_size, limit=limit,
    )))
    print(f"[deeprecur·eval] 结果落 {output_path}（lmms-eval 写 results.json，本阶段再收成 summary.json）")
    return 0


def collect_results(results_json: Path, suite: str, out_dir: Path) -> dict:
    """把 lmms-eval 的 results.json 收成论文表格形态的 summary.json。

    ``results.json`` 结构（lmms-eval 惯例）::

        {"results": {"<task>": {"<metric>": value, ...}, ...}, "configs": {...}}
    """
    data = json.loads(Path(results_json).read_text(encoding="utf-8"))
    results = data.get("results", data)
    summary: dict = {
        "stage": STAGE,
        "harness": HARNESS,
        "suite": suite,
        "paper": "DeepStack arXiv 2406.04334",
        "table": [],
        "per_task": {},
    }
    for bench in suite_benchmarks(suite):
        row = {"benchmark": bench.paper_name, "tasks": list(bench.tasks), "scores": {}}
        found = False
        for task in bench.tasks:
            if task in results:
                metrics = {k: v for k, v in results[task].items() if isinstance(v, (int, float))}
                row["scores"][task] = metrics
                found = True
        row["matched"] = found
        if bench.paper_name in TRAIN_OBSERVED:
            row["paper_footnote"] = "* 训练图在训练中见过"
        if bench.validation_split:
            row["paper_footnote"] = (row.get("paper_footnote", "") + " ‡ 验证集").strip()
        summary["table"].append(row)
        summary["per_task"].update(row["scores"])
    missing = [row["benchmark"] for row in summary["table"] if not row["matched"]]
    if missing:
        raise SystemExit(
            f"[deeprecur·eval] 这些论文基准在结果里没有（task 名对不上或没跑）：{missing}"
            f"\n  已装 harness 时先 `python eval.py --resolve-tasks --suite {suite}` 核对 task 名"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    print(f"[deeprecur·eval] 汇总（suite={summary['suite']}，harness={summary['harness']}）：")
    for row in summary["table"]:
        scores = "; ".join(
            f"{task}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            for task, metrics in row["scores"].items()
        )
        note = f"  [{row['paper_footnote']}]" if row.get("paper_footnote") else ""
        print(f"  {row['benchmark']:<12} {scores}{note}")


def run_suite(args) -> int:
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 可选后端：OpenCompass（接线复用 shensi/stage3_eval）；默认 lmms-eval
    if args.harness == "opencompass":
        from shensi.recipes.paper.deeprecur.stage2_eval.common import opencompass as oc

        if args.resolve_datasets:
            summary = oc.resolve_datasets(args.suite)
            src = summary.pop("_sources", {})
            print(
                f"[deeprecur·eval][opencompass] 双来源解析："
                f"OpenCompass 配置 {src.get('opencompass_configs')} 个；vlmeval：{src.get('vlmeval')}"
            )
            for paper_name, info in summary.items():
                if "source" in info:
                    print(f"  {paper_name:<12} -> [{info['source']}] {info['name']}")
                else:
                    print(f"  {paper_name:<12} ✗ {info['missing']}")
            return 0
        summary = oc.run(
            args.suite, args.arm, args.base_url, output_path, dry_run=args.dry_run
        )
        if args.dry_run:
            print(f"[deeprecur·eval][opencompass] 配置：{summary['config']}")
            print(f"[deeprecur·eval][opencompass] 命令：{summary['command']}")
            return 0
        print_summary(summary)
        return 0

    tasks = task_names(args.suite)
    if args.resolve_tasks:
        from shensi.recipes.paper.deeprecur.stage2_eval.common.benchmarks import resolve_tasks

        resolved = resolve_tasks(args.suite)
        print("[deeprecur·eval] task 名解析：")
        for paper_name, task in resolved.items():
            print(f"  {paper_name:<12} -> {task}")
        return 0
    if args.dry_run:
        return dry_run(args.suite, output_path, args.base_url, args.arm, args.batch_size, args.limit)

    command = build_command(
        tasks=tasks,
        output_path=output_path,
        base_url=args.base_url,
        model_version=args.arm,
        batch_size=args.batch_size,
        limit=args.limit,
        extra=args.extra,
    )
    print(f"[deeprecur·eval] 跑 {HARNESS}：{' '.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"[deeprecur·eval] 找不到 {HARNESS}：{exc}\n"
            f"  安装：pip install {HARNESS_PIP}；端点由 `python serve.py --arm {args.arm} --ckpt <目录>` 起"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[deeprecur·eval] {HARNESS} 退出码 {exc.returncode}（看上面的日志）") from exc

    results_json = _find_results(output_path)
    summary = collect_results(results_json, args.suite, output_path)
    print_summary(summary)
    return 0


def _find_results(output_path: Path) -> Path:
    """找 lmms-eval 写的 results.json（可能在 <output>/<model_version>/ 下）。"""
    direct = output_path / "results.json"
    if direct.is_file():
        return direct
    candidates = sorted(output_path.glob("**/results.json"))
    if not candidates:
        raise SystemExit(f"[deeprecur·eval] 在 {output_path} 下没找到 results.json")
    return candidates[-1]


def selftest() -> int:
    """离线自检：命令组装 + 结果收集 + 缺 harness 的显式报错。**不需要 lmms-eval/数据/网络。**"""
    import tempfile

    checks: list[str] = []

    # 1) 论文套件完整性：四个套件的论文基准名与论文表格逐条一致
    expect = {
        "main": ["VQAv2", "GQA", "TextVQA", "DocVQA", "InfoVQA", "SEED", "POPE", "MMMU", "MM-Vet"],
        "text": ["ChartQA", "DocVQA", "InfoVQA", "MultiDocVQA", "TextVQA"],
        "video": ["EgoSchema", "NextQA", "MSVD", "ActivityNet"],
        "ablation": ["GQA", "POPE", "SEED", "TextVQA", "DocVQA", "ChartQA", "InfoVQA"],
    }
    for suite, names in expect.items():
        got = [b.paper_name for b in suite_benchmarks(suite)]
        assert got == names, f"套件 {suite} 与论文不符：{got} != {names}"
    checks.append("论文四套件基准名逐条一致（Table 1/2/3/4-8）")

    # 2) 命令组装：main 套件的 task 列表 = 论文明细
    tasks = task_names("main")
    command = build_command(
        tasks=tasks, output_path=Path("/tmp/x"), base_url="http://127.0.0.1:8000/v1",
        model_version="deeprecur", batch_size=1, limit=8,
    )
    assert command[0].endswith("lmms-eval") or "lmms-eval" in command[0]
    assert command[command.index("--tasks") + 1] == ",".join(tasks)
    assert command[command.index("--model") + 1] == "openai_compatible"
    checks.append(f"lmms-eval 命令组装正确（main 套件 {len(tasks)} 个 task）")

    # 3) 结果收集：造一份 lmms-eval 形态的 results.json → summary.json（论文表格形态）
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results = {"results": {task: {f"{task}_acc": 0.5 + i / 100} for i, task in enumerate(tasks)}}
        (tmp / "results.json").write_text(json.dumps(results), encoding="utf-8")
        summary = collect_results(tmp / "results.json", "main", tmp)
        assert len(summary["table"]) == len(expect["main"])
        assert all(row["matched"] for row in summary["table"])
        assert (tmp / "summary.json").is_file()
        assert any("训练图在训练中见过" in (row.get("paper_footnote") or "") for row in summary["table"])
        checks.append("results.json → summary.json 收集正确（含论文 * 标注）")

    # 4) 缺基准显式报错
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "results.json").write_text(json.dumps({"results": {}}), encoding="utf-8")
        try:
            collect_results(tmp / "results.json", "main", tmp)
        except SystemExit as exc:
            assert "对不上" in str(exc) or "没有" in str(exc)
        else:
            raise AssertionError("缺基准没有报错")
    checks.append("结果缺基准时显式报错（不静默）")

    # 5) 缺 harness 的提示（本机未装 lmms-eval，resolve 应给安装指引）
    try:
        from shensi.recipes.paper.deeprecur.stage2_eval.common.benchmarks import resolve_tasks

        resolve_tasks("main")
    except SystemExit as exc:
        assert HARNESS_PIP in str(exc), f"缺 harness 的提示应包含安装指引：{exc}"
        checks.append(f"缺 {HARNESS} 时显式报错并给安装指引")
    else:
        checks.append(f"本机已装 {HARNESS}（跳过缺件检查）")

    # 6) 已核实的 harness 口径登记在案（两代版本 + MSVD 缺口 + 候选名按版本排序）
    from shensi.recipes.paper.deeprecur.stage2_eval.common.benchmarks import (
        HARNESS_VERIFIED,
        KNOWN_MISSING_IN_HARNESS,
    )

    assert HARNESS_VERIFIED["paper_era"] and HARNESS_VERIFIED["current"]
    assert "MSVD" in KNOWN_MISSING_IN_HARNESS
    multidoc = next(b for b in suite_benchmarks("text") if b.paper_name == "MultiDocVQA")
    assert multidoc.tasks[0] == "multidocvqa_val", multidoc.tasks
    textvqa = next(b for b in suite_benchmarks("main") if b.paper_name == "TextVQA")
    assert textvqa.tasks[0] == "textvqa_val" and "textvqa" in textvqa.tasks
    checks.append(
        f"harness 口径已登记：论文年代 {HARNESS_VERIFIED['paper_era']}（裸名）/ 当前 "
        f"{HARNESS_VERIFIED['current']}（后缀名）；MSVD 缺口显式登记；候选名按版本排序"
    )

    # 7) OpenCompass 可选后端：映射 + 汇总 + 缺件报错
    from shensi.recipes.paper.deeprecur.stage2_eval.common import opencompass as oc

    for line in oc.selftest():
        checks.append(f"opencompass：{line}")

    print("[deeprecur·eval] 离线自检：")
    for line in checks:
        print(f"  ✓ {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deeprecur 评测（基准套件 + 双 harness）")
    parser.add_argument("--suite", default="main", choices=sorted(SUITES))
    parser.add_argument(
        "--harness",
        default="lmms-eval",
        choices=["lmms-eval", "opencompass"],
        help="评测 harness：default lmms-eval；opencompass 为可选后端",
    )
    parser.add_argument("--arm", default="deeprecur", help="被评的臂（也是 lmms-eval 的 model_version 名）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM 端点（serve.py 起）")
    parser.add_argument("--output", default=None, help="结果目录（默认 <FS>/shensi/runs/deeprecur/stage2_eval/<arm>-<suite>）")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="每任务限量（冒烟用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    parser.add_argument("--resolve-tasks", action="store_true", help="在装了 lmms-eval 的机器上核对 task 名")
    parser.add_argument(
        "--resolve-datasets",
        action="store_true",
        help="（--harness opencompass）在装了 OpenCompass 的机器上核对数据集名",
    )
    parser.add_argument("--selftest", action="store_true", help="离线自检")
    parser.add_argument("--extra", action="append", default=[], help="透传给 harness 的额外参数")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.output is None:
        from shensi.recipes.paper.deeprecur.common.paths import env_paths

        args.output = str(Path(env_paths()["runs"]) / STAGE / f"{args.arm}-{args.suite}")
    return run_suite(args)


if __name__ == "__main__":
    sys.exit(main())
