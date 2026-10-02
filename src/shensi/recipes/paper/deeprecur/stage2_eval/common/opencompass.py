"""OpenCompass 可选后端：基准名 → OpenCompass 数据集映射与结果汇总。

执行接线复用 ``shensi/stage3_eval/opencompass_eval``（端点模型配置 / 数据集枚举 /
summary CSV 解析）；本模块只做映射与汇总。数据集名是候选：``--resolve-datasets`` 在装了
OpenCompass 的机器上双来源核对（OpenCompass 配置 + VLMEvalKit 注册表），对不上显式报错；
视频套件在该 harness 下无对应数据集，显式登记为缺口。
"""

from __future__ import annotations

import json
from pathlib import Path

from .benchmarks import (
    TRAIN_OBSERVED,
    suite_benchmarks,
)

#: 论文基准 → OpenCompass 数据集目录名候选（对不上会报错，不静默）
PAPER_TO_OPENCOMPASS: dict[str, tuple[str, ...]] = {
    "VQAv2": ("vqav2",),
    "GQA": ("gqa",),
    "TextVQA": ("textvqa",),
    "DocVQA": ("docvqa",),
    "InfoVQA": ("infovqa",),
    "SEED": ("seedbench",),
    "POPE": ("pope",),
    "MMMU": ("mmmu",),
    "MM-Vet": ("mmvet",),
    "ChartQA": ("chartqa",),
    "MultiDocVQA": ("multidocvqa",),
    # 视频套件（Table 3）：OpenCompass 无对应数据集
    "EgoSchema": (),
    "NextQA": (),
    "MSVD": (),
    "ActivityNet": (),
}

#: OpenCompass 没有的论文基准（显式登记）
MISSING_IN_OPENCOMPASS = ("EgoSchema", "NextQA", "MSVD", "ActivityNet")


def _glue():
    """OpenCompass 执行接线（懒导入 shensi/stage3_eval）。"""
    from shensi.recipes.shensi.stage3_eval import opencompass_eval as glue

    return glue


def build_config_dict(*, suite: str, arm: str, base_url: str, abbr: str | None = None) -> dict:
    """组 OpenCompass 的 cfg（供复用的 glue.build_config / build_command 消费）。

    数据集用 `datasets` 字段显式列论文基准候选名——glue 会按目录名精确/子串匹配，
    匹配不到就报错。
    """
    names = []
    missing = []
    for bench in suite_benchmarks(suite):
        if not PAPER_TO_OPENCOMPASS.get(bench.paper_name):
            missing.append(bench.paper_name)
            continue
        names.append(PAPER_TO_OPENCOMPASS[bench.paper_name][0])
    if not names:
        raise SystemExit(
            f"[deeprecur·eval][opencompass] 套件 {suite} 全是没有对应数据集的基准：{missing}"
        )
    return {
        "endpoint": {"base_url": base_url, "model": arm},
        "opencompass": {
            "datasets": ",".join(dict.fromkeys(names)),
            "abbr": abbr or arm,
        },
    }


def _vlmeval_names(py) -> tuple[list[str], str]:
    """问 OpenCompass 的 venv 要 VLMEvalKit 的数据集名（OpenCompass 的 VL 基准走它）。

    返回 ``(名字列表, 状态说明)``；vlmeval 未装时返回空列表 + 说明（不静默）。
    """
    import subprocess

    probe = (
        "import json\n"
        "try:\n"
        "    import vlmeval.dataset as d\n"
        "except Exception as e:\n"
        "    print(json.dumps({'names': [], 'why': f'vlmeval 未装（{type(e).__name__}）'}))\n"
        "else:\n"
        "    names = getattr(d, 'dataset_list', None) or getattr(d, 'SUPPORTED_DATASETS', None) or []\n"
        "    if callable(names):\n"
        "        names = names()\n"
        "    print(json.dumps({'names': list(names), 'why': 'ok'}))\n"
    )
    out = subprocess.run([str(py), "-c", probe], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return [], f"vlmeval 探测失败：{out.stderr.strip()[-120:]}"
    try:
        import json as _json

        table = _json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return [], "vlmeval 探测输出不可解析"
    return list(table.get("names") or []), str(table.get("why", "ok"))


def resolve_datasets(suite: str) -> dict:
    """在两个来源里解析论文基准的数据集名（装了 OpenCompass 的机器上跑）。

    来源 1：OpenCompass 的数据集配置模块（`configs/datasets/**`）；
    来源 2：**VLMEvalKit 注册表**（OpenCompass 的 VL 基准走它）。

    返回 ``{论文基准: {"source": ..., "name": ...} | {"missing": 原因}}``——两处都没有的
    显式给出原因（含本机薄装情况），不静默。
    """
    glue = _glue()
    py = glue.venv_python({"opencompass": {}})
    mods = glue.discover_datasets(py)
    vl_names, vl_why = _vlmeval_names(py)
    out: dict = {"_sources": {"opencompass_configs": len(mods), "vlmeval": vl_why}}
    for bench in suite_benchmarks(suite):
        cands = PAPER_TO_OPENCOMPASS.get(bench.paper_name) or ()
        hit = next(
            (m for m in mods if any(glue.dataset_dir(m).lower() == c.lower() for c in cands)), None
        )
        if hit:
            out[bench.paper_name] = {"source": "opencompass", "name": hit}
            continue
        vl_hit = next((n for n in vl_names if any(c.lower() in n.lower() for c in cands)), None)
        if vl_hit:
            out[bench.paper_name] = {"source": "vlmeval", "name": vl_hit}
            continue
        out[bench.paper_name] = {
            "missing": (
                "OpenCompass 配置里没有（本机装的是文本向集合）"
                if not cands
                else f"两处都没有：OpenCompass 配置 {len(mods)} 个里无匹配；{vl_why}"
            )
        }
    return out


def run(suite: str, arm: str, base_url: str, out_dir: Path, dry_run: bool = False) -> dict:
    """跑 OpenCompass（复用 glue）：写配置 → CLI → 汇总成与 lmms-eval 同形的 summary。"""
    glue = _glue()
    cfg = build_config_dict(suite=suite, arm=arm, base_url=base_url)
    missing = [
        b.paper_name for b in suite_benchmarks(suite) if not PAPER_TO_OPENCOMPASS.get(b.paper_name)
    ]
    if missing:
        print(
            f"[deeprecur·eval][opencompass] 这些论文基准 OpenCompass 没有对应数据集，将不产出：{missing}"
        )
    result = glue.run(cfg, out_dir, dry_run=dry_run)
    if dry_run:
        return {"harness": "opencompass", "config": result.get("config"), "command": result.get("command")}
    summary = to_summary(suite, result)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def to_summary(suite: str, oc_result: dict) -> dict:
    """把 OpenCompass 的 {数据集: 分数} 收成论文表格形态（与 lmms-eval 后端同形）。"""
    card: dict = oc_result.get("card", {})
    summary: dict = {
        "stage": "stage2_eval",
        "harness": "opencompass",
        "suite": suite,
        "paper": "DeepStack arXiv 2406.04334",
        "table": [],
        "per_task": {},
        "opencompass": {"overall": oc_result.get("overall"), "summary_csv": oc_result.get("summary_csv")},
    }
    for bench in suite_benchmarks(suite):
        row = {"benchmark": bench.paper_name, "tasks": list(PAPER_TO_OPENCOMPASS.get(bench.paper_name, ())), "scores": {}}
        for key, value in card.items():
            if any(cand and (key.lower() == cand.lower() or cand.lower() in key.lower()) for cand in row["tasks"]):
                row["scores"][key] = {"score": value}
        row["matched"] = bool(row["scores"])
        if bench.paper_name in TRAIN_OBSERVED:
            row["paper_footnote"] = "* 训练图在训练中见过"
        if bench.validation_split:
            row["paper_footnote"] = (row.get("paper_footnote", "") + " ‡ 验证集").strip()
        if bench.paper_name in MISSING_IN_OPENCOMPASS:
            row["note"] = "OpenCompass 无对应数据集（用 lmms-eval 后端或自注册）"
        summary["table"].append(row)
        summary["per_task"].update(row["scores"])
    return summary


def selftest() -> list[str]:
    """离线自检（不需要 OpenCompass）：映射完整性 + 汇总转换；缺 venv 时验证显式报错。

    返回通过项列表；任何一项失败直接抛错。
    """
    checks: list[str] = []

    # 1) 映射完整性：四套件里除视频族外都要有候选名；视频族显式登记为缺口
    for suite in ("main", "text", "ablation"):
        for bench in suite_benchmarks(suite):
            assert PAPER_TO_OPENCOMPASS.get(bench.paper_name), (
                f"{suite}/{bench.paper_name} 在 OpenCompass 映射里没有候选名"
            )
    for name in MISSING_IN_OPENCOMPASS:
        assert name in PAPER_TO_OPENCOMPASS and not PAPER_TO_OPENCOMPASS[name]
    checks.append("映射完整性：main/text/ablation 全有候选名；视频 4 项显式登记为 OpenCompass 缺口")

    # 2) cfg 组装：显式列出候选名，且视频套件要显式报错
    cfg = build_config_dict(suite="main", arm="deeprecur", base_url="http://127.0.0.1:8000")
    assert cfg["endpoint"]["model"] == "deeprecur"
    assert "vqav2" in cfg["opencompass"]["datasets"] and "mmvet" in cfg["opencompass"]["datasets"]
    checks.append(f"cfg 组装：datasets={cfg['opencompass']['datasets']}")

    # 3) 汇总转换：造一份 OpenCompass card → 论文表格形态（含 * 标注与缺口 note）
    fake = {
        "card": {"vqav2": 0.71, "gqa": 0.55, "textvqa": 0.48, "docvqa": 0.62, "infovqa": 0.4,
                 "seedbench": 0.66, "pope": 0.87, "mmmu": 0.35, "mmvet": 0.28},
        "overall": 0.5,
        "summary_csv": "x.csv",
    }
    summary = to_summary("main", fake)
    rows = {row["benchmark"]: row for row in summary["table"]}
    assert abs(rows["VQAv2"]["scores"]["vqav2"]["score"] - 0.71) < 1e-9
    assert all(row["matched"] for row in summary["table"])
    assert "训练图在训练中见过" in (rows["TextVQA"].get("paper_footnote") or "")
    checks.append("汇总转换：9 项全部对上、分数正确、论文 * 标注带上")

    fake_video = {"card": {"egoschema": 0.3}, "overall": 0.3, "summary_csv": "y.csv"}
    video = to_summary("video", fake_video)
    notes = {row["benchmark"]: row.get("note", "") for row in video["table"]}
    assert all("无对应数据集" in notes[name] for name in MISSING_IN_OPENCOMPASS)
    checks.append("视频套件：4 项都带「OpenCompass 无对应数据集」的显式 note")

    # 4) 缺 venv 时的显式报错（本机没装 OpenCompass 的 venv，走的是这条）
    try:
        _glue().venv_python({"opencompass": {}})
    except SystemExit as exc:
        assert "OpenCompass 的 venv" in str(exc), exc
        checks.append("缺 OpenCompass venv 时显式报错并给安装指引")
    else:
        checks.append("本机有 OpenCompass venv（跳过缺件检查）")

    # 5) 双来源解析（有 venv 时真跑一遍）：每项要么给来源、要么给明确缺口原因
    try:
        resolved = resolve_datasets("main")
    except SystemExit as exc:
        checks.append(f"无 venv 时解析显式报错（预期）：{str(exc)[:60]}")
    else:
        src = resolved.pop("_sources")
        assert src.get("opencompass_configs") is not None
        for paper_name, info in resolved.items():
            assert ("source" in info) != ("missing" in info), (paper_name, info)
            if "missing" in info:
                assert "vlmeval" in info["missing"], info
        resolved_n = sum(1 for v in resolved.values() if "source" in v)
        checks.append(
            f"双来源解析：{resolved_n}/{len(resolved)} 命中（OpenCompass 配置 "
            f"{src['opencompass_configs']} 个；vlmeval：{src['vlmeval']}）；缺口都带原因"
        )
    return checks


if __name__ == "__main__":
    for line in selftest():
        print(f"  ✓ {line}")
