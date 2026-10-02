#!/usr/bin/env python3
"""OpenCompass 执行层：生成配置、选数据集、起 CLI 与汇总分数。"""

from __future__ import annotations

import ast
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmarks  # noqa: E402  口径表（数据集名子串 / 集合 / 参考分数）

LEADERBOARD_COLLECTION = "opencompass.configs.dataset_collections.chat_OC15"


TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "common" / "tokenizer" / "MiniCPM5-2B"


def venv_python(cfg: dict) -> Path:
    """定位 OpenCompass 所在 venv 的 python（配置 → 环境变量 → 逐级父目录找）。"""
    raw = (cfg.get("opencompass") or {}).get("venv") or os.environ.get("SHENSI_OPENCOMPASS_VENV")
    candidates = [Path(str(raw))] if raw else []
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / ".venv-opencompass")
    for cand in candidates:
        py = cand / "bin" / "python"
        if py.is_file():
            return py
    raise SystemExit(
        "找不到 OpenCompass 的 venv：给 `opencompass.venv: <路径>` 或 SHENSI_OPENCOMPASS_VENV；"
        "装法见 stage4_eval/README.md 的「OpenCompass」一节（bash setup_env.sh 会装）"
    )


def package_root(py: Path) -> Path:
    """Venv 里 opencompass 包的位置。"""
    out = subprocess.run(
        [
            str(py),
            "-c",
            "import opencompass,pathlib;print(pathlib.Path(opencompass.__file__).parent)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"这个 venv 里没有 opencompass：{py}\n{out.stderr[-400:]}")
    return Path(out.stdout.strip())


def discover_datasets(py: Path, pattern: str | None = None) -> list[str]:
    """枚举安装包里的数据集配置模块，可按名字子串过滤。"""
    root = package_root(py) / "configs" / "datasets"
    mods = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(root).with_suffix("")
        if pattern and pattern.lower() not in str(rel).lower():
            continue
        mods.append("opencompass.configs.datasets." + ".".join(rel.parts))
    return mods


def _dataset_names(tree: ast.Module) -> list[str]:
    """模块里以 ``_datasets`` 结尾的顶层名字（赋值与 import 都算，函数体不算）。"""
    names: list[str] = []

    def visit(node) -> None:
        for child in getattr(node, "body", []) or []:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_datasets"):
                        names.append(target.id)
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    candidate = alias.asname or alias.name
                    if candidate.endswith("_datasets"):
                        names.append(candidate)
            visit(child)

    visit(tree)
    return list(dict.fromkeys(names))


def dataset_vars(py: Path, mods: list[str]) -> dict[str, list[str]]:
    """这些数据集模块里各自的 ``*_datasets`` 名字叫什么：读源码，不 import。

    OpenCompass 的数据集配置大量用 ``with read_base():`` + ``dataset.deepcopy()`` 这类
    只有它自己的解析器才成立的写法，直接 import 会炸（如 ruler_128k_gen）。
    """
    root = package_root(py)
    out: dict[str, list[str]] = {}
    for mod in mods:
        path = root / Path(*mod.split(".")[1:]).with_suffix(".py")
        if not path.is_file():
            raise SystemExit(f"数据集模块找不到文件：{mod}（{path}）")
        names = _dataset_names(ast.parse(path.read_text(encoding="utf-8")))
        if not names:
            raise SystemExit(f"{mod} 里没有以 _datasets 结尾的顶层名字，不能这样展开")
        out[mod] = names
    return out


def _expand(entry: str) -> list[str]:
    members = benchmarks.OC_SETS.get(entry)
    if members:
        return [benchmarks.oc_name(name) for name in members]
    return [entry]


def pick_datasets(cfg: dict, py: Path) -> tuple[str, list[str]]:
    """按配置选数据集：自带集合 / 口径集合 / 指定名字 / 全部。"""
    preset = str((cfg.get("opencompass") or {}).get("datasets") or "leaderboard")
    if preset == "leaderboard":
        return "collection", []
    mods = discover_datasets(py)
    if preset == "all":
        return "modules", mods
    names = [s.strip() for s in preset.split(",") if s.strip()]
    picked: list[str] = []
    misses: list[str] = []
    for name in names:
        for sub in _expand(name):
            hit = next((m for m in mods if sub.lower() in m.lower()), None)
            if hit is None:
                misses.append(sub)
            elif hit not in picked:
                picked.append(hit)
    if misses:
        raise SystemExit(f"这些名字在 OpenCompass 里没匹配到数据集：{misses}")
    return "modules", picked


def build_config(cfg: dict, out_dir: Path) -> Path:
    """生成一份 OpenCompass 配置：模型指向本机端点，数据集按选择展开。"""
    oc = cfg.get("opencompass") or {}
    ep = cfg["endpoint"]
    mode, mods = pick_datasets(cfg, venv_python(cfg))
    lines = ["from mmengine.config import read_base", "", "with read_base():"]
    if mode == "collection":
        lines.append(f"    from {LEADERBOARD_COLLECTION} import datasets")
    else:
        for module, names in dataset_vars(venv_python(cfg), mods).items():
            for name in names:
                lines.append(f"    from {module} import {name}")

        lines += [
            "",
            "datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])",
        ]
    limit = int(oc.get("limit") or 0)
    if limit > 0:
        lines += [
            "",
            "for _d in datasets:",
            f"    _d['reader_cfg']['test_range'] = '[0:{limit}]'",
        ]
    lines += [
        "",
        "from opencompass.models import OpenAI",
        "",
        "models = [",
        "    dict(",
        "        type=OpenAI,",
        f"        abbr={oc.get('abbr', 'looma')!r},",
        f"        path={ep.get('model', 'looma')!r},",
        f"        openai_api_base={ep['base_url'].rstrip('/') + '/v1/chat/completions'!r},",
        "        key='EMPTY',",
        f"        max_seq_len={int(oc.get('max_seq_len', 32768))},",
        f"        max_out_len={int(oc.get('max_out_len', 1024))},",
        f"        batch_size={int(oc.get('batch_size', 8))},",
        f"        query_per_second={float(oc.get('query_per_second', 4))},",
        f"        retry={int(oc.get('retry', 2))},",
        f"        mode={str(oc.get('mode', 'none'))!r},",
        f"        tokenizer_path={str(oc.get('tokenizer') or TOKENIZER_DIR)!r},",
        "        temperature=0.0,",
        "    ),",
        "]",
    ]
    path = out_dir / "opencompass_config.py"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_command(cfg: dict, conf: Path, work_dir: Path) -> list[str]:
    """组装 OpenCompass 的命令行。"""
    oc = cfg.get("opencompass") or {}
    cmd = [
        str(venv_python(cfg).parent / "opencompass"),
        str(conf),
        "-w",
        str(work_dir),
        "--max-num-workers",
        str(int(oc.get("max_num_workers", 8))),
    ]
    if oc.get("debug"):
        cmd.append("--debug")
    return cmd


def collect(work_dir: Path) -> dict:
    """把 OpenCompass 的 summary 汇总成 {数据集: 分数} 并与参考分对照。"""
    csvs = sorted(work_dir.rglob("summary/summary_*.csv"))
    if not csvs:
        return {"card": {}, "overall": 0.0, "summary_csv": None, "reference": {}}
    with open(csvs[-1], encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    card: dict[str, float] = {}
    for row in rows:
        name = (row.get("dataset") or "").strip()

        raw = row.get("score") or row.get("accuracy") or (list(row.values())[-1] if row else None)
        try:
            score = float(raw) if raw is not None else float("nan")
        except ValueError:
            score = float("nan")
        if name and score == score:
            card[name] = score
    overall = sum(card.values()) / len(card) if card else 0.0
    return {
        "card": card,
        "overall": overall,
        "summary_csv": str(csvs[-1]),
        "reference": benchmarks.compare_references(card),
    }


def run(cfg: dict, out_dir: Path, dry_run: bool = False) -> dict:
    """跑一次 OpenCompass：写配置 → 起 CLI → 汇总。"""
    work_dir = out_dir / "opencompass"
    conf = build_config(cfg, out_dir)
    cmd = build_command(cfg, conf, work_dir)
    print(f"[eval][opencompass] 配置：{conf}")
    print("[eval][opencompass] 命令：\n  " + " \\\n    ".join(cmd))
    if dry_run:
        return {"config": str(conf), "command": " ".join(cmd)}
    work_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(work_dir),
        env={**os.environ, "OPENAI_API_KEY": "EMPTY"},
    )
    if proc.returncode != 0:
        raise SystemExit(f"[eval][opencompass] 退出码 {proc.returncode}")
    result = collect(work_dir)
    print(f"[eval][opencompass] 数据集 {len(result['card'])} 个，均值 {result['overall']:.4f}")
    for name, row in result["reference"].items():
        ref = row.get("reference")
        got = row.get("score")
        delta = "—" if ref is None or got is None else f"{got - ref:+.2f}"
        print(
            f"  {name:<28} 实测 {got if got is not None else float('nan'):.2f}  参考 {ref if ref is not None else '—'}  Δ {delta}"
        )
    return result


def selftest() -> int:
    """离线自检：配置生成、命令组装与 summary 解析。"""
    out = Path(os.environ.get("SHENSI_FS", "/tmp")) / "shensi/runs/looma/stage4_eval/selftest"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "endpoint": {"base_url": "http://127.0.0.1:8000", "model": "looma"},
        "opencompass": {"datasets": "leaderboard", "abbr": "looma"},
    }
    conf = build_config(cfg, out)
    text = conf.read_text(encoding="utf-8")
    cmd = build_command(cfg, conf, out / "opencompass")
    checks = [
        (
            "配置里有端点（/v1/chat/completions）",
            "http://127.0.0.1:8000/v1/chat/completions" in text,
        ),
        ("配置里 import leaderboard 集合", LEADERBOARD_COLLECTION in text),
        ("配置里 import OpenAI 模型", "from opencompass.models import OpenAI" in text),
        ("命令走 venv 的 opencompass 入口", cmd[0].endswith("opencompass")),
        (
            "口径表：集合里每一项都有 OpenCompass 名",
            all(benchmarks.oc_name(n) for n in benchmarks.OC_SETS["mini"]),
        ),
    ]
    try:
        mods = discover_datasets(venv_python(cfg), "mmlu")
        checks.append((f"数据集枚举（mmlu 命中 {len(mods)} 个配置）", len(mods) > 0))
    except SystemExit as exc:
        print("[opencompass 自检] 跳过数据集枚举：", str(exc)[:90])
    fake = out / "opencompass/fake/summary/summary_1.csv"
    fake.parent.mkdir(parents=True, exist_ok=True)

    fake.write_text(
        "dataset,version,metric,mode,looma\nmmlu,1,accuracy,gen,0.42\ngsm8k,1,accuracy,gen,0.30\n",
        encoding="utf-8",
    )
    got = collect(out / "opencompass")
    checks.append(("summary 解析", abs(got["overall"] - 0.36) < 1e-9 and len(got["card"]) == 2))
    checks.append(("参考分对照（mmlu/gsm8k 命中口径表）", len(got["reference"]) == 2))
    ok = True
    for name, passed in checks:
        print(f"[opencompass 自检] {name} → {'✓' if passed else '✗'}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
