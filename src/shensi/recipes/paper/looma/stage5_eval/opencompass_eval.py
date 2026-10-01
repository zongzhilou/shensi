#!/usr/bin/env python3
"""OpenCompass 评测：本机 vLLM 端点跑 LLM 基准（leaderboard 集合 / 口径集合 / 指定 / 全部）。

OpenCompass 装在**独立 venv**（它要 numpy<2，与训练侧冲突）：`.venv-opencompass`、环境变量
``SHENSI_OPENCOMPASS_VENV`` 或配置 ``opencompass.venv`` 三选一，见 setup_env.sh。

    python opencompass_eval.py --selftest        # 离线自检：配置生成 + summary 解析 + 数据集枚举
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmarks  # noqa: E402  口径表（数据集名子串 / 集合 / 参考分数）

#: OpenCompass 自带的 leaderboard 集合（17 组：mmlu / cmmlu / ceval / Gaokao / triviaqa / nq /
#: race / winogrande / hellaswag / bbh / gsm8k / math / TheoremQA / humaneval / mbpp / gpqa / IFEval）
LEADERBOARD_COLLECTION = "opencompass.configs.dataset_collections.chat_OC15"

#: 默认分词器（本配方的 vendored MiniCPM5-2B，绝对路径）：OpenCompass 按它算输入长度与截断。
TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "common" / "tokenizer" / "MiniCPM5-2B"


def venv_python(cfg: dict) -> Path:
    """OpenCompass 所在 venv 的 python（配置 `opencompass.venv` → 环境变量 → 逐级父目录找）。"""
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
        "装法见 stage5_eval/README.md 的「OpenCompass」一节（bash setup_env.sh 会装）"
    )


def package_root(py: Path) -> Path:
    """Venv 里 opencompass 包的位置（用来枚举数据集配置）。"""
    out = subprocess.run(
        [str(py), "-c", "import opencompass,pathlib;print(pathlib.Path(opencompass.__file__).parent)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"这个 venv 里没有 opencompass：{py}\n{out.stderr[-400:]}")
    return Path(out.stdout.strip())


def discover_datasets(py: Path, pattern: str | None = None) -> list[str]:
    """枚举安装包里的数据集配置模块（`configs/datasets/**/*.py`），可按名字子串过滤。"""
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


_VAR_PROBE = (
    "import importlib,json,sys;"
    "out={};"
    "\nfor _n in json.loads(sys.argv[1]):"
    "\n    _m=importlib.import_module(_n);"
    "\n    out[_n]=[k for k in vars(_m) if k.endswith('_datasets')];"
    "\nprint(json.dumps(out))"
)


def dataset_vars(py: Path, mods: list[str]) -> dict[str, list[str]]:
    """问 venv：这些数据集模块里各自的 ``*_datasets`` 变量叫什麼（配置里要写成字面量 import）。"""
    out = subprocess.run(
        [str(py), "-c", _VAR_PROBE, json.dumps(mods)],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"枚举数据集变量失败：\n{out.stderr[-400:]}")
    return json.loads(out.stdout.strip() or "{}")


def _expand(entry: str) -> list[str]:
    """把一个选择项展开成子串列表：口径集合名 → 其成员的 OpenCompass 名；其余原样。"""
    members = benchmarks.OC_SETS.get(entry)
    if members:
        return [benchmarks.oc_name(name) for name in members]
    return [entry]


def pick_datasets(cfg: dict, py: Path) -> tuple[str, list[str]]:
    """返回 (模式, 模块列表)：``leaderboard`` 走自带集合，其余按名字子串枚举安装包的数据集。

    一个选择项只认**排序后的第一个**匹配：同一数据集在安装包里常有多份带 hash 的等价配置
    （变量名还都一样，多份会互相覆盖），只取一份才既确定又不会重复跑同一套题。
    """
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
    """生成一份 OpenCompass 配置：模型 = 本机端点，数据集 = 集合 / 口径集合 / 指定 / 全部。

    `with read_base()` 里只能写 ``from … import …``（OpenCompass 的解析器这么定的），所以数据集
    一律展开成字面量 import 行；``*_datasets`` 变量名由 venv 报回来。
    """
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
        # 聚合要写在 read_base 外面（OpenCompass 自己的集合文件就是这么写的）：
        # 块里的 `*_datasets` 都进了 locals()，自己 sum 成 datasets。
        lines += ["", "datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])"]
    limit = int(oc.get("limit") or 0)
    if limit > 0:
        # 样本上限走 dataset 的 test_range（OpenCompass 的标准做法；`--debug` 不是按条数限）
        lines += [
            "",
            "for _d in datasets:",
            f"    _d['reader_cfg']['test_range'] = '[0:{limit}]'",
        ]
    limit = int(oc.get("limit") or 0)
    if limit > 0:
        # 样本上限走 dataset 的 test_range（OpenCompass 的标准做法；`--debug` 不是按条数限）
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
    """组装 OpenCompass CLI：`<venv>/bin/opencompass <配置> -w <工作目录> --max-num-workers N`。"""
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
    """把 OpenCompass 的 summary csv 汇总成 {数据集: 分数} 与均值，并附上口径参考分对照。"""
    csvs = sorted(work_dir.rglob("summary/summary_*.csv"))
    if not csvs:
        return {"card": {}, "overall": 0.0, "summary_csv": None, "reference": {}}
    with open(csvs[-1], encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    card: dict[str, float] = {}
    for row in rows:
        name = (row.get("dataset") or "").strip()
        # 分数列的列名是模型 abbr（我们在生成配置里设的那个），所以取最后一列兜底
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
    """跑一次 OpenCompass：写配置 → 起 CLI → 汇总（含与口径参考分的对照）。"""
    work_dir = out_dir / "opencompass"
    conf = build_config(cfg, out_dir)
    cmd = build_command(cfg, conf, work_dir)
    print(f"[eval][opencompass] 配置：{conf}")
    print("[eval][opencompass] 命令：\n  " + " \\\n    ".join(cmd))
    if dry_run:
        return {"config": str(conf), "command": " ".join(cmd)}
    work_dir.mkdir(parents=True, exist_ok=True)
    # cwd 放到 work_dir：OpenCompass 会在 cwd 下写 tmp/ 之类的中间文件，别落到配方目录里
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
        print(f"  {name:<28} 实测 {got if got is not None else float('nan'):.2f}  参考 {ref if ref is not None else '—'}  Δ {delta}")
    return result


def selftest() -> int:
    """离线自检：配置生成 + summary 解析 + 参考分对照（装了 opencompass 才顺带验数据集枚举）。"""
    out = Path(os.environ.get("SHENSI_FS", "/tmp")) / "shensi/runs/looma_stage5_eval/oc_selftest"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "endpoint": {"base_url": "http://127.0.0.1:8000", "model": "looma"},
        "opencompass": {"datasets": "leaderboard", "abbr": "looma"},
    }
    conf = build_config(cfg, out)
    text = conf.read_text(encoding="utf-8")
    cmd = build_command(cfg, conf, out / "opencompass")
    checks = [
        ("配置里有端点（/v1/chat/completions）", "http://127.0.0.1:8000/v1/chat/completions" in text),
        ("配置里 import leaderboard 集合", LEADERBOARD_COLLECTION in text),
        ("配置里 import OpenAI 模型", "from opencompass.models import OpenAI" in text),
        ("命令走 venv 的 opencompass 入口", cmd[0].endswith("opencompass")),
        ("口径表：集合里每一项都有 OpenCompass 名", all(benchmarks.oc_name(n) for n in benchmarks.OC_SETS["mini"])),
    ]
    try:
        mods = discover_datasets(venv_python(cfg), "mmlu")
        checks.append((f"数据集枚举（mmlu 命中 {len(mods)} 个配置）", len(mods) > 0))
    except SystemExit as exc:
        print("[opencompass 自检] 跳过数据集枚举：", str(exc)[:90])
    fake = out / "opencompass/fake/summary/summary_1.csv"
    fake.parent.mkdir(parents=True, exist_ok=True)
    # 真实形状：分数列的列名是模型 abbr
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
