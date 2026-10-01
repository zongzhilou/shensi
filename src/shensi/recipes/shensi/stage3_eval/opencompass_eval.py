#!/usr/bin/env python3
"""OpenCompass 评测：本机 vLLM 起 OpenAI 兼容端点 → OpenCompass 跑 LLM 基准。

三件事：

1. **数据集**：默认跑 leaderboard 那套集合（OpenCompass 自带的
   `configs/dataset_collections/chat_OC15.py`：mmlu / cmmlu / ceval / GaokaoBench / triviaqa / nq /
   race / winogrande / hellaswag / bbh / gsm8k / math / TheoremQA / humaneval / mbpp / gpqa / IFEval）；
   `opencompass.datasets: all` 展开安装包里 `configs/datasets/**` 的**全部**数据集配置；
   也可以给逗号分隔的名字（按目录/文件名匹配，例如 `mmlu,gsm8k,humaneval`）。
2. **模型**：`opencompass.models.openai_api.OpenAI` 指向 `endpoint.base_url`（vLLM serve 的端点），
   `key` 给占位串（本地服务不校验）。
3. **产物**：OpenCompass 的 work_dir（predictions / results / summary）落在评测输出目录下，
   再把 `summary/summary_*.csv` 汇总进我们的 `summary.json`。

OpenCompass 装在独立 venv（`opencompass.venv` 或环境变量 `SHENSI_OPENCOMPASS_VENV`，默认仓库根的
`.venv-opencompass`）：它的依赖要求 numpy<2，与训练侧（verl/vllm 要 numpy 2.x）冲突，所以不装进训练 venv。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

LEADERBOARD_COLLECTION = "opencompass.configs.dataset_collections.chat_OC15"


def venv_python(cfg: dict) -> Path:
    """OpenCompass 所在 venv 的 python（配置 `opencompass.venv` 优先，其次环境变量与仓库根）。"""
    raw = (cfg.get("opencompass") or {}).get("venv") or os.environ.get("SHENSI_OPENCOMPASS_VENV")
    candidates = [Path(str(raw))] if raw else []
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / ".venv-opencompass")
    for cand in candidates:
        py = cand / "bin" / "python"
        if py.is_file():
            return py
    raise SystemExit(
        "找不到 OpenCompass 的 venv：给 `opencompass.venv: <路径>` 或 SHENSI_OPENCOMPASS_VENV，"
        "装法见 stage3_eval/README.md 的「OpenCompass」一节"
    )


def package_root(py: Path) -> Path:
    """venv 里 opencompass 包的位置（用来枚举数据集配置）。"""
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
    for f in sorted(root.rglob("*.py")):
        if f.name == "__init__.py":
            continue
        rel = f.relative_to(root).with_suffix("")
        if pattern and pattern.lower() not in str(rel).lower():
            continue
        mods.append("opencompass.configs.datasets." + ".".join(rel.parts))
    return mods


def pick_datasets(cfg: dict, py: Path) -> tuple[str, list[str]]:
    """返回 (模式, 模块列表)：leaderboard 模式走自带集合，其余走数据集模块枚举。"""
    preset = str((cfg.get("opencompass") or {}).get("datasets") or "leaderboard")
    if preset == "leaderboard":
        return "collection", []
    mods = discover_datasets(py)
    if preset == "all":
        return "modules", mods
    names = [s.strip() for s in preset.split(",") if s.strip()]
    picked = [m for m in mods if any(n.lower() in m.lower() for n in names)]
    if not picked:
        raise SystemExit(f"这些名字在 OpenCompass 里没匹配到数据集：{names}")
    return "modules", picked


def build_config(cfg: dict, out_dir: Path) -> Path:
    """生成一份 OpenCompass 配置：模型 = 本机端点，数据集 = 集合 / 指定 / 全部。"""
    oc = cfg.get("opencompass") or {}
    ep = cfg["endpoint"]
    mode, mods = pick_datasets(cfg, venv_python(cfg))
    lines = [
        "from mmengine.config import read_base",
        "",
        "with read_base():",
    ]
    if mode == "collection":
        lines.append(f"    from {LEADERBOARD_COLLECTION} import datasets")
    else:
        lines += [
            "    import importlib",
            "    import json",
            "    from pathlib import Path",
            "",
            f"    _mods = json.loads(Path({str(out_dir / 'opencompass_modules.json')!r}).read_text())",
            "    _bags = []",
            "    for _name in _mods:",
            "        _mod = importlib.import_module(_name)",
            "        _bags += [v for k, v in vars(_mod).items() if k.endswith('_datasets')]",
            "    datasets = sum(_bags, [])",
        ]
        (out_dir / "opencompass_modules.json").write_text(
            json.dumps(mods, ensure_ascii=False), encoding="utf-8"
        )
    lines += [
        "",
        "from opencompass.models import OpenAI",
        "",
        "models = [",
        "    dict(",
        "        type=OpenAI,",
        f"        abbr={oc.get('abbr', 'shensi')!r},",
        f"        path={ep.get('model', 'shensi')!r},",
        f"        openai_api_base={ep['base_url'].rstrip('/') + '/v1/chat/completions'!r},",
        "        key='EMPTY',",
        f"        max_seq_len={int(oc.get('max_seq_len', 32768))},",
        f"        max_out_len={int(oc.get('max_out_len', 1024))},",
        f"        batch_size={int(oc.get('batch_size', 8))},",
        f"        query_per_second={float(oc.get('query_per_second', 4))},",
        f"        retry={int(oc.get('retry', 2))},",
        "        temperature=0.0,",
        "    ),",
        "]",
    ]
    path = out_dir / "opencompass_config.py"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_command(cfg: dict, conf: Path, work_dir: Path) -> list[str]:
    """`<venv>/bin/opencompass <config> -w <work_dir> --max-num-workers N`。"""
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
    """把 OpenCompass 的 summary csv 汇总成 {数据集: 分数} 与均值。"""
    csvs = sorted(work_dir.rglob("summary/summary_*.csv"))
    if not csvs:
        return {"card": {}, "overall": 0.0, "summary_csv": None}
    with open(csvs[-1], encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    card = {}
    for row in rows:
        name = (row.get("dataset") or "").strip()
        raw = row.get("score") or row.get("accuracy")
        try:
            score = float(raw) if raw is not None else float("nan")
        except ValueError:
            score = float("nan")
        if name and score == score:  # 跳过 NaN
            card[name] = score
    overall = sum(card.values()) / len(card) if card else 0.0
    return {"card": card, "overall": overall, "summary_csv": str(csvs[-1])}


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
    proc = subprocess.run(cmd, check=False, env={**os.environ, "OPENAI_API_KEY": "EMPTY"})
    if proc.returncode != 0:
        raise SystemExit(f"[eval][opencompass] 退出码 {proc.returncode}")
    result = collect(work_dir)
    print(f"[eval][opencompass] 数据集 {len(result['card'])} 个，均值 {result['overall']:.4f}")
    return result


def selftest() -> int:
    """离线自检：配置生成 + summary 解析（不需要 GPU；装了 opencompass 才顺带验数据集枚举）。"""
    out = Path(os.environ.get("SHENSI_FS", "/tmp")) / "shensi/runs/oc_selftest"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "endpoint": {"base_url": "http://127.0.0.1:8000", "model": "shensi"},
        "opencompass": {"datasets": "leaderboard", "abbr": "shensi"},
    }
    conf = build_config(cfg, out)
    text = conf.read_text(encoding="utf-8")
    cmd = build_command(cfg, conf, out / "opencompass")
    checks = [
        ("配置里有端点（/v1/chat/completions）", "http://127.0.0.1:8000/v1/chat/completions" in text),
        ("配置里 import leaderboard 集合", LEADERBOARD_COLLECTION in text),
        ("配置里 import OpenAI 模型", "from opencompass.models import OpenAI" in text),
        ("命令走 venv 的 opencompass 入口", cmd[0].endswith("opencompass")),
    ]
    try:
        mods = discover_datasets(venv_python(cfg), "mmlu")
        checks.append((f"数据集枚举（mmlu 命中 {len(mods)} 个配置）", len(mods) > 0))
    except SystemExit as exc:
        print("[opencompass 自检] 跳过数据集枚举：", str(exc)[:90])
    fake = out / "opencompass/fake/summary/summary_1.csv"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(
        "dataset,version,metric,mode,score\nmmlu,1,accuracy,gen,0.42\ngsm8k,1,accuracy,gen,0.30\n",
        encoding="utf-8",
    )
    got = collect(out / "opencompass")
    checks.append(("summary 解析", abs(got["overall"] - 0.36) < 1e-9 and len(got["card"]) == 2))
    ok = True
    for name, passed in checks:
        print(f"[opencompass 自检] {name} → {'✓' if passed else '✗'}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
