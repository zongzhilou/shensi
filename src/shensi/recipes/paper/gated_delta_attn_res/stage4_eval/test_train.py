"""评测的预检与冒烟：生成小批题目并给 tiny 检查点评分。"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from shensi import runtime  # noqa: F401

E = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    checks: list[tuple[str, bool, str]] = []

    for name in ("make_depth_retrieval", "run_depth_retrieval"):
        try:
            _load(f"github_{name}", E / f"{name}.py")
            checks.append((f"import {name}", True, "ok"))
        except Exception as exc:  # noqa: BLE001
            checks.append((f"import {name}", False, f"{type(exc).__name__}: {exc}"))

    ckpt = Path("/tmp/gdar_eval_tiny_hf")
    if not (ckpt / "config.json").is_file():
        try:
            from shensi.recipes.paper.gated_delta_attn_res.common.models.vllm import tiny_checkpoint

            info = tiny_checkpoint.build("gdar", ckpt, overwrite=True)
            print(f"  [·] 自建 tiny HF 目录：{ckpt}（{info['parameters']:,} 参数，随机权重）")
        except Exception as exc:  # noqa: BLE001
            print(f"  [·] 自建 tiny HF 目录失败：{type(exc).__name__}: {exc}")
    has_ckpt = (ckpt / "config.json").is_file() and (ckpt / "tokenizer.json").is_file()
    checks.append(("评测模型目录（tiny ckpt）", has_ckpt, str(ckpt)))

    ok = all(c[1] for c in checks)
    for name, good, note in checks:
        print(f"  [{'✓' if good else '✗'}] {name}：{note}")
    if not ok or args.skip_eval:
        print(f"[test_train:stage4_eval] {'PASS（仅预检）' if ok else 'FAIL'}")
        return 0 if ok else 1

    out = Path("/tmp/gdar_eval_smoke")
    out.mkdir(parents=True, exist_ok=True)
    data = out / "dr_smoke.jsonl"
    py = sys.executable
    print(f"[test_train:stage4_eval] 生成 {args.n} 题 → {data}")
    subprocess.run(
        [
            py,
            str(E / "make_depth_retrieval.py"),
            "--out",
            str(data),
            "--n",
            str(args.n),
            "--lengths",
            "512",
            "--ks",
            "1,2,4",
            "--seed",
            "7",
            "--filler-random",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print("[test_train:stage4_eval] 评分（tiny ckpt，随机权重 → 期望在 chance 附近）")
    r = subprocess.run(
        [
            py,
            str(E / "run_depth_retrieval.py"),
            "--model",
            str(ckpt),
            "--data",
            str(data),
            "--device",
            "cuda",
            "--out-json",
            str(out / "score.json"),
        ],
        capture_output=True,
        text=True,
    )
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()][-6:]
    print("\n".join("  " + ln[:140] for ln in tail))
    ok = r.returncode == 0 and (out / "score.json").is_file()
    print(f"[test_train:stage4_eval] {'PASS' if ok else 'FAIL'}（分数：{out / 'score.json'}）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
