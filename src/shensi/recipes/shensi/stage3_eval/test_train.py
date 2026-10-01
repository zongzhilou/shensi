#!/usr/bin/env python3
"""stage3_eval 的集成预检：这条线不训练，查的是"评测起不来"的东西。"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common, tiny_test
from shensi.recipes.shensi.stage2_rl.reward import compute_score
from shensi.recipes.shensi.stage3_eval import eval as eval_mod

STAGE = "stage3_eval"


def _check_longctx_suite(cfg: dict) -> bool:
    """长文套件自检：MRCR 类题面来自 stage3_longctx 的构建产物，且判分要求按出现顺序全对。"""
    loc = cfg.get("local", {})
    root = Path(str(loc.get("longctx_root", "")))
    mrcr = root / "mrcr_eval.jsonl"
    if not mrcr.is_file():
        print(
            f"  ○ 长文套件：{mrcr} 不在——先跑 "
            "`cd stage0_pretrain/stage3_longctx && python build_longctx.py --step mrcr`（评测与训练共用同一批针）"
        )
        return True
    rows = eval_mod.build_local_prompts({"local": {**loc, "capability_sets": []}})
    mr = [r for r in rows if str(r["capability"]).startswith("mrcr")]
    if not mr:
        print(f"  ✗ 长文套件：{mrcr} 在，但没造出 mrcr 题面")
        return False
    row = mr[0]
    needles = row["ground_truth"].split("、")
    correct = "、".join(needles)
    scrambled = "、".join(reversed(needles))
    good = compute_score(row["capability"], correct, row["ground_truth"])
    bad = compute_score(row["capability"], scrambled, row["ground_truth"])
    print(
        f"  {'✓' if good == 1.0 and bad == 0.0 else '✗'} 长文套件（MRCR 类）：{len(mr)} 条 · "
        f"按序全对={good:.1f} · 顺序错={bad:.1f} · 针数={len(needles)}"
    )
    return good == 1.0 and bad == 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Shensi {STAGE} 集成预检")
    ap.add_argument("--profile", default="tiny")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    results = []
    print(f"[test_train:{STAGE}] 预检（profile={args.profile}）")

    cfg_path = here / f"config/{args.profile}.yaml"
    try:
        cfg = common.resolve_cfg(common.load_yaml(cfg_path))
        results.append(tiny_test.expect(True, "配置解析", str(cfg_path)))
    except Exception as exc:  # noqa: BLE001
        results.append(tiny_test.expect(False, "配置解析", f"{type(exc).__name__}: {exc}"))
        cfg = None

    if cfg:
        try:
            serving = cfg.get("serving", {})
            if serving.get("model_path"):
                cmd = eval_mod.build_vllm_command(serving)
                results.append(
                    tiny_test.expect(bool(cmd), "vllm serve 命令", " ".join(cmd[:4]) + " …")
                )
                model_path = Path(str(serving["model_path"]))
                if model_path.exists():
                    tiny_test.expect(True, "模型目录", str(model_path))
                else:
                    # 档里的路径是占位（生产机上的 ckpt）：这里只提示，不判 FAIL
                    print(
                        f"  ○ 模型目录不在本机（{model_path}）；真跑评测时用 --set serving.model_path=<ckpt> 指过去"
                    )
            else:
                print("  ○ serving.model_path 没配，跳过 serve 命令与模型检查")
        except Exception as exc:  # noqa: BLE001
            results.append(
                tiny_test.expect(False, "vllm serve 命令", f"{type(exc).__name__}: {exc}")
            )

    vllm_cli = shutil.which("vllm") or (Path(sys.executable).parent / "vllm")
    results.append(tiny_test.expect(Path(vllm_cli).exists(), "vllm CLI", str(vllm_cli)))
    results.append(tiny_test.expect(tiny_test.gpu_available(), "GPU 可见"))
    for mod in ("vllm", "transformers", "shensi.runtime"):
        try:
            __import__(mod)
            results.append(tiny_test.expect(True, f"import {mod}"))
        except Exception as exc:  # noqa: BLE001
            results.append(tiny_test.expect(False, f"import {mod}", f"{type(exc).__name__}: {exc}"))
    try:
        out = subprocess.run(
            [str(vllm_cli), "--version"], capture_output=True, text=True, timeout=120
        )
        results.append(
            tiny_test.expect(out.returncode == 0, "vllm 可执行", out.stdout.strip()[:80])
        )
    except Exception as exc:  # noqa: BLE001
        results.append(tiny_test.expect(False, "vllm 可执行", f"{type(exc).__name__}: {exc}"))

    if cfg:
        results.append(_check_longctx_suite(cfg))

    ok = all(results)
    print(f"[test_train:{STAGE}] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
