#!/usr/bin/env python3
"""stage3_longctx 的集成测试：tiny 几何 + 本 stage 的档，跑 5 步并校验收尾；再自检三类语料的构建。

比前两段多覆盖的东西：长序列的 YaRN 位置编码（tiny 档把长度压到 128，但走的是同一套 rope 参数路径）与 `context_parallel_size` 的配置面；数据同 stage2。
构建自检用本地合成长文当源（不碰云端语料）：NextLong / EntropyLong / MRCR（含评测集）三类都要产出，
并检查 MRCR 的「针」在材料里各出现一次、ground_truth 的顺序与出现顺序一致、同种子可复现。

跑法：`cd stage0_pretrain/stage3_longctx && python test_train.py`
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import common, tiny_test

STAGE = "stage3_longctx"


def _check_builder() -> bool:
    """三类语料构建的自检（离线，合成源）。"""
    here = Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="longctx_build_") as tmp:
        tmp = Path(tmp)
        source = tmp / "docs.jsonl"
        with source.open("w", encoding="utf-8") as fh:
            for i in range(8):
                fh.write(
                    json.dumps({"text": f"第 {i} 篇：" + "内容句子。" * 800}, ensure_ascii=False)
                    + "\n"
                )
        out = tmp / "out"
        cmd = [
            sys.executable,
            str(here / "build_longctx.py"),
            "--step",
            "all",
            "--source-jsonl",
            str(source),
            "--out",
            str(out),
            "--target-chars",
            "4000",
            "--synth-target-chars",
            "4000",
            "--needles",
            "4",
            "--items",
            "3",
            "--min-chars",
            "500",
        ]
        rc = subprocess.run(
            cmd, cwd=here, capture_output=True, text=True, env=common.subprocess_env()
        )
        if rc.returncode != 0:
            print(f"[test_train:{STAGE}] 构建器退出码 {rc.returncode}\n{rc.stderr[-600:]}")
            return False

        produced = {
            p.name: [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for p in sorted(out.glob("*.jsonl"))
        }
        for name in (
            "Shensi-Longctx-Synth-NextLong.jsonl",
            "Shensi-Longctx-Synth-EntropyLong.jsonl",
            "Shensi-Longctx-MRCR.jsonl",
            "mrcr_eval.jsonl",
        ):
            if not produced.get(name):
                print(f"[test_train:{STAGE}] 缺产物或为空：{name}")
                return False

        for row in produced["mrcr_eval.jsonl"]:
            needles = row["ground_truth"].split("、")
            positions = [row["prompt"].find(needle) for needle in needles]
            if any(pos < 0 for pos in positions) or positions != sorted(positions):
                print(f"[test_train:{STAGE}] MRCR 针的顺序不对：{positions}")
                return False
            if any(row["prompt"].count(needle) != 1 for needle in needles):
                print(f"[test_train:{STAGE}] MRCR 有重复的针：{needles}")
                return False

        again = tmp / "again.jsonl"
        rerun = subprocess.run(
            cmd + ["--eval-out", str(again)],
            cwd=here,
            capture_output=True,
            text=True,
            env=common.subprocess_env(),
        )
        if rerun.returncode != 0 or again.read_text(encoding="utf-8") != (
            out / "mrcr_eval.jsonl"
        ).read_text(encoding="utf-8"):
            print(f"[test_train:{STAGE}] 同种子两次构建结果不一致")
            return False

        print(
            f"[test_train:{STAGE}] 构建自检：NextLong {len(produced['Shensi-Longctx-Synth-NextLong.jsonl'])} 篇 / "
            f"EntropyLong {len(produced['Shensi-Longctx-Synth-EntropyLong.jsonl'])} 篇 / "
            f"MRCR {len(produced['Shensi-Longctx-MRCR.jsonl'])} 篇 + 评测 {len(produced['mrcr_eval.jsonl'])} 条："
            "针各出现一次、顺序正确、同种子可复现"
        )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Shensi {STAGE} 集成测试（tiny 几何）")
    ap.add_argument("--profile", default="debug")
    ap.add_argument(
        "--iters", type=int, default=None, help="覆盖 train_iters（默认用 profile 自己的值）"
    )
    ap.add_argument(
        "--mtp", type=int, default=0, help="MTP 层数（MTP 与 mHC 已打通，1/2 层都实跑过）"
    )
    ap.add_argument("--skip-builder", action="store_true", help="只跑训练门")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    rc = tiny_test.run_stage_tiny(
        STAGE, args.profile, override=args.override, iters=args.iters, mtp_layers=args.mtp
    )
    if rc != 0 or args.skip_builder:
        return rc
    return 0 if _check_builder() else 1


if __name__ == "__main__":
    raise SystemExit(main())
