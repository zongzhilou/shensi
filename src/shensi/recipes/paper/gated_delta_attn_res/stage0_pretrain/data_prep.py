"""预训练段的段级语料准备派发入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUB_STAGES = {"stage1_pretrain": "stage1_pretrain", "stage2_midtrain": "stage2_midtrain"}

STAGE = "stage0_pretrain"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="预训练段（PT-1/PT-2 + 中训练两段）", add_help=False)
    ap.add_argument("--stage", default="stage1_pretrain", choices=sorted(SUB_STAGES))
    args, rest = ap.parse_known_args(argv)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / SUB_STAGES[args.stage] / "data_prep.py"),
        *rest,
    ]
    print("[gdar] 派发：" + " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
