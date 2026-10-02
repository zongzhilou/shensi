"""RL 段的段级训练派发入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUB_STAGES = {
    "stage2_math": "stage2_math",
    "stage2_code": "stage2_code",
    "stage2_agent": "stage2_agent",
    "stage2_writing": "stage2_writing",
}

STAGE = "stage2_rl"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="强化学习段（四个方向 teacher）", add_help=False)
    ap.add_argument("--stage", default="stage2_math", choices=sorted(SUB_STAGES))
    args, rest = ap.parse_known_args(argv)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / SUB_STAGES[args.stage] / "train.py"),
        *rest,
    ]
    print("[gdar] 派发：" + " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
