"""OPD 的学生 rollout：从模型批量采样。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from shensi.recipes.paper.looma import common

STAGE = "stage3_opd"


def main() -> int:
    """学生 rollout 入口：从模型批量采样并落 jsonl。"""
    ap = argparse.ArgumentParser(description="OPD 学生 rollout（第①步）")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--prompts", required=True, help="待采样的 prompts jsonl")
    ap.add_argument("--out", required=True, help="rollout 输出 jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vllm serve 端点")
    ap.add_argument("--model", default=None, help="端点上的模型名")
    ap.add_argument("--load", required=True, help="学生 ckpt（决定 rollout 驱动）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = common.build_config(STAGE, args.profile, [], Path(common.env_paths()["data"]) / STAGE)
    sec = dict(cfg.get("rollout") or {})
    harness_on = bool(common.agent_harness(args.load))
    template = (sec.get("command") if harness_on else sec.get("plain_command")) or ""
    if not template:
        key = "command" if harness_on else "plain_command"
        raise SystemExit(f"[looma] stage3_opd 的 rollout.{key} 没有配置（见 config/default.yaml）")
    fields = {
        "prompts": args.prompts,
        "out": args.out,
        "base_url": args.base_url,
        "model": args.model or "",
    }
    cmd = template.format(**fields)
    env = None
    if harness_on:
        from shensi.recipes.shensi.common import common as base
        from shensi.recipes.shensi.common import harness

        for line in harness.setup_commands(cfg, base_url=args.base_url, model=args.model):
            print(f"[looma] 沙箱侧：{line}")
        env = dict(base.subprocess_env(strip_proxy=True))
        env.update(harness.harness_env(cfg, base_url=args.base_url, model=args.model))
    print(f"[looma] rollout（{'harness' if harness_on else 'plain'}）：{cmd}")
    if args.dry_run:
        return 0
    return subprocess.call(cmd, shell=True, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
