#!/usr/bin/env python3
"""Specialized SFT 入口（论文 §2.5.1）。

`--profile default` = thinking with grounding（box 专家 F_TwG）；
`--profile point`   = thinking with pointing（point 专家 F_TwP）。
70% 通用 + 30% 专项的混合在 data_prep 里按 family 已拌好。
"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, vl_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage1_sft")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    cfg = config.build_config("stage1_sft", args.profile, args.override)
    if cfg["train"]["model"].get("extend_tokenizer", True):
        from shensi.recipes.shensi.common import common as base

        cfg["train"]["model"]["tokenizer_dir"] = str(
            vl_tokens.extend_tokenizer(base.env_paths()["tokenizer"], cfg["train"]["model"]["tokenizer_dir"])
        )
    config.write_run_dir(cfg, "stage1_sft", args.profile)
    if args.dry_run:
        print(f"[stage1] dry-run：box/point 档见 config/{args.profile}.yaml 的 data.train_jsonl")
        return 0
    from shensi.recipes.shensi_vl.common.train import train_loop

    return train_loop.run(cfg, mode="sft")


if __name__ == "__main__":
    raise SystemExit(main())
