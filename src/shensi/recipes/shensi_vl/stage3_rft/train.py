#!/usr/bin/env python3
"""Unified RFT 入口：从基座预训练模型重训（超参同冷启动 SFT，只换数据）→ 统一模型 F。"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, vl_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage3_rft")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    cfg = config.build_config("stage3_rft", args.profile, args.override)
    if cfg["train"]["model"].get("extend_tokenizer", True):
        from shensi.recipes.shensi.common import common as base

        cfg["train"]["model"]["tokenizer_dir"] = str(
            vl_tokens.extend_tokenizer(base.env_paths()["tokenizer"], cfg["train"]["model"]["tokenizer_dir"])
        )
    config.write_run_dir(cfg, "stage3_rft", args.profile)
    if args.dry_run:
        print(f"[stage3] dry-run：起点={cfg['train']['model']['llm_path']}")
        return 0
    from shensi.recipes.shensi_vl.common.train import train_loop

    return train_loop.run(cfg, mode="rft")


if __name__ == "__main__":
    raise SystemExit(main())
