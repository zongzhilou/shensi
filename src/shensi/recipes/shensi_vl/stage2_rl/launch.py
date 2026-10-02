"""stage2_rl 两个子段共用入口：读 config → grpo.run（含 OPD 用的难度分层已由 data_prep 做）。"""

from __future__ import annotations

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, vl_tokens


def launch(stage: str) -> int:
    ap = argparse.ArgumentParser(description=f"shensi_vl {stage}（GRPO）")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    cfg = config.build_config(stage, args.profile, args.override)
    if cfg["train"]["model"].get("extend_tokenizer", True):
        from shensi.recipes.shensi.common import common as base

        cfg["train"]["model"]["tokenizer_dir"] = str(
            vl_tokens.extend_tokenizer(base.env_paths()["tokenizer"], cfg["train"]["model"]["tokenizer_dir"])
        )
    config.write_run_dir(cfg, stage, args.profile)
    if args.dry_run:
        print(f"[{stage}] dry-run：group_n={cfg['train']['rl'].get('group_n')} "
              f"pool={cfg['train']['data']['train_jsonl']}")
        return 0
    from shensi.recipes.shensi_vl.stage2_rl import grpo

    return grpo.run(cfg)
