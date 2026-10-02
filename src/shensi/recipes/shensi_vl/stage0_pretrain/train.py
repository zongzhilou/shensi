#!/usr/bin/env python3
"""预训练入口（HF 侧循环；文本=DSV4F+DSv4 模板，图像=Kimi K3 image_processor）。"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, vl_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage0_pretrain")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--config", default=None, help="与 --profile 等价（config 路径）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    cfg = config.build_config("stage0_pretrain", args.profile, args.override)
    paths_note = cfg["train"]["model"]
    if cfg["train"]["model"].get("extend_tokenizer", True):
        from shensi.recipes.shensi.common import common as base

        out = vl_tokens.extend_tokenizer(
            base.env_paths()["tokenizer"], cfg["train"]["model"]["tokenizer_dir"]
        )
        cfg["train"]["model"]["tokenizer_dir"] = str(out)
    config.write_run_dir(cfg, "stage0_pretrain", args.profile)
    if args.dry_run:
        print(f"[stage0] dry-run：tokenizer={paths_note['tokenizer_dir']} llm={paths_note['llm_path']} "
              f"vision={paths_note['vision_path']} processor={paths_note['processor_path']}")
        return 0
    from shensi.recipes.shensi_vl.common.train import train_loop

    return train_loop.run(cfg, mode="pretrain")


if __name__ == "__main__":
    raise SystemExit(main())
