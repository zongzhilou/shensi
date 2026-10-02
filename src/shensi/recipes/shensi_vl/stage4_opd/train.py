#!/usr/bin/env python3
"""On-Policy Distillation 入口（论文 §2.5.4）。

L_OPD(θ) = Σᵢ wᵢ · D_KL(π_θ ‖ π_Eᵢ)，教师 = {E_TwG, E_TwP}，全词表 logit 蒸馏，
轨迹来自学生自采样（data_prep.py 产出的 opd_train.jsonl；学生对这些轨迹 token 学教师的分布）。
实现走 common/train/train_loop.py 的 opd 模式（CE + Σ wᵢ·反向 KL）。
"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, vl_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="shensi_vl stage4_opd")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    cfg = config.build_config("stage4_opd", args.profile, args.override)
    if cfg["train"]["model"].get("extend_tokenizer", True):
        from shensi.recipes.shensi.common import common as base

        cfg["train"]["model"]["tokenizer_dir"] = str(
            vl_tokens.extend_tokenizer(base.env_paths()["tokenizer"], cfg["train"]["model"]["tokenizer_dir"])
        )
    config.write_run_dir(cfg, "stage4_opd", args.profile)
    if args.dry_run:
        print(f"[stage4] dry-run：teachers={cfg['train'].get('opd', {}).get('teachers')}")
        return 0
    from shensi.recipes.shensi_vl.common.train import train_loop

    return train_loop.run(cfg, mode="opd")


if __name__ == "__main__":
    raise SystemExit(main())
