#!/usr/bin/env python3
import argparse
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common

STAGE = "stage1_pretrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage1_pretrain 主预训练")
    ap.add_argument("--profile", default="default")
    ap.add_argument(
        "--config",
        default=None,
        help="直接给配置档路径（与 --profile 等价，例：config/tiny.yaml）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不启动")
    ap.add_argument(
        "--wait", action="store_true", help="提交后等本机这次 run 跑完再返回（串接多阶段时用）"
    )
    ap.add_argument("--smoke", action="store_true", help="跑仓库内 tiny 配置 5 步")
    ap.add_argument("--tokens", type=int, default=None, help="token 预算，用来换算 train_iters")
    ap.add_argument("--data-dir", default=None, help="预处理产物目录（含 blend.json）")
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号键覆写，可多次"
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=3,
        help="早停耐心（验证指标连续多少次不改善就收尾；默认 3，0 或负数=不看门狗）",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关掉早停看门狗（按 profile 的 train_iters 跑满）",
    )
    ap.add_argument(
        "--early-stop-grace",
        type=float,
        default=600.0,
        help="宽限秒数：这段时间内不判耐心（跑过预热再判）",
    )
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    if args.smoke:
        return common.smoke()
    paths = common.env_paths()
    data_dir = Path(args.data_dir or paths["data"] / STAGE)
    cfg = common.build_config(STAGE, args.profile, args.override, data_dir, tokens=args.tokens)
    opt = str(
        (cfg.get("train", {}).get("model", {}).get("optimizer") or {}).get("optimizer", "")
    ).lower()
    # 默认档就是混合优化器（Muon + AdEMAMix），所以这个前置检查默认就会跑
    if opt and opt != "adamw":
        try:
            __import__("emerging_optimizers")  # noqa: F401
            if opt in ("muon", "adaptive_muon"):
                # mcore 用自己那份 TensorParallelMuon（在 megatron.core.optimizer.emerging_optimizers），
                # 但它依赖 emerging_optimizers 的 Newton–Schulz 内核与 NS 系数表
                from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: F401
                    newton_schulz_tp,
                )
        except ImportError:
            raise SystemExit(
                f"[recipe] 这个档用 {opt}，但环境里没有 emerging-optimizers（>= 0.2）。\n"
                "          非 AdamW 的优化器（muon / adaptive_muon / lion / soap / …）都靠这个包；\n"
                "          装法见本 stage README 的『优化器』小节（默认档就是混合优化器）。\n"
                "          只想先用 AdamW 做对照：--profile adamw。"
            ) from None
        if opt in ("muon", "adaptive_muon"):
            from megatron.core.optimizer.emerging_optimizers import (  # noqa: F401
                TensorParallelMuon,
            )
    patience = 0 if args.no_early_stop else args.early_stop
    return common.run(
        cfg,
        STAGE,
        args.profile,
        args.dry_run,
        wait=args.wait,
        watch=common.watchdog_spec(patience, metric="lm loss value", grace=args.early_stop_grace),
    )


if __name__ == "__main__":
    raise SystemExit(main())
