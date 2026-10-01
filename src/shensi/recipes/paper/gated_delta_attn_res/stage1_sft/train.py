#!/usr/bin/env python3
"""GDAR 配方 · stage1_sft（SFT-1 deep-thinking / SFT-2 agent）。

    python train.py --smoke                                  # mock SFT，5 步
    python train.py --tokens 2e9 --load <Mid-2 ckpt>         # SFT-1（UltraData-SFT-2605）
    python train.py --profile sft2_agent --load <SFT-1 ckpt> # SFT-2（UltraData-SFT-Agent-2609）

`--sft` 走上游打包口径的 SFTDataset（稠密 Qwen3 无 CSA 限制），数据是 messages jsonl
（data_prep.py 产出）；loss mask 由 SFTTokenizer 按 Qwen3 chat 模板生成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage1_sft"


def _smoke_jsonl() -> Path:
    """冒烟用的合成 messages jsonl（16 条短对话，固定内容）。

    SFT 的 mock 不能再走上游 MockSFTDataset（THD 打包口径，与 local 注意力互斥，见
    train/data.py），改成"合成真实 jsonl + tiny 几何"——走的还是 SFT 的真实数据集代码路径。
    """
    out = Path(common.env_paths()["runs"]) / "smoke_stage1_sft" / "smoke_sft.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(16):
            msgs = {
                "messages": [
                    {"role": "user", "content": f"What is {i} plus {i}?"},
                    {"role": "assistant", "content": f"The answer is {2 * i}."},
                ]
            }
            fh.write(json.dumps(msgs, ensure_ascii=False) + "\n")
    print(f"[gdar] 冒烟用合成 messages jsonl：{out}（16 条）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR stage1_sft（SFT-1 / SFT-2 两段）")
    ap.add_argument("--profile", default="default")
    ap.add_argument(
        "--model-algo",
        default=None,
        help=f"模型算法（不给则用 {common.DEFAULT_ALGO}；profile 自带 spec 时用 profile 的）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不启动")
    ap.add_argument("--smoke", action="store_true", help="跑仓库内 tiny 配置 5 步（mock SFT）")
    ap.add_argument(
        "--tokens",
        type=lambda v: int(float(v)),  # 认 1e9 这类科学计数法
        default=None,
        help="token 预算（认 1e9），换算 train_iters",
    )
    ap.add_argument(
        "--data-jsonl",
        default=None,
        help="messages jsonl（默认 <FS>/shensi/data/gated_delta_attn_res/stage1_sft/sft_train.jsonl）",
    )
    ap.add_argument("--load", default=None, help="接续上一段的 ckpt 目录（SFT 从 Mid-2 起）")
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号键覆写，可多次"
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=None,
        help="早停耐心（验证指标连续多少次不改善就收尾）；不给用配置默认（默认就开）",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关掉早停看门狗（默认开：训练步数给无限大，靠早停及时收尾）",
    )
    args = ap.parse_args()
    if args.smoke:
        jsonl = _smoke_jsonl()
        return common.smoke(STAGE, "tiny", [f"train.data.data_path={jsonl}"])
    algo = common.apply_algo_or_die(args.model_algo)
    paths = common.env_paths()
    jsonl = Path(args.data_jsonl or paths["data"] / STAGE / "sft_train.jsonl")
    if not jsonl.is_file() and not args.dry_run:
        raise SystemExit(f"没找到 {jsonl}，先跑 data_prep.py --prepare")
    override = [f"train.data.data_path={jsonl}", *args.override]
    cfg = common.build_config(
        STAGE,
        args.profile,
        override,
        paths["data"] / STAGE,
        tokens=args.tokens,
        model_algo=algo,
        load_ckpt=args.load,
    )
    watch = None if args.no_early_stop else common.early_stop_plan(STAGE, cfg, args.early_stop)
    return common.run(cfg, args.dry_run, watch=watch)


if __name__ == "__main__":
    raise SystemExit(main())
