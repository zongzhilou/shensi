#!/usr/bin/env python3
"""SFT 入口（mcore --sft；注入 sft_train.jsonl）。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage0_pretrain"))

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common

STAGE = "stage1_sft"


def check_tokenizer(paths) -> None:
    """SFTTokenizer 按 chat 模板切轮次；这里只做提示，不拦（模板可能写在 tokenizer_config 或 *.jinja 里）。"""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[stage1_sft] 提示：tokenizer 读不到（{paths['tokenizer']}，{type(exc).__name__}）")
        return
    if not getattr(tok, "chat_template", None):
        print(
            f"[stage1_sft] 提示：{paths['tokenizer']} 的 chat_template 为空，"
            "若逐轮 loss mask 报错就换带模板的 tokenizer（SHENSI_TOKENIZER）"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage1_sft（mcore --sft）")
    ap.add_argument("--profile", default="debug")
    ap.add_argument(
        "--config",
        default=None,
        help="直接给配置档路径（与 --profile 等价，例：config/tiny.yaml）",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--data-jsonl",
        default=None,
        help="messages jsonl（默认 <FS>/shensi/data/stage1_sft/sft_train.jsonl）",
    )
    ap.add_argument("--set", dest="override", action="append", default=[])
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
    check_tokenizer(paths)
    cfg = common.build_config(STAGE, args.profile, args.override, paths["data"] / STAGE)
    data = cfg.get("train", {}).get("data") or {}
    if data.get("mock_data"):
        # 冒烟档：不做 mcore 的 mock SFT（那份是 THD 打包口径，CSA 断言拒绝打包），
        # 改成一份自足的极小 jsonl（十几条单轮问答，ShensiSFTDataset 一条一条喂）；
        # 显式给了 --data-jsonl 且文件在，就按它来（世界模型段落的 SFT 会这么传）
        from shensi.recipes.shensi.stage1_sft import data_prep

        explicit = Path(args.data_jsonl) if args.data_jsonl else None
        if explicit and explicit.is_file():
            jsonl = explicit
        else:
            jsonl = Path(paths["data"] / STAGE / "tiny_sft.jsonl")
            if not jsonl.is_file():
                data_prep.write_tiny_jsonl(jsonl)
        common._set_dotted(cfg, "train.data.mock_data", False)
        common._set_dotted(cfg, "train.data.data_path", str(jsonl))
    else:
        jsonl = Path(args.data_jsonl or paths["data"] / STAGE / "sft_train.jsonl")
        if not jsonl.is_file():
            raise SystemExit(f"没找到 {jsonl}，先跑 data_prep.py --prepare")
        common._set_dotted(cfg, "train.data.data_path", str(jsonl))
    # 验证集不走第二个数据源：mcore 只允许一个数据源（data_path 与 valid_data_path 同时给会 assert），
    # 验证由 `train.data.split`（默认 98,1,1）从同一份 jsonl 切出来——早停看门狗盯的就是它的验证损失。
    patience = 0 if args.no_early_stop else args.early_stop
    return common.run(
        cfg,
        STAGE,
        args.profile,
        args.dry_run,
        watch=common.watchdog_spec(patience, metric="lm loss value", grace=args.early_stop_grace),
    )


if __name__ == "__main__":
    raise SystemExit(main())
