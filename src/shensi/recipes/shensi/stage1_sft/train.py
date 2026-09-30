#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage0_pretrain"))

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import common

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
    ap = argparse.ArgumentParser(description="Shensi stage1_sft（FlagScale --sft）")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--wait", action="store_true", help="提交后等本机这次 run 跑完再返回（串接多阶段时用）"
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--data-jsonl",
        default=None,
        help="messages jsonl（默认 <FS>/shensi/data/stage1_sft/sft_train.jsonl）",
    )
    ap.add_argument("--set", dest="override", action="append", default=[])
    ap.add_argument("--early-stop", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        return common.smoke()
    paths = common.env_paths()
    check_tokenizer(paths)
    jsonl = Path(args.data_jsonl or paths["data"] / STAGE / "sft_train.jsonl")
    if not jsonl.is_file():
        raise SystemExit(f"没找到 {jsonl}，先跑 data_prep.py --prepare")
    override = [f"train.data.data_path={jsonl}", *args.override]
    cfg = common.build_config(STAGE, args.profile, override, paths["data"] / STAGE)
    rc = common.run(cfg, STAGE, args.profile, args.dry_run, wait=args.wait)
    if args.early_stop and not args.dry_run:
        rc = common.watch(cfg, args.early_stop)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
