#!/usr/bin/env python3
"""造一个 Looma 的 tiny 自描述 ckpt（随机权重 + 真 tokenizer + 自带建模代码）。

不训练就能端到端跑通推理链（登记 → 引擎建房 → 生成 → 与 HF 参考对拍）：产物含带 auto_map
与全部 ``looma_*`` 旋钮的 ``config.json``、``model.safetensors``、逐字节拷贝的建模代码
（``trust_remote_code`` 用）、vendored MiniCPM5-2B tokenizer 与 ``generation_config.json``。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .variants import BY_KEY, LoomaVariant, base_of, tiny_base

TOKENIZER = Path(__file__).resolve().parents[2] / "tokenizer" / "MiniCPM5-2B"


def _ensure_chat_template(out: Path) -> None:
    """把 ``chat_template.jinja`` 的模板写进 ``tokenizer_config.json``（缺了就补）。

    transformers 会读 .jinja 文件，vLLM 的 chat 端点只看 ``tokenizer_config.json`` —— 缺了就直接
    400（"default chat template is no longer allowed … provide a chat template"）。
    """
    cfg_path = out / "tokenizer_config.json"
    jinja = out / "chat_template.jinja"
    if not cfg_path.is_file() or not jinja.is_file():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("chat_template"):
        return
    cfg["chat_template"] = jinja.read_text(encoding="utf-8")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(
    out: Path,
    *,
    variant: LoomaVariant | None = None,
    shape: str = "tiny",
    seed: int = 0,
    tokenizer_dir: Path | None = None,
) -> dict:
    """在 ``out`` 造一个 ckpt 并返回报告 dict（变体、参数量、文件清单等）。"""
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
    from shensi.recipes.paper.looma.common.models.transformers.configuration_looma import LoomaConfig
    from shensi.recipes.paper.looma.common.models.transformers.modeling_looma import LoomaForCausalLM

    variant = variant or BY_KEY["looma"]
    tok_src = Path(tokenizer_dir or TOKENIZER)
    if not tok_src.is_dir():
        raise SystemExit(f"[looma·vllm] 找不到 tokenizer：{tok_src}（--tokenizer 可指定）")

    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    vocab_size = len(tokenizer)
    torch.manual_seed(seed)
    config = LoomaConfig(**base_of(shape, vocab_size), **variant.tiny_knobs)
    # 引擎从模型 config 读停止符（eos/pad）：不设就没有停止判据，生成会一路跑到 max_tokens
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = LoomaForCausalLM(config)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "chat_template.jinja"):
        src = tok_src / name
        if src.is_file():
            shutil.copy2(src, out / name)
    _ensure_chat_template(out)
    gen = {"eos_token_id": [config.eos_token_id], "bos_token_id": config.bos_token_id,
           "pad_token_id": config.pad_token_id, "do_sample": False}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")

    files = sorted(p.name for p in out.iterdir())
    report = {
        "variant": variant.key,
        "shape": shape,
        "architecture": variant.architecture,
        "model_type": variant.model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": vocab_size,
        "knobs": {k: getattr(config, k) for k in config.__class__.__annotations__
                  if k.startswith("looma_")},
        "files": files,
        "has_remote_code": ("configuration_looma.py" in files and "modeling_looma.py" in files),
        "tokenizer": str(tok_src),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：造 ckpt，打印报告与下一步的 ``vllm serve`` 命令。"""
    ap = argparse.ArgumentParser(description="Looma tiny 自描述 ckpt")
    ap.add_argument("--out", default="/tmp/looma_smoke/looma")
    ap.add_argument("--variant", default="looma", choices=sorted(BY_KEY))
    ap.add_argument("--shape", default="tiny", choices=["tiny", "0.6b"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args(argv)

    report = build(Path(args.out), variant=BY_KEY[args.variant], shape=args.shape,
                   seed=args.seed, tokenizer_dir=args.tokenizer)
    print(f"[looma·vllm] tiny ckpt：{args.out}")
    for key in ("variant", "shape", "architecture", "model_type", "parameters", "vocab_size",
                "has_remote_code"):
        print(f"  {key:16s}: {report[key]}")
    print(f"  knobs           : {report['knobs']}")
    print(f"  files           : {report['files']}")
    print(f"\n下一步（没登记引擎也能跑，走 auto_map 的远程代码）：")
    print(f"  vllm serve {args.out} --trust-remote-code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
