#!/usr/bin/env python3
"""vLLM 生成冒烟：小样本生成并与 HF 参考对齐。"""



from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .variants import BY_KEY
from .tiny_checkpoint import build

PROMPT = "The capital of France is"

_HF_REFERENCE = """
import json, sys, torch
from transformers import AutoModelForCausalLM
ckpt, tokens, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32,
                                             trust_remote_code=True).eval()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ckpt)
text = sys.argv[4]
cur = tok(text, return_tensors="pt").input_ids
with torch.no_grad():
    for _ in range(tokens):
        logits = model(cur, use_cache=False).logits[:, -1, :]
        cur = torch.cat([cur, logits.argmax(-1, keepdim=True)], dim=1)
json.dump({"ids": cur[0, -tokens:].tolist()}, open(out, "w"))
"""


def _hf_reference(ckpt: Path, tokens: int, prompt: str) -> list[int]:
    out = ckpt.parent / "hf_reference.json"
    code = ckpt.parent / "_hf_ref.py"
    code.write_text(_HF_REFERENCE, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(code), str(ckpt), str(tokens), str(out), prompt], check=True
    )
    return json.loads(out.read_text())["ids"]


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    """生成冒烟入口：小样本生成并与 HF 参考对齐。"""
    ap = argparse.ArgumentParser(description="Looma vLLM 生成冒烟")
    ap.add_argument("--ckpt", default="/tmp/looma_smoke/looma")
    ap.add_argument("--variant", default="looma", choices=sorted(BY_KEY))
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument("--no-compare", action="store_true", help="跳过 HF 参考对拍")
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        report = build(ckpt, variant=BY_KEY[args.variant])
        print(f"[looma·vllm] 现造 tiny ckpt：{ckpt}（{report['parameters']:,} 参数）")

    from .register_model import register_all

    report = register_all()
    _ = report

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(ckpt),
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    outs = llm.generate([args.prompt], sp)
    ids = list(outs[0].outputs[0].token_ids)
    print(f"[looma·vllm] vLLM {vllm.__version__} 生成 {len(ids)} token：{ids}")

    if args.no_compare:
        return 0
    ref = _hf_reference(ckpt, args.tokens, args.prompt)
    lcp = _lcp(ids, ref)
    print(f"[looma·vllm] HF 参考（全量重算）：{ref}")
    print(f"[looma·vllm] 相同前缀：{lcp}/{args.tokens}")
    if lcp != args.tokens:
        print("[looma·vllm] 不一致：原生实现与 HF 参考在 fp32 下应当逐 token 相同")
        return 1
    print("[looma·vllm] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
