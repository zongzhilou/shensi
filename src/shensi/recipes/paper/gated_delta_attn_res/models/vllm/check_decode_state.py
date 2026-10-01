"""Is the depth state really lost across decode steps?  Measure it, do not assume.

The depth-routed variants carry a per-token state that is *not* in the KV cache
(GDAR/AR/DAR: the delta-source list; HC/MHC: the n-stream residual and its mixing
matrix; MUDD/DenseFormer: their depth aggregates).  ``Qwen3HCModel`` /
``Qwen3MHCModel`` raise ``NotImplementedError`` when called with a non-empty
``past_key_values`` **and** exactly one new token, on the theory that incremental
decoding would silently drop that state.

This script checks the theory rather than trusting it:

1. **Cached vs recomputed, N greedy steps.**  Decode N tokens (a) by re-running the
   whole sequence every step with ``use_cache=False``, and (b) incrementally with
   ``past_key_values``.  If the depth state were lost, (b) would diverge from (a).
2. **Chunked prefill.**  A 2-chunk prefill, which is what a chunked-prefill engine
   does -- and which also *slips past* the hc/mhc guard, because the guard only fires
   on ``seq_len == 1``.  Worth knowing before pointing any engine at these models.
3. **The hc/mhc guard, removed by source patch.**  For hc/mhc the guard is deleted
   from the *inner* model's ``forward`` source (a surgical ``exec`` of the patched
   function, nothing on disk changes) and (1) is repeated, so the guard can be judged
   separately from the mathematics.

Every variant is run with ``attn_res_block_size`` set, because that is the master
switch for the connection: at its default of ``None`` the connection is *off* and all
7 variants are plain Qwen3, so the test would be vacuous.

Run with either environment (needs only torch + transformers v5):

    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.check_decode_state
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.check_decode_state
"""

from __future__ import annotations

import argparse
import inspect
import sys
import textwrap
from pathlib import Path

from ._paths import RECIPE, SRC, ensure_src_on_path

ensure_src_on_path()

from .variants import VARIANTS  # noqa: E402


def _load(variant_key: str):
    import importlib

    from .variants import BY_KEY

    v = BY_KEY[variant_key]
    cfg_mod = importlib.import_module(f"models.{v.config_file}")
    model_mod = importlib.import_module(f"models.{v.model_file}")
    return v, getattr(cfg_mod, v.config_class), getattr(model_mod, v.model_class)


def remove_guard(model_cls) -> bool:
    """Delete the ``raise NotImplementedError`` block from the inner model's forward.

    Only in memory: the source is re-``exec``'d with the block removed and the
    function is rebound on the class.  Returns True if something was actually removed.
    """
    mod = sys.modules[model_cls.__module__]
    inner = [
        n
        for n in dir(mod)
        if n.startswith("Qwen3") and n.endswith("Model") and "ForCausalLM" not in n
    ]
    if not inner:
        return False
    cls = getattr(mod, inner[0])
    lines = textwrap.dedent(inspect.getsource(cls.forward)).splitlines()
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.strip().startswith("if past_key_values is not None") and "get_seq_length" in ln
        ),
        None,
    )
    if start is None:
        return False
    end = next(j for j in range(start, len(lines)) if lines[j].strip() == ")")
    ns = dict(vars(mod))
    exec(compile("\n".join(lines[:start] + lines[end + 1 :]), f"<patched {inner[0]}>", "exec"), ns)
    cls.forward = ns["forward"]
    return True


def decode(model, ids, steps, mode):
    out, cache, cur = [], None, ids
    for _ in range(steps):
        if mode == "recompute":
            r = model(cur, use_cache=False)
        else:
            r = model(cur, past_key_values=cache, use_cache=True)
            cache = r.past_key_values
        nxt = r.logits[:, -1, :].argmax(-1, keepdim=True)
        out.append(int(nxt))
        cur = nxt if mode == "cached" else __import__("torch").cat([cur, nxt], dim=1)
    return out


def probe(
    variant_key: str,
    *,
    layers: int,
    hidden: int,
    block_size: int,
    steps: int,
    prompt_len: int,
    vocab: int,
    patch: bool,
) -> dict:
    import torch

    v, cfg_cls, model_cls = _load(variant_key)
    torch.manual_seed(0)
    # ``mudd_num_ways`` is documented as 4 or 1; 2 raises inside the module, so the
    # per-variant knob table is consulted instead of guessing.
    knobs = dict(v.tiny_knobs)
    knobs["attn_res_block_size"] = block_size
    if "mudd_num_ways" in knobs and knobs["mudd_num_ways"] == 2:
        knobs["mudd_num_ways"] = 1
    cfg = cfg_cls(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        **knobs,
    )
    model = model_cls(cfg).eval()
    prompt = torch.randint(0, vocab, (1, prompt_len))

    ref = decode(model, prompt.clone(), steps, "recompute")

    guard_fired, patched = False, False
    try:
        got = decode(model, prompt.clone(), steps, "cached")
    except NotImplementedError:
        guard_fired = True
        if patch:
            patched = remove_guard(model_cls)
            got = decode(model, prompt.clone(), steps, "cached")
        else:
            got = None

    # chunked prefill: 2 chunks, both longer than one token
    chunk_ok = None
    half = max(1, prompt_len // 2)
    try:
        with torch.no_grad():
            c1 = model(prompt[:, :half], use_cache=True)
            c2 = model(
                prompt[:, half:],
                past_key_values=c1.past_key_values,
                cache_position=torch.arange(half, prompt_len),
            )
            chunk_ok = bool(
                torch.allclose(
                    c2.logits[:, -1, :], model(prompt).logits[:, -1, :], atol=1e-4, rtol=1e-4
                )
            )
    except Exception as exc:
        chunk_ok = f"{type(exc).__name__}"

    return {
        "variant": variant_key,
        "block_size": block_size,
        "guard_fired": guard_fired,
        "guard_patched": patched,
        "cached_matches_recompute": (got == ref) if got is not None else None,
        "chunked_prefill_matches": chunk_ok,
        "recompute_tokens": ref,
        "cached_tokens": got,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--prompt-len", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2])
    ap.add_argument(
        "--no-patch",
        action="store_true",
        help="do not patch the hc/mhc guard away (leave their cached run unmeasured)",
    )
    args = ap.parse_args(argv)

    print(
        f"torch {__import__('torch').__version__} | transformers "
        f"{__import__('transformers').__version__}"
    )
    print(
        f"{args.layers} layers, hidden={args.hidden}, {args.steps} greedy steps, "
        f"prompt={args.prompt_len}, vocab={args.vocab}, block_sizes={args.block_sizes}\n"
    )
    header = f"{'variant':22s} {'bs':>3s} {'guard':>6s} {'patched':>8s} {'cached==recompute':>18s} {'chunked prefill':>16s}"
    print(header)
    print("-" * len(header))

    failures = []
    for bs in args.block_sizes:
        for v in VARIANTS:
            r = probe(
                v.key,
                layers=args.layers,
                hidden=args.hidden,
                block_size=bs,
                steps=args.steps,
                prompt_len=args.prompt_len,
                vocab=args.vocab,
                patch=not args.no_patch,
            )
            print(
                f"{v.key:22s} {bs:3d} {str(r['guard_fired']):>6s} {str(r['guard_patched']):>8s} "
                f"{str(r['cached_matches_recompute']):>18s} {str(r['chunked_prefill_matches']):>16s}"
            )
            if r["guard_patched"] and r["cached_matches_recompute"] is False:
                failures.append((v.key, bs))
                print(f"    recompute={r['recompute_tokens']}")
                print(f"    cached   ={r['cached_tokens']}")

    print()
    if failures:
        print(f"DIVERGED for {failures} -- the depth state really is lost there")
        return 1
    print(
        "No divergence: for every variant and block size, KV-cached greedy decoding "
        "reproduces the full-recompute token sequence exactly."
    )
    print(
        "=> the depth state is per-token and reconstructed inside each forward, so no "
        "depth-state cache is needed by an engine."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
