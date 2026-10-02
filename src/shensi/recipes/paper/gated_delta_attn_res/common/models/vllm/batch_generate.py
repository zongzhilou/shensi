"""Continuous batching: many prompts in one engine, one at a time as the control.

``LLM.generate([p1, p2, ...])`` hands every prompt to the engine's scheduler at once;
the scheduler then decides, step by step, how many of them to run *together*.  That
is the continuous-batching path, and it is the one a rollout loop actually uses --
the smoke test only ever submitted a single prompt.

What this checks
----------------
1. **correctness**: every prompt's tokens from the batched call are *identical* to
   the tokens of the same prompt run alone on the same ``LLM`` instance (greedy +
   ``ignore_eos``, so both are deterministic and must agree), and optionally to the
   plain-transformers reference;
2. **that batching really happened**: it counts, per engine step, how many requests
   the scheduler had running.  Requests of different lengths are submitted together,
   so a real continuous-batching run shows several of them in the same step and
   different finish steps -- a sequential engine would only ever show one;
3. **whether a divergence is a defect at all**: the lone calls are repeated, and every
   generated token's top-2 logprob margin is reported.  Greedy decoding is one argmax
   per step, so two *correct* runs can still pick different tokens wherever the top two
   logits are within numerical noise -- which is the regime these random-weight models
   are in (``min top1-top2 margin`` comes back ~1e-4 and smaller).  ``--max-num-seqs 1``
   is the control: the same batched call with co-batching forbidden.

For (2) the engine has to run *in this process* (``VLLM_ENABLE_V1_MULTIPROCESSING=0``,
set below before vLLM is imported): with the default multiprocess engine the scheduler
lives in another process and cannot be observed from here.  The scheduling path is the
same one either way; only the observation point changes.

That has one consequence, measured rather than assumed: vLLM's Transformers backend
**decorates the HF model class in place** (``support_torch_compile`` appends
``TorchCompileWithNoGuardsWrapper`` to ``cls.__bases__`` and replaces ``__init__`` with
one that calls ``get_current_vllm_config()``).  With the engine in-process, every later
``AutoModelForCausalLM.from_pretrained`` of the same architecture then dies with
``AssertionError: Current vLLM config is not set``.  So the plain-transformers reference
below runs in a **fresh subprocess** (``--hf-reference``), which is also the honest
isolation boundary: it is exactly the process a user would get.

Usage
-----
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.batch_generate --variant dar
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.batch_generate --variant gdar --shape 0.6b \\
        --dtype bfloat16 --max-model-len 512
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ._paths import RECIPE, SRC, ensure_src_on_path

ensure_src_on_path()

#: Observation point, not a behaviour switch: the engine must be in-process for the
#: scheduler trace below to exist at all.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from .register_model import register_all  # noqa: E402
from .smoke_generate import _ensure_venv_bin_on_path  # noqa: E402
from .tiny_checkpoint import DEFAULT_TOKENIZER, build  # noqa: E402
from .variants import BY_KEY, SHAPES  # noqa: E402

#: (label, prompt, max_new_tokens) -- four lengths, four different budgets, so the
#: requests neither start nor finish at the same step.
PROMPTS: tuple[tuple[str, str, int], ...] = (
    ("p1-short", "The capital of France is", 8),
    (
        "p2",
        "In a distant future where machines have learned to write poetry, the last human librarian",
        24,
    ),
    (
        "p3-medium",
        "Once upon a time, in a small village at the edge of a dark forest, there lived a",
        16,
    ),
    (
        "p4-long",
        "The committee reviewed the proposal carefully and concluded that the most important "
        "question was not whether the system could be built, but whether it should be built at "
        "all, given the constraints of the surrounding environment and the",
        32,
    ),
)


def _instrument_scheduler() -> list[dict]:
    """Record, per ``Scheduler.schedule`` call, what the engine decided to run."""
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.schedule
    trace: list[dict] = []

    def schedule(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        trace.append(
            {
                "running": len(self.running),
                "waiting": len(self.waiting),
                "scheduled": len(output.num_scheduled_tokens),
            }
        )
        return output

    Scheduler.schedule = schedule  # type: ignore[method-assign]
    return trace


def _hf_reference_subprocess(variant: str, shape: str, dtype: str, device: str) -> dict:
    """Run ``--hf-reference`` in a fresh interpreter; returns ``{label: token ids}``.

    A subprocess, not a function call: see the note in the module docstring -- vLLM's
    compile decorator rewrites the model class in whatever process builds the engine.
    """
    cmd = [
        sys.executable,
        "-m",
        "shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.batch_generate",
        "--hf-reference",
        "--variant",
        variant,
        "--shape",
        shape,
        "--dtype",
        dtype,
        "--hf-device",
        device,
    ]
    proc = subprocess.run(cmd, cwd=str(SRC), env=os.environ.copy(), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise RuntimeError(f"the HF reference subprocess failed (exit {proc.returncode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])["generated"]


def _run_hf_reference(args) -> int:
    """The plain-transformers reference for every prompt, model loaded once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    ckpt = str(Path(args.ckpt_root) / f"{args.variant}-{args.shape}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch_dtype, trust_remote_code=True)
        .eval()
        .to(args.hf_device)
    )
    out: dict[str, list[int]] = {}
    with torch.no_grad():
        for label, prompt, n in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            # full recompute, like smoke_generate.hf_greedy: the depth state is per token
            # and rebuilt inside every forward, so this is the exact reference.
            seq = ids
            for _ in range(n):
                logits = model(seq, use_cache=False).logits[:, -1, :]
                seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
            out[label] = seq[0, ids.shape[1] :].tolist()
    print(json.dumps({"generated": out}))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="dar", choices=sorted(BY_KEY))
    ap.add_argument("--shape", default="tiny", choices=sorted(SHAPES))
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--ckpt-root", default="/tmp/rollout_batch")
    ap.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="cap how many requests the scheduler may run together. 1 is the "
        "control that isolates *co-batching*: the same batched call, but the "
        "engine runs one request at a time (what differs is only the batch "
        "shape the kernels see, not the call)",
    )
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument(
        "--compare-hf",
        action="store_true",
        help="also compare against the plain-transformers reference, which runs in "
        "its own subprocess (see the module docstring)",
    )
    ap.add_argument("--hf-device", default="cpu")
    ap.add_argument(
        "--hf-reference",
        action="store_true",
        help="internal: print the reference tokens for every prompt as JSON and exit",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.hf_reference:
        return _run_hf_reference(args)

    _ensure_venv_bin_on_path()
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    variant = BY_KEY[args.variant]
    ckpt = Path(args.ckpt_root) / f"{variant.key}-{args.shape}"
    if args.rebuild or not (ckpt / "model.safetensors").exists():
        rep = build(
            variant.key, ckpt, shape=args.shape, tokenizer_dir=args.tokenizer_dir, overwrite=True
        )
        print(
            f"built {args.shape} checkpoint: {rep['parameters']:,} params, "
            f"vocab={rep['vocab_size']}, dir={ckpt}"
        )
    register_all()

    print(f"vllm {vllm.__version__} | python {sys.version.split()[0]} | torch {torch.__version__}")
    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
        f"in-process engine: VLLM_ENABLE_V1_MULTIPROCESSING="
        f"{os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}"
    )

    kwargs = dict(
        model=str(ckpt),
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    if args.kv_cache_memory_bytes:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**kwargs)

    trace = _instrument_scheduler()
    # logprobs=2 costs nothing and is what makes a *divergence* diagnosable: greedy
    # decoding only needs the argmax, so a batched run and a lone run can only pick
    # different tokens where the top two are within numerical noise (see below).
    sps = [
        SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, logprobs=2)
        for _, _, n in PROMPTS
    ]
    prompts = [p for _, p, _ in PROMPTS]

    # ---- the batched call (all prompts submitted at once) -------------------
    batched = llm.generate(prompts, sps)
    batch_order = [list(o.outputs[0].token_ids) for o in batched]
    batch_margin = [_min_margin(o.outputs[0]) for o in batched]
    prompt_lens = [len(o.prompt_token_ids) for o in batched]
    steps_batched = len(trace)
    trace_batched = list(trace)
    trace.clear()

    # ---- the same prompts, one call each, same LLM --------------------------
    single, steps_single, single_margin = [], [], []
    for i, (label, prompt, n) in enumerate(PROMPTS):
        trace.clear()
        out = llm.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, logprobs=2)
        )
        single.append(list(out[0].outputs[0].token_ids))
        single_margin.append(_min_margin(out[0].outputs[0]))
        steps_single.append(len(trace))

    # ---- is the *single* call itself reproducible? --------------------------
    # A greedy decode is one argmax per step, so it can only differ between runs where
    # the top two logits are within numerical noise.  Repeating the lone calls says how
    # much of the batched-vs-single difference is that kind of noise.
    single2 = []
    for label, prompt, n in PROMPTS:
        out = llm.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, logprobs=2)
        )
        single2.append(list(out[0].outputs[0].token_ids))
    repeated = sum(a == b for a, b in zip(single, single2))
    identical = sum(a == b for a, b in zip(batch_order, single))
    print(
        f"\n  single call repeated: {repeated}/{len(PROMPTS)} identical to the first run"
        f"   (batched vs single: {identical}/{len(PROMPTS)})"
    )
    if repeated < len(PROMPTS):
        print("  -> the lone calls are not reproducible run-to-run at this dtype/scale, so")
        print("     'batched != single' cannot be read as a batching defect")

    print(
        f"\n{'prompt':12s} {'tokens':>6s} {'new':>4s}  batched == single   "
        f"min top1-top2 margin  (batched / single)"
    )
    failures = 0
    for i, (label, _, n) in enumerate(PROMPTS):
        same = batch_order[i] == single[i]
        failures += not same
        print(
            f"  {label:12s} {prompt_lens[i]:5d} {n:4d}  {'YES' if same else 'NO '}"
            f"                 {batch_margin[i]:.3e} / {single_margin[i]:.3e}"
            f"{'' if same else '   first divergence at step ' + str(_first_diff(batch_order[i], single[i]))}"
        )
        print(f"    batched ids: {batch_order[i]}")
    print(
        f"\n  all {len(PROMPTS)} prompts identical between the batched call and the "
        f"one-at-a-time calls: {failures == 0}"
    )

    # ---- did the engine really batch them? ---------------------------------
    peak = max((t["running"] for t in trace_batched), default=0)
    multi = [i for i, t in enumerate(trace_batched) if t["running"] > 1]
    print(f"\n  scheduler trace over the batched call: {steps_batched} engine steps")
    print(f"    running requests per step : {[t['running'] for t in trace_batched]}")
    print(f"    waiting  requests per step: {[t['waiting'] for t in trace_batched]}")
    print(f"    peak concurrent requests  : {peak} / {len(PROMPTS)} submitted")
    print(f"    steps with >1 request     : {len(multi)}")
    print(f"    steps one-at-a-time       : {steps_single} (sum {sum(steps_single)})")
    batched_ok = peak > 1 and len(multi) > 0 and steps_batched < sum(steps_single)
    failures += not batched_ok
    print(
        f"  continuous batching observed: {batched_ok}  "
        f"(peak {peak} > 1 and {steps_batched} steps < {sum(steps_single)} sequential steps)"
    )

    # ---- optional: the plain-transformers reference (own process) ----------
    hf = None
    if args.compare_hf:
        hf_map = _hf_reference_subprocess(args.variant, args.shape, args.dtype, args.hf_device)
        print("  HF reference (fresh process, full recompute):")
        for i, (label, _, n) in enumerate(PROMPTS):
            ref = hf_map[label]
            lcp = _lcp(batch_order[i], ref)
            failures += lcp != n
            print(f"    {label:12s} lcp={lcp}/{n}  {'OK' if lcp == n else 'MISMATCH'}")
        hf = [hf_map[label] for label, _, _ in PROMPTS]

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "vllm": vllm.__version__,
                    "variant": variant.key,
                    "shape": args.shape,
                    "dtype": args.dtype,
                    "prompts": [{"label": l, "prompt": p, "max_tokens": n} for l, p, n in PROMPTS],
                    "batched": batch_order,
                    "single": single,
                    "hf": hf,
                    "scheduler_trace": trace_batched,
                },
                indent=2,
            )
        )
        print(f"\nreport written to {args.json}")

    print("\n" + ("all checks passed" if not failures else f"FAILED: {failures} check(s)"))
    return 1 if failures else 0


def _min_margin(completion) -> float:
    """Smallest ``top1 - top2`` logprob over the generated tokens (0.0 if unknown).

    A greedy decoder can only be swayed by numerical noise at the steps where this
    is ~1e-3 or less; a model with random weights is exactly the regime where the
    top two logits sit that close together.
    """
    gap = float("inf")
    for step in completion.logprobs or []:
        ranked = sorted(
            step.values(), key=lambda lp: -(lp.logprob if lp.logprob is not None else -1e9)
        )
        if len(ranked) >= 2:
            gap = min(gap, abs(ranked[0].logprob - ranked[1].logprob))
    return 0.0 if gap == float("inf") else gap


def _first_diff(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(main())
