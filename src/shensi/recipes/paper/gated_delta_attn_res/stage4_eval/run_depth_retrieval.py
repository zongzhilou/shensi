"""Score `train/eval_depth_retrieval.py` JSONL and report it the way the reviews demand.

What the reviewers asked for (R1-1/R1-2/R1-12) is not just a number: the task has
to be *fully defined* (length, K, distractors, answer format, item count, chance)
and the report has to be stratified rather than a single pooled accuracy.  This
script is the second half of that pair -- the generator defines the task, this
file scores it and refuses to call a result "usable" unless it beats chance.

Input:  the JSONL produced by `train/eval_depth_retrieval.py` (one question per
line; fields `context`, `choices`, `answer_idx`, `k_pairs`, `context_chars`,
plus a sibling `*.manifest.json` with the task definition).  We do *not* re-derive
the answer: `answer_idx` from the file is the gold.

Scoring: the standard harness multiple-choice rule.  The prompt is everything up
to and including ``Answer:``; each of the four options is appended and its
log-probability under the model is accumulated (teacher forcing, the prompt is
masked out).

    sum   : total log P(option | prompt)          -- "acc"
    mean  : per-token log P(option | prompt)      -- "acc_norm" (length-fairer)

The report contains, in this order:

1. **Provenance header** -- model path, dataset path, item count, and the task
   definition (lengths, K, repeats, seed) read from the manifest.
2. **Pooled accuracy** with a Wilson 95% interval and an exact one-sided
   binomial test against ``chance`` (default **25%**, 4-way MC).
3. **K x L stratification** -- accuracy per (KV pairs) x (context length) cell,
   which is the table that separates "fails at depth" from "fails at length".
4. **Answer-position bias check** -- distribution of the *predicted* option
   position next to the *gold* position distribution (the latter must be ~uniform
   by construction), accuracy conditioned on the gold position, and a scalar
   position-bias index.  A model that scores above chance while always answering
   position 0 is reported as biased, not as good.
5. **Usable flag** -- set only when the Wilson 95% lower bound clears chance
   (overall and per stratum, cells with too few items are marked as such).

Nothing here depends on lm-eval, so this path works even in an environment where
the harness cannot be installed.

Usage
-----
    # real run
    python eval/run_depth_retrieval.py --model /path/to/ckpt --data /tmp/dr_test.jsonl

    # CPU smoke: builds a tiny random GDAR checkpoint + a tiny task file, ~30s
    python eval/run_depth_retrieval.py --smoke

    # sanity control: shuffle the gold labels; pooled accuracy must fall back to
    # ~chance, which is what proves the scoring path has no leak
    python eval/run_depth_retrieval.py --model M --data D --control-shuffle-labels
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CHANCE = 0.25


# ---------------------------------------------------------------------------
# statistics (no scipy dependency)
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """Exact one-sided P(X >= k) for X ~ Binomial(n, p)."""
    if n == 0:
        return 1.0
    k = max(0, min(k, n))
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def stratum_report(hits: int, n: int, chance: float) -> dict:
    lo, hi = wilson_interval(hits, n)
    return {
        "n": n,
        "hits": hits,
        "acc": (hits / n) if n else float("nan"),
        "wilson95": [lo, hi],
        "p_one_sided_vs_chance": binom_tail_ge(hits, n, chance) if n else 1.0,
        "usable": bool(n > 0 and lo > chance),
    }


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_items(path: Path, limit: int | None) -> list[dict]:
    items = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    for i, it in enumerate(items):
        for field in ("context", "choices", "answer_idx"):
            if field not in it:
                raise SystemExit(f"{path}: item {i} lacks required field {field!r}")
        if len(it["choices"]) != 4:
            raise SystemExit(f"{path}: item {i} has {len(it['choices'])} choices, expected 4")
        it.setdefault("k_pairs", -1)
        it.setdefault("context_chars", len(it["context"]))
    return items


def load_manifest(path: Path) -> dict | None:
    man = path.with_suffix(".manifest.json")
    if man.exists():
        try:
            return json.loads(man.read_text())
        except Exception:
            return None
    return None


def declared_cells(manifest: dict | None) -> list[tuple[int, int, int]]:
    """Parse the generator manifest's ``strata`` into ``[(K, L_requested, count), ...]``.

    The generator records what it *asked for* (``K=4,L=8192``); the JSONL only
    records what came out (``context_chars`` ~ L + the inserted fact sentences).
    Reporting on the requested grid is what makes cells comparable across
    datasets, so this is the preferred grid whenever a manifest is present.
    """
    if not manifest:
        return []
    cells = []
    for key, count in (manifest.get("strata") or {}).items():
        try:
            k_part, l_part = key.split(",")
            cells.append((int(k_part.split("=")[1]), int(l_part.split("=")[1]), int(count)))
        except Exception:
            continue
    return sorted(cells)


def assign_cells(preds: list[dict], cells: list[tuple[int, int, int]]) -> tuple[list, dict]:
    """Map each prediction onto a declared (K, L) cell; falls back to the raw values.

    Returns ``(grid_ks, grid_ls)`` plus a per-item ``cell`` annotation.  When there
    is no manifest grid, ``context_chars`` is bucketed instead and the cell key
    carries the bucket label.
    """
    if cells:
        ls = sorted({L for _, L, _ in cells})
        ks = sorted({K for K, _, _ in cells})
        valid = {(K, L) for K, L, _ in cells}
        for p in preds:
            cands = [L for L in ls if (p["k_pairs"], L) in valid] or ls
            L = min(cands, key=lambda v: abs(v - p["context_chars"]))
            p["cell"] = (p["k_pairs"], L)
        return ks, ls
    ks = sorted({p["k_pairs"] for p in preds})
    vals = sorted({p["context_chars"] for p in preds})
    if len(vals) <= 12:
        return ks, vals
    step = max(1, (vals[-1] - vals[0]) // 8 + 1)
    for p in preds:
        p["cell"] = (p["k_pairs"], (p["context_chars"] // step) * step)
    return ks, sorted({p["cell"][1] for p in preds})


# ---------------------------------------------------------------------------
# model + scoring
# ---------------------------------------------------------------------------


def load_model(model_path: str, tokenizer_path: str | None, device: str, dtype: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {}
    if dtype and dtype != "auto":
        kwargs["dtype"] = getattr(torch, dtype)
    else:
        kwargs["dtype"] = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    model.to(device)
    model.eval()
    tok_src = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def score_item(model, tok, item: dict, device: str, max_length: int, batch: int = 4, unk_id=None):
    """Return per-option (sum_logprob, mean_logprob, n_tokens) for one question."""
    import torch

    prompt_ids = encode(tok, item["context"])
    option_ids = [encode(tok, str(c)) for c in item["choices"]]
    max_opt = max((len(o) for o in option_ids), default=0)
    if max_opt == 0:
        raise SystemExit("an option encodes to zero tokens; tokenizer/model mismatch?")
    # left-truncate the prompt so prompt+option fits in max_length (keeps `Answer:`)
    room = max(1, max_length - max_opt)
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    rows, spans = [], []
    for ids in option_ids:
        ids = ids or [tok.eos_token_id]
        full = prompt_ids + ids
        rows.append(full)
        spans.append((len(prompt_ids), len(ids), ids))

    out_sums, out_means, out_n = [], [], []
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        chunk_spans = spans[start : start + batch]
        width = max(len(r) for r in chunk)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, r in enumerate(chunk):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, : len(r)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for i, (p_len, o_len, ids) in enumerate(chunk_spans):
            # logits at position j predict token j+1 -> option token t sits at prompt_len-1+t
            idx = torch.arange(p_len - 1, p_len - 1 + o_len, device=logprobs.device)
            tgt = torch.tensor(ids, dtype=torch.long, device=logprobs.device)
            lp = logprobs[i, idx, tgt]  # advanced indexing -> [o_len]
            total = float(lp.sum())
            out_sums.append(total)
            out_means.append(total / o_len)
            out_n.append(o_len)
    return list(zip(out_sums, out_means, out_n))


def evaluate(
    model,
    tok,
    items: list[dict],
    device: str,
    max_length: int,
    control_shuffle_labels: bool = False,
    control_seed: int = 0,
    quiet: bool = False,
) -> dict:
    """Run every item, return per-item predictions (gold = the file's answer_idx)."""
    try:
        from tqdm import tqdm

        iterator = tqdm(items, desc="depth-retrieval", disable=quiet)
    except Exception:  # pragma: no cover
        iterator = items
    rng = random.Random(control_seed)
    preds = []
    t0 = time.time()
    for it in iterator:
        gold = int(it["answer_idx"])
        if control_shuffle_labels:
            gold = rng.randrange(len(it["choices"]))
        scores = score_item(model, tok, it, device, max_length)
        sums = [s for s, _, _ in scores]
        means = [m for _, m, _ in scores]
        preds.append(
            {
                "id": it.get("id", ""),
                "k_pairs": int(it.get("k_pairs", -1)),
                "context_chars": int(it.get("context_chars", len(it["context"]))),
                "gold": gold,
                "gold_file": int(it["answer_idx"]),
                "pred_sum": int(max(range(len(sums)), key=lambda i: sums[i])),
                "pred_mean": int(max(range(len(means)), key=lambda i: means[i])),
                "sum_logprobs": [round(s, 6) for s in sums],
                "mean_logprobs": [round(m, 6) for m in means],
                "n_tokens": [n for _, _, n in scores],
            }
        )
    elapsed = time.time() - t0
    return {"predictions": preds, "seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt_acc(block: dict, chance: float) -> str:
    if block["n"] == 0:
        return "  n=0"
    flag = "USABLE" if block["usable"] else "not>chance"
    return (
        f"acc={block['acc']:.3f} n={block['n']:4d} wilson95=[{block['wilson95'][0]:.3f},"
        f"{block['wilson95'][1]:.3f}] p={block['p_one_sided_vs_chance']:.2g} [{flag}]"
    )


def build_report(res: dict, items: list[dict], meta: dict) -> str:
    chance = float(meta.get("chance", DEFAULT_CHANCE))
    preds = res["predictions"]
    n = len(preds)
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("depth-retrieval report  (latest-value retrieval, 4-way multiple choice)")
    add("=" * 78)
    add(f"model            : {meta['model']}")
    add(f"dataset          : {meta['data']}")
    add(f"items scored     : {n}" + (f" (limited from {meta['data_n']})" if meta.get("data_n") else ""))
    add(f"tokenizer        : {meta['tokenizer']}")
    add(f"device / dtype   : {meta['device']} / {meta['dtype']}")
    add(f"score modes      : acc = sum log P(option), acc_norm = mean log P(option)/token")
    add(f"max prompt tokens: {meta['max_length']}")
    add(f"CHANCE           : {chance:.2%}  (4-way MC, uniform; distractors = older writes + other keys)")
    if meta.get("control_shuffle_labels"):
        add("CONTROL          : --control-shuffle-labels ON -> gold indices randomised,")
        add("                   so every accuracy below MUST come out ~chance.")
    if meta.get("task_manifest"):
        tm = meta["task_manifest"]
        cells = declared_cells(tm)
        add(
            "task definition  : "
            f"generator n={tm.get('n')} lengths={sorted({L for _, L, _ in cells})} "
            f"K={sorted({K for K, _, _ in cells})} "
            f"repeats={tm.get('repeats_per_query_key')} seed={tm.get('seed')} "
            f"declared chance={tm.get('chance')}"
        )
        add(f"                   {tm.get('answer_format', '')}")
        add(f"                   distractors: {tm.get('distractors', '')}")
        add(f"                   filler source: {tm.get('filler_source', 'n/a')}")
    add("")

    # ---------------- pooled ----------------
    add("-" * 78)
    add("1. POOLED")
    add("-" * 78)
    pooled = {}
    for key, name in (("pred_sum", "acc"), ("pred_mean", "acc_norm")):
        hits = sum(1 for p in preds if p[key] == p["gold"])
        block = stratum_report(hits, n, chance)
        pooled[key] = block
        add(f"  {name:9s} {_fmt_acc(block, chance)}")
    best = max(pooled.values(), key=lambda b: b["acc"] if b["n"] else -1)
    add("")
    add(f"  VERDICT: pooled accuracy is {'ABOVE' if best['usable'] else 'NOT ABOVE'} chance")
    add(f"           (`usable` requires the Wilson 95% lower bound > {chance:.0%}, a stricter")
    add("            bar than point-estimate > chance; it is the flag quoted in EVAL.md)")
    add("")

    # ---------------- K x L ----------------
    add("-" * 78)
    add("2. STRATIFIED BY K (KV pairs) x L (requested context length)")
    add("-" * 78)
    cells = declared_cells(meta.get("task_manifest"))
    ks, ls = assign_cells(preds, cells)
    declared = {(K, L): c for K, L, c in cells}
    grid_note = "requested grid from the generator manifest" if cells else "raw context_chars buckets"
    add(f"  grid: {grid_note}; cells = acc_norm (n_scored/declared) [* = clears chance at 95%]")
    add("  K\\L     " + "".join(f"{f'{v}':>18s}" for v in ls))
    strata = {}
    for k in ks:
        row = f"  {k:<6d}  "
        for L in ls:
            sel = [p for p in preds if p["cell"] == (k, L)]
            hits = sum(1 for p in sel if p["pred_mean"] == p["gold"])
            block = stratum_report(hits, len(sel), chance)
            block["declared_n"] = declared.get((k, L))
            strata[f"K={k},L={L}"] = block
            if block["n"] == 0 and not block["declared_n"]:
                row += f"{'-':>18s}"
                continue
            mark = "*" if block["usable"] else " "
            dn = f"/{block['declared_n']}" if block["declared_n"] else ""
            cell = f"{block['acc']:.3f}({block['n']}{dn}){mark}" if block["n"] else f"-({dn}){mark}"
            row += f"{cell:>18s}"
        add(row)
    add("  * = stratum clears chance (Wilson 95% lower bound > chance). Cell value is acc_norm.")
    add("")
    flat = [v for v in strata.values() if v["n"] > 0]
    if flat:
        n_usable = sum(1 for v in flat if v["usable"])
        add(f"  strata: {len(flat)} non-empty, {n_usable} clear chance")
        best_cell = max(flat, key=lambda v: v["acc"])
        worst_cell = min(flat, key=lambda v: v["acc"])
        add(f"  best  cell acc={best_cell['acc']:.3f} (n={best_cell['n']})")
        add(f"  worst cell acc={worst_cell['acc']:.3f} (n={worst_cell['n']})")
    add("")

    # ---------------- position bias ----------------
    add("-" * 78)
    add("3. ANSWER-POSITION BIAS CHECK")
    add("-" * 78)
    gold_counts = [sum(1 for p in preds if p["gold"] == i) for i in range(4)]
    pred_counts = [sum(1 for p in preds if p["pred_mean"] == i) for i in range(4)]
    add("  position : " + "".join(f"{i:>9d}" for i in range(4)))
    add("  gold  %  : " + "".join(f"{c / max(1, n):>9.3f}" for c in gold_counts))
    add("  pred  %  : " + "".join(f"{c / max(1, n):>9.3f}" for c in pred_counts))
    acc_by_gold = []
    for i in range(4):
        sel = [p for p in preds if p["gold"] == i]
        hits = sum(1 for p in sel if p["pred_mean"] == p["gold"])
        acc_by_gold.append(hits / len(sel) if sel else float("nan"))
    add("  acc|gold : " + "".join(f"{a:>9.3f}" for a in acc_by_gold))
    bias = max(pred_counts) / max(1, n) - min(pred_counts) / max(1, n)
    gold_bias = max(gold_counts) / max(1, n) - min(gold_counts) / max(1, n)
    add("")
    add(f"  gold-position spread = {gold_bias:.3f} (generator guarantees ~0; a large value")
    add("  with a small n just means this subset is unbalanced -- not a task defect)")
    add(f"  position-bias index (max-min of predicted-position mass) = {bias:.3f}")
    add("  interpretation: 0 = perfectly spread; ->1 = the model answers one position")
    add("  regardless of content, in which case any above-chance accuracy is suspect.")
    add("")

    # ---------------- usable ----------------
    add("-" * 78)
    add("4. USABLE FLAG")
    add("-" * 78)
    usable_cells = sorted(k for k, v in strata.items() if v["usable"])
    verdict = {
        "pooled_acc": pooled["pred_sum"]["acc"],
        "pooled_acc_norm": pooled["pred_mean"]["acc"],
        "chance": chance,
        "usable": bool(pooled["pred_sum"]["usable"] or pooled["pred_mean"]["usable"]),
        "wilson95_acc": pooled["pred_sum"]["wilson95"],
        "wilson95_acc_norm": pooled["pred_mean"]["wilson95"],
        "p_value": min(pooled["pred_sum"]["p_one_sided_vs_chance"], pooled["pred_mean"]["p_one_sided_vs_chance"]),
        "position_bias_index": bias,
        "gold_position_spread": gold_bias,
        "acc_by_gold_position": acc_by_gold,
        "strata": strata,
        "usable_cells": usable_cells,
        "n": n,
        "control_shuffle_labels": bool(meta.get("control_shuffle_labels")),
    }
    add(f"  usable = {verdict['usable']}  (chance={chance:.2%}, n={n})")
    add(f"  rule   : usable iff max(acc, acc_norm) has Wilson95 lower bound > chance")
    add(f"  strata clearing chance: {len(usable_cells)}/{len([v for v in strata.values() if v['n']])}"
        + (f" -> {usable_cells}" if usable_cells else ""))
    add("=" * 78)
    return "\n".join(lines), verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate_smoke_data(out: Path) -> Path:
    """Tiny version of the generator: 24 items, one length, low K, builtin filler."""
    cmd = [
        sys.executable,
        str(REPO / "train" / "eval_depth_retrieval.py"),
        "--out",
        str(out),
        "--n",
        "24",
        "--lengths",
        "512",
        "--ks",
        "1,2",
        "--filler-random",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"smoke data generation failed:\n{proc.stdout}\n{proc.stderr}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="score depth-retrieval JSONL (stratified, chance-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", help="checkpoint dir (AutoModelForCausalLM + trust_remote_code)")
    ap.add_argument("--data", help="JSONL from train/eval_depth_retrieval.py")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N items")
    ap.add_argument("--max-length", type=int, default=8192, help="left-truncate prompt tokens to fit")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    ap.add_argument("--chance", type=float, default=DEFAULT_CHANCE, help="chance level for the flags")
    ap.add_argument("--out-json", default=None, help="where to write the machine-readable result")
    ap.add_argument("--control-shuffle-labels", action="store_true",
                    help="randomise gold indices (pipeline sanity control; expect ~chance)")
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="build a tiny random model + tiny dataset, run on CPU (~30s)")
    ap.add_argument("--smoke-dir", default="/tmp/dr_smoke", help="scratch dir for --smoke")
    args = ap.parse_args()

    if args.smoke:
        args.smoke_dir = Path(args.smoke_dir)
        args.smoke_dir.mkdir(parents=True, exist_ok=True)
        if not args.model:
            from eval.smoke_model import make_tiny  # noqa: E402

            args.model = str(make_tiny(args.smoke_dir / "tiny_gdar", variant="gdar"))
        if not args.data:
            print("[smoke] generating a tiny task file via train/eval_depth_retrieval.py ...")
            args.data = str(generate_smoke_data(args.smoke_dir / "dr_smoke.jsonl"))
        args.device = "cpu" if args.device == "auto" else args.device
        args.dtype = "float32" if args.dtype == "auto" else args.dtype
        args.quiet = True
    if not args.model or not args.data:
        raise SystemExit("--model and --data are required (or use --smoke)")

    data_path = Path(args.data)
    items = load_items(data_path, args.limit)
    task_manifest = load_manifest(data_path)

    import torch

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto" and device == "cuda":
        device = "cuda"
    model, tok, device = load_model(args.model, args.tokenizer, device, args.dtype)
    res = evaluate(
        model,
        tok,
        items,
        device,
        args.max_length,
        control_shuffle_labels=args.control_shuffle_labels,
        control_seed=args.control_seed,
        quiet=args.quiet,
    )

    meta = {
        "model": args.model,
        "data": args.data,
        "data_n": len(items),
        "tokenizer": args.tokenizer or args.model,
        "device": device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "control_shuffle_labels": args.control_shuffle_labels,
        "task_manifest": task_manifest,
    }
    report, verdict = build_report(res, items, meta)
    print(report)
    print(f"[timing] scoring took {res['seconds']}s for {len(items)} items")

    out_json = Path(args.out_json) if args.out_json else data_path.with_suffix(".score.json")
    payload = {
        "meta": meta,
        "verdict": verdict,
        "chance": args.chance,
        "predictions": res["predictions"],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[out] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# 移植注记（本配方）：文件逐字来自 gdar_package（``train/eval_depth_retrieval.py`` /
# ``eval/run_depth_retrieval.py``），只把 ``--data-dir`` 的默认值换成本配方的数据根；
# 生成器/评分器都是自包含脚本（``--n``/``--lengths``/``--ks`` 分层 + chance/Wilson/位置偏差），
# 与 EXPERIMENT_MATRIX.md §5 的 T0"受控深度检索"口径一致。
# ---------------------------------------------------------------------------
