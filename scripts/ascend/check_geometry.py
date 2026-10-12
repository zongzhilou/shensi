#!/usr/bin/env python3
"""Offline dry-run of the on-box geometry check (shensi.recipes...args.verify_shensi_geometry).

Why this exists: the training entry refuses to start when the HF geometry json and the
profile disagree (it raises RuntimeError listing every mismatched field). That check needs
no GPU, so this tool replays the same comparison offline -- from a laptop, right after
editing the profile or the json -- and tells you PASS / FAIL before you queue the job.

It also checks the json's *internal* consistency (list lengths against num_hidden_layers) and,
for information, whether the per-layer plan follows the transformers fork's *default* derivation
(`configuration_shensi.py.__post_init__`: 2x heavily-compressed bootstrap + alternating
compressed/heavily, truncated to num_hidden_layers). An explicit list in the json legitimately
overrides that default, so a deviation is reported as a note, not a failure -- the authoritative
V1-N plan is whatever the shipped `2p8b_a1p2b.json` says, and the profile must match it.

Scope: this is an independent re-implementation of the same rules, not an import of the
training code -- it needs no torch / megatron / transformers, and it deliberately covers
only the checks that matter for the geometry (see KNOWN LIMITS below).

KNOWN LIMITS
- The profile is parsed with a flat `key: value` regex, which is enough for the shipped
  profiles (scalars under train.model / train.system) but would miss list or nested values.
- `vocab_size` is compared against the tokenizer vocabulary (129280 for the shipped
  tokenizer) because the padded value is only known at run time; the real check compares
  the padded size, which is a multiple of it.
- Legacy fields the fork pops before validation (`compress_ratios`, `compress_rate_csa`,
  `num_hash_layers`, `qk_rope_head_dim`) are checked for internal consistency only.

Usage:
    python check_geometry.py --repo /root/work/shensi-offline/shensi
    # 默认读仓库自带档位与权威 json：
    #   config = <repo>/src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain/config/default.yaml
    #   hf     = <repo>/src/shensi/recipes/shensi/common/config/hf/2p8b_a1p2b.json
    # 也可 --config / --hf 单独覆盖。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# (json field, profile key) -- mirrors args.SHENSI_HF_PARITY_FIELDS
PARITY_FIELDS = (
    ("n_routed_experts", "shensi_num_experts"),
    ("num_experts_per_tok", "shensi_num_experts_per_tok"),
    ("moe_intermediate_size", "shensi_moe_intermediate_size"),
    ("routed_expert_hidden_size", "shensi_routed_expert_hidden_size"),
    ("head_dim", "shensi_head_dim"),
    ("q_lora_rank", "shensi_q_lora_rank"),
    ("o_lora_rank", "shensi_o_lora_rank"),
    ("o_groups", "shensi_o_groups"),
    ("index_n_heads", "shensi_index_n_heads"),
    ("index_head_dim", "shensi_index_head_dim"),
    ("index_topk", "shensi_index_topk"),
    # hc_* 三项已在 hc 移除后从 json 与档位同时删除（2026-10 上游），不再对拍。
    ("attn_res_block_size", "shensi_attn_res_block_size"),
    ("erc_loss_alpha", "shensi_erc_loss_alpha"),
    ("erc_loss_coef", "shensi_erc_loss_coef"),
    ("num_hidden_layers", "num_layers"),
    ("hidden_size", "hidden_size"),
    ("num_attention_heads", "num_attention_heads"),
    ("router_aux_loss_coef", "moe_aux_loss_coeff"),
    ("num_nextn_predict_layers", "mtp_num_layers"),
)

LAYER_TYPE_TO_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}
RATIO_TO_LAYER_TYPE = {v: k for k, v in LAYER_TYPE_TO_RATIO.items()}

TOKENIZER_VOCAB_SIZE = 129280


def read_profile(path: pathlib.Path, _seen: set | None = None) -> dict[str, float]:
    """Flat `key: value` scalars from a profile yaml, resolving a top-level `base:` first.

    `thru.yaml` / `scale8.yaml` 只写覆盖项并 `base: default.yaml`；不解析 base 就会误报
    「缺几何键」。子档的值覆盖 base。
    """
    _seen = _seen or set()
    text = path.read_text()
    out: dict[str, float] = {}
    m = re.search(r"^base:\s*([^\s#]+)", text, re.M)
    if m:
        base = (path.parent / m.group(1)).resolve()
        if base.exists() and base not in _seen:
            _seen.add(base)
            out.update(read_profile(base, _seen))
    for line in text.splitlines():
        mm = re.match(r"^\s{4,6}([a-z_][a-z0-9_]*):\s*(.+?)\s*(?:#.*)?$", line)
        if not mm:
            continue
        key, raw = mm.group(1), mm.group(2).strip().strip('"').strip("'")
        if raw.startswith("${"):
            continue
        # yaml 的 null/~/空值要当"未设置"，否则会被当成字符串 "null"，
        # 让下游的 `or ""` 兜底失效、`int("null")` 直接抛 ValueError（tiny 档就踩这个）。
        if raw == "" or raw.lower() in ("null", "~", "none"):
            continue
        try:
            out[key] = float(raw)
        except ValueError:
            out[key] = raw  # type: ignore[assignment]
    return out


def derive_layer_types(n: int) -> list[str]:
    """The fork's default: 2x heavily-compressed bootstrap, then alternating CSA/HCA."""
    interleave = [
        "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
        for i in range(max(n - 2, 0))
    ]
    return ["heavily_compressed_attention"] * min(n, 2) + interleave


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo",
        default="/root/work/shensi-offline/shensi",
        help="shensi 仓库根（默认与 migrate.sh 一致）；config/hf 的默认路径由它推出",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="训练档位 yaml；默认 <repo>/.../stage1_pretrain/config/default.yaml",
    )
    ap.add_argument(
        "--hf",
        default=None,
        help="HF 几何 json；默认 <repo>/src/shensi/recipes/shensi/common/config/hf/2p8b_a1p2b.json",
    )
    ap.add_argument("--vocab-size", type=int, default=TOKENIZER_VOCAB_SIZE)
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    config = pathlib.Path(args.config or (
        repo / "src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain/config/default.yaml"
    ))
    hf_path = pathlib.Path(args.hf or (
        repo / "src/shensi/recipes/shensi/common/config/hf/2p8b_a1p2b.json"
    ))

    prof = read_profile(config)
    hf = json.loads(hf_path.read_text())
    bad: list[str] = []
    notes: list[str] = []
    checked = 0

    # ---- 1. json internal consistency + the fork's derivation rule ----
    n = hf.get("num_hidden_layers")
    mtp = hf.get("num_nextn_predict_layers") or 0
    if n is None:
        bad.append("num_hidden_layers: json is missing it")
        n = 0
    lt = [str(t) for t in (hf.get("layer_types") or [])]
    mlp = [str(t) for t in (hf.get("mlp_layer_types") or [])]
    cr = list(hf.get("compress_ratios") or [])

    if not lt:
        bad.append("layer_types: json is missing it")
    elif len(lt) != n:
        bad.append(f"layer_types: {len(lt)} entries but num_hidden_layers={n}")
    else:
        unknown = [t for t in lt if t not in LAYER_TYPE_TO_RATIO]
        if unknown:
            bad.append(f"layer_types: unknown entries {unknown}")
        else:
            checked += 1
            default_plan = derive_layer_types(n)
            if lt == default_plan:
                notes.append("layer_types matches the fork's default derivation")
            else:
                notes.append(
                    "layer_types is an explicit list (fork default would be "
                    f"{[LAYER_TYPE_TO_RATIO[t] for t in default_plan]}) -- fine, "
                    "the profile must match the json, which is what is checked below"
                )
    if not mlp:
        bad.append("mlp_layer_types: json is missing it")
    elif len(mlp) != n:
        bad.append(f"mlp_layer_types: {len(mlp)} entries but num_hidden_layers={n}")
    else:
        n_hash = mlp.count("hash_moe")
        expect_mlp = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
        if mlp != expect_mlp:
            bad.append("mlp_layer_types: hash_moe entries must form a leading prefix")
        else:
            checked += 1
    if cr:
        if len(cr) != n + mtp:
            bad.append(f"compress_ratios: {len(cr)} entries but num_hidden_layers+mtp={n + mtp}")

    # ---- 2. json vs profile (the same field-by-field comparison the box runs) ----
    for field, key in PARITY_FIELDS:
        if field not in hf:
            bad.append(f"{field}: json is missing it")
            continue
        ours = prof.get(key)
        if ours is None:
            bad.append(f"{field}: profile does not set {key}")
            continue
        ours = 0 if ours in (None, "") else ours
        try:
            same = float(ours) == float(hf[field])
        except (TypeError, ValueError):
            same = ours == hf[field]
        if same:
            checked += 1
        else:
            bad.append(f"{field}: profile={ours} json={hf[field]}")

    # layer_types -> ratios must equal the profile's per-layer compress ratios (prefix)
    ours_ratios = [
        int(x) for x in str(prof.get("shensi_compress_ratios") or "").split(",") if x.strip()
    ]
    if lt and all(t in LAYER_TYPE_TO_RATIO for t in lt) and len(lt) == n:
        hf_ratios = [LAYER_TYPE_TO_RATIO[t] for t in lt]
        if ours_ratios[: len(hf_ratios)] == hf_ratios:
            checked += 1
        else:
            bad.append(
                "shensi_compress_ratios (profile) does not match layer_types (json):\n"
                f"      profile[:{len(hf_ratios)}] = {ours_ratios[:len(hf_ratios)]}\n"
                f"      json layer_types  = {hf_ratios}"
            )

    # mlp hash prefix must equal the profile's
    if mlp and len(mlp) == n:
        ours_mlp = [x.strip() for x in str(prof.get("shensi_mlp_layer_types") or "").split(",")]
        ours_n_hash = ours_mlp.count("hash_moe")
        if mlp.count("hash_moe") == ours_n_hash:
            checked += 1
        else:
            bad.append(
                f"mlp hash prefix: json={mlp.count('hash_moe')} profile={ours_n_hash}"
            )

    # sliding_window
    if "sliding_window" in hf:
        ours = prof.get("shensi_sliding_window")
        if ours is not None and float(ours) == float(hf["sliding_window"]):
            checked += 1
        else:
            bad.append(f"sliding_window: profile={ours} json={hf['sliding_window']}")
    else:
        bad.append("sliding_window: json is missing it")

    # vocab_size: the run-time value is the padded size, a multiple of the tokenizer vocab
    if "vocab_size" in hf:
        theirs = int(hf["vocab_size"])
        ours = int(args.vocab_size)
        if theirs > 0 and ours >= theirs and ours % theirs == 0:
            checked += 1
        else:
            bad.append(
                f"vocab_size: tokenizer={ours} json={theirs} "
                "(must be >= and divisible after up-padding)"
            )
    else:
        bad.append("vocab_size: json is missing it")

    # rope: the compressor layers are YaRN; factor / original_max_position must agree
    rp = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
    compress = rp.get("compress") if isinstance(rp, dict) else None
    if not isinstance(compress, dict):
        compress = (
            {k: v for k, v in rp.items() if k not in ("main", "compress")}
            if isinstance(rp, dict)
            else {}
        )
    rtype = str(compress.get("rope_type", compress.get("type", "default")))
    ours_factor = float(prof.get("rotary_scaling_factor") or 1.0)
    if rtype != "yarn":
        if ours_factor != 1.0:
            bad.append(
                f"rope_parameters.compress.rope_type: json={rtype} (no YaRN) but "
                f"profile rotary_scaling_factor={ours_factor}"
            )
        else:
            checked += 1
    elif "factor" not in compress:
        bad.append("rope_parameters.compress.factor: json says yarn but has no factor")
    elif abs(ours_factor - float(compress["factor"])) > 1e-9:
        bad.append(
            f"rope_parameters.compress.factor: profile={ours_factor} json={compress['factor']}"
        )
    else:
        checked += 1
        theirs_orig = compress.get("original_max_position_embeddings")
        if theirs_orig is not None:
            ours_orig = int(prof.get("original_max_position_embeddings") or 0)
            if ours_orig == int(theirs_orig):
                checked += 1
            else:
                bad.append(
                    "rope_parameters.compress.original_max_position_embeddings: "
                    f"profile={ours_orig} json={theirs_orig}"
                )

    print(f"[geom] config={config}")
    print(f"[geom] hf    ={hf_path}")
    print(f"[geom] geometry: num_hidden_layers={n} mtp={mtp} compress_ratios={len(cr)}")
    print(f"[geom] per-layer ratios: {ours_ratios or '[not set in profile]'}")
    for x in notes:
        print(f"[geom] note: {x}")
    if bad:
        print(f"[geom] FAIL: {len(bad)} problem(s) -- the on-box check would refuse to start:")
        for x in bad:
            print(f"  - {x}")
        return 1
    print(f"[geom] PASS: {checked} fields agree -> verify_shensi_geometry should PASS on the box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
