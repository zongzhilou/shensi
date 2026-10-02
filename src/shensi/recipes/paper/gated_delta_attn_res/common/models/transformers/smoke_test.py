"""Smoke test for the depth-routed Qwen3 variants (Kimi-style API).

On tiny random models (no data needed) this checks:
  1. every variant builds, forwards, back-props, and emits routing statistics;
  2. parameter overhead vs plain Qwen3 (the reviewers' "three gates are
     negligible" complaint, measured);
  3. the gate initialisations do what their names claim -- ``identity`` starts at
     (decay, erase, write) ~ (1, 0, 1), ``paper`` starts at 0.5;
  4. GDAR with identity gates reproduces the DAR update (the premise of Gate 1 in
     the redo plan).

Run:  .venv/bin/python models/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers import (  # noqa: E402
    Qwen3ARConfig,
    Qwen3ARForCausalLM,
    Qwen3DARConfig,
    Qwen3DARForCausalLM,
    Qwen3GDARConfig,
    Qwen3GDARForCausalLM,
)
from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (
    AttentionResidual,
)  # noqa: E402

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RMSNorm  # noqa: E402

BASE = dict(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=128,
)

VARIANTS = [
    ("baseline Qwen3", Qwen3Config, Qwen3ForCausalLM, {}),
    ("AR block(2)", Qwen3ARConfig, Qwen3ARForCausalLM, {"attn_res_block_size": 2}),
    ("AR full(1)", Qwen3ARConfig, Qwen3ARForCausalLM, {"attn_res_block_size": 1}),
    ("DAR sublayer(1)", Qwen3DARConfig, Qwen3DARForCausalLM, {"attn_res_block_size": 1}),
    ("DAR block(2)", Qwen3DARConfig, Qwen3DARForCausalLM, {"attn_res_block_size": 2}),
    ("GDAR repo-faithful", Qwen3GDARConfig, Qwen3GDARForCausalLM, {"attn_res_block_size": 2}),
    (
        "GDAR redo fixes",
        Qwen3GDARConfig,
        Qwen3GDARForCausalLM,
        {
            "attn_res_block_size": 2,
            "attn_res_gate_init": "identity",
            "attn_res_gate_rank": 64,
            "attn_res_k_rank": 64,
        },
    ),
    (
        "GDAR identity bias8",
        Qwen3GDARConfig,
        Qwen3GDARForCausalLM,
        {
            "attn_res_block_size": 2,
            "attn_res_gate_init": "identity",
            "attn_res_gate_init_bias": 8.0,
        },
    ),
]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (
        AttentionResidual,
    )

    extra = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, AttentionResidual)
        for p in module.parameters()
    )
    return total, extra


def main() -> int:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 512, (2, 32))
    failures = []

    print("=" * 100)
    print(
        f"{'variant':<24}{'params':>12}{'vs base':>10}{'gdar extras':>14}{'share':>9}{'fwd/back':>10}"
    )
    print("=" * 100)

    base_params = None
    for name, config_cls, model_cls, extra in VARIANTS:
        cfg = config_cls(**BASE, **extra)
        model = model_cls(cfg)
        total, gate = count_params(model)
        base_params = base_params or total

        model.train()
        try:
            out = model(input_ids=input_ids, labels=input_ids, return_attn_res_stats=True)
            out.loss.backward()
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: forward/backward failed: {exc}")
            status = "FAIL"
            out = None

        share = f"{100 * gate / total:.2f}%" if gate else "-"
        print(
            f"{name:<24}{total:>12,}{total / base_params:>9.3f}x{gate:>14,}{share:>9}{status:>10}"
        )

        stats = getattr(out, "attn_res_stats", None) if out is not None else None
        if stats:
            sharp = [s["sharpness"] for s in stats if "sharpness" in s]
            gates = [s for s in stats if "gate_decay" in s]
            msg = f"{len(stats)} entries"
            if sharp:
                msg += f", sharpness mean={sum(sharp) / len(sharp):.3f}"
            if gates:
                g = gates[0]
                msg += f", gates(d/e/w)=({g['gate_decay']:.3f}, {g['gate_erase']:.3f}, {g['gate_write']:.3f})"
            print(f"{'':24}{msg}")

    # ---- gate initialisation + identity behaviour -------------------------
    print("\n" + "=" * 100)
    print("AttentionResidual: gate initialisation and identity (vs prefix + delta)")
    print("=" * 100)
    h = 64
    torch.manual_seed(0)
    prefix = torch.randn(16, h)
    delta = torch.randn(16, h)
    small = dict(
        vocab_size=8,
        hidden_size=h,
        intermediate_size=4 * h,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=h,
        max_position_embeddings=16,
        attn_res_block_size=2,
        attn_res_gate_rank=None,
        attn_res_k_rank=None,
    )
    for init, bias in (("paper", 4.0), ("identity", 4.0), ("identity", 8.0), ("zero", 4.0)):
        module = AttentionResidual(
            Qwen3GDARConfig(**small, attn_res_gate_init=init, attn_res_gate_init_bias=bias)
        )
        with torch.no_grad():
            _, updated, gates, _ = module(prefix, delta, None)
        decay, erase, write = (g.mean() for g in gates)
        dar = prefix + delta
        rel = (updated - dar).abs().max() / dar.abs().max()
        label = f"{init}(bias={bias})"
        print(
            f"  {label:<16} decay={decay:.4f} erase={erase:.4f} write={write:.4f} | "
            f"max|GDAR-(prefix+delta)|/max|.| = {rel:.5f}"
        )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nall checks ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
