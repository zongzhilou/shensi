"""Numerical verification of the model-side switches the E1-E6 ablation tables need.

Nothing here trains anything: these are checks that each switch *does what the
table column says it does*, and that none of them breaks the construction
guarantee ("at initialisation the connection is bit-exactly the DAR update").

E2  source x gate 2x2  : ``Qwen3GDARConfig.gated_ar_preset()`` names the fourth cell
                         (cumulative sources + state address + three gates).  Verified
                         mechanically: the read contexts are the *stream snapshots*
                         taken at block boundaries, bit-equal to the embedding, and
                         the address direction is the accumulated state -- not the
                         delta being written.
E3  gate structure     : ``attn_res_gate_channels`` -- all seven subsets, the single
                         scalar gate and "no gate at all".  Each is the identity at
                         init (bit-exact DAR), each pins exactly the gates it removes
                         *after* the projection (so the parameter set is constant and
                         the removed slices get exactly zero gradient), and the gates
                         it keeps still move.
E4  gate initialisation: identity (deviation + sigmoid biases) vs 0.5-init vs 0-init.
E5  rank               : r in {32, 64, full} -- identity is rank-independent.
E6  block size         : B in {2, 4, 6, 8, 12} -- the read sees ``floor(l/B) + 1``
                         snapshots at layer l, and the identity slice is exact for
                         every B.

Run:  .venv/bin/python models/test_ablation_switches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


from shensi.recipes.paper.gated_delta_attn_res.models.transformers import (
    Qwen3GDARConfig,
    Qwen3GDARForCausalLM,
)  # noqa: E402
from shensi.recipes.paper.gated_delta_attn_res.models.transformers.modeling_qwen3_gdar import (  # noqa: E402
    GATE_CHANNELS,
    AttentionResidual,
    _apply_gate_channels,
)

SMALL = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    max_position_embeddings=64,
    attn_res_block_size=2,
)

#: the E3 rows: only the listed gates exist.  "scalar" collapses the write gate over
#: channels, "none" removes the gate entirely.
SUBSETS = ("d", "e", "w", "de", "dw", "ew")


def _module(**kw) -> AttentionResidual:
    torch.manual_seed(0)
    return AttentionResidual(Qwen3GDARConfig(**SMALL, **kw))


def _model(**kw) -> Qwen3GDARForCausalLM:
    torch.manual_seed(0)
    return Qwen3GDARForCausalLM(Qwen3GDARConfig(**SMALL, **kw))


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
    return ok


def _deviation(p: torch.Tensor, value: float = 1.0) -> float:
    """max|p - value|, for the 'is this gate still at its identity constant' checks."""
    return float((p - value).abs().max())


def _move_off_identity(m: AttentionResidual) -> None:
    """Push every deviation scale away from zero, so the gates are no longer pinned."""
    with torch.no_grad():
        for name in ("decay_scale", "erase_scale", "write_scale"):
            scale = getattr(m, name, None)
            if scale is not None:
                scale.fill_(0.7)


def main() -> int:  # noqa: C901 - one linear battery of checks, as in test_theory.py
    torch.manual_seed(0)
    tokens, hidden = 16, SMALL["hidden_size"]
    prefix = torch.randn(tokens, hidden)
    delta = torch.randn(tokens, hidden)
    blocks = torch.randn(tokens, 3, hidden)
    dar = prefix + delta
    results: list[bool] = []

    # ================= E3: gate structure =================================
    print("E3  gate structure (attn_res_gate_channels)")
    print(f"    values declared by the config: {GATE_CHANNELS}")

    # the default is the reference path: the gates are (1, 0, 1) at init and *move*
    ref = _module(attn_res_gate_param="deviation")
    with torch.no_grad():
        d0, e0, w0 = ref._gates(prefix)
    results.append(
        check(
            "default is 'dew'", ref.gate_channels == "dew", f"gate_channels = {ref.gate_channels!r}"
        )
    )
    _move_off_identity(ref)
    with torch.no_grad():
        d1, e1, w1 = ref._gates(prefix)
    results.append(
        check(
            "'dew' keeps the gate untouched",
            _deviation(d0, 1.0) == 0
            and _deviation(w0, 1.0) == 0
            and float(e0.abs().max()) == 0
            and (_deviation(d1, 1.0) > 0 or float(e1.abs().max()) > 0 or _deviation(w1, 1.0) > 0),
            f"at init max|decay-1|,|erase|,|write-1| = "
            f"{_deviation(d0, 1.0):.1e}, {float(e0.abs().max()):.1e}, {_deviation(w0, 1.0):.1e}; "
            f"after moving the scales max|decay-1| = {_deviation(d1, 1.0):.3f}",
        )
    )

    for channels in GATE_CHANNELS:
        mod = _module(attn_res_gate_param="deviation", attn_res_gate_channels=channels)
        with torch.no_grad():
            decay, erase, write = mod._gates(prefix)
            out, updated, _, _ = mod(prefix, delta, blocks)

        # (a) the construction guarantee survives every value
        ident = bool(torch.equal(updated, dar) and torch.equal(out, dar))
        scalar_ok = write.shape[-1] == 1 if channels == "scalar" else write.shape[-1] == hidden
        results.append(
            check(
                f"'{channels}': identity at init",
                ident and scalar_ok,
                f"updated == prefix+delta: {bool(torch.equal(updated, dar))} "
                f"(whole module: {bool(torch.equal(out, dar))}), gate shape {tuple(write.shape)}",
            )
        )

        # (b) the removed gates are pinned *functionally*, not only at init
        _move_off_identity(mod)
        with torch.no_grad():
            decay, erase, write = mod._gates(prefix)
            moved = mod.update(prefix, delta)[0]
        if channels == "scalar":
            kept = (write - 1.0).abs().amax(dim=-1) > 1e-6  # the write gate moved (either sign)
            pinned = _deviation(decay, 1.0) == 0 and float(erase.abs().max()) == 0
            detail = (
                f"write moves per token: {int(kept.sum())}/{write.shape[0]}, "
                f"one value per token: {write.shape[-1] == 1}, decay/erase pinned: {pinned}"
            )
            ok = bool(kept.all()) and pinned and write.shape[-1] == 1
        elif channels == "none":
            ok = (
                bool(torch.equal(moved, dar))
                and _deviation(decay, 1.0) == 0
                and float(erase.abs().max()) == 0
                and _deviation(write, 1.0) == 0
            )
            detail = (
                f"update == prefix+delta with every scale moved: {bool(torch.equal(moved, dar))}"
            )
        else:
            kept_moved = {
                "d": _deviation(decay, 1.0) > 0,
                "e": float(erase.abs().max()) > 0,
                "w": _deviation(write, 1.0) > 0,
            }
            pinned_ok = (
                (_deviation(decay, 1.0) == 0 if "d" not in channels else True)
                and (float(erase.abs().max()) == 0 if "e" not in channels else True)
                and (_deviation(write, 1.0) == 0 if "w" not in channels else True)
            )
            ok = pinned_ok and all(kept_moved[c] for c in channels) and not torch.equal(moved, dar)
            detail = (
                "kept gates moved "
                + "/".join(f"{c}:{kept_moved[c]}" for c in "dew")
                + f", removed gates exactly pinned: {pinned_ok}"
            )
        results.append(check(f"'{channels}': removed pinned, kept alive", ok, detail))

    # (c) constant parameter set across the E3 rows -- the ablation varies *structure*,
    #     and the price of that choice is that a removed slice is dead weight, so the
    #     dead slices' gradients must be exactly zero (not merely small).  The scales
    #     are moved off zero first: in the deviation parameterisation the gate *head*
    #     receives gradient only through its scale (d gate / d r = sigmoid(r) * scale),
    #     so at the identity point every head gradient is zero by construction.
    n_params = {}
    for channels in GATE_CHANNELS:
        mod = _module(attn_res_gate_param="deviation", attn_res_gate_channels=channels)
        n_params[channels] = sum(p.numel() for p in mod.parameters())
    results.append(
        check(
            "E3 rows share one parameter set",
            len(set(n_params.values())) == 1,
            f"{', '.join(f'{c}:{n}' for c, n in n_params.items())}",
        )
    )

    dead_mod = _module(attn_res_gate_param="deviation", attn_res_gate_channels="e")
    _move_off_identity(dead_mod)
    # one backward pass with *non-zero* scales, so a "kept" head would show a gradient
    out = dead_mod(prefix, delta, None)[0]
    out.sum().backward()
    bias = (
        dead_mod.gate_proj.bias
        if isinstance(dead_mod.gate_proj, nn.Linear)
        else dead_mod.gate_proj[-1].bias
    )
    grad = bias.grad.abs()
    removed = float(grad[:hidden].max())  # decay: pinned to 1, must be dead
    kept = float(grad[hidden : 2 * hidden].max())  # erase: the gate this row keeps
    results.append(
        check(
            "'e': removed slice dead, kept slice alive",
            removed == 0.0 and kept > 0.0,
            f"max|grad bias| decay (removed) = {removed:.3e}, erase (kept) = {kept:.3e}",
        )
    )

    try:
        _module(attn_res_gate_param="deviation", attn_res_gate_channels="dwe")
        results.append(check("invalid value rejected", False, "no error raised"))
    except ValueError as exc:
        results.append(check("invalid value rejected", True, f"ValueError: {str(exc)[:60]}"))

    # a plain function-level check of the switch itself, independent of the module
    ones, zeros, w = _apply_gate_channels(
        torch.ones(2, 4), torch.zeros(2, 4), torch.full((2, 4), 3.0), "w"
    )
    results.append(
        check(
            "_apply_gate_channels('w') pins decay/erase only",
            _deviation(ones, 1.0) == 0
            and float(zeros.abs().max()) == 0
            and _deviation(w, 3.0) == 0,
            f"decay==1: {_deviation(ones, 1.0) == 0}, erase==0: {float(zeros.abs().max()) == 0}, "
            f"write kept at 3.0: {_deviation(w, 3.0) == 0}",
        )
    )

    # ================= E4: gate initialisation ============================
    print("\nE4  gate initialisation (attn_res_gate_init + attn_res_gate_init_bias)")
    # the three E4 rows, exactly as the config documents them
    rows = {
        "identity (deviation)": dict(attn_res_gate_param="deviation"),
        "identity (sigmoid b=4)": dict(attn_res_gate_init="identity", attn_res_gate_init_bias=4.0),
        "identity (sigmoid b=8)": dict(attn_res_gate_init="identity", attn_res_gate_init_bias=8.0),
        "0.5-init (sigmoid b=0)": dict(attn_res_gate_init="zero"),
        "0.5-init (uniform b=0)": dict(attn_res_gate_init="uniform", attn_res_gate_init_bias=0.0),
        "0-init (uniform b=-20)": dict(attn_res_gate_init="uniform", attn_res_gate_init_bias=-20.0),
    }
    measured = {}
    for label, kw in rows.items():
        mod = _module(**kw)
        with torch.no_grad():
            decay, erase, write = mod._gates(prefix)
            _, updated, _, _ = mod(prefix, delta, None)
        rel = float((updated - dar).abs().max() / dar.abs().max())
        measured[label] = rel
        gate_means = (float(decay.mean()), float(erase.mean()), float(write.mean()))
        if label == "identity (deviation)":
            ok = bool(torch.equal(updated, dar)) and gate_means == (1.0, 0.0, 1.0)
        elif label.startswith("identity"):
            ok = rel < 0.05  # (b, -b, b): decay/write open, erase closed
        elif label.startswith("0.5-init"):
            ok = all(abs(m - 0.5) < 0.05 for m in gate_means)
        else:  # 0-init
            ok = all(abs(m) < 1e-6 for m in gate_means) and rel > 0.5
        results.append(
            check(
                f"{label}",
                ok,
                f"gates = ({gate_means[0]:.4f}, {gate_means[1]:.4f}, {gate_means[2]:.4f}), "
                f"rel. deviation from DAR = {rel:.2e}",
            )
        )
    results.append(
        check(
            "sigmoid identity init converges to DAR as b grows",
            measured["identity (sigmoid b=8)"] < measured["identity (sigmoid b=4)"],
            f"b=4 -> {measured['identity (sigmoid b=4)']:.2e}, b=8 -> {measured['identity (sigmoid b=8)']:.2e}",
        )
    )

    # ================= E5: rank ===========================================
    print("\nE5  rank sweep (attn_res_gate_rank / _q_rank / _k_rank)")
    for rank in (32, 64, None):
        kw = dict(attn_res_gate_param="deviation")
        if rank is not None:
            kw.update(attn_res_gate_rank=rank, attn_res_q_rank=rank, attn_res_k_rank=rank)
        mod = _module(**kw)
        with torch.no_grad():
            _, updated, _, _ = mod(prefix, delta, None)
        n = sum(p.numel() for p in mod.parameters())
        results.append(
            check(
                f"rank {'full' if rank is None else rank}: identity at init",
                bool(torch.equal(updated, dar)),
                f"params = {n:,} (gate projection {sum(p.numel() for p in _params(mod.gate_proj)):,}), "
                f"updated == prefix+delta: {bool(torch.equal(updated, dar))}",
            )
        )
    # rank is a parameter-count knob: for hidden H the low-rank form costs 2*r*H per
    # projection against H^2 full-rank, so r = 32 is cheaper than full rank whenever
    # r < H/2.  At H = 64 (the SMALL config) r = 64 is *more* expensive than full rank,
    # which is why the ordering is checked at a size where the comparison is the one E5
    # makes (H = 256: full rank = 3 H^2, r = 32 = 6 * 32 * H).
    n_small = {}
    for rank, label in ((32, "r=32"), (None, "full")):
        kw = dict(
            SMALL,
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            intermediate_size=512,
            attn_res_gate_param="deviation",
        )
        kw["attn_res_block_size"] = 2
        if rank is not None:
            kw.update(attn_res_gate_rank=rank, attn_res_q_rank=rank, attn_res_k_rank=rank)
        n_small[label] = sum(
            p.numel() for p in AttentionResidual(Qwen3GDARConfig(**kw)).parameters()
        )
    results.append(
        check(
            "rank is a parameter-count knob (H=256)",
            n_small["r=32"] < n_small["full"],
            f"{n_small['r=32']:,} (r=32) < {n_small['full']:,} (full rank)",
        )
    )

    # ================= E2: the 2x2 cell named by gated_ar_preset() =========
    print("\nE2  source x gate 2x2 -- gated_ar_preset() is the Gated-AR cell")
    preset = Qwen3GDARConfig.gated_ar_preset()
    results.append(
        check(
            "preset pins the cell",
            preset["attn_res_gate_channels"] == "dew"
            and preset["attn_res_address"] == "state"
            and preset["attn_res_gate_param"] == "deviation"
            and preset["attn_res_block_size"] > 1,
            ", ".join(f"{k}={v!r}" for k, v in preset.items()),
        )
    )
    results.append(
        check(
            "preset is overridable (E6 sweeps the block size)",
            Qwen3GDARConfig.gated_ar_preset(attn_res_block_size=1)["attn_res_block_size"] == 1,
            "gated_ar_preset(attn_res_block_size=1) -> the per-sublayer (GDAR) source",
        )
    )

    # the source axis: what a layer routes over, captured from the real forward pass
    captured: list[torch.Tensor | None] = []
    original_read = AttentionResidual.read

    def spy_read(self, prefix, blocks, state=None):
        captured.append(None if blocks is None else blocks.detach().clone())
        return original_read(self, prefix, blocks, state=state)

    ids = torch.randint(0, SMALL["vocab_size"], (2, 12))
    source_counts = {}
    snapshot_is_stream = {}
    for label, block_size in (("cumulative (gated_ar, B=4)", 4), ("deltas (gdar, B=1)", 1)):
        captured.clear()
        model = Qwen3GDARForCausalLM(
            Qwen3GDARConfig(
                **{**SMALL, **Qwen3GDARConfig.gated_ar_preset(attn_res_block_size=block_size)}
            )
        )
        model.eval()
        AttentionResidual.read = spy_read
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            AttentionResidual.read = original_read
        source_counts[label] = [0 if b is None else b.shape[1] for b in captured]
        # the "cumulative" claim, checked bit-exactly: a snapshot taken at a block
        # boundary is the stream itself -- at layer 0 that is the embedding
        emb = model.model.embed_tokens(ids).reshape(-1, hidden).float()
        snapshot_is_stream[label] = [
            bool(torch.equal(s[:, 0, :].float(), emb))
            for s in captured
            if s is not None and s.shape[1] == 1
        ]

    # 2 layers, B=4: the only snapshot is the embedding, re-read by all four reads.
    ok_cum = source_counts["cumulative (gated_ar, B=4)"] == [1, 1, 1, 1]
    results.append(
        check(
            "B>1 reads cumulative stream snapshots",
            ok_cum,
            f"source counts per read = {source_counts['cumulative (gated_ar, B=4)']} (expected [1, 1, 1, 1]; "
            f"one snapshot per closed block, re-read by every sublayer)",
        )
    )
    same = snapshot_is_stream["cumulative (gated_ar, B=4)"]
    results.append(
        check(
            "the snapshot *is* the stream (bit-equal to the embedding)",
            len(same) == 4 and all(same),
            f"{sum(same)}/{len(same)} captured sources == embed_tokens(input_ids) reshaped to [T, H]",
        )
    )
    ok_delta = source_counts["deltas (gdar, B=1)"] == [1, 2, 3, 4]
    results.append(
        check(
            "B=1 reads per-sublayer deltas",
            ok_delta,
            f"source counts per read = {source_counts['deltas (gdar, B=1)']} (expected [1, 2, 3, 4]: the list grows "
            f"by one sublayer output per write)",
        )
    )

    # the address axis: which direction the erase gate clears.  Measured at the
    # identity point (decay = write = 1, so m = prefix + delta) with lambda -> inf,
    # where the update *is* the projection m - khat <khat, m>:
    #   address="state" -> khat = normalize(norm(prefix + delta)), i.e. the direction of
    #                      m itself, so the whole update is annihilated -- the signature
    #                      that the address came from the *accumulated stream*;
    #   address="delta" -> khat = normalize(delta) (the content being written), so only
    #                      the delta component is removed.
    for addr, expect in (("state", "stream"), ("delta", "delta")):
        mod = _module(
            attn_res_gate_param="deviation", attn_res_update="objective", attn_res_address=addr
        )
        with torch.no_grad():
            mod.k_proj.copy_(torch.eye(hidden))  # khat == normalize(address_src)
            mod.erase_scale.fill_(1e4)  # lambda -> inf: hard projection
            up, _ = mod.update(prefix, delta)
            m = prefix + delta
            state_dir = F.normalize(mod._state(prefix, delta), dim=-1)
            delta_dir = F.normalize(delta, dim=-1)
            scale = m.abs().max()
            rel = {
                "stream": float((state_dir * up).sum(-1).abs().max()) / scale,
                "delta": float((delta_dir * up).sum(-1).abs().max()) / scale,
            }
            kept = float(up.abs().max()) / scale
        if expect == "stream":
            ok = kept < 1e-3 and rel["stream"] < 1e-3
            detail = f"|updated|/|prefix+delta| = {kept:.2e} (the update is annihilated: khat is the stream direction)"
        else:
            ok = kept > 0.5 and rel["delta"] < 1e-3 and rel["stream"] > 0.5
            detail = (
                f"|updated|/|prefix+delta| = {kept:.2f}, |<delta, updated>| = {rel['delta']:.2e}, "
                f"|<stream, updated>| = {rel['stream']:.2f} (only the delta direction is cleared)"
            )
        results.append(check(f"address='{addr}' derives khat from the {expect}", ok, detail))

    # ================= E6: block size =====================================
    print("\nE6  block size sweep (attn_res_block_size)")
    big = dict(SMALL, num_hidden_layers=12)
    for block_size in (2, 4, 6, 8, 12):
        captured.clear()
        model = Qwen3GDARForCausalLM(
            Qwen3GDARConfig(
                **{**big, "attn_res_gate_param": "deviation", "attn_res_block_size": block_size}
            )
        )
        model.eval()
        AttentionResidual.read = spy_read
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            AttentionResidual.read = original_read
        counts = [0 if b is None else b.shape[1] for b in captured]
        # a snapshot is appended when ``layer_idx % B == 0``, *before* that layer's
        # attention read, so both reads of layer `layer` see the floor(layer/B) + 1 snapshots
        expected = [layer // block_size + 1 for layer in range(12) for _ in range(2)]
        results.append(
            check(
                f"B={block_size}: sources = floor(l/B)+1 per layer",
                counts == expected,
                f"counts = {counts} expected {expected}",
            )
        )

    # every B keeps the exact identity slice (gates forced == gates at init)
    from shensi.recipes.paper.gated_delta_attn_res.models.transformers.guarantee import (
        identity_slice_loss,
        rollback_to_identity,
    )  # noqa: E402

    for block_size in (2, 4, 6, 8, 12):
        model = Qwen3GDARForCausalLM(
            Qwen3GDARConfig(
                **{**big, **Qwen3GDARConfig.theory_preset(), "attn_res_block_size": block_size}
            )
        )
        model.train()
        with torch.no_grad():
            base = float(model(input_ids=ids, labels=ids).loss)
            slice_loss = identity_slice_loss(model, ids)
            for m in model.modules():
                if isinstance(m, AttentionResidual):
                    m.decay_scale.fill_(0.7)
                    m.erase_scale.fill_(0.7)
                    m.write_scale.fill_(0.7)
            forced = identity_slice_loss(model, ids)
            moved = float(model(input_ids=ids, labels=ids).loss)
            rollback_to_identity(model)
            after = float(model(input_ids=ids, labels=ids).loss)
        results.append(
            check(
                f"B={block_size}: identity slice exact + rollback",
                abs(slice_loss - base) < 1e-6
                and abs(slice_loss - forced) < 1e-6
                and moved != slice_loss
                and after == slice_loss,
                f"init={base:.4f} slice={slice_loss:.4f} forced={forced:.4f} moved={moved:.4f} "
                f"after rollback={after:.4f}",
            )
        )

    # ================= cross-switch: model-level smoke =====================
    print("\ncross-switch: the full model runs for every E3 value (theory preset + read)")
    for channels in GATE_CHANNELS:
        kw = Qwen3GDARConfig.theory_preset(attn_res_gate_channels=channels)
        model = _model(**kw)
        model.train()
        out = model(input_ids=ids, labels=ids, return_attn_res_stats=True)
        out.loss.backward()
        g = [s for s in out.attn_res_stats if "gate_decay" in s][0]
        bad = sum(
            1
            for _, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        )
        results.append(
            check(
                f"theory preset x '{channels}'",
                bool(torch.isfinite(out.loss)) and bad == 0,
                f"loss = {out.loss.item():.3f}, gates(d/e/w) = "
                f"({g['gate_decay']:.4f}, {g['gate_erase']:.4f}, {g['gate_write']:.4f}), non-finite grads = {bad}",
            )
        )

    # ================= config plumbing =====================================
    print("\nconfig plumbing: the new knob serialises and defaults to the reference")
    cfg = Qwen3GDARConfig(**SMALL, **Qwen3GDARConfig.theory_preset())
    as_dict = cfg.to_dict()
    results.append(
        check(
            "default keeps the reference behaviour",
            as_dict.get("attn_res_gate_channels") == "dew",
            f"to_dict()['attn_res_gate_channels'] = {as_dict.get('attn_res_gate_channels')!r}",
        )
    )
    cfg2 = Qwen3GDARConfig(**SMALL, attn_res_gate_channels="scalar")
    results.append(
        check(
            "non-default knob round-trips through to_dict",
            cfg2.to_dict().get("attn_res_gate_channels") == "scalar",
            f"to_dict()['attn_res_gate_channels'] = {cfg2.to_dict().get('attn_res_gate_channels')!r}",
        )
    )
    results.append(
        check(
            "gated_ar_preset serialises",
            Qwen3GDARConfig(**{**SMALL, **Qwen3GDARConfig.gated_ar_preset()})
            .to_dict()
            .get("attn_res_block_size")
            == 4,
            "attn_res_block_size = 4 (cumulative block snapshots)",
        )
    )

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


def _params(proj):
    if isinstance(proj, nn.Parameter):
        return [proj]
    return list(proj.parameters())


if __name__ == "__main__":
    raise SystemExit(main())
