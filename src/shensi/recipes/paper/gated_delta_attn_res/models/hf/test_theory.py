"""Numerical verification of the propositions behind the theory-optimal GDAR.

Nothing here trains anything: these are checks that the implementation really does
satisfy the statements written in ``modeling_qwen3_gdar.py``.

T1  identity gates      : ``attn_res_gate_param="deviation"`` gives gates that are
                          exactly (decay, erase, write) = (1, 0, 1) at init.
T2  exact DAR limit     : with those gates the update is bit-exactly
                          ``prefix + delta`` (the DAR rule), for both update rules.
T3  Proposition 1       : ``m - lambda/(1+lambda) * khat <khat, m>`` is the unique
                          minimiser of  J(h') = 1/2||h'-m||^2 + (lambda/2)<khat,h'>^2
                          -- checked by numerically minimising J.
T4  Proposition 2       : as lambda -> inf the update becomes the orthogonal
                          projection, which leaves every direction orthogonal to
                          khat exactly invariant (minimum-norm correction).
T5  cascade ladder      : the decay's per-channel time constants are learned, starting on a
                          geometric ladder over [1, 100], and still give decay = 1 at init.

Run:  .venv/bin/python models/test_theory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Qwen3GDARConfig, Qwen3GDARForCausalLM  # noqa: E402
from models.modeling_qwen3_gdar import AttentionResidual  # noqa: E402
from models.modeling_qwen3_gdar import AttentionResidual, _depth_read, _softmax1  # noqa: E402

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


def _module(**kw) -> AttentionResidual:
    torch.manual_seed(0)
    return AttentionResidual(Qwen3GDARConfig(**SMALL, **kw))


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    tokens, hidden = 16, SMALL["hidden_size"]
    prefix = torch.randn(tokens, hidden, dtype=torch.float32)
    delta = torch.randn(tokens, hidden, dtype=torch.float32)
    blocks = torch.randn(tokens, 3, hidden, dtype=torch.float32)
    results = []

    # ---------------- T1: identity gates ---------------------------------
    print("T1  identity gate initialisation (attn_res_gate_param='deviation')")
    mod = _module(attn_res_gate_param="deviation")
    with torch.no_grad():
        decay, erase, write = mod._gates(prefix)
    results.append(check("decay == 1", bool(torch.all(decay == 1)), f"max|decay-1| = {(decay - 1).abs().max():.3e}"))
    results.append(check("erase == 0", bool(torch.all(erase == 0)), f"max|erase| = {erase.abs().max():.3e}"))
    results.append(check("write == 1", bool(torch.all(write == 1)), f"max|write-1| = {(write - 1).abs().max():.3e}"))
    gw = mod.gate_proj.weight if hasattr(mod.gate_proj, "weight") else mod.gate_proj[0].weight
    results.append(check("gate weights zero (flat point)", bool(torch.all(gw == 0)), f"max|W_gate| = {gw.abs().max():.3e}"))

    # ---------------- T2: exact DAR limit ---------------------------------
    print("\nT2  exact DAR limit: updated == prefix + delta")
    for rule in ("shensi", "objective"):
        mod = _module(attn_res_gate_param="deviation", attn_res_update=rule)
        with torch.no_grad():
            _, updated, _, _ = mod(prefix, delta, None)
        dar = prefix + delta
        results.append(
            check(
                f"update_rule='{rule}'",
                bool(torch.equal(updated, dar)),
                f"max|updated-(prefix+delta)| = {(updated - dar).abs().max():.3e}",
            )
        )

    # ... and the whole module, read included: the read's own gate starts at zero
    mod = _module(attn_res_gate_param="deviation", attn_res_update="objective")
    with torch.no_grad():
        out, _, _, _ = mod(prefix, delta, blocks)
    dar = prefix + delta
    results.append(
        check(
            "whole module output (read on)",
            bool(torch.equal(out, dar)),
            f"max|output-(prefix+delta)| = {(out - dar).abs().max():.3e}, read_scale = {mod.read_scale.item()}",
        )
    )

    # ---------------- T3: Proposition 1 ----------------------------------
    print("\nT3  Proposition 1: closed form == numerical minimiser of J(h')")
    khat = F.normalize(torch.randn(tokens, hidden), dim=-1)
    m = torch.randn(tokens, hidden)
    def _J(h, lam):
        return 0.5 * (h - m).pow(2).sum() + 0.5 * lam * (khat * h).sum(-1).pow(2).sum()

    for lam in (0.0, 0.3, 5.0):
        closed = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(-1, keepdim=True)
        # (a) stationarity: the gradient of J vanishes at the closed form
        hc = closed.clone().requires_grad_(True)
        _J(hc, lam).backward()
        grad_norm = hc.grad.abs().max()
        results.append(check(f"lambda={lam}: grad J(h*) == 0", bool(grad_norm < 1e-5), f"max|grad| = {grad_norm:.3e}"))
        # (b) global: J(h*) beats every random point (strong convexity)
        worst = min((_J(closed, lam) - _J(torch.randn_like(m), lam)).item() for _ in range(200))
        results.append(check(f"lambda={lam}: J(h*) < J(random)", bool(worst < 0), f"min margin = {worst:.3e}"))
        # (c) LBFGS converges to the closed form
        h = torch.randn_like(m, requires_grad=True)
        opt = torch.optim.LBFGS([h], lr=1.0, max_iter=200)
        for _ in range(20):
            def closure():
                opt.zero_grad()
                loss = _J(h, lam)
                loss.backward()
                return loss

            opt.step(closure)
        results.append(
            check(f"lambda={lam}: LBFGS -> closed form", bool(torch.allclose(h.detach(), closed, atol=1e-4)), f"max diff = {(h.detach() - closed).abs().max():.3e}")
        )

    # ---------------- T4: Proposition 2 ----------------------------------
    print("\nT4  Proposition 2: lambda->inf projection leaves the orthogonal complement invariant")
    lam_big = 1e8
    closed = m - (lam_big / (1.0 + lam_big)) * khat * (khat * m).sum(-1, keepdim=True)
    # build an explicit orthogonal direction per token
    v = torch.randn(tokens, hidden)
    v = v - khat * (khat * v).sum(-1, keepdim=True)          # v ⟂ khat
    lhs = (v * closed).sum(-1)
    rhs = (v * m).sum(-1)
    results.append(
        check("<v, h'> == <v, m> for v ⟂ khat", bool(torch.allclose(lhs, rhs, atol=1e-4)), f"max diff = {(lhs - rhs).abs().max():.3e}")
    )
    # and the khat component is annihilated
    proj = (khat * closed).sum(-1)
    results.append(check("<khat, h'> == 0", bool(proj.abs().max() < 1e-3), f"max|<khat,h'>| = {proj.abs().max():.3e}"))

    # ---------------- T5: cascade ladder ----------------------------------
    print("\nT5  multi-timescale decay ladder")
    ladder = 8
    mod = _module(attn_res_gate_param="deviation", attn_res_decay_ladder=ladder, attn_res_decay_tau_max=100.0)
    with torch.no_grad():
        decay0, _, _ = mod._gates(prefix)
        # move the decay scale to a non-trivial value and look at the spread
        mod.decay_scale.fill_(0.5)
    decay1, _, _ = mod._gates(prefix)
    tau = mod.decay_tau.exp()  # the parameter holds log tau, so tau stays positive
    results.append(check("tau spans a geometric range", bool(tau.min() == 1.0 and tau.max() > 50), f"tau in [{tau.min():.2f}, {tau.max():.2f}]"))
    results.append(
        check(
            "tau is a learnable parameter (log space)",
            bool(isinstance(mod.decay_tau, nn.Parameter) and mod.decay_tau.requires_grad and tau.gt(0).all()),
            f"nn.Parameter={isinstance(mod.decay_tau, nn.Parameter)}, requires_grad={mod.decay_tau.requires_grad}",
        )
    )
    decay1.sum().backward()
    grad_tau = mod.decay_tau.grad.abs().sum().item()
    results.append(check("tau receives gradient once decay moves", bool(grad_tau > 0), f"|grad tau| = {grad_tau:.3e}"))
    results.append(check("decay == 1 at init (all channels)", bool(torch.all(decay0 == 1)), f"max|decay-1| = {(decay0 - 1).abs().max():.3e}"))
    per_token = decay1[0]
    spread = (per_token.max() - per_token.min()).item()
    results.append(
        check(
            "decay becomes channel-heterogeneous",
            bool(spread > 0.05),
            f"channel spread = {spread:.3f} (tau in [{tau.min():.0f}, {tau.max():.0f}] -> decay in [{per_token.min():.3f}, {per_token.max():.3f}])",
        )
    )

    # ---------------- T7: multi-head read ---------------------------------
    print("\nT7  multi-head read")
    rr = torch.randn(4, 5, 32)                                   # (T, N, D)
    qq = torch.randn(4, 32)
    ref_scores = (rr * qq.unsqueeze(1)).sum(-1) * torch.rsqrt(rr.square().mean(-1) + 1e-6)
    ref = (ref_scores.softmax(-1).unsqueeze(-1) * rr).sum(1)
    out1 = _depth_read(rr, qq, 1e-6, heads=1)
    results.append(check("H=1 == reference formula", bool(torch.allclose(out1, ref, atol=1e-6)), f"max diff = {(out1 - ref).abs().max():.3e}"))
    out4 = _depth_read(rr, qq, 1e-6, heads=4)
    results.append(check("H=4 differs from H=1", bool(not torch.allclose(out4, out1, atol=1e-4)), "heads are not a no-op"))
    _, p4 = _depth_read(rr, qq, 1e-6, heads=4, return_scores=True)
    results.append(check("per-head weights sum to 1", bool(torch.allclose(p4.sum(dim=1), torch.ones_like(p4.sum(dim=1)))), f"shape={tuple(p4.shape)}"))
    try:
        AttentionResidual(Qwen3GDARConfig(**{**SMALL, "attn_res_read_heads": 3}))
        results.append(check("indivisible heads rejected", False, "no error raised"))
    except ValueError:
        results.append(check("indivisible heads rejected", True, "ValueError raised"))

    # ---------------- T8: Softmax_1 null route ----------------------------
    print("\nT8  Softmax_1 null route")
    N = 5
    vals = torch.randn(3, N, 16)
    zero_q = torch.zeros(3, 16)
    _, pn = _depth_read(vals, zero_q, 1e-6, null=True, return_scores=True)
    expected = 1.0 / (1.0 + N)
    results.append(
        check("zero logits -> null mass = 1/(1+N)", bool(abs(float(1 - pn.sum(-1).mean()) - expected) < 1e-6), f"null={float(1 - pn.sum(-1).mean()):.4f} expected={expected:.4f}")
    )
    _, ps = _depth_read(vals, zero_q, 1e-6, null=False, return_scores=True)
    results.append(check("plain softmax has no null mass", bool(abs(float(1 - ps.sum(-1).mean())) < 1e-6), f"null={float(1 - ps.sum(-1).mean()):.2e}"))

    # ---- T8b: Softmax_1 numerical stability at extreme logits -------------
    print("\nT8b Softmax_1 stays finite and differentiable at extreme logits")
    for scale in (1e3, 1e6, -1e3, -1e6):
        z = torch.randn(3, 5, requires_grad=True) * scale
        p = _softmax1(z, dim=-1)
        (g,) = torch.autograd.grad(p.sum(), z)
        null = 1.0 - p.sum(-1)
        ok = bool(
            torch.isfinite(p).all() and torch.isfinite(g).all() and (p.sum(-1) <= 1 + 1e-6).all() and (null >= -1e-6).all()
        )
        results.append(check(f"logits scale {scale:+.0e}", ok, f"probs finite={bool(torch.isfinite(p).all())} grads finite={bool(torch.isfinite(g).all())} null mass={float(null.mean()):.3f}"))

    # ---------------- T9: whitening implements the Mahalanobis read --------
    print("\nT9  whitened read implements v^T Sigma^-1 q (Mahalanobis scoring)")
    tv = torch.randn(2, 6, 16)
    tq = torch.randn(2, 16)
    S = tv.reshape(-1, 16)
    for mode, ridge in (("diag", 1e-2), ("full", 1e-2)):
        got = _depth_read(tv, tq, 1e-6, whiten=mode, ridge=ridge)
        if mode == "diag":
            scale = torch.rsqrt(S.pow(2).mean(dim=0) + ridge)
            v_s, q_s = tv * scale, tq * scale
            expect = (v_s * q_s.unsqueeze(1)).sum(-1) * torch.rsqrt(v_s.square().mean(-1) + 1e-6)
        else:
            cov = (S.transpose(0, 1) @ S) / S.shape[0] + ridge * torch.eye(16)
            evals, evecs = torch.linalg.eigh(cov)
            W = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)
            v_s, q_s = tv @ W, tq @ W
            expect = (v_s * q_s.unsqueeze(1)).sum(-1) * torch.rsqrt(v_s.square().mean(-1) + 1e-6)
        expect = (expect.softmax(-1).unsqueeze(-1) * tv).sum(1)
        results.append(check(f"whiten='{mode}' matches the analytic form", bool(torch.allclose(got, expect, atol=1e-5)), f"max diff = {(got - expect).abs().max():.3e}"))

    # ---------------- T10: address source ---------------------------------
    print("\nT10 address direction switch (pattern separation)")
    d10 = torch.randn(tokens, hidden)
    pk = torch.randn(tokens, hidden)
    for addr in ("delta", "state"):
        mod = _module(attn_res_gate_param="deviation", attn_res_update="objective", attn_res_address=addr)
        kp = mod.k_proj if isinstance(mod.k_proj, nn.Parameter) else mod.k_proj[0].weight
        clears = []
        with torch.no_grad():
            mod.erase_scale.fill_(1e4)                            # lambda -> inf: hard projection
            for prefix_i in (pk, pk + 3.0 * torch.randn_like(pk)):  # the address must not depend on this
                up, _ = mod.update(prefix_i, d10)
                state_i = mod.norm(prefix_i + d10)
                k_delta = F.normalize(F.linear(d10, kp), dim=-1)
                k_state = F.normalize(F.linear(state_i, kp), dim=-1)
                proj_delta = float((k_delta * up).sum(-1).abs().max())
                proj_state = float((k_state * up).sum(-1).abs().max())
                clears.append("delta" if proj_delta < proj_state else "state")
        results.append(check(f"address='{addr}' clears the {addr} direction", set(clears) == {addr}, f"cleared {clears} (both prefixes)"))

    # ---------------- T11: premises are reported --------------------------
    print("\nT11 statistical premises are measured, not assumed")
    model = Qwen3GDARForCausalLM(Qwen3GDARConfig(**{**SMALL, "attn_res_read_null": True, "attn_res_read_heads": 4}))
    ids = torch.randint(0, SMALL["vocab_size"], (2, 16))
    with torch.no_grad():
        out = model(input_ids=ids, return_attn_res_stats=True)
    st = [x for x in out.attn_res_stats if "key_cond" in x]
    results.append(check("premises present in stats", len(st) > 0, f"{len(st)} entries with key_cond/null_mass/head_corr"))
    if st:
        results.append(check("key_cond >> 1 (whitening premise)", st[0]["key_cond"] > 10, f"key_cond = {st[0]['key_cond']:.3e}"))
        results.append(check("null_mass > 0 (abstention premise)", st[0]["null_mass"] > 0, f"null_mass = {st[0]['null_mass']:.4f}"))
        results.append(check("head_corr reported", "head_corr" in st[0], f"head_corr = {st[0]['head_corr']:.4f}"))

    # ---------------- T6: theory preset runs ------------------------------
    print("\nT6  theory preset end-to-end (forward + backward)")
    for name, kw in (
        ("shensi update", dict(attn_res_gate_param="deviation")),
        ("objective update", dict(attn_res_gate_param="deviation", attn_res_update="objective")),
        ("objective + ladder", dict(attn_res_gate_param="deviation", attn_res_update="objective", attn_res_decay_ladder=16)),
        (
            "objective + ladder + lowrank",
            dict(
                attn_res_gate_param="deviation",
                attn_res_update="objective",
                attn_res_decay_ladder=16,
                attn_res_gate_rank=16,
                attn_res_q_rank=16,
                attn_res_k_rank=16,
            ),
        ),
    ):
        try:
            model = Qwen3GDARForCausalLM(Qwen3GDARConfig(**SMALL, **kw))
            model.train()
            ids = torch.randint(0, SMALL["vocab_size"], (2, 16))
            out = model(input_ids=ids, labels=ids, return_attn_res_stats=True)
            out.loss.backward()
            g = [s for s in out.attn_res_stats if "gate_decay" in s][0]
            results.append(
                check(
                    name,
                    True,
                    f"loss={out.loss.item():.3f} gates(d/e/w)=({g['gate_decay']:.6f}, {g['gate_erase']:.2e}, {g['gate_write']:.6f})",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(check(name, False, f"{type(exc).__name__}: {str(exc)[:80]}"))

    # ---------------- T12: training stability of the full preset ----------
    print("\nT12 training stability under AdamW (guards the lambda > -1 pole)")
    try:
        model = Qwen3GDARForCausalLM(Qwen3GDARConfig(**{**SMALL, **Qwen3GDARConfig.theory_preset()}))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, SMALL["vocab_size"], (2, 16))
        losses, bad = [], 0
        for _ in range(5):
            opt.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            losses.append(loss.item())
            bad += sum(1 for _, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
            opt.step()
        results.append(check("loss stays finite over 5 steps", all(l == l for l in losses), f"losses = {[round(l, 2) for l in losses]}"))
        results.append(check("no non-finite gradients", bad == 0, f"{bad} offending tensors"))
        results.append(check("loss decreases", losses[-1] < losses[0], f"{losses[0]:.2f} -> {losses[-1]:.2f}"))
    except Exception as exc:  # noqa: BLE001
        results.append(check("training stability", False, f"{type(exc).__name__}: {str(exc)[:80]}"))

    # ---------------- T13: the lower-bound guarantee -----------------------
    print("\nT13 lower-bound guarantee: identity slice + rollback (train/guarantee.py)")
    from train.guarantee import LowerBoundGuard, force_identity_gates, identity_slice_loss, rollback_to_identity

    model = Qwen3GDARForCausalLM(Qwen3GDARConfig(**{**SMALL, **Qwen3GDARConfig.theory_preset()}))
    ids = torch.randint(0, SMALL["vocab_size"], (2, 16))
    with torch.no_grad():
        base_loss = float(model(input_ids=ids, labels=ids).loss)
        slice_loss = identity_slice_loss(model, ids)
    results.append(check("slice loss == initial loss (GDAR(0) is DAR)", abs(slice_loss - base_loss) < 1e-5, f"slice={slice_loss:.6f} init={base_loss:.6f}"))

    # after moving the gates away, forcing identity must still land exactly on DAR
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, AttentionResidual) and hasattr(m, "erase_scale"):
                m.erase_scale.fill_(0.5)
                m.write_scale.fill_(-0.3)
        moved = float(model(input_ids=ids, labels=ids).loss)
        forced = identity_slice_loss(model, ids)
    results.append(check("identity slice unchanged after gates move", abs(forced - slice_loss) < 1e-5, f"forced={forced:.6f} vs slice={slice_loss:.6f}"))

    # rollback projects back onto the slice
    rollback_to_identity(model)
    with torch.no_grad():
        after = float(model(input_ids=ids, labels=ids).loss)
    results.append(check("rollback returns the model to the slice", abs(after - slice_loss) < 1e-5, f"after={after:.6f} slice={slice_loss:.6f}"))

    # The guard is checked mechanically: an untrained model sits at the uniform
    # entropy floor (loss ~ ln(vocab)), so no gate setting can make it measurably
    # worse -- the scientific question ("does a *trained* GDAR beat a trained DAR")
    # is what train/compare.py measures, not this test.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, AttentionResidual) and hasattr(m, "decay_scale"):
                m.decay_scale.fill_(10.0)
    guard = LowerBoundGuard(model, every=1, delta=0.0)
    rec = guard.maybe_step(1, ids)
    with torch.no_grad():
        current = float(model(input_ids=ids, labels=ids).loss)
    results.append(check("guard measures the gap correctly", abs(rec["gap"] - (rec["loss"] - rec["dar_slice"])) < 1e-6, f"gap={rec['gap']:+.5f} = current-slice")) 
    results.append(check("no rollback while gap <= delta", not rec["rolled_back"], f"gap={rec['gap']:+.5f} (untrained model is at the entropy floor)"))

    guard_forced = LowerBoundGuard(model, every=1, delta=-1.0)  # force the trigger path
    rec2 = guard_forced.maybe_step(1, ids)
    with torch.no_grad():
        repaired = float(model(input_ids=ids, labels=ids).loss)
        slice_now = identity_slice_loss(model, ids)
    results.append(check("trigger -> rollback -> exactly on the slice", bool(rec2["rolled_back"] and abs(repaired - slice_now) < 1e-6), f"repaired={repaired:.6f} slice={slice_now:.6f}"))

    # --- the decay-scale projection must not freeze the gate -----------------------------
    # `x.clamp(min=0)` has *zero* gradient exactly at the boundary, so a naive projection would
    # pin `decay_scale` at 0 forever (decay == 1 for every channel, the multi-timescale ladder
    # dead).  The projection is straight-through instead: clamp the forward value, pass the
    # gradient.  Both properties are asserted here so neither can regress silently.
    for mode in ("free", "project"):
        torch.manual_seed(0)
        cfg = Qwen3GDARConfig(
            vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
            attn_res_block_size=2,
            **Qwen3GDARConfig.theory_preset(attn_res_decay_positivity=mode),
        )
        mod = AttentionResidual(cfg).float().train()
        gates = mod._gates(torch.randn(8, cfg.hidden_size))
        sum(g.sum() for g in gates).backward()
        results.append(check(
            f"decay_positivity={mode}: the decay scale is not frozen at init",
            float(mod.decay_scale.grad) != 0.0,
            f"decay_scale.grad={float(mod.decay_scale.grad):+.4f}",
        ))
        with torch.no_grad():
            mod.decay_scale.fill_(-0.8)
            decay = mod._gates(torch.randn(8, cfg.hidden_size))[0]
        if mode == "project":
            results.append(check(
                "decay_positivity=project: decay <= 1 even with a negative scale",
                float(decay.max()) <= 1.0, f"max(decay)={float(decay.max()):.4f}",
            ))
        else:
            results.append(check(
                "decay_positivity=free: a negative scale can amplify (the failure mode)",
                float(decay.max()) > 1.0, f"max(decay)={float(decay.max()):.4f}",
            ))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
