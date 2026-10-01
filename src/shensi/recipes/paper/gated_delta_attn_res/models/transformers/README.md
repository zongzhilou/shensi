# Models: HuggingFace Reference Implementations

Plain-PyTorch reference implementations of the seven depth-connection variants, used for
*research, evaluation, conversion and rollout* — never for training (training always goes through
the Megatron-Core path in `models/megatron/`).

Each variant is faithful to its own upstream repository, and that faithfulness is checked
numerically: the official operator implementations are vendored under `upstream/` and compared
tensor-by-tensor against ours (`test_upstream_alignment.py`, 13/13).

## Overview

| File | Variant | Upstream |
|------|---------|----------|
| `modeling_qwen3_gdar.py` | GDAR — gated delta attention residual (the paper's model) | this work (Kimi-style depth read + gated delta write) |
| `modeling_qwen3_ar.py` | AR — attention residuals | Kimi Linear (`upstream/kimi_modeling_kimi_linear.py`) |
| `modeling_qwen3_dar.py` | DAR — depth attention residual | Kimi Linear |
| `modeling_qwen3_denseformer.py` | DenseFormer — depth-weighted average | `upstream/denseformer_*.py` |
| `modeling_qwen3_mudd.py` | MUDD — multiway dynamic dense | `upstream/muddformer_*.py` |
| `modeling_qwen3_hc.py` / `modeling_qwen3_mhc.py` | HC / mHC — hyper-connections | shensi branch / paper |
| `guarantee.py` | The identity guarantee helpers shared by the variants | — |

Each variant ships a config class (`configuration_qwen3_*.py`) that carries the connection knobs
(`attn_res_*`), registers `model_type` (e.g. `qwen3_gdar`) and declares `auto_map`, so a checkpoint
saved with `save_pretrained` is loadable with `trust_remote_code=True` in a fresh process — which
is exactly how verl and vLLM consume these models.

## The GDAR Operator (Reference Semantics)

At every sublayer the residual stream is read and written through a depth state:

- **Write** — the sublayer output is written into the state with a gated delta rule: a learned
  per-channel decay (`decay_tau` ladder, positive by construction under
  `attn_res_decay_positivity="project"`), an erase gate and a write gate, all driven by the state.
- **Read** — the sublayer input is a whitened multi-head read over the state's snapshots
  (λ-clamped closed-form update, `Softmax¬1`, learned null source).
- **Identity** — with the paper initialization the connection is exactly the plain residual
  stream: `GDAR(0) == Qwen3` bit-exactly (`modeling_qwen3_gdar.py` docstring carries the proof
  sketch and the measured deviations for the non-identity forms).

Low-rank projections (`attn_res_{gate,q,k}_rank`) make the operator affordable: full rank costs
~45% of an 8B model and `k_proj` alone ~15%.

## Quick Start

```bash
cd models/transformers

# the HF unit suites
python test_theory.py                 # 58/58 — operator algebra, identity, gate semantics
python test_ablation_switches.py      # 64/64 — every knob's effect
python test_autoclass.py              # 42/42 — config/auto_map/round-trip serialization
python smoke_test.py                  # forward+backward over 8 configurations, routing stats

# alignment against the vendored upstream implementations
python test_upstream_alignment.py     # 13/13 (AR / MUDD / DenseFormer / GDAR-per-head bit-exact)
```

## Verification

| Check | Result |
|-------|--------|
| Alignment vs upstream | AR vs Kimi-K3 operator **bit-exact**, MUDD vs MUDDFormer block **bit-exact**, DenseFormer vs official DWAModules **bit-exact**, GDAR vs the shensi branch under `per_head` **bit-exact** (13/13 checks; known deltas listed per variant) |
| Theory suite | 58/58 |
| Ablation switches | 64/64 |
| Autoclass / serialization | 42/42 |
| Forward + backward smoke | 8 configurations, routing statistics (`sharpness`, `entropy`, `n_sources`, gate values) |

`upstream/PROVENANCE.md` records the sha256 of every vendored file; they are kept byte-identical
(the repository's formatter excludes them).

## Not Ported

- shensi's hyper-connection multi-stream (`ShensiHyperConnection`) — a different mechanism.
- shensi's `block_write_layer` / `attn_res_block_layer_types` — this tree uses the DAR-consistent
  block-source semantics (snapshot differences) so GDAR-vs-DAR differ only in the gates.
- DAR's V-stream decoupled attention (`Qwen3AttnResAttention`, its `delta_v` variant).
- MoE / KDA attention — this tree is the dense Qwen3 backbone only.

## Further Reading

- [Recipe README](../../README.md) — where these implementations are used (rollout, conversion, eval)
- [vLLM rollout](../vllm/README.md) — serving these models with the engine
- [LIMITATIONS.md](../../LIMITATIONS.md) — A6 (per-head whitening), A12 (formatting/provenance)
