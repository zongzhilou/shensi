# Stage 0.1: Pretraining (PT-1 stable → PT-2 decay)

Two-phase pretraining: PT-1 holds the learning rate constant to build language ability and give a
clean stability reading (the *stable* phase of a WSD schedule); PT-2 anneals on a high-quality
subset (the *decay* phase). PT-2 starts from PT-1's checkpoint. Every model algorithm
(`--model-algo`) runs this same recipe — that is what makes the architecture comparison clean.

## Overview

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Blends the Ultra corpora and tokenizes to Megatron bin/idx |
| `train.py` | Megatron-Core pretraining; `--model-algo` selects the connection (default: GDAR paper main) |
| `test_train.py` | Integration test: tiny geometry, 5 steps, judged from the log |
| `config/` | Profile configs (`default`, `decay`, `debug`, `tiny`, `perf`, `geoms/*`, `ablations/*`) |

> **Early stopping is on by default**: `train_iters` is effectively unlimited and the watchdog
> (metric `lm loss value`) ends the run when the loss plateaus. The stop counts as success;
> disable with `--no-early-stop`, adjust with `--early-stop N`.

## Profiles

| Profile | What it is |
|---------|------------|
| `default` | PT-1 stable: 9B tokens (90%), seq 2048, LR 6e-4 **constant**, 2% warmup, general blend |
| `decay` | PT-2 decay: 1B tokens (10%), seq 2048, cosine 6e-4 → 6e-5, `decay.json` high-quality blend |
| `debug` | Tiny geometry (4 layers / 256 hidden / seq 512) on real bin/idx — link verification |
| `tiny` | Mock smoke profile (used by `--smoke`) |
| `perf` | Throughput profile: TransformerEngine backbone + the three verified fusions |
| `geoms/qwen3_{1p7b,4b,8b,14b,30b_a3b}.yaml` | Scale ladder. Shared: `--profile geoms/*` resolves from Mid / SFT too (one copy, under this stage's `config/`). The 0.6B geometry is `default` itself (28 layers / 1024 hidden) |
| `ablations/a1a_*` … `ablations/e3_*` | Design-matrix and gate-structure rows (carry their own spec) |
| `minicpm5_2b.yaml` | The released MiniCPM5-2B geometry (back-derived; alignment reference) |

## Data Preparation

```bash
python data_prep.py --prepare                        # default blend
python data_prep.py --prepare --blend decay.json     # PT-2 blend
```

Blend files live in `config/data_prep/`. Output is Megatron bin/idx plus a `blend.json` (per-split
paths) under `$SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_pretrain/`; `train.py` picks it up
automatically.

## Training

```bash
python data_prep.py --prepare
python train.py --tokens 9e9                          # PT-1 (tokens -> train_iters)
python data_prep.py --prepare --blend decay.json
python train.py --profile decay --tokens 1e9 --load <PT-1 ckpt>   # PT-2
python train.py --model-algo base --tokens 9e9        # control arm: same recipe, other model
```

| Flag | Description |
|------|-------------|
| `--profile <name>` | Profile config (default `default`; `decay`, `debug`, `perf`, `geoms/*`, `ablations/*`) |
| `--model-algo <name>` | Connection / baseline (default `qwen3_gdar_paper`; `base` = plain Qwen3) |
| `--tokens <n>` | Token budget — converted to `train_iters` from global batch × seq length |
| `--load <ckpt>` | Continue from a checkpoint (PT-2, or any later stage) |
| `--smoke` / `--dry-run` | Synthetic tiny run / print the command only |
| `--no-early-stop`, `--early-stop N` | Watchdog off / patience override |

## Verification

| Check | Command | Criterion |
|-------|---------|-----------|
| Integration test | `python test_train.py` | tiny, 5 steps: rc=0, last iteration reached, `[after training is done]`, no tracebacks |
| Offline checks | `python -m shensi.recipes.paper.gated_delta_attn_res.train.checks` | identity 27/27 bit-exact, `max|Δ logit| = 0.000e+00`, parameter-cost table, gradient flow |
| Smoke | `python train.py --smoke` | same criteria as the integration test (mock data) |

Throughput: the default is `transformer_impl: local` (the bit-exactness baseline); cluster runs
switch to `--set train.model.transformer_impl=transformer_engine` (verified to run: TE attention /
MLP plus the three fusions in `perf.yaml`). With the local spec, GDAR costs ≈ 2.19× plain Qwen3 per
step (no fused connection kernel yet) — the fused kernel is a separate engineering item.

## Run the Full Paper Experiment (EXPERIMENT_MATRIX.md §2)

Architecture conclusions are carried by pretraining alone: every variant × the scale ladder, three
seeds at 0.6B, single seed elsewhere. Each command is `train.py --model-algo <arm>` — an arm name
is one row of the design matrix.

```bash
cd stage0_pretrain/stage1_pretrain

# 0) both blends
python data_prep.py --prepare
python data_prep.py --prepare --blend decay.json

# 1) 0.6B core matrix (3 seeds) — variants x one-knob rows
ARMS="base qwen3_ar_block4 qwen3_dar_block4 qwen3_denseformer qwen3_mudd qwen3_hc qwen3_mhc \
      qwen3_gdar_paper qwen3_gdar_theory qwen3_gdar_upstream qwen3_gdar_block2 qwen3_gdar_block8 \
      qwen3_gdar_r16 qwen3_gdar_noladder qwen3_gdar_no_output_route a1a_gate_prefix a1b_gate_delta \
      a3_decay_projected a4_lambda_free a6_reference a9_half_init a9_uniform_init"
for seed in 0 1 2; do for algo in $ARMS; do
  # 0.6B geometry is `default` (28 layers / 1024 hidden); decay continues from it
  python train.py --model-algo $algo --tokens 1e10 \
      --set experiment.seed=$seed --set experiment.exp_dir=$SHENSI_FS/shensi/runs/pt06/$algo-s$seed
  python train.py --profile decay --model-algo $algo --tokens 1e9 \
      --load $SHENSI_FS/shensi/runs/pt06/$algo-s$seed/ckpt --set experiment.seed=$seed
done; done

# 2) scale ladder (single seed): 1.7B / 4B / 8B / 14B + the 30B-A3B facade
for size in qwen3_1p7b qwen3_4b qwen3_8b qwen3_14b; do for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  python train.py --profile geoms/$size --model-algo $algo --tokens <budget>
done; done

# 3) mechanism curves: 220M / 1.04B (control depth + width curves)
# 4) evaluation: 0-shot / Chinese / controlled depth retrieval / RULER (>=8B) — see stage4_eval/
```

## Further Reading

- [Stage 0 overview](../README.md) — the 2 + 2 segment design and why
- [Mid-training](../stage2_midtrain/README.md) — what continues from PT-2
- [LIMITATIONS.md](../../LIMITATIONS.md) — early stopping evidence, fusion measurements
