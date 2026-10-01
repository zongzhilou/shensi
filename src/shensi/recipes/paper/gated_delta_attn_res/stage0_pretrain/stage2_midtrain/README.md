# Stage 0.2: Mid-training (Mid-1 capability → Mid-2 long documents)

Continue from PT-2 in two phases: Mid-1 raises the code/math share and the sequence length to 4K
(capability strengthening); Mid-2 switches to Ultra-FineWeb-L3 long documents with 16K sequences
and a further LR decay (distribution adaptation). Budgets, blends and the reasoning behind the
2 + 2 split are in the [stage overview](../README.md).

## Overview

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Blends the mid-training corpora (with the UltraData-Code tiers) into bin/idx |
| `train.py` | Megatron-Core mid-training; `--model-algo` selects the connection |
| `test_train.py` | Integration test (tiny geometry, 5 steps, judged from the log) |

> **Early stopping is on by default**: iterations are effectively unlimited; the watchdog
> (metric `lm loss value`) ends the run at plateau and the stop counts as success.

## Profiles

| Profile | What it is |
|---------|------------|
| `default` | Mid-1 capability: 0.5B tokens (5%), seq 4096, LR 6e-5 constant, Code 40% + Math 30% + UltraX 30% |
| `mid2` | Mid-2 distribution: 0.3B tokens (3%), seq 16384, cosine 6e-5 → 3e-5, L3 70% + UltraX 30% |
| `debug` | Tiny geometry (seq 512) on real bin/idx — link verification |
| `tiny` | Mock smoke profile |
| `geoms/*` | Shared scale-ladder profiles (`--profile geoms/qwen3_8b`); one copy lives under `stage1_pretrain/config/` |

## Quick Start

```bash
cd stage0_pretrain/stage2_midtrain
python data_prep.py --prepare                        # Mid-1 blend
python train.py --tokens 5e8 --load <PT-2 ckpt>
python data_prep.py --prepare --blend mid2.json      # Mid-2 long-document blend
python train.py --profile mid2 --tokens 3e8 --load <Mid-1 ckpt>
```

| Flag | Description |
|------|-------------|
| `--profile <name>` | `default` (Mid-1), `mid2`, `debug`, `tiny` |
| `--model-algo <name>` | Same registry as every stage (default `qwen3_gdar_paper`) |
| `--tokens <n>` | Token budget → `train_iters` |
| `--load <ckpt>` | Start from PT-2 (Mid-1) or Mid-1 (Mid-2) |

## Verification

| Check | Command | Criterion |
|-------|---------|-----------|
| Integration test | `python test_train.py` | tiny, 5 steps: rc=0, last iteration reached, `[after training is done]`, no tracebacks |
| Smoke | `python train.py --smoke` | same criteria (mock data) |

## Run the Full Paper Experiment (EXPERIMENT_MATRIX.md §3)

Mid-training serves the 8B four-variant set and the 30B-A3B facade; the longer sequences are what
make the RULER 16K / 32K buckets meaningful.

```bash
cd stage0_pretrain/stage2_midtrain

# blends
python data_prep.py --prepare                     # Mid-1 (includes the UltraData-Code tiers)
python data_prep.py --prepare --blend mid2.json   # Mid-2 long documents

for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  # 8B
  python train.py --profile geoms/qwen3_8b --model-algo $algo --tokens 5e9 \
      --load <PT-2(8B, $algo) ckpt> --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo
  python train.py --profile geoms/qwen3_8b --model-algo $algo --tokens 3e9 \
      --set train.model.seq_length=16384 \
      --load $SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo/ckpt \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid2/$algo
  # 30B-A3B facade: same commands with --profile geoms/qwen3_30b_a3b
  # 32K variant:      add --set train.model.seq_length=32768
done

# evaluation: same suite as PT (see stage1_pretrain), RULER adds the 16K / 32K buckets
```

`geoms/*` profiles are shared with pretraining (same geometry); the Mid-2 32K variant is a single
`--set train.model.seq_length=32768`.

## Maintenance Notes

- **2026-10-01, one unreproduced CUDA illegal-memory-access**: a `test_train.py` run (tiny geometry,
  L=8, block B=4, 5 steps) hit `CUDA error: an illegal memory access` in the step-2 backward pass;
  five subsequent runs (same run, plus B=4/B=2/B=1 shapes, 3 steps each) all finished cleanly.
  Recorded as an unreproduced flake, not as fixed. To localise if it returns:

```bash
cd stage0_pretrain/stage2_midtrain
CUDA_LAUNCH_BLOCKING=1 python train.py --profile debug \
    --set train.model.train_iters=2 --set train.model.eval_iters=0
```

## Further Reading

- [Stage 0 overview](../README.md) — segment design, budgets, LR rationale
- [Pretraining](../stage1_pretrain/README.md) — the phase Mid-1 continues from
- [LIMITATIONS.md](../../LIMITATIONS.md) — A7 records the CUDA flake and the localisation command
