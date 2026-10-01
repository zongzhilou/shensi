# Stage 2: Reinforcement Learning (four domain teachers in parallel)

Train four domain teachers from the SFT checkpoint — **math**, **code**, **agent**, **writing** —
in parallel and independently. Each teacher is a *specialist*: they are never compared with each
other, they exist to be distilled back into one release model by [OPD](../stage3_opd/).

Training runs on **verl** (GRPO-family estimators) with a **Megatron-Core actor**; the GDAR
registration that lets verl build and serve our checkpoints lives in
[`gdar_bridge.py`](./gdar_bridge.py) and is exercised end-to-end by
[`test_gdar_bridge.py`](./test_gdar_bridge.py).

## Overview

| Component | Description |
|-----------|-------------|
| `stage2_{math,code,agent,writing}/` | One arm each: config, data prep, reward, launcher |
| `gdar_bridge.py` | Imports to register the seven variants with Megatron-Bridge (`VERL_USE_EXTERNAL_MODULES`) |
| `train/export_hf` | Publishes an SFT checkpoint as the HF directory that `model.path` points at |
| `harness_tool.py`, `config/tools/harness.yaml` | Tool/harness wiring for the agent arm |
| `test_train.py` | Preflight: config → verl CLI, reward module, verl import |
| `test_gdar_bridge.py` | End-to-end gate: dispatch, spec, weight load, HF parity, rollout-sync export |

## Model Algorithm Profiles

All four arms share six profiles (config files `config/<name>.yaml`, merged over `default.yaml`):

| Profile | What changes |
|---------|--------------|
| `default` | GRPO, KL-free, clip 0.2/0.28 (the baseline) |
| `dapo` | Decoupled clip + dynamic sampling + token-level loss |
| `drgrpo` | Drop the std normalization of advantages |
| `token_baseline` | Token-level optimal baseline estimator |
| `critic` | GAE + value model (the JustRL-II-style arm) |
| `fsdp` | HF/FSDP actor path (does not go through the mcore bridge) |

## Prerequisites

- An **HF directory** as the training start: publish the SFT checkpoint first
  (verl loads the model through `auto_map`, so an mcore checkpoint is not enough):

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_sft \
    --out  $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_sft/sft2_agent
```

- RL prompts: UltraData-RL-2609, sliced per domain by `data_prep.py --blend <domain>.json`.
- The GDAR bridge is loaded into every verl process by `VERL_USE_EXTERNAL_MODULES` (set by the
  launcher); without it, verl refuses the architecture loudly — it never builds the wrong model.

## Quick Start

```bash
cd stage2_rl/stage2_math
python data_prep.py --prepare             # prompts -> verl train/val parquet
python train.py --dry-run                 # print the verl command
python train.py                           # train (GRPO, LR 1e-6, clip 0.2/0.28)
python train.py --profile dapo            # one of the six algorithm profiles
```

| Flag | Description |
|------|-------------|
| `--profile <name>` | `default`, `dapo`, `drgrpo`, `token_baseline`, `critic`, `fsdp` |
| `--set k=v` | Any config override (`model.path`, `trainer.total_epochs`, ...) |
| `--dry-run` | Print the composed verl CLI |

> **Early stopping is on by default** for RL too: `total_epochs` is effectively unlimited and the
> watchdog watches `val/reward` (higher is better), ending the run when it plateaus.

## Verification

| Check | Command | Result |
|-------|---------|--------|
| Preflight | `python stage2_rl/test_train.py` | config → CLI mapping, rewards, verl import |
| Bridge gate | `python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.test_gdar_bridge` | 12/12: auto_map dispatch, layer spec = GDAR connection, weights load with no missing keys, HF parity `2.4e-07`, export (rollout sync) bit-identical |
| Profiles | `python train.py --profile <p> --dry-run` for all arms | 4 arms × 6 profiles = 24 dry-runs, each asserting `model.path` and the two provider overrides reach the CLI |

## Run the Full Paper Experiment (EXPERIMENT_MATRIX.md §5: the facade teachers)

RL trains the domain teachers for the **30B-A3B facade** only, and explicitly does not carry
architecture conclusions. The rollout driver is chosen from the checkpoint's model type.

```bash
# both arms: each trains its own four teachers from its own SFT-3 checkpoint
for arm in qwen3_gdar_main base; do
  for dir in stage2_math stage2_code stage2_agent stage2_writing; do
    cd stage2_rl/$dir
    python data_prep.py --prepare --limit 200000      # UltraData-RL-2609, sliced by domain
    python train.py                                   # model.path -> this arm's SFT-3 HF dir
  done
done
```

Each teacher is evaluated with the SFT suite plus its domain metric (RUN_EXPERIMENTS.md §5); the
teacher checkpoints are the inputs of `stage3_opd` (one per domain).

## Further Reading

- [OPD](../stage3_opd/README.md) — where the four teachers go
- [Publish + eval](../stage4_eval/README.md) — the HF directory and the T0 retrieval task
- [LIMITATIONS.md](../LIMITATIONS.md) — A9 (profiles), A13 (bridge), A15 (profile merging)
