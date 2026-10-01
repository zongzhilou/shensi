# Stage 1: Supervised Fine-Tuning (SFT-1 → SFT-2 → SFT-3, 400B tokens)

Instruction tuning of the mid-training checkpoint on the UltraData SFT releases: **SFT-1**
deep-thinking (`UltraData-SFT-2605`), **SFT-2** hybrid-thinking, **SFT-3** agent
(`UltraData-SFT-Agent-2609`). The flagship 400B-token budget is 200B deep-thinking + 200B
hybrid-thinking, then the agent phase. The comparison arms run the same data, token budget and LR
schedule, so only the model algorithm differs.

## Overview

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Normalizes UltraData SFT parquet/jsonl into `{"messages": [...]}` jsonl (98/2 train/val) |
| `train.py` | Megatron-Core SFT (`--sft`, unpacked), `train.data.data_path` injected automatically |
| `test_train.py` | Integration test: synthetic messages jsonl, tiny geometry, 5 steps |

> **Early stopping is on by default**: iterations are effectively unlimited; the watchdog
> (metric `lm loss value`) ends the run at plateau and the stop counts as success.

## Profiles

| Profile | Phase | Data | Seq | LR | Budget (0.6B) |
|---------|-------|------|-----|----|---------------|
| `default` | SFT-1 deep-thinking | UltraData-SFT-2605 | 8192 (unpacked) | 2e-5 cosine → 2e-6, 1% warmup | ~2B tokens (flagship: 200B) |
| `sft2_hybrid` | SFT-2 hybrid-thinking | 2605 hybrid subset | 8192 | same | ~2B (flagship: 200B) |
| `sft3_agent` | SFT-3 agent | UltraData-SFT-Agent-2609 | 8192 | same | ~0.2B (flagship: 20B) |
| `debug` / `tiny` | link verification / mock smoke | — | 2048 / tiny | — | 5 steps |

`geoms/*` (the scale ladder) is shared across PT / Mid / SFT: `--profile geoms/qwen3_30b_a3b`
resolves to the single copy under `stage0_pretrain/stage1_pretrain/config/geoms/`.

## Data Path: Unpacked, Chat-Templated

- `--sft` runs the **unpacked** path (`ShensiSFTDataset`: one conversation per sample, right
  padding). The recipe's attention is the local implementation, and
  `DotProductAttention` asserts `packed_seq_params is None` — THD packing would require the TE
  attention path. This matches the shensi recipe's trade-off.
- The loss mask is produced by `SFTTokenizer` from the **Qwen3 chat template** shipped with the
  tokenizer (`sft_tokenizer_prompt_format: default`).
- `data_prep.py` keeps `reasoning_content` (thinking) verbatim in each message.

## Quick Start

```bash
cd stage1_sft
python data_prep.py --prepare --limit 1000            # debug-scale jsonl
python train.py --smoke                               # synthetic messages jsonl + tiny geometry, 5 steps
python data_prep.py --prepare --blend default.json    # SFT-1
python train.py --tokens 2e9 --load <Mid-2 ckpt>      # SFT-1
python data_prep.py --prepare --blend hybrid.json     # SFT-2
python train.py --profile sft2_hybrid --load <SFT-1 ckpt>
python data_prep.py --prepare --blend agent.json      # SFT-3
python train.py --profile sft3_agent --load <SFT-2 ckpt> \
    --data-jsonl <sft_train_agent.jsonl>
```

| Flag | Description |
|------|-------------|
| `--profile <name>` | `default`, `sft2_hybrid`, `sft3_agent`, `geoms/*`, `debug`, `tiny` |
| `--model-algo <name>` | Same registry as every stage (default `qwen3_gdar_paper`) |
| `--tokens <n>` | Token budget → `train_iters` |
| `--load <ckpt>` / `--data-jsonl <file>` | Continuation checkpoint / explicit messages jsonl |
| `--smoke` / `--dry-run` | Synthetic tiny run / print the command |

## Verification

| Check | Command | Criterion |
|-------|---------|-----------|
| Integration test | `python test_train.py` | generates its own synthetic jsonl if no corpus is present; 5 steps: rc=0, `[after training is done]`, no tracebacks |
| Smoke | `python train.py --smoke` | same criteria |

## Run the Full Paper Experiment (EXPERIMENT_MATRIX.md §4: the flagship pair)

SFT serves the **flagship pair** only — `qwen3_gdar_main` and its plain-residual twin `base`, on
the 30B-A3B geometry, same data / tokens / LR by construction.

```bash
cd stage1_sft
python data_prep.py --prepare --blend default.json     # SFT-1: UltraData-SFT-2605 deep-thinking
python data_prep.py --prepare --blend hybrid.json      # SFT-2: hybrid-thinking (200B + 200B = 400B)
python data_prep.py --prepare --blend agent.json       # SFT-3: UltraData-SFT-Agent-2609

for algo in qwen3_gdar_main base; do
  D=$SHENSI_FS/shensi/runs/gdar_30b_sft/$algo
  python train.py --profile geoms/qwen3_30b_a3b --model-algo $algo --tokens 2e11 \
      --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train.jsonl \
      --load <Mid-2(30B, $algo) ckpt> --set experiment.exp_dir=$D-sft1
  python train.py --profile sft2_hybrid --model-algo $algo --tokens 2e11 \
      --load $D-sft1/ckpt --set experiment.exp_dir=$D-sft2
  python train.py --profile sft3_agent --model-algo $algo --tokens 2e10 \
      --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train_agent.jsonl \
      --load $D-sft2/ckpt --set experiment.exp_dir=$D-sft3
done

# evaluation (flagship pair only): mmlu(5-shot) / gsm8k(8-shot) / MATH / HumanEval / MBPP / CMMLU / C-Eval
```

Artifacts land in per-phase directories (`*-sft1 / *-sft2 / *-sft3`), matching the `sft1` / `sft2`
rows of `EXPERIMENT_MATRIX.json`. The RL stage consumes **SFT-2** (or SFT-3 for the agent arm)
after publishing it as an HF directory — see the [RL README](../stage2_rl/README.md).

## Further Reading

- [Recipe README](../README.md) — pipeline overview and `--model-algo`
- [Mid-training](../stage0_pretrain/stage2_midtrain/README.md) — the phase SFT continues from
- [MINICPM5_ALIGNMENT.md](../MINICPM5_ALIGNMENT.md) — the 400B deep-thinking alignment
