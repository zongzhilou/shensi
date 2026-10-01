# Stage 3: On-Policy Distillation (four teachers → one release model)

Distill the four domain teachers back into a **single release model**. The student is the SFT
base; it rolls out on each domain's prompts, the teachers score those rollouts token by token, and
the student learns from them — on *its own* tokens, which is what makes on-policy distillation
stronger than offline distillation.

The pipeline is **Megatron-Core native** in all three steps (no vLLM dependency): rollout data
prep → teacher scoring with mcore's logits saver → KD training with mcore's cached-logits loss.
The loss direction can be **reverse KL** (`KL(student‖teacher)`, the MiniCPM5 OPD recipe) or
forward KL (mcore's default).

## Overview

| Step | What happens | Implementation |
|------|--------------|----------------|
| ① rollout | The student samples on each domain's prompts | `data_prep.py --prepare --blend <domain>.json` |
| ② score | Frozen teacher forward, top-K/top-P log-probs to disk | `python score.py --load <teacher> --out <cache>` → mcore `--logits-save-dir --logits-save-top-k --freeze-all-layers --async-save` |
| ③ distill | The student trains on the rollout sequences with a KD loss | `python train.py --teacher-cache <cache>` → mcore `--logits-load-dir` (`kd_loss_alpha` mixes LM loss) |

## Configuration

| Knob | Value / meaning |
|------|-----------------|
| Student | SFT-3 (agent phase) checkpoint |
| LR | 1e-5 cosine → 1e-6, 1% warmup |
| Sequence | 8192 |
| Budget | 0.5B tokens per round; 2–4 rounds |
| KD weight | `logits_load_kd_loss_alpha` (1.0 = pure KD, lower mixes LM loss against forgetting) |
| Loss direction | forward KL by default; `--set train.model.logits_load_reverse_kl=true` switches to reverse KL |
| Multi-teacher | Route by data domain (each domain trains on its own teacher's cache), or merge caches (≈ averaging in log-prob space) |

> **Early stopping is on by default** (metric `lm loss value`); the stop counts as success.

## Quick Start

```bash
cd stage3_opd
python data_prep.py --prepare --blend math.json                        # ① student rollout -> bin/idx
python score.py --load <math teacher ckpt> --out $CACHE/math --top-k 64  # ② teacher scoring
python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/math   # ③ distillation
python train.py --dry-run                                              # print the command
python test_train.py                                                   # preflight
```

## Publish the Release Model

The release model is an mcore checkpoint; evaluation and serving read HF directories, so publish
it first (geometry comes from the checkpoint's own `run_config.yaml`, connection knobs from the
run's `config.yaml`):

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd \
    --out  $SHENSI_FS/shensi/models/gdar-release-hf
```

## Forward vs Reverse KL

| Direction | Where | Note |
|-----------|-------|------|
| Forward KL (mcore default) | `--logits-load-dir` path, unchanged | Teacher mass is covered; tends to be mode-covering |
| **Reverse KL** (`KL(student‖teacher)`) | `train/reverse_kl.py`, enabled with `--logits-load-reverse-kl` | The MiniCPM5 OPD recipe; mode-seeking. Replaces `topk_kl_div` with a same-signature implementation, so the cache/top-k/TP pipeline is unchanged |

The unit test `train/test_reverse_kl.py` checks both directions against the analytic solution
(2.7e-07) and that they are genuinely different (Δ = 2.8).

## Verification

| Check | Command | Criterion |
|-------|---------|-----------|
| Preflight | `python test_train.py` | tiny geometry, 5 steps: rc=0, `[after training is done]`, no tracebacks |
| Reverse KL | `python -m shensi.recipes.paper.gated_delta_attn_res.train.test_reverse_kl` | analytic match, patch idempotent and installed |

## Run the Full Paper Experiment (EXPERIMENT_MATRIX.md §5 / RECIPE §4)

```bash
# per domain (math / code / agent / writing), then the merged release model
cd stage3_opd
for dom in math code agent writing; do
  python data_prep.py --prepare --blend $dom.json
  python score.py --load <$dom teacher ckpt> --out $CACHE/$dom --top-k 64
  python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/$dom \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/opd_$dom
done
# reverse-KL variant (MiniCPM5's OPD direction) on the same caches
python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/math \
    --set train.model.logits_load_reverse_kl=true --set experiment.exp_dir=$SHENSI_FS/shensi/runs/opd_rkl

# publish, then evaluate (stage4_eval): the release model must not regress on the SFT suite
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt <OPD ckpt> --out $HF/gdar-release
```

## Further Reading

- [RL teachers](../stage2_rl/README.md) — where the teacher checkpoints come from
- [Publish + eval](../stage4_eval/README.md) — the HF directory and the T0 retrieval task
- [MINICPM5_ALIGNMENT.md](../MINICPM5_ALIGNMENT.md) — OPD alignment item (16 experts, reverse KL, prompt reuse)
