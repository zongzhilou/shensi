# Stage 4: Evaluation (after OPD)

Evaluate the **published release model**. The T0 task is **controlled depth retrieval**: every
question is constructed with a known answer position, a declared chance level (4-way multiple
choice = 25%) and a controllable depth (`K` distinct keys written at `L` context length), so the
result can be compared against a random baseline and against the plain-Qwen3 twin.

The stage runs **after** [OPD](../stage3_opd/) and reads a HuggingFace directory — publish the
release checkpoint first.

## Overview

| Component | Description |
|-----------|-------------|
| `make_depth_retrieval.py` | Generator: writes `n` questions at given `K` × `L` grids |
| `run_depth_retrieval.py` | Scorer: per-bucket accuracy, Wilson 95% intervals, chance comparison, position-bias check, optional label shuffle control |
| `test_train.py` | Preflight + smoke: generate 40 questions and score a tiny checkpoint |

## Publish the Model First

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd \
    --out  $SHENSI_FS/shensi/models/gdar-release-hf
```

The exporter reads geometry from the checkpoint's own `run_config.yaml`, the connection knobs from
that run's `config.yaml` (or `--model-algo`), pre-checks tensor shapes, and writes an HF directory
whose `config.json` carries `model_type: qwen3_gdar`, the `auto_map` entries, the two remote-code
`.py` files and the 22 `attn_res_*` knobs — loadable with `trust_remote_code=True`. Earlier stage
checkpoints work the same way (`--ckpt` at their run directory), but the number that gets reported
is the OPD release model.

## Scoring Protocol

1. **Chance / random baseline** — `--chance 0.25` (4-way); every bucket reports whether it is
   above chance.
2. **Interval** — Wilson 95% per bucket, not a raw accuracy point estimate.
3. **Stratification** — `K ∈ {1,2,4,8}` × `L ∈ {1024,2048,4096}` (`--ks` / `--lengths`).
4. **Position bias** — gold/pred position distributions; plus `--control-shuffle-labels` as a
   negative control. Buckets below chance are reported as-is (the `usable` gate), never spun as
   "worse than random".

## Quick Start

```bash
cd stage4_eval
# ① generate (paper scale: >= 1000 questions)
python make_depth_retrieval.py --out $SHENSI_FS/shensi/data/gated_delta_attn_res/eval/dr1000.jsonl \
    --n 1000 --lengths 1024,2048,4096 --ks 1,2,4,8 --seed 42
# ② score a published model
python run_depth_retrieval.py --model $HF/gdar-release --data <dr1000.jsonl> --device cuda \
    --chance 0.25 --out-json <score.json>
# ③ preflight / smoke (generate 40 questions, score a tiny checkpoint)
python test_train.py
```

## Verification

| Check | Command | Result |
|-------|---------|--------|
| Preflight + smoke | `python test_train.py` | imports, tiny checkpoint, 40 questions generated, scored in ~3 s |
| End-to-end chain | `train/export_hf.py` then `run_depth_retrieval.py` | HF directory loads with `trust_remote_code`; `score.json` reports `chance = 0.25` and the `usable` gate |

## Run the Full Paper Experiment (EXPERIMENT_MATRIX.md §5)

```bash
# ① publish both arms of the flagship pair
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd      --out $HF/gdar-release
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd_base --out $HF/base-release

# ② T0 controlled depth retrieval
cd stage4_eval
python make_depth_retrieval.py --out $SHENSI_FS/shensi/data/gated_delta_attn_res/eval/dr1000.jsonl \
    --n 1000 --lengths 1024,2048,4096 --ks 1,2,4,8 --seed 42
python run_depth_retrieval.py --model $HF/gdar-release --data <dr1000.jsonl> --device cuda \
    --chance 0.25 --out-json <score_gdar.json>
python run_depth_retrieval.py --model $HF/base-release --data <dr1000.jsonl> --device cuda \
    --chance 0.25 --out-json <score_base.json>
python run_depth_retrieval.py --model $HF/gdar-release --data <dr1000.jsonl> --device cuda \
    --control-shuffle-labels --out-json <score_shuffled.json>          # negative control

# ③ general capability (lm-eval suite + Chinese) and long context (RULER, >= 8B, with oracle control)
#    scripts ship with the package (eval/run_lm_eval.py, eval/run_ruler.py); same wiring as T0.
```

Criteria: GDAR main must not be worse than its plain-Qwen3 twin on **T0** (the paper's main
claim); lm-eval / RULER must not drop by more than 1 point (including after OPD). Buckets below
chance are reported honestly through the `usable` gate.

## Not Wired Yet

lm-eval (HellaSwag / ARC / PIQA / … / CMMLU / C-Eval) and RULER (≥ 8B, with an oracle control) both
have runnable scripts in the package; they plug in at the same place as T0. Realistic retrieval
(SWDE / FDA / RAG settings) is the T1 plan and runs on the cluster.

## Further Reading

- [OPD](../stage3_opd/README.md) — produces the model being evaluated
- [Pretraining](../stage0_pretrain/stage1_pretrain/README.md) — the 0-shot / Chinese / RULER suite used earlier in the pipeline
- [LIMITATIONS.md](../LIMITATIONS.md) — A8 (stage port), A20 (the publish step)
