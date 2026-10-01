# Models: vLLM Rollout Side

Serve the seven variants with vLLM through the engine's own extension points
(`ModelRegistry.register_model` + the `vllm.general_plugins` entry point) — **no file shipped with
vLLM is modified**. The directory is deliberately self-contained: importing
`models.vllm.register_model` pulls in neither Megatron, nor TransformerEngine, nor verl, because
an engine worker process only needs the registration and the bridge (a check enforces this).

## Overview

| File | Description |
|------|-------------|
| `variants.py` | How the engine sees each variant: `architectures[0]` (dispatch key), `model_type`, the two remote-code file names, the non-default knobs of the tiny smoke checkpoint, `SHAPES` (`tiny` = 2×64, `0.6b` = 28×1024/16 heads) |
| `tiny_checkpoint.py` | Builds a *random-weight* self-describing checkpoint (connection on, `auto_map` + both `.py` files copied next to the weights) |
| `register_model.py` | Registers the 7 architectures; `install` also writes the `vllm.general_plugins` entry point into the venv |
| `vllm_bridge.py` | The registered implementation: vLLM's Transformers backend **with the connection subtrees held out of the engine's module rewrite** |
| `sitecustomize.py` | Makes the registration reach **EngineCore worker processes** (opt-in via `ROLLOUT_PLUGIN_AUTOLOAD=1`) |
| `smoke_generate.py` | End-to-end: checkpoint → real `vllm.LLM` → 16 tokens, optionally diffed against the plain-transformers reference (longest common prefix) |
| `batch_generate.py` | Continuous batching: many prompts at once vs one-by-one, with the engine's running/waiting counters |
| `mudd_fused_qkv_repro.py` | Minimal CPU reproduction of the QKV-fusion failure `mudd` triggers, then proof that fused/unfused agree bit-exactly |
| `check_decode_state.py` | Measures whether depth state is lost across decode steps (it is not: every depth operation is pointwise along the token axis, so incremental decoding equals a full recompute) |
| `_paths.py` | `SRC` (makes `import shensi...` work in any interpreter) and the in-recipe tokenizer path |

## Why a Bridge Subclass Is Needed

vLLM's Transformers backend replaces `nn.Linear` / `RMSNorm` / `nn.Embedding` / QKV inside the HF
model with the engine's own kernels (fused QKV, paged attention, TP-aware linears) — exactly what
the backbone wants, and exactly what breaks the connection. Two measured failures (vLLM 0.30):

1. `TPAwareRMSNorm` has no `.eps`, which the connection's read reads;
2. the connection computes in fp32 while the replaced `nn.Linear` holds bf16 weights → dtype clash.

`DepthTransformersForCausalLM` masks the connection subtrees (swapping them for `nn.Identity`)
while the engine rewrites the backbone, then restores them verbatim: the backbone runs on engine
kernels, the connection keeps its HF definition, and no vLLM file is touched.

## Quick Start

```bash
R=<repo>/src/shensi/recipes/paper/gated_delta_attn_res
export PATH=$(dirname $(which python)):$PATH        # flashinfer JIT needs ninja

# 0) one-time: let the engine discover the plugin (no env vars afterwards)
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model install

# 1) one variant: checkpoint -> 16 tokens
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.tiny_checkpoint gdar /tmp/vllm_smoke/gdar --overwrite
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.smoke_generate --variant gdar --tokens 16 \
    --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 128

# 2) all seven (one subprocess each) and diff against plain transformers
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.smoke_generate --all \
    --tokens 16 --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 128 --json /tmp/vllm_smoke.json

# 3) 0.6B shape (28 layers / 1024 hidden) and continuous batching
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.tiny_checkpoint dar /tmp/vllm_06b/dar --shape 0.6b --overwrite
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.batch_generate --variant dar --compare-hf

# 4) the mudd / QKV-fusion minimal reproduction (CPU)
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.mudd_fused_qkv_repro
```

## Measured Status

- `register_model register --check`: **7/7 architectures registered** (vllm 0.30.1rc0.dev360).
- Self-containment: after importing `models.vllm.register_model`, neither `megatron` nor
  `transformer_engine` is in `sys.modules`.
- End-to-end generation (tiny checkpoint → `vllm.LLM` → diff against the HF reference): **7/7
  variants generate, token-identical with the plain-transformers reference (lcp = 16/16, fp32,
  tiny shape)**.
- The engine does replace norm modules, so the GDAR modeling file caches `eps` inside the
  connection (`self._eps`) — that immunity is the one fix that took gdar from FAIL to OK.
  `tiny_checkpoint` / the HF remote-code cache are keyed by file hash: after editing a modeling
  file, remove the old checkpoints and
  `~/.cache/huggingface/modules/transformers_modules/<name>/`.
- Continuous batching matches the one-by-one results and the scheduler really overlaps (peak
  concurrency > 1). Under bf16, greedy decoding of random weights can flip an argmax when the
  batch shape changes (top-2 gap ~1e-4) — a numerical tie, not a scheduling defect.
- **The depth state needs no cache** (every depth operation is pointwise along the token axis, so
  incremental decoding equals a full recompute) — the premise for writing a native engine kernel.

## Limitations

There is **no native kernel for the connection**: the registered implementation delegates
execution to the HF implementation, so vLLM contributes scheduling, batching, sampling and the
OpenAI-compatible surface, while paged attention acts on the *backbone*, not on the depth routing.
Extreme-throughput work (a fused connection kernel / native paged implementation) is a separate
engineering item.

## Further Reading

- [HF reference implementations](../transformers/README.md) — the models being served
- [RL stage](../../stage2_rl/README.md) — vLLM is the rollout engine there
- [LIMITATIONS.md](../../LIMITATIONS.md) — B6 (fused kernels)
