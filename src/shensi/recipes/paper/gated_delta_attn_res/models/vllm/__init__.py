"""Rollout-side support: register the 7 depth-routed variants with an inference engine.

The training stack (``verl``) lives at transformers 4.57.6; the modeling code in
``models/`` needs transformers v5 (``@strict`` configs).  The inference engine is
therefore these tools are the *only* part of the recipe the engine imports (the
repo's single venv carries transformers + vLLM; a deployment can still split it)
part of the repo it imports.

    rollout/variants.py         the 7 variants, as the engine sees them (+ checkpoint shapes)
    rollout/tiny_checkpoint.py  build a random-weight checkpoint of any variant / shape
    rollout/register_model.py   register the architectures with the engine
    rollout/smoke_generate.py   end-to-end: checkpoint -> engine -> 16 tokens, vs the HF reference
    rollout/batch_generate.py   continuous batching: many prompts at once vs one at a time
    rollout/mudd_fused_qkv_repro.py  the mudd/fused-QKV minimal reproduction (CPU only)
    rollout/sitecustomize.py    make the registration reach engine *worker* processes

Nothing in here patches the engine's installed files.
"""

__all__ = ["variants", "tiny_checkpoint"]
