"""Run our connection modules *inside* vLLM's Transformers backend, unmodified.

Why a subclass is needed
------------------------
vLLM's Transformers backend (``vllm.model_executor.models.transformers``) is the right
way to serve an architecture vLLM has no native kernel for: it builds the HF model,
then walks it and **replaces recognised modules with vLLM's own** so that attention
goes through vLLM's attention layer and the projections are tensor-parallel aware:

    nn.Linear        -> vLLM's ReplicatedLinear / ColumnParallelLinear / ...
    RMSNorm          -> vLLM's RMSNorm / TPAwareRMSNorm   (detected from dataflow)
    nn.Embedding     -> VocabParallelEmbedding
    attention QKV    -> a fused QKVParallelLinear + vLLM's Attention instance

That is exactly what we want for the *backbone*, and exactly what breaks the
*connection*.  Measured on vLLM 0.30 (see ``README.md`` in this directory and the package's
``code/ROLLOUT_ENV.md`` §5
for the raw tracebacks), with the stock ``TransformersForCausalLM``:

1. ``TPAwareRMSNorm`` has no ``.eps``.  ``AttentionResidual.read`` passes
   ``self.norm.eps`` (the unweighted RMSNorm's epsilon) to the depth read, and the
   replaced module simply does not carry that attribute:
   ``AttributeError: 'TPAwareRMSNorm' object has no attribute 'eps'``.
2. The connection computes in float32 on purpose (``prefix.float()``) while the
   replaced ``nn.Linear`` holds bf16 weights, so
   ``RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float !=
   c10::BFloat16``.  Under the plain HF implementation the same model is fine, because
   there the surrounding parameters *are* float32 -- the inference engine is what turns
   the model's fp32-internal arithmetic into a dtype conflict.

Both are the same root cause: the connection reads attributes of, and relies on the
parameterisation of, modules that the engine is entitled to swap out.  The connection
is not part of the backbone's compute graph in any way the engine's rewrite rules
anticipate.

What this class does
--------------------
It overrides ``recursive_replace`` -- the single method that performs the rewrite -- to
*mask off* the connection subtrees (swapping them for ``nn.Identity``) while the
backbone is rewritten, then puts them back verbatim.  The result:

* the backbone is fully engine-native (vLLM's fused QKV, vLLM's paged attention, vLLM's
  TP-aware linears and norms);
* the connection keeps the HF definitions the model was written against, so it runs
  the same arithmetic as ``AutoModelForCausalLM`` would;
* **no file shipped with vLLM is modified**, and ``models/*.py`` is untouched.

What counts as "the connection"
-------------------------------
Not a name list, which would rot: inside a decoder layer everything *except* the four
backbone children is ours, and at the model root everything except the embedding, the
layer stack, the final norm and the rotary embedding is ours.  That rule is derived
from the structure of ``models/modeling_qwen3_*.py`` and holds for all 7 variants even
though no two of them name their connection the same way (``AttentionResidual``,
``DepthRead``, ``HyperConnection``, ``MultiwayDynamicDense``,
``DepthWeightedAverage``, or -- in AR/DAR -- plain ``nn.Linear`` attributes on the
decoder layer such as ``self_attention_res_proj``).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

#: children of a decoder layer that belong to the stock Qwen3 backbone
BACKBONE_LAYER_CHILDREN = frozenset(
    {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
)

#: children of the base model that belong to the stock Qwen3 backbone
BACKBONE_MODEL_CHILDREN = frozenset({"embed_tokens", "layers", "norm", "rotary_emb"})


def connection_module_names(model: nn.Module, architecture: str | None = None) -> list[str]:
    """Qualnames of the modules that must not be rewritten by the engine.

    Purely structural: a decoder layer is any child of an ``nn.ModuleList`` whose
    length equals ``num_hidden_layers``.
    """
    texts = [type(model).__name__, architecture or ""]
    is_hf_root = any("ForCausalLM" in t or t.endswith("Model") for t in texts)

    protected: list[str] = []
    for name, child in model.named_children():
        del child
        if not is_hf_root or name in BACKBONE_MODEL_CHILDREN:
            continue
        protected.append(name)
    # Decoder layers: everything except their four backbone children is ours.
    # ``named_children`` -- submodules live in ``Module._modules``, not ``__dict__``.
    layers = getattr(model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        for i, layer in enumerate(layers):
            for name, _sub in layer.named_children():
                if name in BACKBONE_LAYER_CHILDREN:
                    continue
                protected.append(f"layers.{i}.{name}")
    return protected


@contextlib.contextmanager
def _masked(model: nn.Module, qualnames: list[str]):
    """Swap each named subtree for ``nn.Identity``; restore on exit, always."""
    saved: list[tuple[nn.Module, str, nn.Module]] = []
    try:
        for qual in qualnames:
            parts = qual.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            module = getattr(parent, attr, None)
            if module is None:
                continue
            saved.append((parent, attr, module))
            setattr(parent, attr, nn.Identity())
        yield
    finally:
        for parent, attr, module in saved:
            setattr(parent, attr, module)


def _vllm_base():
    """The engine's Transformers backend base class, or None if unavailable."""
    try:
        from vllm.model_executor.models.transformers import TransformersForCausalLM

        return TransformersForCausalLM
    except Exception:
        return None


_BASE = _vllm_base()

if _BASE is None:  # pragma: no cover - vLLM not installed in this interpreter

    class DepthTransformersForCausalLM(nn.Module):  # type: ignore[no-redef]
        """Placeholder so importing this module never fails; registration will too."""

        def __init__(self, *a, **kw):
            raise RuntimeError(
                "vLLM's Transformers backend is unavailable in this interpreter, so "
                "shensi.recipes.paper.gated_delta_attn_res.models.vllm.vllm_bridge cannot be used. Install vllm (see ../README.md（本目录）与包内 code/ROLLOUT_ENV.md)."
            )

else:

    class DepthTransformersForCausalLM(_BASE):  # type: ignore[misc,valid-type]
        """``TransformersForCausalLM``, minus the rewrite of the connection subtrees."""

        #: filled in by ``register_model.register_all`` so the report can name it
        protected_qualnames: list[str] = []

        def recursive_replace(self):
            model = self.model
            qualnames = connection_module_names(model, getattr(self.config, "model_type", None))
            self.protected_qualnames = qualnames
            if not qualnames:
                return super().recursive_replace()
            with _masked(model, qualnames):
                super().recursive_replace()

        # ``_create_attention_instances`` needs every decoder layer to have dispatched
        # through the attention interface; the mask above must not have hidden the
        # attention modules themselves, which is why they are in
        # BACKBONE_LAYER_CHILDREN.  Assert it rather than trust it.
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            missing = [q for q in self.protected_qualnames if _resolve(self.model, q) is None]
            if missing:  # pragma: no cover - would mean the mask/restore broke
                raise RuntimeError(f"connection modules were not restored: {missing}")


def _resolve(root: nn.Module, qualname: str):
    obj = root
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


__all__ = ["DepthTransformersForCausalLM", "connection_module_names"]
