"""vLLM 原生实现与 HF 参考之间的连接模块对照与解析。"""


from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


BACKBONE_LAYER_CHILDREN = frozenset(
    {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
)


BACKBONE_MODEL_CHILDREN = frozenset({"embed_tokens", "layers", "norm", "rotary_emb"})


def connection_module_names(model: nn.Module, architecture: str | None = None) -> list[str]:
    """列出 vLLM 原生实现里的连接模块名。"""
    texts = [type(model).__name__, architecture or ""]
    is_hf_root = any("ForCausalLM" in t or t.endswith("Model") for t in texts)

    protected: list[str] = []
    for name, child in model.named_children():
        del child
        if not is_hf_root or name in BACKBONE_MODEL_CHILDREN:
            continue
        protected.append(name)


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
    try:
        from vllm.model_executor.models.transformers import TransformersForCausalLM

        return TransformersForCausalLM
    except Exception:
        return None


_BASE = _vllm_base()

if _BASE is None:  # pragma: no cover - vLLM not installed in this interpreter

    class DepthTransformersForCausalLM(nn.Module):  # type: ignore[no-redef]

        def __init__(self, *a, **kw):
            raise RuntimeError(
                "vLLM's Transformers backend is unavailable in this interpreter, so "
                "shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.vllm_bridge cannot be used. Install vllm (see ../README.md（本目录）与包内 code/ROLLOUT_ENV.md)."
            )

else:

    class DepthTransformersForCausalLM(_BASE):  # type: ignore[misc,valid-type]


        protected_qualnames: list[str] = []

        def recursive_replace(self):
            model = self.model
            qualnames = connection_module_names(model, getattr(self.config, "model_type", None))
            self.protected_qualnames = qualnames
            if not qualnames:
                return super().recursive_replace()
            with _masked(model, qualnames):
                super().recursive_replace()





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
