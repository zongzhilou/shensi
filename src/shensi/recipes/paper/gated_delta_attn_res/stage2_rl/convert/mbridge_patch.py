"""Megatron-Bridge 的权重转换补丁：接入本配方的表。"""


from __future__ import annotations

import warnings

import torch

from .convert import AttnLayout, hf_to_mcore, mcore_to_hf
from .tables import SynthesisPolicy, Table, build_table

__all__ = ["WeightConversionMixin"]


def _copy_into(model, converted: dict[str, torch.Tensor], table: Table, report) -> None:
    target = model.state_dict()

    value_head = {k for k in target if k == "output_layer.weight" and tuple(target[k].shape)[0] == 1}
    missing = [
        k for k in target if k not in converted and not k.endswith("_extra_state") and k not in value_head
    ]
    unexpected = [k for k in converted if k not in target]
    shape_bad = [
        (k, tuple(converted[k].shape), tuple(t.shape))
        for k, t in target.items()
        if k in converted and k not in value_head and tuple(converted[k].shape) != tuple(t.shape)
    ]
    if missing or unexpected or shape_bad:
        raise RuntimeError(
            f"{table.variant}: converted checkpoint does not match the Megatron model: "
            f"{len(missing)} missing (e.g. {missing[:4]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:4]}), {len(shape_bad)} wrong shape (e.g. {shape_bad[:2]}). "
            f"This is a converter/model disagreement.\n" + report.describe()
        )
    if value_head:
        warnings.warn(
            f"{table.variant}: the model has a value head (output_layer.weight with one row), so the "
            f"checkpoint's lm_head is not loaded into it -- the same exception mbridge's loader and "
            f"verl's `allowed_mismatched_params` make for critics.",
            RuntimeWarning,
            stacklevel=2,
        )
    with torch.no_grad():
        for name, tensor in converted.items():
            if name in value_head:
                continue
            target[name].copy_(tensor)


class WeightConversionMixin:

    """接入本配方权重表的 Bridge 混入：让 Megatron-Bridge 认识这些变体。"""
    conversion_table: Table | None = None
    conversion_policy: SynthesisPolicy | None = None
    last_report = None



    def conversion_table_for(self) -> Table:
        if self.conversion_table is None:
            variant = getattr(self.variant, "name", None)
            if variant is None:  # pragma: no cover - _build_config always sets it
                variant = str(self.hf_config.model_type).removeprefix("qwen3_")
            self.conversion_table = build_table(
                variant,
                self.hf_config,
                int(self.hf_config.num_hidden_layers),
                policy=self.conversion_policy,
            )
        return self.conversion_table

    def _policy(self) -> SynthesisPolicy:
        if self.conversion_policy is None:
            self.conversion_policy = SynthesisPolicy()
        return self.conversion_policy

    def _attn_layout(self) -> AttnLayout:
        return AttnLayout.from_config(self.hf_config)

    def _read_hf_state_dict(self, weights_path: str) -> dict[str, torch.Tensor]:
        self.safetensor_io = self._get_safetensor_io(weights_path)
        names = self.safetensor_io.load_hf_weight_names()
        if not names:
            raise FileNotFoundError(
                f"no tensors found under {weights_path!r}: mbridge's SafeTensorIO accepts a "
                f"directory of .safetensors (sharded via model.safetensors.index.json or not)"
            )
        state = self.safetensor_io.load_some_hf_weight(list(names))
        if getattr(self.hf_config, "tie_word_embeddings", False) and "lm_head.weight" not in state:


            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        return state

    def _padded_vocab_size(self, models: list) -> int | None:
        if getattr(self, "padded_vocab_size", None) is not None:
            return self.padded_vocab_size
        for model in models:
            for key, value in model.state_dict().items():
                if key == "embedding.word_embeddings.weight":
                    return int(value.shape[0])
        return None



    def load_weights(self, models: list, weights_path: str, memory_efficient: bool = False) -> None:
        if self.mpu.tp_size > 1:
            raise NotImplementedError(
                f"tp={self.mpu.tp_size}: the depth-connection converter produces full (unsharded) "
                f"Megatron tensors and does not shard them across tensor-parallel ranks yet. "
                f"Run with tensor_model_parallel_size=1; the connection parameters are replicated "
                f"rather than TP-parallel, so nothing is lost on that side."
            )
        hf_state = self._read_hf_state_dict(weights_path)
        table = self.conversion_table_for()
        converted, report = hf_to_mcore(
            hf_state,
            table,
            layout=self._attn_layout(),
            padded_vocab_size=self._padded_vocab_size(models),
            policy=self._policy(),
        )
        self.last_report = report
        if table.unsupported:




            warnings.warn(
                f"{table.variant}: the converted checkpoint loads, but this configuration is not "
                f"equivalent to the HF model:\n"
                + "\n".join(f"  - {entry}" for entry in table.unsupported)
                + "\nthe tensor mapping itself is complete (see conversion_table.unsupported).",
                RuntimeWarning,
                stacklevel=2,
            )
        for model in models:
            _copy_into(model, converted, table, report)

    def export_weights(self, models: list):
        if self.mpu.pp_size > 1 or self.mpu.tp_size > 1:
            raise NotImplementedError(
                f"export_weights supports pp=1, tp=1 (got pp={self.mpu.pp_size}, "
                f"tp={self.mpu.tp_size}); the depth variants already reject pp>1 at build time"
            )
        table = self.conversion_table_for()
        converted: dict[str, torch.Tensor] | None = None
        for model in models:


            state = {k: v for k, v in model.state_dict().items() if v is not None and not k.endswith("_extra_state")}





            head = state.get("output_layer.weight")
            if head is not None and tuple(head.shape)[0] == 1:
                del state["output_layer.weight"]
                warnings.warn(
                    f"{table.variant}: the model's output_layer is a value head (one row), so "
                    f"'lm_head.weight' is not exported; every other tensor is.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            converted, report = mcore_to_hf(
                state,
                table,
                layout=self._attn_layout(),
                vocab_size=int(self.hf_config.vocab_size),
                policy=self._policy(),
            )
            self.last_report = report
        if converted is None:
            return
        yield from converted.items()






    load_hf_weights = load_weights
