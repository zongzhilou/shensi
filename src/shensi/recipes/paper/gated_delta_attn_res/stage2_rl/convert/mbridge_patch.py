"""Make ``bridge.load_weights`` / ``export_weights`` work for the depth variants.

``verl``'s Megatron worker calls exactly two weight methods
(``verl/workers/megatron_workers.py``)::

    bridge.load_weights(actor_module, local_model_path)      # HF ckpt -> mcore
    bridge.export_weights(actor_module)                      # mcore -> HF dict

mbridge's implementations of both run off a name table
(``Bridge._ATTENTION_MAPPING`` / ``_MLP_MAPPING`` / ``_OTHER_MAPPING``) that has
no entry for any of our connection tensors, and whose backbone entries assume the
TransformerEngine *fused* layout rather than the local one our specs build.
``VERL_REGISTRATION.md`` §4 is the run that died on the first backbone norm::

    NotImplementedError: Unsupported parameter name: decoder.layers.0.input_layernorm.weight

How this module fixes it, and why this way
------------------------------------------
mbridge's tables could be extended (``_OTHER_MAPPING`` for ``input_layernorm``,
``_MLP_MAPPING`` for ``pre_mlp_layernorm``) and the connection tensors added on
top -- that would keep mbridge's tensor-parallel merge/split machinery, which is
real value.  It cannot be the whole answer, for one structural reason:

    ``Bridge.load_weights`` builds ``to_load_from_disk`` from the mapped HF names
    and then *requires every one of them to exist in the ``.safetensors`` files*.

Our tables contain ``synth`` rows -- the AR/DAR ``read_scale`` gate and the
low-rank ``q/k`` ``up.bias`` -- for which no HF tensor exists at all.  No name
mapping can conjure one up, so with mbridge's loader those tensors can only be
loaded by lying about their name (aliasing them to an unrelated tensor and then
overwriting the value), which is exactly the kind of trick that hides a bug.

So both methods are replaced by a **state-dict conversion** driven by
:mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables`, using mbridge's own ``SafeTensorIO`` for the
I/O (sharded index, single file, ``cached_file`` -- all unchanged) and
``torch.nn.Module.load_state_dict`` for the write.  The conversion report is kept
on the bridge as ``last_report`` and the load is made *strict by hand*: a
missing or unexpected key raises with the report attached instead of being
quietly tolerated.

Cost: ``tp > 1`` is refused (the converter produces full, unsharded Megatron
tensors).  That is not a regression -- with mbridge's stock tables a ``tp > 1``
run cannot get past the first connection tensor anyway -- and the connection
parameters are replicated rather than TP-parallel, so there is nothing to shard
on that side.
"""

from __future__ import annotations

import warnings

import torch

from .convert import AttnLayout, hf_to_mcore, mcore_to_hf
from .tables import SynthesisPolicy, Table, build_table

__all__ = ["WeightConversionMixin"]


def _copy_into(model, converted: dict[str, torch.Tensor], table: Table, report) -> None:
    """Write ``converted`` into ``model``'s **existing** tensors, tensor by tensor.

    Not ``model.load_state_dict(...)``, for a reason the end-to-end run found: the
    modules ``verl`` hands to ``bridge.load_weights`` are DDP-wrapped
    (``DistributedDataParallel.load_state_dict`` returns ``None``), so the
    "call it and inspect the result" pattern raises ``AttributeError: 'NoneType'
    object has no attribute 'missing_keys'`` somewhere inside the first actor
    initialisation.  Reading ``model.state_dict()`` and copying into the tensors it
    returns works for a plain ``GPTModel``, a DDP wrapper and a ``Float16Module``
    alike -- it is what mbridge's own ``load_weights`` does -- and it casts to the
    model's dtype/device on the way in (``bf16`` actors, ``fp32`` checkpoints).

    The strictness is kept by hand, and stays the point: a key the model has and
    the conversion does not (or vice versa), or a shape disagreement, raises with
    the conversion report attached.  This is the "unmapped tensors = 0" check at
    run time, and it must never degrade into a silent partial load.

    One exception, and it is not a hole: a **value head** -- the critic's
    ``output_layer.weight`` of width 1 -- legitimately disagrees with the
    checkpoint's ``lm_head``, because ``verl`` deliberately loads the actor
    checkpoint into the critic.  ``mbridge``'s loader makes the same exception
    (it skips a width-1 ``output_layer``) and ``verl``'s non-vanilla path spells it
    out as ``allowed_mismatched_params=["output_layer.weight"]``.  That one tensor
    is skipped and reported; everything else still has to match exactly.
    """
    target = model.state_dict()
    #: critic-only: a single-row output layer is a value head, not the LM head
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
    """Mixed into ``Qwen3DepthBridge``; uses ``self.hf_config``, ``self.variant``, ``self.mpu``.

    Attributes the mixin adds:

    * ``conversion_table`` -- the :class:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables.Table` for this
      bridge, built once (it is a pure function of the config) and reused.
    * ``conversion_policy`` -- the :class:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables.SynthesisPolicy`
      in force; set it *before* the first load to change the ``read_scale`` policy.
    * ``last_report`` -- the :class:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.convert.ConversionReport` of
      the most recent load or export, for logging and for ``selftest.py``.
    """

    conversion_table: Table | None = None
    conversion_policy: SynthesisPolicy | None = None
    last_report = None

    # -- plumbing ----------------------------------------------------------

    def conversion_table_for(self) -> Table:
        """The table for this bridge's variant and config (built on first use)."""
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
        """The whole checkpoint as a plain dict, through mbridge's own reader."""
        self.safetensor_io = self._get_safetensor_io(weights_path)
        names = self.safetensor_io.load_hf_weight_names()
        if not names:
            raise FileNotFoundError(
                f"no tensors found under {weights_path!r}: mbridge's SafeTensorIO accepts a "
                f"directory of .safetensors (sharded via model.safetensors.index.json or not)"
            )
        state = self.safetensor_io.load_some_hf_weight(list(names))
        if getattr(self.hf_config, "tie_word_embeddings", False) and "lm_head.weight" not in state:
            # mcore keeps one shared Parameter for the embedding and the head, so the
            # table always produces both names; a tied checkpoint only ships one.
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        return state

    def _padded_vocab_size(self, models: list) -> int | None:
        """Megatron's (padded) vocabulary, read off the built model.

        ``self.padded_vocab_size`` is only filled in by ``_model_provider``, which
        has not necessarily run when a caller loads into an existing model, so
        the model itself is the authority.  ``None`` means "no padding".
        """
        if getattr(self, "padded_vocab_size", None) is not None:
            return self.padded_vocab_size
        for model in models:
            for key, value in model.state_dict().items():
                if key == "embedding.word_embeddings.weight":
                    return int(value.shape[0])
        return None

    # -- the two methods verl calls ---------------------------------------

    def load_weights(self, models: list, weights_path: str, memory_efficient: bool = False) -> None:
        """HF checkpoint directory -> Megatron model(s).

        Drop-in for ``Bridge.load_weights``.  ``memory_efficient`` is accepted and
        ignored: mbridge's streaming variant exists to avoid materialising the
        whole checkpoint, but every rank here needs the whole tensor set anyway
        (the connection parameters are replicated), so streaming per tensor would
        buy nothing and add a second code path.

        Raises:
            NotImplementedError: ``tp > 1`` -- see the module docstring.
            RuntimeError: the converted checkpoint does not match the model.  The
                conversion report is attached to the message: a key mismatch here
                is a table bug or a config mismatch, never a checkpoint problem,
                and it must not be silently tolerated.
        """
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
            # Loud on purpose: these are configurations where the Megatron port's
            # arithmetic is *not* the reference's, so the load succeeds but the
            # model does not compute what the checkpoint describes.  Silent would
            # be the worst outcome; failing would strand otherwise usable runs.
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
        """Megatron model(s) -> ``(hf_name, tensor)`` pairs.

        Same generator contract as ``Bridge.export_weights``, so ``save_weights``
        and ``verl``'s rollout-weight plumbing work unchanged.  The synthesized
        Megatron-only rows have no HF name and are therefore not yielded; they
        are listed in ``self.last_report.dropped``.
        """
        if self.mpu.pp_size > 1 or self.mpu.tp_size > 1:
            raise NotImplementedError(
                f"export_weights supports pp=1, tp=1 (got pp={self.mpu.pp_size}, "
                f"tp={self.mpu.tp_size}); the depth variants already reject pp>1 at build time"
            )
        table = self.conversion_table_for()
        converted: dict[str, torch.Tensor] | None = None
        for model in models:
            # `state_dict()` recurses through DDP / Float16Module wrappers, which
            # is what makes this work for the actor verl builds
            state = {k: v for k, v in model.state_dict().items() if v is not None and not k.endswith("_extra_state")}
            # A value head has no HF counterpart to export: `lm_head.weight` is a
            # (vocab, H) tensor and the critic's is (1, H).  Dropping it here beats
            # letting `mcore_to_hf` fail with "Megatron vocab 1 < 151936", which is
            # what a reader would otherwise have to decode.  The critic's weights
            # are not synced to a rollout engine, so nothing depends on this.
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

    #: ``verl`` calls ``load_hf_weights`` instead of ``load_weights`` when
    #: ``actor_rollout_ref.actor.megatron.vanilla_mbridge=False``.  mbridge's
    #: ``Bridge`` has no such method -- it is a ``megatron-bridge`` name that this
    #: verl version already expects -- so the alias is what keeps the non-vanilla
    #: flag from failing with an ``AttributeError`` on the first checkpoint.
    load_hf_weights = load_weights
