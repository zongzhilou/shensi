"""Configuration for Qwen3 + Delta Attention Residuals (DAR).

Mirrors the structure of ``moonshotai/Kimi-K3:configuration_kimi_k3.py``: the
config lives in its own file and ``modeling_qwen3_dar.py`` imports it.

Reference for the connection: ``wdlctc/delta-attention-residuals-code``.
"""

from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def _strict_config(cls):
    """Apply ``huggingface_hub.dataclasses.strict`` where it is applicable.

    transformers v5 turns ``PretrainedConfig`` -- and therefore every subclass --
    into a dataclass, which is precisely the precondition ``strict`` checks, and
    it is what promotes the class-level annotations below into validated fields.
    transformers v4 keeps a plain class there and ``strict`` raises
    ``StrictDataclassDefinitionError``, so the decorator is skipped and the extra
    fields stay ordinary class attributes (``to_dict`` below still serialises
    them).  This is a runtime probe rather than a version test so that whichever
    transformers is installed gets the behaviour that is correct for it.
    """
    try:
        from huggingface_hub.dataclasses import strict
    except ImportError:  # pragma: no cover - huggingface_hub without dataclasses
        return cls
    try:
        return strict(cls)
    except Exception:
        return cls


__all__ = ["Qwen3DARConfig"]


@_strict_config
class Qwen3DARConfig(Qwen3Config):
    """Qwen3Config + Delta Attention Residuals knobs.

    ``attn_res_block_size``:
      * ``None`` -> depth routing disabled, the model is exactly stock Qwen3;
      * ``1``    -> one source per sublayer output (the paper's Delta AttnRes);
      * ``N``    -> one source per block of N layers (Delta Block).

    ``attn_res_use_null_source`` prepends a learnable null source (zeros at
    initialisation) so that a randomly initialised query still yields a
    near-identity update -- the reference uses this for fine-tuning a pretrained
    checkpoint.
    """

    model_type = "qwen3_dar"

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` resolve through this when the
    #: checkpoint is loaded with ``trust_remote_code=True`` -- which is how a
    #: fresh process (verl's, vLLM's) gets the definition: these two files are
    #: copied next to the weights by ``save_pretrained``.
    auto_map = {
        "AutoConfig": "configuration_qwen3_dar.Qwen3DARConfig",
        "AutoModel": "modeling_qwen3_dar.Qwen3DARModel",
        "AutoModelForCausalLM": "modeling_qwen3_dar.Qwen3DARForCausalLM",
    }

    def to_dict(self):
        """Serialise this variant's knobs as well as ``Qwen3Config``'s.

        On transformers v5 every declared field is a dataclass field and
        ``to_dict`` already emits it, so ``setdefault`` is a no-op and the output
        is bit-identical to the inherited implementation.  On v4 the extra fields
        exist only as class attributes and ``PretrainedConfig.to_dict`` -- which
        deep-copies ``self.__dict__`` -- would silently drop them, which would
        make a saved checkpoint lose the connection's geometry.
        """
        output = super().to_dict()
        # ``auto_map`` is a plain class attribute here, and neither transformers
        # serialises plain class attributes (v4's ``to_dict`` copies
        # ``self.__dict__``, v5's emits dataclass fields only) -- yet *both* read
        # it back out of ``config.json`` to resolve the files that ship next to
        # the weights.
        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output

    attn_res_block_size: int | None = None
    attn_res_output_route: bool = True
    attn_res_use_null_source: bool = False


#: the fields this variant adds on top of ``Qwen3Config`` (read by ``to_dict``)
_EXTRA_CONFIG_FIELDS = tuple(Qwen3DARConfig.__annotations__)

# ``model_type = "qwen3_dar"`` is resolved through the HF config registry, so
# importing this module is enough to make ``AutoConfig.from_pretrained`` work.
# ``modeling_qwen3_dar`` registers the model classes and repeats this call.
try:
    AutoConfig.register("qwen3_dar", Qwen3DARConfig)
except ValueError:
    pass

# Marks this module as ''checkpoint-local code'': ``save_pretrained`` then writes
# ``auto_map`` into ``config.json`` and copies this file next to the weights, so
# ``AutoConfig.from_pretrained(path, trust_remote_code=True)`` works in a fresh
# process that has never imported this package (verl's MegatronWorker, vLLM).
try:
    Qwen3DARConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):  # older/newer transformers without the API
    pass
