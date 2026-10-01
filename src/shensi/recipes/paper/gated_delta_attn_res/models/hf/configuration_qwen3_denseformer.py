"""Configuration for Qwen3 + DenseFormer (Depth-Weighted Averaging).

Mirrors the structure of ``moonshotai/Kimi-K3:configuration_kimi_k3.py``: the
config lives in its own file and ``modeling_qwen3_denseformer.py`` imports it.

Reference implementation (official, consulted): ``epfml/DenseFormer`` --

    * ``denseformer/denseformer.py``      -> ``DWAModules`` (the released package)
    * ``experiments/models/denseformer.py`` -> the GPT-2 style experiment model
    * paper: "DenseFormer: Enhancing Information Flow in Transformers via Depth
      Weighted Averaging", Pagliardini, Mohtashami, Fleuret, Jaggi, 2024
      (arXiv:2402.02622; NeurIPS 2024).

Both files were read while writing ``modeling_qwen3_denseformer.py``; the
Python files are kept in ``tmp/official_refs/denseformer/`` for cross-checking.
The formulation implemented here is the official one, not a re-derivation::

    Y_i = DWA_i({X_0, ..., X_i}) = sum_{j<=i} alpha_{i,j} * X_j
    alpha_{i,i} = 1, alpha_{i,j} = 0 elsewhere      # official initialisation

``attn_res_block_size`` uses the same "per layer / per block" semantics as the
AR / DAR / GDAR variants in this directory:

  * ``None`` -> no DWA at all, the model is exactly stock Qwen3;
  * ``1``    -> a DWA module after every block: the full DenseFormer (1x1);
  * ``N``    -> a DWA module every ``N`` blocks: the paper's *periodic*
    DenseFormer (``Nx1`` in the paper's ``kxp`` notation, ``DWAModules(period=N)``
    in the official package).

``attn_res_dwa_dilation`` is the paper's second sparsification knob
(``DWAModules(dilation=k)``): DWA_i averages only the states whose index is
``= i (mod k)``.  ``attn_res_output_route`` decides whether the very last layer
always carries a DWA, which is what makes the model output ``Y_d`` as in the
paper (default) rather than "whatever the last DWA event left in the stream".
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

__all__ = ["Qwen3DenseFormerConfig"]


@_strict_config
class Qwen3DenseFormerConfig(Qwen3Config):
    """Qwen3Config + Depth-Weighted Averaging knobs.

    ``attn_res_block_size``: ``None`` (off) | ``1`` (DWA after every block) |
    ``N`` (DWA every N blocks, the paper's periodic DenseFormer).

    ``attn_res_dwa_dilation``: keep only every ``k``-th source state, i.e.
    ``DWA_i`` averages ``{X_j : j <= i, j = i (mod k)}`` (paper Sec. 3.2).

    ``attn_res_dwa_param``:
      * ``"deviation"`` (default): the alpha vector is ``one_hot(last) + delta``
        with ``delta`` zero-initialised.  Same forward value as the official
        initialisation, but the identity is a *constructed* guarantee that
        survives any re-initialisation of the module (a plain zero-init
        multiplicative gate would be wrong here: ``Y = X + s*(sum a X - X)`` has
        zero gradient w.r.t. both ``s`` and ``a`` at the identity point, so the
        connection could never leave it -- see the modeling file docstring);
      * ``"official"``: a raw ``nn.Parameter`` initialised to the one-hot vector
        (literally ``weight.zero_(); weight[-1] = 1`` from the official code).
    """

    model_type = "qwen3_denseformer"

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` resolve through this when the
    #: checkpoint is loaded with ``trust_remote_code=True`` -- which is how a
    #: fresh process (verl's, vLLM's) gets the definition: these two files are
    #: copied next to the weights by ``save_pretrained``.
    auto_map = {
        "AutoConfig": "configuration_qwen3_denseformer.Qwen3DenseFormerConfig",
        "AutoModel": "modeling_qwen3_denseformer.Qwen3DenseFormerModel",
        "AutoModelForCausalLM": "modeling_qwen3_denseformer.Qwen3DenseFormerForCausalLM",
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
    attn_res_dwa_dilation: int = 1
    attn_res_dwa_param: str = "deviation"


#: the fields this variant adds on top of ``Qwen3Config`` (read by ``to_dict``)
_EXTRA_CONFIG_FIELDS = tuple(Qwen3DenseFormerConfig.__annotations__)

# ``model_type = "qwen3_denseformer"`` is resolved through the HF config registry, so
# importing this module is enough to make ``AutoConfig.from_pretrained`` work.
# ``modeling_qwen3_denseformer`` registers the model classes and repeats this call.
try:
    AutoConfig.register("qwen3_denseformer", Qwen3DenseFormerConfig)
except ValueError:
    pass

# Marks this module as ''checkpoint-local code'': ``save_pretrained`` then writes
# ``auto_map`` into ``config.json`` and copies this file next to the weights, so
# ``AutoConfig.from_pretrained(path, trust_remote_code=True)`` works in a fresh
# process that has never imported this package (verl's MegatronWorker, vLLM).
try:
    Qwen3DenseFormerConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):  # older/newer transformers without the API
    pass
