"""Configuration for Qwen3 + MUDDFormer (Multiway Dynamic Dense connections).

Mirrors the structure of ``moonshotai/Kimi-K3:configuration_kimi_k3.py``: the
config lives in its own file and ``modeling_qwen3_mudd.py`` imports it.

Reference (official, consulted): ``Caiyun-AI/MUDDFormer`` -- paper "MUDDFormer:
Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense
Connections" (Xiao, Meng, Li, Yuan; arXiv:2502.12170, ICML 2025).

    * ``jax/MaxText/layers/mudd.py`` (``Mlp`` / ``Compose``) - the reference the
      released checkpoints were trained with;
    * ``pytorch/muddformer/modeling_muddformer.py`` (``MultiwayDynamicDenseBlock``)
      - the released PyTorch port;
    * ``jax/MaxText/exp.py`` (``MUDDLlama2Medium``) - the canonical flag values.

Defaults here are the paper's core mechanism on a *matched Qwen3 trunk* (so that
MUDD-vs-GDAR differs in the connection only).  ``official_preset()`` switches on
the remaining paper/official-JAX knobs (PrePostDANorm, depth-scaled FFN):

    cfg = Qwen3MUDDConfig(**base, **Qwen3MUDDConfig.official_preset())

Knob by knob:

  ``attn_res_block_size``
      ``None`` -> plain Qwen3 (no dense connection at all).
      ``1``    -> one DA module after every block, the paper's MUDDFormer.
      ``N>1``  -> one DA every N blocks.  This is an *extension* (the official
                  code only ever applies MUDD per layer); it exists so that the
                  harness's ``attn_res_block_size`` sweep means the same thing for
                  every variant.  The DA still aggregates all states up to the
                  current block, only the number of DA modules changes.
  ``attn_res_output_route``
      ``True`` -> the last layer always carries a DA module, so the model output
      is a dense-connection output (official ``dynamic_dense_fix_last_layer`` and
      ``MUDDFormer(X) = Xbar_L^R``).
  ``mudd_num_ways``
      number of decoupled input streams C: ``4`` = (query, key, value, residual),
      the paper's *multiway* MUDD (official ``dense_type='qkvr'``); ``1`` = the
      single-stream variant (official ``dense_type='l'``), which is exactly
      DenseFormer with input-dependent weights.
  ``mudd_dw_norm``
      ``"none"`` (default, official) applies no normalisation to the generated
      connection weights.  ``"softmax"`` is offered because the redo plan assumed
      softmax routing; the paper explicitly removed it (Appendix A: "Softmax is
      removed ... we empirically found that adding more sophisticated ingredients
      in DA (e.g. input dependent keys, softmax) does not bring improvement and
      slow down training"), and softmax breaks the identity initialisation.
  ``mudd_param``
      ``"deviation"`` (default) = the static prior is ``one_hot(last) + delta``
      with ``delta`` zero-initialised, i.e. the *same forward value* as the
      official initialisation but with the identity as a construction guarantee
      (and a live gradient, see the modeling file);
      ``"official"`` = a raw prior parameter initialised one-hot (the official
      JAX initialisation: ``dense_proj2`` kernel 0 + bias one-hot on the last
      state);  ``"random"`` = ``torch.randn`` like the *released PyTorch* release
      (``self.dense_bs = nn.ParameterList([... torch.randn(C, lidx+2) ...])``), which
      does *not* start at the identity -- kept as an ablation.
  ``mudd_pre_norm`` / ``mudd_post_norm``
      paper Sec. 2.5 "Optional Normalization" (PreDANorm / PostDANorm), used by
      the official JAX LLM runs (``mudd_prenorm = mudd_postnorm = True``).  Off by
      default: they change the trunk (extra norms on the state list) and are not
      needed at the scales used in this repository.  With ``post_norm`` the prior
      is initialised to 0 like the official ``dense2_bias_init_value``.
  ``mudd_ffn_depth_scaling``
      paper Sec. 2.4 "Parameter Re-allocation" (Eq. 9; official
      ``dynamic_mlp_dim = True``): the FFN hidden dim grows linearly with depth so
      that the total parameter count is unchanged.  Off by default because it
      makes the trunk differ from the other variants.
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

__all__ = ["Qwen3MUDDConfig"]


@_strict_config
class Qwen3MUDDConfig(Qwen3Config):
    """Qwen3Config + Multiway Dynamic Dense connection knobs."""

    model_type = "qwen3_mudd"

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` resolve through this when the
    #: checkpoint is loaded with ``trust_remote_code=True`` -- which is how a
    #: fresh process (verl's, vLLM's) gets the definition: these two files are
    #: copied next to the weights by ``save_pretrained``.
    auto_map = {
        "AutoConfig": "configuration_qwen3_mudd.Qwen3MUDDConfig",
        "AutoModel": "modeling_qwen3_mudd.Qwen3MUDDModel",
        "AutoModelForCausalLM": "modeling_qwen3_mudd.Qwen3MUDDForCausalLM",
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

    # --- placement, same switch as the AR / DAR / GDAR variants -----------------
    attn_res_block_size: int | None = None
    attn_res_output_route: bool = True

    # --- the MUDD module itself ------------------------------------------------
    mudd_num_ways: int = 4
    mudd_act: str = "gelu"
    mudd_hidden_round: int = 64
    mudd_fix_last_layer: bool = True
    mudd_last_layer_expand: int = 4
    mudd_param: str = "deviation"
    mudd_dw_norm: str = "none"
    mudd_scale_dw: bool = False
    mudd_sepln: bool = True

    # --- official extras (paper Sec. 2.4 / 2.5) --------------------------------
    mudd_pre_norm: bool = False
    mudd_post_norm: bool = False
    mudd_ffn_depth_scaling: bool = False
    mudd_ffn_round: int = 128

    @classmethod
    def official_preset(cls, **overrides):
        """Kwargs for the official JAX training configuration (``MUDDLlama2Medium``).

        ``mudd_prenorm = mudd_postnorm = True`` (PrePostDANorm), FFN parameter
        re-allocation on.  These change the trunk, so they are opt-in: use them to
        reproduce MUDDFormer as published, keep the defaults to compare methods on
        an identical Qwen3 trunk.

        >>> cfg = Qwen3MUDDConfig(**base, **Qwen3MUDDConfig.official_preset())
        """
        preset = dict(
            mudd_pre_norm=True,
            mudd_post_norm=True,
            mudd_ffn_depth_scaling=True,
        )
        preset.update(overrides)
        return preset


#: the fields this variant adds on top of ``Qwen3Config`` (read by ``to_dict``)
_EXTRA_CONFIG_FIELDS = tuple(Qwen3MUDDConfig.__annotations__)

# ``model_type = "qwen3_mudd"`` is resolved through the HF config registry, so
# importing this module is enough to make ``AutoConfig.from_pretrained`` work.
# ``modeling_qwen3_mudd`` registers the model classes and repeats this call.
try:
    AutoConfig.register("qwen3_mudd", Qwen3MUDDConfig)
except ValueError:
    pass

# Marks this module as ''checkpoint-local code'': ``save_pretrained`` then writes
# ``auto_map`` into ``config.json`` and copies this file next to the weights, so
# ``AutoConfig.from_pretrained(path, trust_remote_code=True)`` works in a fresh
# process that has never imported this package (verl's MegatronWorker, vLLM).
try:
    Qwen3MUDDConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):  # older/newer transformers without the API
    pass
