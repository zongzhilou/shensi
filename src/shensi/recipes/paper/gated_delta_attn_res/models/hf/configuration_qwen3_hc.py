r"""Configuration for Qwen3 + Hyper-Connections (HC).

Mirrors the structure of ``moonshotai/Kimi-K3:configuration_kimi_k3.py``: the
config lives in its own file and ``modeling_qwen3_hc.py`` imports it.

HC is the *unprojected* half of the pair: the same n-stream machinery as the mHC
variant next to it, with the residual mixing matrix left free
(``mhc_manifold="none"``) and unconstrained read/write mappings -- i.e. the
original Zhu et al. formulation (arXiv:2409.19606), whose \(A_m, A_r, B\) are
plain learnable coefficients.  Keeping the field names identical to
``Qwen3MHCConfig`` makes the two a one-line ablation of each other.

Reference implementation (ported): ``megatron-core 0.18.2``
``megatron/core/transformer/hyper_connection.py`` -- the module, the initialisation
and the block-level ``input_expand``/``output_contract`` -- with the Sinkhorn
projection switched off and the paper's initialisation (Eq. 14:
\(A_m = e_{k \\bmod n}, A_r = I, B = \\mathbf{1}\)) used for the static parts.
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

__all__ = ["Qwen3HCConfig"]


@_strict_config
class Qwen3HCConfig(Qwen3Config):
    r"""Qwen3Config + Hyper-Connections knobs.

    Defaults are the **paper-faithful, identity-initialised HC**: the static
    mappings start exactly at Eq. 14 (\(A_m = e_k\), \(A_r = I\), \(B = 1\)) and
    the dynamic branches are scaled by \(\\alpha = 0\), so ``HC(0)`` is the
    standard PreNorm residual ``h + f(h)`` bit-exactly (verified with
    ``torch.equal`` in ``test_baselines_hc.py``).  ``megatron_preset()`` keeps
    the same geometry but swaps in the megatron-core activation choices
    (\(\\sigma\) read, \(2\\sigma\) write) for a like-for-like comparison against
    the neighbouring mHC variant.
    """

    model_type = "qwen3_hc"

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` resolve through this when the
    #: checkpoint is loaded with ``trust_remote_code=True`` -- which is how a
    #: fresh process (verl's, vLLM's) gets the definition: these two files are
    #: copied next to the weights by ``save_pretrained``.
    auto_map = {
        "AutoConfig": "configuration_qwen3_hc.Qwen3HCConfig",
        "AutoModel": "modeling_qwen3_hc.Qwen3HCModel",
        "AutoModelForCausalLM": "modeling_qwen3_hc.Qwen3HCForCausalLM",
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

    # --- block layout (identical switch to AR/DAR/GDAR) --------------------
    attn_res_block_size: int | None = None

    # --- stream geometry ---------------------------------------------------
    hc_num_streams: int = 4

    # --- initialisation ----------------------------------------------------
    # "identity": the paper's Eq. 14 (A_m = e_k, A_r = I, B = 1) with the
    #             dynamic part scaled to exactly 0 at step 0.
    # "official": megatron-core's activation/scale init (alpha = 0.01, xavier
    #             mapping_proj, zero biases) with A_r anchored at I, since an
    #             unprojected zero mixing matrix would zero the stream.
    hc_init: str = "identity"

    # --- parameterisation of the three mappings ----------------------------
    hc_dynamic: bool = True
    # H_pre: "linear"  = A_m directly, unconstrained (the HC paper),
    #        "simplex" = softmax (non-negative, sums to 1),
    #        "sigmoid" = sigmoid + eps (megatron-core).
    # The reference implementation (megatron-core `hyper_connection.py`) supports only
    # "simplex" and "sigmoid", so those are the only reads that run in the training/serving
    # stacks; "linear" exists only in this file and is used by the *identity-anchored*
    # presets below because it is the one that keeps the identity bit-exact.  Published-
    # semantics presets use the official "sigmoid"; use one of those (e.g. `megatron_preset`)
    # whenever the model has to run in the training or serving stacks.
    hc_read: str = "linear"
    # H_post: "linear"   = B directly, unconstrained (the HC paper, init ones),
    #         "sigmoid2" = 2*sigmoid (megatron-core).
    hc_write: str = "linear"
    # H_res: HC has no projection.  "none" = raw logits; "doubly_stochastic"
    # keeps the Sinkhorn projection available for a one-switch comparison.
    mhc_manifold: str = "none"
    mhc_sinkhorn_iterations: int = 10
    mhc_sinkhorn_eps: float = 1e-6
    mhc_init_gating_factor: float = 0.01
    mhc_compute_h_eps: float = 1e-6

    # --- n-stream -> 1-stream at the chunk exit ----------------------------
    # "mean" (megatron-core output_contract) | "learned" (official DSv4
    # learned_output_contract) | "sum" (the HC paper's Algorithm 1).
    hc_output_contract: str = "mean"
    hc_output_contract_eps: float = 1e-6

    # ------------------------------------------------------------------ #
    # presets
    # ------------------------------------------------------------------ #
    @classmethod
    def identity_preset(cls, **overrides) -> dict:
        """Kwargs for the exact-identity (paper Eq. 14) configuration."""
        preset = dict(
            hc_init="identity",
            hc_dynamic=True,
            hc_read="linear",  # bit-exact identity (HF-only read)
            hc_write="linear",
            mhc_manifold="none",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset

    @classmethod
    def paper_preset(cls, **overrides) -> dict:
        """Kwargs for the paper's HC exactly as published.

        Static HC (Algorithm 2 / Eq. 14) is the identity-initialised
        configuration above; the paper's dynamic variant (Eqs. 10-13) adds
        ``tanh(theta x)`` branches scaled by a small learnable factor, which is
        what ``hc_dynamic=True`` + ``hc_init="official"`` reproduces here (the
        factor initialised at ``mhc_init_gating_factor``, Megatron's 0.01, and
        not at the paper's "small value" which is unspecified).
        """
        return cls.identity_preset(**overrides)

    @classmethod
    def megatron_preset(cls, **overrides) -> dict:
        """Kwargs for megatron-core's activation choices *without* the Sinkhorn.

        Identical to :meth:`~Qwen3MHCConfig.official_preset` except that H_res
        stays unprojected, which is exactly the "HC vs mHC" ablation: the two
        model files then differ in ``mhc_manifold`` alone.
        """
        preset = dict(
            hc_init="official",
            hc_dynamic=True,
            hc_read="sigmoid",
            hc_write="sigmoid2",
            mhc_manifold="none",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset

    @classmethod
    def lite_preset(cls, **overrides) -> dict:
        """Kwargs for static HC (no input-dependent coefficients).

        The paper's SHC (Section 2.1) costs ``n(n+2)`` parameters per sublayer
        and no extra memory traffic; ``hc_dynamic=False`` reproduces it, keeping
        the identity initialisation.
        """
        preset = dict(
            hc_init="identity",
            hc_dynamic=False,
            hc_read="linear",  # bit-exact identity (HF-only read)
            hc_write="linear",
            mhc_manifold="none",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset


#: the fields this variant adds on top of ``Qwen3Config`` (read by ``to_dict``)
_EXTRA_CONFIG_FIELDS = tuple(Qwen3HCConfig.__annotations__)

# ``model_type = "qwen3_hc"`` is resolved through the HF config registry, so
# importing this module is enough to make ``AutoConfig.from_pretrained`` work.
# ``modeling_qwen3_hc`` registers the model classes and repeats this call.
try:
    AutoConfig.register("qwen3_hc", Qwen3HCConfig)
except ValueError:
    pass

# Marks this module as ''checkpoint-local code'': ``save_pretrained`` then writes
# ``auto_map`` into ``config.json`` and copies this file next to the weights, so
# ``AutoConfig.from_pretrained(path, trust_remote_code=True)`` works in a fresh
# process that has never imported this package (verl's MegatronWorker, vLLM).
try:
    Qwen3HCConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):  # older/newer transformers without the API
    pass
