"""Configuration for Qwen3 + Manifold-Constrained Hyper-Connections (mHC).

Mirrors the structure of ``moonshotai/Kimi-K3:configuration_kimi_k3.py``: the
config lives in its own file and ``modeling_qwen3_mhc.py`` imports it.

Reference implementation (ported, see the modeling file header):
``megatron-core 0.18.2`` -> ``megatron/core/transformer/hyper_connection.py``
(``HyperConnectionModule``, ``_sinkhorn_iterations``, ``learned_output_contract``)
+ ``transformer_config.py`` (``num_residual_streams``, ``mhc_sinkhorn_iterations``,
``mhc_init_gating_factor``, ``use_fused_mhc``) + ``transformer_block.py``
(``input_expand`` / ``output_contract`` / ``learned_output_contract``).

Field groups
------------
``attn_res_block_size``
    Same switch as the AR/DAR/GDAR variants: ``None`` -> stock Qwen3; ``N`` ->
    one n-stream residual per *chunk* of ``N`` decoder layers, expanded at the
    chunk entry (``input_expand``: the single stream is replicated n times) and
    contracted at the chunk exit (``output_contract``).  Set it to
    ``num_hidden_layers`` (or more) for the Megatron-equivalent layout, where the
    whole stack shares one n-stream residual.

``hc_*``
    Stream geometry and how the three mappings are parameterised/read out.

``mhc_*``
    Names copied from the Megatron config so the two can be compared field by
    field: ``mhc_manifold`` is the Sinkhorn projection switch, ``hc_init``
    selects the initialisation point (see ``identity_preset``/``official_preset``).
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


__all__ = ["Qwen3MHCConfig"]


@_strict_config
class Qwen3MHCConfig(Qwen3Config):
    """Qwen3Config + Manifold-Constrained Hyper-Connections knobs.

    Defaults are the **identity-anchored mHC** used for the comparison: the
    connection is initialised at the exact PreNorm-residual point
    (``mHC(0) == h + f(h)`` bit-exactly, verified in ``test_baselines_hc.py``),
    so the model starts on top of the standard Qwen3 lower bound and can only
    move away from it.  ``official_preset()`` restores the Megatron
    initialisation (gating factor 0.01, zero biases, xavier projection) and
    ``paper_preset()`` restores the paper's sigmoid read on top of it.
    """

    model_type = "qwen3_mhc"

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` resolve through this when the
    #: checkpoint is loaded with ``trust_remote_code=True`` -- which is how a
    #: fresh process (verl's, vLLM's) gets the definition: these two files are
    #: copied next to the weights by ``save_pretrained``.
    auto_map = {
        "AutoConfig": "configuration_qwen3_mhc.Qwen3MHCConfig",
        "AutoModel": "modeling_qwen3_mhc.Qwen3MHCModel",
        "AutoModelForCausalLM": "modeling_qwen3_mhc.Qwen3MHCForCausalLM",
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
    # expansion rate n: the residual stream is n x C wide (Megatron:
    # num_residual_streams = 4).  n = 1 degenerates to the standard residual.
    hc_num_streams: int = 4

    # --- initialisation ----------------------------------------------------
    # "identity": exact PreNorm residual at step 0 (streams equal, H_pre a
    #             convex combination summing to 1, H_post = 1, H_res = I or the
    #             uniform doubly stochastic matrix -- both give the identity on
    #             identical streams), dynamic branches scaled by alpha = 0.
    # "official": megatron-core defaults (xavier mapping_proj,
    #             alpha = mhc_init_gating_factor, bias = 0 -> H_pre = 0.5*n,
    #             H_post = 1, H_res = uniform via Sinkhorn).
    hc_init: str = "identity"

    # --- parameterisation of the three mappings (Eq. 7-8 of the paper) -----
    # input-dependent coefficients (the n*C -> n^2+2n projection).  False keeps
    # only the static coefficients -- the "lite" variant, see lite_preset().
    hc_dynamic: bool = True
    # H_pre: "simplex" = softmax(logits) (non-negative, sums to 1 -> exact
    #        convex combination, the identity point is exactly 1/n),
    #        "sigmoid" = sigmoid(logits) + eps (the paper's Eq. 8 / Megatron),
    #        "linear"  = logits (unconstrained, the HC paper's A_m).
    hc_read: str = "simplex"
    # H_post: "sigmoid2" = 2*sigmoid(logits) (the paper's Eq. 8 / Megatron),
    #         "linear" = logits (unconstrained, the HC paper's B).
    hc_write: str = "sigmoid2"
    # H_res: "doubly_stochastic" = Sinkhorn-Knopp projection onto the Birkhoff
    #        polytope (mHC), "none" = the raw logits (plain HC).
    mhc_manifold: str = "doubly_stochastic"
    mhc_sinkhorn_iterations: int = 10
    mhc_sinkhorn_eps: float = 1e-6
    # alpha_pre/alpha_post/alpha_res (Eq. 5/7): the learnable scale of the
    # dynamic part; Megatron default 0.01, identity init uses 0.0 so that the
    # dynamic part contributes exactly nothing at step 0 while keeping a
    # non-zero gradient (the projection weights are a non-zero carrier).
    mhc_init_gating_factor: float = 0.01
    # Megatron's _MHC_COMPUTE_H_EPS: floor added to the sigmoid read.
    mhc_compute_h_eps: float = 1e-6

    # --- n-stream -> 1-stream at the chunk exit ----------------------------
    # "mean"    : official output_contract (average of the streams),
    # "learned" : official learned_output_contract (DSv4; hc_head_fn/base/scale),
    # "sum"     : HC paper Algorithm 1 ("sum rows of H^L").
    hc_output_contract: str = "mean"
    hc_output_contract_eps: float = 1e-6

    # ------------------------------------------------------------------ #
    # presets
    # ------------------------------------------------------------------ #
    @classmethod
    def identity_preset(cls, **overrides) -> dict:
        """Kwargs for the exact-identity (lower-bound) configuration.

        >>> cfg = Qwen3MHCConfig(**base, **Qwen3MHCConfig.identity_preset())
        """
        preset = dict(
            hc_init="identity",
            hc_dynamic=True,
            hc_read="simplex",
            hc_write="sigmoid2",
            mhc_manifold="doubly_stochastic",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset

    @classmethod
    def official_preset(cls, **overrides) -> dict:
        """Kwargs for the megatron-core 0.18.2 initialisation/parameterisation.

        ``H_pre = sigmoid(theta x') + 1e-6``, ``H_post = 2 sigmoid(...)``,
        ``H_res = Sinkhorn(theta x')``, ``alpha = 0.01``, zero static biases and
        a xavier ``mapping_proj``.  This is **not** the standard residual at
        step 0 (measured deviation is printed by ``test_baselines_hc.py``).
        """
        preset = dict(
            hc_init="official",
            hc_dynamic=True,
            hc_read="sigmoid",
            hc_write="sigmoid2",
            mhc_manifold="doubly_stochastic",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset

    @classmethod
    def paper_preset(cls, **overrides) -> dict:
        """Kwargs for the paper's literal mappings on the Megatron init.

        Same as :meth:`official_preset` (the paper's Eq. 8 *is* the Megatron
        parameterisation); kept as a separate name so the redo notes can point at
        "the paper" and "the released code" independently.
        """
        return cls.official_preset(**overrides)

    @classmethod
    def lite_preset(cls, **overrides) -> dict:
        """Kwargs for **mHC-lite** (ours, not in the paper): static-only mHC.

        Drops the input-dependent coefficients of Eq. 7 -- the ``nC -> n^2+2n``
        projection, its RMS-normed argument and the ``alpha`` gating factors --
        while keeping the manifold projection of Eq. 8-9 (H_res doubly
        stochastic via Sinkhorn, H_pre/H_post non-negative) and the identity
        initialisation.  What is saved: ``nC(n^2+2n) + 3`` parameters per
        sublayer and, more importantly for wall-clock, the extra
        ``nC x (n^2+2n)`` read/write of the whole hyper hidden matrix that the
        paper's Section 3.2 measures as the dominant memory-access cost.

        .. warning::
           With ``hc_dynamic=False`` the identity point is an *exactly* stable
           stream-permutation-symmetric saddle: ``dL/dH_res`` is exactly 0 there
           (the Sinkhorn projection annihilates the uniform gradient direction --
           measured in ``test_baselines_hc.py`` section 2b), and with static
           coefficients nothing breaks the symmetry, so the mixing never learns
           (measured stream spread 0.0 after 30 steps).  Use ``hc_dynamic=True``
           or ``hc_init="official"`` when the mixing is supposed to train;
           ``lite_preset`` is for measuring the *static* parameterisation, not
           for a competitive run.
        """
        preset = dict(
            hc_init="identity",
            hc_dynamic=False,
            hc_read="simplex",
            hc_write="sigmoid2",
            mhc_manifold="doubly_stochastic",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset


#: the fields this variant adds on top of ``Qwen3Config`` (read by ``to_dict``)
_EXTRA_CONFIG_FIELDS = tuple(Qwen3MHCConfig.__annotations__)

# ``model_type = "qwen3_mhc"`` is resolved through the HF config registry, so
# importing this module is enough to make ``AutoConfig.from_pretrained`` work.
# ``modeling_qwen3_mhc`` registers the model classes and repeats this call.
try:
    AutoConfig.register("qwen3_mhc", Qwen3MHCConfig)
except ValueError:
    pass

# Marks this module as ''checkpoint-local code'': ``save_pretrained`` then writes
# ``auto_map`` into ``config.json`` and copies this file next to the weights, so
# ``AutoConfig.from_pretrained(path, trust_remote_code=True)`` works in a fresh
# process that has never imported this package (verl's MegatronWorker, vLLM).
try:
    Qwen3MHCConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):  # older/newer transformers without the API
    pass
