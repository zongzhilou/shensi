"""Configuration for Qwen3 + Gated Delta Attention Residuals (GDAR).

Mirrors the structure of ``moonshotai/Kimi-K3:configuration_kimi_k3.py``: the
config lives in its own file and ``modeling_qwen3_gdar.py`` imports it.

Reference for the connection: ``zongzhilou/transformers@shensi`` ->
``ShensiAttentionResidual``.

Defaults reproduce that repository exactly.  The redo plan's two implementation
fixes are opt-in::

    Qwen3GDARConfig(..., attn_res_gate_init="identity",  # (b, -b, b) biases -> GDAR(0) ~ DAR
                          attn_res_gate_rank=64,          # low-rank D -> r -> 3D gates
                          attn_res_q_rank=64,             # low-rank routing query
                          attn_res_k_rank=64)             # low-rank erase direction

Ablation switches (E2-E6 of the redo plan) all live on this config, so an arm is
one `kwargs` dict and nothing in the model file changes between arms:

* **E2** source x gate 2x2 -- ``gated_ar_preset()`` is the fourth cell.  The two
  axes are *which depths the read routes over* (`attn_res_block_size`: ``> 1`` =
  cumulative stream snapshots, the AR source; ``== 1`` = per-sublayer outputs, the
  DAR/GDAR source) and *whether the three gates exist*
  (`attn_res_gate_param` / `attn_res_gate_channels`).  `attn_res_address` is a
  third, independent knob: it picks the **erase direction** (``"state"`` = from the
  accumulated stream, which is also the AR-flavoured choice and the reference
  default, ``"delta"`` = from the content being written).  See the class docstring
  of :meth:`Qwen3GDARConfig.gated_ar_preset` for the verified mapping.
* **E3** gate structure -- ``attn_res_gate_channels``.
* **E4** gate initialisation -- ``attn_res_gate_init`` + ``attn_res_gate_init_bias``.
* **E5** rank -- ``attn_res_gate_rank`` / ``attn_res_q_rank`` / ``attn_res_k_rank``
  (``None`` = full rank, the reference).
* **E6** block size -- ``attn_res_block_size``.
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

__all__ = ["GATE_CHANNELS", "GATED_AR_SOURCE_BLOCK", "Qwen3GDARConfig"]

#: accepted values of ``attn_res_gate_channels`` -- the E3 gate-structure ablation.
#: ``"dew"`` is the reference: all three gates (decay, erase, write) as ``D -> 3D``.
#: A subset keeps only the listed gates and pins the others to their identity value
#: (decay 1, erase 0, write 1).  ``"scalar"`` keeps the write gate only and collapses
#: it over channels, and ``"none"`` removes the gate entirely (the update degenerates
#: to DAR -- the "no gate" row of E3).  Implemented in ``modeling_qwen3_gdar.py``:
#: ``_apply_gate_channels``.
GATE_CHANNELS = ("dew", "d", "e", "w", "de", "dw", "ew", "scalar", "none")

#: block granularity that makes the read contexts *cumulative stream snapshots*
#: (the AR source) instead of per-sublayer deltas -- the source axis of E2/E6.
GATED_AR_SOURCE_BLOCK = 4


@_strict_config
class Qwen3GDARConfig(Qwen3Config):
    """Qwen3Config + Gated Delta Attention Residuals knobs."""

    model_type = "qwen3_gdar"

    #: ``AutoConfig`` / ``AutoModelForCausalLM`` resolve through this when the
    #: checkpoint is loaded with ``trust_remote_code=True`` -- which is how a
    #: fresh process (verl's, vLLM's) gets the definition: these two files are
    #: copied next to the weights by ``save_pretrained``.
    auto_map = {
        "AutoConfig": "configuration_qwen3_gdar.Qwen3GDARConfig",
        "AutoModel": "modeling_qwen3_gdar.Qwen3GDARModel",
        "AutoModelForCausalLM": "modeling_qwen3_gdar.Qwen3GDARForCausalLM",
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

    # "block" source granularity, same switch as the other variants
    attn_res_block_size: int | None = None
    attn_res_output_route: bool = True

    # gates: None + "paper" reproduces shensi; 64 + "identity" is the redo's fix
    attn_res_gate_rank: int | None = None
    #: gate initialisation (E4).  ``b = attn_res_gate_init_bias``.
    #:
    #: ``"identity"``: biases ``(b, -b, b)`` -- decay and write open, erase closed --
    #: so step 0 is at the DAR update.  With the default ``b = 4`` the *bias* value is
    #: ``sigmoid(4) = 0.982`` (``b = 8``: ``0.99966``); the realised gates differ from
    #: it by the random projection weights, which is why the measured deviation from
    #: the DAR update is 3.4e-2 at ``b = 4`` and 6.7e-4 at ``b = 8``
    #: (``test_ablation_switches.py``).  For the *exact* identity use
    #: ``attn_res_gate_param="deviation"`` (the scales start at zero, so the gates are
    #: (1, 0, 1) bit-exactly, whatever ``b`` is).
    #: ``"zero"``: all three biases 0 -> every gate at ``sigmoid(0) = 0.5``, the
    #: "0.5-init" row of E4.  ``"uniform"``: all three biases equal to ``b``, i.e.
    #: gates ``(sigmoid(b),)*3`` -- ``b = 0`` is the 0.5-init row again and
    #: ``b = -20`` is the "0-init" row (``2.1e-9``, i.e. decay = write = 0: the stream
    #: is zeroed by every write).  ``"paper"``: the reference's own ``nn.Linear``
    #: defaults.
    attn_res_gate_init: str = "paper"
    attn_res_gate_init_bias: float = 4.0
    #: which of the three delta-rule gates exist -- the E3 gate-structure ablation.
    #:
    #: ``"dew"`` (default) is the reference and the only value that reaches the gates
    #: untouched; every other value replaces the gates it removes by their identity
    #: constant *after* the projection, so all E3 arms keep the same parameter set
    #: (the removed slices simply receive zero gradient) and the comparison isolates
    #: the gate *structure*.  ``"scalar"`` keeps the write gate only and averages it
    #: over channels, i.e. one learned write value per token instead of one per
    #: channel; ``"none"`` pins ``(decay, erase, write) = (1, 0, 1)`` for every input,
    #: which turns the update into the DAR rule exactly ("no gate" row).  Valid values:
    #: :data:`GATE_CHANNELS`.
    attn_res_gate_channels: str = "dew"

    # --- theory-optimal parameterisation (see modeling_qwen3_gdar.py docstring) ---
    # "sigmoid"   : gate = sigmoid(Linear(state))           -- reproduces shensi
    # "deviation" : gate = 1 + zero-init deviation, so the model starts *exactly*
    #               on the DAR update and can only move away from it
    attn_res_gate_param: str = "sigmoid"
    # Carrier bias of the *write* head.  The identity point is enforced by the
    # deviation scales (init 0), but a carrier that is zero there -- tanh(0) = 0 --
    # would give the write scale a vanishing gradient and the gate could never
    # learn to move.  tanh(-4) = -0.999 keeps the gradient alive while
    # write = 1 + 0 * tanh(-4) = 1 stays exact.
    attn_res_write_carrier_bias: float = -4.0
    # multi-timescale decay ladder (cascade model): >1 gives the channels time
    # constants on a geometric ladder over [1, attn_res_decay_tau_max], which are
    # then *learned*; < 2 turns the multi-timescale decay off (tau == 1)
    attn_res_decay_ladder: int = 0
    attn_res_decay_tau_max: float = 100.0
    # "shensi" (alias "reference"):
    #     updated = decay*h - khat<khat, erase*decay*h> + write*delta
    #     This is the *older* reference rule.  NOTE the naming trap: the
    #     `zongzhilou/transformers@shensi` branch itself now uses the objective rule
    #     below, so `"shensi"` does **not** mean "what the branch does" -- use
    #     `"objective"` for that.
    # "objective": exact minimiser of  J(h') = 1/2||h'-m||^2 + (lambda/2)<khat,h'>^2
    #     (this is what the shensi branch implements today)
    attn_res_update: str = "shensi"

    # --- read side: statistically optimal estimator instead of plain NW ---------
    # multi-head depth read: one softmax per subspace; H=1 reproduces the reference,
    # the ensemble variance (1/H)Var + (1-1/H)Cov makes decorrelated heads strictly
    # better.  Must divide attn_res hidden size.
    attn_res_read_heads: int = 1
    # Softmax_1: p_i = exp(z_i)/(1 + sum_j exp(z_j)).  The leftover mass is a null
    # route: it is the posterior mean of a zero-mean GP with unit noise and the
    # Bayes-optimal abstention rule (Chow 1970), and it removes the need for extreme
    # logits when the right answer is "do not update".
    attn_res_read_null: bool = False
    # whitened (Mahalanobis) scoring: plain softmax is Nadaraya-Watson kernel
    # regression, which double-counts correlated sources; weighting by Sigma^-1
    # gives the minimum-variance (Gauss-Markov) linear read.  "diag" scales each
    # dimension by 1/sqrt(var+ridge), "full" uses the D x D whitening matrix.
    attn_res_read_whiten: str = "off"
    attn_res_read_ridge: float = 1e-3
    # address (erase) direction: "state" reproduces the reference, "delta" derives it
    # from the content being written, "novelty" takes the optimal one -- the part of the
    # new content orthogonal to the retained stream, which maximises the written
    # signal-to-interference ratio in closed form.
    #:
    #: NOTE this knob does **not** select the read contexts: the erase direction is
    #: ``khat = normalize(k_proj(address_src))`` with ``address_src = norm(prefix+delta)``
    #: for "state", the sublayer output for "delta" and its component orthogonal to the
    #: retained stream for "novelty".  ``"state"`` is therefore the *AR-flavoured*
    #: direction (it reads the accumulated stream, as Attention Residuals does) and is
    #: what the reference uses; the E2 cell that pairs it with cumulative sources and
    #: the three gates is :meth:`Qwen3GDARConfig.gated_ar_preset`.
    attn_res_address: str = "state"

    # --- design-ablation knobs (GDAR's own design decisions; see GDAR_ABLATION_DESIGN.md) ---
    # Which tensor drives the three gates.  "state" is the reference and what every run so
    # far used: the normalised (prefix + delta).  "prefix" gives the gates the retained
    # stream only, "delta" the write only (falling back to the state where there is no
    # delta).  This is the ablation of design decision (a).
    attn_res_gate_source: str = "state"
    # Positivity of the decay scale.  decay = exp(-softplus(r) * s_decay * tau) is <= 1
    # exactly when s_decay >= 0, and the raw scale is unconstrained: **measured** to go
    # negative in training (24/24 modules in one E12 run, 39/56 in the 0.6B smoke), which
    # lets the gate amplify the stream.  "free" is what every run so far used; "project"
    # clamps s_decay at 0 in the forward -- identity is preserved bit-exactly (0 clamps to
    # 0) and the paper's bounded-decay claim then holds by construction.
    attn_res_decay_positivity: str = "free"
    # Lower clamp on lambda = mean_c(erase) in the objective update.  -0.5 is our margin
    # inside the strictly convex region (lambda > -1); None removes the clamp -- the
    # ablation that shows why the clamp is there (lambda -> -1 is a pole).
    attn_res_lambda_clamp: float | None = -0.5
    # Space in which the read averages its values: "raw" (ours -- score in whitened space,
    # average the actual values, the GLS/BLUE argument) vs "whitened" (average the whitened
    # values, map the mixture back).  Only meaningful when attn_res_read_whiten != "off".
    attn_res_read_mix: str = "raw"

    @classmethod
    def theory_preset(cls, **overrides):
        """Kwargs for the theory-optimal configuration.

        Exact-identity initialisation (GDAR(0) == DAR bit-exactly, with no
        vanishing gradients), the objective-derived update, and the
        multi-timescale decay ladder -- see ``modeling_qwen3_gdar.py`` and
        ``test_theory.py``.

        >>> cfg = Qwen3GDARConfig(**base, **Qwen3GDARConfig.theory_preset())
        """
        preset = dict(
            # write side
            attn_res_gate_param="deviation",
            attn_res_update="objective",
            attn_res_decay_ladder=64,
            attn_res_address="delta",
            # read side
            attn_res_read_heads=8,
            attn_res_read_null=True,
            attn_res_read_whiten="full",
            # parameterisation
            attn_res_gate_rank=64,
            attn_res_q_rank=64,
            attn_res_k_rank=64,
        )
        preset.update(overrides)
        return preset

    @classmethod
    def gated_ar_preset(cls, **overrides):
        """Kwargs for the E2 fourth cell: **Gated-AR** = AR's connection + GDAR's gates.

        The 2x2 of the plan is *source* x *gates* -- AR (cumulative, ungated), DAR
        (delta, ungated), **Gated-AR** (cumulative, gated) and GDAR (delta, gated) --
        and this preset pins the two axes plus the address that the reference uses:

        * **source = cumulative** -- the read contexts are *cumulative stream
          snapshots*, one per closed block, exactly what
          ``models/modeling_qwen3_ar.py`` routes over (Kimi's ``block_residual``).
          Verified in ``modeling_qwen3_gdar.py``: ``attn_res_block_size > 1`` appends
          ``prefix_sum`` (the stream itself) at ``layer_idx % block_size == 0``;
          ``attn_res_block_size == 1`` appends the *sublayer outputs* instead, i.e.
          deltas, which is the DAR/GDAR source.  So the source axis is
          ``attn_res_block_size``, **not** ``attn_res_address`` (see below).
        * **gates = the three gates** -- ``attn_res_gate_param="deviation"`` (the
          identity construction: ``Gated-AR(0) == AR(0) == DAR`` bit-exactly) with
          ``attn_res_gate_channels="dew"`` (decay, erase, write all present).
        * **address = "state"** -- the erase direction comes from the accumulated
          stream ``norm(prefix + delta)``.  This is the reference's own setting and
          the "AR-flavoured" way to pick the direction to clear, as opposed to
          ``"delta"`` (the DeltaNet-consistent choice that ``theory_preset()`` uses).
          It is *not* what makes the sources cumulative: the sources are set by the
          block size above.  Both halves of the sentence "state address on cumulative
          sources" are what this preset pins.

        The block size defaults to :data:`GATED_AR_SOURCE_BLOCK` and is overridable
        (E6 sweeps it), but overriding it to ``1`` *changes the cell*: the sources
        become per-sublayer deltas, i.e. GDAR.

        >>> cfg = Qwen3GDARConfig(**base, **Qwen3GDARConfig.gated_ar_preset())
        """
        preset = dict(
            attn_res_gate_param="deviation",
            attn_res_gate_channels="dew",
            attn_res_address="state",
            attn_res_block_size=GATED_AR_SOURCE_BLOCK,
        )
        preset.update(overrides)
        return preset

    # routing query / erase-direction projections: None reproduces shensi's full D x D
    attn_res_q_rank: int | None = None
    attn_res_k_rank: int | None = None


#: the fields this variant adds on top of ``Qwen3Config`` (read by ``to_dict``)
_EXTRA_CONFIG_FIELDS = tuple(Qwen3GDARConfig.__annotations__)

# ``model_type = "qwen3_gdar"`` is resolved through the HF config registry, so
# importing this module is enough to make ``AutoConfig.from_pretrained`` work.
# ``modeling_qwen3_gdar`` registers the model classes and repeats this call.
try:
    AutoConfig.register("qwen3_gdar", Qwen3GDARConfig)
except ValueError:
    pass

# Marks this module as ''checkpoint-local code'': ``save_pretrained`` then writes
# ``auto_map`` into ``config.json`` and copies this file next to the weights, so
# ``AutoConfig.from_pretrained(path, trust_remote_code=True)`` works in a fresh
# process that has never imported this package (verl's MegatronWorker, vLLM).
try:
    Qwen3GDARConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):  # older/newer transformers without the API
    pass
