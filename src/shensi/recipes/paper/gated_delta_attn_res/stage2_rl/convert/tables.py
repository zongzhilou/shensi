"""The HF <-> Megatron tensor correspondence: one table per variant.

This module is the single source of truth for *names*.  Nothing else in the
converter hard-codes a tensor name, so the audit in :mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.audit`
can compare the table against a real ``state_dict()`` of a real Megatron model and
a real HF model and answer the only question that matters: **is every tensor on
both sides accounted for?**

The table is *generated* from the HF config, not pasted from a dump, because
several shapes depend on knobs (low-rank vs full-rank projections, presence of a
null source, learned output contraction, ...).  A generated table can be wrong;
the audit is what makes that visible.

Vocabulary
----------
``mcore``
    The Megatron name: ``decoder.layers.3.mlp_attn_res.gate_proj.weight``.
``hf``
    The HuggingFace name(s): ``model.layers.3.mlp_attn_res.gate_proj.weight``.
    A tuple, because two HF tensors can fuse into one Megatron tensor (``q/k/v``
    -> ``linear_qkv``, ``gate/up`` -> ``linear_fc1``).
``kind`` -- how the values travel (see :data:`KINDS`):

==============  ==========================================================
``copy``        one HF tensor, same values, different name
``qkv``         three HF tensors -> one fused ``linear_qkv`` (head-grouped)
``fc1``         ``gate_proj``/``up_proj`` -> one fused ``linear_fc1``
``vocab``       one HF tensor, *truncated* to the HF vocab size on the way back
``synth``       Megatron-only: HF has no such tensor.  The value is a constant
                chosen so that the Megatron module reproduces the HF module's
                arithmetic.  Dropped (and reported) on the way back.
==============  ==========================================================

Why ``synth`` exists at all
---------------------------
The Megatron port of AR/DAR adds a zero-initialised scalar gate on the read
(``read_scale``, `FlagScale/flagscale/models/megatron/depth/depth_connection.py`)
so that step 0 is bit-exactly plain Qwen3 -- see ``BASELINES_MEGATRON.md`` §4
item 5.  The HF implementation has no such tensor: its read *replaces* (AR) or
*adds* (DAR) unconditionally, which is the reference operator and corresponds to
``read_scale = 1`` in the Megatron parameterisation.  A converted checkpoint is
supposed to *behave like the HF model*, so the default is ``1.0``; pass
``SynthesisPolicy(ar_dar_read_scale=0.0)`` to land on the identity anchor
instead (the point the Megatron-only training runs start from).

Nothing is invented silently: every ``synth`` entry has a ``constant`` and a
``note``, and the conversion report lists them with their counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

__all__ = [
    "KINDS",
    "Pair",
    "SynthesisPolicy",
    "Table",
    "build_table",
    "ALL_VARIANTS",
]

KINDS = ("copy", "qkv", "fc1", "vocab", "synth")

Kind = Literal["copy", "qkv", "fc1", "vocab", "synth"]

#: Megatron's GPTModel attribute names / HF's Qwen3 prefix.  Kept as constants so
#: the rewritten names are greppable in one place.
#:
#: The three "ragged" roots are worth spelling out, because ``GPTModel`` does not
#: put everything under one prefix and getting them wrong costs a full audit run:
#: the transformer stack is ``decoder.layers.*`` and the final norm is
#: ``decoder.final_layernorm.weight``, but the embedding is
#: ``embedding.word_embeddings.weight`` and the head is ``output_layer.weight``
#: -- no ``decoder.`` in either.  (mbridge's own tables, e.g.
#: ``models/qwen2.py:_DIRECT_MAPPING``, use exactly these spellings.)
MCORE_ROOT = "decoder"
HF_ROOT = "model"
MCORE_EMBEDDING = "embedding.word_embeddings.weight"
MCORE_FINAL_NORM = f"{MCORE_ROOT}.final_layernorm.weight"
MCORE_OUTPUT = "output_layer.weight"

ALL_VARIANTS = ("ar", "dar", "gdar", "denseformer", "hc", "mhc", "mudd")


@dataclass(frozen=True)
class SynthesisPolicy:
    """Values written into tensors that exist only on the Megatron side.

    Args:
        ar_dar_read_scale: value for the AR/DAR read gate.  ``1.0`` (default)
            makes the Megatron module reproduce the HF/reference operator
            (``prefix + 1 * selected`` == the reference's ``prefix + selected``,
            and ``prefix + 1 * (routed - prefix)`` == the reference's ``routed``
            up to the float association of the subtraction).  ``0.0`` is the
            identity anchor the FlagScale/Megatron tiny runs train from.
        zero: value for the remaining Megatron-only tensors, all of which are
            biases that HF simply does not have (low-rank ``q/k`` ``up.bias``).
            Zero is the only value that makes the factored and composed forms
            agree.
    """

    ar_dar_read_scale: float = 1.0
    zero: float = 0.0


@dataclass(frozen=True)
class Pair:
    """One row of the conversion table."""

    mcore: str
    hf: tuple[str, ...] = ()
    kind: Kind = "copy"
    #: for ``synth``: the value to write (``None`` means "look up the policy")
    constant: float | None = None
    #: for ``synth``: which :class:`SynthesisPolicy` field decides the value
    policy_field: str | None = None
    #: free text, printed by the audit when the row is interesting
    note: str = ""

    @property
    def synthesized(self) -> bool:
        return self.kind == "synth"

    def synthesized_value(self, policy: SynthesisPolicy) -> float:
        if self.policy_field is not None:
            return float(getattr(policy, self.policy_field))
        if self.constant is not None:
            return float(self.constant)
        return float(policy.zero)


@dataclass
class Table:
    """The full name correspondence for one model configuration."""

    variant: str
    pairs: list[Pair] = field(default_factory=list)
    #: configuration facts the table depends on, for the report
    facts: dict = field(default_factory=dict)
    #: configurations the Megatron side cannot represent (reported, not hidden)
    unsupported: list[str] = field(default_factory=list)

    # -- lookups -----------------------------------------------------------

    def by_mcore(self) -> dict[str, Pair]:
        """``{mcore name: Pair}``; raises on a duplicate, which would be a bug."""
        out: dict[str, Pair] = {}
        for pair in self.pairs:
            if pair.mcore in out:
                raise ValueError(f"duplicate mcore name in table: {pair.mcore}")
            out[pair.mcore] = pair
        return out

    def by_hf(self) -> dict[str, tuple[Pair, int]]:
        """``{hf name: (Pair, index within the pair)}``; a name may appear once."""
        out: dict[str, tuple[Pair, int]] = {}
        for pair in self.pairs:
            for index, name in enumerate(pair.hf):
                if name in out:
                    raise ValueError(f"duplicate HF name in table: {name}")
                out[name] = (pair, index)
        return out

    @property
    def mcore_names(self) -> list[str]:
        return [p.mcore for p in self.pairs]

    @property
    def hf_names(self) -> list[str]:
        return [name for pair in self.pairs for name in pair.hf]

    @property
    def synthesized(self) -> list[Pair]:
        return [p for p in self.pairs if p.synthesized]

    def report(self) -> dict:
        counts: dict[str, int] = {}
        for pair in self.pairs:
            counts[pair.kind] = counts.get(pair.kind, 0) + 1
        return {
            "variant": self.variant,
            "rows": len(self.pairs),
            "kinds": counts,
            "mcore_tensors": len(self.mcore_names),
            "hf_tensors": len(self.hf_names),
            "synthesized": {p.mcore: p.constant for p in self.synthesized},
            "facts": self.facts,
            "unsupported": self.unsupported,
        }


# ---------------------------------------------------------------------------
# the backbone, identical for every variant
# ---------------------------------------------------------------------------


def _backbone_pairs(num_layers: int) -> Iterator[Pair]:
    """Stock Qwen3, in the *local* (non-TransformerEngine) layout.

    The two norms are the reason mbridge alone cannot load our checkpoints: its
    tables assume the TransformerEngine fused layout, where the input norm lives
    inside ``linear_qkv.layer_norm_weight`` and the pre-MLP norm inside
    ``linear_fc1.layer_norm_weight``.  Our specs build the local submodules, so
    both are separate modules here.  See :mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.mbridge_patch`.
    """
    yield Pair(MCORE_EMBEDDING, (f"{HF_ROOT}.embed_tokens.weight",), "vocab")
    yield Pair(MCORE_FINAL_NORM, (f"{HF_ROOT}.norm.weight",))
    yield Pair(MCORE_OUTPUT, ("lm_head.weight",), "vocab")
    for layer in range(num_layers):
        m = f"{MCORE_ROOT}.layers.{layer}"
        h = f"{HF_ROOT}.layers.{layer}"
        yield Pair(
            f"{m}.self_attention.linear_qkv.weight",
            (f"{h}.self_attn.q_proj.weight", f"{h}.self_attn.k_proj.weight", f"{h}.self_attn.v_proj.weight"),
            "qkv",
        )
        yield Pair(f"{m}.self_attention.linear_proj.weight", (f"{h}.self_attn.o_proj.weight",))
        yield Pair(f"{m}.self_attention.q_layernorm.weight", (f"{h}.self_attn.q_norm.weight",))
        yield Pair(f"{m}.self_attention.k_layernorm.weight", (f"{h}.self_attn.k_norm.weight",))
        yield Pair(
            f"{m}.mlp.linear_fc1.weight",
            (f"{h}.mlp.gate_proj.weight", f"{h}.mlp.up_proj.weight"),
            "fc1",
        )
        yield Pair(f"{m}.mlp.linear_fc2.weight", (f"{h}.mlp.down_proj.weight",))
        yield Pair(f"{m}.input_layernorm.weight", (f"{h}.input_layernorm.weight",))
        yield Pair(f"{m}.pre_mlp_layernorm.weight", (f"{h}.post_attention_layernorm.weight",))


# ---------------------------------------------------------------------------
# GDAR: gated delta attention residual
# ---------------------------------------------------------------------------


def _gdar_pairs(config, num_layers: int) -> Iterator[Pair]:
    """One ``AttentionResidual`` per sublayer, plus the final read-only module.

    Full rank (``attn_res_*_rank is None``, the default on *both* sides since the
    2026-09-29 alignment) makes ``gate_proj`` a single ``nn.Linear`` -- weight
    and bias -- and ``q_proj``/``k_proj`` a bare ``nn.Parameter``: pure renames.
    Low rank is a pair of ``nn.Linear``s that the two frameworks index
    differently (HF ``.0``/``.1`` ``Sequential`` indices, Megatron ``.down``/
    ``.up`` names), and it is the one place where the parameterisations still
    differ: FlagScale's ``AttentionResidual._make_qk`` calls ``LowRankLinear``
    with its defaults (``down_bias=False, out_bias=True``) while HF's ``_make_qk``
    builds two bias-free linears, so Megatron has a ``q_proj.up.bias`` /
    ``k_proj.up.bias`` with no HF counterpart.  Those are synthesised as zero,
    which is exactly the value that makes ``up(down(x))`` equal HF's
    ``F.linear(x, up.weight @ down.weight)``.
    """
    gate_rank = getattr(config, "attn_res_gate_rank", None)
    q_rank = getattr(config, "attn_res_q_rank", None)
    k_rank = getattr(config, "attn_res_k_rank", None)
    ladder = int(getattr(config, "attn_res_decay_ladder", 0) or 0)
    output_route = bool(getattr(config, "attn_res_output_route", True))
    # ``"deviation"`` replaces the reference's sigmoid gates with
    # ``1 + zero-init deviation``, and *both* sides carry three scalars for it
    # (HF `AttentionResidual.__init__`, Megatron `AttentionResidual.__init__`).
    # Missing rows here cost a whole variant: with the default ``"sigmoid"`` the
    # tensors do not exist at all, so a table that forgot them looks perfect --
    # which is why one of the audit profiles has to use the deviation form.
    deviation = getattr(config, "attn_res_gate_param", "sigmoid") == "deviation"

    def module(m: str, h: str, writer: bool = True) -> Iterator[Pair]:
        """The rows of one ``AttentionResidual``.

        ``writer=False`` is the *output* module, which is read-only: it has a
        ``read_scale`` and a query projection, and **no** gate, no key projection
        and no write path.  Both sides agree on that (HF's
        ``model.output_attn_res_module`` holds ``q_proj`` + ``read_scale``, and the
        Megatron module holds ``q_proj`` + ``read_scale``); emitting a ``k_proj``
        row here is not harmless, because a table row whose source does not exist
        is silently skipped by the converter -- which is why the audit now checks
        ``report.missing_source`` as well.
        """
        if writer:
            if gate_rank is None:
                yield Pair(f"{m}.gate_proj.weight", (f"{h}.gate_proj.weight",))
                yield Pair(f"{m}.gate_proj.bias", (f"{h}.gate_proj.bias",))
            else:
                # both halves carry a bias on both sides (HF
                # `_make_proj(bias=True)` -> Sequential(Linear(bias=True),
                # Linear(bias=True)); Megatron `_make_proj(..., bias=True)` ->
                # `LowRankLinear(down_bias=True, out_bias=True)`), so these four
                # rows are plain renames.
                yield Pair(f"{m}.gate_proj.down.weight", (f"{h}.gate_proj.0.weight",))
                yield Pair(f"{m}.gate_proj.down.bias", (f"{h}.gate_proj.0.bias",))
                yield Pair(f"{m}.gate_proj.up.weight", (f"{h}.gate_proj.1.weight",))
                yield Pair(f"{m}.gate_proj.up.bias", (f"{h}.gate_proj.1.bias",))
        for name, rank in (("q_proj", q_rank), ("k_proj", k_rank)) if writer else (("q_proj", q_rank),):
            if rank is None:
                yield Pair(f"{m}.{name}", (f"{h}.{name}",))
            else:
                yield Pair(f"{m}.{name}.down.weight", (f"{h}.{name}.0.weight",))
                yield Pair(f"{m}.{name}.up.weight", (f"{h}.{name}.1.weight",))
                # The one place the two low-rank parameterisations still differ:
                # HF's `_make_qk` builds `Sequential(Linear(..., bias=False),
                # Linear(..., bias=False))`, while Megatron's `_make_qk` calls
                # `LowRankLinear` with its defaults (`out_bias=True`), so the port
                # has an `up.bias` that the reference simply does not have.  Zero is
                # the only value that makes the factored and the composed forms
                # agree (`up(down(x))` == `F.linear(x, W2 @ W1)`), and it is what
                # `SynthesisPolicy.zero` supplies.  The model pins this bias at zero
                # (``requires_grad_(False)`` in ``_make_qk``), so a trained Megatron
                # checkpoint cannot drift away from the value synthesised here.  (The 2026-09-29 alignment removed
                # exactly this asymmetry from `q`/`k` in the *full-rank* case, where
                # both sides are a bare parameter.)
                yield Pair(
                    f"{m}.{name}.up.bias",
                    (),
                    "synth",
                    note="LowRankLinear defaults to out_bias=True; HF's _make_qk uses bias=False",
                )
        if writer and ladder > 1:
            yield Pair(f"{m}.decay_tau", (f"{h}.decay_tau",))
        if writer and deviation:
            for scale in ("decay_scale", "erase_scale", "write_scale"):
                yield Pair(f"{m}.{scale}", (f"{h}.{scale}",))
        yield Pair(f"{m}.read_scale", (f"{h}.read_scale",))

    for layer in range(num_layers):
        for sub in ("self_attention", "mlp"):
            yield from module(
                f"{MCORE_ROOT}.layers.{layer}.{sub}_attn_res",
                f"{HF_ROOT}.layers.{layer}.{sub}_attn_res",
            )
    if output_route:
        last = num_layers - 1
        yield from module(
            f"{MCORE_ROOT}.layers.{last}.output_attn_res",
            f"{HF_ROOT}.output_attn_res_module",
            writer=False,
        )


# ---------------------------------------------------------------------------
# AR / DAR: additive and delta routers (a weighted RMSNorm + a 1-D projection)
# ---------------------------------------------------------------------------


def _router_pairs(
    m: str,
    h_proj: str,
    h_norm: str,
    h_null: str | None,
    policy: SynthesisPolicy,
) -> Iterator[Pair]:
    yield Pair(f"{m}.proj.weight", (h_proj,))
    yield Pair(f"{m}.norm.weight", (h_norm,))
    if h_null is not None:
        yield Pair(f"{m}.null_source", (h_null,))
    yield Pair(
        f"{m}.read_scale",
        (),
        "synth",
        policy_field="ar_dar_read_scale",
        note=(
            "Megatron-only identity gate (depth_layer.py); 1.0 == the HF reference "
            "operator, 0.0 == the identity anchor the tiny runs start from"
        ),
    )


def _ar_dar_pairs(config, num_layers: int, variant: str, policy: SynthesisPolicy) -> Iterator[Pair]:
    null = bool(getattr(config, "attn_res_use_null_source", False)) and variant == "dar"
    output_route = bool(getattr(config, "attn_res_output_route", True))
    for layer in range(num_layers):
        h = f"{HF_ROOT}.layers.{layer}"
        for sub in ("self_attention", "mlp"):
            yield from _router_pairs(
                f"{MCORE_ROOT}.layers.{layer}.{sub}_attn_res",
                f"{h}.{sub}_res_proj.weight",
                f"{h}.{sub}_res_norm.weight",
                f"{h}.{sub}_null_source" if null else None,
                policy,
            )
    if output_route:
        last = num_layers - 1
        # HF has no null source on the *output* module; Megatron builds one
        # whenever `use_null_source` is on (depth_layer.py `_build_connections`
        # passes the same `extra` to all three routers), so it is a `synth` row
        # with an explicit note rather than a silent mismatch.
        yield from _router_pairs(
            f"{MCORE_ROOT}.layers.{last}.output_attn_res",
            f"{HF_ROOT}.output_attn_res_proj.weight",
            f"{HF_ROOT}.output_attn_res_norm.weight",
            None,
            policy,
        )
        if null:
            yield Pair(
                f"{MCORE_ROOT}.layers.{last}.output_attn_res.null_source",
                (),
                "synth",
                note="Megatron's output router inherits use_null_source; HF's output read has no null source",
            )


# ---------------------------------------------------------------------------
# DenseFormer: a learned depth-weighted average
# ---------------------------------------------------------------------------


def _denseformer_pairs(config, num_layers: int) -> Iterator[Pair]:
    mode = getattr(config, "attn_res_dwa_param", "deviation")
    weight = "alpha" if mode == "official" else "alpha_delta"
    for layer in range(num_layers):
        yield Pair(
            f"{MCORE_ROOT}.layers.{layer}.block_dwa.{weight}",
            (f"{HF_ROOT}.layers.{layer}.dwa.{weight}",),
        )


# ---------------------------------------------------------------------------
# HC / mHC: the official hyper-connection module
# ---------------------------------------------------------------------------


def _hc_pairs(config, num_layers: int) -> Iterator[Pair]:
    # the module attribute names are identical on both sides (both are the
    # official `HyperConnectionModule` / its port), only the prefix differs
    for layer in range(num_layers):
        for sub in ("self_attention", "mlp"):
            for tensor in ("alpha_pre", "alpha_post", "alpha_res", "bias", "mapping_proj.weight"):
                yield Pair(
                    f"{MCORE_ROOT}.layers.{layer}.{sub}_hyper_connection.{tensor}",
                    (f"{HF_ROOT}.layers.{layer}.{sub}_hyper_connection.{tensor}",),
                )
    # block-level contraction (only for the "learned" mode)
    if getattr(config, "hc_output_contract", "mean") == "learned":
        yield Pair(
            f"{MCORE_ROOT}.layers.{num_layers - 1}.head_fn",
            (f"{HF_ROOT}.hc_head_fn",),
            note="one head per chunk exit; HF keeps a single model-level one",
        )
        yield Pair(f"{MCORE_ROOT}.layers.{num_layers - 1}.head_base", (f"{HF_ROOT}.hc_head_base",))
        yield Pair(f"{MCORE_ROOT}.layers.{num_layers - 1}.head_scale", (f"{HF_ROOT}.hc_head_scale",))


# ---------------------------------------------------------------------------
# MUDD: multiway dynamic dense connections (single-stream port)
# ---------------------------------------------------------------------------


def _mudd_pairs(config, num_layers: int) -> Iterator[Pair]:
    mode = getattr(config, "mudd_param", "deviation")
    prior = {"official": "prior", "random": "prior"}.get(mode, "prior_delta")
    pre_norm = bool(getattr(config, "mudd_pre_norm", False))
    post_norm = bool(getattr(config, "mudd_post_norm", False))
    for layer in range(num_layers):
        m = f"{MCORE_ROOT}.layers.{layer}.dense_conn"
        h = f"{HF_ROOT}.layers.{layer}.dense_conn"
        yield Pair(f"{m}.{prior}", (f"{h}.{prior}",))
        yield Pair(f"{m}.w1.weight", (f"{h}.w1.weight",))
        yield Pair(f"{m}.w2.weight", (f"{h}.w2.weight",))
        if pre_norm:
            yield Pair(f"{m}.norm.weight", (f"{h}.norm.weight",))
        if post_norm:
            yield Pair(
                f"{MCORE_ROOT}.layers.{layer}.dense_post_norm.weight",
                (f"{HF_ROOT}.layers.{layer}.dense_post_norm.weight",),
            )


# ---------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------

_BUILDERS = {
    "gdar": _gdar_pairs,
    "ar": lambda config, n, **_kw: _ar_dar_pairs(config, n, "ar", _kw["policy"]),
    "dar": lambda config, n, **_kw: _ar_dar_pairs(config, n, "dar", _kw["policy"]),
    "denseformer": _denseformer_pairs,
    "hc": _hc_pairs,
    "mhc": _hc_pairs,
    "mudd": _mudd_pairs,
}


def build_table(
    variant: str,
    config,
    num_layers: int,
    *,
    policy: SynthesisPolicy | None = None,
    connection: bool | None = None,
) -> Table:
    """Build the table for one variant and one HF configuration.

    Args:
        variant: one of :data:`ALL_VARIANTS`.
        config: the HF config object (read with ``getattr`` and defaults, so a
            plain namespace works too).
        num_layers: ``num_hidden_layers``.
        policy: what to write into Megatron-only tensors.
        connection: ``None`` (default) reads ``attn_res_block_size`` off the
            config; ``False`` forces the backbone-only table (the connection is
            off); ``True`` forces the connection rows even if the config says
            off, which is how the audit catches a config that lies.

    Returns:
        A :class:`Table`.  ``table.unsupported`` names the configurations whose
        Megatron side is missing pieces -- reported, never silently dropped.
    """
    if variant not in _BUILDERS:
        raise KeyError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
    policy = policy or SynthesisPolicy()
    if connection is None:
        connection = getattr(config, "attn_res_block_size", None) is not None

    table = Table(variant=variant)
    table.pairs.extend(_backbone_pairs(num_layers))
    if connection:
        builder = _BUILDERS[variant]
        if variant in ("ar", "dar"):
            table.pairs.extend(builder(config, num_layers, policy=policy))
        else:
            table.pairs.extend(builder(config, num_layers))

    # ---- knob combinations the Megatron port cannot represent -------------
    if connection:
        if variant == "mudd":
            ways = int(getattr(config, "mudd_num_ways", 4) or 4)
            if ways > 1:
                table.unsupported.append(
                    f"mudd_num_ways={ways}: the Megatron port is single-stream only "
                    f"(depth_connection.py:MultiwayDynamicDense hard-codes num_ways=1), so HF's "
                    f"per-way prior/w2 rows and input_layernorm_q/k/v have no destination"
                )
            if getattr(config, "mudd_pre_norm", False):
                table.unsupported.append(
                    "mudd_pre_norm=True: HF adds state_norm on the state list, which the "
                    "Megatron port does not implement"
                )
            if getattr(config, "mudd_ffn_depth_scaling", False):
                table.unsupported.append("mudd_ffn_depth_scaling=True: HF-only FFN re-allocation")
        if variant in ("hc", "mhc"):
            read = getattr(config, "hc_read", "linear")
            if read not in ("simplex", "sigmoid"):
                table.unsupported.append(
                    f"hc_read={read!r}: the Megatron HcConfig only accepts 'simplex'/'sigmoid', "
                    f"so a model built from this config cannot be created at all"
                )
            if getattr(config, "hc_output_contract", "mean") == "learned" and int(
                getattr(config, "attn_res_block_size", 1) or 1
            ) < num_layers:
                table.unsupported.append(
                    "hc_output_contract='learned' with attn_res_block_size < num_hidden_layers: "
                    "Megatron keeps one learned head *per chunk exit*, HF keeps a single "
                    "model-level head"
                )
        if variant == "denseformer":
            if int(getattr(config, "attn_res_dwa_dilation", 1) or 1) > 1:
                table.unsupported.append(
                    "attn_res_dwa_dilation>1: the source *count* per event differs between the "
                    "reference (keeps every block) and the port (keeps event snapshots only)"
                )
        if variant == "dar" and getattr(config, "attn_res_use_null_source", False):
            table.unsupported.append(
                "attn_res_use_null_source=True: the Megatron output router carries a null source "
                "that HF's output read does not have, so the final read is not equivalent"
            )
        if variant == "ar" and int(getattr(config, "attn_res_block_size", 1) or 1) == 1:
            # Measured, not guessed (see code/VERL_CONVERTER.md): with
            # `depth_block_size=1` the port takes its *per-sublayer* source mode
            # (`depth_layer.py: per_sublayer_sources = variant in ("ar","dar") and
            # period == 1`) and appends three sources per layer -- the incoming
            # stream, the attention output and the MLP output -- while HF's AR
            # appends exactly one, the layer input at the block boundary.  The
            # source *lists* therefore differ in content, not just in name, and no
            # knob of the spec controls it.  At block size > 1 the port appends at
            # boundaries only and the two agree to float noise (2.4e-07 on the
            # audit's `ar_block2`).
            table.unsupported.append(
                "attn_res_block_size=1: the Megatron port takes its per-sublayer source mode at "
                "block size 1 (three sources per layer) where HF's AR appends one per layer, so the "
                "two models are not equivalent after conversion; AR is exact at block size > 1"
            )
        if variant in ("hc", "mhc"):
            # The read is a real knob on both sides; the *write* is not.  HF's
            # `hc_write` has two modes and HC's own default is "linear" (the
            # paper's unconstrained static mixing), while the Megatron port
            # implements the mHC-style `sigmoid * 2` only.
            write = getattr(config, "hc_write", "sigmoid2")
            if write != "sigmoid2":
                table.unsupported.append(
                    f"hc_write={write!r}: the Megatron port implements the write operator as "
                    f"'sigmoid2' only (the mHC default), so an HC checkpoint whose write is "
                    f"{write!r} converts tensor-for-tensor but not function-for-function"
                )
    elif getattr(config, "attn_res_block_size", None) is not None:
        table.unsupported.append("connection forced off although attn_res_block_size is set")

    from dataclasses import asdict

    table.facts = {
        "num_layers": num_layers,
        "connection": bool(connection),
        "block_size": getattr(config, "attn_res_block_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None) or (
            getattr(config, "hidden_size", 0) // max(1, getattr(config, "num_attention_heads", 1) or 1)
        ),
        "vocab_size": getattr(config, "vocab_size", None),
        "policy": asdict(policy),
    }
    return table
