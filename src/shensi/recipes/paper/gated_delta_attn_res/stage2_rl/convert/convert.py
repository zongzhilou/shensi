"""The conversion functions: a ``state_dict`` on one side, a ``state_dict`` on the other.

Pure tensor work -- no Megatron, no transformers, no mbridge -- so it can run in
either environment (``.venv`` has transformers 5, ``.venv-flagos`` has Megatron)
and so the audit can exercise it without building anything.

Two directions, both driven by :class:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables.Table`:

``hf_to_mcore(hf_state_dict, table, ...)``
    Rename, fuse (``q/k/v`` -> ``linear_qkv``, ``gate/up`` -> ``linear_fc1``), pad
    the vocabulary, and fill the Megatron-only tensors with their policy value.
``mcore_to_hf(mcore_state_dict, table, ...)``
    The inverse: split, truncate the vocabulary, and **drop** the Megatron-only
    tensors (they are reported, not silently forgotten).

Both return ``(state_dict, report)``.  The report is the point: it lists every
tensor of the *input* that no row matched, every tensor of the table that the
input was missing, and every shape disagreement.  A conversion that drops
something says so.

Fusing q/k/v
------------
Megatron's ``linear_qkv`` is *head-grouped*: the rows are laid out as
``[group][q | k | v]`` over ``num_query_groups`` groups, not as ``[all q | all
k | all v]``.  The reshape below is the same one mbridge uses
(``Bridge._weight_to_mcore_format``) and the same one Megatron's own
``ColumnParallelLinear`` expects, which is what makes the two sides agree for
``num_query_groups < num_attention_heads`` (GQA) as well as for the equal case.

Only ``tensor_model_parallel_size == 1`` is supported here; see
:mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.mbridge_patch` for what that means for the verl path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .tables import Pair, Table

__all__ = ["ConversionReport", "hf_to_mcore", "mcore_to_hf", "AttnLayout", "check_shape"]


@dataclass
class AttnLayout:
    """What the q/k/v fusion needs to know about the attention geometry."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config) -> "AttnLayout":
        hidden = int(getattr(config, "hidden_size"))
        heads = int(getattr(config, "num_attention_heads"))
        return cls(
            num_attention_heads=heads,
            num_key_value_heads=int(getattr(config, "num_key_value_heads", heads)),
            head_dim=int(getattr(config, "head_dim", None) or hidden // heads),
        )


@dataclass
class ConversionReport:
    """What the conversion could and could not account for."""

    direction: str
    variant: str
    produced: int = 0
    synthesized: dict[str, float] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    unmapped_source: list[str] = field(default_factory=list)
    missing_source: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing on the *source* side was left unexplained."""
        return not self.unmapped_source and not self.shape_mismatch

    def summary(self) -> str:
        return (
            f"{self.direction} {self.variant}: produced {self.produced}, "
            f"synthesized {len(self.synthesized)}, dropped {len(self.dropped)}, "
            f"unmapped {len(self.unmapped_source)}, missing {len(self.missing_source)}, "
            f"shape-mismatch {len(self.shape_mismatch)}"
        )

    def describe(self, limit: int = 8) -> str:
        lines = [self.summary()]
        for label, items in (
            ("unmapped (in the source, no table row)", self.unmapped_source),
            ("missing (in the table, absent from the source)", self.missing_source),
            ("dropped (Megatron-only, no HF destination)", self.dropped),
            ("synthesized", [f"{k} = {v}" for k, v in self.synthesized.items()]),
            ("shape mismatch", [f"{k}: {a} vs {b}" for k, a, b in self.shape_mismatch]),
        ):
            if not items:
                continue
            lines.append(f"  {label}: {len(items)}")
            lines.extend(f"    {item}" for item in items[:limit])
            if len(items) > limit:
                lines.append(f"    ... and {len(items) - limit} more")
        return "\n".join(lines)


def _fuse_qkv(layout: AttnLayout, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``[group][q|k|v]`` -- Megatron's ``linear_qkv`` row order."""
    hidden = q.shape[-1]
    group_dim = layout.head_dim * layout.num_attention_heads // layout.num_key_value_heads
    groups = q.shape[0] // group_dim
    q = q.reshape(groups, group_dim, hidden)
    k = k.reshape(groups, layout.head_dim, hidden)
    v = v.reshape(groups, layout.head_dim, hidden)
    return torch.cat([q, k, v], dim=1).reshape(-1, hidden).contiguous()


def _split_qkv(layout: AttnLayout, qkv: torch.Tensor) -> list[torch.Tensor]:
    """The inverse of :func:`_fuse_qkv` (mbridge's ``_weight_to_hf_format``)."""
    hidden = qkv.shape[-1]
    group_dim = layout.head_dim * layout.num_attention_heads // layout.num_key_value_heads
    grouped = qkv.reshape(layout.num_key_value_heads, -1, hidden)
    q_len = group_dim
    q = grouped[:, :q_len].reshape(-1, hidden)
    k = grouped[:, q_len : q_len + layout.head_dim].reshape(-1, hidden)
    v = grouped[:, q_len + layout.head_dim :].reshape(-1, hidden)
    return [q.contiguous(), k.contiguous(), v.contiguous()]


def check_shape(name: str, tensor: torch.Tensor, expected: torch.Tensor | None, report: ConversionReport) -> None:
    """Record (do not raise) a disagreement between a produced tensor and its target."""
    if expected is None or tuple(tensor.shape) == tuple(expected.shape):
        return
    report.shape_mismatch.append((name, tuple(tensor.shape), tuple(expected.shape)))


def _materialize(
    pair: Pair,
    tensors: list[torch.Tensor],
    layout: AttnLayout,
    padded_vocab_size: int | None,
    report: ConversionReport,
) -> torch.Tensor:
    """Build the Megatron tensor for one table row (HF -> Megatron)."""
    if pair.kind == "synth":
        reference = tensors[0]
        value = pair.constant if pair.constant is not None else 0.0
        return torch.full_like(reference, value)
    if pair.kind == "copy":
        return tensors[0]
    if pair.kind == "qkv":
        return _fuse_qkv(layout, *tensors)
    if pair.kind == "fc1":
        return torch.cat(tensors, dim=0).contiguous()
    if pair.kind == "vocab":
        tensor = tensors[0]
        if padded_vocab_size is None or tensor.shape[0] == padded_vocab_size:
            return tensor
        if tensor.shape[0] > padded_vocab_size:
            raise ValueError(f"{pair.mcore}: HF vocab {tensor.shape[0]} > padded {padded_vocab_size}")
        pad = torch.zeros(
            (padded_vocab_size - tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, pad], dim=0)
    raise ValueError(f"unknown kind {pair.kind!r} for {pair.mcore}")


def hf_to_mcore(
    hf_state_dict: dict[str, torch.Tensor],
    table: Table,
    *,
    layout: AttnLayout | None = None,
    padded_vocab_size: int | None = None,
    policy=None,
    dtype: torch.dtype | None = None,
    expected: Iterable[str] | None = None,
) -> tuple[dict[str, torch.Tensor], ConversionReport]:
    """HF ``state_dict`` -> Megatron ``state_dict``.

    Args:
        hf_state_dict: the source, keyed by HF names (extra entries such as
            ``model.rotary_emb.inv_freq`` are reported, not ignored).
        table: from :func:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables.build_table`.
        layout: attention geometry for the q/k/v fusion; read from the config by
            the caller.
        padded_vocab_size: Megatron's vocab (``make_vocab_size_divisible_by``);
            ``None`` or equal to the HF vocab means no padding.
        policy: :class:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables.SynthesisPolicy` for the
            Megatron-only tensors.
        dtype: cast every produced tensor to this dtype when given.
        expected: the Megatron model's own key list, when available: used to
            tighten ``missing`` (a table row whose destination the model does
            not have is a table bug, not a caller bug).

    Returns:
        ``(state_dict, report)``.  The dict has exactly ``table.mcore_names`` as
        keys -- every row produces a tensor, including the synthesised ones.
    """
    from .tables import SynthesisPolicy

    policy = policy or SynthesisPolicy()
    report = ConversionReport(direction="hf->mcore", variant=table.variant)
    if layout is None:
        raise ValueError("AttnLayout is required (q/k/v fusion); build it with AttnLayout.from_config")

    unconsumed = set(hf_state_dict)
    out: dict[str, torch.Tensor] = {}
    for pair in table.pairs:
        tensors: list[torch.Tensor] = []
        for name in pair.hf:
            if name not in hf_state_dict:
                report.missing_source.append(name)
                tensors = []
                break
            tensors.append(hf_state_dict[name])
            unconsumed.discard(name)
        if pair.synthesized:
            value = pair.synthesized_value(policy)
            built = _synth_tensor(pair, table, hf_state_dict, value)
            if built is None:
                report.missing_source.append(f"{pair.mcore} (no sibling tensor to shape the constant)")
                continue
            out[pair.mcore] = built
            report.synthesized[pair.mcore] = float(value)
            continue
        if not tensors:
            continue
        tensor = _materialize(pair, tensors, layout, padded_vocab_size, report)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        out[pair.mcore] = tensor.contiguous()

    report.produced = len(out)
    report.unmapped_source = sorted(unconsumed)
    if expected is not None:
        expected_set = set(expected)
        absent = [name for name in out if name not in expected_set]
        if absent:
            report.notes.append(
                f"{len(absent)} produced tensors are not in the model's state_dict (table/model disagree): "
                f"{absent[:4]}"
            )
    return out, report


def _synth_tensor(
    pair: Pair, table: Table, hf_state_dict: dict, value: float
) -> torch.Tensor | None:
    """The right *shape and dtype* for a synthesised row, taken from its module siblings.

    The Megatron-only tensors are a scalar gate, a bias, or a null source, and
    each one's shape follows from a named sibling **in the same module**:

    * ``read_scale``  -> ``(1,)``
    * ``*.up.bias``   -> the row count of the sibling ``*.up.weight``
    * ``null_source`` -> the width of the sibling ``proj.weight``

    Deriving it from a sibling (rather than hard-coding) means a table row that
    points at the wrong module is caught by a shape mismatch instead of by a
    silently wrong tensor.
    """
    prefix = pair.mcore.rsplit(".", 1)[0]
    siblings = {
        p.mcore: hf_state_dict[p.hf[0]]
        for p in table.pairs
        if p.mcore.startswith(prefix + ".") and p.hf and p.hf[0] in hf_state_dict
    }
    if pair.mcore.endswith(".read_scale"):
        # NOTE: the *policy* value, not zeros.  This used to be ``new_zeros(1)``
        # while the report still recorded ``policy.ar_dar_read_scale`` -- the
        # tensor said 0.0 (read off) and the report said 1.0 (the reference
        # operator), so a converted AR/DAR checkpoint silently ran with the
        # connection's read disabled.  The tensor round trip could not see it
        # (0 would have been the value on the way back too); the forward probe
        # could, which is how it was found.
        for tensor in siblings.values():
            return tensor.new_full((1,), value)
        return None
    if pair.mcore.endswith(".up.bias"):
        weight = siblings.get(pair.mcore.replace(".up.bias", ".up.weight"))
        if weight is None:
            return None
        return weight.new_full((weight.shape[0],), value)
    if pair.mcore.endswith(".null_source"):
        for name, tensor in siblings.items():
            if name.endswith(".proj.weight") and tensor.dim() == 2:
                return tensor.new_full((tensor.shape[-1],), value)
        return None
    for tensor in siblings.values():
        return torch.full_like(tensor, value)
    return None


def mcore_to_hf(
    mcore_state_dict: dict[str, torch.Tensor],
    table: Table,
    *,
    layout: AttnLayout | None = None,
    vocab_size: int | None = None,
    policy=None,
    expected: Iterable[str] | None = None,
) -> tuple[dict[str, torch.Tensor], ConversionReport]:
    """Megatron ``state_dict`` -> HF ``state_dict`` (the inverse direction).

    The Megatron-only rows are *dropped* and listed in ``report.dropped``:
    forgetting them silently is what would make a round trip look better than it
    is.  ``vocab_size`` truncates the padded embedding/lm_head rows.
    """
    from .tables import SynthesisPolicy

    policy = policy or SynthesisPolicy()
    report = ConversionReport(direction="mcore->hf", variant=table.variant)
    if layout is None:
        raise ValueError("AttnLayout is required (q/k/v split); build it with AttnLayout.from_config")

    consumed: set[str] = set()
    out: dict[str, torch.Tensor] = {}
    for pair in table.pairs:
        if pair.mcore not in mcore_state_dict:
            report.missing_source.append(pair.mcore)
            continue
        tensor = mcore_state_dict[pair.mcore]
        consumed.add(pair.mcore)
        if pair.synthesized:
            report.dropped.append(pair.mcore)
            report.synthesized[pair.mcore] = pair.synthesized_value(policy)
            continue
        if pair.kind == "copy":
            pieces = [tensor]
        elif pair.kind == "qkv":
            pieces = _split_qkv(layout, tensor)
        elif pair.kind == "fc1":
            pieces = list(tensor.chunk(2, dim=0))
        elif pair.kind == "vocab":
            if vocab_size is not None and tensor.shape[0] != vocab_size:
                if tensor.shape[0] < vocab_size:
                    raise ValueError(f"{pair.mcore}: Megatron vocab {tensor.shape[0]} < {vocab_size}")
                tensor = tensor[:vocab_size]
            pieces = [tensor]
        else:
            raise ValueError(f"unknown kind {pair.kind!r}")
        for name, piece in zip(pair.hf, pieces):
            out[name] = piece.contiguous()

    report.produced = len(out)
    report.unmapped_source = sorted(set(mcore_state_dict) - consumed)
    if expected is not None:
        absent = [name for name in out if name not in set(expected)]
        if absent:
            report.notes.append(
                f"{len(absent)} produced tensors are not in the reference model's state_dict: {absent[:4]}"
            )
    return out, report


def compare(
    produced: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    label: str = "round-trip",
    atol: float = 0.0,
) -> tuple[int, list[str]]:
    """Bit-exact comparison of two state dicts; returns ``(n_bad, details)``.

    ``atol=0`` (the default) means ``torch.equal``: the round trip either
    reproduces the checkpoint or it does not, and there is no tolerance to hide a
    transposed view in.
    """
    details: list[str] = []
    bad = 0
    for name, tensor in reference.items():
        other = produced.get(name)
        if other is None:
            bad += 1
            details.append(f"{name}: MISSING in {label}")
            continue
        if tuple(other.shape) != tuple(tensor.shape):
            bad += 1
            details.append(f"{name}: shape {tuple(other.shape)} != {tuple(tensor.shape)}")
            continue
        if atol == 0.0:
            same = bool(torch.equal(other, tensor))
        else:
            same = bool(torch.allclose(other.float(), tensor.float(), atol=atol, rtol=0.0))
        if not same:
            bad += 1
            delta = (other.float() - tensor.float()).abs().max().item()
            details.append(f"{name}: values differ (max|d| = {delta:.3e})")
    for name in produced:
        if name not in reference:
            bad += 1
            details.append(f"{name}: EXTRA in {label}")
    return bad, details
