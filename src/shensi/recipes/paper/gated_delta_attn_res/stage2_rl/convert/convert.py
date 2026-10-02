"""转换实现：QKV 融合 / 拆分与双向权重搬运。"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .tables import Pair, Table

__all__ = ["ConversionReport", "hf_to_mcore", "mcore_to_hf", "AttnLayout", "check_shape"]


@dataclass
class AttnLayout:

    """注意力权重的排布约定（QKV 融合顺序与头数）。"""
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

    """一次转换的结果报告（逐张量状态与统计）。"""
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
    hidden = q.shape[-1]
    group_dim = layout.head_dim * layout.num_attention_heads // layout.num_key_value_heads
    groups = q.shape[0] // group_dim
    q = q.reshape(groups, group_dim, hidden)
    k = k.reshape(groups, layout.head_dim, hidden)
    v = v.reshape(groups, layout.head_dim, hidden)
    return torch.cat([q, k, v], dim=1).reshape(-1, hidden).contiguous()


def _split_qkv(layout: AttnLayout, qkv: torch.Tensor) -> list[torch.Tensor]:
    hidden = qkv.shape[-1]
    group_dim = layout.head_dim * layout.num_attention_heads // layout.num_key_value_heads
    grouped = qkv.reshape(layout.num_key_value_heads, -1, hidden)
    q_len = group_dim
    q = grouped[:, :q_len].reshape(-1, hidden)
    k = grouped[:, q_len : q_len + layout.head_dim].reshape(-1, hidden)
    v = grouped[:, q_len + layout.head_dim :].reshape(-1, hidden)
    return [q.contiguous(), k.contiguous(), v.contiguous()]


def check_shape(name: str, tensor: torch.Tensor, expected: torch.Tensor | None, report: ConversionReport) -> None:
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
    """把 HF 权重转成 mcore 权重。"""
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
    prefix = pair.mcore.rsplit(".", 1)[0]
    siblings = {
        p.mcore: hf_state_dict[p.hf[0]]
        for p in table.pairs
        if p.mcore.startswith(prefix + ".") and p.hf and p.hf[0] in hf_state_dict
    }
    if pair.mcore.endswith(".read_scale"):







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
    """把 mcore 权重转回 HF 权重。"""
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
