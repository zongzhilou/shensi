"""权重对应表：由规则生成 HF 与 mcore 张量的配对与合成策略。"""


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











MCORE_ROOT = "decoder"
HF_ROOT = "model"
MCORE_EMBEDDING = "embedding.word_embeddings.weight"
MCORE_FINAL_NORM = f"{MCORE_ROOT}.final_layernorm.weight"
MCORE_OUTPUT = "output_layer.weight"

ALL_VARIANTS = ("ar", "dar", "gdar", "denseformer", "hc", "mhc", "mudd", "realformer")


@dataclass(frozen=True)
class SynthesisPolicy:

    """张量合成策略（切片 / 拼接 / 转置等）的登记。"""
    ar_dar_read_scale: float = 1.0
    zero: float = 0.0


@dataclass(frozen=True)
class Pair:

    """一条权重对应：HF 张量名 ↔ mcore 张量名，以及合成方式。"""
    mcore: str
    hf: tuple[str, ...] = ()
    kind: Kind = "copy"

    constant: float | None = None

    policy_field: str | None = None

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

    """整个模型的权重对应表：按层与模块组织 Pair。"""
    variant: str
    pairs: list[Pair] = field(default_factory=list)

    facts: dict = field(default_factory=dict)

    unsupported: list[str] = field(default_factory=list)



    def by_mcore(self) -> dict[str, Pair]:
        out: dict[str, Pair] = {}
        for pair in self.pairs:
            if pair.mcore in out:
                raise ValueError(f"duplicate mcore name in table: {pair.mcore}")
            out[pair.mcore] = pair
        return out

    def by_hf(self) -> dict[str, tuple[Pair, int]]:
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







def _backbone_pairs(num_layers: int) -> Iterator[Pair]:
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







def _gdar_pairs(config, num_layers: int) -> Iterator[Pair]:
    gate_rank = getattr(config, "attn_res_gate_rank", None)
    q_rank = getattr(config, "attn_res_q_rank", None)
    k_rank = getattr(config, "attn_res_k_rank", None)
    ladder = int(getattr(config, "attn_res_decay_ladder", 0) or 0)
    output_route = bool(getattr(config, "attn_res_output_route", True))






    deviation = getattr(config, "attn_res_gate_param", "sigmoid") == "deviation"

    def module(m: str, h: str, writer: bool = True) -> Iterator[Pair]:
        if writer:
            if gate_rank is None:
                yield Pair(f"{m}.gate_proj.weight", (f"{h}.gate_proj.weight",))
                yield Pair(f"{m}.gate_proj.bias", (f"{h}.gate_proj.bias",))
            else:





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







def _realformer_pairs(config, num_layers: int) -> Iterator[Pair]:
    gate = str(getattr(config, "attn_res_realformer_gate", "deviation"))
    if gate != "deviation":
        return
    for layer in range(1, num_layers):
        yield Pair(
            f"{MCORE_ROOT}.layers.{layer}.self_attention.core_attention.carry_gate",
            (f"{HF_ROOT}.layers.{layer}.self_attn.realformer_gate.delta",),
        )


def _denseformer_pairs(config, num_layers: int) -> Iterator[Pair]:
    mode = getattr(config, "attn_res_dwa_param", "deviation")
    weight = "alpha" if mode == "official" else "alpha_delta"
    for layer in range(num_layers):
        yield Pair(
            f"{MCORE_ROOT}.layers.{layer}.block_dwa.{weight}",
            (f"{HF_ROOT}.layers.{layer}.dwa.{weight}",),
        )







def _hc_pairs(config, num_layers: int) -> Iterator[Pair]:


    for layer in range(num_layers):
        for sub in ("self_attention", "mlp"):
            for tensor in ("alpha_pre", "alpha_post", "alpha_res", "bias", "mapping_proj.weight"):
                yield Pair(
                    f"{MCORE_ROOT}.layers.{layer}.{sub}_hyper_connection.{tensor}",
                    (f"{HF_ROOT}.layers.{layer}.{sub}_hyper_connection.{tensor}",),
                )

    if getattr(config, "hc_output_contract", "mean") == "learned":
        yield Pair(
            f"{MCORE_ROOT}.layers.{num_layers - 1}.head_fn",
            (f"{HF_ROOT}.hc_head_fn",),
            note="one head per chunk exit; HF keeps a single model-level one",
        )
        yield Pair(f"{MCORE_ROOT}.layers.{num_layers - 1}.head_base", (f"{HF_ROOT}.hc_head_base",))
        yield Pair(f"{MCORE_ROOT}.layers.{num_layers - 1}.head_scale", (f"{HF_ROOT}.hc_head_scale",))







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






_BUILDERS = {
    "gdar": _gdar_pairs,
    "ar": lambda config, n, **_kw: _ar_dar_pairs(config, n, "ar", _kw["policy"]),
    "dar": lambda config, n, **_kw: _ar_dar_pairs(config, n, "dar", _kw["policy"]),
    "denseformer": _denseformer_pairs,
    "realformer": _realformer_pairs,
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
    if variant not in _BUILDERS:
        raise KeyError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
    policy = policy or SynthesisPolicy()
    if connection is None:


        connection = any(
            getattr(config, knob, None) is not None
            for knob in ("attn_res_block_size", "attn_res_realformer_gate")
        )

    table = Table(variant=variant)
    table.pairs.extend(_backbone_pairs(num_layers))
    if connection:
        builder = _BUILDERS[variant]
        if variant in ("ar", "dar"):
            table.pairs.extend(builder(config, num_layers, policy=policy))
        else:
            table.pairs.extend(builder(config, num_layers))


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










            table.unsupported.append(
                "attn_res_block_size=1: the Megatron port takes its per-sublayer source mode at "
                "block size 1 (three sources per layer) where HF's AR appends one per layer, so the "
                "two models are not equivalent after conversion; AR is exact at block size > 1"
            )
        if variant in ("hc", "mhc"):




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
