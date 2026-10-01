"""强化学习段的 variants.py 模块。"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Variant",
    "ResolvedSpec",
    "VARIANTS",
    "MODEL_TYPES",
    "ARCHITECTURES",
    "models_dir",
    "variant_for_model_type",
    "variant_for_architecture",
    "resolve_megatron_spec",
    "build_layer_spec",
    "hf_to_megatron_knobs",
]


@dataclass(frozen=True)
class ResolvedSpec:

    module_name: str
    object_name: str
    module: object
    spec: object

    def __str__(self) -> str:
        return f"{self.module_name}:{self.object_name}"


@dataclass(frozen=True)
class Variant:

    name: str
    model_type: str
    config_class: str
    model_class: str
    lm_class: str

    spec_candidates: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    knob_map: dict[str, str] | None = None

    extra_knobs: dict[str, object] = field(default_factory=dict)

    hf_knob_prefix: str = "attn_res_"
    megatron_knob_prefix: str = "gdar_"

    block_size_knob: str = "attn_res_block_size"

    notes: str = ""

    @property
    def config_module(self) -> str:
        return f"configuration_{self.model_type}"

    @property
    def modeling_module(self) -> str:
        return f"modeling_{self.model_type}"

    @property
    def auto_map(self) -> dict[str, str]:
        return {
            "AutoConfig": f"{self.config_module}.{self.config_class}",
            "AutoModel": f"{self.modeling_module}.{self.model_class}",
            "AutoModelForCausalLM": f"{self.modeling_module}.{self.lm_class}",
        }


def _flagscale_spec(module: str, *objects: str) -> tuple[tuple[str, str], ...]:
    return tuple((module, obj) for obj in objects)


_GDAR = "shensi.recipes.paper.gated_delta_attn_res.models.megatron.gdar_spec"
_DEPTH = "shensi.recipes.paper.gated_delta_attn_res.models.megatron.depth_spec"
_HC = "shensi.recipes.paper.gated_delta_attn_res.models.megatron.hc_spec"


_DEPTH_SHARED = {
    "attn_res_block_size": "depth_block_size",
    "attn_res_output_route": "depth_output_route",
}

_HC_SHARED = {
    "attn_res_block_size": "hc_chunk_size",
    "hc_num_streams": "hc_num_streams",
    "hc_init": "hc_init",
    "hc_read": "hc_read",
    "mhc_sinkhorn_iterations": "hc_sinkhorn_iterations",
    "mhc_init_gating_factor": "hc_gating_factor",
    "hc_output_contract": "hc_contract",
}

VARIANTS: dict[str, Variant] = {
    "ar": Variant(
        name="ar",
        model_type="qwen3_ar",
        config_class="Qwen3ARConfig",
        model_class="Qwen3ARModel",
        lm_class="Qwen3ARForCausalLM",
        spec_candidates=_flagscale_spec(_DEPTH, "ar_layer_spec", "ar_layer_spec_reference"),
        knob_map=dict(_DEPTH_SHARED),
        extra_knobs={"depth_ar_reset": "zero"},
    ),
    "dar": Variant(
        name="dar",
        model_type="qwen3_dar",
        config_class="Qwen3DARConfig",
        model_class="Qwen3DARModel",
        lm_class="Qwen3DARForCausalLM",
        spec_candidates=_flagscale_spec(
            _DEPTH, "dar_layer_spec", "dar_layer_spec_reference", "dar_layer_spec_null_source"
        ),
        knob_map=dict(_DEPTH_SHARED, attn_res_use_null_source="depth_use_null_source"),
    ),
    "gdar": Variant(
        name="gdar",
        model_type="qwen3_gdar",
        config_class="Qwen3GDARConfig",
        model_class="Qwen3GDARModel",
        lm_class="Qwen3GDARForCausalLM",
        spec_candidates=_flagscale_spec(
            _GDAR,
            "gdar_layer_spec",
            "gdar_layer_spec_theory",
            "gdar_layer_spec_fullrank",
            "gdar_layer_spec_reference",
        ),
    ),
    "hc": Variant(
        name="hc",
        model_type="qwen3_hc",
        config_class="Qwen3HCConfig",
        model_class="Qwen3HCModel",
        lm_class="Qwen3HCForCausalLM",
        spec_candidates=_flagscale_spec(_HC, "hc_layer_spec", "hc_layer_spec_force_identity", "hc_layer_spec_published"),
        knob_map=dict(_HC_SHARED, hc_family="hc_family"),
        notes="megatron-core 0.18.2 implements hyper-connections natively too (see NATIVE_MHC).",
    ),
    "mhc": Variant(
        name="mhc",
        model_type="qwen3_mhc",
        config_class="Qwen3MHCConfig",
        model_class="Qwen3MHCModel",
        lm_class="Qwen3MHCForCausalLM",
        spec_candidates=_flagscale_spec(_HC, "mhc_layer_spec", "mhc_layer_spec_force_identity", "mhc_layer_spec_published"),
        knob_map=dict(_HC_SHARED, hc_family="mhc_family"),
        notes="megatron-core 0.18.2 implements mHC natively too (see NATIVE_MHC).",
    ),
    "mudd": Variant(
        name="mudd",
        model_type="qwen3_mudd",
        config_class="Qwen3MUDDConfig",
        model_class="Qwen3MUDDModel",
        lm_class="Qwen3MUDDForCausalLM",
        spec_candidates=_flagscale_spec(
            _DEPTH, "mudd_layer_spec", "mudd_layer_spec_reference", "mudd_layer_spec_official_norm", "mudd_layer_spec_random"
        ),
        knob_map=dict(
            _DEPTH_SHARED,
            mudd_num_ways="depth_mudd_num_ways",
            mudd_act="depth_mudd_act",
            mudd_hidden_round="depth_mudd_hidden_round",
            mudd_scale_dw="depth_mudd_scale_dw",
            mudd_param="depth_mudd_param",
            mudd_pre_norm="depth_mudd_use_pre_norm",
            mudd_post_norm="depth_mudd_use_post_norm",
        ),
    ),
    "realformer": Variant(
        name="realformer",
        model_type="qwen3_realformer",
        config_class="Qwen3RealFormerConfig",
        model_class="Qwen3RealFormerModel",
        lm_class="Qwen3RealFormerForCausalLM",
        spec_candidates=(
            (
                "shensi.recipes.paper.gated_delta_attn_res.models.megatron.realformer_spec",
                "realformer_layer_spec",
            ),
            (
                "shensi.recipes.paper.gated_delta_attn_res.models.megatron.realformer_spec",
                "realformer_layer_spec_identity",
            ),
            (
                "shensi.recipes.paper.gated_delta_attn_res.models.megatron.realformer_spec",
                "realformer_layer_spec_reference",
            ),
        ),
        knob_map={
            "attn_res_realformer_gate": "realformer_gate",
            "attn_res_realformer_mean": "realformer_mean",
        },
    ),
    "denseformer": Variant(
        name="denseformer",
        model_type="qwen3_denseformer",
        config_class="Qwen3DenseFormerConfig",
        model_class="Qwen3DenseFormerModel",
        lm_class="Qwen3DenseFormerForCausalLM",
        spec_candidates=_flagscale_spec(
            _DEPTH, "denseformer_layer_spec", "denseformer_layer_spec_official", "denseformer_layer_spec_period4"
        ),
        knob_map=dict(
            _DEPTH_SHARED,
            attn_res_dwa_param="depth_dwa_param",
            attn_res_dwa_dilation="depth_dwa_dilation",
        ),
    ),
}

MODEL_TYPES: tuple[str, ...] = tuple(v.model_type for v in VARIANTS.values())
ARCHITECTURES: tuple[str, ...] = tuple(v.lm_class for v in VARIANTS.values())

_BY_MODEL_TYPE = {v.model_type: v for v in VARIANTS.values()}
_BY_ARCHITECTURE = {v.lm_class: v for v in VARIANTS.values()}


def variant_for_model_type(model_type: str) -> Variant | None:
    return _BY_MODEL_TYPE.get(model_type)


def variant_for_architecture(architecture: str) -> Variant | None:
    return _BY_ARCHITECTURE.get(architecture)


def models_dir() -> Path:
    override = os.environ.get("VERL_PLUGIN_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "models"


def resolve_megatron_spec(variant: Variant, object_name: str | None = None) -> "ResolvedSpec":
    override = os.environ.get("VERL_PLUGIN_SPEC")
    if override:
        module_name, _, obj = override.partition(":")
        candidates = ((module_name, obj or f"{variant.name}_layer_spec"),)
    elif object_name is not None:
        candidates = tuple((mod, object_name) for mod, _ in variant.spec_candidates) or _flagscale_spec(
            variant.name, object_name
        )
    else:
        candidates = variant.spec_candidates

    tried: list[str] = []
    for module_name, obj_name in candidates:
        tried.append(f"{module_name}:{obj_name}")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        spec = getattr(module, obj_name, None)
        if spec is not None:
            return ResolvedSpec(module_name=module_name, object_name=obj_name, module=module, spec=spec)

    raise NotImplementedError(
        f"no Megatron layer spec for model_type={variant.model_type!r}. Tried {tried}. "
        f"A spec is a module-level object (the --spec form) that returns/lifts a "
        f"ModuleSpec or TransformerBlockSubmodules subclassing BaseTransformerLayer; "
        f"point VERL_PLUGIN_SPEC=<module>:<object> at it to wire the variant up."
    )


GDAR_GATE_CHANNELS_MODULE = (
    "shensi.recipes.paper.gated_delta_attn_res.models.megatron.ablation_spec"
)
GDAR_GATE_CHANNEL_SPECS: dict[str, str] = {
    "none": "gated_ar_layer_spec_no_gate",
    "d": "gated_ar_layer_spec_decay_only",
    "e": "gated_ar_layer_spec_erase_only",
    "w": "gated_ar_layer_spec_write_only",
    "de": "gated_ar_layer_spec_decay_erase",
    "dw": "gated_ar_layer_spec_write_decay",
    "ew": "gated_ar_layer_spec_erase_write",
    "dew": None,
    "scalar": "gated_ar_layer_spec_scalar",
}


def build_layer_spec(hf_config, variant: Variant):
    resolved = resolve_megatron_spec(variant)
    if variant.name == "gdar":
        channels = str(getattr(hf_config, "attn_res_gate_channels", "dew"))
        if channels not in GDAR_GATE_CHANNEL_SPECS:
            raise ValueError(
                f"attn_res_gate_channels={channels!r} has no Megatron counterpart. "
                f"Known: {sorted(GDAR_GATE_CHANNEL_SPECS)}; the mapping lives in "
                f"{GDAR_GATE_CHANNELS_MODULE} (one spec object per subset)."
            )
        object_name = GDAR_GATE_CHANNEL_SPECS[channels]
        module = importlib.import_module(GDAR_GATE_CHANNELS_MODULE)
        if object_name is not None:
            spec_obj = getattr(module, object_name, None)
            if spec_obj is None:
                raise ValueError(
                    f"{GDAR_GATE_CHANNELS_MODULE}:{object_name} not found for gate_channels={channels!r}"
                )
            resolved = ResolvedSpec(
                module_name=GDAR_GATE_CHANNELS_MODULE, object_name=object_name, module=module, spec=spec_obj
            )
    spec = resolved.spec
    params = getattr(spec, "params", None)
    knobs, dropped = hf_to_megatron_knobs(hf_config, variant)

    if not knobs or not isinstance(params, dict):
        return spec, resolved, dropped

    spec_cls = type(spec)
    return spec_cls(module=spec.module, submodules=spec.submodules, params={**params, **knobs}), resolved, dropped


def hf_to_megatron_knobs(
    hf_config, variant: Variant, accepted: set[str] | None = None
) -> tuple[dict, list[str]]:
    fields = getattr(type(hf_config), "__annotations__", {})
    knobs: dict[str, object] = {}
    dropped: list[str] = []

    def is_connection_field(name: str) -> bool:
        if name.startswith("attn_res_"):
            return True
        if variant.name in ("hc", "mhc"):
            return name.startswith(("hc_", "mhc_", "gdar_"))
        return name.startswith(f"{variant.name}_")

    if variant.knob_map is None:
        for name in fields:
            if not name.startswith(variant.hf_knob_prefix):
                continue
            if variant.name == "gdar" and name == "attn_res_gate_channels":
                continue
            knob = variant.megatron_knob_prefix + name[len(variant.hf_knob_prefix) :]
            if accepted is not None and knob not in accepted:
                dropped.append(knob)
                continue
            knobs[knob] = getattr(hf_config, name)
        knobs.update(variant.extra_knobs)
        return knobs, dropped

    for hf_name, knob in variant.knob_map.items():
        if hf_name in fields:
            knobs[knob] = getattr(hf_config, hf_name)
    for name in fields:
        if name not in variant.knob_map and is_connection_field(name):
            dropped.append(name)
    knobs.update(variant.extra_knobs)
    return knobs, dropped
