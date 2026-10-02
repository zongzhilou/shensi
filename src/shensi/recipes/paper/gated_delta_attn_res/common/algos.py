"""模型算法注册表：算法名到 mcore 层规格预设的映射（主行、对照臂与消融行）。"""

from __future__ import annotations

_MODELS = "shensi.recipes.paper.gated_delta_attn_res.common.models.megatron"
MODEL_ALGOS: dict[str, tuple[str, str] | None] = {
    "base": None,
    "qwen3_ar": (_MODELS, "ar_layer_spec"),
    "qwen3_ar_block4": (_MODELS, "ar_layer_spec_block4"),
    "qwen3_dar": (_MODELS, "dar_layer_spec"),
    "qwen3_dar_block4": (_MODELS, "dar_layer_spec_block4"),
    "qwen3_denseformer": (_MODELS, "denseformer_layer_spec"),
    "qwen3_realformer": (_MODELS, "realformer_layer_spec"),
    "qwen3_realformer_identity": (_MODELS, "realformer_layer_spec_identity"),
    "qwen3_realformer_reference": (_MODELS, "realformer_layer_spec_reference"),
    "qwen3_realformer_mean": (_MODELS, "realformer_layer_spec_mean"),
    "qwen3_mudd": (_MODELS, "mudd_layer_spec"),
    "qwen3_hc": (_MODELS, "hc_layer_spec"),
    "qwen3_mhc": (_MODELS, "mhc_layer_spec"),
    "qwen3_gdar_paper": (_MODELS, "gdar_layer_spec_paper"),
    "qwen3_gdar": (_MODELS, "gdar_layer_spec"),
    "qwen3_gdar_theory": (_MODELS, "gdar_layer_spec_theory"),
    "qwen3_gdar_upstream": (
        _MODELS,
        "gdar_layer_spec_upstream",
    ),
    "qwen3_gdar_fullrank": (_MODELS, "gdar_layer_spec_fullrank"),
    "qwen3_gdar_block2": (_MODELS, "gdar_layer_spec_block2"),
    "qwen3_gdar_block4": (_MODELS, "gdar_layer_spec_block4"),
    "qwen3_gdar_block8": (_MODELS, "gdar_layer_spec_block8"),
    "qwen3_gdar_block16": (_MODELS, "gdar_layer_spec_block16"),
    "qwen3_gdar_r16": (_MODELS, "gdar_layer_spec_block4_r16"),
    "qwen3_gdar_noladder": (_MODELS, "gdar_layer_spec_paper_noladder"),
    "qwen3_gdar_no_output_route": (_MODELS, "gdar_layer_spec_no_output_route"),
    "qwen3_gated_ar": (_MODELS, "gated_ar_layer_spec"),
    "a1a_gate_prefix": (_MODELS, "gdar_layer_spec_gate_prefix"),
    "a1b_gate_delta": (_MODELS, "gdar_layer_spec_gate_delta"),
    "a3_decay_projected": (_MODELS, "gdar_layer_spec_decay_projected"),
    "a4_lambda_free": (_MODELS, "gdar_layer_spec_lambda_free"),
    "a6_reference": (_MODELS, "gdar_layer_spec_reference"),
    "a9_half_init": (_MODELS, "gdar_half_init_layer_spec"),
    "a9_uniform_init": (_MODELS, "gdar_uniform_init_layer_spec"),
    "e3_scalar_gate": (_MODELS, "gated_ar_layer_spec_scalar"),
    "e3_no_gate": (_MODELS, "gated_ar_layer_spec_no_gate"),
    "e3_decay_only": (_MODELS, "gated_ar_layer_spec_decay_only"),
    "e3_erase_only": (_MODELS, "gated_ar_layer_spec_erase_only"),
    "e3_write_only": (_MODELS, "gated_ar_layer_spec_write_only"),
}

DEFAULT_ALGO = "qwen3_gdar_paper"

_MODELS_ABL = _MODELS + ".ablation_spec"
MODEL_ALGOS.update(
    {
        "qwen3_gdar_main": (_MODELS, "gdar_layer_spec_paper"),
        "qwen3_gdar_main_sublayer": (_MODELS, "gdar_layer_spec_paper_sublayer"),
        "qwen3_gdar_main_b2": (_MODELS, "gdar_layer_spec_paper_b2"),
        "qwen3_gdar_main_b8": (_MODELS, "gdar_layer_spec_paper_b8"),
        "qwen3_gdar_main_b16": (_MODELS, "gdar_layer_spec_paper_b16"),
        "qwen3_gdar_main_rank16": (_MODELS, "gdar_layer_spec_paper_rank16"),
        "qwen3_gdar_main_rankfull": (_MODELS, "gdar_layer_spec_paper_rankfull"),
        "qwen3_gdar_main_decay_free": (_MODELS, "gdar_layer_spec_paper_decay_free"),
        "qwen3_gdar_main_lambda_free": (_MODELS, "gdar_layer_spec_paper_lambda_free"),
        "qwen3_gdar_main_ladder0": (_MODELS, "gdar_layer_spec_paper_ladder0"),
        "qwen3_gdar_main_gate_prefix": (_MODELS, "gdar_layer_spec_paper_gate_prefix"),
        "qwen3_gdar_main_gate_delta": (_MODELS, "gdar_layer_spec_paper_gate_delta"),
        "qwen3_gdar_main_address_state": (_MODELS, "gdar_layer_spec_paper_address_state"),
        "qwen3_gdar_main_address_novelty": (_MODELS, "gdar_layer_spec_paper_address_novelty"),
        "qwen3_gdar_main_update_reference": (_MODELS, "gdar_layer_spec_paper_update_reference"),
        "qwen3_gdar_main_heads1": (_MODELS, "gdar_layer_spec_paper_heads1"),
        "qwen3_gdar_main_null_off": (_MODELS, "gdar_layer_spec_paper_null_off"),
        "qwen3_gdar_main_whiten_diag": (_MODELS, "gdar_layer_spec_paper_whiten_diag"),
        "qwen3_gdar_main_whiten_off": (_MODELS, "gdar_layer_spec_paper_whiten_off"),
        "qwen3_gdar_main_mix_whitened": (_MODELS, "gdar_layer_spec_paper_mix_whitened"),
        "qwen3_gdar_main_no_output_route": (_MODELS, "gdar_layer_spec_paper_no_output_route"),
        "qwen3_gdar_main_carrier0": (_MODELS, "gdar_layer_spec_paper_carrier0"),
        "qwen3_gdar_main_init_paper": (_MODELS, "gdar_layer_spec_paper_init_paper"),
        "qwen3_gdar_main_init_uniform": (_MODELS, "gdar_layer_spec_paper_init_uniform"),
        "qwen3_gdar_main_init_half": (_MODELS, "gdar_layer_spec_paper_init_half"),
        "qwen3_gdar_main_gates_d": (_MODELS_ABL, "gdar_paper_gates_d"),
        "qwen3_gdar_main_gates_e": (_MODELS_ABL, "gdar_paper_gates_e"),
        "qwen3_gdar_main_gates_w": (_MODELS_ABL, "gdar_paper_gates_w"),
        "qwen3_gdar_main_gates_de": (_MODELS_ABL, "gdar_paper_gates_de"),
        "qwen3_gdar_main_gates_dw": (_MODELS_ABL, "gdar_paper_gates_dw"),
        "qwen3_gdar_main_gates_ew": (_MODELS_ABL, "gdar_paper_gates_ew"),
        "qwen3_gdar_main_gates_scalar": (_MODELS_ABL, "gdar_paper_gates_scalar"),
        "qwen3_gdar_main_gates_none": (_MODELS_ABL, "gdar_paper_gates_none"),
        "qwen3_mhc_lite": (_MODELS, "mhc_layer_spec_lite"),
    }
)


def apply_model_algo(cfg: dict, algo: str | None) -> str | None:
    """把算法名对应的层规格写进 ``train.model.spec``；不给算法名时不动配置。"""
    if algo is None:
        return None
    if algo not in MODEL_ALGOS:
        raise SystemExit(
            f"[gdar] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}（base = plain Qwen3）"
        )
    entry = MODEL_ALGOS[algo]
    model = cfg.setdefault("train", {}).setdefault("model", {})
    if entry is None:
        model.pop("spec", None)
    else:
        module, obj = entry
        model["spec"] = [module, obj]
    return algo


def apply_algo_or_die(algo: str | None) -> str | None:
    """校验算法名，不认识就报错退出；认识则原样返回。"""
    if algo is not None and algo not in MODEL_ALGOS:
        raise SystemExit(
            f"[gdar] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}（base = plain Qwen3）"
        )
    return algo
