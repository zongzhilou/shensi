"""模型算法注册表：算法名到 mcore 层规格预设的映射（主行与消融行）。"""

from __future__ import annotations

_SPEC = "shensi.recipes.paper.looma.common.models.megatron.looma_spec"

MODEL_ALGOS: dict[str, tuple[str, str]] = {
    "looma": (_SPEC, "looma_layer_spec"),
    "looma_no_loop": (_SPEC, "looma_layer_spec_no_loop"),
    "looma_iter64": (_SPEC, "looma_layer_spec_iter64"),
    "looma_tol1e3": (_SPEC, "looma_layer_spec_tol1e3"),
    "looma_fixed_count": (_SPEC, "looma_layer_spec_fixed_count"),
    "looma_tau05": (_SPEC, "looma_layer_spec_tau05"),
    "looma_grad0": (_SPEC, "looma_layer_spec_grad0"),
    "looma_rank16": (_SPEC, "looma_layer_spec_rank16"),
    "looma_rankfull": (_SPEC, "looma_layer_spec_rankfull"),
    "looma_heads1": (_SPEC, "looma_layer_spec_heads1"),
    "looma_no_output_route": (_SPEC, "looma_layer_spec_no_output_route"),
    "looma_lambda_free": (_SPEC, "looma_layer_spec_lambda_free"),
    "looma_flat_ladder": (_SPEC, "looma_layer_spec_flat_ladder"),
    "looma_carrier0": (_SPEC, "looma_layer_spec_carrier0"),
}

DEFAULT_ALGO = "looma"


def apply_model_algo(cfg: dict, algo: str | None) -> str | None:
    """把算法名对应的层规格写进 ``train.model.spec``；不给算法名时不动配置。"""
    if algo is None:
        return None
    if algo not in MODEL_ALGOS:
        raise SystemExit(f"[looma] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}")
    module, obj = MODEL_ALGOS[algo]
    cfg.setdefault("train", {}).setdefault("model", {})["spec"] = [module, obj]
    return algo


def apply_algo_or_die(algo: str | None) -> str | None:
    """校验算法名，不认识就报错退出；认识则原样返回。"""
    if algo is not None and algo not in MODEL_ALGOS:
        raise SystemExit(f"[looma] 未知模型算法：{algo!r}。可用：{sorted(MODEL_ALGOS)}")
    return algo
