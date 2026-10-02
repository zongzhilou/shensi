"""配方的公共件入口：按需导出路径、算法注册表、配置组装、语料准备与训练 / RL 启动。

导出是惰性的（PEP 562）：深路径导入（如 mcore 层规格）不会把 config / rl / runner
与它们背后的 verl、mcore 一起拖进来。
"""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, str] = {
    ".algos": "DEFAULT_ALGO MODEL_ALGOS apply_algo_or_die apply_model_algo",
    ".config": (
        "_coerce _set_dotted _stage_cfg add_common_train_args build_config dataprep_config "
        "dataprep_config_for load_blend load_blend_spec load_yaml profile_from_args resolve_cfg "
        "smoke_config train_from_args"
    ),
    ".paths": "RECIPE TOKENIZER_DIR TOKENIZER_ENV _SHARED_GEOMS base env_paths stage_dirs",
    ".rl": "_VERL_CLI_EXTRA agent_harness agent_overrides build_verl_command model_type_of run_verl",
    ".runner": "EARLY_STOP_DEFAULTS early_stop_plan run smoke write_run_dir",
}

_OWNER: dict[str, str] = {
    name: module for module, names in _EXPORTS.items() for name in names.split()
}

__all__ = sorted(_OWNER)


def __getattr__(name: str):
    module = _OWNER.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
