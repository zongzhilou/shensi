"""配方的公共件入口：按需导出路径、算法注册表、配置组装、语料准备与训练 / RL 启动。

导出是惰性的（PEP 562）：verl 在干净 worker 里按 ``VERL_USE_EXTERNAL_MODULES`` 导入
``looma_bridge`` 时，正是靠这一点才不会把 config / rl / runner 拖进来、撞上循环导入。
"""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, str] = {
    ".algos": "DEFAULT_ALGO MODEL_ALGOS apply_algo_or_die apply_model_algo",
    ".config": (
        "add_common_train_args apply_overrides build_config dataprep_config load_blend_spec load_yaml "
        "merge_profile profile_from_args resolve_cfg smoke_config train_from_args"
    ),
    ".paths": "RECIPE TOKENIZER_DIR TOKENIZER_ENV env_paths stage_dirs",
    ".prep": "prep_main prepare",
    ".rl": "agent_harness agent_overrides build_verl_command model_type_of run_verl",
    ".runner": "ENTRY early_stop_plan launch run smoke spawn write_run_dir",
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
