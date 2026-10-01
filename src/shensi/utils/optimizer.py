"""优化器接入：AdaMuon（矩阵腿）+ AdEMAMix / GrokFastAdamW（标量腿），导入即生效。"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import torch

__all__ = ["AdEMAMix", "GrokFastAdamW", "install"]


class _InitGroupShim:
    """把 pytorch_optimizer 的状态初始化接到 mcore 的调用口径上。"""

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        stubbed = []
        for p in group["params"]:
            if p.grad is None:
                if skip_non_grad_params:
                    continue
                p.grad = torch.zeros_like(p.data)
                stubbed.append(p)
        self.init_group(group)
        for p in stubbed:
            p.grad = None


def _import_pytorch_optimizer():
    """pytorch_optimizer 是这条链的硬依赖（pyproject 已声明）；缺了直接报清楚。"""
    try:
        from pytorch_optimizer import AdEMAMix as _AdEMAMix
        from pytorch_optimizer import GrokFastAdamW as _GrokFastAdamW
    except ImportError as exc:  # pragma: no cover - 安装问题
        raise ImportError(
            "shensi 的优化器口径（AdaMuon + AdEMAMix / GrokFastAdamW）需要 pytorch-optimizer："
            "`uv pip install pytorch-optimizer`（pyproject.toml 里已声明）"
        ) from exc
    return _AdEMAMix, _GrokFastAdamW


_PytorchAdEMAMix, _PytorchGrokFastAdamW = _import_pytorch_optimizer()


class AdEMAMix(_InitGroupShim, _PytorchAdEMAMix):
    """pytorch_optimizer 的 AdEMAMix（arXiv:2409.03137）：快/慢两条 EMA + 二阶矩。"""


class GrokFastAdamW(_InitGroupShim, _PytorchGrokFastAdamW):
    """pytorch_optimizer 的 GrokFastAdamW（arXiv:2405.20233）：梯度低通滤波 + AdamW。"""


def _ademamix_kwargs(config: Any, *, lr: float, betas: tuple, weight_decay: float) -> dict:
    """AdEMAMix 的构造参数：三元 betas 按 (adam_beta1, adam_beta2, ademamix_beta3) 组。"""
    three = getattr(config, "ademamix_betas", None)
    if three is None or len(tuple(three)) != 3:
        slow = getattr(config, "ademamix_beta3", None) or 0.9999
        three = (
            float(getattr(config, "adam_beta1", 0.9)),
            float(getattr(config, "adam_beta2", 0.999)),
            float(slow),
        )
    kwargs = {
        "lr": lr,
        "betas": tuple(float(b) for b in three),
        "weight_decay": weight_decay,
        "alpha": float(getattr(config, "ademamix_alpha", 5.0) or 5.0),
    }
    t_alpha_beta3 = getattr(config, "ademamix_t_alpha_beta3", None)
    if t_alpha_beta3:
        kwargs["t_alpha_beta3"] = int(t_alpha_beta3)
    return kwargs


def _grokfast_kwargs(config: Any, *, lr: float, betas: tuple, weight_decay: float) -> dict:
    """GrokFastAdamW 的构造参数：滤波开关与三个滤波超参从配置上取（没挂就用库默认）。"""
    kwargs = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
    for name, caster in (
        ("grokfast", bool),
        ("grokfast_alpha", float),
        ("grokfast_lamb", float),
        ("grokfast_after_step", int),
    ):
        value = getattr(config, name, None)
        if value is not None:
            kwargs[name] = caster(value)
    return kwargs


# 标量腿可选的名字 → (类, 构造参数映射)
# 额外旋钮由 recipes 侧的 `--ademamix-*` / `--grokfast-*` 挂到 OptimizerConfig 上（见 train/builders.py）。
_SCALAR_LEG: dict[str, tuple[type, Callable[..., dict]]] = {
    "ademamix": (AdEMAMix, _ademamix_kwargs),
    "grokfastadamw": (GrokFastAdamW, _grokfast_kwargs),
}

# 主优化器属于 Muon 家族时，标量腿的名字看 `muon_scalar_optimizer`
_MUON_FAMILY = ("muon", "dist_muon", "adaptive_muon")

# 这两个优化器的状态键不止 Adam 的那两个，checkpoint 的按名查找表要跟上
_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "ademamix": ("exp_avg", "exp_avg_sq", "exp_avg_slow"),
    "grokfastadamw": ("exp_avg", "exp_avg_sq", "grok_exp_avg"),
}

_ACTIVE_SCALAR: ContextVar[dict | None] = ContextVar("shensi_scalar_leg", default=None)


class _ScalarLegDispatch:
    """站在上游 `Lion` 的位置上。"""

    _native_lion: type | None = None

    def __new__(
        cls,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.01,
        **kwargs,
    ):
        active = _ACTIVE_SCALAR.get()
        if active is None:
            return cls._native_lion(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
        build = active["build"]
        return active["cls"](params, **build(lr=lr, betas=betas, weight_decay=weight_decay))


def _scalar_state_init(opt: torch.optim.Optimizer, config: Any = None) -> None:
    """Mcore 的 init_state_fn 口径：给每个参数把状态建出来（torch_dist 的加载模板要它）。"""
    for group in opt.param_groups:
        opt._init_group(group, skip_non_grad_params=False)


def _rebind_init_state(result: Any) -> Any:
    """上游 lion 分支给的 init_state_fn 只会建 `exp_avg`；换成按我们优化器自己状态建的一版。"""
    if isinstance(result, tuple):
        raw, _ = result
        return (raw, _scalar_state_init)
    if hasattr(result, "init_state_fn"):
        result.init_state_fn = _scalar_state_init
    return result


def _install_scalar_leg() -> bool:
    """让标量腿（`--muon-scalar-optimizer`）接受 AdEMAMix / GrokFastAdamW。"""
    from megatron.core import optimizer as mcore_optimizer

    original = mcore_optimizer._get_megatron_optimizer_based_on_param_groups
    if getattr(original, "_shensi_scalar_leg", False):
        return False
    native_lion = getattr(mcore_optimizer, "Lion", None)
    if native_lion is None:  # 上游太老（没有 lion 分支）就不动它
        return False

    _ScalarLegDispatch._native_lion = native_lion

    def build_with_scalar_leg(config, model_chunks, param_groups, *args, **kwargs):
        name = str(getattr(config, "optimizer", "") or "").lower()
        entry = _SCALAR_LEG.get(name)
        if entry is None:
            return original(config, model_chunks, param_groups, *args, **kwargs)
        cls, build = entry
        token = _ACTIVE_SCALAR.set({"cls": cls, "build": _bind_build(build, config)})
        saved = config.optimizer
        config.optimizer = "lion"  # 借上游自己的标量腿分支做构造与包装；名字只在这段窗口里改
        try:
            result = original(config, model_chunks, param_groups, *args, **kwargs)
        finally:
            config.optimizer = saved
            _ACTIVE_SCALAR.reset(token)
        return _rebind_init_state(result)

    build_with_scalar_leg._shensi_scalar_leg = True
    mcore_optimizer._get_megatron_optimizer_based_on_param_groups = build_with_scalar_leg
    mcore_optimizer.Lion = _ScalarLegDispatch
    return True


def _bind_build(build: Callable[..., dict], config: Any) -> Callable[..., dict]:
    """把配置绑进"构造参数映射"：dispatch 只需要传 mcore 给的那三个量。"""

    def bound(*, lr: float, betas: tuple, weight_decay: float) -> dict:
        return build(config, lr=lr, betas=betas, weight_decay=weight_decay)

    return bound


def _install_state_keys() -> bool:
    """检查点的参数状态按键名读写：把我们的状态键补进上游那张表。"""
    from megatron.core.optimizer import DistributedOptimizer

    prop = DistributedOptimizer.optimizer_state_keys
    if getattr(prop.fget, "_shensi_state_keys", False):
        return False
    original = prop.fget

    def optimizer_state_keys(self):
        name = str(getattr(self.config, "optimizer", "") or "").lower()
        if name in _MUON_FAMILY:
            name = str(getattr(self.config, "muon_scalar_optimizer", "") or "").lower()
        keys = _STATE_KEYS.get(name)
        return keys if keys is not None else original(self)

    optimizer_state_keys._shensi_state_keys = True
    DistributedOptimizer.optimizer_state_keys = property(optimizer_state_keys)
    return True


def _install_emerging_entries() -> list[str]:
    """把这两个优化器也登记成 mcore 的 emerging 主体（`--optimizer ademamix` / `grokfastadamw`）。"""
    from emerging_optimizers import registry
    from megatron.core.optimizer import emerging_optimizers as mcore_eo

    table = getattr(mcore_eo, "_EMERGING_OPTIMIZERS", None)
    installed = []
    for name, (cls, build) in _SCALAR_LEG.items():
        if name not in registry.get_optimizer_name_list():
            registry.register_optimizer(name)(cls)
        if table is not None and name not in table:
            table[name] = mcore_eo.EmergingOptimizerEntry(
                optimizer_cls=cls,
                config_to_kwargs=lambda config, model_chunks=None, pg_collection=None, _b=build: _b(
                    config,
                    lr=float(config.lr),
                    betas=(
                        float(getattr(config, "adam_beta1", 0.9)),
                        float(getattr(config, "adam_beta2", 0.999)),
                    ),
                    weight_decay=float(config.weight_decay),
                ),
            )
        if table is not None and name in table:
            installed.append(name)
    return installed


def install() -> dict[str, Any]:
    """幂等安装：标量腿扩展 + checkpoint 状态键 + emerging 主体登记。"""
    return {
        "scalar_leg": _install_scalar_leg(),
        "state_keys": _install_state_keys(),
        "emerging_entries": _install_emerging_entries(),
    }


_installed = install()
print(
    "[shensi][optim] AdaMuon（矩阵腿）+ AdEMAMix / GrokFastAdamW（标量腿）已接入："
    f"标量腿={_installed['scalar_leg']}、状态键={_installed['state_keys']}、"
    f"emerging={_installed['emerging_entries']}"
)
