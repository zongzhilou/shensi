"""优化器旋钮：AdaMuon 与 AdEMAMix / GrokFastAdamW 的参数接入。"""

from __future__ import annotations

from typing import Any

SCALAR_OPTIMIZER_KWARG_FIELDS = (
    "ademamix_betas",
    "ademamix_alpha",
    "ademamix_beta3",
    "ademamix_t_alpha_beta3",
    "grokfast_alpha",
    "grokfast_lamb",
    "grokfast_after_step",
)

_SCALAR_OPTIMIZER_NAMES = ("ademamix", "grokfastadamw")

_EXTRA_SCALAR_NAMES = _SCALAR_OPTIMIZER_NAMES


def add_scalar_optimizer_args(group) -> None:
    """把标量腿优化器（AdEMAMix / GrokFastAdamW）的参数注册进 mcore 解析器。"""
    group.add_argument(
        "--gdar-scalar-optimizer",
        default=None,
        help="标量腿优化器（ademamix / grokfastadamw / adam / lion）",
    )
    group.add_argument(
        "--ademamix-betas",
        nargs="+",
        type=float,
        default=None,
        help="AdEMAMix 的 (beta_fast, beta2, beta_slow)",
    )
    group.add_argument("--ademamix-alpha", type=float, default=None, help="AdEMAMix 的 alpha")
    group.add_argument(
        "--ademamix-beta3", type=float, default=None, help="AdEMAMix 的 beta3（单值写法）"
    )
    group.add_argument(
        "--ademamix-t-alpha-beta3",
        type=float,
        default=None,
        help="alpha 的 warmup 步数（t_alpha_beta3）",
    )
    group.add_argument("--grokfast-alpha", type=float, default=None, help="GrokFast 的 alpha")
    group.add_argument("--grokfast-lamb", type=float, default=None, help="GrokFast 的 lamb")
    group.add_argument(
        "--grokfast-after-step", type=float, default=None, help="GrokFast 的 after_step"
    )


def extend_scalar_optimizer_choices(parser) -> list[str]:
    """扩展 ``--muon-scalar-optimizer`` 的可选值（mcore 白名单默认只有 adam/lion）。"""
    added: list[str] = []
    for action in parser._actions:  # noqa: SLF001
        if action.dest == "muon_scalar_optimizer" and isinstance(action.choices, list):
            for name in _EXTRA_SCALAR_NAMES:
                if name not in action.choices:
                    action.choices.append(name)
                    added.append(name)
    return added


def _attach_from_args(config: Any, args: Any) -> list[str]:
    attached: list[str] = []
    for name in SCALAR_OPTIMIZER_KWARG_FIELDS:
        value = getattr(args, name, None)
        if value is None:
            continue
        setattr(config, name, tuple(value) if isinstance(value, list) else value)
        attached.append(name)
    scalar = str(getattr(config, "muon_scalar_optimizer", "") or "").lower()
    if attached and scalar in _SCALAR_OPTIMIZER_NAMES:
        print(
            f"[gdar][optim] {scalar} 超参已挂到 OptimizerConfig："
            f"{ {n: getattr(config, n) for n in attached} }",
            flush=True,
        )
    return attached


def attach_to_container(container: Any, args: Any) -> list[str]:
    """把命令行上的标量腿旋钮落进 mcore 的 OptimizerConfig。"""
    opt_cfg = getattr(container, "optimizer", None)
    if opt_cfg is None:
        return []
    chosen = getattr(args, "gdar_scalar_optimizer", None)
    if chosen:
        opt_cfg.muon_scalar_optimizer = str(chosen)
        print(f"[gdar][optim] 标量腿={chosen}（--gdar-scalar-optimizer）", flush=True)
    return _attach_from_args(opt_cfg, args)
