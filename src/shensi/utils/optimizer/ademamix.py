"""AdEMAMix 的注册适配：数学与状态都在 pytorch_optimizer 的 AdEMAMix 里，这里补三件事。

1. 上游 `emerging_optimizers.registry` 里没有 pytorch_optimizer 的类，先按名字注册一次；
2. mcore 的 emerging 表是**在它自己 import 时**从 provider 注册表扫一遍建好的，我们注册得晚，
   所以再往 mcore 的 `_EMERGING_OPTIMIZERS` 里补一条 `EmergingOptimizerEntry`（不改 mcore 源码）；
3. mcore 的 checkpoint 初始化会调 `_init_group(group, skip_non_grad_params=False)`，
   库自己的入口叫 `init_group`，转一下即可；
注意一条上游约束：AdEMAMix 只能当**主体**优化器（`--optimizer ademamix`，走 emerging 路径），
不能当 `--muon-scalar-optimizer`。mcore 只把"主优化器"那组交给 emerging 表，标量那组一律落到
`_get_megatron_optimizer_based_on_param_groups`，而那里只认 adam/adamw/lion/sgd
（`optimizer/__init__.py` 里那段注释写的就是这个意思）。所以配方默认的标量腿是 Lion，
AdEMAMix 走单独的对照档。
"""

from __future__ import annotations

import torch
from emerging_optimizers import registry
from pytorch_optimizer import AdEMAMix as _AdEMAMix

__all__ = ["AdEMAMix", "install_mcore_entry"]


@registry.register_optimizer("ademamix")
class AdEMAMix(_AdEMAMix):
    """pytorch_optimizer 的 AdEMAMix（arXiv:2409.03137），补 mcore 需要的 `_init_group`。"""

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """按需建状态；torch_dist checkpoint 下无梯度参数也要有状态，用临时零梯度走库的入口。"""
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


def _config_to_kwargs(config, model_chunks=None, pg_collection=None) -> dict:
    """把 mcore 的 OptimizerConfig 映射成 AdEMAMix 的构造参数。

    `ademamix_betas` / `ademamix_beta3` 这几个字段不是 mcore 的 OptimizerConfig 自带项，
    由 build 侧从 `--ademamix-*` 挂上来（见 `recipes/shensi/train/builders.py`）。
    """
    betas = getattr(config, "ademamix_betas", None)
    if betas is None or len(tuple(betas)) != 3:
        beta3 = getattr(config, "ademamix_beta3", None)
        betas = (
            float(config.adam_beta1),
            float(config.adam_beta2),
            float(beta3 if beta3 else 0.9999),
        )
    kwargs = {
        "lr": float(config.lr),
        "betas": tuple(float(b) for b in betas),
        "alpha": float(getattr(config, "ademamix_alpha", 5.0) or 5.0),
        "weight_decay": float(config.weight_decay),
    }
    t_alpha_beta3 = getattr(config, "ademamix_t_alpha_beta3", None)
    if t_alpha_beta3:
        kwargs["t_alpha_beta3"] = int(t_alpha_beta3)
    return kwargs


def install_mcore_entry() -> bool:
    """把 AdEMAMix 补进 mcore 的 emerging 表；返回是否真的补了。"""
    from megatron.core.optimizer import emerging_optimizers as mco_eo

    table = getattr(mco_eo, "_EMERGING_OPTIMIZERS", None)
    if table is None or "ademamix" in table:
        return False
    table["ademamix"] = mco_eo.EmergingOptimizerEntry(
        optimizer_cls=AdEMAMix,
        config_to_kwargs=_config_to_kwargs,
    )
    return True


if install_mcore_entry():
    print("[shensi][optim] AdEMAMix 已补进 mcore 的 emerging 表（--optimizer ademamix 可用）")
