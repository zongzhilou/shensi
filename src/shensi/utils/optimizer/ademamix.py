"""AdEMAMix 的注册适配：数学与状态都在 pytorch_optimizer 的 AdEMAMix 里，这里补三件事。

1. 上游 `emerging_optimizers.registry` 里没有 pytorch_optimizer 的类，先按名字注册一次；
2. mcore 的 checkpoint 初始化会调 `_init_group(group, skip_non_grad_params=False)`，
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

__all__ = ["AdEMAMix"]


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
