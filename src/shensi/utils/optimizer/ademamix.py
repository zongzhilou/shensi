"""AdEMAMix 的注册适配：数学与状态都在 pytorch_optimizer 的 AdEMAMix 里，这里只补两件事。

上游 `megatron/core/optimizer` 按名字从 `emerging_optimizers.registry` 取标量优化器
（`muon_scalar_optimizer` 就是按这个名字解析的），而 pytorch_optimizer 的类不在那个注册表里，
所以在这里注册一次名字；另外 mcore 的 checkpoint 初始化会调
`_init_group(group, skip_non_grad_params=False)`，库自己的入口叫 `init_group`，转一下即可。
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
