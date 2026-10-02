"""恒等保障工具：门退化到恒等、恒等切片损失与回退检查。"""


from __future__ import annotations

from contextlib import contextmanager

import torch

from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (
    AttentionResidual,
)

__all__ = [
    "attn_res_modules",
    "force_identity_gates",
    "identity_slice_loss",
    "rollback_to_identity",
    "LowerBoundGuard",
]


def attn_res_modules(model):
    """列出模型里全部连接模块。"""
    return [m for m in model.modules() if isinstance(m, AttentionResidual)]


@contextmanager
def force_identity_gates(model):
    """把连接门推到恒等（用于对照与回退）。"""
    modules = attn_res_modules(model)
    try:
        for m in modules:
            m._force_identity = True
        yield
    finally:
        for m in modules:
            m._force_identity = False


@torch.no_grad()
def identity_slice_loss(model, input_ids, labels=None, **kwargs) -> float:
    """恒等切片损失：约束连接在初始化附近不偏离恒等。"""
    was_training = model.training
    model.eval()
    try:
        with force_identity_gates(model):
            out = model(
                input_ids=input_ids, labels=labels if labels is not None else input_ids, **kwargs
            )
        return float(out.loss)
    finally:
        model.train(was_training)


@torch.no_grad()
def rollback_to_identity(model) -> None:
    """把模型回退到恒等初始化。"""
    for m in attn_res_modules(model):
        if getattr(m, "gate_param", "") == "deviation":
            for scale in (
                getattr(m, "decay_scale", None),
                getattr(m, "erase_scale", None),
                getattr(m, "write_scale", None),
            ):
                if scale is not None:
                    scale.zero_()


class LowerBoundGuard:

    """恒等下界看门狗：越界时报警或回退。"""
    def __init__(self, model, every: int = 200, delta: float = 0.0):
        self.model = model
        self.every = max(1, int(every))
        self.delta = float(delta)
        self.history: list[dict] = []

    def maybe_step(self, step: int, input_ids, labels=None) -> dict | None:
        if step % self.every != 0:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                current = float(
                    self.model(
                        input_ids=input_ids, labels=labels if labels is not None else input_ids
                    ).loss
                )
            slice_loss = identity_slice_loss(self.model, input_ids, labels)
        finally:
            self.model.train(was_training)

        gap = current - slice_loss
        rolled = gap > self.delta
        if rolled:
            rollback_to_identity(self.model)
        record = {
            "step": int(step),
            "loss": current,
            "dar_slice": slice_loss,
            "gap": gap,
            "rolled_back": rolled,
        }
        self.history.append(record)
        return record
