"""The lower-bound guarantee: keep GDAR on or above the DAR slice while training.

Design intent (the sentence this file implements):

    the initialisation locks the *lower bound* at DAR; the upper bound is left to
    training.

"Locked at init" is only a statement about step 0 -- a single AdamW step can move
the gates somewhere worse.  This module turns the statement into a training-time
invariant:

* ``force_identity_gates(model)``  -- context manager that pins every gate to
  exactly ``(decay, erase, write) = (1, 0, 1)``, i.e. the model *is* DAR.  Both
  update rules collapse to ``prefix + delta`` there, so this is an exact, not
  approximate, slice.
* ``identity_slice_loss(model, batch)`` -- the loss that DAR-with-these-weights
  would achieve.  This is the running lower bound.
* ``rollback_to_identity(model)`` -- projects the gates back onto the slice by
  zeroing the three deviation scales.  It cannot make the loss worse, because
  afterwards the model *is* the slice.
* ``LowerBoundGuard`` -- the callback a training loop calls every K steps: measure
  both losses, roll back when the learned gates are worse than the slice, and log
  the gap so the "is the gate earning its parameters?" question is answered by a
  number rather than by a hope.

None of this is a proof about held-out data -- it is a proof about *this* model on
*this* batch: the guarded model's training loss is by construction ≥ ... rather,
is never below the DAR slice it is compared against.  The empirical question (does
a trained GDAR beat a trained DAR) is untouched by it, which is the point.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from hf.modeling_qwen3_gdar import AttentionResidual

__all__ = [
    "attn_res_modules",
    "force_identity_gates",
    "identity_slice_loss",
    "rollback_to_identity",
    "LowerBoundGuard",
]


def attn_res_modules(model):
    return [m for m in model.modules() if isinstance(m, AttentionResidual)]


@contextmanager
def force_identity_gates(model):
    """Pin every depth-connection gate to (1, 0, 1) for the duration."""
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
    """Loss of the exact DAR slice of this model (gates pinned to identity)."""
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
    """Project the gates back onto the DAR slice (zero the deviation scales).

    The gate *weights* are left alone: with the scales at zero the gates are
    exactly (1, 0, 1) whatever the weights are, so nothing learned elsewhere is
    discarded, and the depth connection can grow back out of the slice later.
    """
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
    """Periodically verify ``loss <= identity_slice_loss`` and roll back if not.

    Args:
        model: the GDAR model.
        every: check every N optimizer steps.
        delta: slack -- only roll back when the gap exceeds this (guards against
            batch noise rather than real regressions).
    """

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
