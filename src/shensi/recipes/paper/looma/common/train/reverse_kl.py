"""reverse KL 的蒸馏损失实现（与 mcore 的 topk_kl_div 同签名）。"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["reverse_kl_from_topk", "install_reverse_kl", "installed"]

_SENTINEL = None


def _sentinel():
    global _SENTINEL
    if _SENTINEL is None:
        from megatron.training.distillation import cached_logits_loss as C

        _SENTINEL = C.CACHED_LOGITS_LOGPROB_SENTINEL
    return _SENTINEL


def reverse_kl_from_topk(
    student_logits: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_group: dist.ProcessGroup,
    add_ghost_token: bool = False,
) -> torch.Tensor:
    """Reverse KL 的蒸馏损失：与 mcore 的 topk_kl_div 同签名同返回。"""
    sentinel = _sentinel()
    student_logits = student_logits.float()
    teacher_topk_logprobs = teacher_topk_logprobs.float()

    logits_max, _ = student_logits.max(dim=-1, keepdim=True)
    if tp_size > 1:
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    student_logits -= logits_max.detach()
    sum_exp = student_logits.exp().sum(dim=-1, keepdim=True)
    if tp_size > 1:
        from megatron.core import tensor_parallel as tp_dist

        sum_exp = tp_dist.dist_nn.functional.all_reduce(
            sum_exp, op=dist.ReduceOp.SUM, group=tp_group
        )
    student_logprobs = student_logits - sum_exp.log()

    local_vocab_size = student_logits.size(-1)
    offset = local_vocab_size * tp_rank
    mask = (
        (teacher_topk_indices >= offset)
        & (teacher_topk_indices < offset + local_vocab_size)
        & (teacher_topk_logprobs != sentinel)
    )
    teacher_local_indices = (teacher_topk_indices - offset).clamp(0, local_vocab_size - 1)
    student_topk_logprobs = torch.gather(student_logprobs, -1, teacher_local_indices)

    if add_ghost_token:
        eps = 1e-8
        student_topk_logprobs_exp = student_topk_logprobs.exp() * mask
        student_topk_exp_sum = student_topk_logprobs_exp.sum(dim=-1, keepdim=True)
        if tp_size > 1:
            from megatron.core import tensor_parallel as tp_dist

            student_topk_exp_sum = tp_dist.dist_nn.functional.all_reduce(
                student_topk_exp_sum, op=dist.ReduceOp.SUM, group=tp_group
            )
        student_residual = torch.log((1.0 - student_topk_exp_sum).clamp(min=eps))
        teacher_residual = torch.log(
            (1.0 - teacher_topk_logprobs.exp().sum(dim=-1, keepdim=True)).clamp(min=eps)
        )
        student_topk_logprobs = torch.cat([student_topk_logprobs, student_residual], dim=-1)
        teacher_topk_logprobs = torch.cat([teacher_topk_logprobs, teacher_residual], dim=-1)
        mask = torch.cat([mask, mask.new_full((*mask.shape[:-1], 1), float(tp_rank == 0))], dim=-1)

    student_probs = student_topk_logprobs.exp() * mask
    kl_div = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
    kl_loss = torch.sum(mask * kl_div, dim=-1)
    return kl_loss.transpose(0, 1).contiguous()


_ORIGINAL = None


def install_reverse_kl() -> bool:
    """用本实现接管 mcore 模块级的 KD 损失名（幂等）。"""
    global _ORIGINAL
    from megatron.training.distillation import cached_logits_loss as C

    if _ORIGINAL is None:
        _ORIGINAL = C.topk_kl_div
    if C.topk_kl_div is reverse_kl_from_topk:
        return True
    C.topk_kl_div = reverse_kl_from_topk
    return True


def installed() -> bool:
    """当前是否已由本实现接管。"""
    from megatron.training.distillation import cached_logits_loss as C

    return C.topk_kl_div is reverse_kl_from_topk
