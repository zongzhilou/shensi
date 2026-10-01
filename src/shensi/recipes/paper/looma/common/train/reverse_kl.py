"""OPD 蒸馏用的 reverse KL：``KL(student‖teacher)`` 的 top-K 实现与运行时接管。

``reverse_kl_from_topk`` 与 ``cached_logits_loss.topk_kl_div`` 同签名，由
``install_reverse_kl`` 原地替换进 KD 链路后生效。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["reverse_kl_from_topk", "install_reverse_kl", "installed"]

_SENTINEL = None


def _sentinel():
    """惰性取回 sentinel：teacher 在该位置没有提供 logprob 时的占位值。"""
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
    """按 teacher 的 top-K 算 reverse KL，返回逐 token loss（形状 ``[T, B]``）。

    student logits 按 TP 分片：max / sum 各做一次 all-reduce 得到全局 log-softmax，再把
    teacher 的全局 top-K 索引映射到本 rank 并 gather 出对应 logprob，落在他 rank 的位置
    由 mask 置零（各 rank 只算自己那一份，求和需调用方跨 TP 归约）。``add_ghost_token=True``
    时给两侧各补一项 top-K 之外的尾部残差，使分布重新归一到 1，且只由 TP rank 0 计入。
    """
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
    """把 ``cached_logits_loss.topk_kl_div`` 换成 ``reverse_kl_from_topk``；幂等，可重复调。"""
    global _ORIGINAL
    from megatron.training.distillation import cached_logits_loss as C

    if _ORIGINAL is None:
        _ORIGINAL = C.topk_kl_div  # 留一份原实现，便于对照
    if C.topk_kl_div is reverse_kl_from_topk:
        return True
    C.topk_kl_div = reverse_kl_from_topk
    return True


def installed() -> bool:
    """当前 KD 链路里的 ``topk_kl_div`` 是否已经是 reverse KL 实现。"""
    from megatron.training.distillation import cached_logits_loss as C

    return C.topk_kl_div is reverse_kl_from_topk
