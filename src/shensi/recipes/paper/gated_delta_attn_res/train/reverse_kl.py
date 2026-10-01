"""OPD 的 reverse KL（MiniCPM5 口径）：把 mcore 的前向 KL 换成 KL(student‖teacher)。

mcore 的离线 KD（``megatron.training.distillation.cached_logits_loss``）计算的是

    KL(teacher ‖ student) = Σ_k p_T(k)·[log p_T(k) − log p_S(k)]

MiniCPM5 的 OPD 用的是 **reverse KL**：``KL(student ‖ teacher)``。两者在同一个**支持集**
（teacher 的 top-K ∪ 一个"幽灵"残差 token）上只差权重侧：把
``p_T·(log p_T − log p_S)`` 换成 ``p_S·(log p_S − log p_T)``。

本文件提供与 ``topk_kl_div`` **同签名同返回**（``[B, S]``）的 ``reverse_kl_from_topk``，
并把整个缓存读取 / TP-aware softmax / 迭代推进的管线**原样留给 mcore**：把模块级的
``topk_kl_div`` 名字换成我们的函数即可（``install_reverse_kl()``，幂等、可逆）。
``train_gdar.py`` 的 ``--logits-load-reverse-kl`` 在构造 KD 损失前调用它。

支持集之外的概率质量在两个方向里都按"幽灵 token"聚成一档处理（与 mcore 的
``add_ghost_token=True`` 同一近似）；TP>1 时幽灵档只在 rank0 计入，同样与 mcore 一致。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["reverse_kl_from_topk", "install_reverse_kl", "installed"]

_SENTINEL = None  # 从 mcore 侧惰性取（常量名随版本可能变）


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
    """``KL(student ‖ teacher)``，签名与返回（``[B, S]``）与 mcore 的 ``topk_kl_div`` 一致。

    逐行对拍 mcore 版本：student 的 TP-aware softmax、按 teacher 的 top-K 索引 gather、
    幽灵残差档（TP>1 只在 rank0 计）都**一模一样**；只有最后两行换成以 student 概率为权。
    """
    sentinel = _sentinel()
    student_logits = student_logits.float()
    teacher_topk_logprobs = teacher_topk_logprobs.float()

    # ---- TP-aware student softmax / log-softmax（与 mcore 相同）----
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

    # ---- gather student log-probs at teacher's top-K positions（与 mcore 相同）----
    local_vocab_size = student_logits.size(-1)
    offset = local_vocab_size * tp_rank
    mask = (
        (teacher_topk_indices >= offset)
        & (teacher_topk_indices < offset + local_vocab_size)
        & (teacher_topk_logprobs != sentinel)
    )
    teacher_local_indices = (teacher_topk_indices - offset).clamp(0, local_vocab_size - 1)
    student_topk_logprobs = torch.gather(student_logprobs, -1, teacher_local_indices)

    # ---- 幽灵残差档（与 mcore 相同：student/teacher 各补一档支持集外的质量）----
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

    # ---- reverse KL：以 **student** 概率为权（唯一与 mcore 不同的两行）----
    student_probs = student_topk_logprobs.exp() * mask
    kl_div = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
    kl_loss = torch.sum(mask * kl_div, dim=-1)
    return kl_loss.transpose(0, 1).contiguous()  # [S, B] -> [B, S]


_ORIGINAL = None


def install_reverse_kl() -> bool:
    """把 ``cached_logits_loss.topk_kl_div`` 换成 reverse KL 版（幂等）。

    管线（缓存读取 / 迭代 / DataLoader / TP 归一）全部沿用 mcore：只换这一个**模块级名字**，
    ``CachedLogitsKDLoss`` 内部对它的调用点（同一模块内的裸调用）随之改变。
    """
    global _ORIGINAL
    from megatron.training.distillation import cached_logits_loss as C

    if _ORIGINAL is None:
        _ORIGINAL = C.topk_kl_div
    if C.topk_kl_div is reverse_kl_from_topk:
        return True
    C.topk_kl_div = reverse_kl_from_topk
    return True


def installed() -> bool:
    from megatron.training.distillation import cached_logits_loss as C

    return C.topk_kl_div is reverse_kl_from_topk
