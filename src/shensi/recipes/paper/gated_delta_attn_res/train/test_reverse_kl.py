#!/usr/bin/env python3
"""reverse KL 的正确性单测：与解析解对拍 + 补丁生效 + 前向 KL 未被破坏。

python -m shensi.recipes.paper.gated_delta_attn_res.train.test_reverse_kl
"""

from __future__ import annotations

import torch

from shensi.recipes.paper.gated_delta_attn_res.train.reverse_kl import (
    install_reverse_kl,
    installed,
    reverse_kl_from_topk,
)

OK = True


def check(name: str, ok: bool, detail: str = "") -> None:
    global OK
    OK &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")


def main() -> int:
    torch.manual_seed(0)
    V, S, B = 8, 3, 2
    student = torch.randn(B, S, V) * 2.0
    teacher_lp = torch.log_softmax(torch.randn(B, S, V) * 2.0, dim=-1)
    # teacher 的 top-K 覆盖全部词表（k=V）：支持集之外质量≈0，幽灵档贡献可忽略
    k = V
    top_lp, top_idx = teacher_lp.topk(k, dim=-1)  # [B,S,K]

    from megatron.training.distillation import cached_logits_loss as C

    # 1) 解析解：float64 全词表的 KL(student‖teacher)
    s_lp = torch.log_softmax(student.double(), dim=-1)
    t_lp = teacher_lp.double()
    ref = (s_lp.exp() * (s_lp - t_lp)).sum(-1)  # (B, S)
    # mcore 的约定：输入/输出都是 [S, B, ...] / [B, S]——把输入转置过去再比
    got = reverse_kl_from_topk(
        student.transpose(0, 1),
        top_lp.transpose(0, 1),
        top_idx.transpose(0, 1),
        1,
        0,
        None,
        add_ghost_token=True,
    )
    d = (got.double() - ref).abs().max().item()
    check("reverse KL == 解析解（全覆盖支持集）", d <= 1e-5, f"max|diff| = {d:.3e}")

    # 2) 前向 KL（mcore 原版）同输入下应当 ≈ 解析前向 KL —— 顺带证明我们对齐了同一个近似
    fwd = C.topk_kl_div(
        student.transpose(0, 1),
        top_lp.transpose(0, 1),
        top_idx.transpose(0, 1),
        1,
        0,
        None,
        add_ghost_token=True,
    )
    ref_f = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
    d2 = (fwd.double() - ref_f).abs().max().item()
    check("前向 KL（mcore 原版）== 解析解", d2 <= 1e-5, f"max|diff| = {d2:.3e}")

    # 3) 方向确实不同（非退化）
    gap = (got - fwd).abs().max().item()
    check("reverse 与前向不是同一个量", gap > 1e-3, f"max|diff| = {gap:.3e}")

    # 4) 补丁：install 后模块级名字被接管、可幂等、原函数仍在
    before = C.topk_kl_div
    install_reverse_kl()
    install_reverse_kl()
    check(
        "install_reverse_kl 幂等且已接管",
        installed() and C.topk_kl_div is reverse_kl_from_topk,
        f"before={before.__name__} now={C.topk_kl_div.__name__}",
    )

    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
