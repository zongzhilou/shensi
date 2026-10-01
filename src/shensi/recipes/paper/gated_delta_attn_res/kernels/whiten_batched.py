"""批量 eigh：把多次白化的协方差叠成一个批次、一次 `torch.linalg.eigh` 算完。

**能接在哪、不能接在哪（实测证据）**：把 `_whitening_transform` 包起来记录调用序列，一次前向
（19 层 × 1024 宽、seq 512）里白化被调 **39 次**，形状序列是

    S=2 ×8、S=3 ×8、S=4 ×8、S=5 ×8、S=6 ×7      （S = 来源数；每次 T=512）

也就是"同形状的 8 次连成一串"——它们是**同一个深度组里 4 层的两个子层**（注意力 + MLP）的读，
层与层之间**严格依赖**（上一层的写决定下一层读的 values）。所以：

* **能批**：把同一调用点上的多次请求攒起来（需要调用方先把请求交出来）——`batched_whitening()`
  就是这个 API，它把 [B, d, d] 的协方差一次解完；
* **不能批**（现状）：同一条深度递推里逐层调用时，后面的 calls 需要前面 calls 的**输出**，
  没有静止的批次点。要真接上，得让连接把"收集请求 → 批量白化 → 回填"拆成两段（那会改动移植件的
  调用结构，风险与工作量都不小）——本文件把收益先量出来，接线留作明确的下一步。

实测（本机，d=1024、fp32，39 次请求）：逐个 388.8 ms → 一次批量 **204.4 ms（1.90×）**，
单项误差 0.0（同一份数学）。见 `bench_whiten.py` 的"批量白化"一段。
"""

from __future__ import annotations

import torch


def batched_whitening(values_list: list[torch.Tensor], mode: str = "full", ridge: float = 1e-3):
    """一次算完多个张量的白化矩阵：返回 [B, d, d]（full）或 [B, d]（diag）。

    `values_list[i]` 形状 [T, S, H]（可不等长）；语义与逐个 `_whitening_transform` 完全一致。
    """
    d = values_list[0].reshape(-1, values_list[0].shape[-1]).shape[-1]
    if mode == "diag":
        return torch.stack(
            [torch.rsqrt(v.reshape(-1, d).float().pow(2).mean(dim=0) + ridge) for v in values_list]
        )
    if mode != "full":
        raise ValueError(f"batched_whitening 只接 'diag'/'full'，收到 {mode!r}")
    eye = torch.eye(d, device=values_list[0].device, dtype=torch.float32) * ridge
    covs = []
    for v in values_list:
        s = v.reshape(-1, d).float().detach()
        cov = (s.transpose(0, 1) @ s) / s.shape[0]
        covs.append(cov + eye)
    stacked = torch.stack(covs)  # [B, d, d]
    evals, evecs = torch.linalg.eigh(stacked)
    return evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(-1, -2)
