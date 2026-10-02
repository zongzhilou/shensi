"""批量白化：把多次同形状的逆平方根请求合成一次批量求解。"""

from __future__ import annotations

import torch


def batched_whitening(values_list: list[torch.Tensor], mode: str = "full", ridge: float = 1e-3):
    """把同形状的多组白化请求合成一次批量求解（eigh 批量档）。"""
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
    stacked = torch.stack(covs)
    evals, evecs = torch.linalg.eigh(stacked)
    return evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(-1, -2)
