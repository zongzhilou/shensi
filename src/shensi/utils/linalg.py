"""Small symmetric eigensolves with an exact CPU compatibility path on Ascend."""

import os

import torch


def symmetric_eigh(matrix: torch.Tensor):
    """Keep CUDA/CPU native; NPU defaults to the same LAPACK formula on CPU.

    GDAR calls this on detached covariance matrices. Only the eigensolve moves;
    eigenvalues/vectors return to the input device. This is deliberately slower
    than a native kernel and does not replace whitening with an approximation.
    Set SHENSI_NPU_EIGH=native only on a stack supporting torch.linalg.eigh.
    Mirrored in the standalone HF model so exported models need no shensi install.
    """
    mode = os.environ.get("SHENSI_NPU_EIGH", "cpu")
    if mode not in ("cpu", "native"):
        raise ValueError("SHENSI_NPU_EIGH must be cpu or native")
    if matrix.device.type == "npu" and mode == "cpu":
        values, vectors = torch.linalg.eigh(matrix.to("cpu"))
        return values.to(matrix.device), vectors.to(matrix.device)
    return torch.linalg.eigh(matrix)
