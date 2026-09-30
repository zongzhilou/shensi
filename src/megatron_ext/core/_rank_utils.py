# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Low-level rank utilities with minimal dependencies to avoid circular imports.

增量说明：本文件 = FL fork 的同名文件（逐字）+ mcore main 新增的默认日志 rank 支持
（`_DEFAULT_LOG_RANKS` / `set_default_log_ranks` / `get_default_log_ranks`）。Bridge@a393057 与
verl 是按 main 的 API 写的（`megatron.bridge.models.megatron_mimo.*` 里
`from megatron.core.utils import set_default_log_ranks`），FL fork 还没有这些名字，import 期就炸。
`log_single_rank` 保持 fork 的签名（`(logger, *args, rank=..., **kwargs)`，与 main 的新签名对调用方
等价），只把"默认 rank"从写死的 0 换成可配置的集合——默认仍是 {0}，行为不变。
"""

import logging
import os
import warnings
from collections.abc import Iterable
from typing import Any, Optional

import torch

from megatron.core._slurm_utils import resolve_slurm_rank, resolve_slurm_world_size


def safe_get_rank() -> int:
    """Get the distributed rank safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_rank() (if initialized)
    2. RANK environment variable (torchrun/torchelastic)
    3. SLURM_PROCID environment variable (SLURM)
    4. Default: 0 (with warning)

    Returns:
        int: The rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    # If torch.distributed is not initialized, try to read environment variables.
    try:
        if "RANK" in os.environ:
            return int(os.environ["RANK"])

        slurm_rank = resolve_slurm_rank()
        if slurm_rank is not None:
            return slurm_rank

        warnings.warn(
            "Could not determine rank from torch.distributed, RANK, or SLURM_PROCID. "
            "Defaulting to rank 0."
        )
        return 0
    except (ValueError, TypeError):
        return 0


def safe_get_world_size() -> int:
    """Get the distributed world size safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_world_size() (if initialized)
    2. WORLD_SIZE environment variable (torchrun/torchelastic)
    3. SLURM_NTASKS environment variable (SLURM)
    4. Default: 1 (with warning)

    Returns:
        The total number of processes in the distributed job.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    slurm_world_size = resolve_slurm_world_size()
    if slurm_world_size is not None:
        return slurm_world_size

    warnings.warn(
        "Could not determine world size from torch.distributed, WORLD_SIZE, or SLURM_NTASKS. "
        "Defaulting to world size 1."
    )
    return 1


# Ranks that log_single_rank / warn_single_rank write on when the caller names no rank.
# 非共卡的 MIMO 会把视觉编码器放在 rank 0、语言模型放在 --mimo-llm-offset，所以单个 rank 只能
# 描述其中一个模型，默认集合要能被改。
_DEFAULT_LOG_RANKS: tuple[int, ...] = (0,)


def set_default_log_ranks(ranks: Iterable[int]) -> None:
    """Set the ranks that ``log_single_rank`` and ``warn_single_rank`` write on by default.

    Call once, after torch distributed is initialized and before the model is built, so
    that setup-time messages are covered. A call site that passes ``rank`` explicitly is
    unaffected.

    Args:
        ranks: Ranks to log on. Duplicates are ignored.
    """
    global _DEFAULT_LOG_RANKS
    _DEFAULT_LOG_RANKS = tuple(sorted(set(ranks)))


def get_default_log_ranks() -> tuple[int, ...]:
    """Return the ranks that the single-rank logging helpers write on by default."""
    return _DEFAULT_LOG_RANKS


def log_single_rank(
    logger: logging.Logger, *args: Any, rank: Optional[int] = None, **kwargs: Any
) -> None:
    """Log a message only on a single rank.

    If torch distributed is initialized, write log on only one rank.

    Args:
        logger: The logger to write the logs.
        *args: All logging.Logger.log positional arguments.
        rank: The rank to write on. Defaults to None, meaning the ranks configured by
            ``set_default_log_ranks`` (rank 0 unless it has been changed).
        **kwargs: All logging.Logger.log keyword arguments.
    """
    current_rank = safe_get_rank()
    should_log = current_rank == rank if rank is not None else current_rank in _DEFAULT_LOG_RANKS
    if should_log:
        logger.log(*args, **kwargs)


def warn_single_rank(
    message: str,
    category: type[Warning] = UserWarning,
    stacklevel: int = 2,
    rank: Optional[int] = None,
) -> None:
    """Issue a warning only on a single rank.

    Use for warnings that describe a property of the job rather than of the calling rank,
    such as deprecated settings and experimental-API notices.
    """
    if torch.distributed.is_initialized():
        current_rank = torch.distributed.get_rank()
    else:
        # safe_get_rank 在完全找不到 rank 时会告警；这里只是要把这条去重，不该再换成另一条。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current_rank = safe_get_rank()

    should_warn = current_rank == rank if rank is not None else current_rank in _DEFAULT_LOG_RANKS
    if should_warn:
        warnings.warn(message, category=category, stacklevel=stacklevel + 1)
