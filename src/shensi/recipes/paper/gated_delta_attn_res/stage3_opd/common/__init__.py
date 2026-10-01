"""OPD 的公共件：训练入口（`--teacher-cache` → mcore 原生 KD）与 rollout 语料准备。"""

from .prep import prep_main
from .train import train_main

__all__ = ["prep_main", "train_main"]
