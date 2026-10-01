"""预训练段的公共件：PT-1/PT-2 与中训练两段共用的训练入口与 bin/idx 语料准备。"""

from .prep import prep_main, prepare
from .train import train_main

__all__ = ["prepare", "prep_main", "train_main"]
