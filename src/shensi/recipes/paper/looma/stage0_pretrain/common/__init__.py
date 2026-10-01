"""预训练段（PT-1/PT-2 与中训练两段）共用的入口。"""

from .prep import prep_main
from .train import train_main

__all__ = ["prep_main", "train_main"]
