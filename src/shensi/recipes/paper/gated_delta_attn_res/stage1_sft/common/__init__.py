"""监督微调段的公共件：训练入口（含合成 jsonl 冒烟）与 messages jsonl 语料准备。"""

from .prep import prep_main
from .train import smoke_jsonl, train_main

__all__ = ["prep_main", "smoke_jsonl", "train_main"]
