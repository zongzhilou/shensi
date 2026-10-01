"""Shensi 的 SFT 数据集：**一条对话一条样本 + 右侧 padding**，不做 THD 打包。"""

from __future__ import annotations

import torch
from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset


class ShensiSFTDataset(SFTDataset):
    """Shensi 的 SFT 数据集（messages jsonl，单条对话一条样本）。"""

    def __getitem__(self, idx: int):
        tokenizer = self.config.tokenizer
        seq_length = self.config.sequence_length

        merged = self.dataset[int(self.indices[idx % len(self.indices)])]
        conversations = self._split_conversations(merged)
        conversation = conversations[idx % len(conversations)]

        tokens, targets = tokenizer.tokenize_conversation(
            conversation, return_target=True, add_generation_prompt=False
        )
        tokens = tokens.tolist()[: seq_length + 1]
        targets = targets.tolist()[: seq_length + 1]

        pad = tokenizer.pad
        pad_len = seq_length + 1 - len(tokens)
        tokens.extend([pad] * pad_len)
        targets.extend([pad] * pad_len)

        # 与上游同一套位移：输入 [:-1]，标签 [1:]（预测下一个 token）
        input_ids = torch.tensor(tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(targets[1:], dtype=torch.int64)

        loss_mask = torch.ones(seq_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0  # padding 段
        loss_mask[labels == IGNORE_INDEX] = 0.0  # prompt 段

        return {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(seq_length, dtype=torch.int64),
        }
