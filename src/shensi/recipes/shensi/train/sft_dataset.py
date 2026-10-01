"""Shensi 的 SFT 数据集：**一条对话一条样本 + 右侧 padding**，不做 THD 打包。

上游的 `SFTDataset` 会把多条对话打进一条 `sequence_length` 的样本并给出 `cu_seqlens`
（THD 打包）。但我们的主干注意力是上游的 CSA/HCA（`DSv4HybridAttention`），它明确断言
`packed_seq_params is None`——打包序列在 Shensi 上走不通，所以这一档换成不打包的口径。

保留的：上游的 tokenize / loss mask 语义（同一个 `SFTTokenizer.tokenize_conversation`、同一套
targets 与 `IGNORE_INDEX`，prompt 段与被 padding 的段都不算 loss）；与上游一致的 `[:-1]` / `[1:]`
位移。换掉的：多条对话打成一包 → 一条对话 + 右 padding，且**不产出 cu_seqlens**。

采样：低层样本的 `merged_conversations` 先按 system 边界切段，取第 `idx % 段数` 段，保证每个
对话都会被采到（上游是把它们全打进一条样本）。

右 padding 对因果注意力无害：有效 token 看不到后面的 pad，pad 段本身也被 loss_mask 排掉。
"""

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
