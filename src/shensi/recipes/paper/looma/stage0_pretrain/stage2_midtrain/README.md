# 中训练（能力强化 → 长文档）

本段两档：Mid-1 把代码/数学占比与序列长度拉到 4K，Mid-2 切长文语料、16K 序列、LR 再衰减一段。
入口与配置都是薄件，实现在 `../../common/train_pt.py` 与 `../../common/prep.py`。

| 档 | 是什么 |
|---|---|
| `default` | Mid-1：0.5B tokens(5%) / seq 4096 / LR 6e-5 恒定 / 代码 40% + 数学 30% + 高质量通用 30% |
| `mid2` | Mid-2：0.3B tokens(3%) / seq 16384 / cosine 6e-5 → 3e-5 / 长文 70% + 高质量通用 30% |

## 快速开始

```bash
python train.py --tokens 5e8 --load <PT-2 检查点>                    # Mid-1
python data_prep.py --prepare --config default --blend data_blend_mid2.json
python train.py --config mid2 --tokens 3e8 --load <Mid-1 检查点>     # Mid-2
python train.py --smoke                                              # tiny 5 步冒烟
```

## 判据

`python train.py --smoke` 5 步跑通、检查点落盘；早停默认开。

## 下一步

- [预训练与中训练总览](../README.md)
- [监督微调](../../stage1_sft/README.md) —— 从 Mid-2 接续
