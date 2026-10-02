# 预训练（stable → decay）

本段两档：PT-1 stable 用恒定学习率建立语言能力，PT-2 decay 在高质量子集上退火收尾。入口与配置
都是薄件，实现在 `../../common/train_pt.py` 与 `../../common/prep_pt.py`。

| 档 | 是什么 |
|---|---|
| `default` | PT-1 stable：9B tokens(90%) / seq 2048 / LR 6e-4 恒定 / warmup 2% |
| `decay` | PT-2 decay：1B tokens(10%) / cosine 6e-4 → 6e-5 / `data_blend_decay.json` |

## 快速开始

```bash
python data_prep.py --prepare --config default    # 语料 → bin/idx
python train.py --tokens 9e9                      # PT-1
python data_prep.py --prepare --config default --blend data_blend_decay.json
python train.py --config decay --tokens 1e9 --load <stable 检查点>
python train.py --smoke                           # tiny 5 步冒烟
```

## 判据

`python train.py --smoke` 5 步跑通、检查点落盘；早停默认开（`lm loss value` 平台后按成功收尾）。

## 下一步

- [预训练与中训练总览](../README.md)
- [中训练](../stage2_midtrain/README.md) —— 从 PT-2 接续的下一段
