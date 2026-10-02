# 阶段 0.2：中训练（Mid-1 能力强化 → Mid-2 长文档）

从 PT-2 接续，两步走：Mid-1 把代码/数学占比与序列长度拉到 4K（能力强化）；Mid-2 切长文语料、
16K 序列、LR 再衰减一段（分布适配）。预算与 2+2 拆分的依据见[阶段总览](../README.md)。

## 总览

| 组件 | 说明 |
|---|---|
| `data_prep.py` | 配比 → bin/idx |
| `train.py` | 训练入口（与预训练共用同一套公共件） |
| `test_train.py` | 集成测试（tiny 5 步） |
| `config/` | `default`（Mid-1）、`mid2`、`tiny`、`debug` |

## 档位

| 档 | 是什么 |
|---|---|
| `default` | Mid-1：0.5B tokens(5%) / seq 4096 / LR 6e-5 恒定 / 代码 40% + 数学 30% + 高质量子集 30% |
| `mid2` | Mid-2：0.3B tokens(3%) / seq 16384 / cosine 6e-5 → 3e-5 / 长文 70% + 高质量子集 30% |
| `debug` / `tiny` | 链路验证（seq 512 + 真实数据）/ mock 冒烟 |

## 快速开始

```bash
python data_prep.py --prepare --config default        # Mid-1 配比
python train.py --config default --tokens 5e8 --load <PT-2 检查点>
python data_prep.py --prepare --config mid2           # Mid-2 长文档
python train.py --config mid2 --tokens 3e8 --load <Mid-1 检查点>
python test_train.py
```

## 完整主跑（8B / 30B-A3B）

中训练只服务 **8B 四变体**与 **30B-A3B 门面**（序列加长后长上下文评测的 16K / 32K 档才有意义）：

```bash
for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  python train.py --config geoms/qwen3_8b --model-algo $algo --tokens 5e9 \
      --load <PT-2(8B, $algo) 检查点> --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo
  python train.py --config geoms/qwen3_8b --model-algo $algo --tokens 3e9 \
      --set train.model.seq_length=16384 \
      --load $SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo/ckpt \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid2/$algo
done
# 30B-A3B 门面：--config geoms/qwen3_30b_a3b；32K 档：--set train.model.seq_length=32768
```

## 判据

| 检查 | 判据 |
|---|---|
| `python test_train.py` | tiny 5 步：rc=0、到最后一 iter、`[after training is done]`、无 Traceback |

## 下一步

- [阶段 0 总览](../README.md) —— 段位设计与预算依据
- [预训练](../stage1_pretrain/README.md) —— Mid-1 接续的那一段
- [阶段 1：SFT](../../stage1_sft/README.md) —— 从 Mid-2 接续
