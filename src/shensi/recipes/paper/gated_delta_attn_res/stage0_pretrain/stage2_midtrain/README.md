# 阶段 0.2：中训练（Mid-1 能力强化 → Mid-2 长文档）

从 PT-2 接续，两步走：Mid-1 把代码/数学占比和序列长度拉到 4K（能力强化）；Mid-2 切
Ultra-FineWeb-L3 长文档、16K 序列、LR 再衰减一段（分布适配）。预算、配比与 2+2 拆分的依据
见[阶段总览](../README.md)。

## 总览

| 组件 | 说明 |
|---|---|
| `data_prep.py` | 混合中训练语料（含 UltraData-Code 分级）并 tokenize 成 bin/idx |
| `train.py` | Megatron-Core 中训练；`--model-algo` 选连接 |
| `test_train.py` | 集成测试（tiny 几何 5 步，按日志判 PASS/FAIL） |

> **早停默认开**：迭代数给到无限大；loss 平台后看门狗（metric `lm loss value`）收尾，按成功处理。

## 档位

| 档 | 是什么 |
|---|---|
| `default` | Mid-1 能力强化：0.5B tokens(5%) / seq 4096 / LR 6e-5 恒定 / Code 40% + Math 30% + UltraX 30% |
| `mid2` | Mid-2 分布适配：0.3B tokens(3%) / seq 16384 / cosine 6e-5 → 3e-5 / L3 70% + UltraX 30% |
| `debug` | 极小几何（seq 512）+ 真实 bin/idx，链路验证用 |
| `tiny` | mock 冒烟档 |
| `geoms/*` | 跨 stage 共享的规模阶梯档（`--profile geoms/qwen3_8b`）；只有一份，落在 `stage1_pretrain/config/` 下 |

## 快速开始

```bash
cd stage0_pretrain/stage2_midtrain
python data_prep.py --prepare                        # Mid-1 混合
python train.py --tokens 5e8 --load <PT-2 ckpt>
python data_prep.py --prepare --blend mid2.json      # Mid-2 长文档混合
python train.py --profile mid2 --tokens 3e8 --load <Mid-1 ckpt>
```

| 参数 | 说明 |
|---|---|
| `--profile <name>` | `default`（Mid-1）、`mid2`、`debug`、`tiny`、`geoms/*` |
| `--model-algo <name>` | 与各 stage 同一份注册表（默认 `qwen3_gdar_paper`） |
| `--tokens <n>` | token 预算 → `train_iters` |
| `--load <ckpt>` | 从 PT-2（Mid-1）或 Mid-1（Mid-2）接续 |

## 判据

| 检查 | 命令 | 判据 |
|---|---|---|
| 集成测试 | `python test_train.py` | tiny 5 步：rc=0、到最后一 iter、`[after training is done]`、无 Traceback |
| 冒烟 | `python train.py --smoke` | 同判据（mock 数据） |

## 跑完整论文实验（EXPERIMENT_MATRIX.md §3）

中训练只服务 **8B 四变体**与 **30B-A3B 门面**；序列加长之后 RULER 的 16K / 32K 档才有意义。

```bash
cd stage0_pretrain/stage2_midtrain

# 配比
python data_prep.py --prepare                     # Mid-1（含 UltraData-Code 分级）
python data_prep.py --prepare --blend mid2.json   # Mid-2 长文档

for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  # 8B
  python train.py --profile geoms/qwen3_8b --model-algo $algo --tokens 5e9 \
      --load <PT-2(8B, $algo) ckpt> --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo
  python train.py --profile geoms/qwen3_8b --model-algo $algo --tokens 3e9 \
      --set train.model.seq_length=16384 \
      --load $SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo/ckpt \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid2/$algo
  # 30B-A3B 门面：同法，--profile geoms/qwen3_30b_a3b
  # 32K 档：--set train.model.seq_length=32768
done

# 评测：与 PT 同一套（见 stage1_pretrain README 的评测清单），RULER 加 16K / 32K
```

`geoms/*` 档与预训练共用（同一份几何）；Mid-2 的 32K 只是加一个
`--set train.model.seq_length=32768`。

## 维护记录

- **2026-10-01 偶发 CUDA illegal memory access（未复现）**：一次 `test_train.py`（tiny 几何、
  L=8、block B=4、5 步）在第 2 步反向报 `CUDA error: an illegal memory access`；随后 5 次复跑
  （同一 run，以及 B=4/B=2/B=1 三形态各 3 步）全部干净收尾。按"未复现的偶发"记录，不当作已修复。
  再遇到时的定位方式：

```bash
cd stage0_pretrain/stage2_midtrain
CUDA_LAUNCH_BLOCKING=1 python train.py --profile debug \
    --set train.model.train_iters=2 --set train.model.eval_iters=0
```

## 延伸阅读

- [阶段 0 总览](../README.md) —— 段位设计、预算、LR 依据
- [预训练](../stage1_pretrain/README.md) —— Mid-1 接续的那一段
- [LIMITATIONS.md](../../LIMITATIONS.md) —— A7 记录了这次 CUDA 偶发与定位命令
