# stage2_midtrain：中训练 + DSA 引入

预训练第二段：把 stage1 训好的稠密主干换成稀疏注意力。按 GLM-5 报告的 "DSA via continued pretraining"
两段式做——先让 Lightning Indexer 追上主干，再全参切稀疏 [2602.15763](https://arxiv.org/abs/2602.15763)。
indexer 本身来自 DeepSeek 的 DSA / Lightning Indexer 一脉，在 DeepSeek-V4 里就是 CSA 的那个 indexer
（[2606.19348](https://arxiv.org/abs/2606.19348)）。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 本段做两件事 | ① `dsa_warmup`：主干全冻、只训 indexer，KL 目标 = **稠密注意力分布**（论文 Eq.3），1000 步、LR 5e-3；② `default`（sparse adaptation）：全参训练，KL 目标切到**被选中的 top-k 集合**（Eq.4），20B tokens，序列拉到 32768 |
| 三个开关 | `csa_dense_mode: false`（indexer 必须在场）、`dsa_indexer_use_sparse_loss` false→true、`shensi_freeze: indexer`（冻主干） |
| 优化器 | 沿用 stage1 的混合口径（Muon + AdEMAMix）；sparse adaptation 的 LR 按报告取常数 1e-5 |

## 2. 超参对照

| 项 | GLM-5 | 本配方 |
| --- | --- | --- |
| dense warm-up 步数 | 1000 步 | 1000 步（`dsa_warmup.yaml`） |
| warm-up 每步序列 | 14 条 × 202,752 tokens | `global_batch_size: 14` × `seq_length: 32768`（按显存下调；warm-up 只需 indexer 追上主干） |
| warm-up LR | 最大 5e-3 | 5e-3 恒定 |
| warm-up 冻结范围 | 主干全冻，只训 indexer | `shensi_freeze: indexer` |
| sparse adaptation | 20B tokens、用中训练的数据与超参 | `--tokens 20e9`；数据 = stage2 的 `min_chars` 过滤版；LR 常数 1e-5 |
| KL 目标 | warm-up 稠密（Eq.3）→ sparse topk（Eq.4） | `dsa_indexer_use_sparse_loss` false → true |
| indexer top-k | 2048（确定性 torch.topk） | `shensi_index_topk: 512`（`ShensiConfig` 默认；推理侧同样固定 512） |
| 序列长度 | 中训练从 32K 起 | 32768 |
| MTP | 3 层共享 | 继承 stage1 |

## 3. 数据

`config/data_prep/data_blend_raw.json` 用 `base:` 继承 stage1 的权重（见 `../README.md` 第 2 节），
只把 `min_chars` 拉高（web/legal 2000、code 1000 量级），让中训练的样本更长。

```bash
python data_prep.py --discover && python data_prep.py --prepare
```

## 4. 运行

```bash
python train.py --profile dsa_warmup --dry-run    # ① 冻主干只训 indexer
python train.py --profile dsa_warmup
python train.py --dry-run                          # ② sparse adaptation
python train.py --tokens 20e9
```

`experiment.load` 默认指向 `shensi/ckpt/stage1_pretrain` / `shensi/ckpt/stage2_dsa_warmup`，按实际 ckpt 路径改。

## 5. 验收判据

1. **warm-up**：日志里 `indexer loss` 非零并下降；**主干权重逐位不变**（这既是本段的定义，也是唯一必须盯的点）；
2. **sparse adaptation**：`indexer loss` 继续下降；`lm loss` 不因切稀疏而跳变（跳变说明 top-k 选得差）；
3. `load_balancing_loss` / `erc loss` / `indexer loss` 三列都在日志里（三个 loss 全开）；
4. 与 dense 前向的 logits 相对偏差在 1e-3 量级内（规模稍大时自建对拍）。

warm-up 的冻结语义已由 `entrypoints/check_shensi_indexer_warmup.py` 离线验证（非 indexer 参数 3 步后逐位不变、
indexer 参数确实被更新、KL 上报非零）。

## 6. 训练侧的三件增量（本轮从「推理侧登记」改到训练侧）

| 件 | 落在哪 | 闸门 |
| --- | --- | --- |
| **DSA TopK 外部内核**（DeepSeek DeepSelect 这类） | `--shensi-index-topk-kernel 包.模块:函数`：训练前向的 top-k 从内置 torch 版换成外部内核（没装就留空），DeepSelect 填这里而不是 vLLM | 同上（S5/S6：桩内核被调用、配错会报错） |
| **MTP draft 单独训练**（DeepSpec 口径） | `config/mtp_draft.yaml`：主干全冻、只训 3 层共享 MTP（`--shensi-freeze mtp`），draft 的接受长度由 mcore 的 MTP loss 反映 | `entrypoints/check_shensi_mtp_draft.py`（5 项：冻结/反向语义、空集合硬失败、冻结档下主干逐位不变、draft 梯度非零） |

## 7. 局限

1. 20B tokens 的 sparse adaptation 对齐的是 GLM-5 的量级，不是 DeepSeek-V3.2 的 943.7B；
2. 报告里 warm-up 每步 202,752 tokens，本档按显存下调到 32768——indexer 追平主干的判据（主干逐位不变）不受影响，
   收敛速度会慢一些；
