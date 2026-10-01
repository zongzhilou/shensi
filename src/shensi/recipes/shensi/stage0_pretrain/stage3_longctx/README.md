# Stage 0.3: 长上下文扩展

预训练第三段：把上下文从 32K 拉到模型上限，分两档跑（对应 GLM-5 mid-training 的后两段）。
稀疏注意力保持打开（继承 stage2），三个 loss 继续生效；优化器沿用 stage1/stage2 的口径（Muon + Lion）。

## Overview

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--profile default`（128K）/ `--profile 1m`（1M），`--tokens` 给 token 预算 |
| `test_train.py` | 集成测试：tiny 几何 5 步（走同一套 rope/YaRN 参数路径） |
| `data_prep.py` | 语料 → bin/idx + `blend.json`（`base:` 继承 stage2，`min_chars` 再拉高、长文档权重上调） |
| `config/` | `default.yaml`（128K）+ `1m.yaml`（1M）+ `debug.yaml` |

| 档 | 序列长度 | token 预算 | GLM-5 对应 | 备注 |
| --- | --- | --- | --- | --- |
| `default` | 131072 | 500B（`--tokens 500e9`） | mid-training 的 128K / 500B | CP 默认 2，按显存调 |
| `1m` | 1048576 | 50B（`--tokens 50e9`） | mid-training 的 200K / 50B | 本模型 compress 层位置编码支持到 1M（YaRN 16 / 原位置 65536），最后一段直接上 1M；要严格照报告就改成 200000 |

| 项 | 值 | 出处 |
| --- | --- | --- |
| 学习率 | 1e-5 恒定续训 | 报告没给长上下文段的具体 LR，按「沿用中训练超参」的口径取 midtrain 收尾量级 |
| loss | aux 0.001 + ERC 1.0/0.5 + indexer KL 0.01 | 配方总览的四条口径 |
| 未启用（登记） | V4.1-Flash 的 CSA2 跨层 KV 复用、FP4 KV 缓存（890 bytes/token）、Causal Encoder-Decoder | 要落到本配方需改推理侧与 ckpt 口径；DeepSeek 侧 1M 能力的来源见 [2606.19348](https://arxiv.org/abs/2606.19348)（1M 下 Pro 的单 token 推理 FLOPs 只有 V3.2 的 27%、KV 缓存 10%）与 [2609.19969](https://arxiv.org/abs/2609.19969) |

## Quick Start

```bash
python test_train.py                                     # 集成测试（tiny 几何，5 步）
python data_prep.py --discover && python data_prep.py --prepare
python train.py --profile default --tokens 500e9 --dry-run
python train.py --profile default --tokens 500e9         # 128K / 500B
python train.py --profile 1m      --tokens 50e9          # 1M / 50B
```

`experiment.load` 默认接 `shensi/ckpt/stage2_midtrain`（128K 档）与 `shensi/ckpt/stage3_128k`（1M 档）。

## 数据

`config/data_prep/data_blend_raw.json` 在 stage2 的基础上把 `min_chars` 拉高（web/legal 20000、code 4000），
并把长文档来源（legal、specialized-generative、PDFs）的权重上调——GLM-5 在后段也是这样 up-sample 长文档的。

## 验收判据

1. 长度切换后 `lm loss` 不出现台阶式恶化（出现说明 RoPE/YaRN 或数据的长度分布没对齐）；
2. `indexer loss` 在长序列上不炸（稀疏选择在 128K/1M 上仍有效）；
3. 1M 档不 OOM：CP 与 micro-batch 按显存调，先把 `micro_batch_size=1` 跑通再往上加；
4. 长文检索类能力（把一篇长文档的某个事实放进 prompt，看能否复述）在 128K 与 1M 上抽测通过；
5. **集成测试**：`python test_train.py` 5 步 PASS。

## 下一步

基座到这里完成 → [Stage 1: SFT](../../stage1_sft/README.md)。

## 局限

**已知缺口（要补语料）**：GLM-5 的长上下文数据还有 ① 自建的自然长文档（书/论文）、② 合成数据
（NextLong / EntropyLong 思路）、③ 200K 段的 MRCR 类数据。Nemotron 预训练集里没有对应物，
本配方先用长文档筛选顶着；要完全对齐需要另外准备这三类语料再进 blend。
