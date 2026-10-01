# Stage 2.1: 多环境可验证奖励 RL（RLVR）

`stage2_rl` 的第一段：单轮为主、答案可校验。奖励由 [`../reward.py`](../reward.py) 按 verifier 规则判
（`string_match` / 精确 / 数字 / `pass_rate` 软标签，兜底 0）。

## 总览

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口（`../common/rl.py` 的薄封装）：`--profile` / `--data-dir` / `--dry-run` / `--set` |
| `test_train.py` | 集成预检：配置→命令、数据 parquet、ray、GPU、import、环境变量 |
| `data_prep.py` | 语料 → verl 的 RL schema（`train.parquet` / `val.parquet`） |
| `config/` | `default.yaml` + `tiny.yaml` + `debug.yaml` + 算法档 `gspo.yaml` / `dapo.yaml` |
| `config/data_prep/` | `data_blend_raw.json`（11 个可校验集）+ `data_blend_tiny.json` + 两个准备档 |

| 项 | 值 |
| --- | --- |
| 目标 | 在可校验任务上把「答对」的比例顶起来（数学 / 科学 / 代码 / 推理） |
| 算法 | GRPO（verl 原生 `algorithm.adv_estimator`），不做 KL（`kl_coef 0`） |
| 采样 | `rollout.n = 8`，`max_response_length = 32768` |
| LR / 截断 | 1e-6 恒定；`clip_ratio_low/high = 0.2/0.28`（双侧截断） |
| 优化器 | AdaMuon（矩阵腿）+ AdEMAMix（标量腿）；单卡档（`tiny` / `debug`）关掉 LayerWise 走普通 bf16 包装 |

**算法档**：默认 GRPO；`gspo.yaml`（序列级重要性比）与 `dapo.yaml`（clip-higher + 动态采样，group 16）
都是 verl 原生 `algorithm.adv_estimator`，`--profile gspo` / `--profile dapo` 直接切。

## 快速开始

```bash
python test_train.py --data-dir <parquet 目录>    # 集成预检（不跑完整 GRPO）
python data_prep.py --discover                     # 看数据在不在
python data_prep.py --prepare                      # → $SHENSI_FS/shensi/data/stage1_rlvr/{train,val}.parquet
python train.py --dry-run                          # 看 verl 命令
python train.py --profile tiny --data-dir <目录>    # 冒烟：本地 tiny 模型（tiny-rl）+ 最小采样预算
python train.py --profile debug --data-dir <目录>   # 极小档（1 epoch、少采样）
python train.py --set model.path=<sft ckpt>        # 正式跑：上一段 ckpt 用 --set 指过去
```

## 数据准备

11 个可校验集（数学 / 科学 / 代码 / 推理），每条数据行带 `verifier`（判分规则）与
`reward_model.ground_truth`；`--discover` 打印每个数据集在不在、条数与字段名。
冒烟可以只用 40 条小语料（`problem` + `answer`）：

```bash
python data_prep.py --prepare --config config/data_prep/tiny.yaml
```

## 训练

| 项 | 值 | 说明 |
| --- | --- | --- |
| 并行 | actor TP=PP=1 | 单机 1~8 卡按机器调 `rollout.tensor_model_parallel_size` 等 |
| 每步权重同步 | mcore actor → vLLM rollout 引擎 | 日志里的 `update_weights done` |
| 优化器 | AdaMuon + AdEMAMix（`actor.optim.*`） | 全档 `use_layer_wise_distributed_optimizer: false`——LayerWise 被 verl 的守卫挡在这套 mcore 之外（见 [`../README.md`](../README.md) 的「与上游的对接口径」） |
| 早停 | 验证准确率（越大越好） | `trainer.early_stop_metric` / `early_stop_mode` 可覆写 |

## 验证

1. **集成预检 PASS**；
2. `critic/score/mean` 高于基线并上行；
3. `actor/entropy` 不塌到 0；同一 prompt 采样 8 次能看到不同解法；
4. 早停：验证准确率超耐心即收尾。

本机实测：从导出的 SFT ckpt 起跑 19/19 步（`Training Progress: 100%`）、权重同步 20 次、0 报错；
换 AdaMuon + AdEMAMix 后重跑同样 100%（dry-run 里能看到 `optim.optimizer=adaptive_muon` 与
`optim.muon_scalar_optimizer=ademamix` 进到 verl 命令行）。

## 局限

1. 奖励只覆盖规则可判的部分；开放式任务的判分走 [`../stage3_align`](../stage3_align/README.md) 的 GenRM 通道；
2. 11 个集的相对权重按本阶段归一，实际 token 占比看 `--discover` 的实测条数；
3. 注意力口径：verl 右 padding，CSA 不接受显式 mask → 丢掉纯右 padding 的 mask，尾部 pad 仍会经压缩块参与计算；
4. 优化器的 LR / 系数沿用预训练口径，未单独扫描。

## 下一步

[`../stage2_agentic`](../stage2_agentic/README.md)（多轮 + 工具 + 环境）。
