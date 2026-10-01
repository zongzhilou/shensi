# Stage 1: SFT（指令微调）

从预训练末段 ckpt 起做多域指令微调：数据是 messages jsonl（含编码 / 推理 / 工具调用），
用 mcore 的 `--sft`（SFTDataset + SFTTokenizer 按 chat 模板生成 loss mask），模型与训练循环跟 PT 同一条。

## 总览

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--profile` / `--config`、`--smoke`、`--data-jsonl`、`--set` |
| `test_train.py` | 集成测试：tiny 几何 5 步（走 `--sft` + `ShensiSFTDataset` 的 loss mask 路径） |
| `data_prep.py` | post-training 语料 → `{"messages": [...]}` jsonl（chat 模板，`truncation` 可配） |
| `encoding_dsv4.py` | DSV4 的编码/解码与 chat 模板实现：与 HF 侧同口径，保持原样（改这里要同步 HF 侧） |
| `config/` | `default.yaml` + `tiny.yaml` + `debug.yaml` |
| `config/data_prep/` | `data_blend_raw.json` / `data_blend_tiny.json` / `debug_local.json` + 两个准备档 |

| 项 | 值 |
| --- | --- |
| 起点 | `shensi/ckpt/stage3_1m`（1M 上下文基座），由 `experiment.load` 指定 |
| 序列长度 | 32768（指令数据比预训练语料短） |
| 全局批 / 步数 | `global_batch_size: 32`、`train_iters: 5000`（`split: 98,1,1` 切验证集，靠早停收尾） |
| 优化器 | AdaMuon（矩阵腿）+ AdEMAMix（标量腿）（微调只换 LR 曲线：1e-5 → 1e-6 cosine、warmup 1%、wd 0） |
| loss | 只算 assistant span（`SFTTokenizer` 生成 mask）；aux/ERC/indexer 三个 loss 按 SFT 口径自动关掉 |
| tokenizer | `SFTTokenizer`：正式档 `default`（tokenizer 自带 chat_template）、极小档 `identity` |

## 快速开始

```bash
python test_train.py                    # 集成测试（tiny 几何 5 步）
python data_prep.py --discover          # post-training 语料面貌
python data_prep.py --prepare           # → $SHENSI_FS/shensi/data/stage1_sft/sft_{train,val}.jsonl
python train.py --profile tiny          # 冒烟（tiny + mock）
python train.py --profile debug         # 真实数据的极小档
python train.py                         # 正式跑（default 档）
```

`data_prep.py` 的输出是**一行一条**的 `{"messages": [{"role": ..., "content": ...}, ...]}`；
mcore 的 `--sft` 直接读 jsonl（不用 bin/idx），loss mask 由 `SFTTokenizer` 按模板切。
`train.py` 会把 `sft_train.jsonl` 写进 `train.data.data_path`（也可 `--data-jsonl` 指别的文件）。

## 数据准备

来源是 post-training 集（`$SHENSI_FS/datasets/llm/post-training/`，配比见 `config/data_prep/data_blend_raw.json`，
`--discover` 看实际列名）。多轮样本按 chat 模板拼成单条 messages；超长样本按 `truncation` 策略处理
（默认 `error`，直接报错提醒调 `max_length`）。

### 不打包

上游的 `SFTDataset` 会把多条对话打进一条样本并给出 `cu_seqlens`（THD），而 CSA/HCA 层明确断言
`packed_seq_params is None`——打包在本家族上走不通。本栈换成 `common/train/sft_dataset.py` 的
`ShensiSFTDataset`：**一条对话一条样本 + 右侧 padding**，tokenize 与 loss mask（prompt 段与 padding 段
都不算 loss）沿用上游同一套口径，只是不产出 `cu_seqlens`；要回上游的 THD 打包口径加 `--shensi-sft-packed`。
右 padding 对因果注意力无害：有效 token 看不到后面的 pad，pad 段本身也被 loss mask 排掉。

## 训练

| 项 | 值 | 说明 |
| --- | --- | --- |
| `--profile` | `default` / `tiny` / `debug` | 入口默认 `debug`（极小档最常用） |
| 覆写 | `--set train.model.train_iters=...` 等 | 与 PT 同一套摊平语义 |
| 早停 | 默认开（盯 `lm loss value`，patience=3、grace=600s） | `split: 98,1,1` 提供验证信号 |

## 验证

1. 极小档能续：`experiment.load` 指向 PT 的 debug ckpt 时要显式给 `no_load_optim/no_load_rng`；
2. 日志里 `lm loss` 下行、`grad norm` 不炸；loss mask 只覆盖 assistant 段与结尾 eos；
3. 指令跟随抽测（固定 prompt 生成，人工看格式与工具调用标签）；
4. **集成测试**：`python test_train.py` 5 步 PASS。

本机实测：离线档 `--prepare --blend config/data_prep/debug_local.json --limit 40` 出 38 行 jsonl；
`--profile debug` 2/2 步（载入 `pt_tiny_debug`，finetune 口径迭代号重开）存 `stage1_sft_debug`；
早停实测 `train_iters=200`、patience=1、grace=5s 在第 42 步收尾（`early_stop.json` 里 `why=patience`）；
导出给 RL / 评测：

```bash
python -m shensi.recipes.shensi.common.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/stage1_sft_debug --out $SHENSI_FS/shensi/models/sft-hf --tiny
# 之后 stage2_rl 用 --set model.path=<out>，stage3_eval 用 --model-path <out>
```

## 局限

1. 微调规模上的优化器口径沿用预训练，LR 另有一组 20 步扫描（极小档、constant 曲线，读数见「验证」）；
2. `encoding_dsv4.py` 的 chat 模板与 HF 侧是同一口径：模板改动要两边同步，否则 loss mask 会不一致；
3. 数据配比走公开 post-training 集，没有自建指令数据——覆盖的是**域**：math / code / agent / safety / 多语。

## 下一步

对齐 / RL 见 [Stage 2: RL](../stage2_rl/README.md)。
