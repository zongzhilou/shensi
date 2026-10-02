# 阶段 1：SFT（deep-thinking → hybrid-thinking → agent）

在中训练检查点上做指令微调：SFT-1 建立深度思考与通用对话能力，SFT-2 扩大覆盖面，SFT-3 补 agent
多轮能力。旗舰档合计 400B tokens（200B + 200B + 20B）。对照臂同数据、同 token 数、同 LR。

## 总览

| 组件 | 说明 |
|---|---|
| `data_prep.py` | 对话数据（parquet/jsonl）→ messages jsonl（98/2 切分） |
| `train.py` | 训练入口（不打包口径；`--data-jsonl` 指向数据集） |
| `test_train.py` | 集成测试（没有语料时自己生成合成 jsonl） |
| `config/` | `default`（SFT-1）、`hybrid`、`agent`、`geoms/*`、`tiny`、`debug` |

## 数据口径

- 走**不打包**口径：一条对话一条样本 + 右 padding（本配方的注意力是 local 实现，不吃 THD 打包）；
- loss mask 由 `SFTTokenizer` 按 tokenizer 自带的 chat 模板生成；
- `reasoning_content`（思考段）原样保留。

| 项 | 说明 |
|---|---|
| 输入 | `$SHENSI_FS/datasets/llm/post-training/<名字>/`（parquet / jsonl） |
| 配比 | `config/data_prep/data_blend_{raw,tiny,hybrid,agent}.json` |
| 输出 | `$SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train<suffix>.jsonl` + `sft_val<suffix>.jsonl` |

## 快速开始

```bash
python data_prep.py --prepare --config tiny           # 小样本 jsonl（调试）
python train.py --smoke                               # 合成 16 条对话 + tiny 几何，5 步
python data_prep.py --prepare --config default        # SFT-1 语料
python train.py --config default --tokens 2e9 --load <Mid-2 检查点>
python data_prep.py --prepare --config hybrid         # SFT-2 语料
python train.py --config hybrid --tokens 2e9 --load <SFT-1 检查点>
python data_prep.py --prepare --config agent          # SFT-3 语料
python train.py --config agent --tokens 2e10 --load <SFT-2 检查点>
```

## 训练

| 参数 | 说明 |
|---|---|
| `--config <名字>` | `default` / `hybrid` / `agent` / `geoms/qwen3_30b_a3b` … |
| `--data-jsonl <文件>` | 显式指定 messages jsonl |
| `--tokens N` / `--load <检查点>` | token 预算 / 接续检查点 |
| `--model-algo <名字>` | 连接 / 基线（默认主行） |

## 完整主跑（旗舰对）

SFT 只服务旗舰对（GDAR 主行 + plain 残差孪生 `base`，30B-A3B 几何）：

```bash
for algo in qwen3_gdar_main base; do
  D=$SHENSI_FS/shensi/runs/gdar_30b_sft/$algo
  python train.py --config geoms/qwen3_30b_a3b --model-algo $algo --tokens 2e11 \
      --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train.jsonl \
      --load <Mid-2(30B, $algo) 检查点> --set experiment.exp_dir=$D-sft1
  python train.py --config hybrid --model-algo $algo --tokens 2e11 --load $D-sft1/ckpt --set experiment.exp_dir=$D-sft2
  python train.py --config agent  --model-algo $algo --tokens 2e10 \
      --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train_agent.jsonl \
      --load $D-sft2/ckpt --set experiment.exp_dir=$D-sft3
done
```

## 判据

| 检查 | 判据 |
|---|---|
| `python test_train.py` | 合成 jsonl + tiny 几何：5 步 rc=0、到最后一 iter、`[after training is done]`、无 Traceback |
| `python train.py --smoke` | 同判据（合成数据） |

## 下一步

- [配方 README](../README.md) —— 管线总览与 `--model-algo`
- [中训练](../stage0_pretrain/stage2_midtrain/README.md) —— SFT 接续的那一段
- [RL](../stage2_rl/README.md) —— 消费 SFT 检查点的下一段
