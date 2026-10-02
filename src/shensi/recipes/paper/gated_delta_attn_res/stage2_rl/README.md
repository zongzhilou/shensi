# 阶段 2：RL（四个方向 teacher 并行分训）

从 SFT 检查点训四个方向的专用 teacher——数学 / 代码 / Agent / 写作——并行、互不共享权重。
teacher 之间不比高低：它们只对 OPD 负责（蒸馏回一个发布模型）。训练跑在 verl 上、actor 是
Megatron-Core；actor 在 verl 侧的模型注册由 `gdar_bridge.py` 提供（导入即注册）。

## 总览

| 组件 | 说明 |
|---|---|
| `stage2_{math,code,agent,writing}/` | 四个臂：入口脚本 + 各自的奖励与配置 |
| `../common/launch_rl.py`、`../common/prep_rl.py` | 四臂共用的启动与语料准备 |
| `gdar_bridge.py` | 把八个 model_type 注册进 Megatron-Bridge（层规格 + 由转换表生成的权重表） |
| `harness_tool.py`、`config/tools/harness.yaml` | Agent 臂的工具/多轮接线 |
| `test_train.py` / `test_gdar_bridge.py` | 预检 / 端到端桥闸门 |

## 算法档位（八档）

| 档 | 变化点 |
|---|---|
| `default` | GRPO、无 KL、clip 0.2/0.28 |
| `dapo` | 解耦截断 + 动态采样 + token 级 loss |
| `drgrpo` | 去掉 advantage 的 std 归一 |
| `token_baseline` | token 级最优基线估计量 |
| `critic` | GAE + value 模型 |
| `fsdp` | actor 走 HF/FSDP 路径（不经过 mcore 桥） |
| `gspo` | **序列级** importance ratio + 序列级截断 |
| `cispo` | 对 importance weight 做 stop-gradient 截断（长 CoT 更稳） |

优化器与训练段同一套：AdaMuon（矩阵腿）+ AdEMAMix（标量腿）。

## 前置条件

- 一个 **HF 目录**作为起点（verl 通过 `auto_map` 加载模型）：先把 SFT 检查点发布出来

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.common.train.export_hf \
    --ckpt <SFT 检查点> --out $SHENSI_FS/shensi/models/sft2-agent-hf
```

- RL prompts：按方向切，配比在各臂的 `config/data_prep/data_blend_*.json`。

## 快速开始

```bash
cd stage2_rl/stage2_math
python data_prep.py --prepare --config tiny      # prompts → parquet（小样本）
python train.py --dry-run                        # 打印 verl 命令
python train.py --profile gspo                   # 起训（八档之一）
python ../test_train.py                          # 四臂预检
python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.test_gdar_bridge   # 桥闸门
```

段级派发：`python stage2_rl/train.py --stage stage2_code --profile dapo --dry-run`。

## 完整主跑

RL 只为 30B-A3B 门面训方向 teacher：

```bash
for arm in stage2_math stage2_code stage2_agent stage2_writing; do
  ( cd stage2_rl/$arm && python data_prep.py --prepare --config default && python train.py )
done
```

> RL 的早停默认开：`total_epochs` 给到无限大，看门狗盯 `val/reward`，平台后收尾。

## 判据

| 检查 | 结果 |
|---|---|
| `python stage2_rl/test_train.py` | 配置 → CLI、奖励、verl 导入的预检 |
| `python -m ...stage2_rl.test_gdar_bridge` | 12/12：auto_map 分发、层规格一致、装载零缺键、HF 对拍 2.4e-07、导出逐位 |
| 各臂 × 八档 `--dry-run` | 全绿（每档都断言关键旋钮进了 verl 命令） |

## 各臂

- [数学](./stage2_math/README.md) · [代码](./stage2_code/README.md) · [Agent](./stage2_agent/README.md) · [写作](./stage2_writing/README.md)

## 下一步

- [OPD](../stage3_opd/README.md) —— 四个 teacher 的归宿
- [评测](../stage4_eval/README.md) —— 发布与打分
