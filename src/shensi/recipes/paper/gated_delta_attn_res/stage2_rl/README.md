# 阶段 2：RL（四方向 teacher 并行分训）

从 SFT ckpt 训四个方向 teacher —— **数学 / 代码 / Agent / 写作** —— 并行、互不共享权重。
每个 teacher 都是**专才**：它们之间不比高低，存在的意义是被 [OPD](../stage3_opd/) 蒸馏回
同一个发布模型。

训练跑在 **verl**（GRPO 系估计量）上，actor 是 **Megatron-Core**；让 verl 能建出并服务我们的
检查点的那套注册在 [`gdar_bridge.py`](./gdar_bridge.py)，端到端闸门是
[`test_gdar_bridge.py`](./test_gdar_bridge.py)。

## 总览

| 组件 | 说明 |
|---|---|
| `stage2_{math,code,agent,writing}/` | 四个臂各一套：配置、数据准备、奖励、启动器 |
| `gdar_bridge.py` | 导入即把七个变体注册进 Megatron-Bridge（`VERL_USE_EXTERNAL_MODULES`） |
| `train/export_hf.py` | 把 SFT ckpt 发布成 `model.path` 要指向的 HF 目录 |
| `harness_tool.py`、`config/tools/harness.yaml` | agent 臂的工具 / harness 接线 |
| `test_train.py` | 预检：配置 → verl CLI、奖励模块、verl 导入 |
| `test_gdar_bridge.py` | 端到端闸门：分发、层规格、权重装载、HF 对拍、rollout 同步导出 |

## 模型算法档位

四个臂共用六个档（`config/<名字>.yaml`，`base: default.yaml` 深合并）：

| 档 | 变化点 |
|---|---|
| `default` | GRPO、无 KL、clip 0.2/0.28（基线） |
| `dapo` | 解耦截断 + 动态采样 + token 级 loss |
| `drgrpo` | 去掉 advantage 的 std 归一 |
| `token_baseline` | token 级最优基线估计量 |
| `critic` | GAE + value 模型（JustRL-II 式那一档） |
| `fsdp` | HF/FSDP actor 路径（不经过 mcore 桥） |

## 前置条件

- 一个 **HF 目录**作为起点：先把 SFT ckpt 发布出来（verl 通过 `auto_map` 加载模型，
  只给 mcore ckpt 不够）：

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_sft \
    --out  $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_sft/sft2_agent
```

- RL prompts：UltraData-RL-2609，`data_prep.py --blend <方向>.json` 按方向切。
- 桥由 `VERL_USE_EXTERNAL_MODULES` 注入每个 verl 进程（启动器负责设置）；没有它时 verl 会
  **明确报错**拒绝这个架构 —— 不会静默建错模型。

## 快速开始

```bash
cd stage2_rl/stage2_math
python data_prep.py --prepare             # prompts → verl train/val parquet
python train.py --dry-run                 # 打印 verl 命令
python train.py                           # 起训（GRPO，LR 1e-6，clip 0.2/0.28）
python train.py --profile dapo            # 六个算法档之一
```

| 参数 | 说明 |
|---|---|
| `--profile <name>` | `default`、`dapo`、`drgrpo`、`token_baseline`、`critic`、`fsdp` |
| `--set k=v` | 任意配置覆盖（`model.path`、`trainer.total_epochs` 等） |
| `--dry-run` | 打印组装好的 verl CLI |

> **RL 的早停同样默认开**：`total_epochs` 给到无限大，看门狗盯 `val/reward`（越大越好），
> 平台后收尾。

## 判据

| 检查 | 命令 | 结果 |
|---|---|---|
| 预检 | `python stage2_rl/test_train.py` | 配置 → CLI 映射、奖励、verl 导入 |
| 桥闸门 | `python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.test_gdar_bridge` | 12/12：auto_map 分发、层规格 = GDAR 连接层、装载零缺键、HF 对拍 `2.4e-07`、导出（rollout 同步）逐位相等 |
| 档位 | 各臂 `python train.py --profile <p> --dry-run` | 4 臂 × 6 档 = 24 个 dry-run，每个都断言 `model.path` 与两份 provider 覆盖真的进了 CLI |

## 跑完整论文实验（EXPERIMENT_MATRIX.md §5：门面 teacher）

RL 只为 **30B-A3B 门面**训方向 teacher，明确不承担架构对比结论。rollout 驱动按起点模型的
model type 自动选择。

```bash
# 两个臂各训自己的四个 teacher（各自从自己的 SFT-3 ckpt 起）
for arm in qwen3_gdar_main base; do
  for dir in stage2_math stage2_code stage2_agent stage2_writing; do
    cd stage2_rl/$dir
    python data_prep.py --prepare --limit 200000      # UltraData-RL-2609 按方向切
    python train.py                                   # model.path → 该臂的 SFT-3 HF 目录
  done
done
```

每条 teacher 用 SFT 那套加方向指标评测（RUN_EXPERIMENTS.md §5）；teacher ckpt 是 `stage3_opd`
的输入（每个方向一个）。

## 延伸阅读

- [OPD](../stage3_opd/README.md) —— 四个 teacher 的归宿
- [发布 + 评测](../stage4_eval/README.md) —— HF 目录与 T0 检索任务
- [LIMITATIONS.md](../LIMITATIONS.md) —— A9（算法档）、A13（桥）、A15（档位合并）
