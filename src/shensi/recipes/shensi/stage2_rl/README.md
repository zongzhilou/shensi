# Stage 2: RL（四个子 stage，verl）

RL 不是一个阶段跑完的。按 GLM-5.x 与 Nemotron-3 的 RL 课程拆成四个子 stage，**同一个 verl 训练器 + 同一套
奖励函数**，差别在数据、采样预算、轨迹长度与环境后端。

## Overview

| 子 stage | 内容 | 数据 | 环境 |
|----------|------|------|------|
| `stage1_rlvr` | ① 多环境可验证奖励 RL（数学/科学/代码/推理，单轮为主） | Nemotron RL Lightning Training Blend | 无（纯 verifier） |
| `stage2_agentic` | ② 长时程 agentic RL（多轮 + 工具 + 环境） | SWE / 终端 / 检索轨迹 | 容器内真机或世界模型（`--profile world_model`） |
| `stage3_align` | ③ 偏好 / 指令 / 安全对齐 | 偏好对 + GenRM 判分 | 判分模型（`reward.judge_model`） |
| `stage4_world_model` | ④ 世界模型：动作 → 观测（CPT → SFT → RL） | 交互轨迹 | 产物即 ② 的 Sim RL 环境 |

公共件：`../rl.py`（yaml → verl CLI 映射、RL schema 归一、启动与环境处理）、`reward.py`（verifier 奖励）、
`agentworld/`（多轮环境的 prompt 资产与判分解析，外部来源保持原样）；每个子 stage 有
`train.py` / `data_prep.py` / `test_train.py` / `config/`。

| 依据 | 怎么落的 |
| --- | --- |
| GLM-5 系列（[报告](https://arxiv.org/abs/2602.15763)、[5.1](https://z.ai/blog/glm-5.1)、[5.2](https://z.ai/blog/glm-5.2)、[5.3](https://z.ai/blog/glm-5.3)） | GRPO、**IcePop 式双侧截断**（`clip_ratio_low/high = 0.2/0.28`）、不做 KL、LR 1e-6 恒定 |
| Nemotron-3（[Nano](https://arxiv.org/abs/2512.20848)、[Super](https://arxiv.org/abs/2604.12374)、[Ultra](https://arxiv.org/abs/2606.15007)、[3.5 Lightning](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/)） | 三段式：**21 环境 RLVR** → **SWE-RL 容器内跑测试** → **RLHF + GenRM 判分** |
| DeepSeek-V4（[2606.19348](https://arxiv.org/abs/2606.19348)） | 后训练仍以可验证奖励为主；长上下文 / 1M 能力在 RL 前就位 |
| Qwen-AgentWorld（[2606.24597](https://arxiv.org/abs/2606.24597)） | 世界模型当环境（Sim RL / 可控扰动）→ `stage2_agentic --profile world_model` |

## Quick Start

```bash
cd stage1_rlvr
python test_train.py --data-dir <parquet 目录>       # 集成预检（配置/数据/ray/GPU/import/环境变量）
python data_prep.py --prepare                        # → train.parquet / val.parquet
python train.py --profile debug --data-dir <目录>     # 极小档：1 epoch、少采样
python train.py --profile default --data-dir <目录>   # 正式跑
```

`train.py --dry-run` 只打印将要执行的 `python -m verl.trainer.main_ppo …` 命令（含所有覆盖项）。

## 现状（mcore main 上）

`stage1_rlvr` 的 debug 档已跑到训练步（rc=0、`step:0` → `step:1`），导入期与配置期的坑由 `shensi.runtime` 收口：

- verl 的 v012 兼容层在版本守卫之前就 import 两个 FL fork 才有的模块 → runtime 补两个最小实现；
- `mcore_fsdp_adapter.FullyShardedDataParallel` 在 main 上是工厂函数，而这份 verl 把它当类做类型判断；
- `dsa_kernel_backend` 在 dsv4_hybrid 下默认 `cudnn`（要 flash_mla）→ runtime 在本机没有融合内核时回退 `none`；
- 注意力口径：mcore 的 CSA 不接受显式 mask，而 verl 会把 response 右 padding 到 `max_response_length` →
  Bridge 的 `ShensiModel.forward` 丢掉纯右 padding 的 mask（左 padding / 文档边界直接报错），
  尾部 pad 仍会经压缩块参与计算（与 FL fork 同口径）。

## 数据

`data_prep.py --prepare` 把语料归一成 verl 的 RL schema（`prompt` + `reward_model.ground_truth` +
`extra_info`），产出 `train.parquet` / `val.parquet`；来源与配比见各子 stage 的
`config/data_prep/data_blend_raw.json`（`--discover` 看实际目录与列名）。

```bash
python data_prep.py --discover                    # 语料面貌（哪些数据集在位、字段名）
python data_prep.py --prepare --blend config/data_prep/debug_sample.json --max-chars 2000
```

## 训练

| 旋钮 | 说明 |
|------|------|
| `--profile` | `debug`（1 epoch、小批）/ `default`（正式）；`stage4_world_model` 另有 `--step cpt/sft/rl` |
| `--data-dir` | parquet 目录，默认 `$SHENSI_FS/shensi/data/<stage>` |
| `--set k=v` | 点号覆写，例如 `--set rollout.n=1 --set actor.optim.lr=5e-6` |
| `--dry-run` | 只打印命令 |

关键默认（`stage1_rlvr/config/default.yaml`）：

| 项 | 值 | 出处 |
| --- | --- | --- |
| 算法 | GRPO + IcePop 式双侧截断（`clip_ratio_low 0.2` / `clip_ratio_high 0.28`）、`kl_coef 0` | GLM-5 系列 |
| 采样 | `rollout.n: 8`、温度 1.0；`log_prob_micro_batch_size_per_gpu: 1` | Nemotron RLVR |
| 并行 | actor：TP=PP=1（单机 1 卡 ~ 8 卡按机器调）；`use_remove_padding: false`（CSA 不支持打包） | 见上式注意力口径 |
| 显存账 | `ray_kwargs.ray_init.include_dashboard: false`、`num_cpus: 8`、`transfer_queue…num_data_storage_units: 2` | 41G 级单机的实测账（见配方 README 的「环境注意事项」） |

## 验收判据

1. **集成预检 PASS**：`python test_train.py --data-dir <目录>`（配置/数据/ray/GPU/import 全 ✓）；
2. 日志里出现 `Training Progress` 与 `step:N`，且 `critic/score/mean` 有非零方差（奖励真的在分化）；
3. `actor/recompute` 的 logprob 偏差在阈值内（rollout 侧与训练侧的口径一致）；
4. 早停看门狗用 `critic/score/mean`（`--mode max`）。

## 下一步

- 策略这条线：`stage1_rlvr` → `stage2_agentic` → `stage3_align`；
- 世界模型这条线（与策略并行）：`stage4_world_model`，产物被 `stage2_agentic --profile world_model` 当环境用；
- 评测见 [Stage 3: 评测](../stage3_eval/README.md)。

> 早停：**默认开**（PT/SFT 盯 `lm loss value`、RL 盯验证准确率，patience=3、grace=600s；`--no-early-stop` 关掉、`--early-stop N` 改耐心）。步数/轮次可以给很大，收尾交给它——见 [配方总览的「早停」一节](../README.md#早停默认开)。

## 局限

1. 四个子段都在本机跑到过训练步：`stage1_rlvr` / `stage2_agentic` / `stage3_align` 各 19/19 步
   （每步把 actor 权重同步给 vLLM），`stage4_world_model` 的三段（CPT → SFT → RL）也全通过
   （RL 用 CPU 桩判分）；真分数与真环境仍要目标机（见各子段 README）；
2. MTP 与 mHC 已打通（配方 README 第 9 节第 2 条）：带 `mtp.*` 的 HF 产物能转换、能进 RL；
   本机的 RL 极小档用的是 MTP=0 的 ckpt，开 MTP 只是把 `--mtp 1` 加上去；
3. agentic / align 的环境与判分模型属于外部依赖：agent harness 统一在 **DeepSeek Harness（dsh）**下
   （`shensi.recipes.shensi.harness`，stage3_eval 与 stage2_agentic 共用同一份接线；vLLM 仍是服务层），
   GenRM 判分模型要自备端点——预检会报缺什么、怎么装。
## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

**RL / 评测都从前面训出来的 ckpt 起**（不再用随机初始化的 tiny 模型）：

```bash
# ① SFT 的 mcore 产物 → HF 目录
python -m shensi.recipes.shensi.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/stage1_sft_debug --out $SHENSI_FS/shensi/models/sft-hf --tiny
# ② 四个子段都拿它当 model.path
cd stage1_rlvr && python train.py --profile debug --data-dir $SHENSI_FS/shensi/data/stage1_rlvr \
    --set model.path=$SHENSI_FS/shensi/models/sft-hf
```

| 子段 | 从 `sft-hf` 起跑的结果 |
| --- | --- |
| stage1_rlvr | 19/19 步（1 epoch），权重同步 20 次 |
| stage2_agentic | 19/19 步（1 epoch），权重同步 20 次 |
| stage3_align | 19/19 步（1 epoch），权重同步 20 次 |
| stage4_world_model | CPT 与 SFT 两段通过、RL 3 步（用 CPU 桩判分端点） |

三个 RLVR/agentic/align 段每步都会把 actor 权重同步给 vLLM（日志里的 `update_weights done`），
说明 HF → mcore（载入）与 mcore → HF（每步同步）两个方向都在真实权重上跑通了。
