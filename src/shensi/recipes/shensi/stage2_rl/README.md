# stage2_rl：强化学习（四个子 stage，verl）

RL 不是一个阶段跑完的。按 GLM-5.x 与 Nemotron-3 的 RL 课程拆成四个子 stage，**同一个 verl 训练器 + 同一套
奖励函数**，差别在数据、采样预算、轨迹长度与环境后端。

> **现状（mcore main）**：`stage1_rlvr` 的 debug 档已跑到 `step:1`（rc=0）。导入期与配置期的坑
> （verl 的 v012 兼容层、FSDP 符号、`dsa_kernel_backend` 默认值）由 `shensi.runtime` 收口；
> 注意力这一侧的口径是「右 padding 的 mask 被丢掉」——mcore 的 CSA 不接受显式 mask
> （`packed_seq_params` 也必须为 None），而 verl 会把 response 右 padding 到 `max_response_length`，
> Bridge 的 `ShensiModel.forward` 因此丢掉纯右 padding 的 mask、拒绝左 padding / 文档边界。
> 注意尾部 pad 仍会通过压缩块参与计算（FL fork 当年收下 mask 却不用它，同口径）：要彻底消除
> 得等上游给 CSA 补 mask 支持，或 verl 那条不 padding 的路径落地。

## 1. 摘要

```text
stage1_rlvr         ① 多环境可验证奖励 RL（数学/科学/代码/推理，单轮为主）
stage2_agentic      ② 长时程 agentic RL（多轮 + 工具 + 环境；环境可真机，也可世界模型）
stage3_align        ③ 偏好 / 指令 / 安全对齐（可接 GenRM 判分模型）
stage4_world_model  ④ 世界模型：把「动作 → 观测」练成模型（CPT → SFT → RL），产物就是 ② 的 Sim RL 环境
```

| 依据 | 怎么落的 |
| --- | --- |
| GLM-5 系列（[报告](https://arxiv.org/abs/2602.15763)、[5.1](https://z.ai/blog/glm-5.1)、[5.2](https://z.ai/blog/glm-5.2)、[5.3](https://z.ai/blog/glm-5.3)） | GRPO、**IcePop 式双侧截断**（`clip_ratio_low/high = 0.2/0.28`）、不做 KL、LR 1e-6 恒定；5.2/5.3 的增益全在 agentic RL（`stage2_agentic`） |
| Nemotron-3（[Nano](https://arxiv.org/abs/2512.20848)、[Super](https://arxiv.org/abs/2604.12374)、[Ultra](https://arxiv.org/abs/2606.15007)、[3.5 Lightning](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/)） | 三段式：**21 环境 RLVR**（`stage1_rlvr`）→ **SWE-RL 容器内跑测试**（`stage2_agentic`）→ **RLHF + GenRM 判分**（`stage3_align`）；数据用它发布的 RL 集与 `*-Training-Blends` |
| DeepSeek-V4（[2606.19348](https://arxiv.org/abs/2606.19348)） | 后训练仍以可验证奖励为主；长上下文 / 1M 能力在 RL 前就位（见 `../stage0_pretrain/stage3_longctx`） |
| Qwen-AgentWorld（[2606.24597](https://arxiv.org/abs/2606.24597)） | 世界模型当环境（Sim RL / 可控扰动 / 虚构世界）→ `stage2_agentic` 的 `--profile world_model`；把世界模型练出来的那一段是 `stage4_world_model` |

## 2. 代码布局

四个子 stage 共用 `rl.py`（yaml → verl CLI 的映射、RL schema 归一、路径解析）与 `reward.py`
（verifier 奖励函数；世界模型那一段用自己的 `reward.py`），每个子 stage 的 `train.py` / `data_prep.py` 只做参数解析：

```text
stage2_rl/
├── README.md
├── rl.py         共用：build_command / launch / prepare / to_rl_row / files_of / resolve_paths
├── reward.py            verifier 奖励：string_match → 精确/数字 → pass_rate 软标签
├── stage1_rlvr/         config/{default,debug,gspo,dapo}.yaml + config/data_prep/* + train.py + data_prep.py
├── stage2_agentic/      环境与工具层；config/world_model.yaml（Sim RL 档）+ config/tools/world_model.yaml
├── stage3_align/        base: ../stage2_agentic/config，同上
└── stage4_world_model/  世界模型：train.py（--step cpt|sft|rl）+ data_prep.py + bench.py + reward.py
```

## 3. 两套环境（真机 / 世界模型）

`stage2_agentic` 有两条并行档：默认档用真环境（NeMo Gym 或 dsh 的容器/工具），`--profile world_model`
把环境换成**语言世界模型**（Qwen-AgentWorld 口径，或 `stage4_world_model` 自训的）——观测由模型预测，
因此环境可以无限扩、可注入扰动、可虚构。两档共用同一份数据与奖励，只是环境后端不同；
细节与依据见 `stage2_agentic/README.md`。

## 4. 奖励与判分

`reward.py` 按数据自带的 verifier 判分（`expected_markers` 全命中 → 数字/精确匹配 → `pass_rate` 软标签 → 兜底 0），
挂法是 verl 的 `reward.custom_reward_function.path/name`（`rl.build_command` 已接好）。
`stage3_align` 起可以把判分换成模型（GenRM，Nemotron-3 的做法）——verl 的 `reward.reward_model` 通道，
数据侧把 `reward_model.ground_truth` 换成偏好/评分规格即可。

## 5. 步数与早停

- 步数不设上限：保持 `trainer.total_training_steps: null`（verl 用 `total_epochs × len(dataloader)` 推总步数，
  所以把 `total_epochs` 给大；注意 `-1` 在 verl 里是**字面步数**不是"无限"），收尾交给看门狗；
- 评估：`test_freq` + `val_before_train: true`；早停用 `../early_stop.py` 盯 `critic/score/mean`（`--mode max`）。

## 6. 跑

```bash
cd stage1_rlvr                                     # 依次 stage2_agentic / stage3_align
python data_prep.py --discover                     # 看数据在不在（云端 post-training 目录）
python data_prep.py --prepare                      # → $SHENSI_FS/shensi/data/stage1_rlvr/{train,val}.parquet
python train.py --dry-run                          # 先看 verl 命令
python train.py                                    # 正式跑（上一段的 ckpt 用 --set model.path=... 指过去）
```

## 7. 验收判据

| 子 stage | 判据 |
| --- | --- |
| `stage1_rlvr` | `critic/score/mean` 高于基线并上行；`actor/entropy` 不塌到 0；同一 prompt 采样 8 次能看到不同解法 |
| `stage2_agentic`（真机） | 任务完成率上行；轨迹长度分布稳定（不塌成 1 轮、不顶上限）；工具调用格式错误率下降 |
| `stage2_agentic`（Sim RL） | 同上；另外看 Sim 档与真机档的完成率差距（差距 = 世界模型的建模误差） |
| `stage3_align` | 安全类不退化（红线用例 0 命中）；结构化输出合法率上升；`critic/score/mean` 不塌 |
| `stage4_world_model` | 见 `stage4_world_model/README.md`（SFT 的 token 级 loss、`bench.py` 的五维总分） |

## 8. RL 栈的升级路线（调研结论）

现在跑在 **verl**（GRPO + Megatron actor + vLLM rollout，已在极小规模上端到端验过）。往上走有两条现成的路：

| 方案 | 它解决什么 | 仓库 / 论文 |
| --- | --- | --- |
| **vime** = slime + vLLM rollout | 服务化 + async rollout、自定义数据生成，和我们的 vLLM(Shensi) 推理栈天然对接；GLM-5.2/5.3 的 agentic RL 就在这一层 | `vllm-project/vime`、`THUDM/slime` |
| **DeepSeek Harness（dsh）** | eval 与 agentic RL 的 **agent 层**（环境 + 工具 + 多轮 + 判分），要一个 DeepSeek 兼容端点（就是我们的 `vllm serve`）；`pip install deepseek-harness-sdk` | `deepseek-ai/deepseek-harness` |
| **NeMo Gym** | 环境 + verifier 判分，RL 与 eval 共用同一套环境（Nemotron-3 的 21 环境就是它） | `NVIDIA-NeMo/Gym`，见 `../stage3_eval` |

其余看过但不直接引入的：Kimi K2 / Mooncake / MoBA（Moonshot）、Qwen3 / Qwen-Agent（Qwen）、GLM 系与 CogView 系
（THUDM）、FlashMLA / DeepGEMM / DualPipe / 3FS（deepseek-ai）——它们要么是推理/基建侧的另一种实现
（我们已有 TE + FlagGems + vLLM 这条线），要么与本仓库的定位重叠（agent 框架已由 dsh/Gym 覆盖）。
与 `../stage0_pretrain/stage2_midtrain/config/mtp_draft.yaml`（主干冻结、只训 MTP draft）；
