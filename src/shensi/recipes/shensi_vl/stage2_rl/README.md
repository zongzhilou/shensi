# Stage 2：Specialized RL（GRPO，两族专家各一段）

论文 §2.5.2：对 F_TwG / F_TwP **分别**做 GRPO（跟随 DSV4F 的算法与超参），得到专家 E_TwG / E_TwP。
两个关键设计照搬：

1. **不监督思考中的原语**：冷启动数据里的框/点已经严格验证过，RL 阶段奖励只看输出文本与最终答案
   → RL 数据只需要（图、问题、spec/答案），来源大幅变宽；
2. **难度分层**（`data_prep.py --rollouts N`）：用 SFT 专家对每题 rollout N 次，
   Easy（全对）/ Normal（部分对）/ Hard（全错），**RL 只喂 Normal-Level**。

| 子 stage | 任务族 | 原语 | Accuracy RM |
| --- | --- | --- | --- |
| [stage1_grounding](./stage1_grounding/) | 计数、空间推理、通用 VQA | box | 计数：R=α·exp(−β·\|ŷ−y\|/(\|y\|+1))，α=0.7 β=3；VQA：精确/数字匹配 |
| [stage2_pointing](./stage2_pointing/) | 迷宫导航、路径追踪 | point | 迷宫：因果探索进度+探索完整性+撞墙惩罚+终路径有效性+答案（五项加权）；轨迹：双向匹配+端点+连续性+答案 |

共享的 [reward.py](../reward.py) 还包含 Format RM（原语文法 + 重复框惩罚）与
Quality RM（LLM GRM 三档 0/0.5/1；需外部判分端点 `SHENSI_VL_QUALITY_URL`，未配则中性跳过）。

## 跑法

```bash
# 任务池 + 难度分层（rollout N 次判难度；冒烟 --rollouts 0 直出全量）
python stage2_rl/data_prep.py --family grounding --rollouts 8
python stage2_rl/data_prep.py --family pointing  --rollouts 8

cd stage2_rl/stage1_grounding
python train.py --profile debug            # tiny LM + 真任务池
python train.py --profile default          # E_TwG
cd ../stage2_pointing && python train.py --profile default   # E_TwP
```

## 实现说明（与 shensi 基座的差异）

shensi 基座的 RL 走 verl（Megatron actor + vLLM rollout）；本配方模型是自定义 VL 结构
（DSV4F+ViT+projector），进不了 verl 的注册路径，所以 GRPO 在配方内自实现
（[grpo.py](../grpo.py)：组采样 → compute_score → 组内归一 advantage → 裁剪比率策略梯度）。
生产规模要回 verl 的话，把导出的 HF 模型注册成 vLLM 架构后复用基座 stage2_rl 的管线即可。
