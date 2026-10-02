# Stage 3：Unified RFT（统一模型 F）

论文 §2.5.3：RFT 模型 F 在各自领域明显好于冷启动 F_TwG/F_TwP，但仍落后专家 E_TwG/E_TwP。
做法：两个专家在更大更杂的数据池上 rollout 生成 RFT 数据 → 难度分层保留
**全部 Normal-Level + 随机 5% Easy-Level**（防灾难性遗忘）→ **从基座预训练模型重新 SFT**，
训练配置（超参、初始 checkpoint 口径）与冷启动 SFT 完全一致，唯一区别是数据配比。

## 跑法

```bash
# ① 专家 rollout → RFT 数据（70% 通用 + 30% RFT 自动拌好）
python data_prep.py --rollouts 8 \
  --general $SHENSI_FS/shensi/data/shensi_vl/stage1_sft/sft_box.jsonl
# 冒烟：已有现成 RFT jsonl 时 --ready <file> --general <file>

# ② 从基座重训
python train.py --profile debug
python train.py --profile default
```

## 产物

`$SHENSI_FS/shensi/ckpt/shensi_vl/stage3_rft/final` = 统一模型 F，下一 stage OPD 的学生初始化。
