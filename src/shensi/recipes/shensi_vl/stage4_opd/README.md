# Stage 4：On-Policy Distillation（在线蒸馏）

论文 §2.5.4：RFT 模型 F 仍落后专家 → OPD 把 {E_TwG, E_TwP} 的能力合进一个统一模型：

```
L_OPD(θ) = Σᵢ wᵢ · D_KL(π_θ ‖ π_Eᵢ)        （反向 KL、全词表 logit 蒸馏）
```

关键在 **on-policy**：学生学的是自己在数据池上采样出的轨迹上的教师分布（不是专家的轨迹）——
`data_prep.py` 先用 F 自采样轨迹，`train.py` 的 opd 模式（`train_loop.run(mode="opd")`）
在响应 token 上加 Σ wᵢ·反向 KL（教师 fp32 前向、无梯度）。

## 跑法

```bash
python data_prep.py --limit 2000            # 学生自采样（正式档全量任务池）
python train.py --profile debug
python train.py --profile default           # 最终模型（= 论文的成品）
```

## 产物

`$SHENSI_FS/shensi/ckpt/shensi_vl/stage4_opd/final` —— 最终模型，进 [stage5_eval](../stage5_eval/)。

## 已知限制

- 教师前向的全词表 logits 显存开销 ≈ 学生前向 × 2（两教师串行）；micro batch 压 1、梯度累积上量。
- 论文的 wᵢ 没给数值，本配方默认两教师等权；要调就 `--set train.opd.weights=[0.7,0.3]`。
- 论文用 FP4（MXFP4）量化跑 RFT/OPD 控成本；HF 侧本配方保持 bf16（MXFP4 内核不进 HF 循环）。
