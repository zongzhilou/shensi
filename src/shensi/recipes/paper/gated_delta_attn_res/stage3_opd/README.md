# 阶段 3：OPD（四个 teacher → 一个发布模型）

把四个方向 teacher 蒸馏回**同一个发布模型**。两条可跑的路：

1. **静态 KD（mcore 原生三步）**：学生 rollout → teacher 用 logits saver 打分落盘 → 学生读缓存
   做 KD 训练（`--logits-load-dir`，默认 forward KL，可切 **reverse KL**）；
2. **RL 式 OPD（verl 原生 distillation）**：学生自己 rollout、teacher **在线**打分，逐 token 的
   reverse KL 当信号，多 teacher 按 `data_source` 路由（`loss_mode=k1` + policy gradient）。

## 总览

| 组件 | 说明 |
|---|---|
| `train.py`（静态 KD） | 读 teacher 缓存训练；`--teacher-cache` / `--reverse-kl` |
| `rollout.py` / `score.py` / `data_prep.py` | 静态路的三步：学生 rollout、teacher 打分、语料成 bin/idx |
| `opd_rl.py` + `config/opd_rl.yaml` | RL 式 OPD 启动器（verl 栈 + 多 teacher 配置） |
| `opd_reward.py` | RL 式的备选奖励：−mean(logπ_student − logπ_teacher)（两个 vLLM 端点） |
| `test_train.py` / `test_opd_reward.py` | 预检 / reward 闸门 |

## 快速开始

```bash
# 静态 KD（三步）
python data_prep.py --prepare --config default                       # ① 学生 rollout → bin/idx
python score.py --load <teacher 检查点> --out $CACHE/math --top-k 64 # ② teacher 打分
python train.py --config default --tokens 5e8 --load <SFT-3 检查点> --teacher-cache $CACHE/math
python train.py --config default --load <SFT-3 检查点> --teacher-cache $CACHE/math --reverse-kl

# RL 式（verl 原生多 teacher 蒸馏）
python opd_rl.py --dry-run
OPD_STUDENT_URL=http://127.0.0.1:8001/v1 OPD_TEACHER_URL=http://127.0.0.1:8002/v1 \
    python opd_reward.py --selftest                                  # reward 自检
```

## 配置

| 旋钮 | 值 / 含义 |
|---|---|
| 学生 | SFT-3（agent 段）检查点；发布基座 |
| LR | 1e-5 cosine → 1e-6，warmup 1% |
| 预算 | 每轮 0.5B tokens，可迭代 2~4 轮 |
| KD 方向 | 静态 KD 默认 forward KL；`--reverse-kl` 或 `--set train.model.logits_load_reverse_kl=true` 切 reverse KL |
| KD 系数 | `logits_load_kd_loss_alpha`（1.0 = 纯 KD，调低混 LM loss 防遗忘） |
| 多 teacher | 静态 KD 按数据域路由（各自缓存）；RL 式由 verl 按 `teacher_key` 路由 |

## 完整主跑

```bash
for dom in math code agent writing; do
  python data_prep.py --prepare --config default
  python score.py --load <$dom teacher 检查点> --out $CACHE/$dom --top-k 64
  python train.py --config default --tokens 5e8 --load <SFT-3 检查点> --teacher-cache $CACHE/$dom \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/opd_$dom
done
# RL 式：opd_rl.py（四个 teacher 已在 config/opd_rl.yaml 里按 key 路由）
```

判据：蒸馏后在四方向的可验证奖励不降于 SFT 基座、接近对应 teacher；通用能力不掉超过 1 个点。

## 发布

蒸馏完的发布模型是 mcore 检查点，评测与上线读 HF 目录：

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.common.train.export_hf \
    --ckpt <OPD 检查点> --out $SHENSI_FS/shensi/models/gdar-release-hf
```

## 判据

| 检查 | 判据 |
|---|---|
| `python test_train.py` | tiny 5 步：rc=0、`[after training is done]`、无 Traceback |
| `python test_opd_reward.py` | 10/10：KL 数学对解析解、拼接处对齐、缓存只打一次、缺端点/空响应报错 |
| `python -m shensi.recipes.paper.gated_delta_attn_res.common.train.test_reverse_kl` | reverse KL 对解析解 2.7e-07、与 forward KL 差异 2.8、补丁幂等 |

## 下一步

- [RL teacher](../stage2_rl/README.md) —— teacher 检查点的来源
- [评测](../stage4_eval/README.md) —— 发布模型的打分
