# 阶段 3：OPD（四个 teacher → 一个发布模型）

把四个方向 teacher 蒸馏回**同一个发布模型**。学生是 SFT 基座；它在各方向的数据域上 rollout，
teacher 给这些 rollout 逐 token 打分，学生在**自己 rollout 的 token** 上学习 —— 这正是
on-policy 蒸馏比离线蒸馏强的地方。

三步全部是 **Megatron-Core 原生实现**（不依赖 vLLM）：student rollout 语料准备 → 用 mcore 的
logits saver 给 teacher 打分 → 用 mcore 的缓存 logits loss 做 KD 训练。损失方向可以是
**reverse KL**（`KL(student‖teacher)`，MiniCPM5 的 OPD 口径）或 forward KL（mcore 默认）。

## 总览

| 步 | 做什么 | 实现 |
|---|---|---|
| ① rollout | 学生在各方向数据域上采样 | `data_prep.py --prepare --blend <方向>.json` |
| ② score | teacher 冻结前向，top-K/top-P logprob 落盘 | `python score.py --load <teacher> --out <缓存>` → mcore `--logits-save-dir --logits-save-top-k --freeze-all-layers --async-save` |
| ③ KD train | 学生在 rollout 序列上训练，loss 换成读缓存的 KD | `python train.py --teacher-cache <缓存>` → mcore `--logits-load-dir`（`kd_loss_alpha` 混 LM loss） |

## 配方

| 旋钮 | 值 / 含义 |
|---|---|
| 学生 | SFT-3（agent 段）ckpt |
| LR | 1e-5 cosine → 1e-6，warmup 1% |
| 序列 | 8192 |
| 预算 | 每轮 0.5B tokens；可迭代 2~4 轮 |
| KD 系数 | `logits_load_kd_loss_alpha`（1.0 = 纯 KD，调低则混 LM loss 防遗忘） |
| 损失方向 | 默认 forward KL；`--set train.model.logits_load_reverse_kl=true` 换成 reverse KL |
| 多 teacher | 按数据域路由（每个方向用自己 teacher 的缓存训自己的域），或合并缓存（≈ 在 logprob 空间平均） |

> **早停默认开**（metric `lm loss value`）；早停按成功处理。

## 快速开始

```bash
cd stage3_opd
python data_prep.py --prepare --blend math.json                        # ① 学生 rollout → bin/idx
python score.py --load <math teacher ckpt> --out $CACHE/math --top-k 64  # ② teacher 打分
python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/math   # ③ 蒸馏
python train.py --dry-run                                              # 打印命令
python test_train.py                                                   # 预检
```

## 发布发布模型

发布模型是 mcore ckpt，而评测与服务读 HF 目录，所以先发布（几何以检查点自带的
`run_config.yaml` 为准，连接旋钮来自那次 run 的 `config.yaml`）：

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd \
    --out  $SHENSI_FS/shensi/models/gdar-release-hf
```

## forward KL 与 reverse KL

| 方向 | 落点 | 说明 |
|---|---|---|
| forward KL（mcore 默认） | `--logits-load-dir` 那条路，未改动 | 覆盖 teacher 的分布质量，偏 mode-covering |
| **reverse KL**（`KL(student‖teacher)`） | `train/reverse_kl.py`，用 `--logits-load-reverse-kl` 打开 | MiniCPM5 的 OPD 口径；偏 mode-seeking。实现与 `topk_kl_div` 同签名，缓存 / top-k / TP 管线全沿用 |

单测 `train/test_reverse_kl.py` 同时校验两个方向对解析解的误差（2.7e-07），并确认两者确实不同
（Δ = 2.8）。

## 判据

| 检查 | 命令 | 判据 |
|---|---|---|
| 预检 | `python test_train.py` | tiny 几何 5 步：rc=0、`[after training is done]`、无 Traceback |
| reverse KL | `python -m shensi.recipes.paper.gated_delta_attn_res.train.test_reverse_kl` | 与解析解一致、补丁幂等且已接管 |

## 跑完整论文实验（EXPERIMENT_MATRIX.md §5 / RECIPE §4）

```bash
# 按方向（math / code / agent / writing），再合出发布模型
cd stage3_opd
for dom in math code agent writing; do
  python data_prep.py --prepare --blend $dom.json
  python score.py --load <$dom teacher ckpt> --out $CACHE/$dom --top-k 64
  python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/$dom \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/opd_$dom
done
# reverse-KL 变体（MiniCPM5 的 OPD 方向），用同一批缓存
python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/math \
    --set train.model.logits_load_reverse_kl=true --set experiment.exp_dir=$SHENSI_FS/shensi/runs/opd_rkl

# 发布，然后评测（stage4_eval）：发布模型在 SFT 套件上不允许回退
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt <OPD ckpt> --out $HF/gdar-release
```

## 延伸阅读

- [RL teacher](../stage2_rl/README.md) —— teacher ckpt 的来源
- [发布 + 评测](../stage4_eval/README.md) —— HF 目录与 T0 检索任务
- [MINICPM5_ALIGNMENT.md](../MINICPM5_ALIGNMENT.md) —— OPD 对齐项（16 专家、reverse KL、复用 prompts）
