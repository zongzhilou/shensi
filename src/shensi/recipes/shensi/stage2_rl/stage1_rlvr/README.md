# stage1_rlvr：多环境可验证奖励 RL（RLVR）

`stage2_rl` 的第一段：单轮为主、答案可校验。数据用 post-training 里带 verifier 的 RL 集与官方
`*-Training-Blends`（它们已经把多环境、多奖励混好了），奖励由 `../reward.py` 按 verifier 规则判。

## 1. 摘要

| 项 | 值 | 出处 |
| --- | --- | --- |
| 目标 | 在可校验任务上把「答对」的比例顶起来（数学 / 科学 / 代码 / 推理） |
| 数据 | `config/data_prep/data_blend_raw.json`：11 个可校验集，权重按本阶段归一 | Nemotron-3 的 21 环境 RLVR 集与 `*-Training-Blends` |
| 算法 | GRPO（verl 原生 `algorithm.adv_estimator`），不做 KL | GLM-5 的 RLVR 段 |
| 采样 | `rollout.n = 8`，`max_response_length = 32768`（推理链要够长） | GLM-5 报告 |
| LR | 1e-6 恒定，`clip_ratio_low/high = 0.2/0.28` | GLM-5 的 IcePop 式双侧截断 |
| 优化器 | verl 的 `actor.optim`（Adam 系），与本段之外的口径一致 | 预训练用 Muon 混合的理由与 SFT 相同：小数据微调要单独扫 LR |

**算法档**：默认 GRPO；`config/gspo.yaml`（序列级重要性比，长思维链更稳）与 `config/dapo.yaml`
（clip-higher + 动态采样，group 拉到 16）都是 verl 原生的 `algorithm.adv_estimator`，`--profile gspo` /
`--profile dapo` 直接切。判分侧：代码类没有单测时可用模型判分（THUDM CodeRM-NT 的思路），
现在 `reward.py` 是 verifier 规则 + 兜底 0。

## 2. 运行

```bash
python data_prep.py --discover          # 看数据在不在
python data_prep.py --prepare           # → $SHENSI_FS/shensi/data/stage1_rlvr/{train,val}.parquet
python train.py --dry-run               # 看 verl 命令
python train.py                         # 正式跑；上一段 ckpt 用 --set model.path=<sft ckpt> 指过去
```

## 3. 验收判据

1. `critic/score/mean` 高于基线并上行；
2. `actor/entropy` 不塌到 0；
3. 同一 prompt 采样 8 次能看到不同解法（多样性还在）；
4. 早停：`../../early_stop.py --metric critic/score/mean --mode max`。

## 4. 局限

1. 奖励只覆盖规则可判的部分（`string_match` / 精确 / 数字 / `pass_rate` 软标签）；
   开放式任务的判分要等 `stage3_align` 的 GenRM 通道；
2. 11 个集的相对权重按本阶段归一，实际 token 占比要看 `--discover` 的实测条数。
