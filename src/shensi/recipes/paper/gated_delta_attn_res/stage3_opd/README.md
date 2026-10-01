# stage3_opd：On-Policy Distillation（把 RL teacher 蒸馏回发布基座）

四个方向（数学/代码/Agent/写作）的 RL teacher 训完后，OPD 把它们的能力蒸馏回**同一个发布
模型**——学生 = SFT 三段之后的基座，老师 = 各方向 RL teacher。学生自己 rollout，老师打分，
学生在**自己 rollout 的 token** 上向老师学——这正是 on-policy 蒸馏优于离线蒸馏的地方。


> **早停默认开**：本 stage 步数/轮次给到无限大，收敛与收尾交给看门狗（metric=`lm loss value`，
> patience 见 `config/default.yaml` 的 `early_stop` 段；超耐心 SIGTERM 收尾、按成功返回；
> `--no-early-stop` 可关；改耐心用 `--early-stop N`）。

## 与 MiniCPM5-2B OPD 的对照（口径 + 差异）

| 项 | MiniCPM5-2B（官方公开口径） | 本配方 |
|---|---|---|
| 专家数 | **16 个 RL 专家（其中 5 个 Agent 专家）** | 4 臂起步（math/code/agent/writing）；加臂即加目录，扩到 16 是加 `stage2_rl/stage2_*` + 一个缓存目录的事 |
| 蒸馏数据 | **复用各 teacher 训练用过的 prompts**（不另做数据策展） | 学生 rollout 的 prompt 直接取自对应方向的 RL prompts（`data_prep.py --blend <方向>.json` 按域路由） |
| 损失方向 | **reverse KL**（KL(student‖teacher)，全词表或 top-k logits 并集）作为 advantage | ③ 步用 mcore 原生 KD（`--logits-load-dir`，**forward KL** + alpha 混合 LM loss）；top-k 截断用 `--logits-save-top-k`。reverse-KL/advantage 的 RL 式变体见下方"升级点" |
| 方法出处 | Thinking Machines 的 OPD + "Rethinking On-Policy Distillation" 的改进 | 同一目标，工程落点改成 mcore 原生三步（下） |

升级点（如实）：把 ③ 的静态 KD 换成 **reverse-KL advantage 的 RL 式 OPD**（学生 rollout → teacher
logprob → advantage = −KL(student‖teacher) → 用 RL 循环更新）时，接 `stage2_rl` 同一套 verl
栈即可；mcore 侧的 `logits_saver`/`LossFuncCallable` 已经把"打分/读回"两端都做成原生的。

## 链路（mcore 原生三步，不再依赖 vLLM）

| 步 | 做什么 | 用什么 |
|---|---|---|
| ① rollout | 学生（当前 OPD ckpt）在各方向数据域上采样 | 任意推理栈（vLLM/`models/vllm/`、或 SFT 数据的复用）；产出与 bin/idx 同源的 jsonl |
| ② score | teacher 冻结前向 + top-K/top-P logprob 落盘 | **`python score.py --load <teacher> --out <缓存>`** → mcore `--logits-save-dir --logits-save-top-k --freeze-all-layers --async-save --use-persistent-ckpt-worker` |
| ③ KD train | 学生在 rollout 序列上训练，loss 换成读缓存的 KD | **`python train.py --teacher-cache <缓存>`** → mcore `--logits-load-dir`（`kd_loss_alpha` 混 LM loss）；**方向可选**：默认 forward KL，加 `--set train.model.logits_load_reverse_kl=true` 换成 **reverse KL（KL(student‖teacher)）——MiniCPM5 的 OPD 口径**（实现与单测见 `train/reverse_kl.py` / `train/test_reverse_kl.py`） |

## 配方（0.6B 档）

- 学生：SFT-3（agent 段）ckpt；`config/default.yaml` 给 LR 1e-5 cosine→1e-6、warmup 1%、seq 8192、
  0.5B tokens/轮（旗舰档按比例放大；官方未公开 OPD 的 token 预算）。
- KD 系数：`logits_load_kd_loss_alpha`（1.0 = 纯 KD；调低则混 LM loss 防遗忘）。
- 多 teacher 融合：按**数据域路由**（每个方向用自己 teacher 的缓存训自己的域），或合并成一份
  混合缓存（≈在 logprob 空间平均）。

## 跑法

```bash
cd stage3_opd
python data_prep.py --prepare --blend math.json                     # ① 学生 rollout → bin/idx
python score.py --load <math teacher ckpt> --out $CACHE/math --top-k 64   # ② 打分
python train.py --tokens 5e8 --load <SFT-3 ckpt> --teacher-cache $CACHE/math   # ③ 蒸馏
```

## 发布（交给评测/上线）

蒸馏完的发布模型是 mcore ckpt，评测与 rollout 读 HF 目录——中间那一步是
`train/export_hf.py`（几何以检查点自带的 `run_config.yaml` 为准，连接旋钮来自 run 的
`config.yaml`）：

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd --out $HF/gdar-release
# 之后：stage4_eval 的 T0/通用评测、以及任何 vLLM 服务，都指向这个目录
```

## 判据与局限

- 判据：蒸馏后学生在各方向的可验证奖励（复用 `stage2_rl/*/reward.py`）不降于 SFT 基座、
  接近对应 RL teacher；通用能力（lm-eval 三件套）不掉超过 1 个点。
- 局限：① 步的采样器随 vLLM 接入补齐（`models/vllm/` 已可 serve 七个变体）；② 步与 ③ 步
  已是 mcore 原生实现（本 stage 的 `score.py`/`train.py` 即接线）。

---

## 跑完整论文实验（EXPERIMENT_MATRIX.md §5 / RECIPE §4：OPD）

把四方向 teacher 蒸馏回**同一个发布基座**（旗舰对的 GDAR 臂与 base 臂各蒸一套）。
链路是 mcore 原生三步（① rollout → ② teacher 打分 → ③ KD 训练），rollout 的驱动按学生的
模型类型自动接线（见 `rollout.py`）。

```bash
cd stage3_opd
BASE=$SHENSI_FS/shensi/runs/gdar_30b_sft/qwen3_gdar_main-sft3        # 学生 = 发布基座
TEACHERS=$SHENSI_FS/shensi/runs/gdar_30b_rl                          # ①四方向 teacher（stage2_rl 的产物）

for d in math code agent writing; do
  CACHE=$SHENSI_FS/shensi/runs/gdar_30b_opd/$d
  # ① 学生 rollout（prompts 复用该方向 teacher 训练用过的 prompts）
  python rollout.py --load $BASE --prompts $TEACHERS/$d/prompts.jsonl \
      --out $CACHE/rollouts.jsonl --base-url http://127.0.0.1:8000/v1
  # ② teacher 打分（冻结前向 + top-K logprob 落盘）
  python score.py --load $TEACHERS/$d/ckpt --data-dir $CACHE --out $CACHE/logprobs --top-k 64
  # ③ 学生 KD 训练（多轮迭代：用上一轮的 ckpt 继续 ①）
  python data_prep.py --prepare --blend $d.json
  python train.py --profile geoms/qwen3_30b_a3b --tokens 5e9 \
      --load $BASE/ckpt --teacher-cache $CACHE/logprobs \
      --set experiment.exp_dir=$CACHE/train
done

# 评测：同 SFT 套 + 方向对应项（数学/代码/Agent/写作）
```

每个方向蒸一轮后，可把 ③ 的产物当新的学生再跑 ①（on-policy），迭代 2–4 轮。
