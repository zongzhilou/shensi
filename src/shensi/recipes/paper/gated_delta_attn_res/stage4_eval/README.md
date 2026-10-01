# 阶段 4：评测（在 OPD 之后）

评测对象是**已发布的发布模型**。T0 任务是受控深度检索：答案位置已知、chance 可声明（四选一 =
25%）、深度可控（长度 L 的上下文里写 K 个不同的键），因此结果能对着随机基线说、也能与
plain-Qwen3 孪生臂比。评测读 HF 目录 —— 先把发布 ckpt 导出来。

## 概览

| 组件 | 说明 |
|---|---|
| `make_depth_retrieval.py` | 生成器：按 K×L 网格合成题目 |
| `run_depth_retrieval.py` | 评分器：逐格准确率、Wilson 95% 区间、chance 对比、位置偏差、标签打乱对照 |
| `test_train.py` | 预检 + 冒烟：生成 40 题并给 tiny ckpt 评分 |
| `config/{default,tiny}.yaml` | 评测参数（题数、网格、chance、对照开关） |
| `config/data_prep/*` | 题集标识与产物路径 |

## 快速开始

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt <OPD ckpt> --out $HF/gdar-release                       # 先发布
python make_depth_retrieval.py --config default                    # 生成 1000 题
python run_depth_retrieval.py --config default --model $HF/gdar-release --out-json score.json
python test_train.py                                               # 40 题冒烟
```

## 评分口径

1. **chance** —— `chance: 0.25`（四选一），逐格报"是否越过 chance"；
2. **区间** —— 逐格 Wilson 95%；
3. **分层** —— `K ∈ {1,2,4,8}` × `L ∈ {1024,2048,4096}`；
4. **位置偏差与阴性对照** —— gold/pred 位置分布；`control_shuffle_labels: true` 打乱标签。
   低于 chance 的格子**如实报**（`usable` 门）。

## 跑完整论文实验

```bash
# ① 发布旗舰对两个臂
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf --ckpt <OPD(GDAR)> --out $HF/gdar-release
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf --ckpt <OPD(base)> --out $HF/base-release
# ② T0 受控深度检索（两个臂 + 一个打乱标签的阴性对照）
python make_depth_retrieval.py --config default
for arm in gdar base; do python run_depth_retrieval.py --config default --model $HF/$arm-release --out-json score_$arm.json; done
python run_depth_retrieval.py --config default --model $HF/gdar-release --control-shuffle-labels --out-json score_shuffled.json
# ③ 通用能力（lm-eval 套件 + 中文）与长上下文（RULER，≥8B，带 oracle 对照）
```

判据：GDAR 主行在 T0 上不低；lm-eval / RULER 不掉超过 1 个点（OPD 之后仍成立才算发布合格）。

## 索引

- [OPD](../stage3_opd/README.md) —— 被评测的模型从哪来
- [预训练](../stage0_pretrain/stage1_pretrain/README.md) —— 链路更早阶段的评测清单
