# 阶段 4：评测（发布之后）

评测对象是**已发布的发布模型**。主任务是"受控检索"：在一段很长的上下文里放入几个位置已知的
键，问其中一个键对应什么值，模型四选一作答。因为答案位置已知、选项数已知（四选一 = 25% 随机
基线），结果能对着随机基线说，也能与 plain 残差的孪生臂逐格比较——这正是这类深度连接要检验的
"能不能在长上下文里把早先写入的信息读出来"。

## 总览

| 组件 | 说明 |
|---|---|
| `make_depth_retrieval.py` | 题目生成器：按 K（键数）× L（上下文长度）网格合成题目 |
| `run_depth_retrieval.py` | 评分器：逐格准确率、Wilson 95% 区间、随机基线对照、位置偏差、打乱标签对照 |
| `test_train.py` | 预检 + 冒烟：生成 40 题并给 tiny 检查点评分 |
| `config/{default,tiny}.yaml` | 评测参数（题数、网格、随机基线、对照开关） |
| `config/data_prep/*` | 题集标识与产物路径 |

## 快速开始

```bash
# 先发布：mcore 检查点 → HF 目录
python -m shensi.recipes.paper.gated_delta_attn_res.common.train.export_hf \
    --ckpt <OPD 检查点> --out $HF/gdar-release

python make_depth_retrieval.py --config default                    # 生成 1000 题
python run_depth_retrieval.py --config default --model $HF/gdar-release --out-json score.json
python test_train.py                                               # 40 题冒烟
```

## 评分口径

1. **随机基线** —— 四选一，`chance = 0.25`，逐格报"是否越过基线"；
2. **置信区间** —— 逐格给 Wilson 95% 区间；
3. **分层** —— `K ∈ {1,2,4,8}` × `L ∈ {1024,2048,4096}` 的网格；
4. **位置偏差与阴性对照** —— 记录 gold / pred 的位置分布；`control_shuffle_labels: true` 打乱
   标签再评一遍。低于基线的格子**如实报**（`usable` 门）。

## 完整主跑

```bash
# ① 发布旗舰对两个臂
python -m shensi.recipes.paper.gated_delta_attn_res.common.train.export_hf --ckpt <OPD(GDAR)> --out $HF/gdar-release
python -m shensi.recipes.paper.gated_delta_attn_res.common.train.export_hf --ckpt <OPD(base)> --out $HF/base-release
# ② 受控检索（两个臂 + 一个打乱标签的阴性对照）
python make_depth_retrieval.py --config default
for arm in gdar base; do python run_depth_retrieval.py --config default --model $HF/$arm-release --out-json score_$arm.json; done
python run_depth_retrieval.py --config default --model $HF/gdar-release --control-shuffle-labels --out-json score_shuffled.json
# ③ 通用能力（lm-eval 套件 + 中文）与长上下文（RULER，8B 以上，带 oracle 对照）
```

判据：GDAR 主行在受控检索上不低于孪生臂；lm-eval / RULER 不掉超过 1 个点（OPD 之后仍成立
才算发布合格）。

## 下一步

- [OPD](../stage3_opd/README.md) —— 被评测的模型从哪来
- [配方 README](../README.md) —— 管线总览
