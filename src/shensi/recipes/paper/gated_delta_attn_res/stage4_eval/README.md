# 阶段 4：评测（在 OPD 之后）

评测对象是**已发布的发布模型**。T0 任务是**受控深度检索**：每道题的答案位置已知、chance 可声明
（四选一 = 25%）、深度可控（在长度 `L` 的上下文里写 `K` 个不同的键），所以结果可以对着随机基线
说、也可以和 plain-Qwen3 孪生臂比。

本 stage 跑在 [OPD](../stage3_opd/) **之后**，读的是 HuggingFace 目录 —— 先把发布 ckpt 导出来。

## 总览

| 组件 | 说明 |
|---|---|
| `make_depth_retrieval.py` | 生成器：按给定的 `K` × `L` 网格写 `n` 道题 |
| `run_depth_retrieval.py` | 评分器：逐格准确率、Wilson 95% 区间、与 chance 对比、位置偏差检查、可选的标签打乱对照 |
| `test_train.py` | 预检 + 冒烟：生成 40 题并给 tiny ckpt 评一遍 |

## 先发布模型

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd \
    --out  $SHENSI_FS/shensi/models/gdar-release-hf
```

导出器从检查点自带的 `run_config.yaml` 读几何、从那一次 run 的 `config.yaml`（或 `--model-algo`）
读连接旋钮，载入前做形状预检，写出的 HF 目录里 `config.json` 带 `model_type: qwen3_gdar`、
`auto_map`、两个随权重走的 `.py` 与 22 个 `attn_res_*` 旋钮 —— `trust_remote_code=True` 直接可加载。
更早的 stage ckpt 用同样方式导出（`--ckpt` 指到那次 run 的目录），但对外报告的是 OPD 发布模型。

## 评分口径

1. **chance / 随机基线** —— `--chance 0.25`（四选一）；每一格都报"是否越过 chance"。
2. **区间** —— 逐格 Wilson 95%，不是裸点估计。
3. **分层** —— `K ∈ {1,2,4,8}` × `L ∈ {1024,2048,4096}`（`--ks` / `--lengths`）。
4. **位置偏差** —— gold/pred 的位置分布；外加 `--control-shuffle-labels` 做阴性对照。低于 chance
   的格子**如实报**（`usable` 门），不包装成"比随机还差"。

## 快速开始

```bash
cd stage4_eval
# ① 生成题（正式档 >= 1000 题）
python make_depth_retrieval.py --out $SHENSI_FS/shensi/data/gated_delta_attn_res/eval/dr1000.jsonl \
    --n 1000 --lengths 1024,2048,4096 --ks 1,2,4,8 --seed 42
# ② 给发布模型评分
python run_depth_retrieval.py --model $HF/gdar-release --data <dr1000.jsonl> --device cuda \
    --chance 0.25 --out-json <score.json>
# ③ 预检 / 冒烟（生成 40 题，给 tiny ckpt 评一遍）
python test_train.py
```

## 判据

| 检查 | 命令 | 结果 |
|---|---|---|
| 预检 + 冒烟 | `python test_train.py` | 导入、tiny ckpt、生成 40 题、~3 秒评完 |
| 端到端链 | `train/export_hf.py` + `run_depth_retrieval.py` | HF 目录 `trust_remote_code` 可加载；`score.json` 里 `chance = 0.25`、`usable` 门在位 |

## 跑完整论文实验（EXPERIMENT_MATRIX.md §5）

```bash
# ① 发布旗舰对的两个臂
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd      --out $HF/gdar-release
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd_base --out $HF/base-release

# ② T0 受控深度检索
cd stage4_eval
python make_depth_retrieval.py --out $SHENSI_FS/shensi/data/gated_delta_attn_res/eval/dr1000.jsonl \
    --n 1000 --lengths 1024,2048,4096 --ks 1,2,4,8 --seed 42
python run_depth_retrieval.py --model $HF/gdar-release --data <dr1000.jsonl> --device cuda \
    --chance 0.25 --out-json <score_gdar.json>
python run_depth_retrieval.py --model $HF/base-release --data <dr1000.jsonl> --device cuda \
    --chance 0.25 --out-json <score_base.json>
python run_depth_retrieval.py --model $HF/gdar-release --data <dr1000.jsonl> --device cuda \
    --control-shuffle-labels --out-json <score_shuffled.json>          # 阴性对照

# ③ 通用能力（lm-eval 套件 + 中文）与长上下文（RULER，>= 8B，带 oracle 对照）
#    脚本在包里（eval/run_lm_eval.py、eval/run_ruler.py），接进来的位置与 T0 相同。
```

判据：GDAR 主行在 **T0** 上不低（论文主结论）；lm-eval / RULER 不掉超过 1 个点
（OPD 之后仍成立才算发布合格）。低于 chance 的格子通过 `usable` 门如实报出。

## 还没接的评测

lm-eval（HellaSwag / ARC / PIQA / … / CMMLU / C-Eval）与 RULER（≥ 8B，带 oracle 对照）在包里都有
可跑脚本，接进来的位置与 T0 相同。真实检索（SWDE / FDA / RAG 设定）是 T1 计划，在集群上做。

## 延伸阅读

- [OPD](../stage3_opd/README.md) —— 被评测的模型从哪来
- [预训练](../stage0_pretrain/stage1_pretrain/README.md) —— 链路更早阶段用的 0-shot / 中文 / RULER 套件
- [LIMITATIONS.md](../LIMITATIONS.md) —— A8（stage 移植）、A20（发布步）
