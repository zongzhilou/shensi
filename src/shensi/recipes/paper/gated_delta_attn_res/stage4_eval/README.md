# stage4_eval：评测（受控深度检索 T0）——**在 stage3_opd 之后**

位置：整条链是 PT → Mid → SFT → RL → **OPD（发布模型）** → **评测**。评测读的是**HF 目录**，而每个 stage 的产物都是 mcore ckpt，所以跑评测前先发布一次：

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd \
    --out  $SHENSI_FS/shensi/models/gdar-release-hf
```

（`export_hf` 的几何以**检查点自己的** `run_config.yaml` 为准，连接旋钮来自那次 run 的`config.yaml` / `--model-algo`；导出的目录自带 `model_type: qwen3_gdar`、`auto_map` 与两个随权重走的 `.py`，`trust_remote_code=True` 直接可加载。）同一个工具也能评更早的 stage（把 `--ckpt` 指到对应的 exp 目录即可），但对外报告的是 OPD 发布模型。

按 `EXPERIMENT_MATRIX.md` §5 的口径，T0 必做项是**受控深度检索**（最新值检索）：题目可定义、
chance 可声明（四选一 = 25%），是本项目唯一能"对着随机基线说话"的检索任务。生成器与评分器
逐字来自 gdar_package（`train/eval_depth_retrieval.py` / `eval/run_depth_retrieval.py`），
只把 `--data-dir` 默认值换成本配方的数据根（`--filler-random` 用内置填充文本，不依赖语料）。

## 口径（评审合规四件套，评分器里都在）

1. **chance / 随机基线**：`--chance 0.25`（四选一），报告里逐层给"是否越过 chance"；
2. **区间**：Wilson 95%；
3. **分层**：`K ∈ {1,2,4,8}` × `L ∈ {1024,2048,4096}`（`--ks` / `--lengths`）；
4. **位置偏差**：gold/pred 位置分布检查；另有 `--control-shuffle-labels` 把标签打乱作阴性对照。
   低于 chance 的格子**如实报**（`usable` 门），不出"比随机还差"的结论。

## 用法

```bash
cd stage4_eval
# ① 生成题（正式档 >= 1000 题）
python make_depth_retrieval.py --out $SHENSI_FS/shensi/data/gated_delta_attn_res/eval/dr1000.jsonl \
    --n 1000 --lengths 1024,2048,4096 --ks 1,2,4,8 --seed 42
# ② 评分（HF 目录，自带 tokenizer）
python run_depth_retrieval.py --model <hf_dir> --data <dr1000.jsonl> --device cuda \
    --out-json <score.json>
# ③ 预检 / 冒烟（生成 40 题 + tiny ckpt 评一遍）
python test_train.py
```

## 实测（本机）

`python test_train.py`：导入 ✓、tiny ckpt ✓、生成 40 题 ✓、评分 3 秒 ✓；
`score.json` 里 `chance=0.25`、`pooled_acc=0.175`（随机权重的 tiny 模型，符合"低于 chance"）。

## 跑完整论文实验（EXPERIMENT_MATRIX.md §5：评测）

```bash
# ① 发布：把 OPD 的发布模型导成 HF 目录（GDAR 主行与 base 对照各一次）
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf     --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd --out $HF/gdar-release
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf     --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage3_opd_base --out $HF/base-release

# ② T0 受控深度检索（≥1000 题；K×L 分层 + Wilson95 + 位置偏差 + 标签打乱阴性对照）
cd stage4_eval
python make_depth_retrieval.py --out $SHENSI_FS/shensi/data/gated_delta_attn_res/eval/dr1000.jsonl     --n 1000 --lengths 1024,2048,4096 --ks 1,2,4,8 --seed 42
python run_depth_retrieval.py --model $HF/gdar-release --data $.../dr1000.jsonl --device cuda     --chance 0.25 --out-json $.../score_gdar.json
python run_depth_retrieval.py --model $HF/base-release --data $.../dr1000.jsonl --device cuda     --chance 0.25 --out-json $.../score_base.json
python run_depth_retrieval.py --model $HF/gdar-release --data $.../dr1000.jsonl --device cuda     --control-shuffle-labels --out-json $.../score_shuffled.json   # 阴性对照

# ③ 通用能力（lm-eval 三件套 + 中文）与长上下文（RULER，≥8B，带 oracle 对照）
#    脚本在包里（eval/run_lm_eval.py / eval/run_ruler.py），接进来的位置与 T0 相同。
```

判据：GDAR 主行相对 base 对照在 **T0** 上不低（论文主结论）；lm-eval/RULER 不掉超过 1 个点
（OPD 之后仍成立才算发布合格）；低于 chance 的格子如实报（`usable` 门），不出"比随机还差"的结论。

## 还没接的评测（按设计清单）

lm-eval 套件（HellaSwag/ARC/PIQA/…/CMMLU/C-Eval）与 RULER（≥8B，含 oracle 对照）在包里都有
可跑脚本（`eval/run_lm_eval.py` / `eval/run_ruler.py`），接进来的位置与 T0 相同；真实检索
（SWDE/FDA/RAG 设定）按 `EXPERIMENT_MATRIX.md` §5 的 T1 计划在集群上做。
