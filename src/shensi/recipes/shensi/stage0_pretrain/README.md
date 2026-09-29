# stage0_pretrain：预训练三阶段

模型是 **DeepSeek-V4-Flash 的 text_model 形态**（[2606.19348](https://arxiv.org/abs/2606.19348)：CSA + HCA
混合注意力、mHC 多流超连接、Muon、1M 上下文），训练课程按 **GLM-5 系列**的预训练口径走
（[报告](https://arxiv.org/abs/2602.15763)：27T 稠密基座 + DSA via continued pretraining + 长上下文 mid-training）。

## 1. 摘要

| 项 | 内容 |
| --- | --- |
| 本目录做什么 | 把基座从零训到 1M 上下文窗口，分三段：稠密主干 → DSA 两段式 → 长上下文 |
| 关键机制 | 两阶段 DSA（冻主干只训 Lightning Indexer，KL 目标从稠密分布切到 top-k 集合）；三个 loss 全开 |
| 优化器 | 三段都默认 Muon（矩阵）+ AdEMAMix（非矩阵）混合，两条腿共用一条 LR（见 `stage1_pretrain/README.md`） |
| 语料 | Nemotron 预训练集按域加权；`Nemotron-Pretraining-Code-v3` 只有元数据，用 `codev3.py` 回 GitHub 落地文本 |
| 未启用（登记） | DeepSeek-V4.1 的 CSA2 跨层 KV 复用、FP4 KV 缓存、Causal Encoder-Decoder —— KV 压缩口径不同，要动推理侧与 ckpt 格式 |

```text
stage1_pretrain   ① 稠密主干（csa_dense_mode=true），4K → 8K，27T 量级
stage2_midtrain   ② 32K；DSA 两段式：dsa_warmup（冻主干只训 indexer，1000 步）→ sparse adaptation（20B tokens）
stage3_longctx    ③ 长上下文：128K / 500B → 1M / 50B，长文档 up-sample
```

## 2. 语料

语料放 `$SHENSI_FS/datasets/llm/pre-training/<目录名>`，目录名去掉 HuggingFace 的 `nvidia/` 前缀。
三个 stage 各有一份 `config/data_prep/data_blend_raw.json`（stage2/3 用 `base:` 继承 stage1 的权重，
只改 `min_chars` 与个别权重），`--prepare` 会把它们分别编码成 `.bin/.idx` + `blend.json`。

来源都在 [nemotron-pre-training-datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets)
集合里，按域加权：

| 域 | 数据集（目录名） | 文本列 |
| --- | --- | --- |
| 英文网页 | `Nemotron-CC-v2.1`（High-Quality / -DQA / -Synthetic / -Translated-To-English 等）、`DCLM-Baseline`、`FineWeb-Edu` | `text` |
| 中文与多语 | `FineWiki`（`en`）、`SkyPile-150B`、`Fineweb-Edu-Chinese-V2.2` | `text` |
| 代码 | `Nemotron-CC-Code-v1`、`Nemotron-Pretraining-Code-v1`、`-v2`（Synthetic-* 系列）、`OpenCoder-Pretrain`、`Ultra-FineWeb-L3`、`UltraData-Code` | `text` / `content` |
| 数学与科学 | `Nemotron-CC-Math-v1`、`UltraData-Math`、`UltraX-Preview` | `text` / `content` / `cleaned_content` |
| 专门领域 | `Nemotron-Pretraining-Specialized-v1/v1.1/v1.2`、`Nemotron-Pretraining-Legal-v1`、`Nemotron-Pretraining-SFT-v1`（类 SFT 合成，小权重）、`FinePDFs` | `text` |
| 只有元数据 | `Nemotron-Pretraining-Code-v3` | 无（见第 3 节） |

`FinePDFs` 是原 `Nemotron-Pretraining-FinePDFs`（云端目录已改名），对应 `HuggingFaceFW/finepdfs`，
列名以 `--discover` 实测为准。`OpenCoder-Instruct`（`OpenCoder-LLM/opc-sft-stage2`，instruction/output/code）
已挪到 post-training，归 `stage1_sft` 用，不进预训练 blend。

## 3. Code-v3：只有元数据时的文本落地

`Nemotron-Pretraining-Code-v3` 的 `Nemotron-Code-Metadata` 只有 `repo / rel_path / language / commit_id`
（1.46 亿行，没有文本）。`codev3.py` 做三件事：

1. **读元数据**：本地 parquet/jsonl（`--v1-meta/--v2-meta/--v3-meta`）或 HF 采样（`--hf-sample N`，调试不下载全量）；
2. **在 v1/v2 基础上分类**：按 `(repo, rel_path)` 把 v3 逐行判成"与 v1/v2 重叠且 commit 相同 / 重叠但 commit 变了 /
   v3 增量"，并给出反向覆盖（v1/v2 的清单有多少还在 v3 里）。v1/v2 的 `Synthetic-*` 配置只有数据集级
   `seed_source`、没有文件级键，不能按文件对接，所以 v1/v2 给的是**文件清单**（避免重复抓/重复训），不是文本；
3. **落地文本**：先查本地文本缓存（`--text-cache`，任何含 `repo/rel_path + text/content` 的文件都能复用），
   未命中再按 `raw.githubusercontent.com/<repo>/<commit>/<rel_path>` 取（路径 percent-encode），
   产出 `{"text": ...}` jsonl（正文首行 `# repo/rel_path @ commit`）+ 账本（404 / 跳过扩展名 / 超体积 / 太短）。

```bash
cd stage1_pretrain
python data_prep.py --codev3 --hf-sample 40 --limit 20       # 调试：分类 + 抓 20 条
python data_prep.py --codev3 --v1-meta <v1元数据> --v2-meta <v2> --v3-meta <v3>    # 正式
python ../codev3.py --selftest                                # 离线自检（分类/URL 转义/缓存/账本）
python data_prep.py --prepare                                  # 落地产物自动进 blend
```

## 4. 三段连跑

```bash
cd stage1_pretrain && python data_prep.py --prepare && python train.py --tokens 27e12
cd ../stage2_midtrain && python data_prep.py --prepare && python train.py --profile dsa_warmup
cd ../stage3_longctx && python data_prep.py --prepare && python train.py --tokens 500e9
```

## 5. 验收判据

| 段 | 判据 |
| --- | --- |
| stage1 | `validation loss` 稳定下行；三个 loss 列都在日志里；ckpt 可存可续（`torch_dist`） |
| stage2 warm-up | `indexer loss` 非零并下降，且**主干权重逐位不变** |
| stage2 sparse | `indexer loss` 继续下降；`lm loss` 不因切稀疏跳变 |
| stage3 | 长度切换后 `lm loss` 无台阶式恶化；1M 档不 OOM；长文检索抽测通过 |

每个子 stage 的 README 有该段的目标、超参对照（与 GLM-5 报告逐项对齐）、数据口径与判据。

## 6. 局限

1. 全规模收敛未验收（只跑过极小几何）；
2. 长上下文段缺 GLM-5 的自建长文档 / 合成长数据 / MRCR 类数据，当前用长文档筛选顶着（见 `stage3_longctx/README.md`）；
3. `FinePDFs` 等数据集的实际列名与分片以 `--discover` 实测为准，本 README 给的是预期值。
