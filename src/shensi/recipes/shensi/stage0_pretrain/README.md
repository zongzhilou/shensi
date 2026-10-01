# Stage 0: Pretraining

把基座从零训到 1M 上下文窗口，分三段：稠密主干 → DSA 两段式 → 长上下文。模型是
**DeepSeek-V4-Flash 的 text_model 形态**（[2606.19348](https://arxiv.org/abs/2606.19348)：CSA + HCA 混合注意力、
mHC 多流超连接、Muon、1M 上下文），课程按 **GLM-5 系列**的预训练口径走
（[报告](https://arxiv.org/abs/2602.15763)：27T 稠密基座 + DSA via continued pretraining + 长上下文 mid-training）。

## Overview

| 组件 | 做什么 |
|------|--------|
| `stage1_pretrain/` | ① 稠密主干（`csa_dense_mode=true`），4K → 8K，27T 量级 |
| `stage2_midtrain/` | ② 32K；DSA 两段式：`dsa_warmup`（冻主干只训 indexer，1000 步）→ sparse adaptation（20B tokens） |
| `stage3_longctx/` | ③ 长上下文：128K / 500B → 1M / 50B，长文档 up-sample |
| `codev3.py` | `Nemotron-Pretraining-Code-v3` 只有元数据 → 按 `repo/rel_path@commit` 回 GitHub 落地文本（见第 3 节） |
| `fetch_code_from_metadata.py` | 同一件事的独立小工具（只需要一批元数据时用它） |

每个子 stage 里都有 `train.py`（入口）、`data_prep.py`（语料 → bin/idx）、`test_train.py`（tiny 几何跑 5 步）、
`config/`（`default.yaml` 全量档 + `debug.yaml` 极小档 + 对照档）。

## Quick Start

```bash
# 集成测试：tiny 几何 + 本段档，5 步判 PASS/FAIL（没准备语料就退回 mock 数据）
cd stage1_pretrain && python test_train.py

# 真实语料的极小档
python data_prep.py --discover                 # 看语料面貌（格式 / 条数 / 字段名 / 权重）
python data_prep.py --prepare --blend config/data_prep/debug_sample.json
python train.py --profile debug                # 单卡 5 步

# 三段连跑（正式）
cd stage1_pretrain && python data_prep.py --prepare && python train.py --tokens 27e12
cd ../stage2_midtrain && python data_prep.py --prepare && python train.py --profile dsa_warmup
cd ../stage3_longctx && python data_prep.py --prepare && python train.py --tokens 500e9
```

日志实时写 `<exp_dir>/logs/host_0_localhost.output`；`train.py --dry-run` 只写 run 目录并打印 torchrun 命令。

## 数据

语料放 `$SHENSI_FS/datasets/llm/pre-training/<目录名>`，目录名去掉 HuggingFace 的 `nvidia/` 前缀。
三个 stage 各有一份 `config/data_prep/data_blend_raw.json`（stage2/3 用 `base:` 继承 stage1 的权重，
只改 `min_chars` 与个别权重），`--prepare` 会把它们分别编码成 `.bin/.idx` + `blend.json`。

产物（`$SHENSI_FS/shensi/data/<stage>/`）：

```text
<stage>/
├── <数据集>__<config>_text_document.bin / .idx     # 一篇文章一条样本 + 尾部 EOD
├── <数据集>__<config>.jsonl                        # 编码前的中间文本（便于回查）
└── blend.json                                      # 权重 × 前缀（交错），train.py 自己注入 data_path
```

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

## Code-v3：只有元数据时的文本落地

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

## 训练

三段共用一套结构（`train/system` 并行与显存、`train/model` 几何与优化器、`train/data` 语料），差异只在档里：

| 档 | 用途 | 关键差异 |
|----|------|---------|
| `stage1_pretrain/config/{default,debug,adamw,lion,muon,ademamix}.yaml` | ① 主预训练 | 全量档 GBS 128 / 35 层 / 8K；`debug` 2 层 5 步；四个对照档换优化器口径 |
| `stage2_midtrain/config/{default,debug,dsa_warmup,mtp_draft}.yaml` | ② 中训练 | `dsa_warmup` 冻主干只训 indexer（LR 5e-3 常数）；`mtp_draft` 冻主干只训 MTP |
| `stage3_longctx/config/{default,debug,1m}.yaml` | ③ 长上下文 | `1m` 把 `seq_length` 拉到 1048576、CP 8 |

覆写与调试：

```bash
python train.py --set train.model.global_batch_size=256        # 改超参
python train.py --profile muon --set experiment.load=<ckpt>    # 从某个 ckpt 续
python train.py --early-stop 20                                # 训练 + 看门狗（验证指标超耐心就收尾）
```

## 验收判据

| 段 | 判据 |
| --- | --- |
| stage1 | `validation loss` 稳定下行；三个 loss 列都在日志里；ckpt 可存可续（`torch_dist`） |
| stage2 warm-up | `indexer loss` 非零并下降，且**主干权重逐位不变**（`--shensi-freeze indexer`） |
| stage2 sparse | `indexer loss` 继续下降；`lm loss` 不因切稀疏跳变 |
| stage3 | 长度切换后 `lm loss` 无台阶式恶化；1M 档不 OOM；长文检索抽测通过 |

集成测试（每段都能单独跑）：`python test_train.py`（tiny 几何 5 步 + 收尾校验，判据见配方 README 第 5 节）。

## 血缘

```mermaid
flowchart TB
    raw["原始语料<br/>Nemotron 预训练集"] --> dp1["data_prep.py<br/>stage1"]
    raw --> dp3["codev3.py<br/>Code-v3 回落地"]
    dp1 --> d1["bin/idx + blend.json"]
    dp3 --> d1
    d1 --> p1["train.py<br/>稠密主干 27T"]
    p1 --> c1["基座 ckpt"]
    c1 --> p2["train.py<br/>dsa_warmup → sparse 20B"]
    p2 --> c2["中训 ckpt"]
    c2 --> p3["train.py<br/>128K 500B → 1M 50B"]
    p3 --> base["1M 上下文基座"]
    base --> next["Stage 1: SFT"]

    style raw fill:#e1f5fe
    style base fill:#e1f5fe
    style next fill:#f3e5f5
```

## 局限

1. 全规模收敛未验收（只跑过极小几何与集成测试）；
2. 长上下文段缺 GLM-5 的自建长文档 / 合成长数据 / MRCR 类数据，当前用长文档筛选顶着（见 `stage3_longctx/README.md`）；
3. `FinePDFs` 等数据集的实际列名与分片以 `--discover` 实测为准，本 README 给的是预期值；
4. MTP 目前只支持 0/1 层（上游 MTP 与家族层的接口差异，见配方 README 第 9 节第 2 条）。

## 下一步

预训练完成后进 [Stage 1: SFT](../stage1_sft/README.md) 做指令微调。环境相关的实测坑（SM120 / ray 内存账 /
vllm 版本钉法）见[配方 README 的「环境注意事项」](../README.md#8-环境注意事项实测)。

## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

全部命令都在本机真跑过（单卡），日志与 run 目录在 `$SHENSI_FS/shensi/runs/`；极小档产物的生成见配方总览的「极小档要两个本地产物」。

三段按顺序连着跑，迭代号是**连续**的（mcore 的迭代计数跨阶段接着算）：

```bash
cd stage1_pretrain   && python data_prep.py --prepare --blend config/data_prep/debug_sample.json --limit 200 \
                     && python train.py --profile debug            # 5/5 步，存 iter_0000005
cd ../stage2_midtrain && python train.py --profile debug            # 载入 iter 5 → 跑到 10
cd ../stage3_longctx  && python train.py --profile debug            # 载入 iter 10 → 跑到 15
```

每段的 debug 档都自带 `checkpoint.load`（指向上一环的产物），所以单独跑 `--profile debug` 就能接上；
从头跑就把 `experiment.load` 与 `train.system.checkpoint.load` 设为空。
