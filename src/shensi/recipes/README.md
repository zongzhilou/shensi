# recipes：训练配方总目录

本目录是 Shensi 的**全部训练配方**，按阶段拆成 4 个顶层 stage；PT 与 RL 各自内建子 stage。
配方的模型出处是 DeepSeek-V4-Flash 的 text_model，训练课程对齐 GLM-5 / 5.1 / 5.2 / 5.3，
数据用法参考 Nemotron-3。

## 1. 目录结构

```text
shensi/
├── README.md                        配方总览（模型与配方的出处、四条口径、目录约定、早停、环境变量）
├── early_stop.py                    日志看门狗：超耐心就发 SIGTERM
├── common.py                        共用：配置合并 / 启动 / bin-idx 编码 / 语料扫描
├── rl.py                            共用：RL 的 yaml → verl CLI 映射与启动
├── config/                          冒烟档（tiny.yaml）与 HF 参考几何（hf/9b_a4b.json）
├── train/                           训练侧运行时：入口 + 上游 mcore 训练循环 + torchrun launcher
├── codev3.py                        Nemotron-Pretraining-Code-v3 的文本落地
├── fetch_code_from_metadata.py      按元数据回 GitHub 取代码
├── stage0_pretrain/                 预训练（三阶段，内建子目录）
│   ├── stage1_pretrain/             ① 稠密主干，4K → 8K
│   ├── stage2_midtrain/             ② 32K + DSA 两段式（dsa_warmup.yaml → default.yaml）
│   └── stage3_longctx/              ③ 128K → 1M（default.yaml / 1m.yaml）
├── stage1_sft/                      SFT（mcore --sft，DeepSeek-V4 chat 编码）
├── stage2_rl/                       RL（verl GRPO + Megatron actor，四个子 stage）
└── stage3_eval/                     评测（vLLM 服务 + dsh 走 NeMo Gym；含离线 local 套件）
```

各 stage 里额外保留的档：`stage1_pretrain/config/{muon,adamw,lion}.yaml`（优化器对照档）、
`stage2_midtrain/config/dsa_warmup.yaml`（冻主干只训 indexer 的那一段）、`stage3_longctx/config/1m.yaml`（1M 档）、
`stage2_rl/stage2_agentic/config/world_model.yaml`（Sim RL：环境交给世界模型）；
`stage2_rl` 的第四个子 stage 是 `stage4_world_model`（把环境模拟练成模型）。

## 2. 每个 stage 的约定

| 项 | 约定 |
| --- | --- |
| 入口 | 训练类 stage 用 `train.py` + `data_prep.py`；评测类 stage 用 `eval.py`（无 `data_prep.py`，基准集由 Gym 自己拉） |
| 配置 | `config/default.yaml`（正式）+ `config/debug.yaml`（极小档）+ `config/data_prep/{default.yaml,data_blend_raw.json,data_blend_tiny.json}` |
| 多子 stage | 父目录只放 `README.md` 与共用代码，子 stage 各自带 `config/`、`train.py`、`data_prep.py` |
| 数据路径 | 语料在 `$SHENSI_FS/datasets/llm/{pre,post}-training/<数据集名>`，产物在 `$SHENSI_FS/shensi/data/<stage>/` |
| 早停 | `early_stop.py --log <exp_dir>/logs/host_0_localhost.output --metric <指标> --mode {min,max}` |

## 3. 从上到下跑一遍

```bash
cd shensi/stage0_pretrain/stage1_pretrain
python data_prep.py --discover
python data_prep.py --prepare
python train.py --dry-run
python train.py --tokens 27e12
```

每个 stage 的 README 里有它自己的数据口径、超参对照（与 GLM-5 报告逐项对齐）、判据与局限。
从 [`shensi/README.md`](shensi/README.md) 进：那里有模型与配方的出处、四条已定的口径、目录约定与环境变量。
