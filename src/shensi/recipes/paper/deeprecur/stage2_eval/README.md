# 阶段 2：评测（stage2_eval）

把训练产物起成 OpenAI 兼容端点，用 harness 跑基准套件，汇总成表格形态的 `summary.json`。默认 harness 为 **lmms-eval**，可选 **OpenCompass**。

## 概述

评测不重写判分：套件定义在 `common/benchmarks.py`，判分交给 harness；本阶段负责端点、命令组装与结果汇总。被评模型由 vLLM 起服（`native` 臂走引擎原生实现，其余臂走桥，见配方 README 的「三镜像」）。

| 组件 | 说明 |
|---|---|
| `serve.py` | 起 vLLM 的 OpenAI 兼容端点（按臂登记桥） |
| `eval.py` | 选套件 → 组装 harness 命令 → 跑 → 收 `summary.json` |
| `common/benchmarks.py` | 套件与基准清单、候选名、口径登记 |
| `common/opencompass.py` | OpenCompass 可选后端（映射 + 结果汇总） |
| `data_prep.py` | 评测数据集审计（落位/缺件报告） |
| `config/` | 套件、端点与批大小配置 |

## 快速开始

```bash
cd src/shensi/recipes/paper/deeprecur/stage2_eval

# 离线自检（不需要 harness/数据/网络）
python eval.py --selftest

# 看命令（不执行）
python eval.py --suite main --dry-run
python eval.py --harness opencompass --suite main --dry-run

# 端到端（先起端点，再评测）
python serve.py --arm deeprecur --ckpt <检查点>
python eval.py --suite main --arm deeprecur --base-url http://127.0.0.1:8000/v1
```

## 基准套件

| 套件 | 基准 |
|---|---|
| `main` | VQAv2、GQA、TextVQA、DocVQA、InfoVQA、SEED (all)、POPE (all)、MMMU、MM-Vet |
| `text` | ChartQA、DocVQA、InfoVQA、MultiDocVQA、TextVQA |
| `video` | EgoSchema、NextQA、MSVD、ActivityNet（零样本） |
| `ablation` | GQA、POPE、SEED、TextVQA、DocVQA、ChartQA、InfoVQA |

### 双 harness

| 项 | lmms-eval（默认） | OpenCompass（可选） |
|---|---|---|
| 覆盖 | 四套件 25 项全落地 | main/text/ablation 有候选名；视频 4 项无对应 |
| 名解析 | `--resolve-tasks`（对 task 清单） | `--resolve-datasets`（双来源：配置 + VLMEvalKit 注册表） |
| 接线 | 本阶段自建（openai_compatible + 结果收集） | 复用 `shensi/stage3_eval/opencompass_eval` |

### task / dataset 名（已对真实 harness 核对）

| 基准 | lmms-eval 0.1.2 | lmms-eval 0.7.3 | OpenCompass |
|---|---|---|---|
| VQAv2 | `vqav2` | `vqav2_val_lite` / `vqav2_val` | 候选 `vqav2` |
| GQA | `gqa` | `gqa` | 候选 `gqa` |
| TextVQA | `textvqa` | `textvqa_val` | 候选 `textvqa` |
| DocVQA | `docvqa` | `docvqa_val` | 候选 `docvqa` |
| InfoVQA | `infovqa` | `infovqa_val` | 候选 `infovqa` |
| SEED | `seedbench` | `seedbench` / `seedbench_lite` | `SeedBench.seedbench_gen` |
| POPE | `pope` | `pope` | 候选 `pope` |
| MMMU | `mmmu` | `mmmu_val` | 候选 `mmmu` |
| MM-Vet | `mmvet` | `mmvet` | 候选 `mmvet` |
| ChartQA | `chartqa` | `chartqa` | 候选 `chartqa` |
| MultiDocVQA | `multidocvqa` | `multidocvqa_val` | 候选 `multidocvqa` |
| EgoSchema / NextQA / ActivityNet | —— | `egoschema` / `nextqa_mc_test` / `activitynetqa` | 无 |
| MSVD | —— | —— | 无（需自注册或等价替代） |

候选名按「当前代 → 论文代」排序，`--resolve-tasks` 自动落到本机 harness 的真实名；对不上（含 MSVD）显式报错并给原因。

## 运行

### 命令行

| 选项 | 说明 |
|---|---|
| `--suite` | `main` / `text` / `video` / `ablation` |
| `--harness` | `lmms-eval`（默认）/ `opencompass` |
| `--arm` | 被评的臂，同时是端点模型名 |
| `--base-url` | vLLM 端点地址 |
| `--output` | 结果目录（默认 `<FS>/shensi/runs/deeprecur/stage2_eval/<arm>-<suite>`） |
| `--limit N` | 每任务限量（冒烟用） |
| `--dry-run` | 只打印将执行的命令 |
| `--resolve-tasks` / `--resolve-datasets` | 核对 task / 数据集名 |
| `--selftest` | 离线自检 |

### 结果格式

`summary.json` 按基准一行，带口径标注（`*` 训练图在训练中见过、`‡` 验证集）：

```json
{
  "harness": "lmms-eval",
  "suite": "main",
  "table": [
    {"benchmark": "VQAv2", "tasks": ["vqav2_val_lite"], "scores": {"vqav2_val_lite": {"acc": 0.71}},
     "matched": true, "paper_footnote": "‡ 验证集"}
  ],
  "per_task": {"...": "..."}
}
```

### 数据

基准数据由 harness 自行下载；离线机器用 `data_prep.py --discover` / `--check` 审计预置情况（不代下）。

## 产物流

```mermaid
flowchart LR
    ckpt["指令检查点"] --> serve["serve.py<br/>(vLLM 端点)"] --> ev["eval.py<br/>(harness)"] --> res["summary.json<br/>(+ harness 原始结果)"]
```

## 上一阶段

- [阶段 1：指令微调](../stage1_sft/README.md)

## 边界

- 判分交 harness：MM-Vet 等需要 judge API 的项要自备配置；`--resolve-*` 对不上时显式报错，不静默跳过。
- 图片分辨率/预算由被评臂自己的 processor 决定——这正是三臂对比要考察的变量之一。
- 真实出分需要装 harness 的机器；本机可跑全部离线自检与 dry-run。
