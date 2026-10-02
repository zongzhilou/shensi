# Stage 4：公开基准评测（vLLM 端点 + OpenCompass）

用 **vLLM 起的 OpenAI 兼容端点**跑 **OpenCompass** 的 LLM 基准：默认跑它自带的 leaderboard 集合
（`chat_OC15`，17 组），也可以点名跑主力口径集合；工具类基准（SWE-Bench / BFCL / τ²-Bench / GAIA /
Terminal-Bench）交给 **deepseek-harness**（dsh），打的是同一个端点。

## Overview

| 组件 | 说明 |
|---|---|
| `setup_env.sh` | 装 OpenCompass 的独立 venv（它要 numpy<2，与训练侧冲突）、dsh 的独立 venv、vLLM 插件入口点，并把 dsh profile 指向本机端点 |
| `eval.py` | 起 `vllm serve` → 跑 OpenCompass → 可选跑 harness → 写 `summary.json` |
| `opencompass_eval.py` | OpenCompass 的执行层：生成配置（模型 = 本机端点）、选数据集、起 CLI、汇总 summary、参考分对照、`--selftest` |
| `benchmarks.py` | 口径表：口径名 ↔ OpenCompass 数据集模块（`oc`）、参考分数、集合定义 |
| `config/{default,tiny}.yaml` | serving / endpoint / OpenCompass / harness 四段配置；`tiny` 是离线小档 |

## Quick Start

```bash
cd src/shensi/recipes/paper/looma/stage4_eval

# ① 一次性准备评测环境（OpenCompass 装 GitHub 最新版 + dsh + vLLM 插件）
bash setup_env.sh

# ② 离线自检：配置生成 / 数据集枚举 / summary 解析 / 参考分对照（不占 GPU）
$SHENSI_ROOT/.venv/bin/python opencompass_eval.py --selftest

# ③ 先看命令（vllm serve + opencompass + dsh）
$SHENSI_ROOT/.venv/bin/python eval.py --dry-run

# ④ 跑：默认 leaderboard 集合；主力口径集合；冒烟子集；工具类
$SHENSI_ROOT/.venv/bin/python eval.py
$SHENSI_ROOT/.venv/bin/python eval.py --suite minicpm5
$SHENSI_ROOT/.venv/bin/python eval.py --suite mini
$SHENSI_ROOT/.venv/bin/python eval.py --suite agent

# ⑤ 端到端小跑：tiny 检查点 + 单个数据集（每集只取少量样本，几分钟跑完）
$SHENSI_ROOT/.venv/bin/python -m shensi.recipes.paper.looma.common.models.vllm.tiny_checkpoint --out /tmp/looma_smoke
$SHENSI_ROOT/.venv/bin/python eval.py --config tiny --set opencompass.datasets=ifeval.IFEval_gen --limit 1
```

## 评测集

数据集取值（`opencompass.datasets`，或用 `--suite` 选集合）：

| 取值 | 内容 |
|---|---|
| `leaderboard`（默认） | OpenCompass 自带 `chat_OC15` 集合：mmlu / cmmlu / ceval / Gaokao / triviaqa / nq / race / winogrande / hellaswag / bbh / gsm8k / math / TheoremQA / humaneval / mbpp / gpqa / IFEval |
| `minicpm5` | 主力口径集合（知识 / 推理 / 数学 / 指令跟随 / 代码共 17 项，见 `benchmarks.py`） |
| `mini` | 冒烟子集：`mmlu_pro` / `math_500` / `ifeval` / `humaneval` |
| `long` | `longbench` / `ruler`（128K 档）/ `needlebench` |
| `agent` | 工具类（走 harness，不经 OpenCompass） |
| `all` | 主力口径项 + 工具类 |
| `逗号名单` | 直接给模块名前缀，如 `mmlu_pro.mmlu_pro_0shot_cot_gen,gsm8k.gsm8k_gen` |

口径表（`benchmarks.py`）：每一项给口径名、展示名、OpenCompass 模块前缀、参考分与备注。名字按住
OpenCompass 的**模块路径**匹配，一个选择项只取排序后的第一个（安装包里同一数据集常有多份带 hash
的等价配置、变量名还都一样，多份会互相覆盖）。跑完 `summary.json` 里附一张对照：

```text
mmlu_pro   实测 0.00  参考 70.8  Δ -70.80
gsm8k      实测 0.00  参考 82.1  Δ -82.10
```

参考分用于同口径对照，**不是 pass/fail 门**。OpenCompass 里没有的项（如 `mmlu_redux`）在表里
`oc=None`：点名（逗号名单 / `--set`）匹配不到就报错；混在集合里（如 `--suite all`）会打印出来
跳过，都不静默丢。

## 命令与配置

```bash
python eval.py [--profile default] [--suite <集合>] [--limit N] [--model-path <HF 目录>] \
               [--dry-run] [--no-serve] [--set 键=值]
```

| 开关 | 说明 |
|---|---|
| `--suite <集合>` | `leaderboard`（默认）/ `minicpm5` / `mini` / `long` / `agent` / `all` |
| `--limit N` | 每个数据集最多评多少条（写进生成配置的 `reader_cfg.test_range`） |
| `--model-path` | 覆盖 `serving.model_path`（默认指 OPD 的发布检查点） |
| `--no-serve` | 端点已起好，直接打（配合 `serving.start: false`） |
| `--set 键=值` | 点号键覆写，可多次；显式给 `opencompass.datasets` 时以它为准 |

配置四段（`config/default.yaml`）：

- `serving`：`vllm serve` 的参数（`model_path` / 端口 / TP / `max_model_len` / `dtype` /
  `trust_remote_code` / `enforce_eager`）。给的 `max_model_len` 超过检查点的
  `max_position_embeddings` 时会自动压回上限。
- `endpoint`：OpenCompass 与 harness 共用的 OpenAI 兼容端点。
- `opencompass`：`datasets` / `abbr` / `max_seq_len` / `max_out_len` / `mode`（`none`·`front`·`mid`·`rear`，
  输入超限时的截断方式）/ `batch_size` / `query_per_second` / `retry` / `max_num_workers` / `debug` /
  `venv`（默认 `<仓库根>/.venv-opencompass`）。
- `harness`：工具类基准的 harness 段（`name: dsh` + 追加参数）。

配置里的 `${oc.env:变量,默认值}` 由 `eval.py` 自己展开（装载器是朴素 YAML，不做插值）。
输入长度按**本配方的分词器**算（生成的配置里 `tokenizer_path` 指向
`common/tokenizer/MiniCPM5-2B`），截断决策与模型侧一致。

## OpenCompass 的安装与版本

装在独立 venv（它要 numpy<2），`setup_env.sh` 装的是 **GitHub 最新版**（依赖走镜像）：

```bash
uv venv .venv-opencompass --python 3.12
uv pip install --python .venv-opencompass/bin/python --torch-backend=cpu \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "opencompass @ git+https://github.com/open-compass/opencompass.git"
```

生成的配置长这样（数据集一律展开成字面量 import，聚合行写在 `read_base()` 外面——OpenCompass 的
配置解析器只允许块里出现 `from … import …`）：

```python
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import OpenAI

models = [dict(type=OpenAI, abbr='looma', path='looma',
               openai_api_base='http://127.0.0.1:8000/v1/chat/completions', key='EMPTY', ...)]
```

产物：OpenCompass 的 work_dir（`predictions/` / `results/` / `summary/summary_*.csv`）落在
`<runs>/looma/stage4_eval/<profile>/opencompass/`，再把 summary 汇总进同目录的 `summary.json`。

## 已验证

| 项 | 命令 | 结果 |
|---|---|---|
| 环境 | `bash setup_env.sh` | OpenCompass（GitHub 最新版）+ dsh 各自独立 venv；vLLM 插件入口点写进训练 venv；dsh profile 指向端点 |
| 离线自检 | `python opencompass_eval.py --selftest` | 9/9：配置含端点与 leaderboard 集合、走 venv 的 CLI、口径表可解、数据集枚举、口径表逐条可解（21 个 oc 项）、summary 解析、参考分对照 |
| 端点 | tiny 检查点 + `--config tiny` | `Resolved architecture: LoomaForCausalLM`（走本配方的 vLLM 原生件），`max_model_len` 用 tiny 档的 256（给超过检查点上限的值会被自动压回），eager 下关掉 torch.compile/CUDAGraph |
| 端到端出分 | `python eval.py --config tiny --limit 2 --set opencompass.datasets=gsm8k.gsm8k_gen` | vLLM 端点（原生实现）→ OpenCompass 拉数据、按 2 条样本推完、出 summary；tiny 是随机初始化的 2 层模型，0 分是应该的，链路本身跑通 |

## 与其它 stage 的关系

- 评测的输入是 **OPD 之后的发布模型**（也可以是任意 HF 目录：SFT、RL teacher、tiny 冒烟检查点）。
- 工具类基准不在 OpenCompass 里：`--suite agent` 走 harness（dsh），打同一个端点。
