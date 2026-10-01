# Stage 3: 评测

用 **vLLM 起服务**，跑四套评测：① **local 套件**（纯 HTTP + 本地打分，单机就能跑）；
② **harness 套件**（默认 DeepSeek Harness 直接跑基准清单，不经 Gym）；③ **NeMo Gym**（harness 的宿主之一）；
④ **官方 MRCR**（`openai/mrcr` 开放集，多针检索）。评测只读模型、不训练；服务层始终是 `vllm serve`。

## 总览

| 组件 | 做什么 |
|------|--------|
| `eval.py` | 入口：`--profile` / `--suite` / `--dry-run` / `--set`；起 vLLM 服务 → 跑基准 → 汇总 `summary.json` |
| `test_train.py` | 集成预检：配置解析、`vllm serve` 命令、vllm CLI、模型目录、GPU、import |
| `mrcr_official.py` | 官方 MRCR：取数（按 token 桶）/ 判分（前缀哈希 + `difflib.SequenceMatcher`）/ `--selftest` |
| `setup_env.sh` | 装 harness / Gym 那套外部依赖（需要目标环境；预检会报缺什么） |
| `config/` | `default.yaml`（全量）/ `tiny.yaml`（云端极小）/ `tiny_local.yaml`（本机离线） |

| 套件（`--suite`） | 需要什么 | 说明 |
|------|---------|------|
| `opencompass` | 本机端点 + OpenCompass（独立 venv） | 跑 OpenCompass 的 LLM 基准：默认 leaderboard 集合（chat_OC15 的 17 组：mmlu / cmmlu / ceval / Gaokao / triviaqa / nq / race / winogrande / hellaswag / bbh / gsm8k / math / TheoremQA / humaneval / mbpp / gpqa / IFEval），`opencompass.datasets: all` 展开安装包里的全部数据集，也可以给逗号分隔的名字 |
| `local` | 本地能力集 + 长上下文语料 | 起 `vllm serve` 后用 HTTP 打本地基准（数学/代码/指令遵循抽样集 + 长文检索 + MRCR 类多针），本地规则打分 |
| `harness` | harness 的容器与基准资产 | 按 `harness.command` 直接跑 `bench.benchmarks` 里的清单 |
| `gym` | NeMo Gym 的检出与基准资产 | Gym 是 harness 的宿主之一，跑同一批基准名字 |
| `mrcr` | `openai/mrcr` 数据（自动下载）+ 长上下文模型 | 官方开放集；判分按数据集卡片口径（前缀哈希必须在开头，随后 SequenceMatcher 比值）；样本按官方 token 桶取 |
| `all` | local + harness + gym | 默认；`mrcr` 要显式点名（或档里 `mrcr.enabled: true`） |

`config/default.yaml` 的关键段：`serving:`（`model_path` / `tp` / `dp` / `kv_cache_dtype` / `max_model_len` /
`reasoning_parser` / `speculative_config`）、`endpoint:`（chat 端点的 `base_url` / `model` / `max_tokens` /
`temperature`）、`bench:`、`harness:`（[`../common/harness.py`](../common/harness.py) 的统一接线，默认 `dsh`）、
`local:`（能力集与长上下文语料根）、`mrcr:`（`needles` / `per_bin` / token 范围）。

## 快速开始

```bash
python test_train.py                       # 集成预检（不真正起服务）
python eval.py --dry-run                   # 只打印 vllm serve + 评测命令
python eval.py --profile tiny              # 极小档：单卡、小模型、短序列
python eval.py --suite mrcr --limit 1      # 官方 MRCR（1 条）
python eval.py --set serving.model_path=<ckpt>   # 指定要评测的 ckpt
```

服务可以单独起（评测之外也常用）：

```bash
vllm serve <ckpt> --served-model-name shensi --port 8000 --tensor-parallel-size 8 \
  --max-model-len 1048576 --kv-cache-dtype fp8
```

端点已经起好时用 `--no-serve`；`--base-url` / `--model` / `--model-path` / `--limit` / `--out` 是快捷覆写。

## OpenCompass

LLM 基准交给 OpenCompass（0.5.4，`opencompass.models.openai_api.OpenAI` 指向本机 vLLM 的
OpenAI 兼容端点）：

```bash
# 装独立 venv（依赖要 numpy<2，与训练侧 numpy 2.x 冲突，不装进训练 venv）
uv venv .venv-opencompass --python 3.12
uv pip install --python .venv-opencompass/bin/python --torch-backend=cpu \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple opencompass

# 跑（vLLM 由本脚本按 serving.* 起；数据集默认 leaderboard 集合）
python eval.py --suite opencompass
python eval.py --suite opencompass --set opencompass.datasets=humaneval --set opencompass.debug=true   # 冒烟
python eval.py --suite opencompass --set opencompass.datasets=all                                      # 全部数据集

# 离线自检（配置生成 + summary 解析 + 数据集枚举）
python opencompass_eval.py --selftest
```

- 配置：`opencompass:` 段（`venv` / `datasets` / `abbr` / `max_seq_len` / `max_out_len` / `batch_size` /
  `query_per_second` / `max_num_workers` / `debug`）；`enabled: true` 时 `--suite all` 会带上它；
- 产物：OpenCompass 的 work_dir（predictions / results / summary）落在 `output_dir/opencompass/`，
  再把 `summary/summary_*.csv` 汇总进我们的 `summary.json` 的 `opencompass` 段；
- 数据集名字按安装包 `configs/datasets/**` 的路径匹配（`--set opencompass.datasets=mmlu,gsm8k`）；
- harness（dsh）那条线不变：harness 只消费 `endpoint.base_url`，OpenCompass 打的是同一个端点。

## 基准与判分

| 基准 | 判分 | 说明 |
|------|------|------|
| 数学 / 科学 | 本地题库 + 规则（数字 / 字符串匹配） | 与 RLVR 的 verifier 同源，便于训练-评测对齐 |
| 代码 | 单测（沙箱执行） | 没有单测的用软标签 |
| 指令遵循 | 结构化输出合法率 + 规则 | 与 `stage2_rl/stage3_align` 同口径 |
| 长文检索（本地 MRCR 类） | 按出现顺序全对 | 题面来自 `stage3_longctx/build_longctx.py --step mrcr` |
| 官方 MRCR | 前缀哈希 + SequenceMatcher 比值 | `openai/mrcr` 开放集，按官方 token 桶取样本 |
| harness / Gym | 宿主自己的判分 | 与公开数字对齐用；本机没装只做预检 |

## 验证

1. **集成预检 PASS**（vllm CLI、serve 命令、GPU、import；`serving.model_path` 不存在时当场报错）；
2. `/v1/models` 有 `shensi`，短 prompt 生成正常；
3. 各基准的分数与训练阶段的判据对得上；
4. 同一 ckpt 多次评测的方差在基准噪声内。

本机实测：`--profile tiny_local --limit 5` 与"从 SFT ckpt 导出后评测"两次都跑通（起服务 → 探活 →
打 local 套件 → 写 `summary.json`）；`mrcr_official.py` 自检 4/4（原样回答满分、缺前缀 0 分、打乱掉分、
空回答 0 分）+ 取数自检（2 针 / 约 5.5K tokens / 16 条消息）。

## 局限

1. 外部依赖（harness 容器、Gym 宿主、基准资产）都要目标环境；预检逐项报「有没有、缺什么、怎么装」。
   **服务层不变**：harness 只消费 `endpoint.base_url`；
2. `config/default.yaml` 的 `serving.model_path` 是占位（生产机上的 ckpt 路径），本机跑要
   `--set serving.model_path=<本机 ckpt>` 或 `--profile tiny_local`；
3. 官方 MRCR 要有长上下文模型才跑得出分数（最小桶也是 4096–8192 tokens）；本地 MRCR 类套件
   用于短上下文链路自检。

## 前序阶段

- [Stage 0: 预训练](../stage0_pretrain/README.md) — 稠密主干、DSA、长上下文
- [Stage 1: SFT](../stage1_sft/README.md) — 指令微调
- [Stage 2: RL](../stage2_rl/README.md) — RLVR / agentic / 对齐 / 世界模型
