# Stage 3: 评测

用 **vLLM 起服务**，跑两套评测：① **NeMo Gym** 基准（走 `dsh` 提交，需要目标环境）；
② 不依赖 Gym 的 **local 套件**（纯 HTTP + 本地打分，单机就能跑）。评测只读模型、不训练。

## Overview

| 组件 | 做什么 |
|------|--------|
| `eval.py` | 入口：`--profile` 选档、`--bench` 选基准、`--dry-run`、`--set`；起 vLLM 服务 → 跑基准 → 汇总 |
| `test_train.py` | 集成预检：配置解析、`vllm serve` 命令、vllm CLI、模型目录、GPU、import |
| `setup_env.sh` | 装 Gym / dsh 那套外部依赖（需要目标环境） |
| `config/` | `default.yaml`（全量档：服务与基准）+ `tiny.yaml`（极小档：单卡小模型/短序列） |

| 部分 | 内容 |
|------|------|
| 首选（Gym） | NeMo Gym 的基准集 + `dsh` 提交；适合与 Nemotron/GLM 的公开数字对齐 |
| local 套件 | 不依赖 Gym：起 `vllm serve` 后用 HTTP 打一批本地基准（数学/代码/指令遵循的抽样集 + 长文检索），本地规则打分 |
| 服务参数 | `config/*.yaml` 的 `serving:` 段（`model_path` / `tp` / `dp` / `kv_cache_dtype` / `max_model_len` / `reasoning_parser` / `tool_call_parser`） |

## Quick Start

```bash
python test_train.py                       # 集成预检（不真正起服务）
python eval.py --dry-run                   # 只打印 vllm serve + 评测命令
python eval.py --profile tiny              # 极小档：单卡、小模型、短序列
python eval.py --set serving.model_path=<ckpt>   # 指定要评测的 ckpt
```

服务本身可以单独起（评测之外也常用）：

```bash
vllm serve <ckpt> --served-model-name shensi --port 8000 --tensor-parallel-size 8 \
  --max-model-len 1048576 --kv-cache-dtype fp8
```

## 数据与基准

| 基准 | 需要什么 | 说明 |
|------|---------|------|
| 数学 / 科学 | 本地题库 + 规则判分（数字/字符串匹配） | 与 RLVR 的 verifier 同源，便于训练-评测对齐 |
| 代码 | 单测（沙箱执行） | 没有单测的用软标签判分 |
| 指令遵循 | 结构化输出合法率 + 规则 | 与 `stage2_rl/stage3_align` 同口径 |
| 长文检索 | 长文档 + 事实复述 | 验 128K / 1M 档的长上下文能力 |
| NeMo Gym 基准 | Gym 的容器与基准资产（`setup_env.sh`） | 与公开数字对齐用；本机没装只做预检 |

## 验收判据

1. **集成预检 PASS**（vllm CLI、serve 命令、GPU、import）；
2. 服务起来后 `/v1/models` 有 `shensi`；短 prompt 生成正常（不是乱码、不是空）；
3. 各基准的分数与训练阶段的判据对得上（例如数学口径与 RLVR 的 verifier 一致）；
4. 同一 ckpt 多次评测的方差在基准的噪声范围内（采样温度固定时应当很小）。

## 局限

1. NeMo Gym / dsh 那套要目标环境（容器、基准资产），本机只做到预检；
2. `config/default.yaml` 里的 `serving.model_path` 是占位（生产机上的 ckpt 路径），
   本机跑要 `--set serving.model_path=<本机 ckpt>`；
3. 长文检索套件是自建的抽样集，不能替代 MRCR 这类标准长上下文基准。

## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

全部命令都在本机真跑过（单卡），日志与 run 目录在 `$SHENSI_FS/shensi/runs/`；极小档产物的生成见配方总览的「极小档要两个本地产物」。

```bash
python eval.py --profile tiny_local --limit 5     # 本机离线档：本地 tiny-rl + post-training 样例集
```

- 起 `vllm serve`（`--enforce-eager`，`max_model_len` 按模型上限自动压回）→ 探活 → 打 local 套件
  5 条 → 写 `summary.json`，全程无报错；
- 分数**没有意义**（随机初始化的 3M 模型 + 极简语料），这一步只证明「服务 → 端点 → 打分 → 汇总」这条链路通；
- 云端档 `--profile tiny` 用生产的 tokenizer/权重与云端能力集；Gym 套件本机没装（`SHENSI_GYM`）。
