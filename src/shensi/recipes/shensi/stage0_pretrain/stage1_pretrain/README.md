# Stage 0.1: 主预训练（稠密主干，从零）

预训练第一段：用短序列把主干训起来，**不引入 DSA**（`csa_dense_mode: true`，Lightning Indexer 不参与），
对应 GLM-5 报告的稠密基座预训练（27T 语料）。架构侧取 DeepSeek-V4-Flash 的 text_model：CSA/HCA 压缩路径、
mHC 多流超连接、3 层共享 MTP（[2606.19348](https://arxiv.org/abs/2606.19348)）。DSA 与长上下文在
`../stage2_midtrain` / `../stage3_longctx` 引入。

## Overview

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--profile` 选档、`--smoke` 跑仓库内 tiny 档、`--tokens` 按预算换算 `train_iters`、`--set` 覆写 |
| `test_train.py` | 集成测试：tiny 几何 + 本档，5 步判 PASS/FAIL（见配方 README 第 5 节） |
| `data_prep.py` | 语料 → mcore `.bin/.idx` + `blend.json`（`--discover` / `--prepare` / `--codev3`） |
| `config/` | `default.yaml`（全量）+ `debug.yaml`（极小）+ `adamw/lion/muon/ademamix`（优化器对照档） |

| 项 | 结论 |
| --- | --- |
| 目标 | 稠密主干收敛到可做中训练的底座；序列 4K 起步、收尾拉到 8K |
| 关键决定 | 优化器用 **Muon（矩阵）+ Lion（非矩阵）** 混合：四家（DeepSeek-V4 / GLM-5 / Kimi K2 / Qwen3.8-Flash-Next）的口径汇成一档；LR 取 V4-Flash 峰值 2.7e-4 |
| 验收 | 集成测试 5 步 PASS + ckpt 存续往返（含优化器状态）+ 离线闸门的 11 项优化器判定 |
| 未启用（登记） | V4.1 的 CSA2 跨层 KV 复用 / FP4 KV / Causal Encoder-Decoder；GLM-5 的 loss-free bias、GLM-5.2 的 IndexShare |

## Quick Start

```bash
python test_train.py                    # 集成测试：tiny 几何 5 步（没准备语料就退回 mock）
python data_prep.py --discover          # 语料面貌（列名 / 条数 / 权重）
python data_prep.py --prepare           # → $SHENSI_FS/shensi/data/stage1_pretrain/{*.bin,*.idx,blend.json}
python train.py --smoke                 # 仓库内 tiny 配置跑几步（验环境/入口）
python train.py --profile debug         # 真实语料的极小档（单卡 5 步）
python train.py --tokens 27e12          # 正式跑
```

`train_iters` 用 token 预算换算：`--tokens 27e12` → `iters = tokens / (global_batch_size × seq_length)`。

## 超参（除数据与几何外与 GLM-5 对齐）

| 项 | 值 | 出处 |
| --- | --- | --- |
| 序列长度 | 4096（主体）→ 8192（收尾，`--set train.model.seq_length=8192`） | GLM-5：base 从 4K 起，mid-training 才拉到 32K/128K/200K |
| 全局批 | `global_batch_size: 128`（= micro_batch × DP，按卡数调） | GLM-5 的 1b 配置口径 |
| 优化器 | **Muon（矩阵）+ Lion（非矩阵）混合**，wd 0.1、grad clip 1.0（见下节） | GLM-5 Muon Split + DeepSeek-V4 的混合口径与 γ |
| 学习率 | 2.7e-4 → 2.7e-5（V4-Flash 峰值），warmup 3%，cosine；两条腿共用 | DeepSeek-V4-Flash 报告 |
| 并行 | EP=8、TP=1、PP=1、CP=1、distributed optimizer + overlap | PP>1 时 mHC/AttnRes 的跨 stage 交接需另测，配方默认 PP=1 |
| 精度 | bf16 + attention softmax fp32 + allreduce 累加 fp32 | 同上 |
| MTP | **本段档默认 0 层**（`mtp_num_layers: 0`） | mHC 下 MTP 跑不通：实测 1 层 + 单流（`hc_mult=1`）能跑，带 mHC 必挂在形状上（配方 README 第 9 节第 2 条） |
| loss | aux 0.001 + ERC 1.0/0.5（本段 indexer KL 为 0，没有 indexer） | 配方总览的四条口径 |
| 深度连接 | **GDAR**（AttnRes 的读写）：四个门缩放合成一个 `g_scale`(4)（初始 0 → init 精确等于 `prefix + delta`，读取也一起静默）；`t` 是可学的逐通道 log 时间常数（初始 log-uniform 铺到 [1, 2×层数]）；写是 gated delta rule 的闭式解（λ=0 退回加性、λ→∞ 清空地址）；读 = 白化打分 + `config.attn_res_read_heads` 头 + Softmax₁；投影是 KimiLinear 风格的低秩具名对 `q_a/q_b`、`g_a/g_b`、`k_a/k_b`（秩 = `routed_expert_hidden_size`，全秩要多约 2.3B 参数） | 本轮改动，逐位自检 + HF/mcore/vLLM 三侧同键 |

## 数据

`config/data_prep/data_blend_raw.json` 是可直接使用的 Nemotron 预训练配置（web / code / math / specialized /
SFT 合成 / legal），权重和 = 1.0；语料来源、目录名与字段见 [`../README.md`](../README.md) 的「数据」一节。
`Nemotron-Pretraining-Code-v3` 这类**只有元数据**的数据集在 `--prepare` 时会明确跳过，
先跑 `--codev3` 落地文本（在 v1/v2 元数据基础上分类后回 GitHub 取），落地后自动进 blend。

产物：`$SHENSI_FS/shensi/data/stage1_pretrain/<数据集>__<config>_text_document.{bin,idx}` + `blend.json`
（权重 × 前缀，`train.py` 自动注入 `data_path`）。极小档用 `config/data_prep/debug_sample.json`
（`Nemotron-Pretraining-Dataset-sample` 的两个 config），tokenizer 可以是自训的小 tokenizer
（`SHENSI_TOKENIZER=<dir>`，冒烟用不着全量词表）。

## 优化器：Muon（矩阵）+ Lion（非矩阵）

默认档（`config/default.yaml`）不用 AdamW：

| 参数 | 走哪条腿 | 口径出处 |
| --- | --- | --- |
| 2D 矩阵：attention 的 q/kv/o 投影、MLA 的 `linear_q_up_proj` / `linear_o_group_proj`、MoE 专家、latent 投影 | **Muon**：Newton–Schulz 正交化 + 每参数 spectral 尺度；`muon_split_qkv` 是 GLM-5 的 **Muon Split**，本家族再叠加 MLA 的**按头**（q_up）与**按组**（o_group）切分 | GLM-5 §架构（[2602.15763](https://arxiv.org/abs/2602.15763)）、DeepSeek-V4 §优化器（[2606.19348](https://arxiv.org/abs/2606.19348)） |
| embedding / 输出头 / norm / **MoE router** / **mHC 静态与 AttnRes 门控** / 哈希嵌入 / 压缩器位置表 | **Lion**（符号动量，状态内存约 AdamW 的一半）：非 AdamW 家族里上游 mcore 明确支持的那一个 | router 归"非矩阵"是 Qwen3.8-Flash-Next 的口径，mHC 静态归"非矩阵"是 DeepSeek-V4 的口径；**为什么不是 AdEMAMix** 见配方 README 第 3 节第 4 条（上游把标量腿锁在 adam/adamw/lion/sgd） |

关键旋钮（`config/default.yaml` 的 `optimizer:` 段）：

- **`muon_extra_scale_factor: 0.18`** = DeepSeek-V4 的 **γ**：把 Muon 的更新幅度归一到 AdamW 量级，
  于是**两条腿共用一条 LR 曲线**（本档 = V4-Flash 的峰值 2.7e-4）。
- `muon_scale_mode: spectral`（Moonlight / V4 的每参数尺度）、`muon_nesterov: true`（V4 用 Nesterov）、
  `muon_coefficient_type: polar_express` + `muon_num_ns_steps: 5`（新一代 NS 系数，低步数更稳）、
  `muon_tp_mode: distributed`（GLM-5 的零冗余分布式 Muon；上游默认 `duplicated` 每步全量 all-gather）。
- **不启用 QK-clip**（Kimi 的 MuonClip）：`qk_layernorm: true` 已把 q/k 归一化——与 V4 同一条理由
  （V4 明说因 q/k RMSNorm 不需要 QK-clip），mcore 也断言 DSv4 hybrid attention 下不能开 `qk_clip`。

各档（`--profile`）：

| profile | 做法 | 用途 |
| --- | --- | --- |
| `default` | Muon（矩阵）+ Lion（非矩阵） | 正式训练 |
| `ademamix.yaml` | **整个模型**用 AdEMAMix（`--optimizer ademamix`，走 emerging 优化器表） | "两条腿都换掉"的对照 |
| `muon.yaml` | 同一套 Muon，但用 V4 的两段式系数（`quintic` + 8 步）与 `blockwise` tp_mode | 系数 / tp_mode 对照 |
| `adamw.yaml` | 老的 AdamW 口径（0.9/0.999、lr 1e-5→1e-6、warmup 3%、cosine） | 回退与对照 |
| `lion.yaml` | Lion 当**主体**优化器 | 另一条"全 Lion"对照 |
| `debug.yaml` | 极小几何 + 极小步数（本段档默认 `mtp_num_layers: 0`） | 冒烟 / 集成测试 |

依赖 `emerging-optimizers >= 0.2`：mcore 的 `TensorParallelMuon` 用它的 Newton–Schulz 内核与 NS 系数表；
缺包时 `train.py` 会在启动前直接报清楚（默认档就会触发这个检查）。

**为什么值得上 Muon**：本配方与架构的两个出处都用它——DeepSeek-V4 用 Muon 换收敛速度与稳定性，
GLM-5 用 Muon Split + 零冗余分布式 Muon；公开结果里 Muon 相对 AdamW 有 1.3~1.5× token 效率
（[2502.16982](https://arxiv.org/abs/2502.16982)，要点正是加 wd 与调每参数更新尺度 = 本档的 `muon_scale_mode: spectral`）；
Qwen3.8-Flash-Next 在同一套口径下把「2× LR 时的 loss spike」从每 10k 步 4.3 压到 0.2，并取消了批大小 warmup
（省 18.8% 步数）——本配方本来就没做批大小 warmup。

**LR 口径**：两条腿共用一条调度器曲线，只写一个 `lr`；要单独调 Muon 侧的实际步长就改 `muon_extra_scale_factor`。

## 验收判据

1. **参数路由不变量**：所有 2D 非标量参数在 Muon 那条腿，embedding / 头 / 1D / 名单（router、mHC、AttnRes 门控、
   哈希嵌入）在标量腿（Lion）；
2. **MLA Muon Split**：`linear_q_up_proj` 按头切开、`linear_o_group_proj` 按组切开；
3. **两条腿共用一条 LR**（`lr` 相同），Muon 侧幅度由 `muon_extra_scale_factor` 归一；
4. **训练健康**：3 步 loss 有限且两条腿的参数都更新；optimizer state（Lion 的 `exp_avg` / Muon 的
   `momentum_buffer`）齐全；
5. **可续训**：极小档跑过完整往返——10 步、第 5 步存（含优化器状态）→ 从 `iter_0000005` 载入后接着训到 10 步
   （`successfully loaded checkpoint ... at iteration 5`、`Traceback=0`）并再存出 `iter_0000010`；
6. **集成测试**：`python test_train.py` 5 步 PASS（rc=0、`[after training is done]`、无 Traceback）。

以上 1–4 由离线闸门全量判定（11 项）；`--profile muon` 的变体系数档也在闸门里跑过。

## 下一步

中训练（DSA 两段式）见 [`../stage2_midtrain/README.md`](../stage2_midtrain/README.md)。
环境相关的实测坑（SM120 / ray 内存账 / vllm 版本钉法）见[配方 README 的「环境注意事项」](../../README.md#8-环境注意事项实测)。

## 局限

1. 全规模收敛未验收：集成测试与极小档只证明「口径正确、能跑、能续」，token 效率要真机预算；
2. AdaMuon / PolarGrad / SOAP 等变体只在 `emerging-optimizers` 里可用，未做端到端验收；
   换档本身（`--profile adamw / lion / muon / ademamix`）在极小档跑得通：档位只换 optimizer，
   其余几何由 `tiny_model.as_cli_overrides` 压到单卡（并行度 1、单进程）；
3. SFT / RL 仍走 Adam 系：Muon 的证据都在预训练规模上，小数据微调要单独扫 LR；
4. MTP 支持 0 / 1 / 2 层且可与 mHC 同开（配方 README 第 9 节第 2 条）。
## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

全部命令都在本机真跑过（单卡），日志与 run 目录在 `$SHENSI_FS/shensi/runs/`；极小档产物的生成见配方总览的「极小档要两个本地产物」。

- `python train.py --profile debug`：5/5 步，`after training is done`，ckpt 落在
  `$SHENSI_FS/shensi/ckpt/pt_tiny_debug/iter_0000005`；
- MTP 探针（几何不动，只开 1 层 MTP 并压成单流）：3/3 步，日志里有 `mtp_1 loss`；
  带 mHC 的 MTP 会挂在形状上（见配方总览「局限」第 2 条）；
- 数据：`data_prep.py --prepare --blend config/data_prep/debug_sample.json --limit 200`
  （两份 Nemotron 样例，共 400 篇 → `blend.json` + `*_text_document.bin/.idx`）。
