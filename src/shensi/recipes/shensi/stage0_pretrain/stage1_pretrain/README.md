# Stage 0.1: 主预训练（稠密主干）

预训练第一段：用短序列把主干训起来，**不引入 DSA**（`csa_dense_mode: true`，Lightning Indexer 不参与），
27T 语料量级。架构侧是家族几何：CSA / HCA 压缩路径、mHC 多流超连接、3 层共享 MTP。DSA 与长上下文在
[`../stage2_midtrain`](../stage2_midtrain/README.md) / [`../stage3_longctx`](../stage3_longctx/README.md) 引入。

## 总览

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--profile` 选档、`--smoke` 跑仓库内 tiny 档、`--tokens` 按预算换算 `train_iters`、`--set` 覆写 |
| `test_train.py` | 集成测试：tiny 几何 + 本档，5 步判 PASS/FAIL（判据见[配方总览](../README.md#验证)） |
| `data_prep.py` | 语料 → mcore `.bin/.idx` + `blend.json`（`--discover` / `--prepare` / `--codev3`） |
| `config/` | `default.yaml`（全量）+ `debug.yaml`（极小）+ `adamw` / `lion` / `muon` / `ademamix` / `grokfast`（优化器对照档） |

| 项 | 结论 |
| --- | --- |
| 目标 | 稠密主干收敛到可做中训练的底座；序列 4K 起步、收尾拉到 8K |
| 关键决定 | 优化器用 **AdaMuon（矩阵腿）+ AdEMAMix（标量腿）**，两条腿共用一条 LR 曲线（峰值 2.7e-4） |
| 验收 | 集成测试 5 步 PASS + ckpt 存续往返（含优化器状态）+ 参数路由 / 分裂 / 状态键逐项核对 |
| 未启用（登记） | CSA2 跨层 KV 复用 / FP4 KV 缓存 / Causal Encoder-Decoder；loss-free bias 负载均衡与 IndexShare 共享 indexer（本配方用三个 loss 的口径，见[配方总览](../../README.md#模型总览)） |

## 快速开始

```bash
python test_train.py                    # 集成测试：tiny 几何 5 步（没准备语料就退回 mock）
python data_prep.py --discover          # 语料面貌（列名 / 条数 / 权重）
python data_prep.py --prepare           # → $SHENSI_FS/shensi/data/stage1_pretrain/{*.bin,*.idx,blend.json}
python train.py --smoke                 # 仓库内 tiny 配置跑几步（验环境/入口）
python train.py --profile debug         # 真实语料的极小档（单卡 5 步）
python train.py --tokens 27e12          # 正式跑
```

`train_iters` 用 token 预算换算：`--tokens 27e12` → `iters = tokens / (global_batch_size × seq_length)`；
同时会把 `lr_decay_iters` 钉在这次预算上，之后放大 `train_iters` 不会拉长 LR 余弦退火。

## 数据准备

`config/data_prep/data_blend_raw.json` 是可直接使用的预训练配比（web / code / math / specialized /
SFT 合成 / legal），权重和 = 1.0；语料来源、目录名与字段见 [`../README.md`](../README.md#数据准备)
的域表。`Nemotron-Pretraining-Code-v3` 这类**只有元数据**的数据集在 `--prepare` 时会明确跳过，
先跑 `--codev3` 落地文本（在 v1/v2 元数据基础上分类后回 GitHub 取），落地后自动进 blend。

产物：`$SHENSI_FS/shensi/data/stage1_pretrain/<数据集>__<config>_text_document.{bin,idx}` + `blend.json`
（权重 × 前缀，`train.py` 自动注入 `data_path`）。极小档用 `config/data_prep/data_blend_tiny.json`
（`Nemotron-Pretraining-Dataset-sample` 的两个 config），tokenizer 可以是自训的小 tokenizer
（`SHENSI_TOKENIZER=<dir>`，冒烟用不着全量词表）。

## 训练

| 项 | 值 |
| --- | --- |
| 序列长度 | 4096（主体）→ 8192（收尾，`--set train.model.seq_length=8192`） |
| 全局批 | `global_batch_size: 128`（= micro_batch × DP，按卡数调） |
| 优化器 | **AdaMuon（矩阵腿）+ AdEMAMix（标量腿）**，wd 0.1、grad clip 1.0（见下节） |
| 学习率 | 2.7e-4 → 2.7e-5，warmup 3%，cosine；两条腿共用 |
| 并行 | EP=8、TP=1、PP=1、CP=1、distributed optimizer + overlap（PP>1 时 mHC/AttnRes 的跨 stage 交接需另测） |
| 精度 | bf16 + attention softmax fp32 + allreduce 累加 fp32 |
| MTP | 全量档 `mtp_num_layers: 3`（与几何的共享 MTP 深度一致）；极小档几何由 `tiny_model.TINY` 注入、MTP 默认 0，要试就 `--set train.model.mtp_num_layers=1`（与 mHC 同开已打通，1 / 2 层都实跑过） |
| loss | aux 0.001 + ERC 1.0/0.5（本段 indexer KL 为 0，没有 indexer） |
| 深度连接 | **GDAR**（AttnRes 的读写）：四个门缩放合成一个 `g_scale(4)`（初始 0 → init 精确等于 `prefix + delta`，读取也一起静默）；`t` 是可学的逐通道 log 时间常数（初始 log-uniform 铺到 [1, 2×层数]）；写是 gated delta rule 的闭式解（λ=0 退回加性、λ→∞ 清空地址）；读 = 白化打分 + `config.attn_res_read_heads` 头 + Softmax₁；投影是低秩具名对 `q_a/q_b`、`g_a/g_b`、`k_a/k_b`（秩 = `routed_expert_hidden_size`，全秩要多约 2.3B 参数） |

### 优化器：AdaMuon（矩阵腿）+ AdEMAMix（标量腿）

全量档（`config/default.yaml`）不用 AdamW：

| 参数 | 走哪条腿 | 依据 |
| --- | --- | --- |
| 2D 矩阵：attention 的 q/kv/o 投影、MLA 的 `linear_q_up_proj` / `linear_o_group_proj`、MoE 专家、latent 投影 | **AdaMuon**（`--optimizer adaptive_muon` → mcore 的 `TensorParallelAdaptiveMuon`）：Newton–Schulz 正交化 + 每参数谱尺度 + 第二矩自适应；`muon_split_qkv` 把 q/kv 拆开，本家族再叠加 MLA 的**按头**（q_up）与**按组**（o_group）切分 | mcore 原生 |
| embedding / 输出头 / norm / **MoE router** / **mHC 静态与 AttnRes 门控** / 哈希嵌入 / 压缩器位置表 | **AdEMAMix**（`--muon-scalar-optimizer ademamix`）：快/慢两条 EMA + 二阶矩；换 GrokFastAdamW 就把名字改成 `grokfastadamw` | [pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer) 的对应类 |

关键旋钮（`config/default.yaml` 的 `optimizer:` 段）：

- **`muon_extra_scale_factor: 0.18`**：把 AdaMuon 的更新幅度归一到 Adam 系量级，于是**两条腿共用
  一条 LR 曲线**；
- `muon_scale_mode: spectral`（每参数谱尺度）、`muon_nesterov: true`、
  `muon_coefficient_type: polar_express` + `muon_num_ns_steps: 5`（新一代 NS 系数，低步数更稳）、
  `muon_tp_mode: distributed`（零冗余分布式 Newton–Schulz；上游默认 `duplicated` 每步全量 all-gather）；
- 标量腿的旋钮：`ademamix_betas`（`(beta_fast, beta2, beta_slow)`）、`ademamix_alpha`；
  换 GrokFastAdamW 时用 `--grokfast-alpha` / `--grokfast-lamb` / `--grokfast-after-step`；
- **不启用 QK-clip**：`qk_layernorm: true` 已把 q/k 归一化，mcore 也断言 DSv4 hybrid attention 下不能开 `qk_clip`。

对照档（`--profile`）：

| profile | 做法 | 用途 |
| --- | --- | --- |
| `default` | AdaMuon（矩阵腿）+ AdEMAMix（标量腿） | 正式训练 |
| `grokfast` | 同上，标量腿换成 **GrokFastAdamW** | 标量腿对照 |
| `ademamix` | **整个模型**用 AdEMAMix（`--optimizer ademamix`，走 emerging 优化器表） | "两条腿都换掉"的对照 |
| `muon` | 纯 Muon，但用两段式系数（`quintic` + 8 步）与 `blockwise` tp_mode | 系数 / tp_mode 对照 |
| `adamw` | 老的 AdamW 口径（0.9/0.999、lr 1e-5→1e-6、warmup 3%、cosine） | 回退与对照 |
| `lion` | 旧口径：Muon + Lion（Lion 当标量腿） | 回退与对照 |
| `debug` | 极小几何 + 极小步数 | 冒烟 / 集成测试 |

**标量腿为什么能是 AdEMAMix / GrokFastAdamW**：上游 mcore 的标量腿分支只认 adam/adamw/lion/sgd，
接入层 `shensi/utils/optimizer.py` 在运行时把那一处包一层（借用上游 lion 分支做构造与全部包装、
按名字补 checkpoint 的状态键、把这两个名字登记进 `emerging_optimizers` 名字表）。
依赖 `pytorch-optimizer` 与 `emerging-optimizers >= 0.2`（pyproject 已声明；缺包时 `train.py` 会在启动前报清楚）。

## 验证

1. **参数路由不变量**：所有 2D 非标量参数在 AdaMuon 那条腿，embedding / 头 / 1D / 名单（router、mHC、
   AttnRes 门控、哈希嵌入）在标量腿（AdEMAMix）；
2. **MLA Muon Split**：`linear_q_up_proj` 按头切开、`linear_o_group_proj` 按组切开；
3. **两条腿共用一条 LR**（`lr` 相同），AdaMuon 侧幅度由 `muon_extra_scale_factor` 归一；
4. **训练健康**：loss 有限且两条腿的参数都更新；optimizer state（AdEMAMix 的 `exp_avg` / `exp_avg_sq` /
   `exp_avg_slow`、AdaMuon 的 `momentum_buffer`）齐全；
5. **可续训**：极小档跑过完整往返——10 步、第 5 步存（含优化器状态，检查点元数据里三种 AdEMAMix 状态键
   与 `momentum_buffer` 都在）→ 从 `iter_0000005` 载入后接着训到 10 步
   （`successfully loaded checkpoint ... at iteration 5`、`Traceback=0`）；
6. **集成测试**：`python test_train.py` 5 步 PASS（rc=0、`[after training is done]`、无 Traceback）。

以上 1–3 由参数分组与状态键核对；`--profile muon` / `ademamix` / `grokfast` 的对照档同样能跑通。

**本机实跑记录**（WSL2 + RTX 5080 16G，单卡）：

- `python train.py --profile debug`：5/5 步，`after training is done`，ckpt 落在
  `$SHENSI_FS/shensi/ckpt/pt_tiny_debug/iter_0000005`；
- 冒烟（mock 数据）：5/5 步，loss 4.91 → 4.45，日志里 `[shensi] ademamix 超参已挂到 OptimizerConfig`
  与 `[shensi][optim] AdaMuon（矩阵腿）+ AdEMAMix / GrokFastAdamW（标量腿）已接入` 各出现一次；
- MTP 探针（几何不动，只开 1 层 MTP）：3/3 步，日志里有 `mtp_1 loss`；
- 数据：`data_prep.py --prepare --blend config/data_prep/data_blend_tiny.json --limit 200`
  （两份 Nemotron 样例，共 400 篇 → `blend.json` + `*_text_document.bin/.idx`）。

## 产物链路

```mermaid
flowchart LR
    raw["预训练语料"] --> dp["data_prep.py"] --> d["bin/idx + blend.json"]
    d --> tr["train.py（稠密主干）"] --> ckpt["pt ckpt<br/>(torch_dist)"]
    ckpt --> next["Stage 0.2: 中训练"]
    style raw fill:#e1f5fe
    style next fill:#f3e5f5
```

## 局限

1. 全规模收敛未验收：集成测试与极小档只证明「口径正确、能跑、能续」，token 效率要真机预算；
2. AdaMuon / GrokFastAdamW 与本档的收敛曲线没做过同预算对照（`muon.yaml` / `ademamix.yaml` 只是留档）；
3. SFT / RL 现在与预训练同一套优化器口径（AdaMuon + AdEMAMix），但微调规模上的 LR 要另外扫
   （本仓默认沿用各自的 LR 曲线，没做扫描）；
4. MTP 与 mHC 可同开（1 / 2 层都在极小档实跑过）。

## 下一步

中训练（DSA 两段式）见 [`../stage2_midtrain/README.md`](../stage2_midtrain/README.md)。
环境相关的实测坑见[配方总览的「环境注意事项」](../../README.md#环境注意事项实测)。
