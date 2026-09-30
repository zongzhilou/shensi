# stage1_pretrain：主预训练（稠密主干，从零）

预训练第一段：用短序列把主干训起来，**不引入 DSA**（`csa_dense_mode: true`，Lightning Indexer 不参与），
对应 GLM-5 报告的稠密基座预训练（27T 语料）。架构侧取 DeepSeek-V4-Flash 的 text_model：CSA/HCA 压缩路径、
mHC 多流超连接、3 层共享 MTP（[2606.19348](https://arxiv.org/abs/2606.19348)）。DSA 与长上下文在
`../stage2_midtrain` / `../stage3_longctx` 引入。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 目标 | 稠密主干收敛到可做中训练的底座；序列 4K 起步、收尾拉到 8K |
| 关键决定 | 优化器换成 **Muon（矩阵）+ AdEMAMix（非矩阵）** 混合：四家（DeepSeek-V4 / GLM-5 / Kimi K2 / Qwen3.8-Flash-Next）的口径汇成一档；LR 取 V4-Flash 峰值 2.7e-4 |
| 验收 | 优化器 11 项判定全过（离线校验）；极小档 5 步训练 + ckpt 存续往返（含优化器状态） |
| 未启用（登记） | V4.1 的 CSA2 跨层 KV 复用 / FP4 KV / Causal Encoder-Decoder；GLM-5 的 loss-free bias、GLM-5.2 的 IndexShare |

## 2. 超参（除数据与几何外与 GLM-5 对齐）

| 项 | 值 | 出处 |
| --- | --- | --- |
| 序列长度 | 4096（主体）→ 8192（收尾，`--set train.model.seq_length=8192`） | GLM-5：base 从 4K 起，mid-training 才拉到 32K/128K/200K |
| 全局批 | `global_batch_size: 128`（= micro_batch × DP，按卡数调） | GLM-5 的 1b 配置口径 |
| 优化器 | **Muon（矩阵）+ AdEMAMix（非矩阵）混合**，wd 0.1、grad clip 1.0（见第 4 节） | GLM-5 Muon Split + DeepSeek-V4 的混合口径与 γ |
| 学习率 | 2.7e-4 → 2.7e-5（V4-Flash 峰值），warmup 3%，cosine；两条腿共用 | DeepSeek-V4-Flash 报告 |
| 并行 | EP=8、TP=1、PP=1、CP=1、distributed optimizer + overlap | PP>1 时 mHC/AttnRes 的跨 stage 交接需另测，配方默认 PP=1 |
| 精度 | bf16 + attention softmax fp32 + allreduce 累加 fp32 | 同上 |
| MTP | 3 层 + 参数共享（`mtp_num_layers: 3`、`mtp_use_repeated_layer: true`） | GLM-5：3 层共享，4 步投机解码接受长度 2.55 → 2.76 |
| loss | aux 0.001 + ERC 1.0/0.5（本段 indexer KL 为 0，没有 indexer） | 配方总览的四条口径 |
| 深度连接 | **GDAR**（AttnRes 的读写）：四个门缩放合成一个 `g_scale`(4)（初始 0 → init 精确等于 `prefix + delta`，读取也一起静默）；`t` 是可学的逐通道 log 时间常数（初始 log-uniform 铺到 [1, 2×层数]）；写是 gated delta rule 的闭式解（λ=0 退回加性、λ→∞ 清空地址）；读 = 白化打分 + `config.attn_res_read_heads` 头 + Softmax₁；投影是 KimiLinear 风格的低秩具名对 `q_a/q_b`、`g_a/g_b`、`k_a/k_b`（秩 = `routed_expert_hidden_size`，全秩要多约 2.3B 参数） | 本轮改动，逐位自检 + HF/mcore/vLLM 三侧同键 |

`train_iters` 用 token 预算换算：`--tokens 27e12` → `iters = tokens / (global_batch_size × seq_length)`。

## 3. 数据

`config/data_prep/data_blend_raw.json` 是可直接使用的 Nemotron 预训练配置（web / code / math / specialized /
SFT 合成 / legal），权重和 = 1.0；语料来源、目录名与字段见 [`../README.md`](../README.md) 第 2 节。
`Nemotron-Pretraining-Code-v3` 这类**只有元数据**的数据集在 `--prepare` 时会明确跳过，
先跑 `--codev3` 落地文本（在 v1/v2 元数据基础上分类后回 GitHub 取），落地后自动进 blend。

```bash
python data_prep.py --discover          # 看目录里有什么（列名 / 条数 / 权重）
python data_prep.py --prepare           # → $SHENSI_FS/shensi/data/stage1_pretrain/{*.bin,*.idx,blend.json}
python data_prep.py --codev3 --hf-sample 40   # Code-v3 只有元数据：分类 + 回捞
```

## 4. 优化器：Muon（矩阵）+ AdEMAMix（非矩阵）

默认档（`config/default.yaml`）不用 AdamW：

| 参数 | 走哪条腿 | 口径出处 |
| --- | --- | --- |
| 2D 矩阵：attention 的 q/kv/o 投影、MLA 的 `linear_q_up_proj` / `linear_o_group_proj`、MoE 专家、latent 投影 | **Muon**：Newton–Schulz 正交化 + 每参数 spectral 尺度；`muon_split_qkv` 是 GLM-5 的 **Muon Split**，本家族再叠加 MLA 的**按头**（q_up）与**按组**（o_group）切分 | GLM-5 §架构（[2602.15763](https://arxiv.org/abs/2602.15763)）、DeepSeek-V4 §优化器（[2606.19348](https://arxiv.org/abs/2606.19348)） |
| embedding / 输出头 / norm / **MoE router** / **mHC 静态与 AttnRes 门控** / 哈希嵌入 / 压缩器位置表 | **AdEMAMix**：Adam 的双 EMA 版本（快 EMA 走 Adam、慢 EMA 留长程梯度，α=5） | 双 EMA 出处 [2409.03137](https://arxiv.org/abs/2409.03137)：1.3B/101B token 与 AdamW 的 197B 打平；router 归 Adam 系是 Qwen3.8-Flash-Next 的口径，mHC 静态归 Adam 系是 DeepSeek-V4 的口径 |

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
| `default` | 上面的混合口径 | 正式训练 |
| `muon.yaml` | 同一套 Muon，但用 V4 的两段式系数（`quintic` + 8 步）与 `blockwise` tp_mode | 系数 / tp_mode 对照 |
| `adamw.yaml` | 老的 AdamW 口径（0.9/0.999、lr 1e-5→1e-6、warmup 3%、cosine） | 回退与对照 |
| `lion.yaml` | Lion（符号更新，状态内存约 AdamW 的一半） | 备选 |
| `debug.yaml` | 极小几何 + 极小步数 | 冒烟 |

依赖 `emerging-optimizers >= 0.2`：mcore 的 `TensorParallelMuon` 在 mcore 自己那份 `emerging_optimizers.py` 里，
用它的 Newton–Schulz 内核与 NS 系数表；缺包时 `train.py` 会在启动前直接报清楚（默认档就会触发这个检查）。

**为什么值得上 Muon**：本配方与架构的两个出处都用它——DeepSeek-V4 用 Muon 换收敛速度与稳定性，
GLM-5 用 Muon Split + 零冗余分布式 Muon；公开结果里 Muon 相对 AdamW 有 1.3~1.5× token 效率
（[2502.16982](https://arxiv.org/abs/2502.16982)，要点正是加 wd 与调每参数更新尺度 = 本档的 `muon_scale_mode: spectral`）；
Qwen3.8-Flash-Next 在同一套口径下把「2× LR 时的 loss spike」从每 10k 步 4.3 压到 0.2，并取消了批大小 warmup
（省 18.8% 步数）——本配方本来就没做批大小 warmup。

**LR 口径**：两条腿共用一条调度器曲线，只写一个 `lr`；要单独调 Muon 侧的实际步长就改 `muon_extra_scale_factor`。

## 5. 运行

```bash
python train.py --dry-run               # 写 run 目录并打印 torchrun 命令
python train.py --smoke                 # 仓库内 tiny 配置跑几步（验环境/入口）
python train.py --tokens 27e12          # 正式跑
```

## 6. 验收判据

1. **参数路由不变量**：所有 2D 非标量参数在 Muon 那条腿，embedding / 头 / 1D / 名单（router、mHC、AttnRes 门控、
   哈希嵌入）在 AdEMAMix 那条腿；
2. **MLA Muon Split**：`linear_q_up_proj` 按头切开、`linear_o_group_proj` 按组切开；
3. **两条腿共用一条 LR**（`lr` 相同），Muon 侧幅度由 `muon_extra_scale_factor` 归一；
4. **训练健康**：3 步 loss 有限且两条腿的参数都更新；optimizer state（`exp_avg_slow` / `momentum_buffer`）齐全；
5. **可续训**：极小档跑过完整往返——10 步、第 5 步存（含优化器状态）→ 从 `iter_0000005` 载入后接着训到 10 步
   （`successfully loaded checkpoint ... at iteration 5`、`Traceback=0`）并再存出 `iter_0000010`。

以上 1–4 由离线闸门全量判定（11 项）；`--profile muon` 的变体系数档也在闸门里跑过。

## 7. 局限

1. 全规模收敛未验收：闸门与极小档只证明「口径正确、能跑、能续」，token 效率要真机预算；
2. AdaMuon / PolarGrad / SOAP 等变体只在 `emerging-optimizers` 里可用，**未做端到端验收**（本档只用 Muon + AdEMAMix）；
3. SFT / RL 仍走 Adam 系：Muon 的证据都在预训练规模上，小数据微调要单独扫 LR。
