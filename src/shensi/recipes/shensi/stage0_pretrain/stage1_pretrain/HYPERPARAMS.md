# 训练配方的逐参数依据（HYPERPARAMS）

**口径**：模型 = 默认几何 `common/config/hf/2p8b_a1p2b.json`（2.2930B 总 / **0.7570B 激活**，19 层 +
1 MTP，256 专家 top-6，前 3 层 hash-MoE）；预算 = 唯一 19B × 3 epoch = **57B passes**（= 1.24 ×
Chinchilla@总计）；硬件 = 8×910B2（3.0 PFLOPS bf16，20–30% MFU = 13.2–19.8 万 token/s）；
优化器 = AdaMuon（矩阵腿）+ GrokFastAdamW（标量腿）。

**怎么用这张表**：改任何一条超参，同时改这里的行与 `config/*.yaml` 的注释；带 ⚠️ 的是**未锚定项**
（公开文献查不到直接依据），各自给了 A/B 方案，别当成结论。文献里的每个数都给了 arXiv 编号与
出处小节；同规模但没有公开数值的（Qwen3 的 LR/aux 系数）标了「未公开」。

---

## A. 主预训练核心超参

| 参数 | 本配方值 | 文献锚点（精确值） | 适用性判断 |
|---|---|---|---|
| 峰值 LR | **2.7e-4** | DeepSeek-V4-Flash 2.7e-4（本架构同源口径）；Cerebras-2.7B **2e-4** @ 53B token（arXiv:2304.03208 Table 1——规模/预算最接近的锚）；Pythia-1.4B 2e-4（2304.01373 Table 1）；phi-1.5 2e-4（2309.05463 §2.3）；DeepSeek-V3 2.2e-4（2412.19437 §4.2）；GLM-4.5 2.5e-4（2508.06471 §2.4）；OLMoE-1B-7B 4e-4（2409.02060 Table 10，5T token 档） | 取 2–2.7e-4 带的上沿（同源架构值）。**整套配方里最该上机标定的一条**：公开锚的步批是 1–64M token、我们 1.05M，AdaMuon+GrokFastAdamW 也没有公开的 η(B) 公式（Moonlight 只发 η(C)/B(C) 拟合，2502.16982 Table 9）→ 上机先跑 `pilot_lr.py` |
| 调度 | **cosine → 10%** | 众数：Llama-2（2307.09288 Table 1）、Pythia（2304.01373）、Cerebras（2304.03208）、DeepSeek-V3（2412.19437 §4.2：2.2e-4→2.2e-5）、OLMoE（4e-4→4e-5）、Moonlight（4.2e-4→4.2e-5）；GLM-4.5 **明确不用 WSD**（§2.4：WSD 在 SimpleQA/MMLU 上更差、stable 段欠训） | 短跑（≤1× Chinchilla）用 cosine 的支撑最广；WSD 要 ≥10% 步数的衰减段（MiniCPM 2404.06395 §4.3；Hägele 2405.18392）——我们没有「多跑一圈复用 ckpt」的需求，不换 |
| min_lr | **10% 峰值**（2.7e-5） | Llama-2 / Pythia / Cerebras / DeepSeek-V3（2.2e-5）/ Moonlight / OLMoE / OLMo-2（「calibrated to reach 10% of peak」2501.00656 §2.3） | 文献众数。SmolLM2/OLMoE 的「退到 0」是另一种口径（2502.02737 §3.1），收益未在短跑里验证 |
| warmup | **3%（≈1.6k 步）** | 绝对 2000 步是众数：Llama-2（2000）、TinyLlama（2000）、DeepSeek-V3（2000）、Moonlight（2k）、OLMo-2 1B/7B（2000）、SmolLM2（2000）、DeepSeek-LLM（2000）；OLMoE 2500；Llama-3-405B 8000（2407.21783）；Pythia 1%；Cerebras-2.7B 347 步（0.7%） | 5.4 万步 × 3% ≈ 1.6k ≈ 众数 2000 ✓ |
| 每步 token | **1.05M**（mbs 8 × DP 8 × 累积 4 × seq 4096） | Cerebras-2.7B **1.08M** @ 53B token（2304.03208 Table 1）；phi-1.5 4.2M（2309.05463 §2.3）；Pythia 2.1M；OLMoE ~4M；Llama-2 4M；OLMo-2-1B ~2.1M；TinyLlama 2M；SmolLM2 2M。MoE：Moonlight 16.8→33.6M（2502.16982 §3.3）、DeepSeek-V3 12.6→62.9M（2412.19437 §4.2）、GLM-4.5 16→64M（2508.06471 §2.4） | 1–3B 模型的公开步批全在 1.05–4.2M；我们原来的 524k（0.19 token/总参）低于所有 1–3B 锚 → 抬到 1.05M。**批次 ramp**（V3 3072→15360、GLM-4.5 16→64M、Llama-3 4→16M）都是前 ~3% token 且 LR 不随动——Cerebras-2.7B 没有 ramp，我们规模也不大 → 不加（要加就按前 3B token 262k→1.05M 线性，LR 不动） |
| weight decay | **0.1**（解耦） | Llama-1/2、DeepSeek-V3（§5.2）、GLM-4.5（§2.4）、Moonlight（「0.1 for all stages」）、Kimi K2、OLMoE/OLMo-2；Pythia 是 0.01（例外） | Muon 家族**必须**有 wd：Moonlight Fig.2 记录无 wd 的权重会涨出 bf16 范围（2502.16982 §2.2） |
| grad clip | **1.0** | Llama-1（2302.13971）、Llama-2、Pythia、Cerebras、DeepSeek-V3（§4.2 明确 1.0）、DeepSeek-LLM、OLMo-2（Table 4）、OLMoE（Table 10）；Moonlight/GLM-4.5/Kimi K2 没提（= 不用） | 众数；没有「Muon 需要另设 clip」的公开依据 |
| AdamW β | **(0.9, 0.95)** | Llama-1（2302.13971 §2）、Llama-2、TinyLlama、Pythia（Table 6）、Cerebras、DeepSeek-V3（§5.2）、OLMoE（Table 10）、SmolLM2、DeepSeek-V4 的 AdamW 组（2606.19348 §4.2.2）；phi-1.5 是 (0.9, 0.98)（离群） | 原来 0.999 改 0.95（本仓此前用的是库默认） |
| AdamW ε | **1e-8**（库默认） | Pythia/OLMoE/Cerebras/OLMo-2/`torch` 默认都是 1e-8；DeepSeek-V4 的 AdamW 组用 1e-20（2606.19348 §4.2.2） | 1e-20 会放大极稀参数的更新，我们没做过数值验证 → 保持 1e-8（记 V4 口径备查） |
| 初始化 std | **0.02** | GPT-2 官方代码与 Megatron/mcore 默认；DeepSeek-V3 用 0.006（2412.19437 §5.1，但配的是他家自己的残差/输出投影缩放） | 与 HF 逐位对拍的前提之一（换值要同步改 HF 侧） |
| dropout | **0** | GLM-4.5「did not use dropout」（2508.06471 §2.4）；Mixtral/V3/V4/OLMoE 都不用 | ✓ |

## B. 优化器内部

| 参数 | 本配方值 | 文献锚点 | 适用性判断 |
|---|---|---|---|
| Muon 动量 | **0.95** | Muon 官方实现默认与文档（「0.95 is usually fine」）；Moonlight §2.2「tuning momentum 没有稳定增益，取 0.95」 | ✓ mcore 用 `lerp_` 的 EMA 形式，与经典 momentum 在 NS 尺度不变性下等价 |
| Nesterov | **on** | Muon blog：每个测过的任务里略好 | ✓ |
| NS 步数 | **5** | Moonlight §2.2（「N=10 不会更好」）；GLM-4.5 §2.4（N=5）；Muon 实现 `newtonschulz5(..., steps=5)` | ✓ |
| NS 系数 | **polar_express** | Polar Express（2505.16932 §4）推荐 T=5 或 6；其 GPT-2-124M 对照 μPolar 3.588 < Jordan 3.639 < AdamW 4.197 | ✓（要更稳可试 6） |
| 更新幅度（extra scale） | **0.18** | DeepSeek-V4-Flash 0.18（本架构同源）；Moonlight 与 GLM-4.5 **0.2**（Moonlight Appendix Table 8 扫过 0.05/0.1/0.2/0.4/0.8）；解析匹配 AdamW `sqrt((1-β1)/(1+β1)) = 0.2294`（β1=0.9） | 三者在 0.18–0.23 带内 → 取同源的 0.18。⚠️ **上机必做一步实测**：AdaMuon（`adamuon` 二阶路径）可能在 NS 之后再归一，`=0.18` 不等于「逐元素更新 RMS = 0.18」——首跑看日志里的 scale factor 行/手算一步更新 RMS，须落在 0.18–0.23；若差 ~√hidden 量级就把 `muon_scale_mode` 换 `unit_rms_norm` |
| 标量腿 | **GrokFastAdamW**（α=0.98, λ=2.0） | GrokFast（2405.20233）推荐 **λ∈[0.5,5]、α∈[0.8,0.99]**；0.98/2.0 是论文的算法/IMDb 任务最优点（§4.1/§4.4） | ⚠️ 未锚定项①：论文任务全是 grokking 级别（模乘/MNIST/QM9/IMDb-LSTM），**没有 LLM 尺度的对照**。值落在推荐带内；库的 `normalize_lr=True`（默认）把标量腿 LR 除以 (1+λ)=3 来补偿 `ĝ=g+λµ` 的放大——设计如此。A/B：`--set train.model.optimizer.grokfast=false`（关掉滤波、腿的 LR 不变）跑 1k 步比 loss |
| 标量腿名单 | embedding / 输出头 / norm / router / AttnRes 门控 / 哈希嵌入 / 压缩器位置表 | DeepSeek-V4：embedding/头/RMSNorm 走 AdamW、其余（含 router 与专家）走 Muon（2606.19348 §4.2.2）；GLM-4.5 同样把 embedding/bias/norm 排除在 Muon 外（2508.06471 §2.4） | ✓ 与文献的「Muon 只吃 2D 矩阵、embedding/norm 另走」一致 |
| 两条腿共用一条 LR | 是（extra scale 归一） | Moonlight §2.2/§3.2：「Muon 可以直接复用为 AdamW 调好的 LR 与 wd」（靠 update-RMS 0.2 对齐）；Kimi K2 Algorithm 1 同样注释「Match Adam RMS」 | ✓ 文献主流；实现细节见上一条 |

## C. MoE 与稀疏注意力专属

| 参数 | 本配方值 | 文献锚点 | 适用性判断 |
|---|---|---|---|
| 负载均衡 | **seq_aux_loss，系数 0.001** | DeepSeek-V2-Lite（与我们同量级）**只用 α1 = 0.001**、无 device/communication 平衡损失（2405.04434 App. B.1）；Qwen 的 LBL 单因子扫描 0.001/0.004/0.008 → **0.001 最好**（2501.11873 Table 4，arXiv 版） | 从 mcore 默认的 Switch 式 `aux_loss` 换成 DeepSeek 系的逐样本口径。V3/V4/GLM-4.5 之后用「序列损失 0.0001 + 路由偏置 γ=0.001」（2412.19437 §4.2 / 2606.19348 §4.2.2 / 2508.06471 §2.4）——但偏置会改**前向**路由，HF 参考里没有（Bridge 侧 `moe_router_enable_expert_bias=False` 是刻意的对拍约束）→ 不采用 |
| hash 层 | **前 3 层** | DeepSeek-V4：`num_hash_layers = 3`（2606.19348 §2.1/§4.2.1）；Hash Layers（2106.04426）：hash 层不需要任何平衡损失、逐层独立 hash 表更好（Table 3：23.21 vs 23.74 PPL） | ✓ mcore 自动豁免 hash 层的 aux（`router.py`: `aux_loss_enabled = not is_hash_layer`）→ 0.001 只作用在 16 个学习路由层上 ✓ |
| 路由打分 | **sqrtsoftplus + topk scaling 1.5 + norm_topk_prob** | DeepSeek-V4：`scoring_func="sqrtsoftplus"`、`routed_scaling_factor=1.5`（TensorRT-LLM/HF 的 V4 config）；V3 世系是 sigmoid + 2.5 | 与几何 JSON 同源 ✓ 不改（改了对拍就炸） |
| router jitter | **0** | Switch 用 1e-2（2101.03961 App. C）；ST-MoE 在 XL 尺度**去掉** jitter 反而更好（2202.08906 §3.2："diminishes quality at XL-scale"） | ✓ |
| 路由 z-loss | 未启用（0） | ST-MoE c_z = 0.001（2202.08906 §3.3，−1.741 vs −1.755）；OLMoE 也用 0.001（2409.02060 Table 1） | 可选稳定项；DeepSeek/GLM 世系不依赖它 → 先不开，若早期看到 logits 漂移再加（A/B 已备） |
| 专家 LR / wd / dropout | **与全局一致 / 0.1 / 0** | 没有任何公开配方给专家单独的 0.1× LR（V3/V4/GLM-4.5/Mixtral/OLMoE 都是单条 LR）；wd 0.1 统一（GLM-4.5 §2.4、V4 §4.2.2）；dropout 0（GLM-4.5 明确） | 「专家 LR ×0.1」属流传的说法，无一手依据 → 不用 |
| MTP 损失权重 λ | **0.3**（后段降 0.1） | DeepSeek-V3：λ=**0.3** 前 10T → **0.1** 余下 4.8T（2412.19437 §4.2）；DeepSeek-V4：0.3，进入 LR 衰减段降 0.1（2606.19348 §4.2.1/§4.2.2）；GLM-4.5：0.3 前 15T、0.1 之后（2508.06471 §2.4） | 三家一致 ✓。mcore 只有单值（默认 0.1），本配方取 0.3；最后 ~15% 想按文献降档就续跑时 `--set train.model.shensi_mtp_loss_scaling_factor=0.1` |
| MTP 深度 | **1**（重复层） | V3/V4/GLM-4.5 全是 depth 1；Meta（2404.19737）Table S7/Fig.3：<3B 时 MTP 增益很小甚至反转，≥7B 才明显；GLM-5 用 3 层**共享参数**（2602.15763 §2.1） | ⚠️ 未锚定项②：2.8B 上 MTP 的净收益没有公开数据 → 深度保持 1、λ 用 0.3，靠 loss_gate 看主干 loss 有没有被拖 |
| indexer KL | warmup 段系数 1（`dsa_warmup.yaml`）；sparse 段 0.01（`midtrain`） | DeepSeek-V3.2-Exp：冻结主干、只训 indexer，1000 步 × 16 seq × 128K、LR 1e-3；**KL 就是 indexer 的目标**（权重 1、输入 detach）；sparse 段在 top-k 集合上继续 KL（报告 §2.1）。GLM-5：1000 步 × 14 seq × 202752、max LR **5e-3**（2602.15763 §2.1.1），之后 ~20B token 的 sparse adaptation | warmup 档与我们 `dsa_warmup.yaml` 的 1000 步 × 14 × 202752 @5e-3 **逐项一致** ✓。⚠️ 未锚定项③：把 KL 折进 LM loss 的系数「0.01」公开配方里没有 → 保持并 A/B（0.01 vs 0.05）看验证 loss 与 top-k 命中率 |

## D. 数据、课程与各段预算

| 决策 | 本配方值 | 文献锚点 | 适用性判断 |
|---|---|---|---|
| 主段预算 | **唯一 19B × 3 = 57B passes** | Chinchilla 20 t/p 总参（2203.15556 Table 3 行：67B↔1.5T = 22 t/p）；细粒度 MoE 表：1B 激活 ≈ **29 t/p 激活**（2402.07871 Table 2）；phi-1.5 23 t/p 发布（2309.05463）；phi-1 1.3B×50B passes 8×A100 4 天发布（2306.11644） | 57B = 24.9 t/p 总参 = 75 t/p 激活 = 2.6× MoE 律最优点 → 「够算力就多训」的一侧，且 ≤4 epoch（见下） |
| 重复 epoch | **3（≤4）** | Muennighoff（2305.16264）：`D_eff = U·(1 + R*·(1−e^−(R−1)/R*))`、**R* = 15.3878**，「up to 4 epochs 的重复对 loss 几乎无损」；我们 3 epoch 的 D_eff = 54.6B（≈ 57B 的 96%） | ✓ 3 < 4 在安全区 |
| 语料构成 | 精选 13–14B + 合成 3–5B + 蒸馏 1.7–3B（合成 ≈ 25–30%） | FineWeb-Edu（2406.17557）：edu 过滤在 350B token 上 MMLU +4pp、ARC +11pp，「~10× token 效率」；DCLM（2406.11794）；合成占比：phi-1.5 80%（2309.05463 §2.3，code/math 靶向）、Phi-4 **40%**（2412.08905 Table 5）、SmolLM2 4%（2502.02737 §4.5） | 通用可发布模型取 Phi-4 式的 25–40%（而不是 phi-1.5 的 80%，那会窄化到 code/math） |
| 序列课程 | 主段 4K → 32K 段 **2B token**（1000 步 × 64 × 32K，尾段 LR）→ 128K 用推理期 YaRN | 位置插值（2306.15595）：32K 扩展只需 1000 步、LR 2e-5；DeepSeek-V3 的 32K 段 = 1000 步 × 1920 × 32K（§4.3）；Qwen2.5（2412.15115 §3.3）：训到 32K，**128K 用 YaRN+DCA 推理期扩展**；长/短混 = 40%/60% | 我们 2B 的 32K 段 = 文献口径的 1000 步 ✓；128K 训练段（1m.yaml）是更大预算可选项（V3 的 128K 段 ≈63B token） |
| 中训练 | 3B（5.3% 主段）、LR 1e-5 常数（峰值 1/27） | OLMo-2：中训练 = **5–10% FLOPs**（2501.00656 §2.1）；Phi-4：LR 比预训练**低 10×**、数据 **70% 回放 + 30% 新采**（2412.08905 §3.3）；DeepSeek sparse 段本身就是常数 7.3e-6 | ✓ 3B/57B = 5.3% 在带内；常数不回暖 ✓ |
| SFT | 1B、LR 1e-5 cosine、2 epoch | Ouro 2.6B：2e-5（2510.25741 Table 6）；Tülu 3：5e-6 @8B（2411.15124 Table 11）；MiniCPM SFT ≈ 6B token（2404.06395 §6.2）；2 epoch 是众数（Tülu/SmolLM2/Ouro/Llama-2） | 1e-5 落在 1–2e-5 带 ✓ |
| 评测与发布口径 | 见下节 | 同量级公开模型的分数（Table 4/9 系）：OLMoE-1B-7B（5.1T token）54.1/80.0/62.1/84.2/79.8/70.2；OLMo-2-1B（4T）26.9/67.5/26.1/—/—/67.8；TinyLlama-1.1B（3T）33.6/60.8/38.1/69.5/71.7/60.1；Pythia-1B（300B）31.1/48.0/31.4/63.4/68.9/52.7 | 它们的 token 是我们的 5–90 倍 → 只作「量级参照」，不作 bar |

## E. 「发布级收敛可测」：预期 loss 与验收

### E1. 理论预期（`loss_gate.py --selftest` 会打印同一张表）

Chinchilla Approach-3 未取整常数（E=1.6934, A=406.4, α=0.3392, B=410.7, β=0.2849，2203.15556 Table A1）
+ N = 0.7570e9 激活 + **MoE 折扣 0.09 nats**（OLMoE 实测：同激活 MoE ≈ dense 的 3× token，
2409.02060 §4.1.1）+ 重复修正（R*=15.3878）：

| passes | epoch | D_eff | 预期 loss（law + MoE 折扣） |
|---|---|---|---|
| 5.7B | 0.30 | 5.7B | 2.681 |
| 17.1B | 0.90 | 17.1B | 2.498 |
| 28.5B | 1.50 | 28.3B | 2.431 |
| 57.0B | 3.00 | 54.6B | **2.357** |

**两个必须说清的偏差**：① 公开跑法（Pile / GPT-2-BPE）的实测比该律**低 ~0.46 nats**（用 Cerebras 的
Table 8 回验：1.3B@26.3B 观测 1.997 vs 律 2.472 等五组，偏置稳定 −0.46~−0.48）——所以绝对阈值必须用
**自己的语料**标定（`--offset`）；② 上表是「完全退火后」的值，训练中段（LR 还高）天然更高。
因此判据是**容差带**：任一检查点 > 预期 + 0.30 nats → 异常（先查数据/LR），终值 > 预期 + 0.15 → warn。

### E2. 闸门怎么跑

```bash
python -m shensi.recipes.shensi.common.train.loss_gate --selftest           # 本机：打印上表
python -m shensi.recipes.shensi.common.train.loss_gate \
    --log runs/stage1_pretrain/logs/host_0_localhost.output \
    --unique-tokens 1.9e10 --epochs 3 --offset <自己的标定值> --out loss_gate.json
```

早期探测（arXiv:2410.11840）：终值从 ~1/3 进度起就可估（用 ≤30% 的 token），估准度 ~4%（小 checkpoint
预算时 5–20%）——所以在 30% 处判「有没有偏」是统计上站得住的。

### E3. 验收 bar（发布可测）

- **loss**：跑完 cosine，0 spike / 0 NaN / 0 skipped；`loss_gate.py` 结论 ≠ fail；
- **评测**（`stage3_eval`，0-shot/5-shot 按各 benchmark 惯例）：HellaSwag ≥ 45、PIQA ≥ 68、
  WinoGrande ≥ 52、ARC-E ≥ 55、MMLU ≥ 26（> random）、ARC-C ≥ 27；SFT 后 HumanEval ≥ 25 **或**
  GSM8K ≥ 10。
  ⚠️ 这是**外推**的 bar：57B passes / 0.76B 激活档没有公开同口径模型（介于 phi-1.5 的 23 t/p 与
  TinyLlama 的 2730 t/p 之间），bar 取 Pythia-1B（300B token）与 Pythia-410M 之间插值——发布时按
  「评测链路可信 + loss 在带内」讲故事，而不是「追平大厂同级」。
- **loss 之外没有第二把尺**：MTP 接受率（推测解码）/top-k 命中率（DSA）是归因用的诊断量，不作 bar。

### E4. 未锚定项清单（⚠️ 要在箱上 A/B，别当结论）

1. **GrokFastAdamW @ LLM 尺度**——论文只到 IMDb-LSTM 级；α=0.98/λ=2.0 在推荐带内但没有 LLM 对照。
   A/B：`optimizer.grokfast=false`，各 1k 步。
2. **MTP @ 2.8B 的净收益**——Meta 的曲线 <3B 时很小/可能为负；V3/V4/GLM-4.5 都是大模型。
   监控主干 loss（λ=0.3 若拖累就降 0.1）。GLM-5 的「3 层共享」是省显存变体，可作后续。
3. **indexer KL 折进 LM loss 的 0.01**——公开配方里 KL 是独立目标（权重 1、detach）。
   A/B：0.01 vs 0.05，看验证 loss 与 top-k 命中率。
4. **ERC 损失（1.0 / α=0.5）**——本仓自有损失，没有公开对照；A/B 只能内部对照（关掉 vs 开）。
5. **hash 层数（3）在 2.8B 的收益**——锚点是 V4 的 3 层，但 V4 是 100B+ 级；Hash Layers 原文
   （2106.04426 §5.3.1）11 层实验里第 1 层放 hash「效果最差但仍可用」。
6. **AdaMuon 的实际更新 RMS**（见 B 表）——首跑必须实测一步，落在 0.18–0.23 才算「幅度对齐」。

## F. 上机顺序（把「理论最优」变成「实测最优」）

```bash
# ① 吞吐（先知道一个 token 要多久）
python -m shensi.recipes.shensi.common.train.bench_throughput --steps 20
# ② LR 三点（唯一必须实测的核心超参；1k 步 × 3 点 ≈ 3.2B token ≈ 预算的 5.6%）
python -m shensi.recipes.shensi.common.train.pilot_lr --steps 1000
# ③ 把 pilot 选中的 LR 回写 default.yaml（同时改注释），正式跑：
python train.py --profile scale8 --tokens 5.7e10
# ④ 跑起来后随时对表（30% 处 > +0.30 nats 就停）：
python -m shensi.recipes.shensi.common.train.loss_gate --log <run>/logs/*.output --unique-tokens 1.9e10
```

**已知的两处「文献与我们不同」**（如实记下，不当结论）：
- 我们的步批 1.05M 低于 **MoE** 公开锚（16–64M token/步）——那些是大集群；在小模型 + 小集群上，
  1–3B dense 的锚（1–4M）更可比。若吞吐实测把 MFU 顶到 30% 且显存有余，可试 2.1M（gbs 512）。
- 我们走 aux-loss 平衡而不是 V4 的 noaux_tc 偏置——这是 HF 对拍约束（前向必须逐位一致）带来的取舍；
  代价是路由均衡弱一点、收益是导出/推理链路完全一致。
