# stage0_pretrain：预训练（stage1）+ 中训练（stage2）

Base training 采用**逐级推进**的两段式（stable → decay）建立基础语言能力与训练稳定性；
随后中训练（mid-training）两段强化目标能力并适配数据分布。语料全部来自同步开源的
Ultra-FineWeb、Ultra-FineWeb-L3、UltraX、UltraData-Code 与 UltraData-Math
（配比 json 在各 stage 的 `config/data_prep/`）。

> 预算与超参以 **0.6B 主对比档**为基准（base 预算 10B tokens，RECIPE.md §3）；
> 1.7B/4B/8B 档按同比例缩放，见根 README 的规模阶梯。

## 段位设计（为什么是 2+2）

| 段 | stage/profile | tokens | seq | LR | 语料 | 目的 |
|---|---|---|---|---|---|---|
| PT-1 stable | `stage1_pretrain`（default） | 9B（90%） | 2048 | 6e-4 **恒定**（WSD stable），warmup 2% | Ultra-FineWeb(en+zh) 85% + Code 10% + Math 5% | 基础语言能力；恒定 LR 是 WSD 的 stable 段，稳定性由 warmup+clip 保证 |
| PT-2 decay | `stage1_pretrain --profile decay` | 1B（10%） | 2048 | cosine 6e-4 → 6e-5（10% 峰值） | ≥50% 切高质量：UltraX 30% + Ultra-FineWeb-L3 40% + Code 15% + Math 15% | 收敛收尾：高质量子集上的退火（Nemotron/GLM 的 decay 口径） |
| Mid-1 能力强化 | `stage2_midtrain`（default） | 0.5B（5%） | 4096 | 6e-5 **恒定**（10% 峰值），warmup 1% | Code 40% + Math 30% + UltraX 30% | 升长度 + 目标能力（代码/数学）强化 |
| Mid-2 分布适配 | `stage2_midtrain --profile mid2` | 0.3B（3%） | 16384 | cosine 6e-5 → 3e-5（末段 5% 峰值） | Ultra-FineWeb-L3 长文档 70% + UltraX 30% | 长文档分布适配，为 SFT 的长序列打底 |

设计依据与取舍：

- **预训练分 2 段（stable+decay）**：WSD 两段式是"逐级推进"的最小实现——stable 段恒定 LR
  保证架构对比的稳定性读数干净，decay 段切高质量子集退火。所有模型算法（`--model-algo`）
  跑**同一份**两段配方，对比才干净。
- **中训练分 2 段**：Mid-1 先"能力强化"（代码/数学占比拉满、长度先升到 4K），
  Mid-2 再"分布适配"（Ultra-FineWeb-L3 长文档占 70%、长度 16K、LR 末段再衰减）——
  能力与分布分两步走，避免一步同时动长度+LR+数据三个变量。
- **LR 峰值 6e-4** 是 0.6B 档的先验（RECIPE.md §3），正式跑前先用
  {3e-4, 6e-4, 1e-3} 各 ~2000 步 pilot 校准（评审 1 质疑点）。
- **接续关系**：PT-2 从 PT-1 的 ckpt 起（`--load`），Mid-1 从 PT-2 起，Mid-2 从 Mid-1 起；
  每段 eval 间隔内用 `early_stop.py` 看门狗兜底。

## 跑法（从零到中训练结束）

```bash
cd stage0_pretrain/stage1_pretrain
python data_prep.py --prepare                       # stable 混合 → bin/idx（Qwen3 tokenizer）
python train.py --tokens 9e9                        # PT-1 stable
python data_prep.py --prepare --blend decay.json    # decay 高质量混合
python train.py --profile decay --tokens 1e9 \
    --load $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_pretrain/default   # PT-2

cd ../stage2_midtrain
python data_prep.py --prepare                       # Mid-1 混合
python train.py --tokens 0.5e9 --load <PT-2 ckpt>   # Mid-1
python data_prep.py --prepare --blend mid2.json     # Mid-2 长文档混合
python train.py --profile mid2 --tokens 0.3e9 --load <Mid-1 ckpt>
```

`--model-algo` 在每个 stage 都可用（默认 `qwen3_gdar_paper`；对照臂换
`--model-algo base / qwen3_ar / qwen3_dar / …`，见根 README 的注册表）。
