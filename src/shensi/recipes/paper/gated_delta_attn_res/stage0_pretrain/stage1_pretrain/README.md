# 阶段 0.1：预训练（PT-1 stable → PT-2 decay）

两段式预训练：PT-1 保持学习率恒定来建立语言能力并给出干净的稳定性读数（WSD 的 *stable* 段）；
PT-2 在高质量子集上退火（*decay* 段），从 PT-1 的 ckpt 接续。所有模型算法（`--model-algo`）
跑的都是这一份配方 —— 这正是架构对比干净的前提。

## 总览

| 组件 | 说明 |
|---|---|
| `data_prep.py` | 混合 Ultra 语料并 tokenize 成 Megatron bin/idx |
| `train.py` | Megatron-Core 预训练；`--model-algo` 选连接（默认 GDAR 论文主行） |
| `test_train.py` | 集成测试：tiny 几何跑 5 步，按日志判 PASS/FAIL |
| `config/` | 档位配置（`default`、`decay`、`debug`、`tiny`、`perf`、`geoms/*`、`ablations/*`） |

> **早停默认开**：`train_iters` 给到无限大，loss 平台后由看门狗（metric `lm loss value`）收尾。
> 早停按**成功**处理；`--no-early-stop` 可关，`--early-stop N` 调耐心。

## 档位

| 档 | 是什么 |
|---|---|
| `default` | PT-1 stable：9B tokens(90%) / seq 2048 / LR 6e-4 **恒定** / warmup 2% / 通用混合；**0.6B 的几何就在这一档里**（28 层 / 1024 hidden） |
| `decay` | PT-2 decay：1B tokens(10%) / seq 2048 / cosine 6e-4 → 6e-5 / `decay.json` 高质量混合 |
| `debug` | 极小几何（4 层 / 256 hidden / seq 512）+ 真实 bin/idx，链路验证用 |
| `tiny` | mock 冒烟档（`--smoke` 用） |
| `perf` | 吞吐档：TransformerEngine 骨干 + 三个实测可用的融合 |
| `geoms/qwen3_{1p7b,4b,8b,14b,30b_a3b}.yaml` | 规模阶梯。**跨 stage 共享**：从 Mid / SFT 也能 `--profile geoms/*`（只维护这一份） |
| `ablations/a1a_*` … `ablations/e3_*` | 设计矩阵与门结构消融行（自带 spec） |
| `minicpm5_2b.yaml` | 发布版 MiniCPM5-2B 几何（逐字段反推，对齐用） |

## 数据准备

```bash
python data_prep.py --prepare                        # default 混合
python data_prep.py --prepare --blend decay.json     # PT-2 混合
```

配比文件在 `config/data_prep/`。产物是 Megatron bin/idx 加一份 `blend.json`（各 split 的路径），
落在 `$SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_pretrain/`；`train.py` 会自动找到它。

## 训练

```bash
python data_prep.py --prepare
python train.py --tokens 9e9                          # PT-1（tokens → train_iters）
python data_prep.py --prepare --blend decay.json
python train.py --profile decay --tokens 1e9 --load <PT-1 ckpt>   # PT-2
python train.py --model-algo base --tokens 9e9        # 对照臂：同配方、换模型
```

| 参数 | 说明 |
|---|---|
| `--profile <name>` | 档位（默认 `default`；另有 `decay`、`debug`、`perf`、`geoms/*`、`ablations/*`） |
| `--model-algo <name>` | 连接 / 基线（默认 `qwen3_gdar_paper`；`base` = plain Qwen3） |
| `--tokens <n>` | token 预算 —— 按 global batch × seq 换算成 `train_iters` |
| `--load <ckpt>` | 从某个 ckpt 接续（PT-2 或更后面的 stage） |
| `--smoke` / `--dry-run` | 合成 tiny 跑 / 只打印命令 |
| `--no-early-stop`、`--early-stop N` | 关看门狗 / 覆盖耐心 |

## 判据

| 检查 | 命令 | 判据 |
|---|---|---|
| 集成测试 | `python test_train.py` | tiny 5 步：rc=0、到最后一 iter、`[after training is done]`、无 Traceback |
| 离线校验 | `python -m shensi.recipes.paper.gated_delta_attn_res.train.checks` | 恒等 27/27 逐位、`max|Δ logit| = 0.000e+00`、参数开销表、梯度流 |
| 冒烟 | `python train.py --smoke` | 与集成测试同判据（mock 数据） |

吞吐：默认 `transformer_impl: local`（逐位恒等验收基线）；集群主跑切
`--set train.model.transformer_impl=transformer_engine`（已验证可跑：TE 注意力 / MLP + `perf.yaml`
里的三个融合）。local spec 下 GDAR 每步约是 plain Qwen3 的 2.19×（连接算子还没有融合内核）
—— 融合内核是独立工程项。

## 跑完整论文实验（EXPERIMENT_MATRIX.md §2）

架构结论只由预训练承担：全部变体 × 规模阶梯，0.6B 三 seed、其余单 seed。每条命令都是
`train.py --model-algo <臂>` —— 臂名即设计矩阵的一行。

```bash
cd stage0_pretrain/stage1_pretrain

# 0) 两套配比
python data_prep.py --prepare
python data_prep.py --prepare --blend decay.json

# 1) 0.6B 核心矩阵（3 seed）—— 变体 × 单旋钮行
ARMS="base qwen3_ar_block4 qwen3_dar_block4 qwen3_denseformer qwen3_mudd qwen3_hc qwen3_mhc \
      qwen3_gdar_paper qwen3_gdar_theory qwen3_gdar_upstream qwen3_gdar_block2 qwen3_gdar_block8 \
      qwen3_gdar_r16 qwen3_gdar_noladder qwen3_gdar_no_output_route a1a_gate_prefix a1b_gate_delta \
      a3_decay_projected a4_lambda_free a6_reference a9_half_init a9_uniform_init"
for seed in 0 1 2; do for algo in $ARMS; do
  # 0.6B 几何就是 `default`（28 层 / 1024 hidden）；decay 从它接续
  python train.py --model-algo $algo --tokens 1e10 \
      --set experiment.seed=$seed --set experiment.exp_dir=$SHENSI_FS/shensi/runs/pt06/$algo-s$seed
  python train.py --profile decay --model-algo $algo --tokens 1e9 \
      --load $SHENSI_FS/shensi/runs/pt06/$algo-s$seed/ckpt --set experiment.seed=$seed
done; done

# 2) 规模阶梯（单 seed）：1.7B / 4B / 8B / 14B + 30B-A3B 门面
for size in qwen3_1p7b qwen3_4b qwen3_8b qwen3_14b; do for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  python train.py --profile geoms/$size --model-algo $algo --tokens <预算>
done; done

# 3) 机制曲线：220M / 1.04B（控制深度与宽度两条曲线）
# 4) 评测：0-shot / 中文 / 受控深度检索 / RULER(≥8B) —— 见 stage4_eval/
```

## 延伸阅读

- [阶段 0 总览](../README.md) —— 2+2 段位设计与依据
- [中训练](../stage2_midtrain/README.md) —— 从 PT-2 接续的那一段
- [LIMITATIONS.md](../../LIMITATIONS.md) —— 早停证据、融合实测
