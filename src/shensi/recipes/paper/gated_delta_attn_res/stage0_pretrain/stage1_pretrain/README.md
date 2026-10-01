# stage1_pretrain：预训练（PT-1 stable → PT-2 decay）

逐级推进的两段式：PT-1 恒定 LR 建基础语言能力与稳定性（WSD 的 stable），PT-2 切高质量
子集做退火收尾（WSD 的 decay）。段位设计、预算与 LR 的完整依据见上一级
[`stage0_pretrain/README.md`](../README.md)；语料全部来自同步开源的 Ultra-FineWeb /
Ultra-FineWeb-L3 / UltraX / UltraData-Code / UltraData-Math。


> **早停默认开**：本 stage 步数/轮次给到无限大，收敛与收尾交给看门狗（metric=`lm loss value`，
> patience 见 `config/default.yaml` 的 `early_stop` 段；超耐心 SIGTERM 收尾、按成功返回；
> `--no-early-stop` 可关；改耐心用 `--early-stop N`）。

## 档位

| 档 | 是什么 |
|---|---|
| `default` | PT-1 stable：9B tokens(90%) / seq 2048 / LR 6e-4 **恒定** / warmup 2% / 通用混合 |
| `decay` | PT-2 decay：1B tokens(10%) / seq 2048 / cosine 6e-4 → 6e-5 / decay.json 高质量混合 |
| `debug` | 极小几何（4L/256h/seq512）+ 真实 bin/idx，链路验证用 |
| `tiny` | mock 冒烟档（`--smoke` 用） |
| `ablations/a1a_*` … `ablations/e3_*` | 设计消融与门结构消融的对照行（自带 spec，见根 README 消融矩阵） |
| `ar` / `dar` / `gdar` | 连接模块对照臂的历史档（等价于 `--model-algo qwen3_ar / qwen3_dar / qwen3_gdar_paper`） |

## 模型算法（--model-algo）

默认 `qwen3_gdar_paper`（论文主行）；`--model-algo base` = plain Qwen3 对照臂。全部可选
值见根 README 的注册表。优先级：`--set train.model.spec=...` > `--model-algo` >
profile 自带 spec > 默认算法。

## 跑法

```bash
python data_prep.py --prepare                        # PT-1 混合 → bin/idx（Qwen3 tokenizer）
python train.py --tokens 9e9                         # PT-1 stable
python data_prep.py --prepare --blend decay.json     # PT-2 高质量混合
python train.py --profile decay --tokens 1e9 \
    --load <PT-1 ckpt>                               # PT-2 decay
python train.py --model-algo base --tokens 9e9       # 对照臂（同配方，只换算法）
```

## 判据

| 检查 | 命令 | 判据 |
|---|---|---|
| 集成测试 | `python test_train.py` | tiny 5 步：rc=0、到最后一 iter、`[after training is done]`、无 Traceback |
| 离线校验 | `python -m ...train.checks` | 恒等 27/27 逐位、logits `0.000e+00`、参数开销表、梯度流 |
| 冒烟 | `python train.py --smoke` | 同集成测试判据（mock，不碰语料） |

吞吐：默认 `transformer_impl: local`（逐位恒等验收的基线）；集群主跑切
`--set train.model.transformer_impl=transformer_engine`（TE 融合注意力/MLP，已验证可跑）。
`--spec` 的实现是 local 子模块时 GDAR ≈ 2.19× base 时间（未融合内核）；连接算子本身的
融合内核是后续吞吐项。集群主跑的数字是占位——本机承担链路验证与 0.6B 档的 pilot。

---

## 跑完整论文实验（EXPERIMENT_MATRIX.md §2 / RUN_EXPERIMENTS.md）

架构结论只由 PT 承担：全部变体 × 阶梯规模，主表 0.6B 三 seed、其余单 seed。
每条命令都是 `train.py --model-algo <臂>`——**臂名即设计矩阵的一行**（main ± 一个旋钮）。

```bash
cd stage0_pretrain/stage1_pretrain

# 0) 语料（stable + decay 两套配比）
python data_prep.py --prepare
python data_prep.py --prepare --blend decay.json

# 1) 0.6B 核心矩阵（3 seed）。臂清单 = 设计矩阵的变体 × 设置：
ARMS="base \
 qwen3_ar_block4 qwen3_dar_block4 \
 qwen3_gdar_main qwen3_gdar_main_sublayer qwen3_gdar_main_b2 qwen3_gdar_main_b8 qwen3_gdar_main_b16 \
 qwen3_gdar_main_rank16 qwen3_gdar_main_rankfull \
 qwen3_gdar_main_gates_d qwen3_gdar_main_gates_e qwen3_gdar_main_gates_w \
 qwen3_gdar_main_gates_de qwen3_gdar_main_gates_dw qwen3_gdar_main_gates_ew \
 qwen3_gdar_main_gates_scalar qwen3_gdar_main_gates_none \
 qwen3_gdar_main_init_paper qwen3_gdar_main_init_uniform qwen3_gdar_main_init_half \
 qwen3_gdar_main_gate_prefix qwen3_gdar_main_gate_delta \
 qwen3_gdar_main_update_reference \
 qwen3_gdar_main_address_state qwen3_gdar_main_address_novelty \
 qwen3_gdar_main_lambda_free qwen3_gdar_main_decay_free qwen3_gdar_main_ladder0 \
 qwen3_gdar_main_heads1 qwen3_gdar_main_null_off qwen3_gdar_main_whiten_diag qwen3_gdar_main_whiten_off \
 qwen3_gdar_main_mix_whitened qwen3_gdar_main_no_output_route qwen3_gdar_main_carrier0 \
 qwen3_dar_null_source \
 qwen3_hc qwen3_hc_published qwen3_mhc qwen3_mhc_published qwen3_mhc_lite \
 qwen3_mudd qwen3_mudd_random qwen3_denseformer qwen3_denseformer_published qwen3_denseformer_period4"

for seed in 0 1 2; do
  for algo in $ARMS; do
    RUN=$SHENSI_FS/shensi/runs/gdar_pt06b/s${seed}/$algo
    python train.py --model-algo $algo --tokens 9e9 \
        --set train.model.seed=$seed --set experiment.exp_dir=$RUN          # PT-1 stable
    python train.py --profile decay --model-algo $algo --tokens 1e9 \
        --set train.model.seed=$seed --set experiment.exp_dir=$RUN-decay \
        --load $RUN/ckpt                                                    # PT-2 decay
  done
done

# 2) 阶梯规模（单 seed）：1.7B / 4B / 8B / 14B（几何档在 config/geoms/）
for g in qwen3_1p7b qwen3_4b qwen3_8b qwen3_14b; do
  for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main \
              qwen3_hc qwen3_mhc qwen3_mudd qwen3_denseformer; do
    python train.py --profile geoms/$g --model-algo $algo --tokens $(case $g in
        qwen3_1p7b) echo 2e10;; qwen3_4b) echo 5e10;; qwen3_8b) echo 1e11;; *) echo 15e10;; esac) \
        --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_$g/$algo
  done
done

# 3) 30B-A3B 门面（四行，只做工程链路与后训练载体）
for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  python train.py --profile geoms/qwen3_30b_a3b --model-algo $algo \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_30b/$algo
done

# 4) 机制曲线专用小档（220M / 1.04B，与 DAR 论文规模对齐；几何按论文数值用 --set 给）
#    python train.py --model-algo qwen3_gdar_main --tokens 2e9 \
#        --set train.model.num_layers=<L> --set train.model.hidden_size=<H> ... \
#        --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_220m_$algo

# 5) 每条臂结束后的评测（谁跑什么见 RUN_EXPERIMENTS.md §2）
#    PPL：grep "on validation set" <run>/logs/host_0_localhost.output
#    0-shot / 中文 / 受控检索 / RULER(≥8B)：见 stage4_eval 与本 README 的「判据」表
#    机制诊断与效率：train/checks.py [4]（参数开销）+ 吞吐/峰值显存按 REVIEW_COMPLIANCE 的 R1-6 口径
```

判定与合规门（每个数字发布前）：PPL 落有意义区间（R1-9）；检索必须带 chance=25% + Wilson95 +
分层 + 位置偏差；效率的分母是 **AR**（不是 GDAR-Full）并标注"无融合内核"；0.6B 之外单 seed 需标注。
