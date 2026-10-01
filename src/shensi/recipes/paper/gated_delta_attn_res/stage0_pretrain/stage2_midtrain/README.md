# stage2_midtrain：中训练（Mid-1 能力强化 → Mid-2 长文档）

从 PT-2 的 ckpt 接续，两步走：Mid-1 把代码/数学占比拉满、长度升到 4K（能力强化）；
Mid-2 切 Ultra-FineWeb-L3 长文档、长度 16K、LR 末段再衰减（分布适配）。两段的具体
预算/LR/语料见上一级 [`stage0_pretrain/README.md`](../README.md)。


> **早停默认开**：本 stage 步数/轮次给到无限大，收敛与收尾交给看门狗（metric=`lm loss value`，
> patience 见 `config/default.yaml` 的 `early_stop` 段；超耐心 SIGTERM 收尾、按成功返回；
> `--no-early-stop` 可关；改耐心用 `--early-stop N`）。

## 档位

| 档 | 是什么 |
|---|---|
| `default` | Mid-1：0.5B tokens(5%) / seq 4096 / LR 6e-5 恒定 / Code 40% + Math 30% + UltraX 30% |
| `mid2` | Mid-2：0.3B tokens(3%) / seq 16384 / cosine 6e-5 → 3e-5 / L3 70% + UltraX 30% |
| `debug` | 极小几何（seq 512）+ 真实 bin/idx，链路验证用 |
| `tiny` | mock 冒烟档 |

## 跑法

```bash
python data_prep.py --prepare                        # Mid-1 混合
python train.py --tokens 5e8 --load <PT-2 ckpt>
python data_prep.py --prepare --blend mid2.json      # Mid-2 长文档混合
python train.py --profile mid2 --tokens 3e8 --load <Mid-1 ckpt>
```

`--model-algo` 与判据同 stage1_pretrain（集成测试走 `python test_train.py`）。

---

## 跑完整论文实验（EXPERIMENT_MATRIX.md §3）

Mid 只服务 **8B 四变体** 与 **30B 门面**：同一套评测口径（RUN_EXPERIMENTS.md §3），
序列加长后 RULER 的 16K/32K 档才有意义。

```bash
cd stage0_pretrain/stage2_midtrain

# Mid-1（能力强化，seq 4096，5% tokens，LR 10% peak 恒定）→ Mid-2（长文档，seq 16K→32K，3%，末段衰到 5%）
python data_prep.py --prepare                     # Mid-1 混合（含 UltraData-Code L2/L3 分级）
python data_prep.py --prepare --blend mid2.json   # Mid-2 混合（L3 长文档）

for algo in base qwen3_ar_block4 qwen3_dar_block4 qwen3_gdar_main; do
  # 8B
  python train.py --profile geoms/qwen3_8b --model-algo $algo --tokens 5e9 \
      --load <PT-2(8B, $algo) ckpt> --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo
  python train.py --profile geoms/qwen3_8b --model-algo $algo --tokens 3e9 \
      --set train.model.seq_length=16384 \
      --load $SHENSI_FS/shensi/runs/gdar_8b_mid1/$algo/ckpt \
      --set experiment.exp_dir=$SHENSI_FS/shensi/runs/gdar_8b_mid2/$algo
  # 30B 门面同法（--profile geoms/qwen3_30b_a3b），Mid-2 的 32K 档：--set train.model.seq_length=32768
done

# 评测：与 PT 同一套（见 stage1_pretrain README 的评测清单），RULER 加 16K/32K
```

`geoms/*` 档与 PT 共用（同一份几何）；Mid-2 的 32K 档直接 `--set train.model.seq_length=32768`。

---

## 维护记录

* **2026-10-01 偶发 CUDA illegal memory access（未复现）**：一次 `test_train.py`（tiny 几何 L=8、
  block B=4、5 步）在第 2 步反向报 `CUDA error: an illegal memory access`；随后 5 次复跑
  （同一 run、以及 B=4/B=2/B=1 三形态各 3 步）全部干净收尾。按"未复现的偶发"记录，不当作已修复。
  再遇到时的定位方式：`CUDA_LAUNCH_BLOCKING=1` 跑同样的 2–3 步；仍在则上 compute-sanitizer。
  ```bash
  cd stage0_pretrain/stage2_midtrain
  CUDA_LAUNCH_BLOCKING=1 python train.py --profile debug \
      --set train.model.train_iters=2 --set train.model.eval_iters=0
  ```
