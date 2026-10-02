# 集群跑法（B5 的三类真跑）

本机是单卡 16GB（RTX 5080 Laptop），**没有集群**：没有 slurm/pbs 调度器、没有 ssh 目标。
所以这个目录干两件事：

1. **本机能跑的那部分真跑**（220M 机制曲线上的主表 A/B、多 seed、受控检索多长度）——
   现成命令，产出带数字的 JSON；
2. **集群那部分的提交件**（三类真跑写成 sbatch + RULER 套件），到集群上一条命令交出去。

| 脚本 | 干什么 | 本机 | 集群 |
|---|---|---|---|
| `b5_mechanism_ab.py` | (a) 主表 A/B、(b) 多 seed（同数据顺序、同种子、固定步数） | ✓ 220M 档 | `--geom geoms/qwen3_1p04b`（1.04B）/ 0.6B / 规模阶梯 |
| `b5_longctx.py` | (c) 受控深度检索的长度曲线（可判分，带 Wilson95 与位置偏差） | ✓ 最长 4K | `--lengths …,131072 --n 200` |
| `b5_ruler.sh` | (c) RULER 官方套件（13 子任务 × 各长度） | `--dry-run` 只校验/打印 | 真跑 |
| `submit_slurm.sh` | 把上面三件写成 sbatch 交出去 | `--dry-run` 打印 sbatch | `sbatch` |

## 本机怎么跑（现在就能跑）

```bash
export SHENSI_ROOT=<仓库根> SHENSI_FS=<存储根>
R=src/shensi/recipes/paper/gated_delta_attn_res

# (a) 主表：主行 vs plain Qwen3（+ 对照臂）
python $R/cluster/b5_mechanism_ab.py --steps 80 \
  --arms qwen3_gdar_main,base --seeds 42 --set train.model.eval_iters=0

# (b) 多 seed：给误差棒（判断臂间差是否过种子噪声）
python $R/cluster/b5_mechanism_ab.py --steps 300 --geom debug \
  --arms qwen3_gdar_main,qwen3_gdar_noladder --seeds 42,43,44 \
  --out $SHENSI_FS/shensi/runs/gated_delta_attn_res/b5_seeds.json

# (c) 长上下文的受控检索曲线（HF 目录来自 train/export_hf.py）
python $R/cluster/b5_longctx.py --model <HF 目录> --lengths 512,1024,2048,4096 --n 40
```

## 集群怎么交

```bash
bash cluster/submit_slurm.sh --dry-run            # 先看要交什么（脚本落在 <WORK>/sbatch/）
bash cluster/submit_slurm.sh --partition gpu --nodes 4 --model <HF 目录>
```

三类真跑的门槛与口径：

| 类 | 规模 | 预算（论文口径） | 判据 |
|---|---|---|---|
| 主表 | 0.6B 对比档 / 1.04B 机制曲线上端 | 每臂 20B–50B tokens（见各 geom 档注释） | 末段 loss、曲线；与四个对照臂同数据同种子 |
| 多 seed | 同上 | ≥3 seed | 臂间 Δ 与 seed σ 比 |
| 长上下文 | 发布档（128K） | RULER 全 13 子任务 × 长度档 | RULER 分数 + 受控检索的 length→acc 曲线 |

**注意**：本机的语料是 sample 级（约 120K tokens），跑出来的曲线是"机制级"对照——同一份数据
顺序、同一种子、只换算法，比较的是优化行为，不是数据质量结论。集群上把语料换成
Ultra-FineWeb / UltraX / UltraData 全套（各 stage 的 `config/data_prep/*.json` 已经写好清单），
同一套命令即论文口径。
