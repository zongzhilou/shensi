# 集群跑法与三类真跑

本机是单卡 16GB（RTX 5080 Laptop），没有集群：没有 slurm/pbs 调度器、没有 ssh 目标。所以这个
目录干两件事：

1. **本机能跑的那部分真跑**——220M 机制曲线上的主表 A/B、多 seed、不同上下文长度的检索曲线，
   现成命令，产出带数字的 JSON；
2. **集群那部分的提交件**——三类真跑写成 sbatch，RULER 长上下文套件连同提交，到集群上一条命令
   交出去。

| 脚本 | 干什么 | 本机 | 集群 |
|---|---|---|---|
| `b5_mechanism_ab.py` | 主表 A/B 与多 seed（同数据顺序、同种子、固定步数） | 可跑（220M 档） | `--geom geoms/qwen3_1p04b`（1.04B）/ 0.6B / 规模阶梯 |
| `b5_longctx.py` | 上下文长度曲线：长度 L 的上下文里放 K 个位置已知的键，逐长度档测准确率，并给随机基线与位置偏差 | 可跑（最长 4K 上下文） | `--lengths …,131072 --n 200` |
| `b5_ruler.sh` | 公开的长上下文基准套件 RULER（13 个子任务 × 各长度档） | `--dry-run` 只校验与打印命令 | 真跑 |
| `submit_slurm.sh` | 把上面三类真跑写成 sbatch 提交 | `--dry-run` 打印 sbatch | `sbatch` |

## 本机怎么跑（现在就能跑）

```bash
export SHENSI_ROOT=<仓库根> SHENSI_FS=<存储根>
R=src/shensi/recipes/paper/gated_delta_attn_res

# ① 主表：主行 vs plain 残差（可加对照臂）
python $R/common/cluster/b5_mechanism_ab.py --steps 80 \
  --arms qwen3_gdar_main,base --seeds 42 --set train.model.eval_iters=0

# ② 多 seed：给误差棒，判断臂间之差是否大过种子噪声
python $R/common/cluster/b5_mechanism_ab.py --steps 300 --geom debug \
  --arms qwen3_gdar_main,qwen3_gdar_noladder --seeds 42,43,44 \
  --out $SHENSI_FS/shensi/runs/gated_delta_attn_res/b5_seeds.json

# ③ 上下文长度曲线（HF 目录来自 export_hf）
python $R/common/cluster/b5_longctx.py --model <HF 目录> --lengths 512,1024,2048,4096 --n 40
```

## 集群怎么交

```bash
bash $R/common/cluster/submit_slurm.sh --dry-run            # 先看要交什么（脚本落在 <WORK>/sbatch/）
bash $R/common/cluster/submit_slurm.sh --partition gpu --nodes 4 --model <HF 目录>
```

## 昇腾（Ascend）集群怎么交

同一套 `submit_slurm.sh` / `b5_ruler.sh` 在昇腾集群上照用，改三处设备面：

| 项 | CUDA 集群 | 昇腾集群 |
|---|---|---|
| 可见设备 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES`（配套 `ASCEND_HOME_PATH` 由 `set_env.sh` 设好） |
| 集合通信 | NCCL | HCCL（`HCCL_*` 环境，多机时按集群的 rank table 配） |
| 并行与算子 | TE + apex 融合（按 `perf` 档） | MindSpeed 承接并行与算子；`MegatronAdaptor + TransformerEngineNPU` 提供 mcore/TE 的 NPU 实现；融合算子走 MindSpeed-Ops（`torch.ops.mindspeed_ops.*`，AscendC/Triton-Ascend 按芯片自动分发） |

上机前的装配检查：`python -m shensi.utils.ascend_env`（CANN、torch↔torch_npu 配对、设备、组件
import、已知差异逐项报）。依赖清单在仓库根 `pyproject.ascend.toml`；容器化与多机拉起的做法与
常规昇腾集群一致（CANN 基础镜像 + 仓库内 3rdparty/ascend 组件按各自 README 安装）。

**注意**：本机没有 NPU，昇腾路径按组件文档整理、`--dry-run` 可校验命令面，**未上 NPU 实测**；
读数请以在集群上真跑出来的为准。

## 三类真跑的门槛与口径

| 类 | 规模 | 预算 | 判据 |
|---|---|---|---|
| 主表 | 0.6B 对比档 / 1.04B 机制曲线上端 | 每臂 20B–50B tokens（见各几何档注释） | 末段 loss 与曲线；四个对照臂同数据、同种子 |
| 多 seed | 同上 | 至少 3 个种子 | 臂间差与种子标准差之比 |
| 长上下文 | 发布档（128K） | RULER 全 13 子任务 × 各长度档 | RULER 分数 + 长度曲线 |

**注意**：本机的语料是 sample 级（约 120K tokens），跑出来的是"机制级"对照——同一份数据顺序、
同一种子、只换算法，比较的是优化行为，不是数据质量结论。集群上把语料换成完整训练集（各段
`config/data_prep/*.json` 已写好清单），同一套命令即论文口径。
