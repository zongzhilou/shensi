#!/usr/bin/env bash
# B5 的集群提交入口：三类真跑（主表 / 多 seed 消融 / RULER 长上下文）一次交出去。
#
#   bash cluster/submit_slurm.sh --dry-run                 # 本机：打印 sbatch 脚本（可核查）
#   bash cluster/submit_slurm.sh --partition gpu --nodes 4 --partition-time 2-00:00:00
#
# 说明：本机没有集群（无 slurm/pbs 调度器、无 ssh 目标），所以这份脚本按"到集群上怎么交"
# 写死；--dry-run 会把要交的 sbatch 脚本落到 <WORK>/sbatch/ 下，交之前先看一眼。
#
# 三类真跑对应的命令：
#   主表      cluster/b5_mechanism_ab.py --geom geoms/qwen3_1p04b --arms <臂表> --seeds 42,43,44
#   多 seed   cluster/b5_mechanism_ab.py --geom geoms/qwen3_0p22b --arms <主行>,<消融行> --seeds 42,43,44,44+1,44+2
#   长上下文  cluster/b5_longctx.py --model <HF 目录> --lengths 8192,16384,32768 （+ cluster/b5_ruler.sh）
set -euo pipefail

RECIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RECIPE/../../../../.." && pwd)"          # src/shensi/recipes/paper/<配方> → 仓库根
WORK="${SHENSI_FS:-/root/work/filestorage}/shensi/runs/gated_delta_attn_res/b5"
PARTITION="${PARTITION:-gpu}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TIME="${TIME:-2-00:00:00}"
ARMS_MAIN="${ARMS_MAIN:-base,qwen3_ar_block4,qwen3_dar_block4,qwen3_gdar_main}"
ARMS_SEED="${ARMS_SEED:-qwen3_gdar_main,qwen3_gdar_noladder}"
SEEDS="${SEEDS:-42,43,44}"
GEOM_MAIN="${GEOM_MAIN:-geoms/qwen3_1p04b}"
GEOM_SEED="${GEOM_SEED:-geoms/qwen3_0p22b}"
MODEL="${MODEL:-}"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORK/sbatch"
PY="${PY:-$REPO/.venv/bin/python}"

write_job() {
  local name="$1"; shift
  local body="$1"; shift
  cat > "$WORK/sbatch/${name}.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=gdar-${name}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --time=${TIME}
#SBATCH --output=${WORK}/sbatch/${name}-%j.log
set -euo pipefail
export SHENSI_ROOT="$REPO"
export SHENSI_FS="${SHENSI_FS:-/root/work/filestorage}"
cd "$REPO"
${body}
EOF
  echo "  写好 $WORK/sbatch/${name}.sbatch"
}

echo "[b5] 仓库 $REPO｜工作目录 $WORK｜分区 $PARTITION｜节点 $NODES×${GPUS_PER_NODE} GPU"
echo "[b5] 三类真跑："
echo "  (a) 主表：${GEOM_MAIN} × 臂 [$ARMS_MAIN]"
echo "  (b) 多 seed：${GEOM_SEED} × 臂 [$ARMS_SEED] × 种子 [$SEEDS]"
echo "  (c) 长上下文：受控检索多长度 + RULER（$([[ -n $MODEL ]] && echo "$MODEL" || echo '待给 --model'))"

write_job "b5-main" "$PY $RECIPE/cluster/b5_mechanism_ab.py --geom $GEOM_MAIN --arms $ARMS_MAIN --seeds 42 --steps 2000 --out $WORK/b5_main.json"
write_job "b5-seeds" "$PY $RECIPE/cluster/b5_mechanism_ab.py --geom $GEOM_SEED --arms $ARMS_SEED --seeds $SEEDS --steps 2000 --out $WORK/b5_seeds.json"
if [[ -n "$MODEL" ]]; then
  write_job "b5-longctx" "$PY $RECIPE/cluster/b5_longctx.py --model $MODEL --lengths 8192,16384,32768,131072 --n 200 --out $WORK/b5_longctx.json"
  write_job "b5-ruler" "bash $RECIPE/cluster/b5_ruler.sh --model $MODEL --work $WORK/ruler"
else
  echo "  ! 没给 --model：长上下文那两个 job 不写（先 export_hf 出 HF 目录再交）"
fi

if [[ "$DRY" == "1" ]]; then
  echo "[b5] dry-run：sbatch 脚本已落盘，交之前先看一遍；真跑用 sbatch $WORK/sbatch/b5-main.sbatch"
  exit 0
fi
for f in "$WORK"/sbatch/*.sbatch; do
  sbatch "$f"
done
