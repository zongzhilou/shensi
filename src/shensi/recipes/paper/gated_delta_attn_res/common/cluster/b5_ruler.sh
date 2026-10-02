#!/usr/bin/env bash
# B5(c) 集群侧：RULER（长上下文官方套件）——各长度档上的 13 个子任务。
#
# 本机跑不了（要 32K/128K 推理、要下 RULER 的语料，而且这台机器**没有外网**：clone 会失败）。
# 所以这份脚本的作用是"把到集群上按什么步骤跑写死"，并带 --dry-run（本机可跑：校验前置件、
# 打印计划）。RULER 的 CLI 旗标是按其 README 的常见写法给的默认值，**上集群后请按所钉 commit
# 的 README 核对一遍**（可以整段覆盖：RULER_PREPARE_ARGS / RULER_RUN_ARGS / RULER_EVAL_ARGS）。
#
#   bash cluster/b5_ruler.sh --dry-run                     # 本机：校验 + 打印计划
#   bash cluster/b5_ruler.sh --model /path/to/hf --work /path/to/work
set -euo pipefail

RULER_REPO="${RULER_REPO:-https://github.com/NVIDIA/RULER.git}"
RULER_COMMIT="${RULER_COMMIT:-main}"           # 上集群时换成固定 commit（可复现）
RECIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=""
WORK="${SHENSI_FS:-/root/work/filestorage}/shensi/runs/gated_delta_attn_res/b5/ruler"
LENGTHS="${LENGTHS:-4096,8192,16384,32768,131072}"
DRY=0

# 默认旗标（按上游脚本习惯给的；不一致就整段覆盖，见文件头注释）
RULER_PREPARE_ARGS="${RULER_PREPARE_ARGS:---tok_path <TOKENIZER> --max_seq_length <L>}"
RULER_RUN_ARGS="${RULER_RUN_ARGS:---model_template qwen3 --max_seq_length <L>}"
RULER_EVAL_ARGS="${RULER_EVAL_ARGS:---benchmark synthetic}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

TOKENIZER="$RECIPE/tokenizer/Qwen3-0.6B"
MODEL_SHOWN="${MODEL:-<HF 目录>}"
echo "[ruler] 仓库 $RULER_REPO（commit $RULER_COMMIT）"
echo "[ruler] 工作目录 $WORK"
echo "[ruler] tokenizer $TOKENIZER（长度档 $LENGTHS）"
echo "[ruler] 注意：RULER 的 CLI 旗标请按该 commit 的 README 核对（本机无外网，没法验证）"

missing=0
[[ -d "$TOKENIZER" ]] || { echo "  ✗ 缺 tokenizer：$TOKENIZER"; missing=1; }
if [[ -n "$MODEL" ]]; then
  [[ -f "$MODEL/config.json" ]] || { echo "  ✗ 缺模型 config.json：$MODEL"; missing=1; }
  [[ -f "$MODEL/tokenizer.json" ]] || echo "  ! 模型目录里没有 tokenizer.json（RULER 会用 --tok_path 那份）"
else
  [[ "$DRY" == "1" ]] || { echo "  ✗ 真跑必须给 --model" ; missing=1; }
fi
[[ $missing == 0 ]] || { echo "[ruler] 前置件不全，先补齐" >&2; exit 1; }

cat <<EOF
1) clone + 钉 commit
   git clone --depth 1 $RULER_REPO "$WORK/RULER" && git -C "$WORK/RULER" checkout $RULER_COMMIT
2) 每个长度档生成合成语料（tokenizer 用本配方那份）
   cd "$WORK/RULER/scripts"
   for L in ${LENGTHS//,/ }; do
     python prepare.py --save_dir "$WORK/data/\$L" ${RULER_PREPARE_ARGS//<TOKENIZER>/"$TOKENIZER"}
   done
3) 推理（HF 目录 + 本配方自带 auto_map，trust_remote_code 加载）
   for L in ${LENGTHS//,/ }; do
     python run.py --model_path "$MODEL_SHOWN" --data_dir "$WORK/data/\$L" \\
       --save_dir "$WORK/outputs/\$L" ${RULER_RUN_ARGS//<L>/\$L}
   done
4) 评分（13 个子任务 → 一张表）
   for L in ${LENGTHS//,/ }; do
     python eval/evaluate.py --data_dir "$WORK/outputs/\$L" --save_dir "$WORK/scores/\$L" $RULER_EVAL_ARGS
   done
EOF

if [[ "$DRY" == "1" ]]; then
  echo "[ruler] dry-run：计划已打印（前置件 OK）。真跑去掉 --dry-run 并给 --model。"
  exit 0
fi

mkdir -p "$WORK"
git clone --depth 1 "$RULER_REPO" "$WORK/RULER"
git -C "$WORK/RULER" checkout "$RULER_COMMIT"
for L in ${LENGTHS//,/ }; do
  mkdir -p "$WORK/data/$L" "$WORK/outputs/$L" "$WORK/scores/$L"
  ( cd "$WORK/RULER/scripts" && python prepare.py --save_dir "$WORK/data/$L" \
      ${RULER_PREPARE_ARGS//<TOKENIZER>/"$TOKENIZER"} ${RULER_PREPARE_ARGS//<L>/$L} )
  ( cd "$WORK/RULER/scripts" && python run.py --model_path "$MODEL" \
      --data_dir "$WORK/data/$L" --save_dir "$WORK/outputs/$L" ${RULER_RUN_ARGS//<L>/$L} )
  ( cd "$WORK/RULER/scripts" && python eval/evaluate.py --data_dir "$WORK/outputs/$L" \
      --save_dir "$WORK/scores/$L" $RULER_EVAL_ARGS )
  echo "[ruler] 长度 $L 完成 → $WORK/scores/$L"
done
echo "[ruler] 全部长度完成；把 \$WORK/scores/*/ 的 13 个子任务分数汇总进 LIMITATIONS 的 B5 行"
