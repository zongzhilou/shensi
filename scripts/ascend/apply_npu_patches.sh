#!/usr/bin/env bash
# 把 NPU 补丁集打到 mcore（Megatron-LM）与 MindSpeed 子模块上。
#
#   bash scripts/ascend/apply_npu_patches.sh [/仓库根] [--check]
#
#   --check   只做 dry-run（不改任何文件），逐条报告 适用/已应用/不适用。
#
# 目标树可用环境变量覆盖（默认按 shensi-offline 部署布局）：
#   MCORE_DIR / MINDSPEED_DIR
#
# 幂等：补丁带 --forward，已应用则跳过；"--check" 与正式执行走同一判定。
# 补丁内容与来源见同目录 README.md（0001 是防 NaN 的必需项，失败必须中止）。
set -u

ROOT=""
CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) ROOT="$1" ;;
  esac
  shift
done
ROOT="${ROOT:-/root/work/shensi-offline/shensi}"
PKG="$(cd "$(dirname "$0")" && pwd)"

M="${MCORE_DIR:-$ROOT/3rdparty/common/Megatron-LM}"
MS="${MINDSPEED_DIR:-$ROOT/3rdparty/ascend/MindSpeed}"

say() { echo "[npu-patches] $*"; }
command -v patch >/dev/null 2>&1 || { say "ERROR: 需要 GNU patch"; exit 2; }
[ -d "$M" ] || { say "ERROR: mcore 树不存在: $M（先 git submodule update）"; exit 2; }
[ -d "$MS" ] || { say "ERROR: MindSpeed 树不存在: $MS"; exit 2; }

FAIL=0
apply_one() { # 树 补丁文件 标签
  local dir="$1" p="$2" tag="$3" name
  name="$(basename "$p")"
  if patch -p1 -d "$dir" --dry-run --forward -r - < "$p" >/dev/null 2>&1; then
    if [ "$CHECK" = 1 ]; then
      say "适用   $tag/$name（--check：未落盘）"
    elif patch -p1 -d "$dir" --forward -r - < "$p" >/dev/null 2>&1; then
      say "applied $tag/$name"
    else
      say "ERROR: dry-run 通过但正式应用失败: $tag/$name"; FAIL=1
    fi
  elif patch -p1 -d "$dir" --dry-run --reverse -r - < "$p" >/dev/null 2>&1; then
    say "already $tag/$name（已应用，跳过）"
  else
    say "ERROR: $tag/$name 既不适用、也不是已应用——目标树版本不对？"
    [ "$tag" = "mcore" ] && say "       （0001 是防 NaN 的必需项，别跳过；预期 mcore revision c16f739b1600）"
    FAIL=1
  fi
}

for p in "$PKG"/patches/mcore/*.patch; do
  [ -f "$p" ] || continue
  apply_one "$M" "$p" mcore
done
for p in "$PKG"/patches/mindspeed/*.patch; do
  [ -f "$p" ] || continue
  apply_one "$MS" "$p" mindspeed
done

[ "$FAIL" = 0 ] || { say "补丁阶段失败——不要带半套补丁训练"; exit 3; }

if [ "$CHECK" = 1 ]; then
  say "check 完成（未改文件）"
else
  say "完成。落地终态 sha（与 team 的 expected_sha256.txt 对照）："
  for f in \
    megatron/core/transformer/experimental_attention_variant/csa.py \
    megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py \
    megatron/training/argument_utils.py \
    megatron/core/optimizer/layer_wise_optimizer.py; do
    [ -f "$M/$f" ] && sha256sum "$M/$f" | sed "s|$M/|  |"
  done
  [ -f "$MS/mindspeed/megatron_adaptor.py" ] && sha256sum "$MS/mindspeed/megatron_adaptor.py" | sed "s|$MS/|  |"
fi
