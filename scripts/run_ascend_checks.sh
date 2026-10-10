#!/usr/bin/env bash
# Run in an existing, version-matched CANN/torch_npu environment. No package replacement.
set -euo pipefail
SHENSI_TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SHENSI_TASK_ROOT"
export PYTHONPATH="$SHENSI_TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SHENSI_BACKEND=npu
export SHENSI_NPU_EIGH="${SHENSI_NPU_EIGH:-cpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$SHENSI_TASK_ROOT/artifacts/ascend_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
"$PYTHON_BIN" -m shensi.utils.ascend_env --scope reference --json "$OUT/preflight.json"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
    -m shensi.utils.npu_collective | tee "$OUT/hccl.log"
"$PYTHON_BIN" -m pytest -q tests/test_ascend_incremental.py --junitxml="$OUT/pytest.xml"
MODULE=shensi.recipes.paper.gated_delta_attn_res.common.tiny_reference
"$PYTHON_BIN" -m "$MODULE" --device npu --dtype float32 --out "$OUT/reference_fp32"
"$PYTHON_BIN" -m "$MODULE" --device npu --dtype bfloat16 --out "$OUT/reference_bf16"
"$PYTHON_BIN" -m "$MODULE" --device npu --whiten per_head --out "$OUT/reference_per_head"
if [[ "${RUN_NATIVE_MEGATRON:-0}" == 1 ]]; then
    "$PYTHON_BIN" -m shensi.recipes.paper.gated_delta_attn_res.common.native_tiny \
        --backend npu --out "$OUT/native_megatron"
fi
printf 'Completed reference checks. Results: %s\n' "$OUT"
printf 'This does not certify verl/vLLM/OpenCompass or a production training run.\n'
