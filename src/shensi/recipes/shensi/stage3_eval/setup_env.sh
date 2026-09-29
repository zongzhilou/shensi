#!/usr/bin/env bash
# stage3_eval 的环境准备：装 Gym 与 dsh（独立 venv，不动训练用的 .venv）、备 Gym 检出、
# 初始化 dsh profile 并把模型端点指到我们的 vLLM。跑一遍就能直接 `python eval.py`。
#
#   bash setup_env.sh              # 用默认路径（$SHENSI_ROOT / $SHENSI_FS）
#   PROXY=http://127.0.0.1:7897 bash setup_env.sh
set -euo pipefail

ROOT="${SHENSI_ROOT:-/root/work/shensi}"
FS="${SHENSI_FS:-/root/work/filestorage}"
VENV="${SHENSI_GYM_VENV:-$ROOT/.venv_gym}"
GYM_DIR="${SHENSI_GYM_DIR:-$ROOT/Gym}"
DSH_HOME_DIR="${DSH_HOME:-$FS/shensi/dsh-home}"
ENDPOINT="${SHENSI_ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL="${SHENSI_SERVED_MODEL:-shensi}"
export UV_HTTP_TIMEOUT=180
CURL_PROXY=()
[ -n "${PROXY:-}" ] && CURL_PROXY=(-x "$PROXY") && export https_proxy="$PROXY" http_proxy="$PROXY"

echo "== ① 独立 venv（训练用的 .venv 不动）"
uv venv "$VENV" --python 3.12 --allow-existing   # 幂等：已经在就复用

echo "== ② 装 Gym 与 dsh"
uv pip install -p "$VENV/bin/python" nemo-gym deepseek-harness-sdk

echo "== ③ 备 Gym 检出（benchmarks/environments 都在里面）"
[ -d "$GYM_DIR" ] || git clone --depth 1 https://github.com/NVIDIA-NeMo/Gym.git "$GYM_DIR"

echo "== ④ 初始化 dsh profile（$DSH_HOME_DIR）"
export DSH_HOME="$DSH_HOME_DIR"
mkdir -p "$DSH_HOME"
"$VENV/bin/dsh" --profile sdk-minimal --dump-default-config >/dev/null

echo "== ⑤ 把 profile 的模型端点指到我们的 vLLM"
cat > "$DSH_HOME/profiles/sdk-minimal/cordis.patch.yml" <<YAML
# 只覆盖 llm-deepseek 这一行：端点指向我们自己的 vLLM（OpenAI/DeepSeek 兼容），其余沿用随包附带配置。
- id: llm-deepseek
  config:
    apiKeyEnv: DEEPSEEK_API_KEY
    baseUrl: $ENDPOINT
    model: $MODEL
YAML

echo "== ⑥ 自检（离线可验）：profile 组合出来应包含我们的端点"
"$VENV/bin/dsh" --profile sdk-minimal --dump-config | grep -q "$ENDPOINT" \
  && echo "  ✔ dsh profile 已指向 $ENDPOINT（model=$MODEL）" \
  || { echo "  ✗ profile 里没看到端点，检查 ⑤"; exit 1; }
"$VENV/bin/dsh" --version

cat <<TXT

环境就绪。跑评测：
  export DSH_HOME=$DSH_HOME_DIR DEEPSEEK_API_KEY=dummy
  export SHENSI_ROOT=$ROOT SHENSI_FS=$FS
  cd <recipes>/shensi/stage3_eval
  python eval.py --dry-run        # 先看命令（vLLM + gym eval + harness_agent=dsh）
  python eval.py                  # 起服务 + 跑 Gym 基准（dsh 当 harness）
TXT
