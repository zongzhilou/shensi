#!/usr/bin/env bash
# stage5_eval 的环境准备：建独立评测 venv（不动训练用的 .venv），装 EvalScope 与 deepseek-harness
# SDK，并把 dsh profile 的模型端点指到本配方起的 vLLM。跑一遍就能直接 `python eval.py`。
#
#   bash setup_env.sh
#   PROXY=http://127.0.0.1:7897 bash setup_env.sh      # 需要代理时
set -euo pipefail

ROOT="${SHENSI_ROOT:-/root/work/shensi}"
FS="${SHENSI_FS:-/root/work/filestorage}"
VENV="${SHENSI_EVAL_VENV:-$ROOT/.venv_eval}"
DSH_HOME_DIR="${DSH_HOME:-$FS/shensi/dsh-home}"
ENDPOINT="${SHENSI_ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL="${SHENSI_SERVED_MODEL:-looma}"
export UV_HTTP_TIMEOUT=180
[ -n "${PROXY:-}" ] && export https_proxy="$PROXY" http_proxy="$PROXY"

echo "== ① 独立 venv（训练用的 .venv 不动）"
uv venv "$VENV" --python 3.12 --allow-existing

echo "== ② 装 EvalScope（评测）与 deepseek-harness SDK（agent 类基准）"
# langdetect：IFEval 的指标要它，但 evalscope 1.12 的依赖元数据里没声明（报错里提的
# `evalscope[ifeval]` extra 在本版本不存在），所以显式装。
uv pip install -p "$VENV/bin/python" evalscope langdetect deepseek-harness-sdk

echo "== ③ 装 vLLM 插件入口点（每个 vLLM 进程都能登记本配方的原生实现）"
"$ROOT/.venv/bin/python" -m shensi.recipes.paper.looma.common.models.vllm.register_model install \
  || echo "  ! 插件安装失败（训练 venv 不在？）：端点会退回 remote-code 路径"

echo "== ④ 初始化 dsh profile（$DSH_HOME_DIR）"
export DSH_HOME="$DSH_HOME_DIR"
mkdir -p "$DSH_HOME"
"$VENV/bin/dsh" --profile sdk-minimal --dump-default-config >/dev/null

echo "== ⑤ 把 profile 的模型端点指到本配方的 vLLM"
mkdir -p "$DSH_HOME/profiles/sdk-minimal"
cat > "$DSH_HOME/profiles/sdk-minimal/cordis.patch.yml" <<YAML
- id: llm-deepseek
  config:
    apiKeyEnv: DEEPSEEK_API_KEY
    baseUrl: $ENDPOINT
    model: $MODEL
YAML

echo "== ⑥ 自检（离线可验）"
"$VENV/bin/python" -c "import evalscope; print('  ✔ evalscope', evalscope.__version__)"
"$VENV/bin/dsh" --profile sdk-minimal --dump-config | grep -q "$ENDPOINT" \
  && echo "  ✔ dsh profile 已指向 $ENDPOINT（model=$MODEL）" \
  || { echo "  ✗ profile 里没看到端点，检查 ⑤"; exit 1; }

cat <<TXT

环境就绪。跑评测：
  export DSH_HOME=$DSH_HOME_DIR DEEPSEEK_API_KEY=dummy SHENSI_ROOT=$ROOT SHENSI_FS=$FS
  cd <recipes>/paper/looma/stage5_eval
  python eval.py --dry-run          # 先看命令（vllm serve + evalscope + dsh）
  python eval.py                    # 起服务 → EvalScope 基准 → 可选 agent 类基准
TXT
