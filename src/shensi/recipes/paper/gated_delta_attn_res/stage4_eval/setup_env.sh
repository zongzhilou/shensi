#!/usr/bin/env bash
# stage4_eval 的环境准备：装 OpenCompass（独立 venv，numpy<2 与训练侧冲突）与 deepseek-harness
# SDK，初始化 dsh profile 指向我们的 vLLM 端点。跑一遍就能直接 `python eval.py`。
#
#   bash setup_env.sh
#   PROXY=http://127.0.0.1:7897 bash setup_env.sh      # 需要代理时
set -euo pipefail

ROOT="${SHENSI_ROOT:-/root/work/shensi}"
FS="${SHENSI_FS:-/root/work/filestorage}"
OC_VENV="${SHENSI_OPENCOMPASS_VENV:-$ROOT/.venv-opencompass}"
DSH_VENV="${SHENSI_DSH_VENV:-$ROOT/.venv_eval}"
DSH_HOME_DIR="${DSH_HOME:-$FS/shensi/dsh-home}"
ENDPOINT="${SHENSI_ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL="${SHENSI_SERVED_MODEL:-gdar}"
export UV_HTTP_TIMEOUT=300
[ -n "${PROXY:-}" ] && export https_proxy="$PROXY" http_proxy="$PROXY"

echo "== ① OpenCompass 的独立 venv（CPU torch，依赖走镜像；OpenCompass 本体装 GitHub 最新版）"
uv venv "$OC_VENV" --python 3.12 --allow-existing
uv pip install --python "$OC_VENV/bin/python" --torch-backend=cpu \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "opencompass @ git+https://github.com/open-compass/opencompass.git"

echo "== ② dsh 的独立 venv（agent 类基准的 harness）"
uv venv "$DSH_VENV" --python 3.12 --allow-existing
uv pip install -p "$DSH_VENV/bin/python" deepseek-harness-sdk

echo "== ③ 装 vLLM 插件入口点（每个 vLLM 进程都能登记本配方的原生实现）"
"$ROOT/.venv/bin/python" -m shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.register_model install \
  || echo "  ! 插件安装失败（训练 venv 不在？）：端点会退回 remote-code 路径"

echo "== ④ 初始化 dsh profile（$DSH_HOME_DIR）"
export DSH_HOME="$DSH_HOME_DIR"
mkdir -p "$DSH_HOME"
"$DSH_VENV/bin/dsh" --profile sdk-minimal --dump-default-config >/dev/null

echo "== ⑤ 把 profile 的模型端点指到本配方的 vLLM"
mkdir -p "$DSH_HOME/profiles/sdk-minimal"
cat > "$DSH_HOME/profiles/sdk-minimal/cordis.patch.yml" <<YAML
# 只覆盖 llm-deepseek 这一行：端点指向我们自己的 vLLM（OpenAI/DeepSeek 兼容），其余沿用随包附带配置。
- id: llm-deepseek
  config:
    apiKeyEnv: DEEPSEEK_API_KEY
    baseUrl: $ENDPOINT
    model: $MODEL
YAML

echo "== ⑥ 自检（离线可验）"
"$OC_VENV/bin/python" -c "import opencompass; print('  ✔ opencompass', opencompass.__version__)"
"$OC_VENV/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import benchmarks, opencompass_eval
py = Path(sys.executable)
mods = opencompass_eval.discover_datasets(py)
print(f"  ✔ 数据集配置 {len(mods)} 个；口径集合 minicpm5 的每一项都能解开："
      f"{all(any(benchmarks.oc_name(n) and benchmarks.oc_name(n) in m for m in mods) for n in benchmarks.OC_SETS['minicpm5'])}")
PY
"$DSH_VENV/bin/dsh" --profile sdk-minimal --dump-config | grep -q "$ENDPOINT" \
  && echo "  ✔ dsh profile 已指向 $ENDPOINT（model=$MODEL）" \
  || { echo "  ✗ profile 里没看到端点，检查 ⑤"; exit 1; }

cat <<TXT

环境就绪。跑评测：
  export DSH_HOME=$DSH_HOME_DIR DEEPSEEK_API_KEY=dummy SHENSI_ROOT=$ROOT SHENSI_FS=$FS
  cd <recipes>/paper/gated_delta_attn_res/stage4_eval
  python eval.py --dry-run              # 先看命令（vllm serve + opencompass + dsh）
  python eval.py --suite minicpm5       # 口径主力项；不带 --suite 跑 OpenCompass 自带集合
TXT
