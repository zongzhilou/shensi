# 模型：vLLM rollout 侧

用 vLLM 服务七个变体，全部走上游自己的扩展点（`ModelRegistry.register_model` +
`vllm.general_plugins` 入口点）—— **不改 vLLM 的任何文件**。本目录刻意自包含：import
`models.vllm.register_model` 不会把 Megatron、TransformerEngine 或 verl 带进来，因为引擎的
worker 进程只需要"注册 + 桥接"这一小片（有检查脚本守着这一点）。

## 总览

| 文件 | 说明 |
|---|---|
| `variants.py` | 引擎眼里的七个变体：`architectures[0]`（分发键）、`model_type`、两份 remote-code 文件名、tiny 冒烟要带的非默认旋钮；`SHAPES`（`tiny` = 2×64、`0.6b` = 28×1024/16 头） |
| `tiny_checkpoint.py` | 造**随机权重**的自描述检查点（连接打开，`auto_map` + 两个 `.py` 拷到权重旁边） |
| `register_model.py` | 登记 7 个架构；`install` 子命令顺带把 `vllm.general_plugins` 入口点写进 venv |
| `vllm_bridge.py` | 真正被登记的实现：vLLM 的 Transformers 后端 + **把连接子树排除出引擎的模块重写** |
| `sitecustomize.py` | 让注册抵达 **EngineCore worker 进程**（`ROLLOUT_PLUGIN_AUTOLOAD=1` 时才动作） |
| `smoke_generate.py` | 端到端：checkpoint → 真 `vllm.LLM` → 16 tokens，可选与纯 transformers 参考对拍（最长公共前缀） |
| `batch_generate.py` | 连续批处理：多 prompt 一次生成 vs 逐条生成，附引擎的 running/waiting 计数 |
| `mudd_fused_qkv_repro.py` | `mudd` 触发 QKV 融合失败的最小复现（CPU），再证明融合/不融合逐位一致 |
| `check_decode_state.py` | 实测"深度状态是否跨 decode 步丢失"（没有丢：所有深度操作都是 token 轴逐点的，缓存增量解码 = 全量重算） |
| `_paths.py` | `SRC`（让 `import shensi...` 在任何解释器成立）与配方内 tokenizer 路径 |

## 为什么需要一个桥接子类

vLLM 的 Transformers 后端会把 HF 模型里的 `nn.Linear` / `RMSNorm` / `nn.Embedding` / QKV 换成
引擎自己的实现（融合 QKV、paged attention、TP-aware 线性层）—— **这正是骨干想要的、也正是连接
要坏的**。实测两个失败（vLLM 0.30）：

1. `TPAwareRMSNorm` 没有 `.eps`，而连接的读要读它；
2. 连接在 fp32 里算，被换掉的 `nn.Linear` 是 bf16 → dtype 冲突。

`DepthTransformersForCausalLM` 的做法：引擎重写骨干期间把连接子树临时换成 `nn.Identity`，
重写完再原样放回。骨干全走引擎内核，连接保持 HF 定义，vLLM 文件零改动。

## 快速开始

```bash
R=<repo>/src/shensi/recipes/paper/gated_delta_attn_res
export PATH=$(dirname $(which python)):$PATH        # flashinfer JIT 要 ninja

# 0) 一次性：让引擎自己发现插件（之后不需要任何环境变量）
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.register_model install

# 1) 单个变体：造 ckpt → 生成 16 tokens
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.tiny_checkpoint gdar /tmp/vllm_smoke/gdar --overwrite
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.smoke_generate --variant gdar --tokens 16 \
    --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 128

# 2) 七个一起（每个变体一个子进程），并与纯 transformers 对拍
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.smoke_generate --all \
    --tokens 16 --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 128 --json /tmp/vllm_smoke.json

# 3) 0.6B 形状（28 层 / 1024 hidden，真几何）与连续批处理
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.tiny_checkpoint dar /tmp/vllm_06b/dar --shape 0.6b --overwrite
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.batch_generate --variant dar --compare-hf

# 4) mudd / QKV 融合最小复现（CPU）
python -m shensi.recipes.paper.gated_delta_attn_res.models.vllm.mudd_fused_qkv_repro
```

## 实测状态

- `register_model register --check`：**7/7 架构登记成功**（vllm 0.30.1rc0.dev360）。
- 自包含性：import `models.vllm.register_model` 之后，`megatron` / `transformer_engine`
  都**未进 `sys.modules`**。
- 端到端生成（tiny checkpoint → `vllm.LLM` → 与 HF 参考对拍）：**7/7 变体生成、与纯 transformers
  参考逐 token 一致（lcp = 16/16，fp32、tiny 形状）**。
- 引擎确实会换掉 norm 类模块，所以 GDAR 的建模文件把 `eps` 缓存在连接内部（`self._eps`）——
  这一处免疫是 gdar 从 FAIL 到 OK 的修复。`tiny_checkpoint` / HF 远程代码缓存**按文件哈希复用**：
  改过 modeling 文件后要清掉旧 checkpoint 与
  `~/.cache/huggingface/modules/transformers_modules/<name>/` 下的旧副本。
- 连续批处理与逐条结果一致、调度器确实并行（峰值并发 > 1）。bf16 下随机权重的贪心解码在批形状
  变化时可能翻转 argmax（top-2 差距 ~1e-4）—— 是数值并列，不是调度缺陷。
- **深度状态不需要 cache**（深度操作都是 token 轴逐点的 ⇒ 缓存增量解码 = 全量重算）——
  这是写原生引擎实现的前提。

## 局限

连接模块**没有**原生内核：登记的实现把执行委托给 HF 实现，所以 vLLM 贡献的是调度 / 批处理 /
采样 / OpenAI 兼容面，paged attention 内核作用于**骨干**、不作用于深度路由。极限吞吐
（连接算子的融合内核 / 原生 paged 实现）是另一个工程项。

## 延伸阅读

- [HF 参考实现](../transformers/README.md) —— 被服务的模型
- [RL 阶段](../../stage2_rl/README.md) —— vLLM 在那里是 rollout 引擎
- [LIMITATIONS.md](../../LIMITATIONS.md) —— B6（融合内核）
