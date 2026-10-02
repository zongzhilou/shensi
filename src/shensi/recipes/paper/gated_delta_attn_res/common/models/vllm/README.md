# 模型：vLLM rollout 侧

用 vLLM 服务七个深度连接变体（RealFormer 除外，原因见下）。注册走 vLLM 自己的扩展点
（`ModelRegistry.register_model` + `vllm.general_plugins` 入口点），**不改 vLLM 任何文件**。
本目录刻意自包含：import 它不会把 Megatron / TransformerEngine / verl 带进来（引擎 worker
只需要"注册 + 桥接"这一小片）。

## 总览

| 文件 | 说明 |
|---|---|
| `variants.py` | 七个变体在引擎眼里的样子：`architectures[0]`、`model_type`、随权重走的两个 `.py`、tiny 冒烟旋钮、`SHAPES` |
| `register_model.py` | 登记架构；`install` 顺带写入 `vllm.general_plugins` 入口点 |
| `vllm_bridge.py` | 被登记的实现：vLLM 的 Transformers 后端 + 把连接子树排除出引擎的模块重写 |
| `sitecustomize.py` | 让注册抵达 EngineCore worker 进程（`ROLLOUT_PLUGIN_AUTOLOAD=1` 时动作） |
| `tiny_checkpoint.py` | 造随机权重的自描述检查点（连接打开、`auto_map` 齐备） |
| `smoke_generate.py` / `batch_generate.py` | 端到端生成与连续批处理（可与纯 transformers 对拍） |
| `mudd_fused_qkv_repro.py` / `check_decode_state.py` | QKV 融合的最小复现 / 深度状态是否跨 decode 丢（没有丢） |

## 为什么需要桥接子类

引擎会把 HF 模型里的 `nn.Linear` / `RMSNorm` / `nn.Embedding` / QKV 换成自己的实现（融合 QKV、
paged attention、TP-aware 线性层）——这正是骨干想要的，也正是连接要坏的：实测两个失败
（`TPAwareRMSNorm` 没有 `.eps`；连接在 fp32 里算而被换掉的层是 bf16）。
`DepthTransformersForCausalLM` 在引擎重写骨干期间把连接子树临时换成 `nn.Identity`，重写完
再原样放回：骨干全走引擎内核，连接保持 HF 定义。

> **RealFormer 不在此列**：它的注意力必须物化分数矩阵，而引擎要求每层注意力经 HF 注意力
> 接口分发——登记过、被实测拒绝（`ValueError: Layer 0 (full_attention) does not dispatch
> through the Transformers attention interface`）。要服务它得写 vLLM 原生实现，单列工程项；
> 作为训练侧对照臂它不需要 rollout。

## 快速开始

```bash
export PATH=$(dirname $(which python)):$PATH        # flashinfer JIT 要 ninja

P=shensi.recipes.paper.gated_delta_attn_res.common.models.vllm
python -m $P.register_model install                  # 一次性：装插件入口点
python -m $P.tiny_checkpoint gdar /tmp/vllm_smoke/gdar --overwrite
python -m $P.smoke_generate --variant gdar --tokens 16 \
    --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 128
python -m $P.smoke_generate --all --tokens 16 \
    --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 128 --json /tmp/vllm_smoke.json
```

## 实测

- `register_model register --check`：**7/7 架构登记成功**；
- 自包含性：import 之后 `megatron` / `transformer_engine` 都**未进 `sys.modules`**；
- 端到端生成：**7/7 变体与纯 transformers 参考逐 token 一致（最长公共前缀 = 16/16）**；
- 连续批处理与逐条结果一致、调度器确实并行；
- **深度状态不需要 cache**（所有深度操作都是 token 轴逐点的，缓存增量解码 = 全量重算）；
- bf16 下随机权重贪心解码在批形状变化时可能翻转 argmax（top-2 差距约 1e-4 的数值并列）。

## 局限

连接模块没有原生内核：登记的实现把执行委托给 HF 实现，vLLM 贡献的是调度 / 批处理 / 采样 /
OpenAI 兼容面；paged attention 作用于骨干、不作用于深度路由。极限吞吐（连接算子的融合内核 /
原生 paged 实现）是独立工程项。

## 下一步

- [HF 参考实现](../transformers/README.md) —— 被服务的模型
- [RL 阶段](../../../stage2_rl/README.md) —— vLLM 在那里是 rollout 引擎
- [配方 README](../../../README.md) —— 已知边界（「边界」一节）
