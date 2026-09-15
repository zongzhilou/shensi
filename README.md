# Shensi

Shensi 模型族的 **Megatron 插件库**：以 DeepSeek-V4（DSv4）的上游实现为基座，只写差异。

目录与上游一一对应（把包名 `megatron` 换成 `shensi`，其余路径照搬）：

| 本仓库 | 上游 | 内容 |
|---|---|---|
| `src/shensi/core/` | Megatron-LM 的 `megatron/core/` | Shensi 独有的层 / 模块 / spec / 模型 |
| `src/shensi/bridge/` | Megatron-Bridge 的 `src/megatron/bridge/` | provider、bridge（HF↔mcore 配置与参数映射） |
| `tests/` | 上游同名相对路径 | 移植件逐位对拍 + 转换/前向对拍 |

放置规则：**上游有且能直接用 → import 上游**；上游有但必须改 → 逐字复制到相同相对路径；上游没有 →
自写，文件名统一加 `shensi_` 前缀（一眼区分"副本"与"自写件"）。

## 必须自写的四处（HF 有、上游无对应物）

| 文件 | 说明 |
|---|---|
| `core/transformer/shensi_hyper_connection.py` | 超连接：`pre/route/post_fn` 三组门控 + 固定流/路由 top-k 的**局部**回写；上游 mHC 是 `mapping_proj` + `alpha_*` + Sinkhorn 的另一套参数化，不可互换 |
| `core/transformer/shensi_attention_residual.py` | 块级记忆：delta-rule 写入/擦除/衰减 + 块库软路由；上游完全没有对应物 |
| `core/transformer/shensi_hash_mlp.py` | hash 层：dense MLP × `deepemb(input_ids)` 门控（上游的 `moe_n_hash_layers` 机制在当前 mcore 修订无消费者） |
| `core/transformer/shensi_layer.py`、`shensi_block.py` | 三态传递的层与块（打包 `[s,b,(2·hc+blocks·hc)·H]`）；出口收缩放在块的 `layer_norm` 槽位 |

其余一律复用上游：注意力 / 压缩器 / Indexer / MoE 外壳与 latent MoE、DSv4 hybrid 的 spec 工厂、
以及 Bridge 的专家映射（`FusedGatedExpertMapping` / `FusedExpertMapping`）。

## 环境

- 训练栈是 conda 环境 **`pre`**（Python 3.12 + Megatron-LM + Megatron-Bridge + TransformerEngine），
  因此本包 `requires-python >=3.12`；安装：`uv pip install -e .`。
- 额外依赖：`fast-hadamard-transform`（DSA 必需，须从 git tag v1.1.0 安装）、`megatron-energon`
  （Bridge 的 diffusion 子包在 import 时拉它）。

## 验证

```sh
python -m pytest -q tests/            # 全部（9 项，约 20 秒）
python -m pytest -q tests/unit_tests  # 只要移植件的逐位对拍
```

- **移植件逐位**（`tests/unit_tests`）：4 个自建模块 vs HF 参考实现 `torch.equal`（0 位差），
  毫秒级、不受下游内核精度影响 —— 移植保真的最硬判据；
- **权重往返**：键集合与检查点完全一致（216/216，读层的 gate/experts 因共享被去重），
  最大绝对差 **0.000e+00**；
- **fp32 前向对拍**：用"全 sliding"检查点（压缩层的 Hadamard 旋转只支持 bf16，绕开它才能在
  fp32 下比），逐层三态与 logits 满足 `diff ≤ 2e-5 + 2e-3·max|x|` —— 这个量级由环境决定：
  mcore 的线性层是 TE 的、fp32 下走 TF32（相对 ~5e-3），HF 侧是真 fp32。

## 边界

- **上游 DSv4 为 TP=1**（`DSv4HybridAttention` 里有断言），扩展靠 DP/EP。
- **PP>1 不可用**：`TransformerConfig.__post_init__` 对 `enable_mhc_connections` + PP>1 直接抛
  `NotImplementedError`（层间激活是 n-stream，而 pipeline 的 p2p 缓冲按 `hidden_size` 定尺寸，
  绕过等于静默截断；修它必须改 mcore）。因此参数只能靠 EP 分片，其余部分需在单卡内容纳。
- 压缩层（CSA/HCA）**不支持推理**：`DSv4HybridAttention` 断言不得传入推理上下文。
- **MTP 未适配**：超连接 + MTP 需要上游的 HybridModel 契约；Shensi 各版配置均为
  `num_nextn_predict_layers = 0`，不触发。
- hash 层的 `input_ids` 走模块级暂存（`core/transformer/context.py`），以便反向重算时也能取到；
  代价是同一进程内不能交错执行两个 microbatch 的前向与反向。

## NPU（下一步）

`npu/` 下已按 Ascend 官方矩阵钉好版本（MindSpeed / MegatronAdaptor / Megatron-LM = `core_r0.18.0`，
TransformerEngineNPU = `main`，MindSpeed-LLM = `master`，后者含 DeepSeek-V4 实现与 HF↔mcore 转换
脚本）。注意：0.18.2 的 Megatron-LM **没有 mHC、也没有 DSv4 hybrid 注意力**，因此 NPU 侧需要
移植这两块（可参考 MindSpeed-LLM 的 `tasks/models/transformer/deepseek4/`）。本仓库当前面向的
是本地 CUDA 栈（mcore main + Megatron-Bridge）。

## 许可

Apache-2.0。
