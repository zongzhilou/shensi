# 模型：HuggingFace 参考实现

七个深度连接变体的纯 PyTorch 参考实现，用于**研究、评测、转换与 rollout** —— 从不用于训练
（训练一律走 `models/megatron/` 的 mcore 路径）。

每个变体都忠实于**它自己的上游仓库**，而这份忠实是数值验证出来的：官方算子实现 vendored 在
`upstream/` 下，与我们的实现逐张量对拍（`test_upstream_alignment.py`，13/13）。

## 总览

| 文件 | 变体 | 上游 |
|---|---|---|
| `modeling_qwen3_gdar.py` | GDAR —— 门控 delta 注意力残差（论文的模型） | 本工作（Kimi 式深度读 + 门控 delta 写） |
| `modeling_qwen3_ar.py` | AR —— attention residuals | Kimi Linear（`upstream/kimi_modeling_kimi_linear.py`） |
| `modeling_qwen3_dar.py` | DAR —— depth attention residual | Kimi Linear |
| `modeling_qwen3_denseformer.py` | DenseFormer —— 深度加权平均 | `upstream/denseformer_*.py` |
| `modeling_qwen3_mudd.py` | MUDD —— multiway dynamic dense | `upstream/muddformer_*.py` |
| `modeling_qwen3_hc.py` / `modeling_qwen3_mhc.py` | HC / mHC —— hyper-connections | shensi 分支 / 论文 |
| `guarantee.py` | 各变体共用的恒等性保证工具 | — |

每个变体带一个配置类（`configuration_qwen3_*.py`），承载连接旋钮（`attn_res_*`）、注册
`model_type`（如 `qwen3_gdar`）并声明 `auto_map` —— 于是 `save_pretrained` 存下的检查点在
新进程里用 `trust_remote_code=True` 就能加载，这正是 verl 与 vLLM 消费这些模型的方式。

## GDAR 算子（参考语义）

每个子层的残差流都通过一份深度状态被读写：

- **写** —— 子层输出带门控 delta 规则写进状态：逐通道可学 decay（`decay_tau` ladder，
  `attn_res_decay_positivity="project"` 下由构造保证为正）、erase 门与 write 门，都由状态驱动。
- **读** —— 子层输入是状态快照上的白化多头读（λ 夹紧的闭式更新、`Softmax¬1`、可学习 null source）。
- **恒等** —— 在论文初始化下，整条连接精确等于 plain 残差流：`GDAR(0) == Qwen3` 逐位成立
  （证明要点与非恒等形态的实测偏差写在 `modeling_qwen3_gdar.py` 的 docstring 里）。

低秩投影（`attn_res_{gate,q,k}_rank`）让算子可负担：全秩在 8B 上要花约 45% 的时间，其中
`k_proj` 一项就约 15%。

## 快速开始

```bash
cd models/transformers

# HF 单测套件
python test_theory.py                 # 58/58 —— 算子代数、恒等性、门语义
python test_ablation_switches.py      # 64/64 —— 每个旋钮的效应
python test_autoclass.py              # 42/42 —— 配置 / auto_map / 序列化往返
python smoke_test.py                  # 8 种配置的前向+反向，回传路由统计

# 与 vendored 上游实现对拍
python test_upstream_alignment.py     # 13/13（AR / MUDD / DenseFormer / GDAR-per-head 逐位）
```

## 判据

| 检查 | 结果 |
|---|---|
| 与上游对齐 | AR vs Kimi-K3 算子**逐位**、MUDD vs MUDDFormer block **逐位**、DenseFormer vs 官方 DWAModules **逐位**、GDAR vs shensi 分支在 `per_head` 下**逐位**（13/13；已知 delta 逐变体列出） |
| 理论套件 | 58/58 |
| 消融开关 | 64/64 |
| 自动类 / 序列化 | 42/42 |
| 前向+反向冒烟 | 8 种配置，回传路由统计（`sharpness`、`entropy`、`n_sources`、门值） |

`upstream/PROVENANCE.md` 记录每个 vendored 文件的 sha256；它们保持逐字节一致（本仓的格式化
已把它们排除在外）。

## 尚未移植

- shensi 的 hyper-connection 多流（`ShensiHyperConnection`）—— 属于另一套机制。
- shensi 的 `block_write_layer` / `attn_res_block_layer_types` —— 本目录用与 DAR 一致的块源语义
  （快照差分），保证 GDAR-vs-DAR 只差门控。
- DAR 仓库的 V-stream 解耦注意力（`Qwen3AttnResAttention` 及其 `delta_v` 变体）。
- MoE / KDA 注意力 —— 本目录只做 Qwen3 稠密基座。

## 延伸阅读

- [配方总 README](../../README.md) —— 这些实现在哪些环节被用到（rollout、转换、评测）
- [vLLM rollout](../vllm/README.md) —— 用引擎服务这些模型
- [LIMITATIONS.md](../../LIMITATIONS.md) —— A6（逐头白化）、A12（格式化与出处）
