# Depth-routed Qwen3 变体（AR / DAR / GDAR）

基座固定 Qwen3，只替换**深度连接模块**：注意力、MLP、RMSNorm、RoPE、LM head 全部沿用
`transformers.models.qwen3`。因此变体之间的差异只有连接这一处，比较是干净的。

## 文件结构（对齐 `moonshotai/Kimi-K3`）

Kimi 的仓库是 `configuration_kimi_k3.py` + `modeling_kimi_linear.py` 成对出现，本目录一一对应：

| 变体 | 配置文件 | 建模文件 |
|---|---|---|
| AR | `configuration_qwen3_ar.py` | `modeling_qwen3_ar.py` |
| DAR | `configuration_qwen3_dar.py` | `modeling_qwen3_dar.py` |
| GDAR | `configuration_qwen3_gdar.py` | `modeling_qwen3_gdar.py` |

**没有共享模块**：每个 modeling 文件自带 norm、路由算子、解码层、backbone、LM head 和
auto-class 注册，只从自己的 configuration 文件（相对导入）取配置——与 Kimi 那份文件完全同构。
把 `modeling_X.py` + `configuration_X.py` 成对拷到任何包里即可独立使用（已验证）。

建模文件里沿用的 Kimi 范式：

| Kimi | 本目录 |
|---|---|
| 模块级 `_apply_attn_res(prefix_sum, block_residual, proj, norm)` | AR 文件同名函数（逐行一致，只多一个诊断用 `return_probs`） |
| `prefix_sum` = 在途累加量，就是该层 hidden state | 同名同义 |
| `block_residual` = `(num_tokens, num_blocks, hidden)` 张量 | 同名同形；DAR/GDAR 存 delta 源 |
| 解码层 `_forward_attn_residual` 返回 `(prefix_sum, residual)` | 同名同结构 |
| 主干循环携带张量状态 + 末端 output routing | `Qwen3XModel.forward` |
| `config.attn_res_block_size`（`None` = 普通 Qwen3） | 同名同义 |
| `KimiRMSNorm` / `KimiMLP` / `KimiDinamicCache` 等自带件 | AR 自带 `RMSNorm`；GDAR 自带 `UnweightedRMSNorm`（见下） |
| `@check_model_inputs` | 改用 `@merge_with_config_defaults`（5.17 里前者已废弃并告警） |

## 算子各自对齐仓库

| 变体 | 算子出处 | 对齐要点 |
|---|---|---|
| **AR** | `modeling_kimi_linear.py:_apply_attn_res` | 逐行一致：`cat([blocks, prefix])` → rsqrt 方差归一 → `score_weight = norm.weight * proj.weight` → softmax → 加权求和；路由结果**替换**子层输入 |
| **DAR** | `wdlctc/delta-attention-residuals-code:delta_attn_res` | 保留该仓库形式：`K = norm(V)`、`logits = <query, K>`、`softmax(dim=0)`、`output = partial_block + selected`（加性）；并实现其 `null_source`（零初始化空源，微调时保证近恒等） |
| **GDAR** | `zongzhilou/transformers@shensi:ShensiAttentionResidual` | 逐行一致：**无权重** state norm、`gate_proj` 全秩 `Linear(D,3D)`、`q_proj`/`k_proj` 为裸 `Parameter(D,D)`、路由用 `rsqrt(mean(v²))` 缩放 + `q_proj(state)`、`values = cat([blocks, updated])`、`output = updated + routed` |

### 接线上的必要差异（已在文件 docstring 标注）

shensi 的连接模块嵌在它的 hyper-connection 多流架构里：模块同时产出**子层输入**
（`updated + routed`）与**新流**（`updated`），而 shensi 的 decoder 里 MLP 那一步传给模块的
`delta` 就是 `prefix_sum` 本身，子层输出的回流由 hyper-connection 承担。

本目录的基座是纯 Qwen3、没有 hyper-connection，所以把「最新子层输出」显式作为 `delta`
穿过下一层（`prev_delta` 状态）。这样到位后：`write` 门把子层输出写进流里，恒等初始化下
`updated = prefix + delta`，**GDAR 严格退化为 DAR**，Gate 1 的前提成立。

## 理论最优配置（`Qwen3GDARConfig.theory_preset()`）

```python
cfg = Qwen3GDARConfig(**base, **Qwen3GDARConfig.theory_preset())
# -> gate_param="deviation", update="objective", decay_ladder=64, gate/q/k 低秩 64
```

**全方位预设的每一项都是某个明确子问题的最优解**，且前提可测量。`models/test_theory.py` 逐条数值验证，**41/41 通过**。

| 组件 | 开关 | 最优性陈述 | 前提（已实测） |
|---|---|---|---|
| 恒等初始化 | `gate_param="deviation"` | `GDAR(0) ≡ DAR` **逐位精确** | — |
| 流更新 | `update="objective"` | `J(h')` 的**唯一闭式最小化点**（λ=0 精确退回 DAR，λ→∞ 正交投影） | J 严格凸 ⟺ **λ > −1** |
| 衰减 | `decay_ladder=64` | 多时间尺度幂律遗忘（cascade 模型），DAR 不可表示 | 学到的 τ 分布异质 |
| 地址 | `address="delta"` | 精确投影 + 串扰 ∝ `⟨k_i,k_j⟩`（pattern separation） | 存在干扰瓶颈 |
| 读·估计量 | `read_whiten="full"` | Mahalanobis 打分 = **BLUE/MMSE**（最小方差线性估计） | 源 Gram 非各向同性：实测 **key_cond ≈ 4e11** ✅ |
| 读·多头 | `read_heads=8` | 集成方差 `(1/H)Var+(1−1/H)Cov`，最优头投影互相正交 | 头间去相关：实测 **head_corr ≈ 0.10** ✅ |
| 读·弃权 | `read_null=True` | 零均值 GP 后验均值 ≡ Chow 最优弃权规则 | 存在该弃权的 token：实测 **null_mass ≈ 1/(1+N)** ✅ |
| 输出路由 | `DepthRead` | 纯读，无死参数 | 死参数 = 0 ✅ |

**前测量化内置**：`return_attn_res_stats=True` 会回传每个子层的 `key_cond` / `null_mass` / `head_corr` / `sharpness` / `entropy` / 三个门均值——"条件严格更优"的前提是**测量出来的**，不是假设的。

**① 精确恒等初始化（保证"不比 DAR 差"）**
门参数化成**相对 DAR 的零偏差**，而不是 `sigmoid(Linear)`：

```
gate_proj 权重 = 0，bias = 0（三个头）
decay_scale = erase_scale = write_scale = 0，read_scale = 0
decay = exp(-softplus(r_decay) * scale_d * tau_c)   -> 恰好 1
erase = softplus(r_erase) * scale_e                  -> 恰好 0
write = 1 + tanh(r_write) * scale_w                  -> 恰好 1
```

结果：`GDAR(0)` **整块**逐位等于 `prefix + delta`（实测 `max|diff| = 0.000e+00`；`update` 部分对两种更新式都成立，读取部分靠零初始化的 `read_scale` 一起为 0），也就是精确的 DAR。加上 `gate_proj` 权重为 0，恒等点还是**输入无关的平坦点**——优化器必须"学会离开它"，而不是被随机的第一步扰出去。而且在恒等点**三个偏差的梯度都不消失**（实测 decay 1e-3~1e-2、erase 7e-4~3e-3、write 1e-3~7e-3）。
读取门 `read_scale` 同样是可学习标量（初始 0，读取随训练淡入），它的梯度 `⟨∂L/∂routed, routed⟩` 在 0 点不消失。

**② 更新式 = 明确目标的闭式最优解（命题 1 / 命题 2）**

```
J(h') = 1/2 ||h' - m||^2 + (lambda/2) <khat, h'>^2 ,  m = decay*h + write*delta
h'    = m - lambda/(1+lambda) * khat * <khat, m>          <- 唯一最小化点
```

- λ=0 **精确**退化为 `decay*h + write*delta`（即 DAR）；λ→∞ 变成正交投影
- 验证：`∇J(h*) = 0`、`J(h*) < J(任意随机点)`、LBFGS 收敛到同一个解（λ=0/0.3/5 三档全过）
- 命题 2：λ→∞ 时**与 k̂ 正交的方向精确不变**（`<v,h'> = <v,m>`），`<k̂,h'> = 0`——最小范数修正
- 与 shensi 原式并存：`attn_res_update="shensi" | "objective"`，默认仍是 `shensi`（**旧参考式**）。
  ⚠️ 命名注意：`transformers@shensi` 分支**现在实现的是 `objective`**（闭式解），`shensi` 只表示更早的参考规则；别名 `reference` 等价于 `shensi`。

**③ 多时间尺度衰减阶梯（cascade 模型；DAR 表达不了的能力）**

`attn_res_decay_ladder=C` 给每个通道一个 τ，**初始**在 [1, τ_max] 上几何分布，然后作为 `nn.Parameter` 自由学习
（存 log τ，保证 τ>0——τ<0 会把遗忘变成放大），`decay_c = exp(−softplus(r_c)·τ_c)`。
初始时全部等于 1（恒等精确），一旦尺度离开 0，通道间就出现异质视界（实测通道跨度 0.707）。
阶梯的 C 个值被通道平铺共享，所以每个时间尺度是从 hidden_size/C 个通道的梯度里估出来的。
这是 GDAR 相对 DAR 唯一有理论支撑的差异化能力：DAR 的读只能给每个源一个标量权重，**无法让不同通道有不同的记忆视界**。

**④ 输出端改成纯读（`DepthRead`）**
参考模块的"读 + 写"在输出端没有 delta 可写，门/擦除方向全是死参数；改成纯读后死参数归零，参数也更省。

### 理论约束（不是工程细节）

**λ 必须 > −1**：`J` 的 Hessian 是 `I + λ·k̂k̂ᵀ`，特征值为 `1`（重数 D−1）与 `1+λ`，**只在 λ > −1 时严格凸**，且在 λ = −1 有极点。而参数化本身不保证学到的 erase 为正——AdamW 第一轮实测就把它推到负值区，随后**梯度全变 NaN**（125 个参数），训练直接崩。修法是在闭式解里把 λ 夹到 `[−0.5, ∞)`：保留一段可用的"反擦除"区间，同时永远待在严格凸域内。这条是**理论约束驱动的实现**，不是数值技巧；`test_theory.py` 的 T12 用 5 步 AdamW 做回归保护。

**白化矩阵必须 detach**：它是从源统计估计出来的**预条件子**（白化文献的标准做法），不能进反传——`torch.linalg.eigh` 在特征值重根时 backward 是 NaN，而这里的协方差恰好是"rank ≤ 源数 + ridge"的重根结构。

### 已修掉的两个真 bug（都是在本轮对照测试里抓到的）

1. **写门梯度恒为 0**：`write = 1 + tanh(r)·s`，恒等点 `tanh(0)=0` ⇒ `∂/∂s = 0`，写门**永远学不起来**。修法是给写头一个非零载体 bias（`attn_res_write_carrier_bias=-4`，`tanh(-4)=-0.999`），恒等仍精确但梯度存活。这类"零载体"陷阱已写进 `test_theory.py` 的覆盖范围。
2. **最后一层 MLP 输出被静默丢弃**：shensi 的流由 hyper-connection 承担，写入延到下一次调用；搬到纯残差流后最后一层没有"下一次"。改成 `read` / `update` 分离、每个子层输出**立即写入**（`AttentionResidual.read/update`），模型状态回到二元组。

## 默认值与"重做修复"

默认配置**逐项复现各仓库**（GDAR：`attn_res_gate_rank=None`、`attn_res_gate_init="paper"`、
`attn_res_q_rank/k_rank=None`）。重做方案的两处修复是开关：

```python
Qwen3GDARConfig(
    ...,
    attn_res_gate_init="identity",  # 门 bias 设 (b, -b, b)
    attn_res_gate_rank=64,  # gate_proj 低秩
    attn_res_q_rank=64,  # 路由 query 低秩
    attn_res_k_rank=64,
)  # erase 方向低秩
```

## 实测（`.venv/bin/python models/smoke_test.py`）

### 门初始化与恒等性（`AttentionResidual` 直接测）

| init / bias | decay | erase | write | 相对 `prefix+delta` 的每步偏差 |
|---|---|---|---|---|
| paper（仓库默认） | 0.4995 | 0.4990 | 0.4941 | **57.5%** |
| identity / 4 | 0.9787 | 0.0210 | 0.9789 | **3.2%** |
| identity / 8 | 0.9996 | 0.0004 | 0.9996 | **0.07%** |
| zero | 0.4990 | 0.5034 | 0.5073 | 57.2% |

→ 方案里"bias=+4 即 GDAR(0) ≡ DAR"只是近似（每子层 3.2%，多层累积）；
   要真正站住 Gate 1 建议 `attn_res_gate_init_bias=8.0`。
→ 另注意：erase 门必须**从关开始**（`(b,-b,b)`）。三个门都设 +4 会让 `erase=σ(4)=0.98`
   把残差流擦掉，恒等性不成立。

### 参数开销（每子层，r=64）

| 模型 | 子层数 | 仓库原样（q+k+3D² 全秩） | 占比 | 低秩 r=64 | 占比 |
|---|---|---|---|---|---|
| 0.6B | 56 | 0.29B | **48.9%** | 29.4M | 4.9% |
| 1.7B | 56 | 1.17B | **69.1%** | 58.7M | 3.5% |
| 4B | 72 | 2.36B | **59.0%** | 94.4M | 2.4% |
| 8B | 72 | 6.04B | **73.7%** | 151.0M | 1.8% |
| 14B | 80 | 10.49B | **70.8%** | 209.7M | 1.4% |
| 32B | 128 | 16.78B | **51.2%** | 335.5M | 1.0% |

⚠️ 方案里"低秩后 8B 门参数 <1%"只算了 `gate_proj`（3D²）。shensi 原样还有 `q_proj`、`k_proj`
各一个 D×D，合计每子层 **5D²**，8B 上就是 6.04B 参数（模型的 73.7%）。**低秩不是可选优化，
是能否训练的前提**；三者都降到 r=64 后才是 1–4.9%（且已并入 `attn_res_q_rank`/`attn_res_k_rank`）。

### 冒烟测试覆盖

`baseline / AR block / AR full / DAR sublayer / DAR block / GDAR 仓库默认 / GDAR 重做修复 /
GDAR identity bias8` 全部前向+反向通过，并回传路由统计
（`sharpness`、`entropy`、`n_sources`、`gate_decay/erase/write`）——E7 路由坍缩图可直接用。

## 尚未移植

* shensi 的 hyper-connection 多流（`ShensiHyperConnection`）——属于另一套机制。
* shensi 的 `block_write_layer` / `attn_res_block_layer_types` 分块写回——本目录用与 DAR
  一致的块源语义（快照差分），保证 GDAR-vs-DAR 只差门控。
* DAR 仓库的 V-stream 解耦注意力（`Qwen3AttnResAttention`，其 `delta_v` 变体）——未纳入，
  需要时可加。
* MoE / KDA 注意力——本目录只做 Qwen3 稠密基座。
