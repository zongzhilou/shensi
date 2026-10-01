# 白化的算子级实现（B6）

连接算子领先的一环是**白化读**（`read_whiten="full"`，论文主行）。`train/bench_connection.py`
量出来的口径是：主行整步 **351.0 ms**、关掉白化 **129.7 ms** —— 白化占整步 **≈63%**，是连接
里最大的一笔开销。这个目录就是冲这一笔的算子级实现，单独立项、可单独验证、可单独计时。

## 白化在算什么

参考实现（`models/megatron/gdar_connection.py::_whitening_transform`）三步：

```text
C = SᵀS / N                      # 协方差（GEMM）
W = (C + ridge·I)^{-1/2}         # eigh（LAPACK）→ rsqrt(clamp(λ)) → 组装
S_s = S @ W,  Q_s = Q @ W        # 两次应用（GEMM）
```

`ridge` 平移先加在协方差对角上，所以 `eig(A+ridge·I) = eig(A)+ridge`、`clamp` 不生效——
白化就是**精确的** `(C + ridge·I)^{-1/2}`，没有别的花样。这一步的性质让"换算法"是安全的：
只要算的还是同一个矩阵，语义不变。

## 这里的实现

| 文件 | 内容 |
|---|---|
| `whiten_ns.py` | 免 LAPACK 的逆平方根：幂迭代估 λmax → 多项式初值（相对误差拟合）→ 牛顿–舒尔茨抛光（**全是 matmul**）；`install()/uninstall()` 把它换进连接模块 |
| `whiten_triton.py` | Triton 协方差内核（`+ridge·I` 与除法融进去；精度 `ieee`/`tf32` 可选）+ 逐头协方差内核 |
| `bench_whiten_stages.py` | 白化的分段计时（先量：钱花在哪一步） |
| `bench_whiten.py` | 各实现的总计时 + 误差 + 折算到整步的收益 |
| `test_whiten.py` | 四类闸门：协方差、逆平方根、变换、端到端换入后 `read` 一致且可卸回 |

**没做的部分（如实）**：`values @ W` 那两次应用留给 cuBLAS —— 它们本来就是 GEMM，重写不会
更快；真正剩下的是把"协方差 + 逆平方根 + 应用"融成一个 kernel（省掉 S 的二次读与中间张量），
那是下一块砖。

## 怎么开

```bash
# 训练入口（默认 eager，不动原路径）
python train.py --config default --model-algo qwen3_gdar_main \
  --set train.system.gdar_whiten_impl=ns        # ns | triton

# 闸门与计时
python kernels/test_whiten.py
python kernels/bench_whiten.py --seq 2048 --sources 5 --hidden 1024
python kernels/bench_whiten_stages.py
```

## 实测（本机 RTX 5080 Laptop 16GB，fp32，2026-10-02）

### 白化到底占多少（真实训练路径，`bench_whiten_insitu.py`）

几何：19 层 × 1024 宽、micro-batch 1 × seq 1024、论文主行 spec（B=4、白化 full）。

| 量 | 数 |
|---|---|
| 白化调用次数 | **78 次 / 微步**（形状 `[1024, 2..6, 1024]`：每块位置 × 层各一次） |
| 单次耗时（eager，full） | **47–80 ms**（同一个操作孤立测见过 9.5 / 58 / 217 ms 三种值：随显存压力与时钟状态摆） |
| 白化合计 | **3.0–3.6 s / 微步**；整步 5.8 s ⇒ 占 **52–100%**（GDAR 主行在这一档基本就是白化） |
| eigh 占白化 | **99%**（孤立分段：cov 0.87 + eigh 217 + 组装 W 0.24 + 应用 1.10 ms） |
| 逐头档（`per_head`，上游默认） | 单次 **~1.1 ms**（8×128² 批量 eigh），比 full 便宜 50–70× |
| 整步对比（220M、32 微步/迭代、每档 1–2 迭代实测） | plain **5.7 s**、主行+full-eager **330–435 s**、主行+NS **138 s**、上游档+per_head **260 s** |

### 实现精度（`test_whiten.py`，12 条闸门全过）

| 闸门 | 结果 |
|---|---|
| Triton 协方差 vs torch | ieee **4.65e-06**；tf32 7.84e-04 |
| Triton 逐头协方差 | **0.00e+00**（逐位） |
| NS 逆平方根 vs eigh | ridge 1e-3：**4.33e-06**；ridge 1e-6（病态）：4.40e-06 |
| 变换 vs 参考实现（full） | 1.72e-06；Triton-ieee 2.52e-06；Triton-tf32 3.96e-04 |
| 逐头白化 vs 参考 | 1.40e-06 |
| **端到端**：把 NS 换进连接后 `AttentionResidual.read` | **0.00e+00（逐位）**；`uninstall()` 后逐位回参考 |

### 为什么"NS 替代 eigh"在本机没赢（如实）

* **FLOP 账**：eigh ≈ 10·d³ ≈ 5 个 matmul 当量；NS 到 1e-5 精度要 12–14 步 × 3 matmul ≈ **40 个当量**
  （残差自查每步再加 2 个）。NS 只在 **matmul 比 LAPACK 快一个量级以上** 时才划算。
* **本机的 matmul 速率**：1024³ fp32 **0.53 ms**（4.0 TF/s）、bf16 0.32 ms（6.8 TF/s）——小矩阵只有个位数
  TF/s（笔记本功耗与占用受限）；而 eigh 在 9.5–217 ms 之间摆。
* **实测**：NS 单次 **479 ms** vs eager **47–80 ms** ⇒ 本机慢 1.7–6×。数据中心卡上 bf16 matmul 高两个
  数量级、eigh 不变 ⇒ 同一实现预期反超 4–8×（交叉点：`matmul 速率 / LAPACK 速率 ≳ 10`）。
* **换成 `per_head` 也救不了整步**（本来以为可以）：它的白化单次只要 ~1.1 ms，但整步仍有 **260 s**
  ⇒ 剩下的 8 s/微步 **不是白化**，是连接其余部分在 micro-batch 1 下的 fp32 往返与访存
  （`[1024, 2..6, 1024]` 的多次 `.float()`、einsum 读、以及 15.2 GB 常驻下的分配器压力）。
* **NS 在本机反超 eigh（138 s vs 330–435 s）不是因为 FLOP 更省**（NS 的 FLOP 是 eigh 的 ~8 倍），
  而是**躲开了 LAPACK 在显存压力下的分配抖动**——同一句 `eigh` 在 9.5 / 58 / 217 ms 之间摆就是证据。
* **本机连接的单价**：每 token 8–13 ms（背骨干 0.18 ms）⇒ 连接比骨干贵 **50–70×**（micro-batch 1）。
  对照：`train/bench_connection.py` 在 batch 2、8 层时连接增量只占整步 4.74× ⇒ **生产档别用
  micro-batch 1 跑这个连接**，这一条比任何单点内核都值钱。

### 下一步（写清楚，不装完成）

1. **批量 eigh**：78 次/微步 是"每次一个 1024²"；把同形状的调用攒起来做 batched eigh（一次 kernel、
   一份 workspace）预期能拿回大部分开销——要动连接的调用结构（白化请求排队后统一处理），没做。
2. **带自适应谱带的迭代**（polar-express 式分步系数）把 NS 从 ~14 步压到 4–5 步：本机的多项式初值
   在 4 个数量级的谱上拟合不出来（已实测失败），正确做法是分步 minimax 系数表。
3. `per_head` 的逐头分支内联在 `_depth_read` 里，还没接开关（本目录只给了等价性闸门）。


## 边界

- 换算法换的是**同一数学**（`(C+ridge·I)^{-1/2}`）：`GDAR(0)` 的恒等路径不受影响（它的
  `read_whiten` 是 `off`），恒等闸门与权重对拍照旧逐位。
- `tf32` 档只动白化的 GEMM 精度（`max rel` 见上表）；要给论文出的数，用 `ieee`。
- `per_head`（上游默认档）的逐头协方差已有 Triton 内核与等价性闸门，但连接里的逐头分支是
  内联在 `_depth_read` 里的，**还没接上开关** —— 接它要把那个分支也换成可替换实现（下一步）。
