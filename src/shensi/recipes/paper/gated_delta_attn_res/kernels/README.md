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
| `whiten_fused.py` | **融合读**：`values/query @ W` + 归一化点积 + softmax₁/softmax + 加权和，一次 Triton kernel（把参考实现的 ~15 个 torch 算子压成 3 次 kernel） |
| `whiten_per_head.py` | `per_head` 档（上游默认）的白化换成 Triton 逐头协方差 + NS，`install()` 换进 `_depth_read` |
| `whiten_batched.py` | **批量 eigh**：把多次白化的协方差叠成 [B, d, d] 一次解（含"能不能接上"的实测证据） |
| `test_whiten.py` / `test_whiten_extra.py` | 闸门：协方差、逆平方根、变换、融合读、per_head 开关、批量白化、端到端换入/卸回 |
| `bench_whiten{,_stages,_insitu}.py` | 三把尺子：孤立分段、各实现对比、**真实训练路径上的计量** |

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

### 三件新件与本轮实测（2026-10-02，本机 RTX 5080 Laptop 16GB）

**撤回一条**：上一版 README 写过"把白化换成 NS 后整步 330–435 s → 138 s（3.14×）"——那次跑的
NS 是**坏版本**（多项式初值发散，损失已经是 NaN，日志里 `lm loss: nan` 被我的正则漏掉了）。
修好迭代之后重测：**本机 eigh 更快**（下表），所以那条读数作废，按下面的表为准。

**白化/读的孤立计时**（`bench_whiten.py`，values [1024, 5, 1024] fp32，10 次取中位）：

| 实现 | ms | 相对参考 | 误差 |
|---|---|---|---|
| 变换：eigh（参考实现） | **11.26** | 1.00× | — |
| 变换：NS（无 LAPACK） | 33.02 | 0.34× | 4.4e-06 |
| 变换：Triton 协方差 + NS（ieee/tf32） | — | — | **NS 残差 8.9e-06 > 容差 ⇒ 当场拒绝**（fail-loud） |
| 读：参考（~15 个 torch 算子） | 11.19 | 1.00× | — |
| 读：**融合读**（Triton，3 次 kernel） | **11.08** | **1.01×** | 2.5e-06 |
| per_head 档：参考（逐头 eigh） | 1.88 | 1.00× | — |
| per_head 档：Triton 逐头协方差 + NS | 15.88 | 0.12× | 见闸门（读级 1.2e-05） |
| 白化 39 次请求：逐个 | 388.8 | 1.00× | — |
| 白化 39 次请求：**一次批量 eigh** | **204.4** | **1.90×** | 0.0 |

**结论（如实）**：在这张笔记本卡上，**没有任何内核改写能在孤立计时里赢过参考实现的白化/读**
（eigh 11.3 ms、融合读 1.01×、per_head NS 0.12×）——因为这些 shape 上 cuBLAS/cuSOLVER 本来就够快，
而 in-training 的 47–80 ms/次是**上下文**造成的（15.2 GB 常驻下的分配抖动 + 时钟/功耗状态，
同一句 eigh 在 9.5 / 58 / 217 ms 之间摆）。真正拿到手的收益只有两处：
**批量 eigh 1.90×**（要调用方把请求交出来）与 **融合读的等价性**（1.01×，留给数据中心卡当积木）。

### 精度包线（开 `ns`/`triton` 前必读）

| 数据 | κ(A) | NS（40 步）W 相对误差 |
|---|---|---|
| 各向同性随机 | ~4e3 | 2e-05 级（λmax 幂迭代步数足够时） |
| 各向异性 ×10² | ~1e5 | 2e-01 级 |
| 各向异性 ×10⁴ | ≳1e8 | 2e-01 ~ 9e-01 |

* `full` 档的协方差由真实隐状态而来，条件数大 ⇒ **NS 不是等价实现**（`inv_sqrt_ns` 收敛不了会抛
  `NSNotConverged`；训练路径 `whitening_transform_ns` 只警告一次，不静默）；
* `per_head` 档（上游默认，dh×dh = 128²）条件数好得多，NS 等价（闸门：读级 1.2e-05，`attention`
  整条 read 逐位相同）；
* 想把 full 档也做出等价加速，得走"自适应谱带的分步系数"（polar-express 式，把迭代压到 4–5 步，
  每步的 minimax 系数按当前谱带拟合）——没做，写在下一步。
### 下一步（写清楚，不装完成）

1. **批量 eigh 接线**：API 与收益（1.90×）都有了，但当前深度递推是**逐层串行**的（实测调用序列：
   `S=2 ×8, S=3 ×8, S=4 ×8, S=5 ×8, S=6 ×7`，8 次同形状来自同一深度组的 4 层 × 2 子层，层间严格依赖），
   要批就得让连接把"收集请求 → 批量白化 → 回填"拆成两段——会动移植件的调用结构，没做。
2. **自适应谱带的迭代**：把 full 档的 NS 从 40+ 步压到 4–5 步（分步 minimax 系数表）。
3. **数据中心卡的复测**：融合读（1.01×）与批量 eigh（1.90×）在 kernel 启动/CPU 开销占比高的机器上
   收益会放大；`bench_whiten*.py` 就是那台机器上的同一把尺子。
4. `per_head` 的逐头分支现在**接了开关**（`--gdar-whiten-impl per_head`）；`full` 的融合读也接了
   （`fused`，与参考同源、只融读）。两者都已过闸门。
