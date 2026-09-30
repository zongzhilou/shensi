# src/ 的布局与对接方式

```text
src/
└── shensi/           包本体
    ├── runtime.py    与 Megatron-Bridge / verl 对接的唯一入口
    ├── utils/        ckpt 摘要、AdEMAMix 注册适配
    └── recipes/      训练配方
```

判据一句话：**给上游库提一个"支持 Shensi"的最小 PR，那个 PR 里会有的东西放上游的子模块里
（模型实现在 Megatron-Bridge 的 `src/megatron/bridge/models/shensi/`）；我们自己的工具、机制、
配方放 `shensi/`。**

## 模型实现去哪了

Shensi 的模型、层规格、mHC / AttnRes / MoE、HF↔Megatron 的桥都在 Megatron-Bridge 里，按官方文档
`docs/adding-new-models.md` 贡献：

```text
3rdparty/common/Megatron-Bridge/src/megatron/bridge/models/shensi/   ← 模型与 bridge（本仓那份检出切在 shensi 分支）
3rdparty/common/Megatron-Bridge/tests/{unit,functional}_tests/.../shensi/
```

mcore 不改（上游 main 自带 `set_default_log_ranks`、`get_backend`、grouped-mxfp8 那套符号，
`dsv4_hybrid` / CSA / DCA 这些注意力变体也是上游的）。

## shensi.runtime：第三方要的东西在这里登记

`import shensi.runtime` 即生效（顺序有讲究，见 `setup()` 里的注释）：

| 做的事 | 为什么 |
| --- | --- |
| 补两个 mcore main 上已删掉的模块（`strategies.async_utils` / `filesystem_async`） | verl 的 v012 兼容层在**版本守卫之前**就 import 它们，装着 mcore main 时 import verl 直接 `ModuleNotFoundError`（守卫永远走不到） |
| 把 `mcore_fsdp_adapter.FullyShardedDataParallel` 从工厂函数换成 V1 类 | mcore main 明确写了"type check 请用 V1/V2 类"，而这份 verl 把它当类用（`megatron_FSDP \| DDP`、类名元组）——`function \| type` 会 TypeError |
| 注册 `nvidia_noipc` CUDA 平台 | WSL2 上跨进程 CUDA IPC 不可用；verl 的 engine 模块一 import 就查 `VERL_PLATFORM`，所以这步要最先做 |
| 导入 `megatron.bridge.models.shensi`、`shensi.utils.optimizer.ademamix` | 这两张注册表都是"导入即注册"（Bridge 的 HF↔Megatron 桥表、`emerging_optimizers` 的标量优化器表） |
| 没装 DSA 融合内核时把 Bridge provider 的 `dsa_kernel_backend` 默认改成 `none` | mcore 给 `dsv4_hybrid` 的默认是 `cudnn`（要 `flash_mla`）；没有内核时构造配置就抛错。训练入口走 `args.py` 的同名回退 |
| 补 verl 的 flat-buffer 判空 | verl 在 `use_distributed_optimizer=False` 时无条件解引用 `param_data`（同一文件里上游自己判过空） |

配方入口会显式 import 它；verl 侧按 verl 自己的约定用
`VERL_USE_EXTERNAL_MODULES=shensi.runtime`，于是每个 import verl 的进程（driver、ray worker、
vLLM server）都会走一遍。

## 装环境（本机的实际顺序）

```bash
uv lock                                    # 解算清单（284 个包，验证 pyproject/source 成立）

# uv sync 是 exact 的：不在 lock 里的包会被卸掉，所以先 sync、后补那三个"本机另编"的
uv sync --no-install-package vllm --no-install-package transformer-engine \
        --no-install-package fast-hadamard-transform
uv pip install --no-deps -e /path/to/Megatron-Bridge            # 本地那份 Bridge（含 models/shensi）
VLLM_VERSION_OVERRIDE=0.30.1rc0.dev360+g54c5060a1 \
  uv pip install --no-build-isolation --no-deps 3rdparty/common/vllm   # vllm 现场编
# TransformerEngine 与 fast-hadamard-transform 见下面「本机补丁」与小节说明
```

四处绕行，都是上游打包 + 本机环境决定的，不是我们的偏好：

- **`megatron-bridge` 不进 `[tool.uv.sources]`**：它自己的 `pyproject.toml` 里
  `megatron-core = { path = "3rdparty/Megatron-LM/" }` 指到一个未初始化的子模块目录，uv 解算
  Bridge 的元数据时会直接失败。所以 Bridge 单独用 `--no-deps` 装。
- **vllm 要现场编**：隔离构建时它的 CMake 会去调一个已经不存在的构建环境 `bin/ninja`
  （uv 每次构建用新的临时环境，而 CMake 缓存里钉着上一次的路径）——先 `rm -rf
  3rdparty/common/vllm/build` 清缓存，再按上面那行用 `--no-build-isolation` 编（venv 里有
  cmake/ninja/setuptools-rust）。清单里也把构建期依赖列进了
  `[tool.uv.extra-build-dependencies] vllm`。
- **TransformerEngine**：清单按 Megatron-Bridge 钉的 rev 走 git 源码（`NVTE_WITH_NCCL_EP=0`，
  因为 uv 的 git checkout 不会 init 它的 `nccl_ep` 子模块）。本机实测**从源码编 >90 分钟没编完**
  （单机笔记本），venv 里目前是 TE-FL 那份二进制，全部闸门都是在它上面跑通的。
- **`transferqueue` / `tile-kernels`**：`uv sync` 会把它们当"不在 lock 里"卸掉（它们是 FL 时代
  手工装的）。跑 RL 前如果报缺，按 `verl` 的 `verl-core` extra 补装。

## 本机补丁

`patches/` 只放"上游二进制包在本机编不过"的补丁，用的时候 clone 上游源码、打补丁、本地编，
不打进 `3rdparty/`：

- [`patches/fast-hadamard-transform-sm120.patch`](../patches/fast-hadamard-transform-sm120.patch)：
  上游 `setup.py` 的 `-gencode` 列表没有 SM120，RTX 50 系上 DSA indexer 的 Hadamard 旋转会
  `no kernel image is available`。用法：

  ```bash
  git clone https://github.com/Dao-AILab/fast-hadamard-transform /tmp/fht
  git -C /tmp/fht checkout f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
  git -C /tmp/fht apply patches/fast-hadamard-transform-sm120.patch
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps --no-cache /tmp/fht
  ```

  （`uv sync` 会按清单里那个 git rev 重装成没打补丁的版本，SM120 上要再走一遍这三行。）

（另有两条不改文件的运行时开关，写在 `recipes/shensi/train/launcher.py` 里：把本 venv 的 `bin`
放到 `PATH` 最前，让 mcore 的 `core/datasets/Makefile` 用本 venv 的 `python3` 现场编 helper；
`TE_FL_PREFER=vendor` 绕开 FlagGems 的 flagos 后端在 SM120 上的段错误。）

## 两条硬规矩

1. **只新增，不覆盖**：本仓不出现与上游同路径的文件，也不往 `3rdparty/` 里写文件——上游需要的行为差异
   一律通过 `shensi.runtime` 在运行时表达（`patches/` 是唯一的例外：只对本机编不过的第三方源码包用，
   且从不落进 `3rdparty/`）。
2. **上游库按原样 import**：`import megatron` / `import megatron.bridge` / `import verl` 拿到的就是
   `3rdparty/` 里那份（mcore 与 Bridge 按可编辑方式装），我们不接管它们的命名空间。
