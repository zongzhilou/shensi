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
| 导入 `megatron.bridge.models.shensi`、`shensi.utils.optimizer` | 导入即注册：Bridge 的 HF↔Megatron 桥表；优化器口径（**AdaMuon 矩阵腿 + AdEMAMix / GrokFastAdamW 标量腿**：标量腿扩展、checkpoint 状态键、`emerging_optimizers` 名字表） |
| 没装 DSA 融合内核时把 Bridge provider 的 `dsa_kernel_backend` 默认改成 `none` | mcore 给 `dsv4_hybrid` 的默认是 `cudnn`（要 `flash_mla`）；没有内核时构造配置就抛错。训练入口走 `args.py` 的同名回退 |
| 补 verl 的 flat-buffer 判空 | verl 在 `use_distributed_optimizer=False` 时无条件解引用 `param_data`（同一文件里上游自己判过空） |

配方入口会显式 import 它；verl 侧按 verl 自己的约定用
`VERL_USE_EXTERNAL_MODULES=shensi.runtime`，于是每个 import verl 的进程（driver、ray worker、
vLLM server）都会走一遍。

## 装环境（本机的实际顺序）

```bash
uv lock                                    # 解算清单（284 个包，验证 pyproject/source 成立）

# uv sync 是 exact 的：不在 lock 里的包会被卸掉，所以先 sync、后补这几个"本机另编/另装"的
uv sync --no-install-package vllm --no-install-package transformer-engine \
        --no-install-package fast-hadamard-transform
uv pip install --no-deps -e /path/to/Megatron-Bridge            # 本地那份 Bridge（含 models/shensi）
uv pip install transferqueue                                    # ver-core extra 里那个（RL 要用）
# 然后编 vllm 与 hadamard（见下），TE 见下
```

**vllm（本机实测可用的编法，约 40 分钟）**：

```bash
rm -rf 3rdparty/common/vllm/build          # 清掉被污染的 CMake 缓存（见下）
VLLM_VERSION_OVERRIDE=0.30.1rc0.dev360+g54c5060a1 MAX_JOBS=8 NVCC_THREADS=1 \
  uv pip install --no-build-isolation --no-deps --reinstall 3rdparty/common/vllm
```

四处绕行，都是上游打包 + 本机环境决定的，不是我们的偏好：

- **`megatron-bridge` 不进 `[tool.uv.sources]`**：它自己的 `pyproject.toml` 里
  `megatron-core = { path = "3rdparty/Megatron-LM/" }` 指到一个未初始化的子模块目录，uv 解算
  Bridge 的元数据时会直接失败。所以 Bridge 单独用 `--no-deps` 装。
- **vllm 的版本号必须带 `VLLM_VERSION_OVERRIDE`**：fork 只有分支没有 tag，vcs-versioning 会算出
  `0.1.devNNNN+g<sha>`，而 verl 的闸门要求 ≥0.18（`verl/third_party/vllm/__init__.py` 读的是
  dist 元数据）。不带 override 编出来的 wheel 装上去，RL 会直接 `ValueError: vllm version ... not supported`。
- **vllm 的 CMake 缓存坑**：隔离构建时它的 CMake 会去调一个已经不存在的构建环境 `bin/ninja`
  （uv 每次构建用新临时环境，而 CMake 缓存钉着上一次的路径），所以先删 `build/`、再用
  `--no-build-isolation` 编（venv 里有 cmake/ninja/setuptools-rust），并且 `MAX_JOBS` 压到 8
  （24 个并行 nvcc 把 WSL 的 47G 虚拟机打到重启过）。清单里也把构建期依赖列进了
  `[tool.uv.extra-build-dependencies] vllm`。
- **TransformerEngine**：清单按 Megatron-Bridge 钉的 rev 走 git 源码（`NVTE_WITH_NCCL_EP=0`，
  因为 uv 的 git checkout 不会 init 它的 `nccl_ep` 子模块）。本机实测**从源码编 >90 分钟没编完**
  （单机笔记本），venv 里目前是 TE-FL 那份二进制，全部闸门都是在它上面跑通的。
- **`transferqueue` / `tile-kernels`**：`uv sync` 会把它们当"不在 lock 里"卸掉（`verl-core` 的
  extra 没被请求）。跑 RL 前按上面第 2 行补装。

## 装环境（昇腾 / NPU 机）

清单是 `pyproject.ascend.toml`（与 NVIDIA 侧的差别只有依赖集、index/source、组件安装顺序三处；
两处相同的取向照旧：**上游 Megatron-LM 用 main**、mcore/Bridge 按可编辑方式装）。

### 依赖清单

| 层 | 要什么 | 出处 |
| --- | --- | --- |
| 硬件 | Atlas A2/A3 训练卡（组件文档的实测环境是 Ascend 950DT） | 各组件 README |
| 驱动/固件 | 与 CANN 版本配套的那一套 | CANN 版本配套表 |
| CANN | 9.2.0，含 Ascend C、Bisheng 编译器、HCCL、HCOMM 的头与库 | DeepEP-Ascend README 明确列了这些 |
| 系统 | Linux + Python **3.12**（TransformerEngineNPU 要 >=3.12，本仓 `requires-python` 也是 3.12） | TE-NPU `pyproject.toml` |
| torch | CPU wheel（`pytorch-cpu` index）+ `torch_npu` **严格配对**：组件文档里一组可用组合是 torch 2.13.0+cpu ↔ torch_npu 2.13.0rc1；MegatronAdaptor 的表写的是 CANN 9.2.0 ↔ torch_npu 26.2.0 | DeepEP / MegatronAdaptor README |
| 适配层 | `MegatronAdaptor`（让 mcore 在 NPU 上跑）、`TransformerEngineNPU`（TE 的 NPU 后端） | 组件 README |
| mcore 侧补丁 | `MindSpeed`（对 mcore 打补丁）、`MindSpeed-Ops`（Triton-Ascend 融合算子，自带 `triton-ascend==3.2.2` 约束） | MindSpeed / MindSpeed-Ops README |
| 可选内核 | `DeepGEMM-Ascend`、`DeepEP-Ascend`（EP / GEMM 走 DeepSeek 昇腾原生内核时） | 两者 README |
| 训练/推理栈 | `megatron-core`(main) · `megatron-bridge`(本仓检出) · `verl` · `verl-hardware-plugin` · `vllm` · `vllm-ascend` · `transformers` | 同 NVIDIA 侧 |
| 其它 | `emerging-optimizers` · `pytorch-optimizer`（Muon/AdEMAMix）· `ray[default]` · `omegaconf` · `pyarrow` · `zstandard` · `pybind11`/`ninja`/`cmake` | 同 NVIDIA 侧 |
| **不要装** | `fast-hadamard-transform`（CUDA 扩展）、`flashinfer-python`（CUDA 专用） | 见下面的说明 |

### 从零开始的步骤

```bash
# 1) 代码与子模块（mcore / Bridge / verl / vllm / transformers + 昇腾四件套 + 两个 DeepSeek 内核）
git clone <本仓> shensi && cd shensi
git submodule update --init --recursive

# 2) CANN 环境（提供 ASCEND_HOME_PATH；组件的 setup.py 都读它）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 3) venv + 上游依赖（清单换成昇腾侧那份）
cp pyproject.ascend.toml pyproject.toml
uv venv --python 3.12 && uv sync \
    --no-install-package vllm --no-install-package mindspeed-ops   # 这两个要本地编（见下面第 4 步）

# 4) 昇腾组件：顺序不能反（适配层 → mcore → 打补丁的 MindSpeed 系）
uv pip install -e 3rdparty/ascend/MegatronAdaptor
uv pip install -e 3rdparty/ascend/TransformerEngineNPU --no-build-isolation
uv pip install -e 3rdparty/common/Megatron-LM                 # 上游 main
uv pip install -e 3rdparty/common/Megatron-Bridge --no-deps    # shensi 模型在它里面
uv pip install -e 3rdparty/ascend/MindSpeed
uv pip install -e 3rdparty/ascend/MindSpeed-Ops --no-build-isolation --no-deps \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

# 5) 两个可选内核（要 EP/GEMM 原生内核时；DeepGEMM-Ascend 的 install_requires 里还带 tilelang）
uv pip install --no-build-isolation 3rdparty/ascend/DeepGEMM-Ascend
uv pip install --no-build-isolation 3rdparty/ascend/DeepEP-Ascend

# 6) 装配自查：一句话把整条链 import 一遍（缺哪层就报哪层）
#    包名按各组件的顶层包目录：TransformerEngineNPU 是 drop-in，装出来的就是 transformer_engine；
#    `megatron.plugin` 由 MindSpeed 提供——mcore 侧补丁生效的判据就是它能不能 import
python -c "import torch, torch_npu, transformer_engine; \
           import megatron.core, megatron.bridge, megatron.plugin; \
           import megatron_adaptor, mindspeed, mindspeed_ops, verl, vllm; \
           print('ok, npu count =', torch.npu.device_count())"
#    可选内核装了就再补两句：import deep_gemm / import deep_ep

# 7) 本地产物 + 环境变量（与 NVIDIA 侧同一套）
export SHENSI_FS=/home/<user>/fsdata            # 数据/ckpt/模型 的根
export SHENSI_TOKENIZER=$SHENSI_FS/shensi/models/tiny-tok
python -m shensi.recipes.shensi.tiny_artifacts --out $SHENSI_FS   # tiny-tok / tiny-rl

# 8) 逐段跑（命令与 NVIDIA 侧一样；昇腾上 RL 侧的平台名不同，见下）
cd src/shensi/recipes/shensi
python -m shensi.recipes.shensi.stage0_pretrain.stage1_pretrain.data_prep --prepare
```

逐段的入口与档位见各 stage README；九段的 `test_train.py` 是同一套极小档门（跑通链路用），
正式训练用各段的 `train.py --profile default`（大档）。RL 四段在昇腾上要设：

```bash
export VERL_PLATFORM=huawei                      # verl 的 NPU 平台名（platform_npu.py 里注册的就是它）
export VERL_USE_EXTERNAL_MODULES=shensi.runtime
# 注意：NVIDIA 侧那两条本机专有开关在这里**不要**用：
#   VERL_PLATFORM=nvidia_noipc 是 WSL2 没有 CUDA IPC 的绕行；VLLM_PLUGINS="" 是 NVIDIA fork 的绕行
```

### 昇腾侧的五处已知差异

1. **DSA indexer 的 Hadamard 旋转**：`fast-hadamard-transform` 是 CUDA 扩展，昇腾上编不了；MindSpeed-Ops
   里也没有现成的 Hadamard（在 `3rdparty/ascend/MindSpeed-Ops` 里搜过）。它由 mcore
   `dsa.py::rotate_activation` 使用，开关是 `dsa_indexer_rotate_activation`（默认 True）。三条路：
   ① 自备一个 NPU Hadamard 挂到 mcore 的 import 名上；② 层计划只用不需要旋转的层（`shensi_compress_ratios`
   里去掉 4、只用 128 / sliding——stage1_pretrain 的档就是这种写法改一份）；③
   `--dsa-indexer-rotate-activation false` 关掉旋转，能跑但**数值口径与参考不同**，只适合链路/吞吐对照。
2. **MindSpeed 与 mcore 的版本配对**：MindSpeed 官方配的是 `core_v0.12.1`，本仓装的是上游 main
   （两侧统一口径）。若它的补丁对 main 不生效，按 MindSpeed 的分支表换 mcore 版本，或在本仓把 mcore
   钉到它支持的 rev——这步必须在 NPU 机上实测确认。
3. **numpy**：MindSpeed 的 requirements 写 `numpy<=1.26.0`，而 verl/vllm 这条线要 numpy 2.x；清单不写死，
   装完按第 6 步逐个 import 验，真撞上再按 MindSpeed 的约束调。
4. **不要装 CUDA 专属包**：`flashinfer-python` 若被 vllm 的解析拉进来，用
   `uv sync --no-install-package flashinfer-python` 排掉（它只在 CUDA 上编译/加载）。
5. **未上 NPU 实测**：本仓库没有昇腾机器，上面这份清单与步骤是按四个组件的 README、它们的
   `requirements.txt`/`pyproject.toml` 以及本机对子模块源码的检查整理的，命令没有在 NPU 上跑过。
   昇腾相关的“登记未接”项（含本条的边界）也在包根 README 的「环境与已知限制」里。

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
