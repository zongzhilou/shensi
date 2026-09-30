# src/ 的布局与对接方式

```text
src/
├── megatron_ext/     mcore / Megatron-Bridge 侧增量（AttnRes、mHC、MoE、Shensi 模型与 bridge 等必要实现）
├── flagscale_ext/    FlagScale 侧增量（训练入口 + 示例配置）
└── shensi/           包本体
    ├── runtime.py    与第三方对接的唯一入口
    ├── utils/        ckpt 摘要、优化器适配
    └── recipes/      训练配方
```

判据一句话：**给上游库提一个"支持 Shensi"的最小 PR，那个 PR 里会有的东西放 `*_ext`；
我们自己的工具、机制、配方放 `shensi/`。**

## megatron_ext / flagscale_ext：按上游路径摆放，只新增

```python
from megatron_ext.core.models.shensi import ShensiModel          # 直接导入，不走任何接管机制
from megatron_ext.bridge.models.shensi import ShensiBridge
```

```text
megatron_ext/core/transformer/shensi/...   ↔  megatron/core/transformer/shensi/...
megatron_ext/core/models/shensi/...        ↔  megatron/core/models/shensi/...
megatron_ext/bridge/models/shensi/...      ↔  megatron/bridge/models/shensi/...
flagscale_ext/train/megatron/...           ↔  FlagScale/flagscale/train/megatron/...
```

`flagscale_ext/train/megatron/train_shensi.py` 由 FlagScale 的 runner 按 experiment 配置里的
`entrypoint` 启动；配置里写的是**本仓的绝对路径**（`${oc.env:SHENSI_ROOT}/src/flagscale_ext/...`），
所以不需要往 FlagScale 树里拷文件。

## shensi.runtime：第三方要的东西在这里登记

`import shensi.runtime` 即生效，做四件事（都只做加法）：

| 做的事 | 为什么 |
| --- | --- |
| 登记 `megatron.core.transformer.mla_qk_norm_config`、`megatron.training.models.gpt` 等模块 | Bridge / verl 按这些名字 import，而 FL fork 里没有 |
| 导入 `megatron_ext.bridge.models.shensi`、`shensi.utils.optimizer.ademamix` | 这两张注册表都是"导入即注册"（Bridge 的 bridge 表、`emerging_optimizers` 的标量优化器表） |
| `megatron_ext.core.backfill.apply()` | FL fork 缺的 mcore-main 符号（`set_default_log_ranks`、FSDP 的 V1/V2 别名、`get_backend`、grouped mxfp8 的判空原语） |
| 两个运行时补丁 | verl 在 `use_distributed_optimizer=False` 时的 flat-buffer 判空；WSL2 的 no-IPC CUDA 平台 |

配方入口会显式 import 它；verl 侧按 verl 自己的约定用
`VERL_USE_EXTERNAL_MODULES=shensi.runtime`，于是每个 import verl 的进程（driver、ray worker、
vLLM server）都会走一遍。

## 两条硬规矩

1. **只新增，不覆盖**：`src/*_ext` 里不许出现与上游同路径的文件（要改上游行为就 `import` + 最小子类/包装）。
   例外只有 `tools/ext_invariant.py` 里 `BACKFILL` 登记的几处，每处写明"为什么必须回填"。
   `src/megatron_ext/__init__.py`、`src/flagscale_ext/__init__.py` 只是我们这两棵树的包标记，
   不落盘、也不覆盖上游的包入口。检查：
   `python3 tools/ext_invariant.py`（逐个比对 `3rdparty/common/*` 的 HEAD，有同名文件就报错退出）。
2. **不往 3rdparty 里写文件**：`3rdparty/` 只是安装环境（git 子模块，只克隆）。上游需要的行为差异
   一律通过 `shensi.runtime` 在运行时表达。
