# src/ 的布局与增量机制

```text
src/
├── shensi/            包本体（安装成 site-packages 里的 shensi）
│   ├── recipes/       各阶段配方与配置
│   └── utils/         我们自己的工具与机制（ckpt 摘要、优化器适配）—— 不属于"适配某个库"
├── megatron_ext/      shensi 对 Megatron / Megatron-Bridge 的适配增量（含 AttnRes / mHC / MoE / layer 等必要实现）
└── flagscale_ext/     shensi 对 FlagScale 的适配增量（训练入口 + 示例配置）
```

判据一句话：**如果给某个上游库提一个"适配 shensi"的最小 PR，那个 PR 里会有的东西就放对应的 `*_ext`；
其余（我们自己的工具、机制、脚本）放 `shensi/utils`。**

## megatron_ext：直接导入，没有任何接管机制

```python
from megatron_ext.core.transformer.shensi.attn_res import ShensiAttentionResidual
from megatron_ext.core.models.shensi import ShensiModel
```

目录按上游包路径摆放，所以"这个文件对应上游哪个位置"一眼可见：

```text
megatron_ext/core/transformer/shensi/...   ↔  megatron/core/transformer/shensi/...
megatron_ext/core/models/shensi/...        ↔  megatron/core/models/shensi/...
megatron_ext/bridge/models/shensi/...      ↔  megatron/bridge/models/shensi/...
megatron_ext/training/models/...           ↔  megatron/training/models/...
```

## flagscale_ext：由 finder 挂到 `flagscale.*`

FlagScale 的运行器按 `flagscale.*` 导入入口（配置里 `entrypoint: flagscale/train/megatron/train_shensi.py`），
所以 `import shensi` 会装一个 meta-path finder，把 `flagscale_ext/**` 当 `flagscale.**` 提供。
另外**入口文件必须真的在 FlagScale 树里**（运行器按文件路径起进程）：

```bash
shensi apply-ext --root <含 FlagScale/ 的目录>     # 通常 --root <工作区>/3rdparty/common
```

它按 `LAYOUT` 落盘：`flagscale_ext/**` → `FlagScale/{flagscale,examples}/**`，
以及 `megatron_ext/tests/**` → `Megatron-Bridge/tests/**`（我们的 Bridge 侧测试，跑 pytest 前落一次）。
megatron 侧的**代码**不落盘——直接 `import megatron_ext` 即可。

多进程那一侧由 `shensi.activate(env)` 兜住：配方起子进程（`flagscale.run`、torchrun、verl、`vllm serve`）时，
它在临时目录放一个只有 `import shensi` 的 `sitecustomize.py` 并挂到子进程 `PYTHONPATH` 最前，于是每个新进程
一开始就把 flagscale 侧的接管装上。

上游库本身在 `3rdparty/` 下（git 子模块，只克隆、不修改）：`common/` 两平台共用，`nvidia/`、`ascend/` 放平台专属；
`vllm` / `transformers` 的 Shensi 实现直接在被引用的分支里（`zongzhilou/vllm@shensi`、`zongzhilou/transformers@shensi`）。

## 两条硬规矩

1. **只新增，不覆盖**：`src/*_ext` 里不许出现与上游同路径的文件（要改上游行为就 `import` + 最小子类/包装）。
   检查：`python3 tmp/others/ext_invariant.py`（逐个比对 `3rdparty/common/*`，有同名文件就报错退出）。
   这条是踩出来的：早先 `csa.py` 是同名覆盖件，把上游代码冻在旧版本上，任何 lint 都抓不到。
2. **导入顺序**：`import shensi` 要早于 `import flagscale`（finder 只在包还没进 `sys.modules` 时接管）。
   `src/flagscale_ext/train/megatron/train_shensi.py` 顶部就做了 `import shensi`（顺带注册 AdEMAMix）。
