# Shensi 训练配方

从零训练 Shensi 的完整链路：模型取 **DeepSeek-V4-Flash 的 text_model 架构**（CSA + HCA 混合注意力、mHC
多流超连接、Muon、1M 上下文，[2606.19348](https://arxiv.org/abs/2606.19348)），训练课程对齐 **GLM-5 / 5.1 / 5.2 / 5.3**
（[报告](https://arxiv.org/abs/2602.15763)、[5.1](https://z.ai/blog/glm-5.1)、[5.2](https://z.ai/blog/glm-5.2)、
[5.3](https://z.ai/blog/glm-5.3)），数据用法参考 **Nemotron-3**（[Nano](https://arxiv.org/abs/2512.20848)、
[Super](https://arxiv.org/abs/2604.12374)、[Ultra](https://arxiv.org/abs/2606.15007)）。

## 1. 阶段划分

```text
stage0_pretrain/stage1_pretrain   预训练①：主预训练，稠密主干（csa_dense_mode=true），4K → 8K，27T 量级
stage0_pretrain/stage2_midtrain   预训练②：中训练 + DSA 引入（warmup 冻主干只训 indexer → sparse adaptation），32K
stage0_pretrain/stage3_longctx    预训练③：长上下文扩展，128K(500B) → 1M(50B)
stage1_sft                        SFT：mcore 的 --sft（SFTTokenizer 模板），数据取 post-training 的 SFT 集
stage2_rl                         RL：verl（GRPO + agent loop）；第四个子 stage（stage4_world_model）是世界模型
stage3_eval                       评测：NeMo Gym 基准（vLLM 起服务 + gym eval）+ 不依赖 Gym 的 local 套件
```

`stage2_rl/stage4_world_model` 与策略这条线并行：它从 `stage0_pretrain` 的基座出发，产物被 `stage2_rl/stage2_agentic`
当环境用（`--profile world_model`），不改变策略的训练流程。依据：AgentWorld 报告里 Sim RL 在 4k OOD 环境上
Claw-Eval 65.4 → 69.7、单轮 LWM RL warm-up 迁移到多轮工具调用（[2606.24597](https://arxiv.org/abs/2606.24597)）。

每个 stage 目录里是同样五件事：

| 文件 | 做什么 |
|------|--------|
| `train.py` | 入口（`--profile` 选档、`--dry-run` 只打印命令、`--set` 点号覆写、`--smoke` 跑仓库内 tiny 档） |
| `test_train.py` | 集成测试：tiny 几何跑 5 步并校验收尾（RL / 评测那几个是预检，见第 5 节） |
| `data_prep.py` | 语料准备（`--discover` 看面貌、`--prepare` 产出 bin/idx 或 parquet） |
| `config/` | `default.yaml`（全量档）+ `debug.yaml`（极小档）+ 数据配比 json（+ 对照档见各 stage README） |
| `README.md` | 这个阶段的数据口径、超参对照、判据与踩过的坑 |

## 2. 工具分工

PT 与 SFT 用**上游 mcore 的训练循环**（本目录 `train/train_shensi.py` 入口 = 上游 `pretrain_gpt.py` 的等价物，
模型从 **Megatron-Bridge** 的 `models/shensi/` 取），启动由 `train/launcher.py` 摊平配置直接起 torchrun；
RL 用 **verl**，评测与推理用 **vLLM**。数据准备由各 stage 的 `data_prep.py` 完成（轻量实现，不依赖 Ray），
产出口径与 Nemotron 库的 `nemotron.data_prep` 一致。

`train/` 里的运行时是 FlagScale 时代那份入口的替代：配置仍是同样的三段
（`train.{system,model,data}` 摊平成 mcore CLI，`no_xxx: true` 表示关掉、list 摊成 `--key v1 v2 ...`），
但不再有 runner/调度层——单机 1~8 卡用不上，返回码就是训练进程的返回码。

公共件：

| 文件 | 做什么 |
|------|--------|
| `common.py` | 路径（`env_paths`）/ 配置合并（`build_config`）/ 启动（`run`）/ 语料（`discover`、`prepare`） |
| `tiny_model.py` | **极小几何的单一出处**：`as_cli_overrides()` 给 launcher，`make_tiny_shensi_provider()` 给 Bridge 侧 |
| `tiny_test.py` | 集成测试的公共部分（跑 tiny 档 + 按日志判 PASS/FAIL + preflight 小工具） |
| `rl.py` | RL 的 yaml → verl CLI 映射与启动 |
| `early_stop.py` | 日志看门狗：验证指标超耐心就发 SIGTERM |

## 3. 四条已定的口径

1. **三个 loss 全开**：MoE aux `0.001`、ERC `coef 1.0 / alpha 0.5`、DSA indexer KL `0.01`
   （DeepSeek-V3.2 §2.1；DeepSeek-V4 的 CSA 同样带 Lightning Indexer）。
2. **两阶段 DSA（因为从零训练）**：主干先在稠密模式下训（`csa_dense_mode: true`，对应 GLM-5 报告的稠密基座），
   再"冻主干、只训 Lightning Indexer"（KL 目标是稠密注意力分布）→ 最后切稀疏（top-k 目标）全参训。
3. **模型几何 = `ShensiConfig` 默认**：35 主干层 + 3 MTP 层的 `compress_ratios`
   （`4` = CSA 带 indexer、`128` = HCA、`0` = 滑窗）、1M 上下文（compressor 层 YaRN factor 16 / 原位置 65536）。
4. **优化器默认是 Muon + Lion 混合**：2D 矩阵走 Muon（GLM-5 的 Muon Split + DeepSeek-V4 的 γ=0.18 与
   spectral 尺度，本家族再叠加 MLA 的按头/按组切分），非矩阵（embedding / 输出头 / norm / **MoE router** /
   **mHC 静态与 AttnRes 门控**）走 Lion；两条腿共用一条 LR 曲线（V4-Flash 峰值 2.7e-4）。
   口径、旋钮与逐项验收见 `stage0_pretrain/stage1_pretrain/README.md` 的「优化器」小节。
   **为什么不是 AdEMAMix**：上游 mcore 只把**主体**优化器那组交给 emerging 优化器表，标量那组一律落到
   `_get_megatron_optimizer_based_on_param_groups`，而那里只认 adam/adamw/lion/sgd——`--muon-scalar-optimizer ademamix`
   在上游实现里构造不出来（FL fork 同样如此）。AdEMAMix 当主体是支持的，留了对照档
   `stage1_pretrain/config/ademamix.yaml`（`--optimizer ademamix`）。`--profile adamw` / `--profile lion`
   仍是旧口径对照。**SFT 与 RL 仍走 Adam 系**（SFT 走 AdamW、RL 走 verl 的 `actor.optim`）。

显式**不采用**的两项（按项目口径取舍，各 stage README 里写了影响面）：GLM-5 的 loss-free bias 负载均衡、
DeepSeek-V3.2 的 dense warmup 之外再叠一层 indexer 蒸馏。

### 3.1 训练侧的三件增量

| 件 | 旋钮 | 落点 |
| --- | --- | --- |
| DSA TopK 外部内核（DeepSeek DeepSelect 这类） | `shensi_index_topk_kernel: "包.模块:函数"`（空串走内置 torch 版） | 训练前向的 top-k 入口 |
| MTP draft 单独训练（DeepSpec 口径） | `--profile mtp_draft`（主干全冻、只训 MTP） | `stage2_midtrain/config/mtp_draft.yaml` + `--shensi-freeze mtp` |
| ERC loss | `--shensi-erc-loss-coef / --shensi-erc-loss-alpha` | `train/erc.py`（HF 口径：全模型分组平均） |

## 4. 快速开始

```bash
# ① 冒烟：仓库内 tiny 档（2 层 / mock 数据 / 5 步），不需要任何真实语料
cd stage0_pretrain/stage1_pretrain
python train.py --smoke

# ② 集成测试：tiny 几何 + 该 stage 的档，跑 5 步并自动判 PASS/FAIL
python test_train.py                 # 有 data_prep 产物就用真实 bin/idx，没有就退回 mock
python test_train.py --profile adamw # 换档对照

# ③ 真实数据的极小档
python data_prep.py --discover                 # 语料面貌
python data_prep.py --prepare --blend config/data_prep/debug_sample.json
python train.py --profile debug                # 极小档真跑几步

# ④ 正式跑（27T 预算；按卡数与显存调 GBS）
python train.py --tokens 27e12
```

每次都把最终配置与完整命令落到 `<exp_dir>/config.yaml` 与 `<exp_dir>/run.sh`，日志实时写
`<exp_dir>/logs/host_0_localhost.output`（`early_stop.py` 看的就是这个文件）。

## 5. 集成测试与判据

`test_train.py` 是每个 stage 的"这条链路还活着吗"闸门，判据统一在 `tiny_test.py`：

| stage 类型 | 测什么 | 判据 |
|-----------|--------|------|
| 预训练 / SFT（`stage0_pretrain/*`、`stage1_sft`） | tiny 几何跑 5 步（有真实数据用真实数据，否则 mock） | rc=0、跑到最后一次 iteration、出现 `[after training is done]`、日志无 Traceback/Error |
| RL（`stage2_rl/*`） | 预检：配置→命令、数据 parquet、ray、GPU、import、环境变量 | 每项 ✓ 才算 PASS（环境变量缺只提示） |
| 评测（`stage3_eval`） | 预检：配置、`vllm serve` 命令、vllm CLI、模型目录、GPU、import | 同上（模型目录不在本机只提示） |

极小几何的定义在 `tiny_model.py`：2 层（一层 CSA 带 indexer + 一层 HCA）、一层 hash-MoE + 一层 MoE、
`hc_mult 16`、AttnRes block 4、hidden 128、seq 128 —— 家族独有的每一处结构都保留，只是规模压到单卡几十秒。

## 6. 早停与评估口径

步数都往"接近无穷"给，靠**评估间隔 + 耐心**收尾（mcore 的 `eval_interval`/`eval_iters`、
verl 的 `test_freq`/`val_before_train` 都只做评估，本身不会早停；早停由看门狗做）：

| stage | 步数 | 评估 | 看门狗指标 |
| --- | --- | --- | --- |
| stage1_pretrain | `train_iters: 1000000`（`--tokens` 按 `tokens/(GBS×seq)` 换算） | `eval_interval: 500` / `eval_iters: 20` | `validation loss`，`--mode min` |
| stage2_midtrain / stage3_longctx | 按 token 预算（20B / 500B / 50B）换算 | 同上 | 同上 |
| stage1_sft | `train_iters: 5000` | `eval_interval: 100` / `eval_iters: 20` | `validation loss`，`--mode min` |
| stage2_rl | `total_training_steps: null` + `total_epochs` 给大（verl 里 `-1` 是字面值不是"无限"） | `test_freq` + `val_before_train: true` | `critic/score/mean`，`--mode max` |

```bash
python early_stop.py --log <exp_dir>/logs/host_0_localhost.output \
    --metric "validation loss" --mode min --patience 20 --max-wait 24
python early_stop.py --log <rl 日志> --metric "critic/score/mean" --mode max --patience 10
```

单机 1~8 卡：评估别太密（PT 500 步 / SFT 100 步量级足够看出趋势），耐心 10~20 次评估，再加 `--max-wait <小时>` 兜底。

## 7. 目录与环境变量

| 用途 | 路径 |
| --- | --- |
| 预训练语料 | `$SHENSI_FS/datasets/llm/pre-training/<数据集名>/`（Nemotron 预训练集，目录名去掉 `nvidia/`） |
| 后训练语料 | `$SHENSI_FS/datasets/llm/post-training/<数据集名>/`（Nemotron post-training v3、UltraData、OpenCoder-Instruct 等） |
| 产物 | `$SHENSI_FS/shensi/{data,ckpt,logs,runs}/` |
| 权重 / tokenizer | `$SHENSI_FS/models/DeepSeek-V4-Flash-0731/`（ModelScope `deepseek-ai/DeepSeek-V4-Flash-0731`） |

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `SHENSI_ROOT` | `/root/work/shensi`（不给就从本文件往上找带 `3rdparty/common` 的那层） | 代码工作区（`3rdparty/common/{Megatron-LM,Megatron-Bridge,verl,vllm}`） |
| `SHENSI_FS` | `/root/work/filestorage` | 存储根（语料 / 产物 / 权重） |
| `SHENSI_TOKENIZER` | `$SHENSI_FS/models/DeepSeek-V4-Flash-0731` | tokenizer 目录（极小档可用自训的小 tokenizer） |

## 8. 环境注意事项（实测）

装环境本身（哪些包要现场编、`uv sync` 的 exact 语义、SM120 的 hadamard 补丁）见
[`src/README.md`](../../../README.md) 的「装环境」与「本机补丁」两节。这里只列**训练时才暴露**的坑，
launcher（`train/launcher.py`）已经把能自动的都自动了：

| 现象 | 原因 | 现在怎么处理 |
|------|------|-------------|
| `make: No targets specified and no makefile found` / `pybind11 not found` | mcore 起训时会 `make -C core/datasets` 现编 helper；隔离装法不带 Makefile，也不带 pybind11 | launcher 把本 venv 的 `bin` 放到 `PATH` 最前（`python3` 就能找到 pybind11）；清单里也声明了 `pybind11`/`ninja` |
| `no kernel image is available`（DSA indexer 的 Hadamard 旋转） | `fast-hadamard-transform` 上游 `setup.py` 没有 SM120 的 `-gencode` | `patches/fast-hadamard-transform-sm120.patch` 本地编（步骤见 src/README） |
| 前向段错误（SM120） | FlagGems 的 flagos 后端在 `te_general_grouped_gemm` 上崩 | launcher `setdefault TE_FL_PREFER=vendor`（走 TE 自带 CUDA kernel） |
| flashinfer 起不来 | SM120 上它要 JIT 补稀疏 MLA 内核 | launcher 在 `/usr/local/cuda/bin/nvcc` 存在时 `setdefault CUDA_HOME=/usr/local/cuda` |
| `vllm version ... not supported`（verl 闸门要求 ≥0.18） | fork 只有分支没 tag，vcs-versioning 会算出 `0.1.devNNNN` | 编 vllm 时带 `VLLM_VERSION_OVERRIDE=0.30.1rc0.dev360+g54c5060a1`（清单的 `extra-build-variables` 里也写了） |
| RL 起不来：ray/vLLM 引擎初始化失败 | 代理环境变量被 ray worker 继承 | `rl.launch` 起子进程前删掉 `http(s)_proxy` 等 |
| RL 起不来：`Unknown platform 'nvidia_noipc'` | WSL2 没有跨进程 CUDA IPC，且 verl 的 engine 模块一 import 就查平台名 | `VERL_PLATFORM=nvidia_noipc` + `shensi.runtime` 先注册平台再让别的模块 import（顺序写在 `runtime.setup()` 注释里） |
| OOM（41G 级机器） | ray dashboard（6 个约 1.4G 的进程）+ 按核数预起的 worker + 每个 TransferQueue unit 约 0.9G | 默认档已关 dashboard、`num_cpus: 8`、`num_data_storage_units: 2`（见 `stage2_rl/stage1_rlvr/config/default.yaml` 的 `ray_kwargs` / `transfer_queue`） |
| `AttributeError: 'NoneType' object has no attribute 'storage'` | verl 在 `use_distributed_optimizer=false` 时无条件解引用 flat buffer | `shensi.runtime` 补判空（同一文件里上游自己判过） |
| `ademamix optimizer is not supported` | 标量腿只认 adam/adamw/lion/sgd（见第 3 节第 4 条） | 默认档改用 Lion；要 AdEMAMix 就用 `--optimizer ademamix` 的对照档 |

## 9. 局限

1. 全部配方在极小几何上验证过（集成测试 + ckpt 往返 + RL 跑到训练步），**全规模收敛结论需要真机预算**；
2. **MTP 只能 0/1 层**：上游 main 的 MTP 在 mHC 配置下用 `[s,b,n*h]` 多流张量串起各 stage（由它自己的
   `HyperConnectionTransformerLayer` 消费），而本家族的 `ShensiTransformerLayer` 吃的是收缩后的 `[s,b,h]`
   （层内自己展开成流）——两边接口不一致，`mtp_num_layers>1` 会在第二个 MTP stage 上撞形状错误。
   极小/集成测试默认把 MTP 关掉（`--mtp 0`）；**生产几何（3 层共享 MTP）在修好这层适配前不要起大跑**；
3. 长上下文段缺 GLM-5 那三类自建/合成长数据（见 `stage0_pretrain/stage3_longctx/README.md` 第 7 节）；
4. 昇腾路径的命令按清单与厂商文档编写，未上 NPU 实测（见包根 `README.md` 的「环境与已知限制」）。
