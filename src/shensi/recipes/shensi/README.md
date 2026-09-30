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

## 2. 工具分工

PT 与 SFT 用**上游 mcore 的训练循环**（本目录 `train/train_shensi.py` 入口 = 上游 `pretrain_gpt.py` 的等价物，
模型从 **Megatron-Bridge** 的 `models/shensi/` 取），启动由 `train/launcher.py` 摊平配置直接起 torchrun；
RL 用 **verl**，评测与推理用 **vLLM**。数据准备由本目录的 `data_prep.py` 完成（轻量实现，不依赖 Ray），
产出口径与 Nemotron 库的 `nemotron.data_prep` 一致。

`train/` 里的运行时是 FlagScale 时代那份入口的替代：配置仍是同样的三段
（`train.{system,model,data}` 摊平成 mcore CLI，`no_xxx: true` 表示关掉、list 摊成 `--key v1 v2 ...`），
但不再有 runner/调度层——单机 1~8 卡用不上，返回码就是训练进程的返回码。

## 3. 四条已定的口径

1. **三个 loss 全开**：MoE aux `0.001`、ERC `coef 1.0 / alpha 0.5`、DSA indexer KL `0.01`
   （DeepSeek-V3.2 §2.1；DeepSeek-V4 的 CSA 同样带 Lightning Indexer）。
2. **两阶段 DSA（因为从零训练）**：主干先在稠密模式下训（`csa_dense_mode: true`，对应 GLM-5 报告的稠密基座），
   再"冻主干、只训 Lightning Indexer"（KL 目标是稠密注意力分布）→ 最后切稀疏（top-k 目标）全参训。
3. **模型几何 = `ShensiConfig` 默认**：35 主干层 + 3 MTP 层的 `compress_ratios`
   （`4` = CSA 带 indexer、`128` = HCA、`0` = 滑窗）、1M 上下文（compressor 层 YaRN factor 16 / 原位置 65536）。
4. **优化器默认就是 Muon 混合**：2D 矩阵走 Muon（GLM-5 的 Muon Split + DeepSeek-V4 的 γ=0.18 与 spectral 尺度，
   本家族再叠加 MLA 的按头/按组切分），非矩阵（embedding / 输出头 / norm / **MoE router** / **mHC 静态与 AttnRes 门控**）
   走 **AdEMAMix**（最新一代 Adam 变体，[2409.03137](https://arxiv.org/abs/2409.03137)）；两条腿共用一条 LR 曲线
   （V4-Flash 峰值 2.7e-4）。口径、旋钮与逐项验收见 `stage0_pretrain/stage1_pretrain/README.md` 的「优化器」小节，
   `--profile adamw` 保留旧口径做对照。预训练三段（stage0）都默认这一档；**SFT 与 RL 仍走 Adam 系**
   （SFT 走 AdamW、RL 走 verl 的 `actor.optim`）：Muon 的证据都在预训练规模上，小数据微调要用得先单独扫 LR。

显式**不采用**的两项（按项目口径取舍，各 stage README 里写了影响面）：GLM-5 的 loss-free bias 负载均衡、

### 3.1 训练侧的三件增量

| 件 | 旋钮 | 落点 |
| --- | --- | --- |
| DSA TopK 外部内核（DeepSeek DeepSelect 这类） | `shensi_index_topk_kernel: "包.模块:函数"`（空串走内置 torch 版） | 同上（训练前向的 top-k 入口） |
| MTP draft 单独训练（DeepSpec 口径） | `--profile mtp_draft`（主干全冻、只训 MTP） | `stage2_midtrain/config/mtp_draft.yaml` + `--shensi-freeze mtp` |


## 4. 早停与评估口径

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

## 5. 目录与环境变量

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
| `SHENSI_TOKENIZER` | `$SHENSI_FS/models/DeepSeek-V4-Flash-0731` | tokenizer 目录 |

`data_prep.py --discover` 先打印实际目录里有什么（格式、条数、字段名、权重），`--prepare` 再据此产出
mcore 的 `.bin/.idx` 与 `blend.json`；`train.py` 读 `blend.json` 把 `data_path` 注入配置，权重不用手工拼。

## 6. 跑一个 stage 的标准动作

```bash
cd stage0_pretrain/stage1_pretrain
python data_prep.py --discover                 # 看语料面貌（必要时改 data_blend_raw.json 的权重）
python data_prep.py --prepare                  # 产出 .bin/.idx + blend.json
python train.py --dry-run                      # 写 run 目录（config.yaml / run.sh）并打印 torchrun 命令
python train.py --smoke                        # 用仓库内 tiny 配置跑几步（mock 数据），确认环境/入口没坏
python train.py --tokens 27e12                 # 正式跑（27T 预算；按卡数与显存调 GBS）
```

每次都把最终配置与完整命令落到 `<exp_dir>/config.yaml` 与 `<exp_dir>/run.sh`，日志实时写
`<exp_dir>/logs/host_0_localhost.output`（`early_stop.py` 看的就是这个文件）。

## 7. 局限

1. 全部配方在极小几何上验证过（闸门 + 极小档训练 + ckpt 往返），**全规模收敛结论需要真机预算**；
2. 长上下文段缺 GLM-5 那三类自建/合成长数据（见 `stage0_pretrain/stage3_longctx/README.md` 第 7 节）；
3. 昇腾路径的命令按清单与厂商文档编写，未上 NPU 实测（见包根 `README.md` 第 7 节）；
4. **RL 侧（stage2_rl）在 mcore main 上还没打通**：上游的 CSA 只支持隐式 causal mask
   （`attention_mask` 必须为 None、`packed_seq_params` 必须为 None），而 verl 这个版本的 RL batch
   把 prompt/response padding 到等长（bshd）或直接打包（thd）——两条路都撞在 CSA 的约束上。
   换成上游 main 之后，RL 的进程能一路走到模型前向（导入期与配置期的坑都由 `shensi.runtime` 与
   Bridge 侧收口了），最后停在这条约束上；要么等上游给 CSA 补 mask/打包支持，要么 RL 侧改成
   单序列不 padding 的口径（`rollout.n=1` + 每个 prompt 一条），要么继续用 FL fork 跑 RL。
   预训练与 SFT 这两段不受影响（数据侧本来就不造 mask，见 `train/args.py` 的
   `disable_dataloader_attention_mask`）。
