# stage2_rl：四方向专用 RL teacher（并行分开训练）

起点统一是 **SFT-2 的 ckpt**（stage1_sft 的 sft2_agent 产物）；四个方向**并行分开训**、
互不共享权重——它们不是发布模型，是 OPD（stage3_opd）的老师。算法统一 verl GRPO +
Megatron actor（mbridge），启动复用 shensi 配方的 `rl.launch`（同一套 yaml → verl CLI
映射与环境处理）。

| 臂 | 方向 | rollout | 奖励 | 数据（post-training 落位后） |
|---|---|---|---|---|
| `stage2_math` | 数学 | 单轮 n=8 | 答案比对（\boxed/末数值，0/1） | GSM8K/MATH/AIME 类可验证题 |
| `stage2_code` | 代码 | 单轮 n=8 | 单元测试执行通过（0/1；生产换隔离沙箱） | 代码题 + 隐藏测试 |
| `stage2_agent` | Agent | **多轮工具**（multi_turn.enable） | 任务成功标志（环境回填） | 工具环境任务集 |
| `stage2_writing` | 写作 | 单轮 n=8 | rubric（骨架）/ RM（生产口径） | 写作题 + 参考 |

## 算法档（`--profile`，六档都已 dry-run 验证）

| profile | 是什么 | 关键覆盖 |
|---|---|---|
| `default` | GRPO 基线（含双侧截断 0.2/0.28、KL 关闭） | — |
| `dapo` | DAPO 口径 | `filter_groups`（动态采样）+ `loss_agg_mode=token-mean` + `clip_ratio_c` |
| `drgrpo` | Dr.GRPO 口径 | `norm_adv_by_std_in_grpo=false`（去 std 归一，消除长度/难度偏置）+ token 级 loss |
| `token_baseline` | token 级最优基线估计量 | `adv_estimator=optimal_token_baseline` |
| `critic` | value/GAE（JustRL II 式的 token 级 credit assignment 路线） | `adv_estimator=gae` + `critic.*`（起点与 actor 同源，由 train.py 注入） |
| `fsdp` | **不经 Bridge 的 HF-actor 路径**：actor 走 `models/transformers`、rollout 走已注册的 `models/vllm` | `actor.strategy=fsdp`、`use_mbridge=false` |

选型建议（2026 检索结论）：长 CoT/数学偏 `dapo`/`token_baseline`；关注长度膨胀用 `drgrpo`；
MoE 稳定性首选 GSPO（本仓 verl 无此估计量，升级路径见 `LIMITATIONS.md` B3）；代码方向 DAPO 系
token 级更合适。四方向 teacher 因此**按方向配档**，不搞一刀切。

## 跑法（每臂一样）

```bash
cd stage2_rl/stage2_math
python data_prep.py --prepare            # prompts → verl 的 train/val parquet
python train.py --dry-run                # 打印 verl 命令
python train.py                          # 起训（verl GRPO；LR 1e-6、GRPO 双侧截断）
```

## 起点：先发布成 HF 目录

`stage2_rl/*/config/default.yaml` 里的 `model.path` 是**HF 目录**（verl 按 `auto_map` 加载），
而 SFT stage 的产物是 mcore ckpt，所以起训前先发布（每个 SFT 子段各一次）：

```bash
python -m shensi.recipes.paper.gated_delta_attn_res.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_sft --out $SHENSI_FS/shensi/ckpt/gated_delta_attn_res/stage1_sft/sft2_agent
# 之后 --set model.path=<out>（或让 yaml 的默认值指向它）
```

## 判据与局限

- 判据：各方向验证集的可验证奖励（test_freq=20 出一次 val reward），教师间不比高低——
  它们只对 OPD 负责；OPD 的判据在 stage3_opd/README。
- **verl 侧的 GDAR 注册已就位**（`gdar_bridge.py`）：导入即注册七个 `model_type`
  （上游 `MegatronModelBridge.register_bridge`，不改 site-packages 任何文件）；provider 的
  层规格来自 `variants.build_layer_spec`（与训练器同一份代码，论文主版本的旋钮组合与之
  逐项相等）；权重表由 `convert/tables.py` 生成，`synth` 行（HF 没有对应物的张量）在模型里
  恒零且不可训练，所以导出时丢弃它们不会让被服务的模型和被训练的模型分叉。
  `common.run_verl` 把该模块挂进 `VERL_USE_EXTERNAL_MODULES`。
  闸门：`python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.test_gdar_bridge`
  （tiny 端到端：分发 / 规格 / 装载 / HF 对拍 / 导出逐位）。

---

## 跑完整论文实验（EXPERIMENT_MATRIX.md §5：门面 teacher）

RL 只为 **30B-A3B 门面**训练方向 teacher（四方向：数学 / 代码 / Agent / 写作），
明确不承担架构对比结论；每条 teacher 的 rollout 驱动按起点模型的类型自动接线。

```bash
# 起点：旗舰对两臂各自的 SFT-3 ckpt（GDAR 与 base 各一套 teacher）
for arm in qwen3_gdar_main base; do
  for dir in stage2_math stage2_code stage2_agent stage2_writing; do
    cd stage2_rl/$dir
    python data_prep.py --prepare --limit 200000                     # UltraData-RL-2609 按方向切
    python train.py                                                  # 起训（yaml 里的 model.path 指向该臂 SFT-3）
  done
done

# 预检（配置→CLI/奖励/依赖）：cd stage2_rl && python test_train.py
# 端到端闸门（auto_map 分发 / 层规格 / 装载 / 导出，tiny 规模）：
python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.test_gdar_bridge
```

每条 teacher 结束后的评测见 RUN_EXPERIMENTS.md §5（同 SFT 套 + 方向对应项）；
teacher 的产出是 `stage3_opd` 的输入（每个方向一个 ckpt）。
