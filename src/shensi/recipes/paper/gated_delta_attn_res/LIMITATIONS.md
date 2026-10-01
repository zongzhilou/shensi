# 局限清单与处置（逐条：问题 → 处置 → 证据）

更新时间 2026-10-01（B1 结案与几何两处修正，见 A13–A20）。口径：**能解决的已解决**（附实测证据），**不能在本机解决的写明为什么 + 具体升级路径**，
不留模糊说法。所有命令都能在仓库里重跑。

## A. 已解决

| # | 曾经的局限 | 处置 | 证据 |
|---|---|---|---|
| A1 | 训练没有默认早停，"无限步数"会跑飞 | 四个训练 stage（PT/Mid/SFT/OPD）**默认带看门狗**；RL 走同一套（`common.run_verl`）；`train_iters`/`total_epochs` 给到无限大；`--no-early-stop` 可关 | 触发式实测：patience=0 → 看门狗 SIGTERM 训练组、写 `early_stop.json`（`why=patience, best=10.93065`），launcher 按**成功**返回 rc=0；训练停在 iteration 13/400（未跑完） |
| A2 | 早停 metric 会抓到 iteration 号（正则取"指标名后第一个数"） | 默认 metric 改 `"lm loss value"`（日志 `... \| lm loss value: X`）而非 `"validation loss"` | 修正前 `best=5.0`（iteration 号）→ 修正后 `best=10.93065`（真 loss） |
| A3 | 看门狗路径/活性没人守（起不来会静默跑到底） | 路径按 `recipes/shensi/early_stop.py` 解析 + 启动后 1s 活性检查，起不来**直接报错退出** | 实测：路径写错时立即 `SystemExit` 并打印子进程输出 |
| A4 | RL 的 `config/rl.yaml` 名与 `rl.launch` 默认不符、argv 被吞、`early_stop.*` 被当未映射键 | 配置改名 `default.yaml`；argv = `sys.argv[1:] + 覆盖项`；`build_verl_command` 跳过本仓库私有段（`harness`/`early_stop`） | 6 个 RL profile dry-run 全 rc=0 |
| A5 | 融合开关"在 SM120 上不可用"（结论过粗） | 逐项隔离后结论：**TE + bias_swiglu/bias_gelu/grad_accum 三个融合可用**；`masked_softmax` 需要 apex 的 `scaled_masked_softmax_cuda`、`persist_layer_norm` 不被 torch LayerNorm 支持——这两个保持关闭 | 三融合 smoke：launcher rc=0、日志 `done=1`；打开 masked_softmax → `ModuleNotFoundError: scaled_masked_softmax_cuda`；打开 persist_ln → `AssertionError: persist_layer_norm not supported by torch LayerNorm` |
| A6 | `read_whiten="per_head"`（上游逐头白化）在 mcore 侧被白名单拒 | `GdarConfig.validated` 白名单补 `per_head`；mcore `_depth_read` 补同款分支 | 27 臂 GPU 扫里 `gdar_layer_spec_upstream` finite=True |
| A7 | 潮汐式"未复现即不管"的风险 | 偶发 CUDA illegal access 记录在案 + 5 次复跑未复现 + 定位命令（`CUDA_LAUNCH_BLOCKING=1`）写进 `stage2_midtrain/README.md` 维护记录 | 复跑记录：B=4/B=2/B=1 各 3 步 + 完整 test_train 共 5 次全绿 |
| A8 | 没有评测 stage（设计里 T0 必做项） | 移植 `stage4_eval/`：`make_depth_retrieval.py`（生成器）+ `run_depth_retrieval.py`（评分器，含 chance/Wilson/位置偏差/usable 门）+ `test_train.py` 预检 | 冒烟实测：40 题 3 秒跑完，`score.json` 里 `chance=0.25`、`pooled_acc=0.175`（随机权重 tiny ckpt，符合"低于 chance"的预期） |
| A9 | RL 算法只有 GRPO 一种 | 落地 5 个可跑 profile（见 §C 超越清单）+ CLI 映射扩展（`clip_ratio_c`/`loss_agg_mode`/`filter_groups`/`norm_adv_by_std_in_grpo`/`override_transformer_config.*`/`critic.*`） | **四个臂 × 六个 profile = 24 个 dry-run 全绿**，且每个都验证了 `model.path` 与两份 apex 覆盖真的出现在 verl 命令里（A15） |
| A10 | `HarnessTool` 起不来（缺 `tool_schema`） | 按 WorldModelTool 的写法补 `DEFAULT_SCHEMA` + `model_validate` | 实测 `execute({"action": "echo harness-ok"})` → `'harness-ok'`；dsh 缺失时给出明确提示 |
| A11 | OPD 只有 forward KL（MiniCPM5 用 reverse KL） | 静态 KD 侧实现 **reverse KL**：`train/reverse_kl.py` 提供与 mcore `topk_kl_div` 同签名同返回的版本，并接管模块级名字（缓存/TP/迭代管线全沿用 mcore）；`--logits-load-reverse-kl` 开关 | 单测 `train/test_reverse_kl.py`：reverse == 解析解（2.7e-7）、前向 == 解析解（3.7e-7）、两者非退化（差 2.8）、补丁幂等且已接管 |

| A12 | 收尾跑格式化时 `ruff format` 把 `models/**` 一并重排（纯格式） | 已把 `models/**` 与两个评测移植件加进 `[tool.ruff] extend-exclude`（今后不再参与本仓格式化）；**vendored 官方件已按原样恢复**（sha256 与 `upstream/PROVENANCE.md` 逐位一致，kimi 前缀 `9e3564c70ac21854` ✓）；移植件本身是**格式重排、语义未动**——由三组数值闸门兜底（下游重跑：对拍 13/13、HF 单测、GPU 27 臂扫） | `upstream/PROVENANCE.md` 的 sha 表；`test_upstream_alignment.py`；`test_theory/test_ablation_switches/test_autoclass` |

| A13 | RL 的 Megatron actor 缺 GDAR 注册（verl 里跑 mcore actor 需要 Bridge 认这个架构） | 按本仓 Bridge 体系重写：`stage2_rl/gdar_bridge.py`（导入即用上游 `MegatronModelBridge.register_bridge` 注册七个 model_type；provider 层规格来自 `variants.build_layer_spec`，与本配方训练器同一份；权重表由 `convert/tables.py` **生成**；`synth` 行用 `allow_hf_name_mismatch` + 一个"Megatron 独有"mapping 处理）+ `common.run_verl` 把该模块挂进 `VERL_USE_EXTERNAL_MODULES` | 闸门 `stage2_rl/test_gdar_bridge.py`（tiny，12/12 全绿）：auto_map 分发到本桥 ✓、层规格=GDAR 连接层 ✓、**桥上的规格与 `gdar_layer_spec_paper` 生效配置逐项相等** ✓、装载零缺键零意外键 ✓、HF vs verl 建出的 mcore `max\|Δ\|=2.4e-07` ✓、导出（rollout 同步那条路）**名字集合一致且逐位相等（max\|Δ\|=0）** ✓；另 `convert/test_convert_tiny.py`（往返位级 80/80） |
| A14 | mcore 侧低秩 `q/k` 的 `up.bias` 零初始化但**可训练**（HF 的 `_make_qk` 两段都 `bias=False`，上游 FlagScale 移植件同样带这个偏置） | `gdar_connection.AttentionResidual._make_qk` 里 `requires_grad_(False)` 钉死在零（mcore 的参数组构建与 FP16 optimizer 都按 `requires_grad` 过滤，实测不影响优化器）；表的 `synth` 注释同步说明 | 闸门新增断言"合成行恒零且不可训练"（9 个 `up.bias`，漂移/可训练 0）；装载仍零缺键；logits 对拍不变 |
| A15 | RL 各 profile 的 yaml 写着"与 default 深合并"，但**没有任何合并发生**（`--profile dapo` 连 `model.path` 都没有，会以 verl 默认路径起训）；且各臂没设 `override_transformer_config.gradient_accumulation_fusion`（本机无 apex，建列并行层即报错） | profile yaml 加 `base: default.yaml`（`_load_with_base` 现成的深合并机制）；四个臂的 `default.yaml` 给 actor/ref 补 `override_transformer_config.gradient_accumulation_fusion: false`；`build_verl_command` 增加 `override_transformer_config.*` 直传；`common.run_verl` 的 `VERL_USE_EXTERNAL_MODULES` 带上 `gdar_bridge` | 24 个 RL profile dry-run 全绿且都带 `model.path` + 两份 apex 覆盖（键数 54→59/61） |
| A16 | `stage1_sft/test_train.py` 的"合成 jsonl"一路**不自足**：jsonl 由 `train.py --smoke` 生成，集成测试本身不生成——换机或清过 `runs/` 后 `data_path` 落到 `exp_dir`，`datasets` 会把 `exp_dir/config.yaml` 当 JSON 读（`ArrowInvalid`） | 测试自己复用 `train.py` 的 `_smoke_jsonl()` 生成 16 条合成对话并显式 `train.data.data_path=<jsonl>` | 冷跑（先删 `runs/stage1_sft_tiny`）实测：`rc=0 最后 iteration=5 收尾标记=True 报错行=0`，PASS |

| A17 | 训练侧声明 GQA（`num_query_groups < num_attention_heads`）但实际训的是 **MHA** | mcore 的 `--num-query-groups` 只在 `--group-query-attention` 在场时生效（`training/argument_utils.py`：否则 `num_query_groups=None` → `TransformerConfig` 退回 `num_attention_heads`）。`train/launcher.py` 现在按配置声明派生该开关 | 干跑命令实测：`--num-query-groups 8` + `--group-query-attention`（default 档为 16 头/8 组）；tiny 档 4 头/2 组；重训后检查点 `run_config.yaml` 为 `num_query_groups: 2` ✓（此前是 4） |
| A18 | Qwen3 的 q/k 归一化没开——训出来的不是 Qwen3 | mcore 不传 `--qk-layernorm` 就不建 `q_layernorm/k_layernorm`；HF 参考实现与论文模型都有这两组。launcher 现在按架构派生（配置显式写 `qk_layernorm: true` 时不重复加） | 参数计数实测：4 层 tiny 下 `qk_layernorm=False` = 81,900,577（= 修复前那次 run 的计数），`True` = 81,901,089；重训后检查点里出现 `self_attention.q_layernorm.weight` ✓ |
| A19 | 冒烟档的检查点互相覆盖（都写到 `ckpt/gated_delta_attn_res/iter_*`） | `common.smoke_config` 现在与 `build_config` 同一条规则：`checkpoint.save = <ckpt>/gated_delta_attn_res/<stage>/<profile>` | 重跑 stage1_sft 冒烟：落在 `ckpt/gated_delta_attn_res/stage1_sft/tiny/iter_0000005` ✓，与它 exp 目录的 config 一一对应 |
| A20 | 每个 stage 的产物都是 mcore ckpt，而 RL 的 `model.path`、评测、上线都读 HF 目录——中间缺一步 | 新增 `train/export_hf.py`（本配方自己的桥 + 由 `convert/tables.py` 生成的权重表）：几何以**检查点自带的 `run_config.yaml`** 为准（stage 的 config.yaml 只作兜底，两者不一致时以检查点为准并打印警告），连接旋钮来自那次 run 的 `config.yaml` 的 `train.model.spec` 或 `--model-algo`，载入前按检查点元数据做形状预检，检查点里的"规范名"（`linear_qkv.layer_norm_weight` 等，mcore 自己的 state_dict_hooks 表）在载入前后各改写一次 | 端到端实测（tiny）：`export_hf` → HF 目录（`model_type: qwen3_gdar`、`auto_map`、两个 `.py`、22 个 `attn_res_*` 旋钮、kv=2）→ transformers 前向 finite → `stage4_eval` 40 题 3 秒出分（`chance=0.25`、`usable` 就位）。两个坑也是实测出来的：tokenizer 目录是完整 Qwen3 快照，**不能整目录复制**（会把合成的 `config.json` 盖成 stock Qwen3 的）；`dist_checkpointing.load` 是裸载，不做 norm 改写 |

> 影响范围（A17/A18）：这两处修复前**训出来的检查点**是 MHA + 无 q/k 归一化 的几何——代码已修，重训即得 Qwen3 几何；已训的 old ckpt 仍然可被 `export_hf` 忠实导出（模型几何跟随检查点），但不要把它们当作 Qwen3 对照臂去和外部模型比。

## B. 本机解决不了、已写明升级路径

| # | 局限 | 为什么本机不行 | 升级路径（具体到文件/命令） |
|---|---|---|---|
| B2 | OPD 的 **RL 式** reverse-KL advantage（用 −KL 当 reward 的 on-policy 循环）还没接 | 静态 KD 侧已换成 reverse KL（见 A11）；advantage 形式要接在 RL 循环里，需要 teacher 的在线 logprob 端点 | 用 stage2_rl 的 verl 栈 + 一个 `opd_reward.py`（custom_reward_function 已经是插件式）：reward = −KL(student‖teacher)，teacher 走 vLLM 端点（`prompt_logprobs`），拿现成的 `models/vllm` 起服务即可 |
| B3 | GSPO（序列级 importance ratio）不在本仓 verl 的估计量里 | verl 0.10 的 `AdvantageEstimator` 只有 gae/grpo/rppo/rloo/opo/gpg/gdpo/token-baseline 等，**没有 gspo** | Qwen3 全系用 GSPO 稳 MoE（见 §C 引文）。升级路径：升级 verl 到带 `gspo` 的版本并加 `algorithm.adv_estimator=gspo` 映射（一行）；或在 `actor.loss_agg_mode` 上用 seq-mean 近似（已支持，但非等价） |
| B4 | 30B-A3B 的 `moe_router_topk`/专家 FFN 在 CLI 上不可设 | 本仓 mcore 的 MoE 参数组只有 `--num-experts`/`--moe-router-load-balancing-type` 等（探针实测）；`moe_router_topk` 是 config-only | 集群上按 `--help` 核对（不同 mcore 版本名字有差）；或用 `yaml_cfg` 路径给 config 字段；门面模型不承担架构结论，影响可控 |
| B5 | 集群真跑（主表/消融多 seed/RULER 长上下文） | 本机单卡 16GB、无集群 | 配方与命令已就绪：各 stage README 末尾的"跑完整论文实验"块 + `MINICPM5_ALIGNMENT.md` §6 |
| B6 | 连接算子的融合内核（吞吐再进一步） | 需要写 CUDA/CUTLASS 内核 | 现状已给出可用的最大吞吐（TE 骨干 + 三融合 + vLLM rollout）；内核项独立立项 |

## C. 相对 MiniCPM5-2B 的超越清单（每条都有落点）

| 维度 | MiniCPM5-2B | 本配方 | 落点 |
|---|---|---|---|
| 架构（论文主张） | 标准 Llama 稠密 | 深度连接（GDAR main）与 8 个对照/基线（AR/DAR/HC/mHC/MUDD/DenseFormer…）**逐位对齐各自上游** | `models/megatron`、`models/transformers/test_upstream_alignment.py`（13/13，AR/MUDD/DenseFormer/GDAR 逐位） |
| RL 算法 | JustRL II（GRPO + critic + 三段过滤） | GRPO 基线（含 DAPO 的双侧截断）+ **5 个可选 profile**：`dapo`（动态采样/token 级 loss/clip-higher）、`drgrpo`（去 std 归一）、`token_baseline`（token 级最优基线估计量）、`critic`（GAE + value，即 JustRL II 式）、`fsdp`（不经过 mcore 注册的那条）；Megatron actor 的 GDAR 注册已落地（A13） | `stage2_rl/*/config/*.yaml`、`stage2_rl/gdar_bridge.py`；§C 引文见下 |
| RL 稳定性 | 未公开 GSPO 采用 | 预留 `algorithm.adv_estimator` 扩展点，GSPO 升级路径写明（B3） | `common._VERL_CLI_EXTRA` |
| 早停/收敛 | 未公开 | **全 stage 默认早停**（metric=`lm loss value` / RL=`val/reward`），步数/轮次无限 | A1–A3 |
| 评测合规 | 未公开细节 | 受控检索带 **chance=25% + Wilson95 + 分层 + 位置偏差 + usable 门**（已冒烟）；RULER 规划里带 oracle 对照 | `stage4_eval/` |
| 吞吐 | 未公开 | TE 骨干 + 三个融合开关实测可用（§A5）+ vLLM 七变体 rollout（lcp=16/16 对齐 HF） | `perf.yaml`、`models/vllm/` |
| 数据/训练方案 | UltraData 五段式 | 逐项对齐（见 `MINICPM5_ALIGNMENT.md`），并补齐它没写的段内设计（2+2 段、LR 计划、分级代码数据） | `MINICPM5_ALIGNMENT.md` |
| 可复现性 | 公开配方 | 每 stage 可跑命令 + 每条读数带来源 + 逐值对拍测试 | 各 README「跑完整论文实验」块 |

### 引文（RL 算法，2026 检索）

- GRPO 系与 2026 共识（critic 多数场景不必需、`std` 归一会放大近解问题、token 级 loss 与动态采样是 DAPO 的有效修复）：见本轮检索综述（DAPO / Dr.GRPO / GSPO / VAPO / CISPO / REAL / NSR / HISPO）。
- GSPO 被 Qwen3 全系采用（MoE 稳定性），但在代码任务上弱于 DAPO——**已有研究结论支持"按方向选算法"**（我们的四方向 teacher 正好按方向配 profile）。
- token 级 baseline（`optimal_token_baseline`）与 critic（VAPO 式 value 预训练）是长 CoT 稀疏奖励下的两条 credit assignment 路线；本配方两条都有可跑档。
