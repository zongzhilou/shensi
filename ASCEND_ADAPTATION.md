# GDAR Ascend 增量适配与验证记录

日期：2026-10-02。基线：`8f8835b`，本地分支：`ascend-gdar-tiny`。

**当前交付已通过单卡 Ascend 910B2 上的 HF 参考链路和原生 Megatron PT → midtrain → SFT 极小参数训练。verl、vLLM-Ascend 与多卡/生产规模流程尚未验收，不能据此宣称整个仓库在 NPU 上全流程无问题。**

## 实际运行情况

CANN Lab 账户当时有 100 卡时余额，但启动新建的 `shensi_tiny` 单卡 910B3 实例时，平台返回 HTTP 400：`当前资源不足，请稍后重试.`。该平台未运行 NPU 训练。随后改用下文的焕新平台完成真机验证。本机 Windows 上只有 PyTorch CPU。

平台当时可选镜像为 CANN 9.0.0 / Python 3.11。仓库固定的 MegatronAdaptor 文档推荐 CANN 9.2.0、PyTorch 2.10.0、torch_npu 26.2.0、Python 3.12；不能把两个版本栈视为已兼容。来源为随包保留的 `3rdparty/ascend/MegatronAdaptor/README.md` 和 `TransformerEngineNPU/docs/zh/release_notes_tenpu.md`。全量 `pyproject.ascend.toml` 依赖尚未在 NPU 环境解算、构建或实跑，没有凭推测替换其版本约束。

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 后端、启动配置、白化前后向回归 | 15 passed，1 skipped | skipped 是真实 NPU 用例 |
| 原有 GDAR 理论判据 | 58/58 | CPU |
| 原有 AutoClass / 导出重载判据 | 126/126 | CPU，多种连接变体 |
| HF ↔ mcore 权重表往返 | 80 张量逐位一致 | `--tables-only`，没有运行 mcore 模型 |
| 极小 HF 参考流程 FP32/full | 通过 | CPU，45,705 参数 |
| 极小 HF 参考流程 BF16/full | 通过 | CPU，45,705 参数 |
| 极小 HF 参考流程 FP32/per_head | 通过 | CPU，45,705 参数 |
| 检查点+优化器恢复，独立进程 HF 重载 | 三组最大差异均为 0 | 同设备重载 |
| 原生 Megatron PT → midtrain → SFT | 已生成并检查 tiny 启动计划 | 此行为 10 月 2 日的 CPU 阶段；真机结果见下文 |
| NPU 环境自查 | 正确失败 | 本机缺 CANN / torch_npu，无静默 CPU 回退 |
| HCCL、NPU kernel、原生训练、verl、vLLM、OpenCompass | 当时未验收 | 后续 HCCL、参考 kernel、原生训练在焕新平台通过，其他项见下文 |

FP32/full 的预训练损失为 `4.131524 → 4.001785`，中训练末损失 `3.853172`，SFT 末损失 `3.743316`，最终模拟评估损失 `3.629844`。模拟数据只检查数值和链路，不代表模型能力。

`validation/2026-10-02/` 保存报告、日志、上机计划、极小模型导出产物和平台错误记录。原转换测试的首次完整运行因缺少 Megatron 停止，日志也保留；没有把部分通过算作完整通过。

## 2026-10-04 焕新平台真实 NPU 验证

用户提供的焕新开发环境 `shensi-npu-tiny` 通过 **VSCode** 使用了 1 张 Ascend 910B2（64 GB），镜像为 CANN 9.1.0-beta.1、Python 3.11.15、PyTorch 2.10.0+cpu、torch_npu 2.10.0。只在新目录 `/root/work/shensi-ascend` 复制项目和安装独立虚拟环境；原有 `/root/work/shensi-offline` 与 `/root/work/filestorage` 均未编辑。没有使用 Jupyter。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| CANN / torch_npu 真机自查 | 通过；1 张 NPU，矩阵乘与反向正常 | `validation/2026-10-04/artifacts/ascend_actual_retry/preflight.json` |
| HCCL 单进程通信 | 通过 | `validation/2026-10-04/artifacts/ascend_actual_retry/hccl.log` |
| 增量回归 | **16/16 通过**，0 失败/错误/跳过 | `validation/2026-10-04/artifacts/ascend_actual_retry/pytest.xml` |
| tiny FP32/full、BF16/full、FP32/per_head | 三组均通过真实 `npu:0` 的预训练→中训练→SFT→参考 RL→参考 OPD→恢复→HF 导出/重载/生成 | 三组 `report.json`、`checks_retry.log` |
| checkpoint 参数、导出后 logits | 三组最大差异均为 0 | 三组 `report.json` |
| 原生 MegatronAdaptor/Megatron PT → midtrain → SFT | **三阶段各 1 步真实 NPU 训练全部通过**，返回码均为 0；中训练和 SFT 从前一阶段加载检查点并各自保存新检查点 | `validation/2026-10-04/native_final/artifacts/native_210_trial4/native_report.json`、三个 stage 的日志与 checkpoint marker |
| 原生训练依赖自查 | 通过；Python 3.12.4、torch/torch_npu 2.10.0、Megatron-Core 0.20.0、TransformerEngineNPU 2.17.1 | `validation/2026-10-04/native_final/artifacts/native_210_trial4/preflight.json` |

原始日志和报告位于 `validation/2026-10-04/`，其中远端取回的 `reference-evidence.tgz` 已做 SHA256 校验。FP32/full、BF16/full、FP32/per_head 各自的首个预训练损失到末个 SFT 损失依次为 `4.131523 → 3.743316`、`4.131402 → 3.748037`、`4.131524 → 3.743295`。合成数据只用来验证数值与链路，不代表模型能力。

首次 pytest 遇到镜像内预装 `torchvision` 与 PyTorch 算子不匹配（缺 `torchvision::nms`），导致 Transformers 文本模型导入失败。在新项目中用 `sitecustomize.py` 让本次纯文本流程不发现 torchvision 后，完整重跑通过；没有改动镜像原有包。可复现的可选兼容脚本在 `scripts/compat/text_only_no_torchvision/sitecustomize.py`，仅当环境出现同样故障时，将其目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH="$PWD/scripts/compat/text_only_no_torchvision:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/run_ascend_checks.sh artifacts/ascend_reference
```

原生训练使用独立 Python 3.12 虚拟环境，安装固定源码中的 MegatronAdaptor、TransformerEngineNPU、Megatron-Core 和 transformers；只使用新建的 `/root/work/shensi-ascend` 目录。运行时先解决了 Python 3.12 头文件、`datasets` 缺包及 vendor CLI 对分布式后端参数的兼容问题。三个阶段的日志均有训练完成与检查点保存标志，证据 tar 的 SHA256 为 `2c0934c62376362a924fe8d7acc8e4f988910abaac96060c1b39978accac721e`。`validation/2026-10-04/native_final/` 保留提取出的报告、日志和 marker，未将大型模型检查点塞入源码包。

参考流程中的 RL/OPD 是数学链路参考，不是原生 verl 训练或缓存式 teacher 流程。焕新环境中 `verl`、`vllm`、`vllm-ascend` 未安装，离线 aarch64 wheelhouse 没有 vllm-ascend；因此 rollout/actor、服务生成及 OpenCompass 端点没有真机通过结论。本次只有一张卡，亦未验证多卡通信和并行训练。

2026-10-04 后续按源码做了增量预审，修复了 verl 外部模块列表被已有环境变量覆盖、NPU 上空 `VLLM_PLUGINS` 禁用插件、vLLM `.pth` 注册钩子未执行、HF 导出备用路径的张量未转 CPU、转换审计在 NPU 上跳过前向却误报成功、评测 `--config` 未生效及已有端点模式仍要求本地模型、NPU 深度检索评测自动设备选择、评测环境硬编码项目路径等问题。评测脚本现在可用 `SHENSI_VLLM_PYTHON` 指定实际安装 vLLM 的 Python，`SHENSI_VLLM_BIN` 指定其可执行文件。**这些后续源码修正只经过静态检查，尚未在 NPU 上重跑；上表的实测结论属于修正前的代码版本。**

## 增量改动

- `SHENSI_BACKEND=auto|cuda|npu|cpu`（也支持 `nvidia` / `ascend` 别名），明确请求 NPU 时，缺包或设备不可用直接失败。
- 在 GDAR 训练入口、verl bridge 和共享 runtime 第一次导入 Megatron 前加载官方 `megatron_adaptor`。不修改第三方 Megatron 源码；CUDA 分支不导入该适配层。
- 独立通信测试直接使用 HCCL；原生 MegatronAdaptor/MindSpeed 的 CLI 需传其接受的 `--distributed-backend nccl`，厂商适配层处理 NPU 通信，传字面量 `hccl` 会在其参数包装器处被拒绝。按 `ASCEND_RT_VISIBLE_DEVICES` / `ASCEND_VISIBLE_DEVICES` 计算进程数。
- GDAR 数据搬运、导出设备和连接初始化 RNG 支持 NPU；保持 CUDA 选择分支。
- RL 的 NPU 平台选择为当前固定 verl 源码中的 `huawei`，避免固定 `nvidia_noipc`，保留 vLLM 插件发现。
- NPU 的 full/per-head 白化默认仅把小型协方差的 `eigh` 求解移到 CPU，再返回 NPU。公式、谱截断和 detach 语义保持原实现；没有用对角白化或近似递推替代原公式。这是正确性优先的兼容路径，存在同步开销，**不是 NPU 性能优化**。`SHENSI_NPU_EIGH=native` 可在支持该算子的版本上显式尝试。
- 修复 `--smoke` 忽略 `--set`、`--load`、`--dry-run` 和 OPD teacher 参数的问题；修复 profile 中 `base` 未解析，使 OPD tiny 不再继承百万步大模型配置。
- 自查真正导入组件并做 NPU 分配、矩阵乘和反向；缺依赖、零设备、导入失败都会返回非零退出码。TransformerEngineNPU 使用实际模块名 `transformer_engine.pytorch`。

适配范围以请求中的 `gated_delta_attn_res` 配方和必要共享工具为主；没有声称仓库内其他研究配方或全部生产并行配置已经适配验收。

## 安装及复现
 
源码压缩包包括仓库源码、全部递归 Git 子模块源码和版本清单，不含 `.git`、虚拟环境、缓存或用户凭据。脚本在 Linux/NPU 上运行；Windows 本次仅作 CPU 校验。

先使用配套的 CANN/PyTorch/torch_npu 环境。不要让普通 `pip install torch` 覆盖镜像里的厂商配套版本。参考流程不要求安装 Megatron、verl 或 vLLM：

```bash
cd /path/to/shensi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -r requirements.tiny-reference.txt
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
python -m shensi.utils.ascend_env --scope reference --json artifacts/preflight.json
bash scripts/run_ascend_checks.sh
```

`run_ascend_checks.sh` 默认依次运行真实 NPU 自查、单卡 HCCL、回归测试，以及 FP32/BF16/full/per-head 的参考链路。有两卡时可设 `NPROC_PER_NODE=2` 验证两卡通信。它不会自动购置算力或安装/替换 CANN。

参考流程的预训练、中训练、SFT、组相对策略梯度、reverse-KL 蒸馏、保存恢复、导出与生成均使用真正 GDAR 模型张量计算。RL 和 OPD 是单进程数学链路验证，**不是 verl 的四个方向 RL 训练，也不是原生缓存式 OPD**。

CPU 复现：

```bash
SHENSI_BACKEND=cpu python -m pytest -q tests/test_ascend_incremental.py
SHENSI_BACKEND=cpu python -m shensi.recipes.paper.gated_delta_attn_res.common.tiny_reference \
  --device cpu --steps 3 --out artifacts/tiny_cpu
python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.test_convert_tiny --tables-only
```

原生 Megatron 真机闸门（已在焕新平台按下列命令通过）：

```bash
export SHENSI_BACKEND=npu
python -m shensi.utils.ascend_env --scope train --json artifacts/native_preflight.json
python -m shensi.recipes.paper.gated_delta_attn_res.common.native_tiny \
  --backend npu --steps 1 --out artifacts/native_tiny
```

该原生入口使用 3 层、hidden=32、2 个注意力头、词表 64、序列长度 16 的 GDAR full 白化模型，自动生成本地 tokenizer 和 SFT jsonl，并串接 PT → midtrain → SFT 检查点。默认 Adam，仅隔离验证基础训练路径；`--optimizer adaptive_muon` 尚未实跑。执行后检查退出码、迭代数、完成标志和 checkpoint marker，不把命令打印成功当作训练成功。焕新镜像若缺 Python 3.12 开发头文件，可在运行前指向已有 Python 3.12 源码树的 `Include` 目录设置 `C_INCLUDE_PATH` / `CPLUS_INCLUDE_PATH`；本次路径为 `/root/work/Python-3.12.4/Include:/root/work/Python-3.12.4`。

在没有 NPU 的开发机上，使用 `SHENSI_BACKEND=cpu ...native_tiny --backend npu --dry-run --out artifacts/native_plan` 只生成计划。`native_report.json` 的 `status=planned` 明确表示未实跑。

原有逐 stage 入口仍可用，例如：

```bash
R=src/shensi/recipes/paper/gated_delta_attn_res
python "$R/stage0_pretrain/stage1_pretrain/train.py" --smoke --set train.model.train_iters=3
python "$R/stage1_sft/train.py" --smoke --dry-run --set train.model.hidden_size=64
python "$R/stage2_rl/convert/test_convert_tiny.py" --device npu
```

## 尚待实机完成

1. 从原生 GDAR checkpoint 转 HF 后，验证 vllm-ascend 自定义模型加载、实际生成和 OpenCompass 端点；单设备参考 `generate` 不覆盖服务链。
2. 安装并验证 verl 的实际 rollout/actor/ref 进程、四类 reward、HCCL 权重同步，以及原生 OPD teacher-cache 链路。
3. 验证 AdaMuon/AdEMAMix、多卡 TP/PP/CP/EP 和 CUDA 硬件回归。本次只有单张 NPU，没有这些项目的通过结论。

## 2026-10-11/12 八卡 hc-free 基线的适配件落仓

八卡生产链路的迁移与实跑由团队在焕新 8×910B2 上完成（本期结论以该实跑为准，本机未复跑 NPU）：
链路打通到"8 卡稳定训练"，冒烟 5/5 步 nan=0、selfcheck 15 项生效；实测 1807 token/s、
≈0.32% MFU、显存 ~22–29 MiB/token/卡（上限约 1500–2000 token/卡）、EP=2 优于 EP=1；
按该吞吐 57B passes 需约一年——**MindSpeed 落地前"一周档"不成立**。

本期把机器侧逐项 debug 出来的适配件收进本仓（原 tgz 交付改为直接 push，原始包与重建包见
`validation/2026-10-11/`）：

- `src/shensi/npu_runtime_fixes.py`——NPU 运行时补丁段（回退计数、NPU 安全 RoPE、attn_sink
  有限初始化、cdist Gram、compile 解包、显存探针、patch inventory selfcheck）。并入时修掉
  fix27 兜底入口 `_orig96` 在导入顺序不利时会 NameError 的隐患（改为调用时惰性解析）。
- `src/shensi/mindspeed_adapter.py`——MindSpeed 确定性接入：预检逐点只读解析（缺位跳过并点名，
  不再"中途失败留部分应用"；其 `parse_path` 会对缺属性 `setattr(None)` 污染命名空间，故未复用）、
  应用失败带完整报告即抛、config 字段快照差分自检。启用/停用由此变成可回退开关。
- `scripts/ascend/`——mcore 4 补丁（CSA 有限哨兵防 NaN、回退死探针、`init=False` 跳过、
  LayerWise allgather 按 dtype 分桶）+ MindSpeed 关停补丁 + 幂等应用脚本 + 移植漂移清单 +
  离线几何对拍。
- `recipes/shensi/common/param_ledger.py`（接在 builders 构建后）——构建期参数账审计：
  每块共享 experts 对象、hash 层共表、dedup/state_dict/激活三口径。**机上实测 4.53B 与账面
  2.2930B 对不上的差额，恰好等于"16 组专家未共享 + 3 张 deepemb 未共表"的算术**——训练启动后
  看 `[shensi][ledger]` 一行即可判定该疑案。

边界：1807 token/s 等数字来自团队机上实测；分块 shim 的 `group_list` 语义按 torch_npu 公开
契约实现、本机以语义替身验证（昇腾首跑前几步留意）；MindSpeed 仍确定性关停，移植完成后
删 `scripts/ascend/patches/mindspeed/0001-*.patch` 即可安全启用。
