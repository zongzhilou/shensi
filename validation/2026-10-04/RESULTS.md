# 焕新平台单卡 NPU 验证结论（2026-10-04）

平台：`shensi-npu-tiny`，VSCode，Ascend 910B2 64 GB，CANN 9.1.0-beta.1。项目仅放在新建的 `/root/work/shensi-ascend`；`/root/work/shensi-offline` 和 `/root/work/filestorage` 原资料未修改。

| 范围 | 实际结果 | 主要证据 |
|---|---|---|
| CANN、torch_npu、NPU 前后向、单进程 HCCL | 通过 | `artifacts/ascend_actual_retry/preflight.json`、`hccl.log` |
| 单卡参考 GDAR FP32/full、BF16/full、FP32/per_head | 三组通过；各自覆盖 PT→midtrain→SFT→参考 RL/OPD 数学计算→检查点恢复→HF 导出/重载/生成 | 三组 `report.json` |
| NPU 增量回归 | 16/16 通过 | `artifacts/ascend_actual_retry/pytest.xml` |
| 原生 Megatron GDAR tiny BF16 PT→midtrain→SFT | 三阶段各 1 步真实训练全部通过；三份训练结束日志、三份检查点 marker（值均为 1）；后两阶段成功从上一阶段检查点载入 | `native_final/artifacts/native_210_trial4/native_report.json`、三个 stage 日志及 marker |
| 本地 CPU 增量回归 | 15 passed、1 skipped（NPU 专项） | 本机 pytest 复跑 |

原生训练虚拟环境为 Python 3.12.4，torch 2.10.0+cpu 与 torch_npu 2.10.0，固定子模块 Megatron-Core 0.20.0、MegatronAdaptor 0.1.0、TransformerEngineNPU 2.17.1；详见 `native_final/artifacts/native_210_trial4/preflight.json`。原生训练模型是 3 层、hidden 32、2 头、词表 64、序列长度 16，数据用模拟生成和本地 JSONL。验证记录包含调试失败过程，不将失败尝试算为通过。最终证据压缩包 `native-final-evidence.tgz` SHA256 为 `2c0934c62376362a924fe8d7acc8e4f988910abaac96060c1b39978accac721e`。

**未验证：**原生 verl actor/rollout/ref、原生 OPD teacher cache、vLLM-Ascend 服务、OpenCompass、多卡并行、AdaMuon/AdEMAMix 和生产规模。环境中 `verl`、`vllm`、`vllm-ascend` 未安装，离线 wheelhouse 缺 vllm-ascend。参考 RL/OPD 的通过不代表原生进程通过。

平台“保存镜像”显示剩余镜像资源 **0 GB**，无法持久化新增的镜像层；没有删除或改动既有镜像。截图为 `kunlun-image-quota.png`。源码、日志和结果已收回本地并纳入交付包；若后续实例重建，依赖环境可能需要按 `ASCEND_ADAPTATION.md` 重新安装。
