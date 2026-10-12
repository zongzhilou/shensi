# Ascend NPU 适配件（补丁集 + 工具）

这里放 8 卡 NPU 部署需要的、**不在任何上游子模块里**的东西。机器侧逐项验证过的
适配件都收在仓里，机器上直接 `git pull` + 跑一个脚本就行，不再依赖单独传包。

```
scripts/ascend/
├── apply_npu_patches.sh        打 mcore / MindSpeed 补丁（幂等；--check 只报告不改）
├── patches/mcore/              4 个 mcore 补丁（目标 revision c16f739b1600）
│   ├── 0001-csa-finite-sentinel.patch               CSA 掩码 -inf → -1e4（防 NaN，必需）
│   ├── 0002-revert-dsv4-debug-probe.patch           回退 dsv4 死探针
│   ├── 0003-argument-utils-skip-init-false.patch    构造器跳过 init=False 字段
│   └── 0004-layer-wise-allgather-mixed-dtype.patch  LayerWise all-gather 按 dtype 分桶
│                                                    （DP>1 时 bf16+fp32 混桶会抛错）
├── patches/mindspeed/0001-disable-mindspeed-features.patch
│                                 确定性关掉 MindSpeed 特征补丁（见下）
├── MindSpeed-PORT-NOTES.md     30 处 mcore 引用漂移清单（移植用）
└── check_geometry.py           离线几何对拍（无需 torch/megatron，笔记本上可跑）
```

## 用法（机器上）

```bash
git -C /root/work/shensi-offline/shensi pull          # 拿到本目录与运行时适配
bash scripts/ascend/apply_npu_patches.sh /root/work/shensi-offline/shensi --check   # 先看
bash scripts/ascend/apply_npu_patches.sh /root/work/shensi-offline/shensi           # 再打
python scripts/ascend/check_geometry.py --repo /root/work/shensi-offline/shensi     # 几何预检
```

目标树可用 `MCORE_DIR` / `MINDSPEED_DIR` 覆盖（默认按 shensi-offline 布局）。

## MindSpeed 的现状（重要）

补丁把 MindSpeed 的特征补丁**确定性关掉**（可临时 `SHENSI_ENABLE_MINDSPEED=1` 试开）。
原因：它对当前 mcore 有约 30 处引用漂移，`MindSpeedPatchesManager.apply_patches()` 会中途
失败、留下"部分应用"状态（实测会让 config 字段丢失、训练起不来）。

仓里的运行时已经把这条路改安全了（`src/shensi/mindspeed_adapter.py`）：预检逐点只读解析
（缺位跳过并点名）、应用失败带完整报告即抛、config 字段快照差分自检。所以**移植完成后
删掉这个补丁即可启用**——不会再出现"全量或全崩"。清单与"早失败/晚失败"机制见
`MindSpeed-PORT-NOTES.md`。

## 与运行时适配的分工

- 补丁集（本目录）：**源码级**修改，只动 mcore / MindSpeed 子模块，随子模块版本走。
- 运行时适配（`src/shensi/`）：`runtime.py`（薄基座 + `npu_grouped_matmul` 128 分组分块 shim +
  MindSpeed 确定性适配接线）、`npu_runtime_fixes.py`（机器侧 NPU 补丁段：回退计数、NPU 安全
  RoPE、attn_sink 有限初始化、cdist Gram、compile 解包、显存探针……）、`mindspeed_adapter.py`。
- 构建期校验：`recipes/shensi/common/param_ledger.py`（tie 审计 + 参数口径，接在 builders 里）。

## 组件 SHA 对齐表（hc-free 基线）

| 组件 | SHA |
|---|---|
| shensi | `f585242`（本仓 main） |
| Megatron-Bridge | `22fe0632`（fork `shensi` 分支） |
| transformers | `f1b85b4a17`（fork `shensi` 分支） |
| vllm | `147f5cad7d`（fork `shensi` 分支，需重编译才生效） |
| mcore（Megatron-LM） | `c16f739b1600`（不 bump；补丁按此 revision 写） |
