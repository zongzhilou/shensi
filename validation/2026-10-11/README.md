# 2026-10-11/12：hc-free 基线 8 卡部署的适配件落仓

来源：团队 8 卡迁移的两个交付包（`upgrade-hc-free.tgz` sha `01fabec4`、
`shensi-migration-final3.tgz` sha `690b35d1`）。本目录记录**落进本仓**的对应物与重建包，
原始包内容不再单独维护。

## 落进本仓的东西

| 内容 | 位置 | 来源 |
|---|---|---|
| NPU 运行时补丁段（1510 行） | `src/shensi/npu_runtime_fixes.py` | migration 包 `overwrite/runtime.py` 追加段（并入时修了 fix27 兜底入口的 NameError 隐患） |
| mcore 4 补丁 + MindSpeed 关停补丁 | `scripts/ascend/patches/` | migration 包 `patches/`、`patches-mindspeed/`（逐位一致） |
| 补丁应用脚本（幂等 + `--check`） | `scripts/ascend/apply_npu_patches.sh` | 改写自 migration 包 `migrate.sh` 的补丁段 |
| 移植漂移清单 | `scripts/ascend/MindSpeed-PORT-NOTES.md` | migration 包同文件（逐位一致） |
| 离线几何对拍 | `scripts/ascend/check_geometry.py` | migration 包 `tools/`（逐位一致） |

## 重建的交付包（供尚未升级的机器）

| 包 | sha256 | 内容 |
|---|---|---|
| `upgrade-hc-free-v2.tgz` | `ad909b99bec5270ff8f689b75355ccd7be396925414db3b50eceded33e432f3d` | payload = shensi `f585242` + Bridge `22fe0632` + transformers `f1b85b4a17`；包内 sha 校验 14/14 |
| `shensi-migration-final4.tgz` | `15a3ac436fdf1101091f01f4f9b1ed31e8103d4d32ca8bcdbe221ee98039cbf8` | 覆盖 shensi 三个模块（runtime / npu_runtime_fixes / mindspeed_adapter）+ 同一套 mcore/MindSpeed 补丁 |

两包都做过的验证：包内 sha 清单（14/14 与 3/3，mcore 5 条终态 sha 与 final3 相同、需机器上核）、
`bash -n`、`upgrade.sh` 在假树上的端到端演练（完整性 → 替换 → 依赖预检 → AST/UTF8 闸门 148 文件）。

## 组件 SHA（hc-free 基线）

shensi `f585242` / Megatron-Bridge `22fe0632` / transformers `f1b85b4a17` / vllm `147f5cad7d` /
mcore `c16f739b1600`（不动）。
