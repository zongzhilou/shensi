"""测试专用的 `megatron_adaptor` 替身（昇腾侧由 MindSpeed 的 MegatronAdaptor 提供）。

`accelerator.bootstrap_megatron()` 在 NPU 后端上会在**第一次 megatron/TE 导入之前** import 它
（它负责装 `transfer_to_npu` 与可选的 MindSpeed 补丁）。这个替身只记录"被 import 过"，
供 `test_npu_sim.py` 断言接线顺序；真机不需要它。
"""

from __future__ import annotations

IMPORTED_MARKER = "shensi_npu_sim_megatron_adaptor_imported"
imported = True
