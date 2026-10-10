"""昇腾模拟器（仅测试用）：把 `npu_sim/` 加进 `sys.path` 后，`import torch_npu` / `megatron_adaptor`
会命中这里的替身，从而在没有 NPU 的机器上跑通 recipe 的 NPU 分支。

由 `test_npu_sim.py` 临时挂载/卸载；**不要**把本目录常驻 `PYTHONPATH`（那会让所有进程以为自己在昇腾上）。
"""
