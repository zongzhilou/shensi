"""本地训练运行时：直接用上游 mcore 的训练循环跑 Shensi（不经 FlagScale）。

- `train_shensi.py`：入口，等价上游 `pretrain_gpt.py`，模型换成 Bridge 的 Shensi。
- `args.py`：`--shensi-*` 旋钮、与 HF config 的几何对拍、检查点告警。
- `builders.py`：`ShensiModelConfig` / `ShensiModelBuilder`（上游 `ModelConfig`/`ModelBuilder` 接口）。
- `erc.py`：ERC loss（DeepSeek-V4 的路由专家耦合正则）。
- `probes.py`：权重加载探针、fp32 保持与 mHC 相关的一次性装置。
- `data.py`：GPT 数据集 provider（mock / 真实 bin-idx 两种口径）。
- `launcher.py`：配置 → mcore CLI → torchrun（取代 FlagScale 的 runner）。
"""
