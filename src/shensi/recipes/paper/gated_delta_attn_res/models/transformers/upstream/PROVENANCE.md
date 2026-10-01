# upstream/：各变体"各自 GitHub 仓库"的官方参考件（vendored，原样未改）

**用途**：`test_upstream_alignment.py` 拿这些官方件与我们的移植件做**数值对拍**——"严格对齐各自
上游仓库"不是声明，是可跑出来的读数。文件本身**逐字未改**（各自带原来的 license/作者处），
下游任何格式化/lint 都不要动它们。

| 文件 | 上游 | 我们对应 |
|---|---|---|
| `muddformer_*.py` / `_README.md` | **MUDDFormer** 官方仓库（CURRENTF/MUDDFormer；包内 `tmp/official_refs/muddformer/` 为当时取回的原文） | `models/transformers/modeling_qwen3_mudd.py`（HF）与 `models/megatron/depth_connection.py::MultiwayDynamicDense`（mcore） |
| `denseformer_*.py` / `_README.md` | **DenseFormer** 官方实现（包内 `tmp/official_refs/denseformer/`） | `modeling_qwen3_denseformer.py`（`DepthWeightedAverage`）与 mcore 侧同名件 |
| `kimi_modeling_kimi_linear.py` | **MoonshotAI/Kimi-K3** 的 `modeling_kimi_linear.py`（AR 算子的可运行官方出处；sha256 见 `code/AR_PROVENANCE.md`） | `modeling_qwen3_ar.py::_apply_attn_res`（逐行） |
| （不 vendored）shensi 分支 `ShensiAttentionResidual` | **本仓 `3rdparty/common/transformers` 就是该分支**：`src/transformers/models/shensi/modular_shensi.py`（GDAR 的真上游，就地可测） | `modeling_qwen3_gdar.py::AttentionResidual` 与 mcore 侧 `gdar_connection.py` |
| （不 vendored）DAR：`wdlctc/delta-attention-residuals-code` | 论文仓库未 vendored（离线环境取不回）；`modeling_qwen3_dar.py` 的 docstring 给出其 `delta_attn_res` 公式，对拍按公式级参考实现做 | `modeling_qwen3_dar.py::DeltaRouter` |

快照校验（本目录文件，sha256 前 16 位；换快照时重算）：
- `muddformer_modeling.py` 19255B `6e7f1c16f5029e47`
- `muddformer_configuration.py` 2581B `16b7a800a07a9002`
- `muddformer_layers.py` 8232B `b7a3ab3ca92cc1fa`
- `muddformer_README.md` 2215B `2073ecd3d32e352c`
- `denseformer_denseformer.py` 2162B `0a2d6adeef70598f`
- `denseformer_models.py` 19246B `074cb2af2fe33727`
- `denseformer_README.md` 1607B `f30b82f0a7e9903c`
- `kimi_modeling_kimi_linear.py` 51506B `9e3564c70ac21854`
