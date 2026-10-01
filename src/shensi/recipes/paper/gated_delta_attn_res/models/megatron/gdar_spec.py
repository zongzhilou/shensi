# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layer specs for GDAR, exposed through Megatron's official ``--spec`` extension point.

``--spec`` takes a ``<module> <object>`` pair and ``spec_utils.import_module`` returns
that module-level *object* verbatim (it is not called), so a spec has to exist before
the config does.  That is why every preset here is a plain module-level
``ModuleSpec`` carrying the GDAR knobs in its ``params``: ``GdarTransformerLayer``
picks the knobs up in ``__init__`` (and builds its submodules from the *config*, so
nothing config-dependent is frozen at import time).

Enable it from a recipe task YAML::

    model:
      spec:
        - shensi.recipes.paper.gated_delta_attn_res.models.megatron.gdar_spec
        - gdar_layer_spec_paper

Configuration layout (mirrors ``EXPERIMENT_MATRIX.json``)
---------------------------------------------------------
* ``_PAPER`` — the **main** row: theory(objective update / decay ladder 64 / delta
  address / 8 read heads / Softmax_1 / full whitening) + ``decay_positivity="project"``
  + rank 64, at **block granularity ``B = 4``** (the main table's uniform value; the
  per-sublayer form is reported alongside via ``gdar_layer_spec_paper_sublayer``).
* every other ``gdar_layer_spec_paper_*`` preset is ``_PAPER`` with **exactly one**
  knob moved, so every row of the design-ablation table can be diffed against the
  main row directly (``RECIPE.md`` §6.2, ``GDAR_ABLATION_DESIGN.md``);
* the gate-subset rows (``gates=d|e|w|de|dw|ew|scalar|none``) live in
  ``ablation_spec.gdar_paper_gates_*`` — they need the ``AblationGdarLayer`` wrapper
  that rebinds the gates, and are built from the same ``_PAPER`` knobs.

``block_size`` must stay **strictly below** the layer count: with ``B >= L`` the whole
model is one block that never closes, so the connection is inert (measured: 27
tensors with ``grad is None``).  ``GdarTransformerLayer`` prints a warning when a
preset violates it.

Legacy names (still referenced by the recipe's ablation profiles and the docs) are
kept as aliases of the corresponding ``_PAPER``-based row.
"""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .gdar_layer import GdarTransformerLayer

__all__ = [
    # main + its forms
    "gdar_layer_spec_paper",
    "gdar_layer_spec_paper_sublayer",
    "gdar_layer_spec_paper_b2",
    "gdar_layer_spec_paper_b4",
    "gdar_layer_spec_paper_b8",
    "gdar_layer_spec_paper_b16",
    # rank rows
    "gdar_layer_spec_paper_rank16",
    "gdar_layer_spec_paper_rankfull",
    # one-knob rows
    "gdar_layer_spec_paper_decay_free",
    "gdar_layer_spec_paper_lambda_free",
    "gdar_layer_spec_paper_ladder0",
    "gdar_layer_spec_paper_gate_prefix",
    "gdar_layer_spec_paper_gate_delta",
    "gdar_layer_spec_paper_address_state",
    "gdar_layer_spec_paper_address_novelty",
    "gdar_layer_spec_paper_update_reference",
    "gdar_layer_spec_paper_heads1",
    "gdar_layer_spec_paper_null_off",
    "gdar_layer_spec_paper_whiten_diag",
    "gdar_layer_spec_paper_whiten_off",
    "gdar_layer_spec_paper_mix_whitened",
    "gdar_layer_spec_paper_no_output_route",
    "gdar_layer_spec_paper_carrier0",
    "gdar_layer_spec_paper_init_paper",
    "gdar_layer_spec_paper_init_uniform",
    "gdar_layer_spec_paper_init_half",
    # other families
    "gdar_layer_spec",
    "gdar_layer_spec_reference",
    "gdar_layer_spec_fullrank",
    "gdar_layer_spec_theory",
    "gdar_layer_spec_upstream",
    "gdar_layer_spec_block2",
    "gdar_layer_spec_block4",
    "gdar_layer_spec_block8",
    "gdar_layer_spec_block16",
    "gdar_layer_spec_block4_r64",
    "gdar_layer_spec_block4_r16",
    # legacy aliases
    "gdar_layer_spec_gate_prefix",
    "gdar_layer_spec_gate_delta",
    "gdar_layer_spec_decay_projected",
    "gdar_layer_spec_lambda_free",
    "gdar_layer_spec_read_whitened_mix",
    "gdar_layer_spec_paper_noladder",
    "gdar_layer_spec_no_output_route",
    "DESIGN_ABLATIONS",
    "make_gdar_spec",
]

#: default knobs of the connection.  See ``gdar_connection.GdarConfig``.
_DEFAULT = dict(
    gdar_block_size=1,
    gdar_output_route=True,
    # --- write side: exact identity at init (GDAR(0) == the plain residual update)
    gdar_gate_param="deviation",
    gdar_update="shensi",
    gdar_address="state",
    gdar_write_carrier_bias=-4.0,
    gdar_decay_ladder=0,
    # --- read side: the reference's plain single-head softmax read
    gdar_read_heads=1,
    gdar_read_null=False,
    gdar_read_whiten="off",
    # --- parameterisation: the redo plan's low-rank fix (full rank costs ~10 H^2/layer).
    # r=64 is the training default the docstring above, the recipe (RECIPE.md item 1) and
    # check_gdar.py all promise; `gdar_layer_spec_fullrank` is the explicit full-rank arm.
    gdar_gate_rank=64,
    gdar_q_rank=64,
    gdar_k_rank=64,
)

#: **论文主配置（main 行）**：设计矩阵 EXPERIMENT_MATRIX.json 的 "main"——
#: theory(objective / ladder 64 / address=delta / 8 读头 / Softmax_1 / 全白化)
#: + ``decay_positivity="project"`` + rank 64，block 形态 ``B = 4``（主表统一取值）。
_PAPER = dict(
    _DEFAULT,
    gdar_block_size=4,
    gdar_update="objective",
    gdar_decay_ladder=64,
    gdar_address="delta",
    gdar_read_heads=8,
    gdar_read_null=True,
    gdar_read_whiten="full",
    gdar_decay_positivity="project",
)


def make_gdar_spec(**knobs) -> ModuleSpec:
    """A GDAR layer spec with ``gdar_*`` knobs overriding the defaults."""
    params = dict(_DEFAULT)
    params.update(knobs)
    return ModuleSpec(module=GdarTransformerLayer, submodules=None, params=params)


def _paper(**one_knob) -> ModuleSpec:
    """``_PAPER`` with exactly one knob moved (every ablation row is built through this)."""
    return make_gdar_spec(**dict(_PAPER, **one_knob))


# --------------------------------------------------------------------------
# main row and its forms
# --------------------------------------------------------------------------
#: **main**：论文主表行（block 形态，B=4）。
gdar_layer_spec_paper: ModuleSpec = _paper()
#: main 的 per-sublayer 形态（B=1；论文里与 block 形态并列报告）。
gdar_layer_spec_paper_sublayer: ModuleSpec = _paper(gdar_block_size=1)
#: block 形态扫描 B ∈ {2,4,8,16}；``_b4`` 就是 main。
gdar_layer_spec_paper_b2: ModuleSpec = _paper(gdar_block_size=2)
gdar_layer_spec_paper_b4: ModuleSpec = _paper(gdar_block_size=4)
gdar_layer_spec_paper_b8: ModuleSpec = _paper(gdar_block_size=8)
gdar_layer_spec_paper_b16: ModuleSpec = _paper(gdar_block_size=16)

# --------------------------------------------------------------------------
# rank rows
# --------------------------------------------------------------------------
#: 参数匹配行（r=16，对齐 HC/mHC 量级）。
gdar_layer_spec_paper_rank16: ModuleSpec = _paper(gdar_gate_rank=16, gdar_q_rank=16, gdar_k_rank=16)
#: 开销证据行（全秩；论文里只用于"开销不可忽略"，不作主表设置）。
gdar_layer_spec_paper_rankfull: ModuleSpec = _paper(
    gdar_gate_rank=None, gdar_q_rank=None, gdar_k_rank=None
)

# --------------------------------------------------------------------------
# one-knob rows（A1–A16；每行只改一处，与 main 直接可比）
# --------------------------------------------------------------------------
#: A3 对照：decay 正性不投影（"free"，有界性变测量事实）。
gdar_layer_spec_paper_decay_free: ModuleSpec = _paper(gdar_decay_positivity="free")
#: A4 对照：去掉 λ 钳制（λ 只保留严格凸域的理论约束）。
gdar_layer_spec_paper_lambda_free: ModuleSpec = _paper(gdar_lambda_clamp=None)
#: A11 对照：单时间尺度（去掉衰减阶梯）。
gdar_layer_spec_paper_ladder0: ModuleSpec = _paper(gdar_decay_ladder=0)
#: A1a/A1b：门的输入换成 prefix / delta。
gdar_layer_spec_paper_gate_prefix: ModuleSpec = _paper(gdar_gate_source="prefix")
gdar_layer_spec_paper_gate_delta: ModuleSpec = _paper(gdar_gate_source="delta")
#: A7：地址换成 state（参考式）/ novelty（对保留流做正交化）。
gdar_layer_spec_paper_address_state: ModuleSpec = _paper(gdar_address="state")
gdar_layer_spec_paper_address_novelty: ModuleSpec = _paper(gdar_address="novelty")
#: A6：更新式换回旧参考规则（objective 的对照）。
gdar_layer_spec_paper_update_reference: ModuleSpec = _paper(gdar_update="reference")
#: A12 读侧三前提逐条：单头 / 关弃权（Softmax₁）/ 白化 diag、off。
gdar_layer_spec_paper_heads1: ModuleSpec = _paper(gdar_read_heads=1)
gdar_layer_spec_paper_null_off: ModuleSpec = _paper(gdar_read_null=False)
gdar_layer_spec_paper_whiten_diag: ModuleSpec = _paper(gdar_read_whiten="diag")
gdar_layer_spec_paper_whiten_off: ModuleSpec = _paper(gdar_read_whiten="off")
#: A5：在白化空间里平均（GLS/BLUE 论证的对照）。
gdar_layer_spec_paper_mix_whitened: ModuleSpec = _paper(gdar_read_mix="whitened")
#: A16：关掉末端输出路由。
gdar_layer_spec_paper_no_output_route: ModuleSpec = _paper(gdar_output_route=False)
#: A10：写门载体偏置归零（梯度陷阱的对照；已由 test_theory 闭环，不必跑）。
gdar_layer_spec_paper_carrier0: ModuleSpec = _paper(gdar_write_carrier_bias=0.0)
#: A9 初始化行：发表初始化（sigmoid+paper）与两个失败锚点（uniform/half）。
gdar_layer_spec_paper_init_paper: ModuleSpec = _paper(
    gdar_gate_param="sigmoid", gdar_gate_init="paper"
)
gdar_layer_spec_paper_init_uniform: ModuleSpec = _paper(
    gdar_gate_param="sigmoid", gdar_gate_init="uniform", gdar_gate_init_bias=-20.0
)
gdar_layer_spec_paper_init_half: ModuleSpec = _paper(
    gdar_gate_param="sigmoid", gdar_gate_init="zero"
)

# --------------------------------------------------------------------------
# other families
# --------------------------------------------------------------------------
#: 训练默认档（shensi 更新式、per-sublayer、低秩 64）——A6 的 shensi 更新式行。
gdar_layer_spec: ModuleSpec = make_gdar_spec()

#: the shensi repository as-is (no exact identity, full-rank projections)
gdar_layer_spec_reference: ModuleSpec = ModuleSpec(
    module=GdarTransformerLayer,
    submodules=None,
    params=dict(
        _DEFAULT,
        gdar_gate_param="sigmoid",
        gdar_gate_init="paper",
        gdar_gate_rank=None,
        gdar_q_rank=None,
        gdar_k_rank=None,
    ),
)

#: deviation gates with the reference's full-rank projections
gdar_layer_spec_fullrank: ModuleSpec = make_gdar_spec(
    gdar_gate_rank=None, gdar_q_rank=None, gdar_k_rank=None
)

#: 理论版（main 去掉正性投影；= ``_paper_decay_free`` 的旧名）。
gdar_layer_spec_theory: ModuleSpec = _paper(gdar_decay_positivity="free")

#: 与上游 ``transformers@shensi`` 的 ``ShensiAttentionResidual`` **逐位对齐**的预设：
#: 论文主行 + 逐头白化（上游当前实现）。默认的 ``gdar_layer_spec_paper`` 用我们自己的全局
#: 白化（`read_whiten="full"`）——两者的实测差见 ``models/transformers/test_upstream_alignment.py``
#: 的 "known deltas"（read_scale=1 时 max|diff| ≈ 2.7）。
gdar_layer_spec_upstream: ModuleSpec = _paper(gdar_read_whiten="per_head")

#: block-granularity snapshots（shensi 更新式家族，E6 的对照臂）
gdar_layer_spec_block2: ModuleSpec = make_gdar_spec(gdar_block_size=2)
gdar_layer_spec_block4: ModuleSpec = make_gdar_spec(gdar_block_size=4)
gdar_layer_spec_block8: ModuleSpec = make_gdar_spec(gdar_block_size=8)
gdar_layer_spec_block16: ModuleSpec = make_gdar_spec(gdar_block_size=16)
#: block-4 variants at an explicit rank, for scans whose iso-FLOP plan must know the
#: parameterisation (the default r=64 and the parameter-matched r=16 of the paper).
gdar_layer_spec_block4_r64: ModuleSpec = make_gdar_spec(
    gdar_block_size=4, gdar_gate_rank=64, gdar_q_rank=64, gdar_k_rank=64
)
gdar_layer_spec_block4_r16: ModuleSpec = make_gdar_spec(
    gdar_block_size=4, gdar_gate_rank=16, gdar_q_rank=16, gdar_k_rank=16
)

# --------------------------------------------------------------------------
# legacy aliases（配置档与文档仍在引用这些名字）
# --------------------------------------------------------------------------
gdar_layer_spec_gate_prefix: ModuleSpec = gdar_layer_spec_paper_gate_prefix
gdar_layer_spec_gate_delta: ModuleSpec = gdar_layer_spec_paper_gate_delta
#: "project" 行 = main 本身（main 已含 ``decay_positivity="project"``）。
gdar_layer_spec_decay_projected: ModuleSpec = gdar_layer_spec_paper
gdar_layer_spec_lambda_free: ModuleSpec = gdar_layer_spec_paper_lambda_free
gdar_layer_spec_read_whitened_mix: ModuleSpec = gdar_layer_spec_paper_mix_whitened
gdar_layer_spec_paper_noladder: ModuleSpec = gdar_layer_spec_paper_ladder0
gdar_layer_spec_no_output_route: ModuleSpec = gdar_layer_spec_paper_no_output_route

#: 设计消融矩阵（``EXPERIMENT_MATRIX.json`` 的 GDAR 行）→ spec 预设；
#: 门结构（gates=…）在 ``ablation_spec.gdar_paper_gates_*``。
DESIGN_ABLATIONS: dict[str, ModuleSpec] = {
    "main": gdar_layer_spec_paper,
    "block_size=1": gdar_layer_spec_paper_sublayer,
    "block_size=2": gdar_layer_spec_paper_b2,
    "block_size=4": gdar_layer_spec_paper_b4,
    "block_size=8": gdar_layer_spec_paper_b8,
    "block_size=16": gdar_layer_spec_paper_b16,
    "rank=64": gdar_layer_spec_paper,
    "rank=16": gdar_layer_spec_paper_rank16,
    "rank=full": gdar_layer_spec_paper_rankfull,
    "init=identity": gdar_layer_spec_paper,
    "init=paper": gdar_layer_spec_paper_init_paper,
    "init=uniform": gdar_layer_spec_paper_init_uniform,
    "init=half": gdar_layer_spec_paper_init_half,
    "gate_source=state": gdar_layer_spec_paper,
    "gate_source=prefix": gdar_layer_spec_paper_gate_prefix,
    "gate_source=delta": gdar_layer_spec_paper_gate_delta,
    "update=objective": gdar_layer_spec_paper,
    "update=reference": gdar_layer_spec_paper_update_reference,
    "address=delta": gdar_layer_spec_paper,
    "address=state": gdar_layer_spec_paper_address_state,
    "address=novelty": gdar_layer_spec_paper_address_novelty,
    "lambda_clamp=-0.5": gdar_layer_spec_paper,
    "lambda_clamp=None": gdar_layer_spec_paper_lambda_free,
    "decay_positivity=project": gdar_layer_spec_paper,
    "decay_positivity=free": gdar_layer_spec_paper_decay_free,
    "ladder=64": gdar_layer_spec_paper,
    "ladder=0": gdar_layer_spec_paper_ladder0,
    "read_heads=8": gdar_layer_spec_paper,
    "read_heads=1": gdar_layer_spec_paper_heads1,
    "read_null=on": gdar_layer_spec_paper,
    "read_null=off": gdar_layer_spec_paper_null_off,
    "read_whiten=full": gdar_layer_spec_paper,
    "read_whiten=diag": gdar_layer_spec_paper_whiten_diag,
    "read_whiten=off": gdar_layer_spec_paper_whiten_off,
    "read_mix=raw": gdar_layer_spec_paper,
    "read_mix=whitened": gdar_layer_spec_paper_mix_whitened,
    "output_route=on": gdar_layer_spec_paper,
    "output_route=off": gdar_layer_spec_paper_no_output_route,
    "carrier_bias=-4": gdar_layer_spec_paper,
    "carrier_bias=0": gdar_layer_spec_paper_carrier0,
}
