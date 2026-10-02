#!/usr/bin/env python3
"""GDAR 两塔的闸门：关闭态逐位对拍上游、开启态前向/反传、块划分与银行行数。

python -m shensi.recipes.paper.deeprecur.common.models.transformers.smoke_test_gdar
"""

from __future__ import annotations

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from ...data import MockVLDataset, VLCollator
from ...model import tiny_processor
from .configuration_qwen3_vl_gdar import Qwen3VLGdarConfig
from .modeling_qwen3_vl_gdar import Qwen3VLGdarForConditionalGeneration, block_boundaries

TEXT = {
    "vocab_size": 151669,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 4096,
}
VISION = {
    "depth": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 2,
    "patch_size": 16,
    "out_hidden_size": 128,
    "num_position_embeddings": 256,
    "deepstack_visual_indexes": [],
}


def _batch():
    processor = tiny_processor()
    dataset = MockVLDataset(n=2)
    return VLCollator(processor, None, 2048)([dataset[0], dataset[1]])


def check_boundaries() -> str:
    """块划分：floor(i×depth/n)，块数恒为 n，余数归末块。"""
    assert block_boundaries(4, 2) == [0, 2]
    assert block_boundaries(4, 4) == [0, 1, 2, 3]
    assert block_boundaries(27, 9) == list(range(0, 27, 3))
    assert len(block_boundaries(9, 4)) == 4 and block_boundaries(9, 4)[-1] == 6
    return "4/2=[0,2]、4/4=[0,1,2,3]、27/9=步长 3、9/4 末块吃余数"


def check_off_parity() -> str:
    """关闭态（recur_blocks=None）与上游逐位一致：树相同 → 严格装载 → logits 相等。"""
    batch = _batch()
    torch.manual_seed(0)
    upstream = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    ).eval()
    torch.manual_seed(0)
    gdar = Qwen3VLGdarForConditionalGeneration(
        Qwen3VLGdarConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    ).eval()
    gdar.load_state_dict(upstream.state_dict(), strict=True)
    with torch.no_grad():
        for name, inputs in (("文本", {"input_ids": batch["input_ids"]}), ("多模态", dict(batch))):
            delta = (upstream(**inputs).logits - gdar(**inputs).logits).abs().max().item()
            assert delta == 0.0, f"{name}路径关闭态与上游不一致：max|Δ|={delta}"
    return "严格装载通过；文本 / 多模态两条路径 logits 逐位相同（max|Δ|=0）"


def check_gdar_forward_backward() -> str:
    """开启态（recur_blocks=2）：多模态前向有限、反传覆盖全部 AR 参数。"""
    from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (
        AttentionResidual,
    )

    batch = _batch()
    torch.manual_seed(0)
    config = Qwen3VLGdarConfig(text_config=dict(TEXT), vision_config=dict(VISION), recur_blocks=2)
    model = Qwen3VLGdarForConditionalGeneration(config).train()
    n_ar = sum(1 for m in model.modules() if isinstance(m, AttentionResidual))
    assert n_ar == 2 * (TEXT["num_hidden_layers"] + VISION["depth"]), f"AR 模块数不对：{n_ar}"
    out = model(**batch)
    assert torch.isfinite(out.loss) and torch.isfinite(out.logits).all()
    out.loss.backward()
    missing, total = [], 0.0
    for name, param in model.named_parameters():
        if any(k in name for k in (".self_attention_attn_res.", ".mlp_attn_res.", ".output_attn_res_module.")):
            if param.grad is None:
                missing.append(name)
            else:
                total += float(param.grad.abs().sum())
    assert not missing, f"AR 参数没有梯度：{missing[:4]}"
    assert total > 0, "AR 参数梯度全为零"
    return (
        f"AR 模块 {n_ar} 个（两塔每层 2 个 + 两塔输出读各 1）；loss={out.loss.item():.3f}；"
        f"全部 AR 参数有梯度（|grad| 合计 {total:.3f}）"
    )


def check_block_rows() -> str:
    """银行行数：vision 塔的读-源峰值 - 1 = 块数（每块边界写一行）。"""
    batch = _batch()
    torch.manual_seed(0)
    config = Qwen3VLGdarConfig(text_config=dict(TEXT), vision_config=dict(VISION), recur_blocks=2)
    model = Qwen3VLGdarForConditionalGeneration(config).eval()
    stats: list = []
    with torch.no_grad():
        model.model.visual(
            batch["pixel_values"], grid_thw=batch["image_grid_thw"], attn_res_stats=stats
        )
    assert stats, "没有收集到 attn-res 统计"
    rows = max(entry["n_sources"] for entry in stats) - 1
    assert rows == 2, f"vision 银行行数 {rows} != recur_blocks=2"
    return f"读-源峰值 {rows + 1} → 银行行数 {rows} = recur_blocks（块边界写行生效）"


def check_deeprecur() -> str:
    """DeepRecur 顶层：块交织前向/反传有限、feedback 门有梯度、两开关真的改变前向。"""
    from .modeling_qwen3_vl_gdar import Qwen3VLGdarDeepRecurForConditionalGeneration

    batch = _batch()
    torch.manual_seed(0)
    config = Qwen3VLGdarConfig(text_config=dict(TEXT), vision_config=dict(VISION), recur_blocks=2)
    model = Qwen3VLGdarDeepRecurForConditionalGeneration(config).train()
    out = model(**batch)
    assert torch.isfinite(out.loss) and torch.isfinite(out.logits).all()
    out.loss.backward()
    grad = model.model.feedback_gate.grad
    assert grad is not None and float(grad.abs().sum()) > 0, "feedback 门没有梯度"

    model.eval()
    with torch.no_grad():
        model.model.feedback_gate.fill_(1.0)  # 门初始为 0（tanh(0)=0），拉大再看 feedback 的效果
        loss_feedback_on = float(model(**batch).loss)
        model.model.do_feedback = False
        loss_feedback_off = float(model(**batch).loss)
        model.model.do_feedback = True
        model.model.do_reinject = False
        loss_no_reinject = float(model(**batch).loss)
        model.model.do_reinject = True
    d_feedback = abs(loss_feedback_on - loss_feedback_off)
    d_reinject = abs(loss_feedback_on - loss_no_reinject)
    assert d_feedback > 0, "feedback 开关没有改变前向"
    assert d_reinject > 0, "reinject 开关没有改变前向"
    return (
        f"loss={float(out.loss.detach()):.3f}，feedback 门有梯度；feedback 生效（Δ={d_feedback:.3e}）、"
        f"reinject 生效（Δ={d_reinject:.3e}）"
    )


def check_pt_fairness() -> str:
    """公平口径：refresh_alignment_weights 只重随机 projector/deepstack，其余权重逐位不动。"""
    from transformers import Qwen3VLForConditionalGeneration

    from ...model import refresh_alignment_weights

    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    ).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.5)  # 先扰动：新模型的 LN=1/0、bias=0 本来就是默认值，不然看不出重随机
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    counts = refresh_alignment_weights(model)

    prefixes = ("model.visual.merger", "model.visual.deepstack_merger_list")
    touched = set()
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            touched.update(f"{name}.{param}" for param, _ in module.named_parameters(recurse=False))

    not_refreshed = [n for n in touched if torch.equal(before[n], dict(model.named_parameters())[n].detach())]
    collateral = [
        n
        for n, p in model.named_parameters()
        if n not in touched and not torch.equal(before[n], p.detach())
    ]
    assert sum(counts.values()) > 0, "没有重随机任何对齐件"
    assert not not_refreshed, f"该重随机的没变：{not_refreshed[:3]}"
    assert not collateral, f"不该动的被改了：{collateral[:3]}"
    return (
        f"重随机 {sum(counts.values())} 个对齐件（merger/deepstack）；"
        f"其余 {len(before) - len(touched)} 个张量逐位不动"
    )


CHECKS = (
    check_boundaries,
    check_off_parity,
    check_gdar_forward_backward,
    check_block_rows,
    check_deeprecur,
    check_pt_fairness,
)


def main() -> int:
    print("[deeprecur·gdar] 逐项闸门：")
    for check in CHECKS:
        print(f"  ✓ {check.__name__}: {check()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
