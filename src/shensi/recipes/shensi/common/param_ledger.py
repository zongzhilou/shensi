#!/usr/bin/env python3
"""参数账闸门：构建后立刻对账 tie_mlp 的共享关系，别让"参数悄悄翻倍"拖到训练时才发现。

背景（2026-10-11 机上事件）：8 卡实测总参 4.53B，与账面 2.2930B 对不上，差额恰好等于
"16 组专家未共享 + 3 张 deepemb 未共表"的算术——即 tie_mlp/共表疑似没在某个构建路径上
生效，而 ERC 的 groups=8/8 又像是对的。本闸门把这件事变成构建期的硬检查：

- 专家组：每个 attention-residual 块的 MoE 层必须共享**同一个** experts 对象
  （块的 MoE 层数 >1 时才构成"共享"，单层块自然独占）；
- deepemb：在场 ≥2 个 hash 层时必须共享**同一张**表；
- 账面数字：dedup 总参 / state_dict 总参 / 每 token 激活（给了词表与 top-k 时）。

对 HF（`model.model.layers`）与 Bridge（`model.decoder.layers`，按 `layer_number`
还原全局层号，PP 分 stage 构建也成立）通用。MTP 块不在审计范围（它的专家组独立是设计）。
"""

from __future__ import annotations


def block_layer_types(num_layers: int, n_hash: int, block_size: int) -> list[str]:
    """全栈的块写入计划（与两仓的 `attn_res_block_layer_types` 同一条规则）。"""
    return [
        "block_write_layer"
        if i == 0 or (i >= n_hash and (i - n_hash) % block_size == 0)
        else "block_read_layer"
        for i in range(num_layers)
    ]


def _iter_layers(model):
    """返回 (层列表, 全局层号列表)。HF 走 model.model.layers，Bridge 走 model.decoder.layers。"""
    stack = getattr(getattr(model, "model", None), "layers", None)
    if stack is None:
        stack = getattr(getattr(model, "decoder", None), "layers", None)
    if stack is None:
        raise ValueError(f"找不到层列表（model.model.layers / model.decoder.layers）：{type(model).__name__}")
    layers = list(stack)
    indices = [int(getattr(layer, "layer_number", i + 1)) - 1 for i, layer in enumerate(layers)]
    return layers, indices


def audit_layers(
    layers,
    global_indices,
    *,
    mlp_layer_types: list[str],
    attn_res_block_layer_types: list[str],
) -> dict:
    """对一层列表审计共享关系；违约抛 ValueError（信息直接指向修法）。"""
    if len(global_indices) != len(layers):
        raise ValueError(f"层号与层数不一致：{len(global_indices)} vs {len(layers)}")
    n_total = len(mlp_layer_types)
    if any(g < 0 or g >= n_total for g in global_indices):
        raise ValueError(f"全局层号越界：{sorted(global_indices)[:8]}…（全栈 {n_total} 层）")

    write_layers = [i for i, t in enumerate(attn_res_block_layer_types) if t == "block_write_layer"]
    moe, hash_layers = [], []
    for g, layer in zip(global_indices, layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if hasattr(mlp, "deepemb"):
            hash_layers.append((g, mlp))
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            bids = [k for k, w in enumerate(write_layers) if w <= g]
            if not bids:
                raise ValueError(
                    f"全局层 {g} 在所有写入层之前（write_layers={write_layers}）："
                    "检查 attn_res_block_size / hash 层数与层计划是否一致"
                )
            moe.append((g, bids[-1], experts))

    mismatches: list[str] = []
    groups: dict[int, set[int]] = {}
    for g, bid, experts in moe:
        groups.setdefault(bid, set()).add(id(experts))
    measured_sets = len({id(experts) for _, _, experts in moe})
    expected_sets = len(groups)
    if measured_sets != expected_sets:
        mismatches.append(
            f"专家组 {measured_sets} != 块数 {expected_sets}——tie_mlp 没生效"
            "（参数会按未共享口径翻倍；先查构建路径有没有绕过 tie）"
        )
    for bid, ids in sorted(groups.items()):
        if len(ids) > 1:
            members = sorted(g for g, b, _ in moe if b == bid)
            mismatches.append(f"块 {bid}（层 {members}）内出现 {len(ids)} 个不同 experts 对象")

    deepemb_ids = {id(mlp.deepemb) for _, mlp in hash_layers}
    if len(hash_layers) >= 2 and len(deepemb_ids) != 1:
        mismatches.append(
            f"deepemb 表 {len(deepemb_ids)} 张（hash 层 {sorted(g for g, _ in hash_layers)}）"
            "——共表没生效（每多一张 ~41M 参数）"
        )

    if mismatches:
        raise ValueError("参数账审计失败：" + "；".join(mismatches))
    return {
        "expert_sets": measured_sets,
        "moe_layers": len(moe),
        "deepemb_tables": len(deepemb_ids) if hash_layers else 0,
        "hash_layers": len(hash_layers),
    }


def _param_totals(model, *, n_routed_experts=None, num_experts_per_tok=None) -> dict:
    dedup = sum(p.numel() for p in model.parameters())
    state_dict = sum(v.numel() for v in model.state_dict().values())
    out = {"total_dedup": dedup, "total_state_dict": state_dict}
    if n_routed_experts and num_experts_per_tok:
        # 路由专家参数 = 全部**去重**的 experts 模块之和（tie 后每组一份）
        expert_params, seen = 0, set()
        for m in model.modules():
            experts = getattr(m, "experts", None)
            if experts is not None and id(experts) not in seen:
                seen.add(id(experts))
                expert_params += sum(p.numel() for p in experts.parameters())
        if expert_params:
            out["active"] = dedup - expert_params * (n_routed_experts - num_experts_per_tok) / n_routed_experts
    return out


def audit_hf_model(model) -> dict:
    """HF 侧入口：类型计划直接取自 config。"""
    cfg = model.config
    report = audit_layers(
        *_iter_layers(model),
        mlp_layer_types=list(cfg.mlp_layer_types),
        attn_res_block_layer_types=list(cfg.attn_res_block_layer_types),
    )
    report.update(
        _param_totals(
            model,
            n_routed_experts=getattr(cfg, "n_routed_experts", None),
            num_experts_per_tok=getattr(cfg, "num_experts_per_tok", None),
        )
    )
    return report


def audit_bridge_model(model) -> dict:
    """Bridge 侧入口：从 ShensiTransformerConfig 还原类型计划；MTP 块不在范围。"""
    cfg = model.config
    n_hash = int(getattr(cfg, "moe_n_hash_layers", 0) or 0)
    block_size = int(getattr(cfg, "attn_res_block_size", 4) or 4)
    num_layers = int(getattr(cfg, "num_layers", 0) or 0)
    types = block_layer_types(num_layers, n_hash, block_size)
    report = audit_layers(*_iter_layers(model), mlp_layer_types=types, attn_res_block_layer_types=types)
    if hasattr(model, "parameters"):
        report["total_dedup"] = sum(p.numel() for p in model.parameters())
    return report
