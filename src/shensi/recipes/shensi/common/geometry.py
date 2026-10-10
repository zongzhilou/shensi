"""几何单一来源：`common/config/hf/<名字>.json` → `train.model.*` 的 CLI 覆写列表。

与 `tiny_model.geometry_overrides` 同一套做法（YAML 合并之后、`--set` 之前注入，显式覆写仍然能盖掉），
区别只是几何来自 HF config.json 而不是写死在代码里：**改几何只改那份 JSON**，各 stage 的 profile 只写
一行 `geometry: <名字>`，训练侧入口的 `verify_shensi_geometry` 会逐字段对拍兜底（不一致直接报错）。

覆盖的字段 = 入口几何对拍的 22 项 + 层计划（压缩比 / MLP 类型）+ sliding / vocab / rope / hf_config 指针
（多注入的键要么被上游忽略、要么就是 profile 本来要写的；**不注入** seq_length / 批大小 / LR 这类
训练口径，它们按 stage 与预算各自给）。
"""

from __future__ import annotations

import json
from pathlib import Path

HF_DIR = Path(__file__).resolve().parent / "config" / "hf"

__all__ = ["available", "geometry_overrides", "load_json"]


def available() -> list[str]:
    """可用的几何名（`config/hf/*.json` 的文件名，去后缀）。"""
    return sorted(p.stem for p in HF_DIR.glob("*.json"))


def load_json(name: str) -> dict:
    path = HF_DIR / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"[recipe] 没有这个几何：{name}；可用：{available()}（{HF_DIR}）")
    return json.loads(path.read_text(encoding="utf-8"))


def _int_if_integral(value):
    """`--rotary-base` / `--max-position-embeddings` 这类上游是 int：JSON 里写成 10000.0 要收成 int。"""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    return int(as_float) if as_float.is_integer() else as_float


def geometry_overrides(name: str) -> list[str]:
    """把 HF 几何翻成 `点号键=值` 覆写（供 build_config 注入）。"""
    hf = load_json(name)
    model = {
        # 与 verify_shensi_geometry 的 22 项对拍字段一一对应
        "num_layers": hf["num_hidden_layers"],
        "hidden_size": hf["hidden_size"],
        "num_attention_heads": hf["num_attention_heads"],
        "mtp_num_layers": hf["num_nextn_predict_layers"],
        "shensi_num_experts": hf["n_routed_experts"],
        "shensi_num_experts_per_tok": hf["num_experts_per_tok"],
        "shensi_moe_intermediate_size": hf["moe_intermediate_size"],
        "shensi_routed_expert_hidden_size": hf["routed_expert_hidden_size"],
        "shensi_head_dim": hf["head_dim"],
        "shensi_q_lora_rank": hf["q_lora_rank"],
        "shensi_o_lora_rank": hf["o_lora_rank"],
        "shensi_o_groups": hf["o_groups"],
        "shensi_index_n_heads": hf["index_n_heads"],
        "shensi_index_head_dim": hf["index_head_dim"],
        "shensi_index_topk": hf["index_topk"],
        "shensi_hc_mult": hf["hc_mult"],
        "shensi_hc_active_streams": hf["hc_active_streams"],
        "shensi_hc_fixed_streams": hf["hc_fixed_streams"],
        "shensi_attn_res_block_size": hf["attn_res_block_size"],
        "moe_aux_loss_coeff": hf["router_aux_loss_coef"],
        "shensi_erc_loss_alpha": hf["erc_loss_alpha"],
        "shensi_erc_loss_coef": hf["erc_loss_coef"],
        # 层计划（名字式与数值式都给，入口二选一都能对上）
        "shensi_compress_ratios": ",".join(str(v) for v in hf["compress_ratios"]),
        "shensi_mlp_layer_types": ",".join(str(t) for t in hf["mlp_layer_types"]),
        # 注意力窗口 / 位置 / 词表 / rope（与 HF 侧同源）
        "shensi_sliding_window": hf["sliding_window"],
        "max_position_embeddings": hf["max_position_embeddings"],
        "rotary_base": _int_if_integral(hf["rope_theta"]),
        "shensi_partial_rotary_factor": hf["partial_rotary_factor"],
    }
    compress = (hf.get("rope_parameters") or {}).get("compress") or {}
    if str(compress.get("rope_type", "")) == "yarn":
        model["rotary_scaling_factor"] = compress.get("factor", 1.0)
        model["original_max_position_embeddings"] = compress.get(
            "original_max_position_embeddings", hf["max_position_embeddings"]
        )
    items = [f"train.model.{k}={v}" for k, v in model.items()]
    items.append(f"train.data.tokenizer.vocab_size={hf['vocab_size']}")
    # 与 9B 档同一位点（train.model → `--shensi-hf-config`），保证几何对拍能找到这份 JSON
    items.append(f"train.model.shensi_hf_config=${{experiment.shensi_hf_dir}}/{name}.json")
    return items
