"""入口参数：`--shensi-*` 旋钮、与 HF config 的几何对拍、检查点相关的告警。

只在解析期做事：不改第三方代码，解析完 `postprocess_args()` 一次性把派生量写回 args。
"""

from __future__ import annotations

import json
import os

from megatron.bridge.models.shensi.transformer_config import (
    inject_shensi_fields_into_args,
    resolve_csa_compress_ratios,
    resolve_moe_n_hash_layers,
)
from megatron.training import print_rank_0

from shensi.utils.dsa import fused_dsa_kernels_available

# HF config.json 字段 → mcore args 字段 → 改了它的 CLI 旋钮（对拍报错时提示用哪个开关）
SHENSI_HF_PARITY_FIELDS = (
    ("n_routed_experts", "num_moe_experts", "--shensi-num-experts"),
    ("num_experts_per_tok", "moe_router_topk", "--shensi-num-experts-per-tok"),
    ("moe_intermediate_size", "moe_ffn_hidden_size", "--shensi-moe-intermediate-size"),
    ("routed_expert_hidden_size", "moe_latent_size", "--shensi-routed-expert-hidden-size"),
    ("head_dim", "v_head_dim", "--shensi-head-dim"),
    ("q_lora_rank", "q_lora_rank", "--shensi-q-lora-rank"),
    ("o_lora_rank", "o_lora_rank", "--shensi-o-lora-rank"),
    ("o_groups", "o_groups", "--shensi-o-groups"),
    ("index_n_heads", "dsa_indexer_n_heads", "--shensi-index-n-heads"),
    ("index_head_dim", "dsa_indexer_head_dim", "--shensi-index-head-dim"),
    ("index_topk", "dsa_indexer_topk", "--shensi-index-topk"),
    ("hc_mult", "num_residual_streams", "--shensi-hc-mult"),
    ("hc_active_streams", "hc_active_streams", "--shensi-hc-active-streams"),
    ("hc_fixed_streams", "hc_fixed_streams", "--shensi-hc-fixed-streams"),
    ("attn_res_block_size", "attn_res_block_size", "--shensi-attn-res-block-size"),
    ("erc_loss_alpha", "erc_loss_alpha", "--shensi-erc-loss-alpha"),
    ("erc_loss_coef", "erc_loss_coef", "--shensi-erc-loss-coef"),
    ("num_hidden_layers", "num_layers", "--num-layers"),
    ("hidden_size", "hidden_size", "--hidden-size"),
    ("num_attention_heads", "num_attention_heads", "--num-attention-heads"),
    ("router_aux_loss_coef", "moe_aux_loss_coeff", "--moe-aux-loss-coeff"),
    ("num_nextn_predict_layers", "mtp_num_layers", "--mtp-num-layers"),
)

SHENSI_LAYER_TYPE_TO_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _extend_choices(parser, dest: str, extra: list[str]) -> None:
    """给上游已有的选项补可选值（不覆盖上游定义）。"""
    for action in parser._actions:
        if action.dest == dest and action.choices:
            action.choices = [*action.choices, *(c for c in extra if c not in action.choices)]


def add_shensi_args(parser):
    """把 Shensi 家族的旋钮加到 mcore 的 parser 上。

    上游已经有同名的参数（`--muon-scalar-optimizer`、`--csa-compress-ratios` 这类）就不再重复添加，
    只按需补 choices——重复添加 argparse 会直接报冲突。
    """
    group = parser.add_argument_group(title="shensi", description="Shensi 家族参数")
    # AdEMAMix 走 emerging_optimizers 的名字注册表（shensi.utils.optimizer.ademamix 导入即注册）
    _extend_choices(parser, "optimizer", ["ademamix"])
    _extend_choices(parser, "muon_scalar_optimizer", ["ademamix"])
    # 名字与 pytorch_optimizer.AdEMAMix 的构造参数一致：mcore 按 `{名字}_{参数}` 从 config 取
    group.add_argument(
        "--ademamix-betas",
        nargs=3,
        type=float,
        default=(0.9, 0.999, 0.9999),
        help="AdEMAMix 的 (beta_fast, beta2, beta_slow)",
    )
    group.add_argument(
        "--ademamix-alpha", type=float, default=5.0, help="AdEMAMix 慢 EMA 在更新里的权重"
    )
    group.add_argument(
        "--ademamix-t-alpha-beta3",
        type=int,
        default=0,
        help="alpha / beta_slow 的 warmup 步数（库里的 t_alpha_beta3）",
    )
    group.add_argument(
        "--shensi-attn-layer-types",
        type=str,
        default="",
        help="逗号分隔：sliding_attention / compressed_sparse_attention / heavily_compressed_attention",
    )
    group.add_argument(
        "--shensi-mlp-layer-types", type=str, default="", help="逗号分隔：hash_moe / moe"
    )
    group.add_argument(
        "--shensi-compress-ratios",
        type=str,
        default="",
        help="逐层压缩比（0=滑窗 / 4=CSA / 128=HCA），长度含 MTP 层；接受逗号分隔的整数，"
        "或上游那种列表表达式 `[0,0]+[4,128]*15+[4]+[0,0,0]`。与 --shensi-attn-layer-types 二选一",
    )
    group.add_argument("--shensi-hc-mult", type=int, default=16)
    group.add_argument("--shensi-hc-active-streams", type=int, default=4)
    group.add_argument("--shensi-hc-fixed-streams", type=int, default=2)
    group.add_argument("--shensi-hc-conv-kernels", nargs="+", type=int, default=[4, 8, 12])
    group.add_argument("--shensi-o-groups", type=int, default=4)
    group.add_argument("--shensi-o-lora-rank", type=int, default=0)
    group.add_argument(
        "--shensi-sliding-window",
        type=int,
        default=128,
        help="滑窗宽度（默认 = HF 的 128；tiny 侧要在 yaml 显式写小值）",
    )
    group.add_argument("--shensi-head-dim", type=int, default=0)
    group.add_argument("--shensi-q-lora-rank", type=int, default=0)
    group.add_argument("--shensi-partial-rotary-factor", type=float, default=0.0)
    group.add_argument("--shensi-routed-expert-hidden-size", type=int, default=None)
    group.add_argument(
        "--shensi-index-topk",
        type=int,
        default=512,
        help="indexer 的 top-k（默认 = HF 的 512；tiny 侧要在 yaml 显式写小值）",
    )
    group.add_argument("--shensi-index-n-heads", type=int, default=0)
    group.add_argument("--shensi-index-head-dim", type=int, default=0)
    group.add_argument(
        "--shensi-num-experts",
        type=int,
        default=256,
        help="路由专家数（默认 = HF 的 256；tiny 侧要在 yaml 显式写小值）",
    )
    group.add_argument(
        "--shensi-num-experts-per-tok",
        type=int,
        default=6,
        help="每 token 激活专家数（默认 = HF 的 6；tiny 侧要在 yaml 显式写小值）",
    )
    group.add_argument("--shensi-moe-intermediate-size", type=int, default=None)
    group.add_argument(
        "--shensi-router-aux-loss-coef",
        type=float,
        default=None,
        help="MoE 路由 aux loss 系数；不给就用 HF 声明的 router_aux_loss_coef（0.001），"
        "显式给 0 可关掉（HF 参考只在 output_router_logits=True 时加它）",
    )
    group.add_argument(
        "--shensi-freeze",
        type=str,
        choices=("none", "indexer", "non-indexer", "mtp", "non-mtp"),
        default="none",
        help="冻参数组：indexer = 只训 Lightning Indexer（DSA dense warmup，主干冻结）；"
        "non-indexer = 冻结 indexer（RL 阶段按论文冻 indexer）；mtp = 只训 MTP 头（DeepSpec 口径）；"
        "non-mtp = 冻结 MTP 头；none = 不冻",
    )
    group.add_argument(
        "--shensi-indexer-loss-coeff",
        type=float,
        default=None,
        help="DSA indexer 的 KL 损失系数（论文 §2.1 的 L^I）；不给就用家族默认 0.01，"
        "显式给 0 可关掉",
    )
    group.add_argument("--shensi-attn-res-block-size", type=int, default=4)
    group.add_argument("--shensi-erc-loss-coef", type=float, default=1.0)
    group.add_argument("--shensi-erc-loss-alpha", type=float, default=0.5)
    group.add_argument(
        "--shensi-hf-config",
        type=str,
        default="",
        help="HF 侧的 config.json（或含它的目录）。给了就与 --shensi-* 解析出的几何逐字段"
        "对拍，缺失/不一致直接报错；不给时会尝试从 --load / --pretrained-checkpoint 自动找",
    )
    group.add_argument(
        "--shensi-pure-weight-auto-downgrade",
        dest="shensi_pure_weight_auto_downgrade",
        action="store_true",
        default=False,
        help="纯权重检查点（pure_weight.json）且未给 --no-load-optim/--no-load-rng 时，"
        "自动把这两个开关置 True（旧行为，需显式打开；默认只告警并提示上游开关）",
    )
    group.add_argument(
        "--no-shensi-hc-fp32-keep",
        dest="shensi_hc_fp32_keep",
        action="store_false",
        default=True,
        help="关掉 mHC/AttnRes/hc_head 在 bf16 下的 fp32 保持（默认保持，对齐 HF）",
    )
    group.add_argument(
        "--no-shensi-attn-res-pp-state-transfer",
        dest="shensi_attn_res_pp_state_transfer",
        action="store_false",
        default=True,
        help="关掉 AttnRes block 状态跨 PP stage 交接（默认开；关掉时 PP>1 会被布局校验拦下）",
    )
    group.add_argument(
        "--shensi-erc-ep-legacy-local",
        dest="shensi_erc_ep_legacy_local",
        action="store_true",
        default=False,
        help="EP>1 时退回旧的『逐卡本地 ERC』口径（对照/诊断用；默认走全局 all-gather）",
    )
    return parser


def apply_shensi_compress_ratios(args) -> None:
    """`--shensi-compress-ratios` → 上游的 `--csa-compress-ratios`（逐层压缩比）。"""
    from megatron.training.arguments import _eval_pattern

    raw = str(getattr(args, "shensi_compress_ratios", "") or "").strip()
    if not raw:
        return
    if getattr(args, "csa_compress_ratios", None):
        raise ValueError("--shensi-compress-ratios 与上游 --csa-compress-ratios 都给了，二选一")
    if "[" in raw:
        values = [int(v) for v in _eval_pattern(raw)]
    else:
        values = [int(v) for v in raw.replace(" ", "").split(",") if v]
    args.csa_compress_ratios = values


def apply_shensi_hf_rope_scaling(args) -> None:
    """HF 侧 compressor 层用 YaRN：没显式给就照 HF config 填上游的 YaRN 参数。"""
    path = find_hf_shensi_config(args)
    if path is None:
        return
    with open(path) as f:
        info = json.load(f)
    rp = info.get("rope_parameters") or info.get("rope_scaling") or {}
    if not isinstance(rp, dict) or not rp:
        return
    compress = rp.get("compress")
    if not isinstance(compress, dict):
        compress = {k: v for k, v in rp.items() if k not in ("main", "compress")}
    if str(compress.get("rope_type", compress.get("type", "default"))) != "yarn":
        return
    for attr, key, cast, default in (
        ("rotary_scaling_factor", "factor", float, 1.0),
        ("original_max_position_embeddings", "original_max_position_embeddings", int, 4096),
        ("beta_fast", "beta_fast", float, 32.0),
        ("beta_slow", "beta_slow", float, 1.0),
    ):
        if key not in compress:
            continue
        cur = getattr(args, attr, None)
        if cur in (None, 0, default):
            setattr(args, attr, cast(compress[key]))
    if getattr(args, "mscale", None) in (None, 0, 1.0):
        args.mscale = 1.0


def find_hf_shensi_config(args) -> str | None:
    """找 HF 侧的 config.json：显式给的最优先，其次从 --load / --pretrained-checkpoint 里认。"""
    cands: list[str] = []
    for raw in (
        getattr(args, "shensi_hf_config", "") or "",
        os.environ.get("SHENSI_HF_CONFIG", "") or "",
    ):
        if raw:
            cands.append(raw)
            cands.append(os.path.join(raw, "config.json"))
    for attr in ("load", "pretrained_checkpoint"):
        p = getattr(args, attr, None)
        if p:
            cands.append(os.path.join(str(p), "config.json"))
    for path in cands:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                info = json.load(f)
        except Exception:
            continue
        arch = " ".join(info.get("architectures") or [])
        if str(info.get("model_type", "")).lower() == "shensi" or "Shensi" in arch:
            return path
    return None


def verify_shensi_geometry(args, hf_config_path: str | None = None) -> dict:
    """与 HF config.json 逐字段对拍几何：不一致直接报错（不给就跳过并告警）。"""
    path = hf_config_path or find_hf_shensi_config(args)
    defaulted = sorted(
        f"{knob}={getattr(args, args_attr, None)}"
        for _, args_attr, knob in SHENSI_HF_PARITY_FIELDS
        if getattr(args, knob.lstrip("-").replace("-", "_"), None) in (None, 0, 0.0)
    )
    if path is None:
        print_rank_0(
            "[shensi][geom] [warn] 未找到 HF config.json（--shensi-hf-config / $SHENSI_HF_CONFIG "
            "/ --load/config.json）→ **跳过几何对拍**，本 run 的几何完全按解析结果用。"
            "生产请显式给 `--shensi-hf-config <HF ckpt 目录>`（或让 --load 指向 HF 权重目录）"
            f"以启用逐字段对拍。当前『未显式给、按 hidden 派生』的旋钮：{defaulted or '无'}"
        )
        return {"status": "skipped", "hf_config": None, "checked": 0, "defaulted": defaulted}
    with open(path) as f:
        hf = json.load(f)
    bad: list[str] = []
    checked = 0

    def _cmp_num(hf_field, ours, knob):
        nonlocal checked
        if hf_field not in hf:
            bad.append(f"{hf_field}: HF config **缺该字段**（{path}）")
            return
        theirs = hf[hf_field]
        if ours is None:
            ours = 0
        try:
            same = float(ours) == float(theirs)
        except (TypeError, ValueError):
            same = ours == theirs
        if same:
            checked += 1
        else:
            bad.append(f"{hf_field}: 本入口解析出 {ours}，HF config 是 {theirs}（用 {knob} 改）")

    for hf_field, args_attr, knob in SHENSI_HF_PARITY_FIELDS:
        _cmp_num(hf_field, getattr(args, args_attr, None), knob)
    ours_ratios = list(getattr(args, "csa_compress_ratios", None) or [])
    hf_types = [str(t) for t in (hf.get("layer_types") or [])]
    if not hf_types:
        bad.append("layer_types: HF config **缺该字段**")
    else:
        unknown = [t for t in hf_types if t not in SHENSI_LAYER_TYPE_TO_RATIO]
        if unknown:
            bad.append(f"layer_types: HF 侧出现未知类型 {unknown}（无法推出压缩比）")
        else:
            hf_ratios = resolve_csa_compress_ratios(hf_types)
            if ours_ratios[: len(hf_ratios)] == hf_ratios:
                checked += 1
            else:
                bad.append(
                    f"layer_types: 本入口 csa_compress_ratios={ours_ratios}（前 "
                    f"{len(hf_ratios)} 项）!= HF {hf_types} -> {hf_ratios}"
                    "（用 --shensi-attn-layer-types 改）"
                )
    hf_mlp = [str(t) for t in (hf.get("mlp_layer_types") or [])]
    if not hf_mlp:
        bad.append("mlp_layer_types: HF config **缺该字段**")
    else:
        hf_n_hash = resolve_moe_n_hash_layers(hf_mlp)
        if int(getattr(args, "moe_n_hash_layers", -1) or -1) != int(hf_n_hash):
            bad.append(
                f"mlp_layer_types: HF {hf_mlp} 推得 hash 前缀 {hf_n_hash}，本入口 "
                f"moe_n_hash_layers={getattr(args, 'moe_n_hash_layers', None)}"
                "（用 --shensi-mlp-layer-types 改）"
            )
    if "sliding_window" in hf:
        ours_w = getattr(args, "csa_window_size", None)
        theirs_w = hf["sliding_window"]
        if ours_w is not None and float(ours_w) <= float(theirs_w):
            checked += 1
        else:
            bad.append(
                f"sliding_window: 本入口 csa_window_size={ours_w} > HF {theirs_w}"
                "（该字段只允许向下夹到 min(seq_length, w)；用 --shensi-sliding-window 改）"
            )
    else:
        bad.append("sliding_window: HF config **缺该字段**")
    if "vocab_size" in hf:
        ours_v = int(getattr(args, "actual_vocab_size", 0) or 0)
        theirs_v = int(hf["vocab_size"])
        if theirs_v > 0 and ours_v >= theirs_v and ours_v % theirs_v == 0:
            checked += 1
        else:
            bad.append(
                f"vocab_size: 本入口 actual_vocab_size={ours_v} 与 HF {theirs_v} 不是"
                "『向上 padding 且整除』的关系"
            )
    else:
        bad.append("vocab_size: HF config **缺该字段**")
    rp = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
    compress = rp.get("compress") if isinstance(rp, dict) else None
    if not isinstance(compress, dict):
        compress = (
            {k: v for k, v in rp.items() if k not in ("main", "compress")}
            if isinstance(rp, dict)
            else {}
        )
    rtype = str(compress.get("rope_type", compress.get("type", "default")))
    ours_factor = float(getattr(args, "rotary_scaling_factor", 1.0) or 1.0)
    if rtype != "yarn":
        if ours_factor != 1.0:
            bad.append(
                f"rope_parameters.compress.rope_type: HF 是 {rtype}（无 YaRN），本入口 "
                f"rotary_scaling_factor={ours_factor}（用 --rotary-scaling-factor 1.0 或改 HF 侧）"
            )
        else:
            checked += 1
    elif "factor" not in compress:
        bad.append("rope_parameters.compress.factor: HF 侧 rope_type=yarn 但缺 factor")
    elif abs(ours_factor - float(compress["factor"])) > 1e-9:
        bad.append(
            f"rope_parameters.compress.factor: 本入口 rotary_scaling_factor={ours_factor}，"
            f"HF 是 {compress['factor']}（用 --rotary-scaling-factor 改）"
        )
    else:
        checked += 1
        theirs_orig = compress.get("original_max_position_embeddings")
        if theirs_orig is not None:
            ours_orig = int(getattr(args, "original_max_position_embeddings", 0) or 0)
            if ours_orig == int(theirs_orig):
                checked += 1
            else:
                bad.append(
                    "rope_parameters.compress.original_max_position_embeddings: 本入口 "
                    f"original_max_position_embeddings={ours_orig}，HF 是 {theirs_orig}"
                    "（用 --original-max-position-embeddings 改）"
                )
    if bad:
        raise RuntimeError(
            f"[shensi][geom] 与 HF config 对拍**失败**（{path}）：\n  - "
            + "\n  - ".join(bad)
            + "\n处置：① 在 yaml/CLI 里把对应 `--shensi-*` 写成 HF config 的值；"
            "② 或者确认 HF 侧 config.json 才是权威（改 HF 侧）；"
            "③ 纯粹做对照实验时可显式传 `--shensi-hf-config ''` 关掉对拍（会打 [warn]）。"
        )
    print_rank_0(
        f"[shensi][geom] 与 HF config 对拍 PASS：{checked} 个字段逐字段一致（{path}）｜"
        f"未显式给、按 hidden 派生的旋钮：{defaulted or '无'}"
    )
    return {"status": "ok", "hf_config": path, "checked": checked, "defaulted": defaulted}


def warn_missing_checkpoint_tracker(args) -> None:
    """`--load` 目录里没有迭代记账文件时提醒：上游会静默退化成随机初始化。"""
    load_dir = getattr(args, "load", None)
    if not load_dir:
        return
    if not os.path.isdir(load_dir):
        return
    tracker = os.path.join(load_dir, "latest_checkpointed_iteration.txt")
    has_iter = any(
        n.startswith("iter_") and os.path.isdir(os.path.join(load_dir, n))
        for n in os.listdir(load_dir)
    )
    if os.path.isfile(tracker) or has_iter:
        return
    print_rank_0(
        f"[shensi][ckpt] [warn] --load {load_dir} 里没有 latest_checkpointed_iteration.txt / "
        "iter_* 目录：上游会**静默**退化成『从随机初始化开始训』（checkpointing.py 只打一句 "
        "info）。要让它变成硬报错请加**上游开关** `--exit-on-missing-checkpoint`；"
        "本入口不替你开（那会改语义）。"
    )


def _pure_weight_marker_of(load_path) -> str | None:
    if not load_path:
        return None
    base = load_path.rstrip("/\\")
    cands = [
        os.path.join(base, "pure_weight.json"),
        os.path.join(os.path.dirname(base), "pure_weight.json"),
    ]
    for c in cands:
        if os.path.isfile(c):
            try:
                with open(c) as f:
                    info = json.load(f)
            except Exception:
                continue
            keys = set(info.get("top_level_keys") or [])
            if info.get("pure_weight") and "optimizer" not in keys:
                return c
    return None


def _auto_downgrade_requested(args) -> bool:
    if bool(getattr(args, "shensi_pure_weight_auto_downgrade", False)):
        return True
    return str(os.environ.get("SHENSI_PURE_WEIGHT_AUTO_DOWNGRADE", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def warn_pure_weight_load(args, downgrade: bool = False) -> None:
    """纯权重检查点（无 optimizer/rng）会撞上游的 KeyError：要么给上游开关，要么显式降级。"""
    marker = _pure_weight_marker_of(getattr(args, "load", None))
    if not marker:
        return
    flags_ok = bool(getattr(args, "no_load_optim", False)) and bool(
        getattr(args, "no_load_rng", False)
    )
    if flags_ok:
        print_rank_0(
            f"[shensi][ckpt] --load {args.load} 是**纯权重**检查点（见 {marker}）；已按上游"
            "要求给出 --no-load-optim/--no-load-rng，原样不动。"
        )
        return
    old = (getattr(args, "no_load_optim", None), getattr(args, "no_load_rng", None))
    if downgrade:
        args.no_load_optim = True
        args.no_load_rng = True
        print_rank_0(
            f"[shensi][ckpt] [warn] --load {args.load} 是**纯权重**检查点（见 {marker}），"
            f"而 --shensi-pure-weight-auto-downgrade 被显式打开 -> 按旧行为把 "
            f"no_load_optim/no_load_rng 从 {old} 置为 (True, True)。"
            "注意：这是**入口侧**的兼容降级，不是上游默认；生产建议直接用上游开关。"
        )
        return
    print_rank_0(
        f"[shensi][ckpt] [warn] --load {args.load} 是**纯权重**检查点（见 {marker}：顶层键里"
        f"没有 optimizer/opt_param_scheduler/rng_state），而 no_load_optim/no_load_rng="
        f"{old} —— mcore 会在 state_dict['optimizer'] 上 KeyError。**本入口不再替你改参数**"
        "（那会静默改语义）。请二选一：① 上游开关 `--no-load-optim --no-load-rng`；"
        "② 上游开关 `--finetune`（同样跳过 optimizer/rng 状态）。"
        "确实要旧行为时显式加 `--shensi-pure-weight-auto-downgrade`。"
    )


def _fused_dsa_kernels_available() -> bool:
    """上游的 cudnn DSA 融合内核要的两个包是否都在。"""
    return fused_dsa_kernels_available()


def apply_dsa_kernel_backend_fallback(args) -> None:
    """`dsv4_hybrid` 会默认选 cudnn 的融合 DSA 内核；本机没装就退回 PyTorch 路径。

    显式给了 `--dsa-kernel-backend` 就不动（上游会自己报缺什么包）。
    """
    if getattr(args, "dsa_kernel_backend", None):
        return
    if _fused_dsa_kernels_available():
        return
    args.dsa_kernel_backend = "none"
    print_rank_0(
        "[shensi][dsa] 本机没有 flash_mla / nvidia-cudnn-frontend[cutedsl] → "
        "dsa_kernel_backend=none（走 PyTorch 回退实现，数值同口径、速度慢）。"
        "生产上装好融合内核后去掉这个回退即可（上游会自动选 cudnn）。"
    )


def disable_dataloader_attention_mask(args) -> None:
    """本家族用 CSA：它只接受隐式 causal（`attention_mask` 必须为 None），所以关掉 dataloader 造掩码。

    上游的 `--no-create-attention-mask-in-dataloader` 就是干这个的；这里显式给个提示，
    免得直接撞在 CSA 的 forward 断言上。
    """
    if not getattr(args, "create_attention_mask_in_dataloader", False):
        return
    args.create_attention_mask_in_dataloader = False
    print_rank_0(
        "[shensi][mask] 已关掉 dataloader 的 attention mask（→ 等价于 "
        "`--no-create-attention-mask-in-dataloader`）：CSA 只支持隐式 causal mask，"
        "收到显式 mask 会直接报错。要自己控制就显式加这个上游开关。"
    )


def postprocess_args(args) -> None:
    """解析完 args 后固定顺序的收尾：压缩比 → YaRN → 派生量 → DSA 后端 → mask → 几何对拍 → 检查点告警。"""
    apply_shensi_compress_ratios(args)
    apply_shensi_hf_rope_scaling(args)
    inject_shensi_fields_into_args(args)
    apply_dsa_kernel_backend_fallback(args)
    disable_dataloader_attention_mask(args)
    verify_shensi_geometry(args)
    warn_pure_weight_load(args, downgrade=_auto_downgrade_requested(args))
    warn_missing_checkpoint_tracker(args)
