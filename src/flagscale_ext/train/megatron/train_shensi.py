# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


import json
import os

import torch
import torch.nn.functional as F

from flagscale.train.megatron import train_gpt
from shensi import runtime  # noqa: F401  导入即登记/补齐第三方要的东西

from megatron_ext.core.models.shensi import (
    ShensiModel,
    get_shensi_decoder_block_spec,
    get_shensi_mtp_layer_spec,
)
from megatron_ext.core.transformer.shensi.transformer_config import inject_shensi_fields_into_args
from megatron.core.utils import unwrap_model
from megatron.training import get_args, print_rank_0

# 导入即把 AdEMAMix 注册进 emerging_optimizers 的注册表（上游按名字取标量优化器，
# `muon_scalar_optimizer="ademamix"` 走的就是它）
import shensi.utils.optimizer.ademamix  # noqa: F401,E402

from shensi.utils.ckpt_digest import (
    digest_state_dict,
    sample_keys,
    summarize,
    tensor_digest,
    weight_keys,
)


def erc_loss_func(
    router_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    alpha: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    R = router_weight
    norm_R = torch.norm(R, dim=1)
    distances = torch.cdist(R, R, p=2)
    distances = distances.masked_fill(
        torch.eye(R.size(0), dtype=torch.bool, device=distances.device), float("inf")
    )
    min_dist, _ = torch.min(distances, dim=1)
    eps = min_dist / 2 / norm_R
    low = (1 - eps).unsqueeze(1)
    high = (1 + eps).unsqueeze(1)
    if noise is None:
        noise = torch.rand_like(R)
    R_tilde = (low + noise * (high - low)) * R
    proxy = F.linear(R_tilde, down_proj_weight)
    M = torch.norm(torch.einsum("jDd,id->ijD", gate_up_proj, proxy), dim=-1)
    row_diff = M - alpha * torch.diag(M).unsqueeze(1)
    row_diff_clamped = torch.clamp(row_diff, min=0.0)
    col_diff = M - alpha * torch.diag(M).unsqueeze(0)
    col_diff_clamped = torch.clamp(col_diff, min=0.0)
    mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
    total_diff = (row_diff_clamped + col_diff_clamped) * mask
    return total_diff.mean()


def _print_erc(msg: str) -> None:
    try:
        from megatron.training import print_rank_last

        print_rank_last(msg)
    except Exception:
        print(msg, flush=True)


_ERC_GROUP_AUDIT_LOGGED = set()


def _resolve_ep():
    try:
        from megatron.core import parallel_state as _ps

        if not _ps.is_initialized():
            return 1, 0, None
        size = int(_ps.get_expert_model_parallel_world_size())
        if size <= 1:
            return 1, 0, None
        return (
            size,
            int(_ps.get_expert_model_parallel_rank()),
            _ps.get_expert_model_parallel_group(),
        )
    except Exception:
        return 1, 0, None


def _erc_noise_for(router_weight: torch.Tensor) -> torch.Tensor | None:
    ep_size, _, ep_group = _resolve_ep()
    if ep_size <= 1 or ep_group is None:
        return None
    if str(os.environ.get("SHENSI_ERC_EP_SYNC_NOISE", "1")).strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    import torch.distributed as dist

    src = dist.get_global_rank(ep_group, 0)
    if dist.get_rank() == src:
        noise = torch.rand_like(router_weight.detach().float())
    else:
        noise = torch.empty_like(router_weight.detach().float())
    dist.broadcast(noise, src=src, group=ep_group)
    return noise


def check_erc_ep_support(coef: float) -> None:
    ep_size = 1
    try:
        from megatron.core import parallel_state as _ps

        if _ps.is_initialized():
            ep_size = int(_ps.get_expert_model_parallel_world_size())
    except Exception:
        ep_size = 1
    if ep_size <= 1 or float(coef) == 0.0:
        return
    allow_local = str(os.environ.get("SHENSI_ERC_EP_ALLOW_LOCAL", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if allow_local:
        _print_erc(
            f"[shensi][erc] [warn] SHENSI_ERC_EP_ALLOW_LOCAL=1：EP={ep_size}>1 时退回"
            "**逐卡本地** ERC（耦合矩阵只覆盖本地专家、梯度只进本地专家）—— 仅供诊断/对照，"
            "训练请用默认的全局口径。"
        )
        return
    _print_erc(
        f"[shensi][erc] EP={ep_size}>1：走**全局**口径（moe.erc_gather_expert_weights 的 "
        "all-gather + 反向按 rank 切片回切；各 rank 噪声同步 -> ERC 标量一致）。"
        "需要逐卡本地口径做对照时设 SHENSI_ERC_EP_ALLOW_LOCAL=1。"
    )


def erc_group_plan(cfg) -> dict[int, list[int]]:
    from megatron_ext.core.transformer.shensi.attn_res import attn_res_block_layer_types

    num_layers = int(getattr(cfg, "num_layers", 0) or 0)
    if num_layers <= 0:
        return {}
    n_hash = int(getattr(cfg, "moe_n_hash_layers", 0) or 0)
    block_size = int(getattr(cfg, "attn_res_block_size", 0) or 0)
    if block_size <= 0:
        return {}
    types = attn_res_block_layer_types(num_layers, n_hash, block_size)
    write_layers = [i for i, t in enumerate(types) if t == "block_write_layer"]
    if not write_layers:
        return {}
    groups: dict[int, list[int]] = {}
    for i in range(num_layers):
        if i < n_hash:
            continue
        gid = max(w for w in write_layers if w <= i)
        groups.setdefault(gid, []).append(i)
    return groups


def _local_global_layer_indices(model) -> list[int] | None:
    layers = list(getattr(getattr(model, "decoder", None), "layers", None) or [])
    if not layers:
        return None
    nums = [getattr(ly, "layer_number", None) for ly in layers]
    if not all(n for n in nums):
        return None
    return sorted(int(n) - 1 for n in nums)


def check_erc_groups_complete(model, owners, verbose: bool = True) -> dict:
    cfg = getattr(model, "config", None)
    plan = erc_group_plan(cfg) if cfg is not None else {}
    n_owners = len(owners or {})
    if not plan:
        if verbose and "skip_no_cfg" not in _ERC_GROUP_AUDIT_LOGGED:
            _ERC_GROUP_AUDIT_LOGGED.add("skip_no_cfg")
            _print_erc(
                "[shensi][erc] [warn] 全局分组对账**跳过**：config 里没有 "
                "num_layers/moe_n_hash_layers/attn_res_block_size（离线脚本或伪造 model）"
            )
        return {"status": "skipped", "global_groups": 0, "local_groups": n_owners, "layers": []}
    local_idx = _local_global_layer_indices(model)
    if local_idx is None:
        if verbose and "skip_no_layer_number" not in _ERC_GROUP_AUDIT_LOGGED:
            _ERC_GROUP_AUDIT_LOGGED.add("skip_no_layer_number")
            _print_erc(
                "[shensi][erc] [warn] 全局分组对账**跳过**：拿不到本 stage 的全局层号"
                "（decoder.layers 为空 / layer_number 缺失）"
            )
        return {
            "status": "skipped",
            "global_groups": len(plan),
            "local_groups": n_owners,
            "layers": [],
        }
    have = sorted(gid for gid, ls in plan.items() if set(ls) & set(local_idx))
    missing = sorted(set(plan) - set(have))
    if missing or n_owners != len(have):
        raise RuntimeError(
            "[shensi][erc] ERC 分组**不完整**：本 stage（= 真正算 ERC 的 PP 末 stage）只覆盖 "
            f"{len(have)}/{len(plan)} 个 AttnRes block 分组，缺组 {missing}"
            f"（本地层={local_idx}，moe_group_owners 条目={n_owners}）。"
            "ERC 的语义是『全模型每组算一次、再对全模型组数取平均』"
            "（HF `ShensiForCausalLM.forward`），而 `loss_func` 只在 PP 末 stage 执行 —— "
            "所以末 stage **必须**拥有每一组的参数，否则算出来的不是 HF 口径（会静默按本地"
            "组数平均）。三种处置：① `--shensi-erc-loss-coef 0` 关掉 ERC；② 改 "
            "`pipeline_model_parallel_layout` / ERC 相关层分配，让全部 AttnRes block 的 MoE "
            "层都落在末 stage；③ 需要 ERC 的 PP>1 生产配置改用 `--shensi-erc-loss-coef 0` + "
            "单独的阶段（当前实现不支持跨 stage 的 ERC）。"
        )
    if verbose and "ok" not in _ERC_GROUP_AUDIT_LOGGED:
        _ERC_GROUP_AUDIT_LOGGED.add("ok")
        _print_erc(
            f"[shensi][erc] 全局分组对账 PASS：本 stage 覆盖 {len(have)}/{len(plan)} 组"
            f"（组 = AttnRes block 写层 {have}；全局层号 {local_idx}）→ "
            "平均口径 = HF 的『按全模型组数平均』"
        )
    return {
        "status": "ok",
        "global_groups": len(plan),
        "local_groups": len(have),
        "layers": local_idx,
    }


def shensi_erc_loss(model, alpha: float = 1.0, verbose: bool = False) -> torch.Tensor | None:
    owners = getattr(model, "moe_group_owners", None)
    if not owners:
        _print_erc(
            "[shensi][erc] 本 stage 没有 MoE 组（moe_group_owners 为空）→ ERC 不上报、不反传；"
            "这是**本 rank 的分片事实**（PP>1 的非末 stage 天然如此），不是 erc_loss_coef=0。"
        )
        return None
    cfg = getattr(model, "config", None)
    check_erc_ep_support(float(getattr(cfg, "erc_loss_coef", 0.0) or 0.0))
    audit = check_erc_groups_complete(model, owners, verbose=verbose)
    total = None
    num_groups = 0
    for mlp in owners.values():
        from megatron_ext.core.transformer.shensi.moe import erc_gather_expert_weights

        router_weight, down_proj_weight, gate_up_proj = erc_gather_expert_weights(mlp)
        group_loss = erc_loss_func(
            router_weight.float(),
            down_proj_weight.float(),
            gate_up_proj.float(),
            alpha=alpha,
            noise=_erc_noise_for(router_weight),
        )
        if verbose:
            _print_erc(
                f"[shensi][erc] erc={float(group_loss.detach()):.6e} "
                f"gate_up={tuple(gate_up_proj.shape)} ep={_resolve_ep()[0]} "
                f"groups={audit.get('local_groups')}/{audit.get('global_groups')}"
            )
        total = group_loss if total is None else total + group_loss
        num_groups += 1
    if total is None or num_groups == 0:
        return None
    return total / num_groups


def attach_erc_loss(
    activation: torch.Tensor,
    erc: torch.Tensor | None,
    coef: float,
    calculate_per_token_loss: bool = True,
    num_tokens: int | None = None,
) -> torch.Tensor:
    if erc is None or coef == 0.0:
        return activation
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    scaled = erc * coef
    if calculate_per_token_loss:
        tokens = num_tokens if num_tokens is not None else activation.shape[0]
        return MoEAuxLossAutoScaler.apply(activation, scaled * tokens)
    return MoEAuxLossAutoScaler.apply(activation, scaled)


def verify_erc_attach_scaling(
    num_tokens: int = 7, coef: float = 3.0, main_loss_backward_scale: float = 1.0
) -> dict:
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    saved = MoEAuxLossAutoScaler.main_loss_backward_scale
    out: dict[str, object] = {
        "num_tokens": int(num_tokens),
        "coef": float(coef),
        "main_loss_backward_scale": float(main_loss_backward_scale),
    }
    try:
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            float(main_loss_backward_scale)
        )
        for per_token in (False, True):
            act = torch.ones(4, dtype=torch.float32, requires_grad=True)
            erc = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
            loss = attach_erc_loss(
                act, erc, coef, calculate_per_token_loss=per_token, num_tokens=num_tokens
            )
            loss.sum().backward()
            expected = (
                float(coef)
                * float(main_loss_backward_scale)
                * (float(num_tokens) if per_token else 1.0)
            )
            tag = "per_token" if per_token else "no_per_token"
            out[f"{tag}_grad_erc"] = float(erc.grad)
            out[f"{tag}_expected"] = expected
            out[f"{tag}_identity_ok"] = bool(torch.equal(loss.detach(), act.detach()))
            out[f"{tag}_grad_activation_ok"] = bool(torch.equal(act.grad, torch.ones(4)))
            out[f"{tag}_ok"] = bool(
                abs(float(erc.grad) - expected) < 1e-6
                and torch.equal(loss.detach(), act.detach())
                and torch.equal(act.grad, torch.ones(4))
            )
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = saved
    out["ok"] = bool(out["per_token_ok"] and out["no_per_token_ok"])
    return out


def add_safe_globals_for_torch_ckpt() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.enums import AttnBackend

    torch.serialization.add_safe_globals([SimpleNamespace, AttnBackend, F.silu])


def warn_missing_checkpoint_tracker(args) -> None:
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
        "iter_* 目录：上游会**静默**退化成『从随机初始化开始训』（checkpointing.py:1364-1373，"
        "只打一句 info）。要让它变成硬报错请加**上游开关** `--exit-on-missing-checkpoint`；"
        "本入口不替你开（那会改语义）。"
    )


def fix_first_iteration_loss_logging() -> None:
    if os.environ.get("SHENSI_LOG_KEEP_UPSTREAM", "").strip() in ("1", "true", "True"):
        print_rank_0(
            "[shensi][log] SHENSI_LOG_KEEP_UPSTREAM=1 -> 保持上游 training_log 口径"
            "（进程内第二条 iteration 日志是两步平均；仅用于对照取证）"
        )
        return
    import megatron.training.training as _training

    if getattr(_training.training_log, "_shensi_first_iter_fix", False):
        return
    _orig_training_log = _training.training_log

    def training_log_fs_reset(*a, **kw):
        if getattr(get_args(), "log_interval", 1) == 1:
            if "is_first_iteration" in kw:
                kw["is_first_iteration"] = False
            elif len(a) > 12:
                a = list(a)
                a[12] = False
                a = tuple(a)
        return _orig_training_log(*a, **kw)

    training_log_fs_reset._shensi_first_iter_fix = True
    _training.training_log = training_log_fs_reset
    print_rank_0(
        "[shensi][log] 已安装 log_interval==1 的累加器清零修正"
        "（每条 iteration 日志只统计自己那一步；训练数学不受影响）"
    )


def get_device_arch_version_cpu_safe():
    from megatron.plugin.platform import get_platform

    cur_platform = get_platform()
    try:
        return cur_platform.get_device_properties(cur_platform.device(0)).major
    except NotImplementedError:
        return None


def install_cpu_platform_compat() -> bool:
    try:
        from megatron.plugin.decorators import register_override_method
    except Exception as exc:
        print_rank_0(f"[shensi][cpu] 跳过 CPU 平台补丁（上游插件机制不可用：{exc!r}）")
        return False
    register_override_method(
        "common_utils.get_device_arch_version", get_device_arch_version_cpu_safe
    )
    return True


def add_shensi_args(parser):
    group = parser.add_argument_group(title="shensi", description="Shensi 家族参数")
    # AdEMAMix：上游 emerging_optimizers 按名字取标量优化器，这里只把名字放进 choices 与参数组
    # （`shensi.utils.optimizer.ademamix` 导入即注册；--muon-scalar-optimizer 选的就是这个名字）
    for action in parser._actions:
        if action.dest == "optimizer" and action.choices and "ademamix" not in action.choices:
            action.choices = [*action.choices, "ademamix"]
    group.add_argument(
        "--muon-scalar-optimizer",
        default="adam",
        choices=["adam", "ademamix"],
        help="Muon 的非矩阵那条腿用哪个标量优化器（默认 adam）",
    )
    # 名字与 pytorch_optimizer.AdEMAMix 的构造参数一致，mcore 按 `{名字}_{参数}` 从 config 取
    group.add_argument("--ademamix-betas", nargs=3, type=float, default=(0.9, 0.999, 0.9999),
                       help="AdEMAMix 的 (beta_fast, beta2, beta_slow)")
    group.add_argument("--ademamix-alpha", type=float, default=5.0,
                       help="AdEMAMix 慢 EMA 在更新里的权重")
    group.add_argument("--ademamix-t-alpha-beta3", type=int, default=0,
                       help="alpha / beta_slow 的 warmup 步数（库里的 t_alpha_beta3）")
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
        help="滑窗宽度（默认 = HF 的 128；tiny 侧必须在 yaml 显式写小值）",
    )
    group.add_argument("--shensi-head-dim", type=int, default=0)
    group.add_argument("--shensi-q-lora-rank", type=int, default=0)
    group.add_argument("--shensi-partial-rotary-factor", type=float, default=0.0)
    group.add_argument("--shensi-routed-expert-hidden-size", type=int, default=None)
    group.add_argument(
        "--shensi-index-topk",
        type=int,
        default=512,
        help="indexer 的 top-k（默认 = HF 的 512；tiny 侧必须在 yaml 显式写小值）",
    )
    group.add_argument("--shensi-index-n-heads", type=int, default=0)
    group.add_argument("--shensi-index-head-dim", type=int, default=0)
    group.add_argument(
        "--shensi-num-experts",
        type=int,
        default=256,
        help="路由专家数（默认 = HF 的 256；tiny 侧必须在 yaml 显式写小值）",
    )
    group.add_argument(
        "--shensi-num-experts-per-tok",
        type=int,
        default=6,
        help="每 token 激活专家数（默认 = HF 的 6；tiny 侧必须在 yaml 显式写小值）",
    )
    group.add_argument("--shensi-moe-intermediate-size", type=int, default=None)
    group.add_argument(
        "--shensi-router-aux-loss-coef",
        type=float,
        default=None,
        help="MoE 路由 aux loss 系数；不给就用 ckpt/HF 声明的 router_aux_loss_coef（0.001），"
        "显式给 0 可关掉（HF 参考只在 output_router_logits=True 时加它）",
    )
    group.add_argument(
        "--shensi-freeze",
        type=str,
        choices=("none", "indexer", "non-indexer"),
        default="none",
        help="冻参数组（GLM-5 配方用）：indexer = 只训 Lightning Indexer（DSA dense warmup，"
        "主干冻结）；non-indexer = 冻结 indexer（RL 阶段按论文冻 indexer）；none = 不冻",
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


def install_parse_and_validate_args() -> None:
    from megatron.training.arguments import parse_and_validate_args as _upstream_parse

    if getattr(train_gpt.parse_and_validate_args, "_shensi_patched", False):
        return

    def parse_and_validate_args_with_shensi(**kwargs):
        inner = kwargs.pop("extra_args_provider", None)

        def provider(p):
            if inner is not None:
                p = inner(p)
            return add_shensi_args(p)

        kwargs["extra_args_provider"] = provider
        args = _upstream_parse(**kwargs)
        apply_shensi_compress_ratios(args)
        apply_shensi_hf_rope_scaling(args)
        inject_shensi_fields_into_args(args)
        verify_shensi_geometry(args)
        warn_pure_weight_load(args, downgrade=_auto_downgrade_requested(args))
        warn_missing_checkpoint_tracker(args)
        return args

    parse_and_validate_args_with_shensi._shensi_patched = True
    train_gpt.parse_and_validate_args = parse_and_validate_args_with_shensi


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
SHENSI_KNOB_OF_ARGS_FIELD = {args_attr: knob for _, args_attr, knob in SHENSI_HF_PARITY_FIELDS}

SHENSI_LAYER_TYPE_TO_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def find_hf_shensi_config(args) -> str | None:
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
    from megatron_ext.core.transformer.shensi.transformer_config import (
        resolve_csa_compress_ratios,
        resolve_moe_n_hash_layers,
    )

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
        "想『权重 + 优化器一起带走』：用 SHENSI_CKPT_OPTIM_PLACEHOLDER=1 重跑 "
        "build_mcore_ckpt_from_hf.py 生成带空占位（零动量/step=0/master=模型权重）的检查点，"
        "然后不要加任何 no-load 开关。确实要旧行为时显式加 "
        "`--shensi-pure-weight-auto-downgrade`。"
    )


def _downgrade_pure_weight_load(args) -> None:
    warn_pure_weight_load(args, downgrade=True)


_UPSTREAM_LOSS_FUNC = train_gpt.loss_func
_ERC_DISABLED_LOGGED = False


def _shensi_erc_of(model):
    modules = unwrap_model(model)
    model_obj = modules[0] if isinstance(modules, (list, tuple)) else modules
    cfg = getattr(model_obj, "config", None)
    alpha = float(getattr(cfg, "erc_loss_alpha", 0.5))
    coef = float(getattr(cfg, "erc_loss_coef", 0.0) or 0.0)
    if getattr(model_obj, "moe_group_owners", None) is None:
        return model_obj, None, coef, alpha, "no_groups"
    if coef == 0.0:
        return model_obj, None, coef, alpha, "coef0"
    return model_obj, shensi_erc_loss(model_obj, alpha=alpha), coef, alpha, "ok"


def shensi_loss_func(loss_mask, output_tensor, model=None):
    global _ERC_DISABLED_LOGGED
    loss, num_tokens, report = _UPSTREAM_LOSS_FUNC(loss_mask, output_tensor, model=model)
    model_obj, erc, coef, alpha, reason = _shensi_erc_of(model)
    if erc is None:
        if not _ERC_DISABLED_LOGGED:
            _ERC_DISABLED_LOGGED = True
            if reason == "no_groups":
                _print_erc(
                    "[shensi][erc] 本 stage 没有 MoE 组（moe_group_owners 为空）→ 不上报 "
                    "erc loss/total loss，也不参与反传。**原因不是 coef=0**：PP>1 时这只说明"
                    "本 rank 不是『持有 MoE 组』的那个 stage（ERC 在末 stage 算）。"
                )
            else:
                _print_erc(
                    f"[shensi][erc] ERC 关闭：erc_loss_coef={coef}（模型 config）→ "
                    "不上报 erc loss/total loss，也不参与反传"
                )
        return loss, num_tokens, report
    args = get_args()
    calc_per_token = bool(getattr(args, "calculate_per_token_loss", False))
    tokens = num_tokens.item() if torch.is_tensor(num_tokens) else num_tokens
    tokens = int(tokens)
    losses = output_tensor.view(-1).float()
    mask = loss_mask.view(-1).float()
    scaled_losses = attach_erc_loss(
        losses, erc, coef, calculate_per_token_loss=calc_per_token, num_tokens=tokens
    )
    loss_out = torch.sum(scaled_losses * mask)
    if not torch.allclose(loss_out.detach(), loss.detach(), rtol=0.0, atol=1e-4):
        _print_erc(
            f"[shensi][erc] [warn] 重算的 lm loss({float(loss_out):.6f}) 与上游"
            f"({float(loss):.6f}) 不一致 → 本步 ERC 只上报、不反传"
        )
        loss_out = loss
    denom = torch.tensor(float(tokens), dtype=torch.float, device=loss.device).view(1)
    erc_value = float(erc.detach())
    erc_sum = torch.tensor(coef * erc_value * tokens, dtype=torch.float, device=loss.device).view(1)
    report = dict(report)
    report["erc loss"] = torch.cat([erc_sum, denom])
    report["total loss"] = torch.cat([loss.detach().view(1) + erc_sum, denom])
    audit = check_erc_groups_complete(
        model_obj, getattr(model_obj, "moe_group_owners", None), verbose=False
    )
    _print_erc(
        f"[shensi][erc] coef={coef} alpha={alpha} "
        f"groups={audit['local_groups']}/{audit['global_groups']}（本 stage/全局）"
        f"status={audit['status']} erc={erc_value:.6e} tokens={tokens} "
        f"per_token_loss={calc_per_token} → 计入反传的 erc 项 = {coef * erc_value:.6e}"
    )
    return loss_out, num_tokens, report


def install_erc_loss() -> None:
    if getattr(train_gpt.loss_func, "_shensi_erc_installed", False):
        return
    shensi_loss_func._shensi_erc_installed = True
    train_gpt.loss_func = shensi_loss_func


def install_load_probe() -> None:
    from megatron_ext.core.models.shensi.shensi_model import ShensiModel

    if getattr(ShensiModel.load_state_dict, "_shensi_load_probe", False):
        return
    original = ShensiModel.load_state_dict

    def probe(self, state_dict, *args, **kwargs):
        model_keys = {k for k in self.state_dict() if not k.endswith("_extra_state")}
        ckpt_keys = {k for k in state_dict if not k.endswith("_extra_state")}
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        n_extra = len(state_dict) - len(ckpt_keys)
        ret = original(self, state_dict, *args, **kwargs)
        sd_now = self.state_dict()
        digest, n = digest_state_dict(sd_now)
        try:
            args_now = get_args()
            load_dir = getattr(args_now, "load", None)
            save_dir = getattr(args_now, "save", None)
        except Exception:
            load_dir, save_dir = None, None
        strict = kwargs.get("strict", args[0] if args else True)
        print_rank_0(
            f"[shensi][load] successfully loaded checkpoint from {load_dir}: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"（模型侧 {len(model_keys)} 个权重键 / 检查点侧 {len(ckpt_keys)} 个，"
            f"_extra_state {n_extra} 个已排除）| 参数摘要 sha256={digest}（{n} 个张量）"
        )
        if missing:
            print_rank_0(f"[shensi][load] missing 明细（前 10）：{missing[:10]}")
        if unexpected:
            print_rank_0(f"[shensi][load] unexpected 明细（前 10）：{unexpected[:10]}")
        probe_strict = str(os.environ.get("SHENSI_LOAD_PROBE_STRICT", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if probe_strict and (missing or unexpected):
            raise RuntimeError(
                "[shensi][load-probe] SHENSI_LOAD_PROBE_STRICT=1 且加载有缺口："
                f"missing={len(missing)}（前 10：{missing[:10]}）"
                f"unexpected={len(unexpected)}（前 10：{unexpected[:10]}）。"
                "上游 checkpointing.py:1939-1946 会把 strict 失败吞掉降级成 strict=False，"
                "所以这里显式拦下（否则只打日志）。"
            )
        if save_dir and _probe_should_write():
            try:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(save_dir, "shensi_load_probe.json")
                info = summarize(sd_now)
                info.update(
                    {
                        "load": load_dir,
                        "missing": missing,
                        "unexpected": unexpected,
                        "num_extra_state_ckpt": n_extra,
                        "strict": strict,
                        "probe_rank": _probe_rank(),
                        "per_tensor_sha256": {
                            k: tensor_digest(sd_now, k) for k in sample_keys(sd_now, n=8)
                        },
                    }
                )
                with open(out, "w") as f:
                    json.dump(info, f, indent=1, ensure_ascii=False)
                print_rank_0(f"[shensi][load] 探针 JSON 已写：{out}")
            except Exception as exc:
                print_rank_0(f"[shensi][load] 探针 JSON 写入失败（忽略）：{exc!r}")
        return ret

    probe._shensi_load_probe = True
    ShensiModel.load_state_dict = probe
    _family_digest_parity_report()
    print_rank_0("[shensi][load] 已安装检查点加载探针（missing/unexpected + sha256 摘要）")


def _probe_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _probe_should_write() -> bool:
    return _probe_rank() == 0


def _family_digest_parity_report() -> None:
    try:
        from models.shensi.ckpt_digest import (
            digest_state_dict as _ws_digest,
            weight_keys as _ws_weight_keys,
        )
    except Exception:
        print_rank_0(
            "[shensi][digest] 工作区 models.shensi 不可导入（生产机正常）→ 跳过与它的"
            "摘要等价自证；本进程用的是树内 shensi.utils.ckpt_digest"
        )
        return
    probe_sd = {
        "a.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "b.weight": torch.zeros(2, dtype=torch.bfloat16),
        "c._extra_state": torch.ones(3),
    }
    ours, n_ours = digest_state_dict(probe_sd)
    theirs, n_theirs = _ws_digest(probe_sd)
    keys_same = weight_keys(probe_sd) == _ws_weight_keys(probe_sd)
    same = ours == theirs and n_ours == n_theirs and keys_same
    print_rank_0(
        f"[shensi][digest] 树内/工作区摘要等价自证：{'PASS' if same else 'FAIL'} "
        f"（ours={ours[:16]}… theirs={theirs[:16]}… n={n_ours}/{n_theirs} "
        f"weight_keys 一致={keys_same}）"
    )
    if os.environ.get("SHENSI_KEEP_UPSTREAM_ASSERT", "").strip() in ("1", "true", "True"):
        print_rank_0(
            "[shensi][P8] SHENSI_KEEP_UPSTREAM_ASSERT=1 -> 保留上游 "
            "deepseek_model.py:76 的无条件断言（PP>1 非首 stage 会 AssertionError）"
        )
        return
    from flagscale.models.megatron.deepseek_v4.deepseek_model import DeepSeekModel

    if getattr(DeepSeekModel.forward, "_shensi_relaxed_input_ids", False):
        return
    _orig_forward = DeepSeekModel.forward

    def forward(self, input_ids, position_ids, *args, **kwargs):
        if input_ids is None and not bool(getattr(self.config, "use_engram", False)):
            input_ids = torch.zeros((1, 1), dtype=torch.long)
            if position_ids is None:
                position_ids = torch.zeros((1, 1), dtype=torch.long)
        return _orig_forward(self, input_ids, position_ids, *args, **kwargs)

    forward._shensi_relaxed_input_ids = True
    DeepSeekModel.forward = forward
    print_rank_0(
        "[shensi][P8] 已放宽 deepseek_model.py:76 的 input_ids 断言"
        "（仅 use_engram=False 的非首 stage 走占位；数值不受影响）"
    )


def _family_owned_defect_entrypoints():
    import megatron_ext.core.transformer.shensi.attn_res as _pp
    from megatron_ext.core.models.shensi.shensi_model import ShensiModel
    from megatron_ext.core.transformer.shensi.attn_res import ShensiAttnResState

    return (
        (
            "C-2",
            "ShensiModel._postprocess（n 流收缩门控）",
            ShensiModel._postprocess,
            "shensi_model.py",
        ),
        (
            "C-2",
            "ShensiModel.shensi_output_contract",
            ShensiModel.shensi_output_contract,
            "shensi_model.py",
        ),
        (
            "C-4",
            "ShensiAttnResState.state_for_pp（header 设备）",
            ShensiAttnResState.state_for_pp,
            "attn_res.py",
        ),
        ("C-5", "attn_res._recv_payload（recv 缓冲设备）", _pp._recv_payload, "attn_res.py"),
        (
            "C-6",
            "_ShensiAttnResPPBridge.forward（grad 缓冲设备）",
            _pp._ShensiAttnResPPBridge.__dict__["forward"].__func__,
            "attn_res.py",
        ),
        (
            "C-6",
            "_ShensiAttnResPPBridge.backward（grad 缓冲设备）",
            _pp._ShensiAttnResPPBridge.__dict__["backward"].__func__,
            "attn_res.py",
        ),
        (
            "C-7",
            "ShensiAttnResPPHandoff._record（numpy 前 .cpu()）",
            _pp.ShensiAttnResPPHandoff._record,
            "attn_res.py",
        ),
    )


def verify_c_defect_ledger(verbose: bool = True) -> dict:
    import inspect

    import megatron_ext.core.transformer.shensi.attn_res as _pp
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    marks = ("_pp2_relay_patch", "_pp2_relay_shim", "_shensi_patched")

    def _src(fn) -> str:
        src = inspect.getsourcefile(fn) or ""
        if not src:
            src = str(getattr(getattr(fn, "__code__", None), "co_filename", "") or "")
        return src.replace("\\", "/")

    def _check_native(tag, desc, fn, want):
        src = _src(fn)
        if not src.endswith(want):
            raise AssertionError(
                f"[shensi][C-ledger] {tag} {desc} 的实现来自 {src}，不是原生 {want}"
                "（家族正经修被覆盖了）"
            )
        got = [m for m in marks if hasattr(fn, m)]
        if got:
            raise AssertionError(f"[shensi][C-ledger] {tag} {desc} 带 monkeypatch 标记 {got}")

    rows = []
    for tag, desc, fn, want in _family_owned_defect_entrypoints():
        _check_native(tag, desc, fn, want)
        rows.append(f"{tag}:{want}")
    _check_native(
        "C-3",
        "OptimizerParamScheduler.__init__",
        OptimizerParamScheduler.__init__,
        "optimizer_param_scheduler.py",
    )
    rows.append("C-3:上游原生开关（无 patch）")
    if not callable(getattr(_pp, "payload_device", None)):
        raise AssertionError("[shensi][C-ledger] 家族 attn_res.payload_device 缺失")
    rows.append(f"CUDA原生:payload_device={_pp.payload_device()}")
    out = {"rows": rows}
    if verbose:
        print_rank_0(
            "[shensi][C-ledger] C-2..C-7 收口核对 PASS：家族正经修 5 条 + 配置开关 1 条"
            "（上游 P8 已随新架构移除：本入口跑 mcore 的 ShensiModel，不再经 FlagScale 的 "
            "DeepSeekModel.forward）｜ " + " ".join(rows)
        )
    return out


def install_shensi_patches() -> None:
    add_safe_globals_for_torch_ckpt()
    install_cpu_platform_compat()
    install_parse_and_validate_args()
    install_erc_loss()
    fix_first_iteration_loss_logging()
    install_load_probe()
    verify_c_defect_ledger()


def install_shensi_patches_twice_for_test() -> dict:
    from megatron_ext.core.models.shensi.shensi_model import ShensiModel

    before_loss = train_gpt.loss_func
    before_probe = ShensiModel.load_state_dict
    install_erc_loss()
    install_load_probe()
    return {
        "loss_func_same": train_gpt.loss_func is before_loss,
        "probe_same": ShensiModel.load_state_dict is before_probe,
        "erc_marker": bool(getattr(train_gpt.loss_func, "_shensi_erc_installed", False)),
        "probe_marker": bool(getattr(ShensiModel.load_state_dict, "_shensi_load_probe", False)),
    }


install_shensi_patches()


def _assert_mtp_on_last_stage(args) -> None:
    layout = str(getattr(args, "pipeline_model_parallel_layout", "") or "")
    if not layout or "|" not in layout:
        return
    segs = layout.split("|")
    bad = [i for i, seg in enumerate(segs[:-1]) if "m" in seg.lower()]
    if bad:
        raise NotImplementedError(
            f"layout={layout!r} 把 MTP 放在非末 pp stage（段 {bad}）；本模型的 MTP 必须与末 stage 收缩同 stage"
        )


def shensi_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
    from megatron_ext.core.transformer.shensi.transformer_config import (
        ShensiTransformerConfig,
        apply_shensi_overrides_from_args,
        shensi_config_from_args,
    )

    print_rank_0("building Shensi model (dsv4_hybrid + AttnRes + 流路由 mHC + ERC) ...")
    if config is None:
        if args.yaml_cfg is not None:
            raise NotImplementedError("shensi_builder 暂不支持 --yaml-cfg")
        config = shensi_config_from_args(args)
        apply_shensi_overrides_from_args(config, args)
    if not isinstance(config, ShensiTransformerConfig):
        raise TypeError(
            f"shensi_builder 需要 ShensiTransformerConfig，得到 {type(config).__name__}"
        )
    if args.spec is not None:
        raise NotImplementedError("Using custom spec is not supported with shensi builder.")
    if args.heterogeneous_layers_config_path is not None:
        raise NotImplementedError(
            "Using heterogeneous layers is not supported with shensi builder."
        )
    use_te = args.transformer_impl == "transformer_engine"
    transformer_layer_spec = get_shensi_decoder_block_spec(
        config=config,
        use_transformer_engine=use_te,
        normalization=args.normalization,
        qk_l2_norm=args.qk_l2_norm,
        vp_stage=vp_stage,
        use_moe=True,
    )
    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None):
        _assert_mtp_on_last_stage(args)
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            get_shensi_mtp_layer_spec(config=config, use_transformer_engine=use_te),
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )
    model = ShensiModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    print_rank_0(f"Model = {model}")
    apply_shensi_freeze(model, getattr(args, "shensi_freeze", "none"))
    return model


def apply_shensi_freeze(model, mode: str) -> None:
    if mode in (None, "", "none"):
        return
    allowed = ("indexer", "non-indexer", "mtp", "non-mtp")
    if mode not in allowed:
        raise ValueError(f"--shensi-freeze 只认 {allowed}，得到 {mode!r}")
    n_frozen = n_train = 0
    for name, param in model.named_parameters():
        is_indexer = ".indexer." in name or name.endswith(".indexer")
        # MTP 头在 mcore 里叫 mtp / multi_token_prediction；主干冻结、只训 draft（DeepSpec 口径的训练侧）
        is_mtp = ".mtp." in name or name.startswith("mtp") or "multi_token_prediction" in name
        if mode == "non-indexer":
            freeze = is_indexer
        elif mode == "non-mtp":
            freeze = is_mtp
        elif mode == "mtp":
            freeze = not is_mtp
        else:
            freeze = not is_indexer
        param.requires_grad_(not freeze)
        if freeze:
            n_frozen += 1
        else:
            n_train += 1
    if n_train == 0:
        raise ValueError(
            f"--shensi-freeze {mode} 把全部参数都冻上了：本模型里没有对应参数"
            "（indexer 只有 ratio==4 的 CSA 层才有；MTP 要 mtp_num_layers>0，检查层计划）"
        )
    print_rank_0(
        f"[shensi][freeze] mode={mode}：可训练 {n_train} 个参数、冻结 {n_frozen} 个"
        f"（可训练元素数 {sum(p.numel() for p in model.parameters() if p.requires_grad)}）"
    )


if __name__ == "__main__":
    train_gpt.main(shensi_builder)
