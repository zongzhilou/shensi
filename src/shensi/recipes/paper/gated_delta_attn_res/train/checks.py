"""离线校验：恒等初始化、前向恒等、参数开销与梯度流。"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common
from shensi.recipes.shensi.train import launcher as base_launcher


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def report(name: str, ok, extra: str = "") -> bool:
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"  [{tag}] {name}{(' -- ' + extra) if extra else ''}", flush=True)
    return bool(ok)


def _gdar_extra_args(parser):
    """让 mcore 的解析器也认本配方的标量腿旋钮（生产档里有 AdaMuon + AdEMAMix）。"""
    from shensi.recipes.paper.gated_delta_attn_res.train import optimizer_knobs

    optimizer_knobs.add_scalar_optimizer_args(parser.add_argument_group("gdar-scalar-optimizer"))
    optimizer_knobs.extend_scalar_optimizer_choices(parser)
    return parser


def load_args_from_profile(profile: str, extra_argv=()):
    cfg = common.smoke_config("stage1_pretrain", profile)
    argv = base_launcher.flatten_train_section(cfg["train"]) + list(extra_argv)
    sys.argv = ["checks"] + argv
    from megatron.training.arguments import parse_args

    args = parse_args(extra_args_provider=_gdar_extra_args)
    if getattr(args, "bf16", False):
        args.params_dtype = torch.bfloat16
    elif getattr(args, "fp16", False):
        args.params_dtype = torch.float16
    else:
        args.params_dtype = torch.float32
    if args.config_logger_dir is None:
        args.config_logger_dir = ""
    if getattr(args, "padded_vocab_size", None) is None:
        multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
        args.padded_vocab_size = -(-args.vocab_size // multiple) * multiple
    return args, list(argv)


def baseline_spec(config):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    return get_gpt_layer_local_spec(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,
        normalization=config.normalization,
        qk_l2_norm=config.qk_l2_norm,
    )


def reseed(seed: int) -> None:
    torch.manual_seed(seed)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(seed)


def build_model(config, args, spec, device):
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.module import Float16Module

    raw = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_base=args.rotary_base,
        rotary_percent=args.rotary_percent,
    )
    return Float16Module(config, raw).to(device).eval()


def make_batch(args, device, seed: int = 1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    seq, bsz = args.seq_length, args.micro_batch_size
    input_ids = torch.randint(0, args.padded_vocab_size, (bsz, seq), generator=g).to(device)
    return input_ids


def forward_logits(model, args, device, seed: int = 1234):
    input_ids = make_batch(args, device, seed)
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=None, attention_mask=None)
    return out[0] if isinstance(out, (tuple, list)) else out


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR 移植离线校验（恒等 / 参数开销 / 梯度流）")
    ap.add_argument(
        "--profile", default="tiny", help="配方冒烟档（pretrain/config/<profile>.yaml）"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-grads", action="store_true")
    ap.add_argument(
        "--no-bda-fusion",
        action="store_true",
        help="加 --no-bias-dropout-fusion：eager 的 bias_dropout_add 与连接的写路径逐位可复现",
    )
    args_cli = ap.parse_args()

    ok = True

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29579")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if str(args_cli.device).startswith("cuda"):
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
        torch.cuda.set_device(args_cli.device)
    else:
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    extra_argv = ["--no-bias-dropout-fusion"] if args_cli.no_bda_fusion else []
    args, argv = load_args_from_profile(args_cli.profile, extra_argv)

    from megatron.core.parallel_state import initialize_model_parallel
    from megatron.training.arguments import core_transformer_config_from_args

    config = core_transformer_config_from_args(args)
    initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        context_parallel_size=config.context_parallel_size,
    )
    reseed(args.seed)

    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_layer_spec

    device = torch.device(args_cli.device)

    line("config")
    print("  " + " ".join(argv))
    print(
        f"  layers={config.num_layers} hidden={config.hidden_size} ffn={config.ffn_hidden_size} "
        f"heads={config.num_attention_heads} kv={config.num_query_groups} "
        f"seq={args.seq_length} vocab={args.padded_vocab_size} dtype={config.params_dtype} "
        f"seed={args.seed}"
    )

    line("[1] initialisation identity: same seed, plain spec vs GDAR spec")
    reseed(args.seed)
    plain = build_model(config, args, baseline_spec(config), device)
    reseed(args.seed)
    gdar = build_model(config, args, gdar_layer_spec, device)
    plain_n = sum(p.numel() for p in plain.parameters())
    gdar_n = sum(p.numel() for p in gdar.parameters())
    print(f"  plain params = {plain_n:,}")
    print(f"  gdar  params = {gdar_n:,}")

    plain_sd = {k: v for k, v in plain.state_dict().items() if not k.endswith("_extra_state")}
    gdar_sd = {k: v for k, v in gdar.state_dict().items() if not k.endswith("_extra_state")}
    extra_keys = sorted(set(gdar_sd) - set(plain_sd))
    bad = [
        key
        for key, tensor in plain_sd.items()
        if gdar_sd.get(key) is None
        or tensor.shape != gdar_sd[key].shape
        or not torch.equal(gdar_sd[key].to(tensor.device), tensor)
    ]
    ok &= report(
        "every shared parameter is bit-equal after the same manual_seed",
        not bad,
        f"{len(plain_sd) - len(bad)}/{len(plain_sd)} bit-equal"
        + (f"; MISMATCH {bad}" if bad else "")
        + f"; gdar-only keys={len(extra_keys)}（连接模块）",
    )

    line("[2] forward identity (eval mode, so no dropout RNG)")
    lp = forward_logits(plain, args, device)
    lg = forward_logits(gdar, args, device)
    diff = (lp.float() - lg.float()).abs().max().item()
    ok &= report(
        "logits identical for the same input",
        bool(torch.equal(lp, lg)),
        f"max|diff| = {diff:.3e}, torch.equal = {torch.equal(lp, lg)}",
    )

    line("[2b] training-mode forward (dropout active; informational)")
    plain.train()
    gdar.train()
    reseed(args.seed)
    lp = forward_logits(plain, args, device)
    reseed(args.seed)
    lg = forward_logits(gdar, args, device)
    ndiff = int((lp != lg).sum().item())
    detail = (
        f"differing elements = {ndiff}/{lp.numel()}, "
        f"max|diff| = {(lp.float() - lg.float()).abs().max().item():.3e}"
    )
    if config.bias_dropout_fusion:
        report("logits identical in training mode", None, detail + "  [informational: fused bda]")
        print("      -> re-run with --no-bda-fusion for the bit-exact statement")
    else:
        ok &= report("logits identical in training mode", bool(torch.equal(lp, lg)), detail)
    plain.eval()
    gdar.eval()

    line("[4] parameter cost（各 spec 预设的增量参数，tiny 几何）")
    import shensi.recipes.paper.gated_delta_attn_res.models.megatron as M

    specs = {
        "gdar_layer_spec (训练默认 r64)": M.gdar_layer_spec,
        "gdar_layer_spec_paper (论文主行)": M.gdar_layer_spec_paper,
        "gdar_layer_spec_theory": M.gdar_layer_spec_theory,
        "gdar_layer_spec_fullrank": M.gdar_layer_spec_fullrank,
        "gdar_layer_spec_block4": M.gdar_layer_spec_block4,
        "gdar_layer_spec_block4_r16 (参数匹配)": M.gdar_layer_spec_block4_r16,
        "ar_layer_spec": M.ar_layer_spec,
        "dar_layer_spec": M.dar_layer_spec,
        "denseformer_layer_spec": M.denseformer_layer_spec,
        "mudd_layer_spec": M.mudd_layer_spec,
        "hc_layer_spec": M.hc_layer_spec,
        "mhc_layer_spec": M.mhc_layer_spec,
        "gated_ar_layer_spec": M.gated_ar_layer_spec,
    }
    base_params = sum(p.numel() for p in plain.parameters())
    print(f"  plain = {base_params:,}")
    for name, spec in specs.items():
        reseed(args.seed)
        m = build_model(config, args, spec, device)
        total = sum(p.numel() for p in m.parameters())
        del m
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(
            f"    {name:<38} +{total - base_params:>12,} "
            f"({(total - base_params) / base_params * 100:6.2f}%)"
        )

    if not args_cli.skip_grads:
        line("[5] gradient flow at the identity point")
        import torch.nn.functional as F

        reseed(args.seed)
        model = build_model(config, args, gdar_layer_spec, device).train()
        input_ids = make_batch(args, device)
        labels = torch.randint(0, args.padded_vocab_size, input_ids.shape, device=device)
        vocab = args.padded_vocab_size

        def step_loss():
            out = model(input_ids=input_ids, position_ids=None, attention_mask=None)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits.shape[0] != input_ids.shape[1]:
                logits = logits.transpose(0, 1)
            return F.cross_entropy(
                logits[:-1].float().reshape(-1, vocab), labels.transpose(0, 1)[1:].reshape(-1)
            )

        loss = step_loss()
        loss.backward()
        groups = {
            "decay_scale": "scale",
            "erase_scale": "scale",
            "write_scale": "scale",
            "read_scale": "scale",
            "gate_proj": "gate weights",
            "q_proj": "read query",
            "k_proj": "erase direction",
        }

        def collect(tag):
            stats = {k: [] for k in groups}
            for name_, param in model.named_parameters():
                if param.grad is None:
                    continue
                if "attn_res" not in name_:
                    continue
                for key in groups:
                    if key in name_:
                        stats[key].append(param.grad.abs().max().item())
            for key, vals in stats.items():
                if vals:
                    print(f"  [{tag}] max|grad| {key:<12} ({groups[key]:<13}) = {max(vals):.6e}")
            print(f"  [{tag}] loss = {loss.item():.6f}")

        collect("step 0 (identity init)")
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss = step_loss()
        loss.backward()
        collect("step 1 (after one AdamW step)")

    line("summary")
    print("  ALL CHECKS PASSED" if ok else "  SOME CHECKS FAILED")
    torch.distributed.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
