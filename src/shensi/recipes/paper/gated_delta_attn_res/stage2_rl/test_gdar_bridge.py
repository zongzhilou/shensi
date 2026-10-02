"""verl 桥的闸门：登记、装载零缺键与与 HF 的单步对拍。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/louzo/code/shensi/shensi/src")
os.environ.setdefault("SHENSI_ROOT", "/home/louzo/code/shensi/shensi")
os.environ.setdefault("SHENSI_FS", "/home/louzo/fsdata")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29598")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<56} {detail}")


def main() -> int:
    from shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.tiny_checkpoint import build

    tmp = Path(tempfile.mkdtemp(prefix="gdar_bridge_"))
    ckpt = tmp / "gdar_tiny"
    info = build("gdar", ckpt, shape="tiny", overwrite=True)
    auto_map = info.get("auto_map") or {}
    check(
        "tiny HF 检查点（auto_map 齐备）", "AutoModelForCausalLM" in auto_map, str(sorted(auto_map))
    )

    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl import gdar_bridge
    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl import variants as gdar_variants

    check(
        "导入即注册（与 variants.VARIANTS 逐一对应）",
        set(gdar_bridge.REGISTERED) == {v.model_type for v in gdar_variants.VARIANTS.values()}
        and "qwen3_realformer" in gdar_bridge.REGISTERED,
        f"{len(gdar_bridge.REGISTERED)} 个：{sorted(gdar_bridge.REGISTERED)}",
    )

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(ckpt, trust_remote_code=True)
    check(
        "AutoBridge 按 auto_map 分发到本模块的桥",
        isinstance(bridge._model_bridge, gdar_bridge.DepthBridge)
        and type(bridge._model_bridge).__name__ == "Qwen3GDARForCausalLMBridge",
        type(bridge._model_bridge).__name__,
    )

    import torch.distributed as dist
    from megatron.core.parallel_state import initialize_model_parallel

    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(0)

    provider = bridge.to_megatron_provider(load_weights=False)
    spec = provider.transformer_layer_spec
    hf_cfg = bridge.hf_pretrained.config
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer import (
        GdarTransformerLayer,
    )

    check(
        "层规格 = 本配方的 GDAR 连接层（非 stock Qwen3）",
        spec.module is GdarTransformerLayer,
        f"{spec.module.__module__}.{spec.module.__name__}",
    )
    check(
        "HF 旋钮进了 spec.params",
        int(spec.params.get("gdar_block_size", -1)) == int(hf_cfg.attn_res_block_size)
        and spec.params.get("gdar_update") == hf_cfg.attn_res_update
        and int(spec.params.get("gdar_read_heads", -1)) == int(hf_cfg.attn_res_read_heads),
        f"block_size={spec.params.get('gdar_block_size')} update={spec.params.get('gdar_update')} "
        f"read_heads={spec.params.get('gdar_read_heads')}",
    )

    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import gdar_spec
    from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.configuration_qwen3_gdar import (
        Qwen3GDARConfig,
    )
    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.variants import (
        VARIANTS,
        build_layer_spec,
    )

    paper_kwargs = {
        "attn_res_" + key[len("gdar_") :]: value for key, value in gdar_spec._PAPER.items()
    }
    paper_cfg = Qwen3GDARConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        **paper_kwargs,
    )
    paper_spec, _, _ = build_layer_spec(paper_cfg, VARIANTS["gdar"])
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer import (
        gdar_knobs_from_kwargs,
    )

    got = gdar_knobs_from_kwargs(dict(paper_spec.params))
    want = gdar_knobs_from_kwargs(dict(gdar_spec.gdar_layer_spec_paper.params))
    check(
        "桥上的规格 = 论文主版本规格（生效配置逐项）",
        got == want,
        ""
        if got == want
        else f"diff={
            {
                k: (getattr(got, k, None), getattr(want, k, None))
                for k in set(got.__dataclass_fields__)
                if getattr(got, k) != getattr(want, k)
            }
        }",
    )

    provider_overrides = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "variable_seq_lengths": True,
        "gradient_accumulation_fusion": False,
    }
    if hasattr(provider, "apply_overrides_and_finalize"):
        provider.apply_overrides_and_finalize(dtype=torch.float32, overrides=provider_overrides)
    else:  # pragma: no cover - 只在更老的 Bridge 上走
        provider.params_dtype = torch.float32
        provider.fp16 = False
        provider.bf16 = False
        for name, value in provider_overrides.items():
            setattr(provider, name, value)
        provider.finalize()
    check(
        "provider finalize 后 init_method 就位",
        provider.init_method is not None and provider.embedding_init_method is not None,
        f"init={provider.init_method is not None} embedding={provider.embedding_init_method is not None}",
    )

    model = provider.provide_distributed_model(
        wrap_with_ddp=False, ddp_config=None, fp16=False, bf16=False, use_megatron_fsdp=False
    )
    model = model if isinstance(model, list) else [model]
    bridge.load_hf_weights(model, str(ckpt))

    target = model[0].state_dict()
    pinned = {
        n: p
        for n, p in model[0].named_parameters()
        if n.endswith((".q_proj.up.bias", ".k_proj.up.bias"))
    }
    drifting = [n for n, p in pinned.items() if p.requires_grad or p.count_nonzero().item()]
    check(
        "Megatron 独有行恒零且不可训练",
        pinned and not drifting,
        f"{len(pinned)} 个，漂移/可训练 {drifting[:3]}",
    )
    check(
        "装载后无全零行（骨干真的灌进去了）",
        all(
            target[k].count_nonzero().item()
            for k in ("embedding.word_embeddings.weight", "output_layer.weight")
        ),
        "",
    )

    torch.manual_seed(1)
    ids = torch.randint(0, int(hf_cfg.vocab_size), (1, 16))
    hf_model = bridge.hf_pretrained.model.eval()
    hf_device = next(hf_model.parameters()).device
    with torch.no_grad():
        hf_logits = hf_model(ids.to(hf_device)).logits.float().cpu()
        mc_logits = model[0](ids.cuda(), position_ids=None, attention_mask=None)
    mc_logits = mc_logits[0] if isinstance(mc_logits, (tuple, list)) else mc_logits
    mc_logits = mc_logits.float().cpu()
    if mc_logits.shape[:2] != hf_logits.shape[:2]:
        mc_logits = mc_logits.transpose(0, 1)
    d = (hf_logits - mc_logits).abs().max().item()
    check("logits 对拍（HF vs TE-mcore，底噪口径 |Δ| ≤ 2e-2）", d <= 2e-2, f"max|Δ| = {d:.3e}")

    exported = {name: tensor for name, tensor in bridge.export_hf_weights(model, cpu=True)}
    hf_state = {k: v for k, v in bridge.hf_pretrained.model.state_dict().items()}
    missing = sorted(set(hf_state) - set(exported))
    extra = sorted(set(exported) - set(hf_state))
    check(
        "导出：无缺名/无多名", not missing and not extra, f"missing={missing[:3]} extra={extra[:3]}"
    )
    worst, worst_name = 0.0, ""
    for name in sorted(set(hf_state) & set(exported)):
        delta = (exported[name].float().cpu() - hf_state[name].float().cpu()).abs().max().item()
        if delta > worst:
            worst, worst_name = delta, name
    check(
        "导出：与 HF 逐位相等",
        not missing and not extra and worst == 0.0,
        f"max|Δ| = {worst:.3e} ({worst_name})",
    )

    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
