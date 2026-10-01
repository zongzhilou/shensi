#!/usr/bin/env python3
"""闸门：verl 的 Megatron 通路（AutoBridge → provider → 模型 → 权重双向），tiny 规模。

转换表本身由 ``convert/test_convert_tiny.py`` 把关；这里让**上游自己的** AutoBridge 按 verl
实际调用的顺序走一遍，把"表对了"提升为"verl 能加载并导出我们的检查点"：

1. tiny HF 检查点（``models/vllm/tiny_checkpoint.py`` 产物：带 ``auto_map``，两个 .py 随权重走）;
2. ``AutoBridge.from_hf_pretrained(dir, trust_remote_code=True)``：架构分发到本模块注册的桥；
3. ``to_megatron_provider()``：provider 的层规格必须是**变体自己的连接层**（不是 stock Qwen3），
   且 HF 旋钮真的进了 spec；论文主版本的旋钮组合要与 ``gdar_spec.gdar_layer_spec_paper`` 逐项相等；
4. ``provide_distributed_model(wrap_with_ddp=False)`` + ``load_hf_weights``：无缺键/无意外键，
   Megatron 独有行恒零且不可训练；
5. logits 对拍（HF vs mcore，核数值底噪口径）;
6. ``export_hf_weights``：名字集合与 HF ``state_dict`` 一致、数值逐位相等——这就是 rollout
   引擎取权重的那条路。

    python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.test_gdar_bridge
"""

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
    # ---------------- 1) tiny HF 检查点（带 auto_map）
    from shensi.recipes.paper.gated_delta_attn_res.models.vllm.tiny_checkpoint import build

    tmp = Path(tempfile.mkdtemp(prefix="gdar_bridge_"))
    ckpt = tmp / "gdar_tiny"
    info = build("gdar", ckpt, shape="tiny", overwrite=True)
    auto_map = info.get("auto_map") or {}
    check(
        "tiny HF 检查点（auto_map 齐备）", "AutoModelForCausalLM" in auto_map, str(sorted(auto_map))
    )

    # ---------------- 2) 注册 + 分发
    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl import gdar_bridge

    check(
        "导入即注册（七个变体）", len(gdar_bridge.REGISTERED) == 7, sorted(gdar_bridge.REGISTERED)
    )

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(ckpt, trust_remote_code=True)
    check(
        "AutoBridge 按 auto_map 分发到本模块的桥",
        isinstance(bridge._model_bridge, gdar_bridge.DepthBridge)
        and type(bridge._model_bridge).__name__ == "Qwen3GDARForCausalLMBridge",
        type(bridge._model_bridge).__name__,
    )

    # ---------------- 3) provider：层规格必须是变体自己的连接层
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
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron.gdar_layer import (
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

    # 论文主版本：把 `gdar_spec._PAPER` 的旋钮写成 HF 配置，桥上跑出来的 spec 必须与
    # 论文训练用的 `gdar_layer_spec_paper` 逐项相等（否则 RL 训的就不是论文那个模型）。
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_spec
    from shensi.recipes.paper.gated_delta_attn_res.models.transformers.configuration_qwen3_gdar import (
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
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron.gdar_layer import (
        gdar_knobs_from_kwargs,
    )

    # 比对"生效配置"而不是 params 字典：显式写出默认值与省略它，对层是同一件事（两条路都进
    # GdarConfig 的同一个字段），字典比对会把这类差异误报成规格不一致。
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

    # ---------------- 4) 建模型 + 灌权重（verl 的调用顺序）
    # verl 在这里调 apply_overrides_and_finalize（BridgeTransformerConfig 把 mcore 的
    # __post_init__ 推迟到 finalize，init_method 那些派生字段都在那一步才算出来）；旧版回退到手写
    # setattr + finalize。这里照抄同一段逻辑，只保留结构性的 overrides。
    provider_overrides = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "variable_seq_lengths": True,  # verl 也设这个（本配方的 depth 层另有自己的处理）
        # 与各臂 config/default.yaml 里 actor/ref 的 override_transformer_config 同一项：
        # 本环境没有 apex，mcore 默认开着的 gradient_accumulation_fusion 会在建列并行层时报错。
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

    # ---------------- 5) 前向对拍（HF fp32 vs mcore fp32，核数值底噪口径）
    torch.manual_seed(1)
    ids = torch.randint(0, int(hf_cfg.vocab_size), (1, 16))
    hf_model = bridge.hf_pretrained.model.eval()
    hf_device = next(hf_model.parameters()).device  # AutoBridge 可能已经把 HF 模型放上了 GPU
    with torch.no_grad():
        hf_logits = hf_model(ids.to(hf_device)).logits.float().cpu()
        mc_logits = model[0](ids.cuda(), position_ids=None, attention_mask=None)
    mc_logits = mc_logits[0] if isinstance(mc_logits, (tuple, list)) else mc_logits
    mc_logits = mc_logits.float().cpu()
    if mc_logits.shape[:2] != hf_logits.shape[:2]:
        mc_logits = mc_logits.transpose(0, 1)
    d = (hf_logits - mc_logits).abs().max().item()
    check("logits 对拍（HF vs TE-mcore，底噪口径 |Δ| ≤ 2e-2）", d <= 2e-2, f"max|Δ| = {d:.3e}")

    # ---------------- 6) 导出（rollout 同步走的路）：名字齐、数值逐位
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
