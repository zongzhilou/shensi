"""RealFormer 的 mcore 侧单测：恒等、共享权重与门可学。"""


from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/louzo/code/shensi/shensi/src")
os.environ.setdefault("SHENSI_ROOT", "/home/louzo/code/shensi/shensi")
os.environ.setdefault("SHENSI_FS", "/home/louzo/fsdata")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29561")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

from megatron.core.parallel_state import initialize_model_parallel  # noqa: E402

initialize_model_parallel(1, 1)

from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")


def make_config(**over):
    kw = dict(
        num_layers=4,
        hidden_size=256,
        ffn_hidden_size=768,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=64,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        gated_linear_unit=True,
        init_method_std=0.02,
        add_bias_linear=False,
        qk_layernorm=True,
        transformer_impl="local",
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )
    kw.update(over)
    return TransformerConfig(**kw)


def build(cfg, spec):
    return GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=1024,
        max_sequence_length=128,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=1000000,
    ).cuda()


def main() -> int:
    from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import realformer_spec as rf

    cfg = make_config()
    ids = torch.randint(0, 1024, (1, 16)).cuda()

    for algo, spec, expect_identity in (
        ("identity (gate=zero)", rf.realformer_layer_spec_identity, True),
        ("deviation (gate=0 起手)", rf.realformer_layer_spec, True),
        ("reference (上游 gate=1)", rf.realformer_layer_spec_reference, False),
    ):
        model_parallel_cuda_manual_seed(0)
        plain = build(cfg, get_gpt_layer_local_spec(
            num_experts=None, moe_grouped_gemm=False, qk_layernorm=True, normalization="RMSNorm"
        )).eval()
        model_parallel_cuda_manual_seed(0)
        rf_model = build(cfg, spec).eval()

        with torch.no_grad():
            a = plain(ids, position_ids=None, attention_mask=None)
            b = rf_model(ids, position_ids=None, attention_mask=None)
        a = a[0] if isinstance(a, (tuple, list)) else a
        b = b[0] if isinstance(b, (tuple, list)) else b
        d = (a.float() - b.float()).abs().max().item()
        if expect_identity:
            check(f"{algo}：与 plain Qwen3 逐位一致", d == 0.0, f"max|Δ| = {d:.3e}")
        else:
            check(f"{algo}：与 plain Qwen3 不同（连接真的在起作用）", d > 0.0, f"max|Δ| = {d:.3e}")

        ps = {
            k: v
            for k, v in rf_model.state_dict().items()
            if "carry_gate" not in k and isinstance(v, torch.Tensor)
        }
        pl = {k: v for k, v in plain.state_dict().items() if isinstance(v, torch.Tensor)}
        same_keys = set(ps) == set(pl)
        same_vals = same_keys and all(torch.equal(ps[k], pl[k]) for k in ps)
        check(f"{algo}：共享权重与 plain 逐位相同", same_vals, "" if same_keys else "键集合不同")
        del plain, rf_model
        torch.cuda.empty_cache()

    model_parallel_cuda_manual_seed(0)
    model = build(cfg, rf.realformer_layer_spec).train()
    out = model(ids, position_ids=None, attention_mask=None)
    out = out[0] if isinstance(out, (tuple, list)) else out
    loss = out.float().pow(2).mean()
    loss.backward()
    gates = {n: p for n, p in model.named_parameters() if n.endswith("carry_gate")}
    check("deviation 档每层一个 gate（layer 0 除外）", len(gates) == cfg.num_layers - 1, f"{len(gates)} 个")
    check("每个 gate 都拿到梯度", all(p.grad is not None for p in gates.values()), "")
    check(
        "gate 命名落在 core_attention（转换表按这个名字找）",
        all(".self_attention.core_attention.carry_gate" in n for n in gates),
        next(iter(gates), ""),
    )
    with torch.no_grad():
        for p in gates.values():
            p.fill_(0.5)
    out2 = model(ids, position_ids=None, attention_mask=None)
    out2 = out2[0] if isinstance(out2, (tuple, list)) else out2
    check("gate 非零后输出变化（残差注意力接上了）", not torch.equal(out.detach(), out2.detach()), "")

    pp_cfg = make_config(pipeline_model_parallel_size=2, pipeline_dtype=torch.float32)
    try:
        from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.realformer_spec import (
            realformer_layer_spec,
        )

        t = realformer_layer_spec.module.__new__(realformer_layer_spec.module)
        from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.realformer_layer import (
            RealFormerTransformerLayer,
        )

        RealFormerTransformerLayer.__init__(
            t, config=pp_cfg, layer_number=1, pg_collection=None, realformer_gate="deviation"
        )
        check("pp>1 被拒绝", False, "没有报错")
    except NotImplementedError as exc:
        check("pp>1 被拒绝", "pipeline" in str(exc), str(exc)[:60])

    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
