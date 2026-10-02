#!/usr/bin/env python3
"""RealFormer 的恒等与转写对照单测。"""



from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.configuration_qwen3_realformer import (  # noqa: E402
    Qwen3RealFormerConfig,
)
from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_realformer import (  # noqa: E402
    Qwen3RealFormerForCausalLM,
    residual_attention,
)
from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.upstream.realformer_torch_reference import (  # noqa: E402
    upstream_residual_attention,
)

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")


def tiny_config(**knobs) -> Qwen3RealFormerConfig:
    return Qwen3RealFormerConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        **knobs,
    )


def main() -> int:
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 12))


    ref = None
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    for mode in ("deviation", "zero"):
        cfg = tiny_config(attn_res_realformer_gate=mode)
        cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        model = Qwen3RealFormerForCausalLM(cfg).eval()
        if ref is None:

            qcfg = cfg.to_dict()
            for key in list(qcfg):
                if key.startswith("attn_res_") or key in ("auto_map", "architectures"):
                    qcfg.pop(key)
            qcfg["model_type"] = "qwen3"
            qcfg["architectures"] = ["Qwen3ForCausalLM"]
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

            torch.manual_seed(0)
            ref_cfg = Qwen3Config(**{k: v for k, v in qcfg.items() if k in Qwen3Config.__annotations__})
            ref_cfg._attn_implementation = "eager"
            ref = Qwen3ForCausalLM(ref_cfg).eval()



        ours = {n: v for n, v in model.named_parameters() if not n.endswith("delta")}
        theirs = dict(ref.named_parameters())
        if set(ours) != set(theirs):
            check(f"恒等：gate={mode} 的参数名集合与 plain Qwen3 一致", False,
                  f"多={sorted(set(ours) - set(theirs))[:3]} 少={sorted(set(theirs) - set(ours))[:3]}")
        else:
            same = all(torch.equal(ours[k], theirs[k]) for k in ours)
            check(f"恒等：gate={mode} 的权重与 plain Qwen3 逐位相同", same, "")
        with torch.no_grad():
            a = model(ids).logits.float()
            b = ref(ids).logits.float()

        d = (a - b).abs().max().item()
        check(f"恒等：gate={mode} 的 logits 与 plain Qwen3 逐位一致", d == 0.0, f"max|Δ| = {d:.3e}")


    from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.upstream.realformer_torch_reference import (
        upstream_attention_scores,
        upstream_residual_from_scores,
    )

    torch.manual_seed(2)
    query = torch.randn(2, 4, 6, 16)
    key = torch.randn(2, 4, 6, 16)
    scores = upstream_attention_scores(query, key)
    prev = torch.randn(2, 4, 6, 6) * 0.5
    mask = torch.triu(torch.full((2, 1, 6, 6), float("-inf")), diagonal=1)

    probs_one, cur_one = residual_attention(scores, prev, torch.ones(()), mask, num_prev_layers=1)
    ref_probs, ref_cur = upstream_residual_from_scores(
        scores, prev, 1, attention_mask=mask, mask_is_binary=False
    )
    check(
        "gate=1 时概率与上游转写逐位一致",
        (probs_one - ref_probs).abs().max().item() == 0.0,
        f"max|Δ| = {(probs_one - ref_probs).abs().max().item():.3e}",
    )
    check("gate=1 时传下去的 cur_attention 逐位一致", torch.equal(cur_one, ref_cur), "")

    probs_zero, cur_zero = residual_attention(scores, prev, torch.zeros(()), mask, num_prev_layers=1)
    ref_probs0, ref_cur0 = upstream_residual_from_scores(
        scores, None, 1, attention_mask=mask, mask_is_binary=False
    )
    check(
        "gate=0 时与上游 layer 0（无 prev）逐位一致",
        (probs_zero - ref_probs0).abs().max().item() == 0.0 and torch.equal(cur_zero, ref_cur0),
        "",
    )
    check(
        "cur_attention 是 softmax **前**的累加（不等于概率）",
        not torch.allclose(cur_one, probs_one),
        "",
    )


    probs_m, cur_m = residual_attention(
        scores, prev, torch.ones(()), mask, use_running_mean=True, num_prev_layers=3
    )
    ref_probs_m, _ = upstream_residual_from_scores(
        scores, prev, 3, use_running_mean=True, attention_mask=mask, mask_is_binary=False
    )
    check(
        "running mean：概率与上游逐位一致",
        (probs_m - ref_probs_m).abs().max().item() == 0.0,
        f"max|Δ| = {(probs_m - ref_probs_m).abs().max().item():.3e}",
    )
    check("running mean：传下去的 cur 与不除时相同（上游语义）", torch.equal(cur_m, cur_one), "")


    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    cfg_one = tiny_config(attn_res_realformer_gate="one")
    cfg_one._attn_implementation = "eager"
    torch.manual_seed(1)
    model_one = Qwen3RealFormerForCausalLM(cfg_one).eval()
    attn = model_one.model.layers[1].self_attn
    torch.manual_seed(4)
    hidden = torch.randn(2, 6, 64)
    pos = torch.arange(6).unsqueeze(0).expand(2, 6)
    with torch.no_grad():
        cos, sin = model_one.model.rotary_emb(hidden, pos)
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        kk = repeat_kv(k, attn.num_key_value_groups)
        vv = repeat_kv(v, attn.num_key_value_groups)
        out, cur_e2e = attn(
            hidden_states=hidden,
            attention_mask=mask,
            position_embeddings=(cos, sin),
            prev_attention=prev,
            num_prev_layers=1,
        )
        ref_out, ref_cur_e2e = upstream_residual_attention(
            q, kk, vv, prev, 1, attention_mask=mask, mask_is_binary=False
        )
        ref_out = attn.o_proj(ref_out.transpose(1, 2).contiguous().reshape(2, 6, -1))
    check(
        "端到端：注意力输出与上游转写逐位一致",
        torch.equal(out, ref_out),
        f"max|Δ| = {(out - ref_out).abs().max().item():.3e}",
    )
    check("端到端：carry 与上游逐位一致", torch.equal(cur_e2e, ref_cur_e2e), "")



    cfg_dec = tiny_config()
    cfg_dec._attn_implementation = "eager"
    torch.manual_seed(5)
    gen_model = Qwen3RealFormerForCausalLM(cfg_dec).eval()
    prefix = torch.randint(0, 256, (1, 8))

    with torch.no_grad():
        full = gen_model(prefix.new_tensor([[*prefix[0].tolist(), 7, 9, 11]])).logits

    with torch.no_grad():
        out = gen_model(prefix, use_cache=True)
        past = out.past_key_values
        logits = []
        for tok in (7, 9, 11):
            out = gen_model(
                torch.tensor([[tok]]), past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            logits.append(out.logits[:, -1, :])
        inc = torch.cat(logits, dim=0).unsqueeze(0)
    d = (full[:, -3:, :].float() - inc.float()).abs().max().item()



    check(
        "增量解码（KV cache）与全序列前向一致（核噪声口径 ≤ 1e-6）",
        d <= 1e-6,
        f"max|Δ| = {d:.3e}（carry 是层间状态 ⇒ 解码无需额外缓存）",
    )


    cfg = tiny_config()
    torch.manual_seed(3)
    m = Qwen3RealFormerForCausalLM(cfg)
    gates = [mod for mod in m.modules() if type(mod).__name__ == "GateScale"]
    check("layer>=1 每层一个 gate（layer 0 没有可加项，不建）",
          len(gates) == cfg.num_hidden_layers - 1, f"{len(gates)} 个")
    check("deviation 档的 gate 可训练且初始为 0", all(g.value().requires_grad and float(g.value()) == 0.0 for g in gates), "")
    with torch.no_grad():
        for g in gates:
            g.delta.fill_(0.25)
    with torch.no_grad():
        a = m(ids).logits
    g0 = [gg for gg in m.modules() if type(gg).__name__ == "GateScale"]
    with torch.no_grad():
        for gg in g0:
            if gg.mode == "deviation":
                gg.delta.zero_()
    with torch.no_grad():
        b = m(ids).logits
    check("gate 非零时输出确实变了（连接真的在起作用）", not torch.allclose(a, b), f"max|Δ| = {(a - b).abs().max().item():.3e}")


    check(
        "auto_map 三件套齐备",
        set(cfg.auto_map) == {"AutoConfig", "AutoModel", "AutoModelForCausalLM"},
        str(sorted(cfg.auto_map)),
    )
    out = m(ids, labels=ids)
    out.loss.backward()
    grads = [p.grad is not None for n, p in m.named_parameters() if n.endswith("delta")]
    check("反向可走通且每个 gate 都拿到梯度", all(grads) and len(grads) == cfg.num_hidden_layers - 1,
          f"{len(grads)} 个 delta，全部有梯度={all(grads)}")

    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
