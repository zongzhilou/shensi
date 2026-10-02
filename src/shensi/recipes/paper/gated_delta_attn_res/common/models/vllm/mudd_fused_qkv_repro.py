"""Minimal reproduction: MUDD breaks in an inference engine *because* it fuses QKV.

``mudd`` was the only one of the 7 variants that could not generate under vLLM
(``../README.md（本目录）与包内 code/ROLLOUT_ENV.md`` 5).  The engine fuses ``q_proj`` / ``k_proj`` / ``v_proj``
into one ``qkv_proj`` and **deletes the three attributes**, while MUDD's
``Qwen3MUDDDecoderLayer._multiway_attention`` -- which feeds three different streams
to ``MHA(LN(X^Q), LN(X^K), LN(X^V))``, the "multiway" in MUDDFormer -- reads
``attn.q_proj`` / ``attn.k_proj`` / ``attn.v_proj`` directly.

This script isolates that claim from everything else, using

* **the engine's own fusion code** (``vLLM``'s ``QKVFuser``: the very class that
  prints ``Fused: q_proj + k_proj + v_proj (...) -> qkv_proj (QKVParallelLinear)``
  and that raises ``ValueError("Layer N does not dispatch ...")`` if it does not
  match), applied by hand to a plain transformers model on the **CPU**, with no
  ``vllm.LLM`` and no GPU anywhere, and
* the **unmodified** ``models/modeling_qwen3_mudd.py`` forward.

It reports, in order:

1. the plain (unfused) HF forward, as the reference logits;
2. the engine's fusion applied to every ``self_attn``, and the identity
   ``qkv_proj(x) == cat(q_proj(x), k_proj(x), v_proj(x))`` that makes the fusion
   arithmetically neutral -- i.e. fusing is not "wrong", it just renames;
3. the pre-fix expression ``attn.q_proj(xq)`` executed against the fused module:
   the ``AttributeError`` the engine reported, reproduced here without an engine;
4. the *current* ``_multiway_attention`` on the same fused module, compared with
   the reference of step 1 -- the check that the fix (slicing the fused linear
   back into three projections) is arithmetically exact.

Usage
-----
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.mudd_fused_qkv_repro            # both num_ways
    .venv/bin/python -m shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.mudd_fused_qkv_repro --num-ways 4
"""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path

from ._paths import RECIPE, SRC, ensure_src_on_path

ensure_src_on_path()

from .variants import BY_KEY, tiny_base  # noqa: E402

#: The three projections the engine's QKV fuser consumes.
QKV_NAMES = ("q_proj", "k_proj", "v_proj")


# --------------------------------------------------------------------------- #
# the engine's fusion, applied by hand
# --------------------------------------------------------------------------- #
def _vllm_config_env():
    """vLLM's fusion code needs a ``VllmConfig`` and an initialized TP group.

    The engine has both; a standalone script has neither.  ``gloo`` + world size 1
    is enough -- the fused linear is built with ``tp_size == 1``, which is the
    geometry the engine builds too (and the only one this smoke test uses).
    """
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config

    import vllm.distributed.parallel_state as ps

    if not torch.distributed.is_initialized():
        ps.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
            backend="gloo",
        )
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        if ps.model_parallel_is_initialized() is False:
            ps.initialize_model_parallel(1)
    return vllm_config


def _load_fused_weights(module, kept: dict) -> None:
    """Put the original weights into the modules the fusion built.

    This is the engine's own load path -- ``packed_modules_mapping`` routes
    ``q_proj``/``k_proj``/``v_proj`` to ``qkv_proj`` with shard ids ``q``/``k``/``v``,
    and ``QKVParallelLinear``'s ``weight_loader`` decides where the rows go
    (``_get_shard_offset_mapping``: ``q`` at 0, ``k`` at ``num_heads * head_size``,
    ``v`` after it).  A fused linear and a re-classed ``o_proj`` are born with
    *uninitialised* weights -- the engine fills them from the checkpoint later -- so
    skipping this would compare NaNs.
    """
    import torch

    fused = module.qkv_proj
    for name, shard_id in zip(QKV_NAMES, ("q", "k", "v")):
        weight = kept[name].weight
        with torch.no_grad():
            fused.weight.weight_loader(fused.weight, weight.to(fused.weight.dtype), shard_id)
            if fused.bias is not None and kept[name].bias is not None:
                fused.bias.weight_loader(fused.bias, kept[name].bias, shard_id)
    new_o, old_o = getattr(module, "o_proj", None), kept["o_proj"]
    if new_o is not None and new_o is not old_o:
        with torch.no_grad():
            try:
                target = new_o.weight
                target.weight_loader(target, old_o.weight.to(target.dtype))
            except Exception:
                new_o.weight.copy_(old_o.weight.to(new_o.weight.dtype))
            if getattr(new_o, "bias", None) is not None and old_o.bias is not None:
                new_o.bias.copy_(old_o.bias.to(new_o.bias.dtype))


def fuse_qkv(model, vllm_config) -> list[dict]:
    """Run vLLM's ``QKVFuser`` over every ``self_attn``; returns one record each.

    This is the same sequence ``recursive_replace`` performs per module
    (``vllm/model_executor/models/transformers/base.py``):
    ``Fusers.__getitem__`` -> ``QKVFuser.match`` + ``update_forward`` (rewrite the
    class's forward source), then ``fuse()`` -> ``update_attrs`` (build the fused
    linear, **delete q/k/v**, re-class ``o_proj``).
    """
    from vllm.model_executor.models.transformers.fusers import QKVFuser
    from vllm.model_executor.models.transformers.fusers.base import fused_head_size
    from vllm.model_executor.models.transformers.fx_utils import trace

    records: list[dict] = []
    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        graph = trace(module)
        fuser = QKVFuser.match(graph, module)  # the engine's own matcher
        if fuser is None:
            raise RuntimeError(f"vLLM's QKVFuser did not match {name} ({type(module).__name__})")
        kept = {q: getattr(module, q) for q in (*QKV_NAMES, "o_proj")}
        fuser.update_forward(module)  # rewrite Qwen3Attention.forward (source)
        fuser.fuse(module, name, vllm_config)  # update_attrs + rebind forward
        _load_fused_weights(module, kept)
        records.append(
            {
                "name": name,
                "module": module,
                "fuser": fuser,
                "kept": kept,
                "head_size": fused_head_size(module, vllm_config),
                "qkv_proj": getattr(module, "qkv_proj"),
            }
        )
    return records


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--num-ways",
        type=int,
        default=0,
        help="mudd_num_ways (0 = sweep 1 and 4); 2 is invalid in the model",
    )
    ap.add_argument(
        "--sepln",
        action="store_true",
        default=True,
        help="mudd_sepln: give q/k/v their own layernorms (distinct streams)",
    )
    ap.add_argument("--no-sepln", dest="sepln", action="store_false")
    ap.add_argument("--seq", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    from shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.modeling_qwen3_mudd import (
        Qwen3MUDDConfig,
        Qwen3MUDDForCausalLM,
    )

    print(f"torch {torch.__version__} | python {sys.version.split()[0]} | device=cpu")
    vllm_config = _vllm_config_env()
    print("vLLM fusion environment ready (gloo, world_size=1, VllmConfig())\n")

    ways = [args.num_ways] if args.num_ways else [1, 4]
    failures = 0
    for num_ways in ways:
        failures += _check_one(
            num_ways=num_ways,
            sepln=args.sepln,
            seq=args.seq,
            seed=args.seed,
            vllm_config=vllm_config,
            torch=torch,
            cfg_cls=Qwen3MUDDConfig,
            model_cls=Qwen3MUDDForCausalLM,
        )

    print("=" * 100)
    print("all checks passed" if not failures else f"FAILED: {failures} check(s)")
    return 1 if failures else 0


def _check_one(*, num_ways, sepln, seq, seed, vllm_config, torch, cfg_cls, model_cls) -> int:
    variant = BY_KEY["mudd"]
    base = tiny_base(256)
    extra = dict(variant.tiny_knobs, mudd_num_ways=num_ways, mudd_sepln=sepln)
    print("=" * 100)
    print(
        f"mudd_num_ways={num_ways} mudd_sepln={sepln}  "
        f"(heads={base['num_attention_heads']} kv_heads={base['num_key_value_heads']} "
        f"head_dim={base['head_dim']} layers={base['num_hidden_layers']})"
    )
    print("=" * 100)

    failures = 0

    def check(cond, msg, detail=""):
        nonlocal failures
        print(f"  [{'ok ' if cond else 'FAIL'}] {msg}{(' -> ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    torch.manual_seed(seed)
    config = cfg_cls(**base, **extra)
    model = model_cls(config).eval()  # unmodified models/ code
    fused = copy.deepcopy(model)  # the same weights, then fused
    ids = torch.randint(0, 256, (1, seq))
    pos = torch.arange(seq).unsqueeze(0)

    with torch.no_grad():
        ref = model(ids, position_ids=pos, use_cache=False).logits

    head_dim = model.model.layers[0].self_attn.head_dim
    q_dim = base["num_attention_heads"] * head_dim
    kv_dim = base["num_key_value_heads"] * head_dim
    print(
        f"\n  HF trunk geometry per layer: q={q_dim} k={kv_dim} v={kv_dim} "
        f"(fused qkv_proj = {q_dim + 2 * kv_dim} rows)"
    )

    # ---- 1./2. the engine's fusion -----------------------------------------
    print("\n1. vLLM's own QKVFuser applied to every self_attn (no engine, no GPU)")
    records = fuse_qkv(fused, vllm_config)
    check(
        len(records) == base["num_hidden_layers"],
        f"fused {len(records)} attention modules (one per layer)",
    )
    rec = records[0]
    fuser = rec["fuser"]
    lm = fused.model.layers[0].self_attn
    check(
        not any(hasattr(lm, q) for q in QKV_NAMES),
        "the engine deleted q_proj/k_proj/v_proj from the attention module",
    )
    check(
        getattr(lm, "qkv_proj", None) is not None, "qkv_proj (QKVParallelLinear) took their place"
    )
    print(f"     fuser: {fuser.info(rec['name'])}")
    print(
        f"     QKVParallelLinear: weight={tuple(lm.qkv_proj.weight.shape)} "
        f"output_sizes={lm.qkv_proj.output_sizes} tp_size={lm.qkv_proj.tp_size}"
    )

    # the fusion is arithmetically neutral: qkv(x) == cat(q(x), k(x), v(x))
    x = torch.randn(5, base["hidden_size"])
    with torch.no_grad():
        cat = torch.cat([rec["kept"][q](x) for q in QKV_NAMES], dim=-1)
        got = lm.qkv_proj(x)
    check(
        torch.equal(cat, got),
        "qkv_proj(x) == cat(q_proj(x), k_proj(x), v_proj(x))   [bit-exact]",
        f"max|diff| = {float((cat - got).abs().max()):.3e}",
    )

    # ---- 3. the pre-fix expression ----------------------------------------
    print("\n2. what the model's code did before the fix (the engine's error, reproduced)")
    attn = lm
    try:
        with torch.no_grad():
            _ = attn.q_norm(attn.q_proj(x).view(5, -1, head_dim)).transpose(1, 2)
        check(False, "attn.q_proj(x) should have raised")
    except AttributeError as exc:
        print(f"     attn.q_proj(x)  ->  AttributeError: {exc}")
        print(
            f"     (models/modeling_qwen3_mudd.py, _multiway_attention: "
            f"{'q_proj' in str(exc)}, line of the old call)"
        )
        check(
            "q_proj" in str(exc) and type(attn).__name__ in str(exc),
            "raises exactly the engine's AttributeError, with no engine involved",
            str(exc),
        )
        print("     captured traceback:")
        print("\n".join("       " + ln for ln in traceback.format_exc().strip().splitlines()[-6:]))

    # ---- 4. current code on the fused module ------------------------------
    print("\n3. current _multiway_attention on the same fused model vs the HF reference")
    try:
        with torch.no_grad():
            out = fused(ids, position_ids=pos, use_cache=False).logits
    except Exception as exc:
        check(False, "fused model forward failed", f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc()[-2000:])
        return failures
    diff = float((out - ref).abs().max())
    check(
        diff == 0.0,
        "fused model logits == unfused HF logits   [bit-exact]",
        f"max|diff| = {diff:.3e}",
    )
    check(bool(torch.equal(out, ref)), "torch.equal(fused, unfused)")
    if diff != 0.0:
        print(f"     non-zero but tiny: {diff:.3e} (fp noise, not a layout error)")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
