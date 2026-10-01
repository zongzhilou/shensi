#!/usr/bin/env python3
"""把 mcore ``torch_dist`` 检查点导出成 HF 目录（评测、rollout 与上线都读它）。

从检查点自己的 ``run_config.yaml`` 读几何与词表，建 HF 配置并经 ``AutoBridge`` 建出 mcore
模型（层规格由 ``looma_bridge.build_layer_spec`` 从 ``looma_*`` 旋钮生成，与配方训练器是同一
份 ``make_looma_spec``）；用 ``dist_checkpointing.load`` 裸载权重，载入前把独立 norm 名改成
检查点里的融合名；由 ``bridge.save_hf_pretrained`` 反向导出，再复制 tokenizer 与
``configuration_looma.py`` / ``modeling_looma.py``。``--verify`` 会重新加载导出目录，与 mcore
模型逐张量、逐 logits 比对（判据是容差而非逐位：骨干两侧走不同的注意力内核）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

RECIPE = Path(__file__).resolve().parents[2]

_GEOMETRY: dict[str, tuple[str, object]] = {
    "num_hidden_layers": ("model.transformer.num_layers", None),
    "hidden_size": ("model.transformer.hidden_size", None),
    "intermediate_size": ("model.transformer.ffn_hidden_size", None),
    "num_attention_heads": ("model.transformer.num_attention_heads", None),
    "num_key_value_heads": ("model.transformer.num_query_groups", None),
    "head_dim": ("model.transformer.kv_channels", None),
    "max_position_embeddings": ("model.seq_length", None),
    "rms_norm_eps": ("model.transformer.layernorm_epsilon", 1e-6),
    "rope_theta": ("model.transformer.rotary_base", 1000000.0),
}
_VOCAB_DIVISIBLE = ("model.make_vocab_size_divisible_by", 64)

_TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    }
)

_NORM_MODEL_TO_CKPT = {
    "self_attention.input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
    "mlp.pre_mlp_layernorm.weight": "mlp.linear_fc1.layer_norm_weight",
}
_NORM_CKPT_TO_MODEL = {v: k for k, v in _NORM_MODEL_TO_CKPT.items()}

_VERIFY_TOL = {"bf16": 5e-2, "fp32": 1e-3, "single_token": 5e-2, "wiring": 1e-3}


def _dig(obj, dotted: str):
    """按点分路径从嵌套字典里取值，任一层缺失返回 None。"""
    node = obj
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _remap(d: dict, table: dict) -> dict:
    """按后缀表把参数名从一侧的命名改写成另一侧的命名。"""
    out = {}
    for name, value in d.items():
        for old, new in table.items():
            if name.endswith(old):
                name = name[: -len(old)] + new
                break
        out[name] = value
    return out


def _init_distributed(device: str) -> None:
    """单进程点起分布式与 mcore 并行组：``sharded_state_dict()`` 与建模型都要它们在场。"""
    import os

    from megatron.core import parallel_state as mpu

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    if device == "cuda":
        torch.cuda.set_device(0)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)


def _bind_pg_collection(provider) -> None:
    """给 provider 挂上建模型时要读的 ``ProcessGroupCollection``。"""
    from megatron.core.process_groups_config import ProcessGroupCollection

    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()  # noqa: SLF001


def _ckpt_tensor_shapes(iter_dir: Path) -> dict[str, tuple[int, ...]]:
    """读检查点元数据，返回 ``{参数名: 全局形状}``。"""
    from torch.distributed.checkpoint import FileSystemReader

    meta = FileSystemReader(str(iter_dir)).read_metadata().state_dict_metadata
    shapes: dict[str, tuple[int, ...]] = {}
    for key, entry in meta.items():
        size = getattr(entry, "size", None)
        if size is not None:
            shapes[key] = tuple(int(s) for s in size)
    return shapes


def _check_shapes(model, iter_dir: Path) -> None:
    """载入前逐张量对形状，把 ``Global shape mismatch`` 变成可照做的提示。"""
    shapes = _ckpt_tensor_shapes(iter_dir)
    bad = []
    for name, tensor in model.sharded_state_dict().items():
        expected = shapes.get(name)
        if expected is None:
            continue
        got = getattr(tensor, "local_shape", None) or getattr(tensor, "shape", None)
        if got is None:
            got = getattr(getattr(tensor, "data", None), "shape", None)
        if got is None:
            continue
        if tuple(got) != tuple(expected):
            bad.append((name, tuple(got), expected))
    if bad:
        preview = "\n  ".join(f"{n}: 模型 {a} vs 检查点 {b}" for n, a, b in bad[:5])
        raise SystemExit(f"[looma·export] 有 {len(bad)} 个张量形状对不上：\n  {preview}")


def _resolve_iter_dir(ckpt: Path, load_iter: int | None) -> Path:
    """定位要载入的 iter 目录。

    依次看：``ckpt`` 本身（含 ``metadata.json``）、``--load-iter`` 指的目录、
    ``latest_checkpointed_iteration.txt`` 记的编号、编号最大的 ``iter_*``。
    """
    if (ckpt / "metadata.json").is_file():
        return ckpt
    if load_iter is not None:
        cand = ckpt / f"iter_{load_iter:07d}"
        if not cand.is_dir():
            raise SystemExit(f"[looma·export] 没有这个 iter：{cand}")
        return cand
    latest = ckpt / "latest_checkpointed_iteration.txt"
    if latest.is_file():
        it = int(latest.read_text().strip())
        cand = ckpt / f"iter_{it:07d}"
        if cand.is_dir():
            return cand
    iters = sorted(p for p in ckpt.glob("iter_*") if p.is_dir())
    if not iters:
        raise SystemExit(f"[looma·export] {ckpt} 里没有 iter_* 目录")
    return iters[-1]


def _build_hf_config(ckpt_rc: dict, run_cfg: dict, tok_vocab: int, args) -> object:
    """建 ``LoomaConfig``：几何优先取检查点里 mcore 自己写的记录，``--run-config`` 只兜底。

    词表以检查点为准，tokenizer 的长度只在检查点没记录时才用（冒烟档的 NullTokenizer 与真
    tokenizer 长度不同）。返回 ``(配置, 词表补齐粒度, 几何取值)``。
    """
    sys.path.insert(0, str(RECIPE.parents[4]))
    from shensi.recipes.paper.looma.common.models.transformers.configuration_looma import (
        LoomaConfig,
    )

    values: dict = {}
    missing: list[str] = []
    for field, (path, default) in _GEOMETRY.items():
        value = _dig(ckpt_rc, path)
        if value is None:
            value = (
                _dig(
                    run_cfg,
                    path.replace("model.transformer.", "train.model.").replace(
                        "model.seq_length", "train.model.max_position_embeddings"
                    ),
                )
                or default
            )
        if value is None:
            missing.append(field)
        else:
            values[field] = value
    if missing:
        raise SystemExit(
            f"[looma·export] 检查点没记录这些几何：{missing}"
            "（用 --run-config 指到训练目录的 config.yaml 兜底）"
        )
    divisible = _dig(ckpt_rc, _VOCAB_DIVISIBLE[0]) or _VOCAB_DIVISIBLE[1]
    ckpt_vocab = _dig(ckpt_rc, "model.vocab_size")
    vocab_size = int(ckpt_vocab) if ckpt_vocab else int(tok_vocab)
    values["max_position_embeddings"] = int(
        args.max_position_embeddings or values["max_position_embeddings"]
    )
    cfg = LoomaConfig(
        vocab_size=int(vocab_size),
        tie_word_embeddings=False,
        **{
            k: int(v) if k != "rms_norm_eps" and k != "rope_theta" else float(v)
            for k, v in values.items()
        },
    )
    cfg.architectures = ["LoomaForCausalLM"]
    cfg._attn_implementation = "eager"
    return cfg, int(divisible), values


def _trained_spec(run_cfg: dict):
    """读训练时那份 ``train.model.spec``（``(模块, 对象)``），没有就返回 None。"""
    spec = _dig(run_cfg, "train.model.spec")
    if not spec:
        return None
    if isinstance(spec, str):
        spec = [s for s in spec.replace(",", " ").split() if s]
    return list(spec) if len(spec) >= 2 else None


def _verify_weights(out: Path, model, bridge) -> tuple[bool, str]:
    """V1：导出文件里的每张量与 mcore 模型里对应的那张逐位相等。

    同名行直接逐位比；融合行（q/k/v 与 gate/up）两侧布局不同，跳过计数、留给 V2 的 logits
    覆盖。查表按导出方向（mcore → HF）走，反向查会在同名不同布局的档上命中融合行而误报。
    """
    from safetensors.torch import load_file

    state = load_file(str(out / "model.safetensors"))
    registry = bridge._model_bridge.mapping_registry()  # noqa: SLF001
    by_name = dict(model.named_parameters())
    compared, skipped, bad = 0, 0, []
    for mc_name, param in by_name.items():
        if mc_name.endswith("_extra_state") or not mc_name.startswith(
            ("decoder.", "embedding.", "output_layer.")
        ):
            continue
        mapping = registry.megatron_to_hf_lookup(mc_name)
        if mapping is None or not isinstance(mapping.hf_param, str):
            skipped += 1
            continue
        tensor = state.get(mapping.hf_param)
        if tensor is None:
            bad.append((mapping.hf_param, mc_name, "导出文件里没有这个张量"))
            continue
        if tuple(param.shape) != tuple(tensor.shape):
            bad.append(
                (mapping.hf_param, mc_name, f"形状 {tuple(param.shape)} vs {tuple(tensor.shape)}")
            )
            continue
        if not torch.equal(param.detach().to(tensor.dtype).cpu(), tensor.cpu()):
            delta = float((param.detach().float().cpu() - tensor.float().cpu()).abs().max())
            bad.append((mapping.hf_param, mc_name, f"max|Δ| = {delta:.3e}"))
            continue
        compared += 1
    if bad:
        preview = "\n  ".join(f"{h} <- {m}: {why}" for h, m, why in bad[:6])
        return False, f"{len(bad)} 个张量对不上：\n  {preview}"
    return True, f"{compared} 个张量逐位相等（{skipped} 个融合行由 logits 校验覆盖）"


def _interleave_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_groups: int, head_dim: int
):
    """把分开的 q/k/v 拼回 mcore 的**交错**融合布局：每组 ``[q_heads..., k, v]``。

    ``Attention.get_query_key_value_tensors`` 就按这个约定沿最后一维切开，反向拼回来必须同一
    套约定；这一条只看得见融合行，而 V1 恰好覆盖不到那里。
    """
    per_group_q = q.shape[0] // num_groups
    q = q.reshape(num_groups, per_group_q, -1)
    k = k.reshape(num_groups, head_dim, -1)
    v = v.reshape(num_groups, head_dim, -1)
    return torch.cat([q, k, v], dim=1).reshape(-1, q.shape[-1])


def _verify_fused(out: Path, model) -> tuple[bool, str]:
    """V3：把导出文件里的 q/k/v 与 gate/up 按交错约定重建，与 mcore 的融合参数逐位比。"""
    from safetensors.torch import load_file

    state = load_file(str(out / "model.safetensors"))
    by_name = dict(model.named_parameters())
    num_layers = int(model.config.num_layers)
    num_groups = int(model.config.num_query_groups)
    head_dim = int(model.config.kv_channels)
    bad, compared = [], 0
    for layer in range(num_layers):
        qkv_name = f"decoder.layers.{layer}.self_attention.linear_qkv.weight"
        qkv = by_name.get(qkv_name)
        if qkv is None:
            bad.append((qkv_name, "", "模型里没有这个参数"))
            continue
        pieces = [
            state.get(f"model.layers.{layer}.self_attn.{proj}_proj.weight")
            for proj in ("q", "k", "v")
        ]
        if any(x is None for x in pieces):
            bad.append((qkv_name, "", "导出文件里缺 q/k/v"))
            continue
        rebuilt = _interleave_qkv(
            *(x.to(qkv.dtype) for x in pieces), num_groups=num_groups, head_dim=head_dim
        )
        if tuple(rebuilt.shape) != tuple(qkv.shape) or not torch.equal(
            rebuilt.cpu(), qkv.detach().cpu()
        ):
            delta = float((rebuilt.float().cpu() - qkv.detach().float().cpu()).abs().max())
            bad.append((qkv_name, "q/k/v -> 融合", f"max|Δ| = {delta:.3e}"))
        else:
            compared += 1

        fc1_name = f"decoder.layers.{layer}.mlp.linear_fc1.weight"
        fc1 = by_name.get(fc1_name)
        gate = state.get(f"model.layers.{layer}.mlp.gate_proj.weight")
        up = state.get(f"model.layers.{layer}.mlp.up_proj.weight")
        if fc1 is None or gate is None or up is None:
            bad.append((fc1_name, "gate/up -> 融合", "缺张量"))
            continue
        rebuilt = torch.cat([gate.to(fc1.dtype), up.to(fc1.dtype)], dim=0)
        if tuple(rebuilt.shape) != tuple(fc1.shape) or not torch.equal(
            rebuilt.cpu(), fc1.detach().cpu()
        ):
            delta = float((rebuilt.float().cpu() - fc1.detach().float().cpu()).abs().max())
            bad.append((fc1_name, "gate/up -> 融合", f"max|Δ| = {delta:.3e}"))
        else:
            compared += 1
    if bad:
        preview = "\n  ".join(f"{a}（{b}）: {c}" for a, b, c in bad[:6])
        return False, f"{len(bad)} 个融合行对不上：\n  {preview}"
    return True, f"{compared} 个融合行按 mcore 交错约定重建后逐位相等"


def _verify_wiring(model, reloaded, hf_cfg, device: str) -> tuple[bool, str]:
    """V2a：两侧的块都只跑一步（``max_iter=1``）后逐长度比 logits。

    两侧的迭代次数都是 1，接线上的任何错误（张量放错名字或位置、宽度打包错、K/V 冻结错）都会在
    所有长度上原样显形，阈值因此可以钉到 1e-3；参考设置（``max_iter`` 由配置给）下的差值只作报数。
    """
    saved = [(layer, layer.looma_cfg.max_iter) for layer in model.decoder.layers]
    saved_hf = [(layer, layer.solver_max_iter) for layer in reloaded.model.layers]
    try:
        for layer, _ in saved:
            layer.looma_cfg.max_iter = 1
        for layer, _ in saved_hf:
            layer.solver_max_iter = 1
        worst, worst_len = 0.0, 0
        for seq_len in (1, 2, 4, 8, 16):
            ids = torch.randint(0, int(hf_cfg.vocab_size), (1, seq_len), device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            with torch.no_grad():
                ref = model(ids.clone(), positions, None)
                ref = (ref[0] if isinstance(ref, tuple) else ref).float()
                got = reloaded(ids.clone()).logits.float()
            delta = float((ref[..., : got.shape[-1]] - got).abs().max())
            if delta > worst:
                worst, worst_len = delta, seq_len
    finally:
        for layer, value in saved:
            layer.looma_cfg.max_iter = value
        for layer, value in saved_hf:
            layer.solver_max_iter = value
    ok = worst < _VERIFY_TOL["wiring"]
    return ok, (
        f"单步（max_iter=1）各长度最差 max|Δ| = {worst:.3e}（seq={worst_len}，"
        f"阈值 {_VERIFY_TOL['wiring']:.1e}）"
    )


def _verify(out: Path, model, hf_cfg, device: str, dtype: str, bridge) -> int:
    """导出后的三道校验：V1 逐张量逐位、V3 融合行、V2a 单步接线；V2 各档只作报数。"""
    ok, detail = _verify_weights(out, model, bridge)
    print(f"[looma·export] V1 逐张量：{detail}")
    if not ok:
        return 1

    ok, detail = _verify_fused(out, model)
    print(f"[looma·export] V3 融合行：{detail}")
    if not ok:
        return 1

    from transformers import AutoModelForCausalLM

    reloaded = (
        AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32)
        .to(device)
        .eval()
    )
    model = model.float()
    ok, detail = _verify_wiring(model, reloaded, hf_cfg, device)
    print(f"[looma·export] V2a 单步接线：{detail}")
    if not ok:
        return 1

    results = {}
    probes = ["fp32"] + ([dtype] if dtype != "fp32" else [])
    for probe_dtype in probes:
        if probe_dtype != "fp32":
            model = model.to(torch.bfloat16 if probe_dtype == "bf16" else torch.float32)
            print(f"[looma·export] V2：mcore 侧回到 {probe_dtype}（与 HF 侧同精度）")
        torch_dtype = torch.bfloat16 if probe_dtype == "bf16" else torch.float32
        reloaded = (
            AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch_dtype)
            .to(device)
            .eval()
        )
        for seq_len in (1, 4, 5, 16):
            ids = torch.randint(0, int(hf_cfg.vocab_size), (1, seq_len), device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            with torch.no_grad():
                reference = model(ids.clone(), positions, None)
                reference = reference[0] if isinstance(reference, tuple) else reference
                got = reloaded(ids.clone()).logits.float()
            ref = reference.float()[..., : got.shape[-1]]
            delta = float((ref - got).abs().max())
            agree = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
            results[(probe_dtype, seq_len)] = (delta, agree)
            print(
                f"[looma·export] V2 logits（算术 {probe_dtype}，seq={seq_len}）："
                f"max|Δ| = {delta:.3e}，argmax 一致 {agree * 100:.1f}%"
            )
        del reloaded

    same_prec = results[(dtype, 1)] if (dtype, 1) in results else results[("fp32", 1)]
    print(
        f"[looma·export] V2 判据：同精度（{dtype}）单 token max|Δ| = {same_prec[0]:.3e}"
        f"（阈值 {_VERIFY_TOL['single_token']:.1e}），argmax 一致 {same_prec[1] * 100:.1f}%"
    )
    if same_prec[0] > _VERIFY_TOL["single_token"] or same_prec[1] < 1.0:
        print("[looma·export] V2 不过：单 token 上两侧就不是同一个函数")
        return 1
    long_delta = results[(dtype, 16)][0]
    if long_delta > 10 * max(same_prec[0], 1e-3):
        print(
            f"[looma·export] 注：参考设置（max_iter={hf_cfg.looma_max_iter}）下长序列的差 "
            f"{long_delta:.3e} 大于单步判据的 {same_prec[0]:.1e}：块的停止是整批的残差统计，"
            "两侧可能停在不同迭代次数。要逐位复现请用单步档或固定次数档"
        )
    print(
        "[looma·export] 校验通过：V1（逐张量）/ V3（融合行）逐位 + V2a（单步接线）"
        "（V2 是报数，见注）"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """导出入口：解析参数、建模型、载入权重并写出 HF 目录，``--verify`` 时接着跑校验。"""
    ap = argparse.ArgumentParser(description="Looma：mcore ckpt → HF 目录")
    ap.add_argument("--ckpt", required=True, help="ckpt 根目录（含 iter_*）或某个 iter 目录")
    ap.add_argument("--out", required=True, help="导出的 HF 目录")
    ap.add_argument("--run-config", default=None, help="训练目录的 config.yaml（几何/spec 兜底）")
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer 目录（默认本配方的 vendored 那份）"
    )
    ap.add_argument("--model-algo", default=None, help="覆盖 spec（默认从 --run-config 读回来）")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--load-iter", type=int, default=None)
    ap.add_argument("--max-position-embeddings", type=int, default=None)
    ap.add_argument("--verify", action="store_true", help="导出后重新加载并比 logits（容差判据）")
    ap.add_argument(
        "--fixed-count",
        action="store_true",
        help="两侧都跑满 max_iter（容差压到不可触及）：验证「长序列的差来自批量统计停止」这个诊断，"
        "也是服务侧「精确可复现」的口径",
    )
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt)
    iter_dir = _resolve_iter_dir(ckpt, args.load_iter)
    out = Path(args.out)
    ckpt_rc = OmegaConf.to_container(OmegaConf.load(iter_dir / "run_config.yaml"), resolve=False)
    run_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.run_config), resolve=False)
        if args.run_config
        else {}
    )
    if not run_cfg:
        print(
            "[looma·export] 没给 --run-config：几何仍从检查点读（权威），但训练时的 spec 无法对账"
            "（连接旋钮按默认值导出）"
        )

    from transformers import AutoTokenizer

    tok_dir = (
        Path(args.tokenizer) if args.tokenizer else RECIPE / "common" / "tokenizer" / "MiniCPM5-2B"
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    hf_cfg, divisible, geometry = _build_hf_config(ckpt_rc, run_cfg, len(tokenizer), args)
    has_real_tokenizer = int(hf_cfg.vocab_size) == int(len(tokenizer))

    from shensi.recipes.paper.looma.common.models.megatron.looma_layer import (
        looma_knobs_from_kwargs,
    )
    from shensi.recipes.paper.looma.stage2_rl.looma_bridge import layer_spec_knobs

    dtype = args.dtype
    if dtype == "auto":
        bf16 = bool(_dig(ckpt_rc, "model.transformer.bf16") or False)
        dtype = "bf16" if bf16 else "fp32"
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

    print(
        f"[looma·export] 几何：vocab={hf_cfg.vocab_size}（补齐 {divisible}）"
        f" layers={hf_cfg.num_hidden_layers} hidden={hf_cfg.hidden_size}"
        f" heads={hf_cfg.num_attention_heads} kv={hf_cfg.num_key_value_heads}"
        f" head_dim={hf_cfg.head_dim} dtype={dtype}"
    )

    _init_distributed(args.device)

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    from shensi.recipes.paper.looma.stage2_rl import looma_bridge  # noqa: F401

    trained = _trained_spec(run_cfg)
    if args.model_algo:
        sys.path.insert(0, str(RECIPE))
        from shensi.recipes.paper.looma import common as looma_common

        module, obj = looma_common.MODEL_ALGOS[args.model_algo]
        trained = [module, obj]
    trained_knobs = None
    if trained:
        import importlib

        trained_spec = getattr(importlib.import_module(trained[0]), trained[1])
        trained_knobs = looma_knobs_from_kwargs(
            dict(trained_spec.params), _FakeCfg(hf_cfg.num_hidden_layers)
        )
        for field in trained_knobs.__dataclass_fields__:
            if field in ("init_std", "residual_dropout", "decay_tau_max"):
                continue
            setattr(hf_cfg, f"looma_{field}", getattr(trained_knobs, field))
        if args.fixed_count:
            hf_cfg.looma_tol = 1e-12
        print(
            f"[looma·export] 旋钮来自训练的 {trained[1]}（max_iter={hf_cfg.looma_max_iter}"
            f" tol={hf_cfg.looma_tol} rank={hf_cfg.looma_rank}"
            f" read_heads={hf_cfg.looma_read_heads} output_route={hf_cfg.looma_output_route}）"
        )
    else:
        print(
            "[looma·export] 没给 spec 来源：旋钮按参考默认值导出（--run-config 或 --model-algo 可指定）"
        )

    bridge = AutoBridge.from_hf_config(hf_cfg)
    provider = bridge.to_megatron_provider(load_weights=False)
    if trained_knobs is not None:
        bridged_knobs = looma_knobs_from_kwargs(
            dict(provider.transformer_layer_spec.params), _FakeCfg(hf_cfg.num_hidden_layers)
        )
        diff = {
            k: (getattr(bridged_knobs, k), getattr(trained_knobs, k))
            for k in trained_knobs.__dataclass_fields__
            if getattr(bridged_knobs, k) != getattr(trained_knobs, k)
        }
        if args.fixed_count:
            diff.pop("tol", None)
        if diff:
            raise SystemExit(f"[looma·export] 规格不一致（桥上 vs 训练）：{diff}")
        print(f"[looma·export] 规格对账：与训练的 {trained[1]} 逐旋钮一致")

    overrides = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": False,
        "seq_length": int(hf_cfg.max_position_embeddings),
        "should_pad_vocab": True,
        "make_vocab_size_divisible_by": divisible,
        "gradient_accumulation_fusion": False,
        "attention_backend": "unfused",
        "persist_layer_norm": False,
        "masked_softmax_fusion": False,
        "apply_rope_fusion": False,
        "bias_activation_fusion": False,
        "bias_dropout_fusion": False,
    }
    if hasattr(provider, "cuda_graph_modules"):
        provider.cuda_graph_modules = None
    if hasattr(provider, "apply_overrides_and_finalize"):
        provider.apply_overrides_and_finalize(dtype=torch_dtype, overrides=overrides)
    else:  # pragma: no cover
        provider.params_dtype = torch_dtype
        provider.bf16 = dtype == "bf16"
        for name, value in overrides.items():
            setattr(provider, name, value)
        provider.finalize()

    print(f"[looma·export] 导出侧连接的旋钮：{provider.transformer_layer_spec.params}")
    _bind_pg_collection(provider)
    model = provider.provide()
    if args.device == "cuda":
        model = model.cuda()
    _check_shapes(model, iter_dir)
    print(
        f"[looma·export] 建好 mcore 模型（{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M 参数，"
        f"device={args.device}）；载入 {iter_dir}"
    )

    from megatron.core import dist_checkpointing

    loaded = dist_checkpointing.load(
        _remap(model.sharded_state_dict(), _NORM_MODEL_TO_CKPT), str(iter_dir)
    )
    loaded = _remap(loaded, _NORM_CKPT_TO_MODEL)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    real_missing = [k for k in missing if not k.endswith("_extra_state")]
    if real_missing:
        raise SystemExit(
            f"[looma·export] 有 {len(real_missing)} 个参数没在检查点里找到（前几个）：{real_missing[:5]}"
        )
    print(
        f"[looma·export] 权重已载入（missing={len(missing)}（应为 _extra_state）"
        f"unexpected={len(unexpected)}）"
    )

    for key, value in layer_spec_knobs(hf_cfg).items():
        if key != "looma_decay_tau_max":
            setattr(hf_cfg, key, value)

    out.mkdir(parents=True, exist_ok=True)
    if has_real_tokenizer:
        copied = [f for f in tok_dir.iterdir() if f.is_file() and f.name in _TOKENIZER_FILES]
        for f in copied:
            shutil.copy2(f, out / f.name)
        print(f"[looma·export] tokenizer：复制自 {tok_dir}（{len(copied)} 个文件）")
    else:
        copied = []
        print(
            f"[looma·export] tokenizer：跳过——{tok_dir} 的词表 {len(tokenizer)} != 导出的词表 "
            f"{hf_cfg.vocab_size}（这个 ckpt 是别的 tokenizer 训的，冒烟档就是这种）"
        )

    try:
        bridge.save_hf_pretrained([model], str(out), show_progress=True)
    except ModuleNotFoundError as exc:
        if "modelopt" not in str(exc):
            raise
        print("[looma·export] 没装 nvidia-modelopt，改用 export_hf_weights + safetensors 直接写")
        from safetensors.torch import save_file

        hf_cfg.save_pretrained(str(out))
        state = {name: tensor for name, tensor in bridge.export_hf_weights([model])}
        save_file(
            {k: v.contiguous() for k, v in state.items()},
            str(out / "model.safetensors"),
            metadata={"format": "pt"},
        )
        print(f"[looma·export] 写了 {len(state)} 个张量到 model.safetensors")

    src_dir = RECIPE / "common" / "models" / "transformers"
    for name in ("configuration_looma.py", "modeling_looma.py"):
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, out / name)
            print(f"[looma·export] 随权重复制的模块：{name}")

    if has_real_tokenizer and not (out / "generation_config.json").is_file():
        (out / "generation_config.json").write_text(
            json.dumps(
                {
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (out / "SHENSI_EXPORT.json").write_text(
        json.dumps(
            {
                "recipe": "looma",
                "source_ckpt": str(iter_dir),
                "run_config": args.run_config,
                "dtype": dtype,
                "geometry": {
                    k: (int(v) if isinstance(v, (int, float)) else v) for k, v in geometry.items()
                },
                "trained_spec": trained,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[looma·export] 完成：{out}")

    if args.verify:
        return _verify(out, model, hf_cfg, args.device, dtype, bridge)
    return 0


class _FakeCfg:
    """``looma_knobs_from_kwargs`` 的配置替身：提供 ``num_layers`` 与初始化口径。"""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.init_method_std = 0.02


if __name__ == "__main__":
    raise SystemExit(main())
