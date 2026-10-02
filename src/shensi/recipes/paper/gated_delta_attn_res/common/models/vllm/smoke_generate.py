"""vLLM 生成冒烟：逐变体跑小样本并核对输出。"""


from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from ._paths import RECIPE, SRC, ensure_src_on_path

ensure_src_on_path()

from .register_model import describe_resolution, register_all  # noqa: E402
from .tiny_checkpoint import DEFAULT_TOKENIZER, build  # noqa: E402
from .variants import BY_KEY, SHAPES, VARIANTS, Variant  # noqa: E402

DEFAULT_PROMPT = "The capital of France is"
DEFAULT_TOKENS = 16





def hf_greedy(ckpt: str, prompt: str, max_new_tokens: int, dtype: str, device: str = "cpu") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]




    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch_dtype, trust_remote_code=True
    ).eval()



    model = model.to(device)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)


    out = ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(out, use_cache=False).logits[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    gen = out[0, ids.shape[1] :].tolist()
    return {
        "prompt_ids": ids[0].tolist(),
        "generated_ids": gen,
        "text": tokenizer.decode(gen, skip_special_tokens=False),
        "class": type(model).__name__,
        "device": device,
    }





def _ensure_venv_bin_on_path() -> list[str]:
    import shutil

    bindir = Path(sys.executable).resolve().parent
    if bindir.is_dir() and str(bindir) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    return [p for p in ("ninja",) if shutil.which(p)] if shutil.which("ninja") else []


def engine_generate(
    ckpt: str,
    prompt: str,
    max_new_tokens: int,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    model_impl: str | None,
    kv_cache_memory_bytes: int | None = None,
) -> dict:
    _ensure_venv_bin_on_path()
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=ckpt,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    if model_impl:
        kwargs["model_impl"] = model_impl
    if kv_cache_memory_bytes:





        kwargs["kv_cache_memory_bytes"] = kv_cache_memory_bytes
    llm = LLM(**kwargs)

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
    outputs = llm.generate([prompt], sp)
    o = outputs[0]
    gen_ids = list(o.outputs[0].token_ids)
    return {
        "prompt_ids": list(o.prompt_token_ids),
        "generated_ids": gen_ids,
        "text": o.outputs[0].text,
        "engine_class": _engine_model_class(llm),
        "model_impl": getattr(llm.llm_engine.model_config, "model_impl", None),
    }


def _engine_model_class(llm) -> str:
    for path in (
        ("llm_engine", "engine_core", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model", "model"),
    ):
        obj = llm
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return f"{type(obj).__name__} ({'.'.join(path)})"
        except AttributeError:
            continue
    return "unavailable"





def run_variant(
    variant: Variant,
    *,
    ckpt_root: Path,
    prompt: str,
    tokens: int,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    rebuild: bool,
    compare_hf: bool,
    model_impl: str | None,
    tokenizer_dir: str | Path | None,
    kv_cache_memory_bytes: int | None = None,
    shape: str = "tiny",
    hf_device: str = "cpu",
) -> dict:
    from transformers import AutoTokenizer

    res: dict = {
        "variant": variant.key,
        "architecture": variant.architecture,
        "shape": shape,
        "ok": False,
    }
    ckpt = ckpt_root / f"{variant.key}-{shape}"
    print(f"\n===== {variant.key}  ({variant.architecture}) =====")

    if rebuild or not (ckpt / "model.safetensors").exists():
        rep = build(variant.key, ckpt, shape=shape, tokenizer_dir=tokenizer_dir, overwrite=True)
        print(
            f"  built {shape} checkpoint: {rep['parameters']:,} params, vocab={rep['vocab_size']}, "
            f"files={rep['remote_code_files']}"
        )
    res["ckpt"] = str(ckpt)
    res["params"] = sum(1 for _ in (ckpt).glob("*.safetensors"))

    print(f"  registry before: {describe_resolution([variant.architecture])[variant.architecture]}")
    reg = register_all()
    res["registered"] = variant.architecture in reg["registered"] + reg["already"]
    print(f"  registry after : {describe_resolution([variant.architecture])[variant.architecture]}")

    try:
        eng = engine_generate(
            str(ckpt),
            prompt,
            tokens,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_impl=model_impl,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
        )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-2500:]
        print(f"  ENGINE FAILED: {res['error']}")
        return res

    res["ok"] = True
    res["engine"] = eng
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    print(f"  engine model class : {eng['engine_class']}")
    print(f"  model_impl         : {eng['model_impl']}")
    print(f"  prompt ids         : {eng['prompt_ids']}")
    print(f"  generated ids      : {eng['generated_ids']}")
    print(f"  generated text     : {eng['text']!r}")
    print(f"  decoded (ids->text): {tok.decode(eng['generated_ids'], skip_special_tokens=False)!r}")

    if compare_hf:
        try:
            ref = hf_greedy(str(ckpt), prompt, tokens, dtype, hf_device)
            lcp = 0
            for a, b in zip(eng["generated_ids"], ref["generated_ids"]):
                if a != b:
                    break
                lcp += 1
            res["hf_reference"] = ref
            res["longest_common_prefix"] = lcp
            print(f"  HF reference ids   : {ref['generated_ids']}  ({ref['class']})")
            print(f"  identical prefix   : {lcp}/{tokens}")
        except Exception as exc:
            res["hf_reference_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  HF reference FAILED: {res['hf_reference_error']}")
    return res


def main(argv: list[str] | None = None) -> int:
    """生成冒烟入口：逐变体跑小样本并核对输出。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="gdar", choices=sorted(BY_KEY))
    ap.add_argument("--all", action="store_true", help="run every variant")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--ckpt-root", default="/tmp/rollout_smoke")
    ap.add_argument(
        "--shape",
        default="tiny",
        choices=sorted(SHAPES),
        help="backbone geometry of the checkpoint: tiny (2x64) or 0.6b (28x1024); "
        "the checkpoint lives at <ckpt-root>/<variant>-<shape>",
    )
    ap.add_argument(
        "--hf-device",
        default="cpu",
        help="device for the plain-transformers reference (cpu is enough for tiny; "
        "use cuda for a 0.6B-shaped checkpoint)",
    )
    ap.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))





    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    ap.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=None,
        help="pin the KV cache size instead of letting vLLM profile free memory "
        "(needed when the GPU is shared: profiling asserts if another process "
        "releases memory between its two measurements)",
    )
    ap.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="allow CUDA graphs (our connections have data-dependent shapes; eager is safer)",
    )
    ap.add_argument(
        "--model-impl",
        default=None,
        help="force a vLLM model_impl; default is the registered class",
    )
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-compare-hf", action="store_true")
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument(
        "--fan-out",
        action="store_true",
        help="run each variant in its own subprocess (used by --all; see below)",
    )
    args = ap.parse_args(argv)





    if args.all and not args.fan_out:
        return _fan_out(args)

    import vllm

    print(
        f"vllm {vllm.__version__} | python {sys.version.split()[0]} | "
        f"ROLLOUT_PLUGIN_AUTOLOAD={os.environ.get('ROLLOUT_PLUGIN_AUTOLOAD', '<unset>')}"
    )
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    variants = list(VARIANTS) if args.all else [BY_KEY[args.variant]]
    ckpt_root = Path(args.ckpt_root)

    results = []
    for v in variants:
        results.append(
            run_variant(
                v,
                ckpt_root=ckpt_root,
                prompt=args.prompt,
                tokens=args.tokens,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=not args.no_enforce_eager,
                rebuild=args.rebuild,
                compare_hf=not args.no_compare_hf,
                model_impl=args.model_impl,
                tokenizer_dir=args.tokenizer_dir,
                kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                shape=args.shape,
                hf_device=args.hf_device,
            )
        )

    ok = [r for r in results if r.get("ok")]
    print(f"\n===== summary: {len(ok)}/{len(results)} variants generated =====")
    for r in results:
        note = ""
        if r.get("ok"):
            note = (
                f"lcp={r.get('longest_common_prefix')}/{args.tokens}"
                if "longest_common_prefix" in r
                else f"impl={r['engine']['model_impl']}"
            )
        else:
            note = r.get("error", "")
        print(f"  {r['variant']:12s} {'OK ' if r.get('ok') else 'FAIL'}  {note}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "vllm": vllm.__version__,
                    "python": sys.version.split()[0],
                    "args": vars(args),
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"\nreport written to {args.json}")

    return 0 if len(ok) == len(results) else 1


def _fan_out(args) -> int:
    import subprocess

    out_root = Path(args.ckpt_root)
    out_root.mkdir(parents=True, exist_ok=True)
    per: list[tuple[str, Path]] = []
    for v in VARIANTS:
        j = out_root / f"{v.key}.result.json"
        cmd = [
            sys.executable,
            "-m",
            "shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.smoke_generate",
            "--variant",
            v.key,
            "--tokens",
            str(args.tokens),
            "--prompt",
            args.prompt,
            "--ckpt-root",
            args.ckpt_root,
            "--tokenizer-dir",
            args.tokenizer_dir,
            "--shape",
            args.shape,
            "--hf-device",
            args.hf_device,
            "--dtype",
            args.dtype,
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--json",
            str(j),
            "--fan-out",
        ]
        if args.no_enforce_eager:
            cmd.append("--no-enforce-eager")
        if args.rebuild:
            cmd.append("--rebuild")
        if args.no_compare_hf:
            cmd.append("--no-compare-hf")
        if args.model_impl:
            cmd += ["--model-impl", args.model_impl]
        if args.kv_cache_memory_bytes:
            cmd += ["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)]

        print(f"\n########## {v.key} ##########", flush=True)


        proc = subprocess.run(cmd, env=os.environ.copy())
        print(f"########## {v.key} exit={proc.returncode} ##########", flush=True)
        per.append((v.key, j))

    results, ok = [], 0
    for key, j in per:
        try:
            r = json.loads(j.read_text())["results"][0]
        except Exception as exc:
            r = {"variant": key, "ok": False, "error": f"no result file ({exc})"}
        results.append(r)
        ok += bool(r.get("ok"))
        note = (
            f"lcp={r.get('longest_common_prefix')}/{args.tokens}"
            if "longest_common_prefix" in r
            else r.get("error", "")
        )
        print(f"  {key:12s} {'OK ' if r.get('ok') else 'FAIL'}  {note}")

    print(f"\n===== summary: {ok}/{len(results)} variants generated =====")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "args": {k: v for k, v in vars(args).items()},
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"report written to {args.json}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
