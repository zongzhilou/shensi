"""转换审计：逐张量核对转换结果与参考。"""


from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CODE = Path(__file__).resolve().parents[2]

import torch  # noqa: E402

from .convert import AttnLayout, compare, hf_to_mcore, mcore_to_hf  # noqa: E402
from .profiles import PROFILES, Profile, profile  # noqa: E402
from .tables import SynthesisPolicy, build_table  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []

GAPS: list[str] = []

SKIPPED: list[str] = []

OBSERVED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}", flush=True)
    return ok


def check_gap(
    name: str, offenders: list[str], expected: tuple[str, ...], ok_detail: str = "0"
) -> bool:
    if not offenders:
        return check(name, True, ok_detail)
    unexplained = [item for item in offenders if not any(marker in item for marker in expected)]
    if unexplained:
        return check(name, False, f"{len(unexplained)} undeclared of {len(offenders)}: {unexplained[:3]}")
    OBSERVED.extend(marker for marker in expected if any(marker in item for item in offenders))
    GAPS.append(f"{name}: {len(offenders)} declared-by-design tensor(s): {', '.join(offenders[:4])}")
    print(f"  [GAP]  {name:<46} {len(offenders)} declared: {offenders[:3]}", flush=True)
    return True


def note(text: str) -> None:
    print(f"  [INFO] {text}", flush=True)


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


def check_synthesized(synthesized: dict[str, float], declared: tuple[str, ...]) -> bool:
    undeclared = [
        name for name in synthesized if not any(marker in name for marker in declared)
    ]
    unfired = [
        marker for marker in declared if not any(marker in name for name in synthesized)
    ]
    detail = f"{len(synthesized)} row(s)"
    if undeclared or unfired:
        detail = (
            f"undeclared {undeclared[:3]}"
            if undeclared
            else f"declared but never synthesised: {unfired}"
        )
    return check("synthesized rows are all declared", not undeclared and not unfired, detail)







def init_distributed(port: int = 29541) -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu, tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1, init_method="env://")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def make_hf_reference(out: Path, hf_python: Path, only: list[str] | None) -> dict:
    command = [str(hf_python), "-m", "shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.hf_reference", "--out", str(out)]
    if only:
        command += ["--only", *only]
    env = dict(os.environ, PYTHONPATH=str(CODE))
    print(f"  $ {' '.join(command)}")
    result = subprocess.run(command, cwd=str(CODE), env=env, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
    index_file = out / "index.json"
    if not index_file.exists():
        raise RuntimeError(
            f"the HF reference could not be produced ({hf_python} returned {result.returncode}). "
            f"Pass --hf-python if the transformers-5 environment is somewhere else."
        )
    return json.loads(index_file.read_text())







def build_mcore(profile: Profile, root: Path, *, cpu: bool = True):
    import shensi.recipes.paper.gated_delta_attn_res.stage2_rl as verl_plugin
    from transformers import AutoConfig
    from verl.models.mcore.mbridge import AutoBridge

    directory = verl_plugin.materialize_config_dir(
        verl_plugin.VARIANTS[profile.variant], root / profile.name, **profile.config_kwargs()
    )
    hf_config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    bridge = AutoBridge.from_config(hf_config, dtype=torch.float32)
    bridge.set_extra_args(use_cpu_initialization=cpu)
    model = bridge._model_provider([])(pre_process=True, post_process=True)
    if not cpu:
        model.cuda(torch.cuda.current_device())
    return bridge, model, hf_config


def slim(state: dict) -> dict:
    return {k: v for k, v in state.items() if v is not None and not k.endswith("_extra_state")}


def audit_skipped_build(profile: Profile, reference_path: Path, root: Path) -> bool:
    print(f"\n--- {profile.name} ({profile.variant}): {profile.about}")
    blob = torch.load(reference_path, map_location="cpu", weights_only=False)
    note(f"HF side builds fine: {len(blob['state_dict'])} tensors")




    from transformers import AutoConfig

    import shensi.recipes.paper.gated_delta_attn_res.stage2_rl as verl_plugin

    directory = verl_plugin.materialize_config_dir(
        verl_plugin.VARIANTS[profile.variant], root / profile.name, **profile.config_kwargs()
    )
    hf_config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    table = build_table(profile.variant, hf_config, int(hf_config.num_hidden_layers))
    for entry in table.unsupported:
        note(f"the table declares: {entry}")

    offenders = [f"table: {entry}" for entry in table.unsupported]
    try:
        build_mcore(profile, root)
    except Exception as exc:  # noqa: BLE001 - failing is the expected outcome here
        offenders.append(f"build: {type(exc).__name__}: {exc}")
    else:
        offenders.append("build: SUCCEEDED -- the documented gap no longer exists, drop skip_build")
    return check_gap("the gap is declared and the build refuses", offenders, profile.expect_unsupported)


def audit_profile(profile: Profile, reference_path: Path, root: Path, policy: SynthesisPolicy) -> bool:
    print(f"\n--- {profile.name} ({profile.variant}): {profile.about}")
    blob = torch.load(reference_path, map_location="cpu", weights_only=False)
    hf_state = blob["state_dict"]

    bridge, model, hf_config = build_mcore(profile, root)



    model_state = slim(model.state_dict())
    model_keys = list(model_state)
    table = build_table(profile.variant, hf_config, int(hf_config.num_hidden_layers), policy=policy)
    layout = AttnLayout.from_config(hf_config)
    note(
        f"{profile.variant}: HF {len(hf_state)} tensors, Megatron {len(model_keys)} tensors, "
        f"table {len(table.pairs)} rows"
    )
    if table.unsupported:
        for entry in table.unsupported:
            note(f"unsupported by the Megatron port: {entry}")



        for marker in profile.expect_unsupported:
            if any(marker in entry for entry in table.unsupported):
                OBSERVED.append(marker)

    ok = True

    converted, report = hf_to_mcore(
        hf_state,
        table,
        layout=layout,
        padded_vocab_size=int(model_state["embedding.word_embeddings.weight"].shape[0]),
        policy=policy,
    )
    produced_keys = set(converted)
    expected_keys = set(model_keys)
    only_model = sorted(expected_keys - produced_keys)
    only_table = sorted(produced_keys - expected_keys)
    ok &= check(
        "every Megatron tensor has a source",
        not only_model and not only_table,
        f"table->model {len(only_table)} missing, model->table {len(only_model)} extra"
        + (f"; missing={only_model[:4]}" if only_model else "")
        + (f"; extra={only_table[:4]}" if only_table else ""),
    )
    shape_bad = [
        k for k in sorted(expected_keys & produced_keys)
        if tuple(converted[k].shape) != tuple(model_state[k].shape)
    ]
    ok &= check("every converted tensor has the model's shape", not shape_bad, f"{len(shape_bad)} wrong {shape_bad[:4]}")

    try:
        model.load_state_dict(converted, strict=True)
        ok &= check("strict load_state_dict accepts it", True, f"{len(converted)} tensors")
    except RuntimeError as exc:
        ok &= check("strict load_state_dict accepts it", False, str(exc)[:200])

    unmapped_hf = report.unmapped_source
    expected = profile.expect_unsupported
    ok &= check_gap("unmapped HF tensors (this direction)", unmapped_hf, expected)




    ok &= check(
        "every table row found its HF source",
        not report.missing_source,
        f"{len(report.missing_source)} row(s) produced nothing: {report.missing_source[:4]}",
    )
    note(
        f"synthesized {len(report.synthesized)}: "
        + ", ".join(f"{k.split('.', 2)[-1]}={v}" for k, v in list(report.synthesized.items())[:4])
        + (" ..." if len(report.synthesized) > 4 else "")
    )
    ok &= check_synthesized(report.synthesized, profile.expect_synthesized)


    back, back_report = mcore_to_hf(slim(model_state), table, layout=layout, vocab_size=int(hf_config.vocab_size), policy=policy)
    bad, details = compare(back, hf_state, label="round-trip")


    offenders = [line.split(":", 1)[0] for line in details]
    ok &= check_gap(
        "HF -> mcore -> HF is bit-exact",
        offenders if bad else [],
        expected,
        ok_detail=f"{len(back)}/{len(hf_state)} tensors, torch.equal everywhere",
    )
    ok &= check(
        "unmapped Megatron tensors (this direction)",
        not back_report.unmapped_source,
        f"{len(back_report.unmapped_source)}: {back_report.unmapped_source[:4]}",
    )
    ok &= check(
        "every table row present in the Megatron state_dict",
        not back_report.missing_source,
        f"{len(back_report.missing_source)} row(s) absent: {back_report.missing_source[:4]}",
    )
    if back_report.dropped:
        note(f"dropped on the way back (Megatron-only): {len(back_report.dropped)}")


    ok &= audit_mbridge_path(bridge, model, hf_state, hf_config, profile, root)


    ok &= audit_forward(profile, blob, converted, root)





    unfired = [marker for marker in expected if marker not in OBSERVED]
    ok &= check(
        "every declared expectation fired",
        not unfired,
        f"declared but never seen: {unfired}" if unfired else f"{len(expected)} declared",
    )
    return ok







FORWARD_ATOL = 1e-3


def audit_forward(profile: Profile, blob: dict, converted: dict, root: Path) -> bool:
    if "logits" not in blob:
        return check(
            "the converted model computes the HF logits", False, "the HF reference has no forward probe"
        )
    if not torch.cuda.is_available():
        note("no CUDA: mcore's RoPE allocates on `torch.cuda.current_device()`, so the forward "
             "probe cannot run; the tensor checks above are unaffected")
        return True

    _, model, _ = build_mcore(profile, root, cpu=False)
    model.load_state_dict(converted, strict=True)
    model.eval()
    input_ids = blob["input_ids"]
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        got = model(
            input_ids=input_ids.to("cuda"),
            position_ids=position_ids.to("cuda"),
            attention_mask=None,
        ).float()
    reference = blob["logits"]
    if tuple(got.shape) != tuple(reference.shape):
        return check(
            "the converted model computes the HF logits",
            False,
            f"shape {tuple(got.shape)} != {tuple(reference.shape)}",
        )
    got = got.cpu()
    diff = (got - reference).abs()
    worst = diff.max().item()
    scale = reference.abs().max().item()
    del model
    inside = worst <= FORWARD_ATOL
    if profile.expect_forward_gap is not None:




        if inside:
            return check(
                "the converted model computes the HF logits",
                False,
                f"declared a forward gap but the logits match (max|delta| {worst:.2e}) -- "
                f"drop the declaration: {profile.expect_forward_gap}",
            )
        GAPS.append(f"forward gap (declared): {profile.expect_forward_gap}")
        print(
            f"  [GAP]  {'the converted model computes the HF logits':<46} "
            f"max|delta| {worst:.2e} -- declared, see the reason below",
            flush=True,
        )
        print(f"         reason: {profile.expect_forward_gap}", flush=True)
        return True
    return check(
        "the converted model computes the HF logits",
        inside,
        f"max|delta| {worst:.2e} (atol {FORWARD_ATOL:g}), mean|delta| {diff.mean().item():.2e}, "
        f"max|logit| {scale:.2f}",
    )


def audit_mbridge_path(bridge, model, hf_state: dict, hf_config, profile: Profile, root: Path) -> bool:
    from safetensors.torch import save_file

    directory = root / f"{profile.name}_hf"
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in hf_state.items()}, str(directory / "model.safetensors"))





    import json
    import shutil

    (directory / "config.json").write_text(json.dumps(hf_config.to_dict(), indent=2, sort_keys=True))
    for extra in ("configuration_qwen3_%s.py" % profile.variant, "modeling_qwen3_%s.py" % profile.variant):
        source = CODE / "models" / extra
        if source.exists():
            shutil.copy2(source, directory / extra)

    fresh = bridge._model_provider([])(pre_process=True, post_process=True)
    try:
        bridge.load_weights([fresh], str(directory))
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        return check(
            "bridge.load_weights (the verl path)",
            False,
            f"{type(exc).__name__}: {str(exc)[:160]}",
        )
    ok = check(
        "bridge.load_weights (the verl path)",
        True,
        f"loaded {len(hf_state)} HF tensors into {len(slim(fresh.state_dict()))} Megatron tensors",
    )

    exported = dict(bridge.export_weights([fresh]))
    bad, details = compare(exported, hf_state, label="bridge export")
    ok &= check_gap(
        "bridge.export_weights reproduces the checkpoint",
        [line.split(":", 1)[0] for line in details] if bad else [],
        profile.expect_unsupported,
        ok_detail=f"{len(exported)} tensors, bit-exact",
    )



    direct = slim(fresh.state_dict())
    differing = [
        k for k, v in direct.items()
        if k in model.state_dict() and not torch.equal(v, model.state_dict()[k])
    ]
    ok &= check(
        "the two load paths agree tensor-for-tensor",
        not differing,
        f"{len(differing)} differing {differing[:4]}",
    )
    return ok







def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HF <-> Megatron conversion audit")
    parser.add_argument("--only", nargs="*", default=None, help="profile names (default: all)")
    parser.add_argument("--hf-python", type=Path, default=CODE / ".venv" / "bin" / "python")
    parser.add_argument("--reference", type=Path, default=None, help="reuse a reference dir instead of re-dumping")
    parser.add_argument("--work", type=Path, default=None, help="scratch dir (default: a temp dir)")
    parser.add_argument("--read-scale", type=float, default=1.0, help="value for the AR/DAR Megatron-only gate")
    parser.add_argument("--json", type=Path, default=None, help="write the per-profile report here")
    args = parser.parse_args(argv)

    section("environment")
    import megatron.core

    import shensi.recipes.paper.gated_delta_attn_res.stage2_rl as verl_plugin

    init_distributed()
    print(f"  python        {sys.version.split()[0]} ({sys.executable})")
    print(f"  torch         {torch.__version__}")
    print(f"  megatron.core {megatron.core.__version__}  ({Path(megatron.core.__file__).parent})")
    report = verl_plugin.install()
    print(f"  plugin        {verl_plugin.__version__}; specs: {list(report['specs'])}")
    ok = check("all seven model types registered", len(report["specs"]) == 7, f"{len(report['specs'])}/7")
    ok &= check("hf reference interpreter exists", args.hf_python.exists(), str(args.hf_python))

    names = args.only or [p.name for p in PROFILES]
    profiles = [profile(n) for n in names]

    work = args.work or Path(tempfile.mkdtemp(prefix="verl_convert_audit_"))
    work.mkdir(parents=True, exist_ok=True)
    reference_dir = args.reference or (work / "hf_reference")
    section("HF reference checkpoints (transformers 5 interpreter)")
    if args.reference is not None and (args.reference / "index.json").exists():
        index = json.loads((args.reference / "index.json").read_text())
        print(f"  reusing {args.reference}")
    else:
        index = make_hf_reference(reference_dir, args.hf_python, names)

    section("per-profile conversion")
    policy = SynthesisPolicy(ar_dar_read_scale=args.read_scale)
    print(f"  synthesis policy: ar_dar_read_scale={policy.ar_dar_read_scale}, zero={policy.zero}")
    per_profile: dict[str, dict] = {}
    for prof in profiles:
        entry = index.get(prof.name)
        if not entry or not entry.get("ok"):
            reason = (entry or {}).get("error", "no HF reference")
            SKIPPED.append(f"{prof.name}: HF model could not be built: {reason}")
            print(f"\n--- {prof.name}: [SKIP] HF model could not be built: {str(reason)[:200]}")
            continue
        checks_before, gaps_before = len(RESULTS), len(GAPS)
        OBSERVED.clear()
        try:
            if prof.skip_build is not None:
                passed = audit_skipped_build(prof, reference_dir / entry["file"], work)
            else:
                passed = audit_profile(prof, reference_dir / entry["file"], work, policy)
        except Exception as exc:  # noqa: BLE001 - a build failure is a finding, not a crash
            SKIPPED.append(f"{prof.name}: {type(exc).__name__}: {exc}")
            print(f"\n--- {prof.name}: [SKIP] {type(exc).__name__}: {str(exc)[:400]}")
            continue
        mine = RESULTS[checks_before:]
        per_profile[prof.name] = {
            "ok": passed,
            "variant": prof.variant,
            "checks": len(mine),
            "failed": [name for name, ok_, _ in mine if not ok_],
            "gaps": [gap for gap in GAPS[gaps_before:]],
        }


    section("summary")
    passed = sum(1 for _, ok_, _ in RESULTS if ok_)
    failed = [name for name, ok_, _ in RESULTS if not ok_]
    print(f"  PASS {passed}/{len(RESULTS)}   FAIL {len(failed)}   declared-GAP lines {len(GAPS)}")
    for name in failed:
        print(f"    FAILED: {name}")
    print("\n  per profile:")
    for name, entry in per_profile.items():
        verdict = "PASS" if entry["ok"] else "FAIL"
        extra = f", {len(entry['gaps'])} declared gap line(s)" if entry["gaps"] else ""
        print(f"    {name:<20} {entry['variant']:<12} {verdict}  {entry['checks']} checks{extra}")
        for line in entry["gaps"]:
            print(f"        GAP  {line}")
    if GAPS:
        print(f"\n  {len(GAPS)} declared-by-design gap line(s) in total (not failures, not passes):")
    if SKIPPED:
        print(f"\n  {len(SKIPPED)} profile(s) not audited at all (reported, not silently skipped):")
        for gap in SKIPPED:
            print(f"    - {gap}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in RESULTS],
                    "profiles": per_profile,
                    "gaps": GAPS,
                    "skipped": SKIPPED,
                    "policy": {"ar_dar_read_scale": policy.ar_dar_read_scale},
                },
                indent=2,
            )
        )
        print(f"\n  report written to {args.json}")
    print(f"  scratch dir: {work}")


    return 0 if not failed and not SKIPPED else 1


if __name__ == "__main__":
    raise SystemExit(main())
