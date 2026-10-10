"""Real tiny GDAR reference workflow on CPU/CUDA/NPU, with synthetic token IDs.

This checks model math and serialization. It does NOT certify Megatron, verl,
vLLM, OpenCompass, distributed training or a production environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from shensi.utils.accelerator import device as select_device


def _optimize(model, optimizer, loss):
    if not torch.isfinite(loss).item():
        raise AssertionError(f"Non-finite loss: {loss}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise AssertionError("Missing or non-finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
    if not torch.isfinite(norm).item() or norm.item() == 0:
        raise AssertionError("Gradient norm must be finite and nonzero")
    optimizer.step()
    return {"loss": loss.item(), "grad_norm": norm.item()}


def _logits(model, tokens):
    return model(input_ids=tokens, use_cache=False).logits.float()


def _lm_loss(model, tokens, assistant_only=False):
    logits = _logits(model, tokens)
    labels = tokens[:, 1:].clone()
    if assistant_only:
        labels[:, : tokens.size(1) // 2] = -100
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels.reshape(-1))


def run(args, report):
    from transformers import __version__ as transformers_version

    from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers import (
        Qwen3GDARConfig,
        Qwen3GDARForCausalLM,
    )

    dev = select_device(None if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    if dev.type != "cpu":
        getattr(torch, dev.type).manual_seed_all(17)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    config = Qwen3GDARConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        attn_res_block_size=1,
        attn_res_gate_rank=8,
        attn_res_q_rank=8,
        attn_res_k_rank=8,
        attn_res_gate_param="deviation",
        attn_res_update="objective",
        attn_res_address="delta",
        attn_res_decay_ladder=8,
        attn_res_decay_positivity="project",
        attn_res_read_heads=2,
        attn_res_read_null=True,
        attn_res_read_whiten=args.whiten,
        attn_res_lambda_clamp=-0.5,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3GDARForCausalLM(config).to(device=dev, dtype=dtype).train()
    if {p.device.type for p in model.parameters()} != {dev.type}:
        raise AssertionError("Model parameters are on the wrong device")
    report.update(
        device=str(dev),
        dtype=str(dtype),
        torch=torch.__version__,
        transformers=transformers_version,
        parameters=sum(p.numel() for p in model.parameters()),
        whitening=args.whiten,
    )
    # Fixed synthetic arithmetic sequences. No external tokenizer, datasets or weights.
    tokens = ((torch.arange(32).reshape(2, 16) * 7 + 3) % 60 + 4).long().to(dev)
    (args.out / "synthetic_tokens.json").write_text(
        json.dumps(tokens.cpu().tolist()), encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, foreach=False)
    stages = report["stages"]
    for stage in ("pretrain", "midtrain", "sft"):
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        records = []
        for _ in range(args.steps):
            records.append(_optimize(model, optimizer, _lm_loss(model, tokens, stage == "sft")))
        if not any(not torch.equal(before[k], p) for k, p in model.named_parameters()):
            raise AssertionError(f"{stage} did not update weights")
        stages[stage] = records
        model.save_pretrained(args.out / stage, safe_serialization=True)
        print(f"[{stage}] loss={records[-1]['loss']:.6f}", flush=True)

    # Compare uninterrupted and restored model+optimizer for the very next step.
    checkpoint = args.out / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location=dev, weights_only=True)
    resumed = Qwen3GDARForCausalLM(copy.deepcopy(config)).to(device=dev, dtype=dtype).train()
    resumed.load_state_dict(saved["model"], strict=True)
    restored_opt = torch.optim.AdamW(resumed.parameters(), lr=5e-4, foreach=False)
    restored_opt.load_state_dict(saved["optimizer"])
    _optimize(model, optimizer, _lm_loss(model, tokens))
    _optimize(resumed, restored_opt, _lm_loss(resumed, tokens))
    resume_delta = max(
        (a - b).float().abs().max().item()
        for a, b in zip(model.parameters(), resumed.parameters(), strict=True)
    )
    if resume_delta > (1e-6 if dtype == torch.float32 else 2e-3):
        raise AssertionError(f"Resume drift: {resume_delta}")
    stages["resume"] = {"max_parameter_delta": resume_delta}
    del resumed, restored_opt, saved

    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    records = []
    for _ in range(args.steps):
        # Four one-token on-policy rollouts from one prompt; group-relative reward.
        prompts = tokens[:1, :8].repeat(4, 1)
        with torch.no_grad():
            old_logits = _logits(model, prompts)[:, -1]
            actions = torch.multinomial(old_logits.softmax(-1), 1)
            old_logp = old_logits.log_softmax(-1).gather(-1, actions).squeeze(-1)
            rewards = actions.squeeze(-1).float() / (config.vocab_size - 1)
            advantage = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
        logp = _logits(model, prompts)[:, -1].log_softmax(-1).gather(-1, actions).squeeze(-1)
        ratio = (logp - old_logp).exp()
        loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
        record = _optimize(model, optimizer, loss)
        record["rewards"] = rewards.cpu().tolist()
        records.append(record)
    stages["rl_reference"] = records
    model.save_pretrained(args.out / "rl", safe_serialization=True)

    records = []
    for _ in range(args.steps):
        # On-policy one-token student samples followed by full-vocabulary reverse KL.
        with torch.no_grad():
            prompt = tokens[:, :8]
            action = torch.multinomial(_logits(model, prompt)[:, -1].softmax(-1), 1)
            rollout = torch.cat((prompt, action), dim=1)
            teacher_logp = _logits(teacher, rollout).log_softmax(-1)
        student_logp = _logits(model, rollout).log_softmax(-1)
        loss = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1).mean()
        records.append(_optimize(model, optimizer, loss))
    stages["opd_reference"] = records
    model.eval()
    with torch.no_grad():
        logits = _logits(model, tokens)
        report["evaluation"] = {
            "loss": _lm_loss(model, tokens).item(),
            "finite_logits": torch.isfinite(logits).all().item(),
        }
        generated = model.generate(
            tokens[:1, :8], max_new_tokens=3, do_sample=False, use_cache=False, pad_token_id=0
        )
        report["evaluation"]["generated_tokens"] = generated.cpu().tolist()
    exported = args.out / "export_hf"
    model.save_pretrained(exported, safe_serialization=True)
    torch.save({"tokens": tokens.cpu(), "logits": logits.cpu()}, args.out / "export_probe.pt")
    # A fresh process must load only the exported remote code, on the SAME device.
    probe = """
import json, sys, torch
from transformers import AutoModelForCausalLM
path, probe, device, dtype = sys.argv[1:]
if device.startswith('npu'):
    import torch_npu
    torch.npu.set_device(0)
torch.set_num_threads(2)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
    local_files_only=True, attn_implementation='eager').to(device=device, dtype=getattr(torch, dtype)).eval()
data = torch.load(probe, weights_only=True)
with torch.no_grad():
    actual = model(data['tokens'].to(device), use_cache=False).logits.float().cpu()
delta = (actual - data['logits']).abs().max().item()
torch.testing.assert_close(actual, data['logits'], atol=2e-3 if dtype=='bfloat16' else 1e-5,
    rtol=2e-3 if dtype=='bfloat16' else 1e-5)
print('EXPORT_RESULT=' + json.dumps({'max_logit_delta': delta, 'device': device}))
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(exported),
            str(args.out / "export_probe.pt"),
            str(dev),
            args.dtype,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    (args.out / "export_reload.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"Fresh-process export reload failed: {proc.stderr[-3000:]}")
    stages["export_reload"] = json.loads(
        next(
            line.removeprefix("EXPORT_RESULT=")
            for line in proc.stdout.splitlines()
            if line.startswith("EXPORT_RESULT=")
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "npu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--whiten", choices=("full", "per_head", "diag", "off"), default="full")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": False,
        "scope": "single-device HF reference model workflow",
        "not_validated": [
            "Megatron training",
            "verl RL",
            "vLLM serving",
            "OpenCompass",
            "distributed training",
        ],
        "requested_device": args.device,
        "stages": {},
        "platform": platform.platform(),
    }
    start = time.monotonic()
    try:
        run(args, report)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
