#!/usr/bin/env python3
"""GRPO（Group Relative Policy Optimization）紧凑实现 —— Specialized RL 的训练循环。

论文 §2.5.2 跟随 DSV4F 用 GRPO；本配方的模型是自定义 VL 结构（DSV4F+ViT+projector），
进不了 verl 的 Megatron actor，所以 RL 在配方内自实现：采样 N 条/题 → compute_score 打分 →
组内归一 advantage → 裁剪比率的策略梯度。奖励 = stage2_rl/reward.py（Format+Quality+Accuracy）。

论文的两个关键设计照搬：
  - **不监督思考过程中的原语**（冷启动数据里已验证过）：奖励只看输出文本与最终答案，
    因此 RL 数据只需要（图、问题、spec/答案）；
  - 数据按难度分层（stage2_rl/data_prep.py 先 rollout 分层，只喂 Normal-Level）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shensi import runtime  # noqa: F401,E402
from shensi.recipes.shensi_vl.common import processors as proc_mod  # noqa: E402
from shensi.recipes.shensi_vl.common import render  # noqa: E402
from shensi.recipes.shensi_vl.common.train import train_loop  # noqa: E402
from shensi.recipes.shensi_vl.stage2_rl import reward as R  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        yield from fh


def encode_prompt(tok, iproc, row: dict, *, trigger: bool = True) -> dict:
    """（图, 问题）→ prompt 张量（含图像占位），结尾锚在 <｜Assistant｜><think>。"""
    image = None
    counts = []
    if row.get("image"):
        from PIL import Image

        image = Image.open(row["image"]).convert("RGB")
        counts = [proc_mod.image_token_count(iproc, *image.size, tok)]
    user = {
        "role": "user",
        "content": render.build_user_content(render.finalize_text(row["question"]), 1 if image else 0, trigger=trigger),
    }
    enc = render.encode_sample(tok, [user], counts)
    ids = enc["input_ids"] + tok("<｜Assistant｜>", add_special_tokens=False)["input_ids"]
    ids += tok("<think>", add_special_tokens=False)["input_ids"]
    pos = enc["image_positions"]
    return {
        "prompt_ids": ids,
        "image_positions": pos,
        "pixel_values": proc_mod.preprocess_images(iproc, [image]) if image is not None else None,
    }


def batch_prompts(tok, batch: list[dict]) -> dict:
    """Prompt 拼 batch（右 padding）。"""
    pad = train_loop.pad_token_id(tok)
    L = max(len(b["prompt_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    pos = torch.zeros((len(batch), L), dtype=torch.bool)
    for i, b in enumerate(batch):
        m = len(b["prompt_ids"])
        ids[i, :m] = torch.tensor(b["prompt_ids"])
        attn[i, :m] = 1
        if b["image_positions"]:
            pos[i, torch.tensor(b["image_positions"], dtype=torch.long)] = True
    pixels = None
    if batch[0]["pixel_values"] is not None:
        pixels = torch.cat([b["pixel_values"] for b in batch], dim=0)
    return {"input_ids": ids, "attention_mask": attn, "image_positions": pos, "pixel_values": pixels}


def to_device(pb: dict, device) -> dict:
    """张量搬设备（pixel_values 可能是 None：无图样本）。"""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in pb.items()}


def response_logps(model, full_ids, prompt_len, attention_mask, pixel_values, image_positions):
    """响应段每 token 的 log p（前向一次，gather 目标位）。"""
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    labels[attention_mask == 0] = -100
    attn_pad = torch.cat(
        [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
    )[:, : full_ids.shape[1]]
    loss, logits = model(
        input_ids=full_ids, attention_mask=attn_pad,
        pixel_values=pixel_values, image_positions=image_positions, labels=labels,
    )
    # 自己再取响应位的 log p（loss 是均值；PG 需要 per-token）
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = full_ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (labels[:, 1:] != -100).float()
    return tok_logp, mask, loss


def run(cfg: dict) -> int:
    from shensi.recipes.shensi_vl.common import vl_tokens

    t = cfg["train"]
    tok = train_loop.build_tokenizer(cfg)
    img_id = vl_tokens.token_ids(tok)[vl_tokens.IMAGE_TOKEN]
    model = train_loop.make_model(cfg, img_id)
    iproc = proc_mod.load_image_processor(t["model"]["processor_path"])
    rows = [json.loads(line) for line in _read_jsonl(t["data"]["train_jsonl"]) if line.strip()]
    if not rows:
        raise SystemExit("[grpo] 任务池为空")
    device = model.device
    o = t["optim"]
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(o["lr"]), betas=tuple(o.get("betas", (0.9, 0.95))), weight_decay=float(o.get("weight_decay", 0.0)),
    )
    n_group = int(t["rl"].get("group_n", 8))
    prompts_per_step = int(t["rl"].get("prompts_per_step", 8))
    clip_lo = float(t["rl"].get("clip_ratio_low", 0.2))
    clip_hi = float(t["rl"].get("clip_ratio_high", 0.28))
    max_new = int(t["rl"].get("max_new_tokens", 1024))
    temperature = float(t["rl"].get("temperature", 1.0))
    total = int(t["batch"]["train_iters"])
    rng = __import__("random").Random(int(cfg["experiment"].get("seed", 42)))
    ckpt_dir = Path(t["model"].get("save") or cfg["experiment"]["exp_dir"])
    t0 = time.time()
    model.train()
    for step in range(1, total + 1):
        batch_rows = rng.sample(rows, min(prompts_per_step, len(rows)))
        prompts = [encode_prompt(tok, iproc, r) for r in batch_rows]
        pb = to_device(batch_prompts(tok, prompts), device)
        if pb.get("pixel_values") is None:
            pb["pixel_values"] = None
        B, L = pb["input_ids"].shape
        # 展开 n_group 份做组采样
        expand = lambda x, n: x.repeat_interleave(n, dim=0)  # noqa: E731
        gen_ids = model.generate(
            expand(pb["input_ids"], n_group), expand(pb["attention_mask"], n_group),
            expand(pb["pixel_values"], n_group) if pb["pixel_values"] is not None else None,
            expand(pb["image_positions"], n_group),
            max_new_tokens=max_new, temperature=temperature,
        )
        texts = tok.batch_decode(gen_ids, skip_special_tokens=False)
        # 打分 + 组内归一 advantage
        rewards = []
        for gi, row in enumerate(batch_rows):
            gt = row.get("spec") or row.get("ground_truth") or {}
            sols = texts[gi * n_group : (gi + 1) * n_group]
            rs = [R.compute_score(row.get("task", "count"), s, gt, {}) for s in sols]
            mean = sum(rs) / n_group
            std = (sum((x - mean) ** 2 for x in rs) / n_group) ** 0.5 or 1.0
            rewards.append([(x - mean) / std for x in rs])
        adv = torch.tensor([a for grp in rewards for a in grp], device=device, dtype=torch.float32)
        # 策略梯度（裁剪比率；单 inner epoch）
        full = torch.cat(
            [expand(pb["input_ids"], n_group), gen_ids.to(pb["input_ids"].dtype)], dim=1
        )
        full_attn = torch.cat(
            [expand(pb["attention_mask"], n_group), torch.ones_like(gen_ids)], dim=1
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tok_logp, mask, _ = response_logps(
                model, full, L, full_attn,
                expand(pb["pixel_values"], n_group) if pb["pixel_values"] is not None else None,
                expand(pb["image_positions"], n_group),
            )
        ratio = torch.exp(tok_logp - tok_logp.detach())
        pg = -torch.min(ratio * adv.unsqueeze(1), (ratio.clamp(1 - clip_lo, 1 + clip_hi)) * adv.unsqueeze(1))
        loss = (pg * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(o.get("clip_grad", 1.0)))
        opt.step()
        if step % int(t["batch"].get("log_steps", 1)) == 0:
            acc = sum(1 for grp in rewards for a, s in zip(grp, texts) if a > 0)
            print(
                f"[grpo] step {step}/{total} loss={float(loss):.4f} "
                f"reward>0 比例={acc / max(1, len(texts)):.2f} "
                f"({(time.time() - t0) / step:.1f}s/step)",
                flush=True,
            )
        if step % int(t["batch"].get("save_steps", 100)) == 0 or step == total:
            train_loop.save(model, tok, ckpt_dir / f"iter_{step:07d}")
    train_loop.save(model, tok, ckpt_dir / "final")
    print(f"[grpo] 完成：ckpt {ckpt_dir / 'final'}")
    return 0
