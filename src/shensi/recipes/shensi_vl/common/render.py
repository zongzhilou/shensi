"""文本渲染：DSV4F 的 DSv4 chat 模板 + 图像槽位 → input_ids / labels / 图像位置。

与 shensi 的 stage1_sft 完全同一条编码路径（官方 encoding_dsv4.py，跨配方直接 import）；
差别只有两处：
  1. user 消息里的 ``<image>`` 占位（每张图一个，按出现顺序对应 images 列表）被展开成
     N 个 ``<|image_pad|>``（N 由 Kimi K3 image_processor 按该图分辨率给出）；
  2. 数据侧占位（<box>/<point>…）先换成原语特殊 token（<|box|>…）。
loss mask 与 stage1_sft 同口径：只算 assistant 段（``<｜Assistant｜>`` 之后到 eos，含 eos），
``<think>``/``</think>`` 脚手架不参训；图像位永远 mask。
"""

from __future__ import annotations

from shensi.recipes.shensi_vl.common import primitives as P
from shensi.recipes.shensi_vl.common import vl_tokens

# 论文里的触发词（§4 Limitations：目前靠显式触发词激活）：system/user 里带上
TRIGGER = "Use visual primitives (boxes/points) in your thinking."


def build_user_content(question: str, n_images: int, *, trigger: bool = True) -> str:
    head = f"{TRIGGER}\n" if trigger else ""
    slot = "<image>" * n_images
    return f"{head}{slot}\n{question}" if slot else f"{head}{question}"


def render_text(messages: list[dict]) -> str:
    """messages（role/content，content 可含 <image> 与原语占位）→ DSv4 chat 文本。"""
    from shensi.recipes.shensi.stage1_sft import encoding_dsv4 as enc

    msgs = []
    for m in messages:
        role = m["role"]
        content = str(m.get("content") or "")
        if role == "assistant":
            reasoning = m.get("reasoning_content") or ""
            msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
        else:
            msgs.append({"role": role, "content": content})
    mode = "thinking" if any(m.get("reasoning_content") for m in msgs) else "chat"
    return enc.encode_messages(msgs, thinking_mode=mode)


def encode_sample(
    tok,
    messages: list[dict],
    image_counts: list[int] | None,
) -> dict:
    """一条样本 → input_ids / loss_mask / image_positions。

    messages: 已含 <image> 占位与原语占位（数据侧格式）；image_counts: 每张图的图像 token 数。
    <image> 占位不进 tokenizer：先按它切分文本分段编码，再拼接 N 个 <|image_pad|>
    （BPE 会把夹在长文本里的 marker 合并成别的切分，逐 id 匹配不可靠）。
    """
    vocab = vl_tokens.token_ids(tok)
    img_id = vocab[vl_tokens.IMAGE_TOKEN]
    counts = list(image_counts or [])
    text = render_text(messages)

    ids: list[int] = []
    for k, seg in enumerate(text.split("<image>")):
        if k:
            n = counts.pop(0) if counts else 0
            ids.extend([img_id] * int(n))  # marker 处展开成该图的图像 token
        if seg:
            ids.extend(tok(seg, add_special_tokens=False)["input_ids"])
    a_id = tok("<｜Assistant｜>", add_special_tokens=False)["input_ids"][0]
    eos = tok.eos_token_id

    # loss mask：assistant 段（<｜Assistant｜> 后到 eos，含 eos）；图像位永远 0
    think_close = tok("</think>", add_special_tokens=False)["input_ids"]
    mask = [0] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] != a_id:
            i += 1
            continue
        j = i + 1
        if ids[j : j + len(think_close)] == think_close:
            j += len(think_close)
        else:
            j += 1
        while j < len(ids) and ids[j] != eos:
            mask[j] = 1
            j += 1
        if j < len(ids):
            mask[j] = 1
        i = j + 1
    for k, v in enumerate(ids):
        if v == img_id:
            mask[k] = 0

    return {
        "input_ids": ids,
        "loss_mask": mask,
        "image_positions": [k for k, v in enumerate(ids) if v == img_id],
        "image_token_id": img_id,
    }


def finalize_text(text: str) -> str:
    """数据侧占位 → 训练侧原语 token（写数据后、渲染前调用一次）。"""
    return P.to_train_text(text)
