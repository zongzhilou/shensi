"""可训规模注册表（**接口层：只登记型号与几何，不下载任何权重**）。

单卡昇腾 910B（64GB 档）口径下的三档：

- ``2b``：**主档** —— 三臂（native / unified / deeprecur）的 PT + SFT + 多 seed 都放得下；
- ``4b``：**第二档** —— PT 必做；SFT 视显存（须梯度检查点 + 冻结主干）；
- ``8b``：**趋势点** —— 只做 PT（冻结主干，只训对齐/连接件）。

ModelScope 集合 ``Qwen3-VL-5c7a94c8cb144b``（2026-10-02 浏览器核对，共 37 条 = 36 模型 + 1 创空间）
的结构：**6 个规模 × 2 范式 × 3 格式**——

| 规模 | Instruct | Thinking | 额外格式 |
|---|---|---|---|
| 2B / 4B / 8B | ✓ | ✓（同几何，换名即可） | FP8、GGUF（**仅推理**） |
| 30B-A3B / 32B | ✓ | ✓ | FP8、GGUF（仅推理） |
| 235B-A22B | ✓ | ✓ | FP8、GGUF（仅推理） |

训练一律用 **BF16 的 Instruct / Thinking**；FP8 给 vLLM 推理、GGUF 给 llama.cpp。30B-A3B /
32B / 235B-A22B 单卡放不下，排除。几何取自各型号官方 ``config.json``（2026-10 核对）；
**deepstack 注入点各型号不同**（2B/4B = [5,11,17]，8B = [8,16,24]）——native 臂直接用它，
deeprecur 臂置空（设计如此）。

权重获取走 ``model.placeholder.name``（HF / ModelScope 同名；也可先用 modelscope/ms 下到
本地目录再填路径，或设 ``SHENSI_DEEPRECUR_MODEL``）。本模块只提供接口，不做任何拉取。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLSize:
    """一个可训规模的完整接口描述。"""

    key: str
    hf_id: str
    params: str
    text_layers: int
    text_hidden: int
    vision_depth: int
    vision_hidden: int
    deepstack_indexes: tuple[int, ...]
    vocab_size: int
    image_token_id: int
    #: 两塔共享的递归块数（deeprecur/gdar 臂用）：取使两塔块内层数都是整数的公约数
    recur_blocks: int
    #: 单卡 910B 的档位说明
    tier: str


SIZES: dict[str, VLSize] = {
    "2b": VLSize(
        key="2b",
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        params="2.1B",
        text_layers=28,
        text_hidden=2048,
        vision_depth=24,
        vision_hidden=1024,
        deepstack_indexes=(5, 11, 17),
        vocab_size=151936,
        image_token_id=151655,
        recur_blocks=7,  # text 4 层/块 · vision 3 层/块（各 7 块）
        tier="主档：三臂 PT+SFT+多 seed",
    ),
    "4b": VLSize(
        key="4b",
        hf_id="Qwen/Qwen3-VL-4B-Instruct",
        params="4.4B",
        text_layers=36,
        text_hidden=2560,
        vision_depth=24,
        vision_hidden=1024,
        deepstack_indexes=(5, 11, 17),
        vocab_size=151936,
        image_token_id=151655,
        recur_blocks=6,  # text 6 层/块 · vision 4 层/块（12 块=text 3 · vision 2 也可选）
        tier="第二档：PT 必做；SFT 视显存（开梯度检查点）",
    ),
    "8b": VLSize(
        key="8b",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        params="8.8B",
        text_layers=36,
        text_hidden=4096,
        vision_depth=27,
        vision_hidden=1152,
        deepstack_indexes=(8, 16, 24),
        vocab_size=151936,
        image_token_id=151655,
        recur_blocks=9,  # text 4 层/块 · vision 3 层/块（各 9 块）
        tier="趋势点：只做 PT（冻结主干，只训对齐/连接件）",
    ),
}


def profile_name(key: str) -> str:
    """型号 key → 几何档名（``--profile geoms/<name>``）。"""
    if key not in SIZES:
        raise SystemExit(f"[deeprecur] 不认识这个规模：{key}（可选：{sorted(SIZES)}）")
    return f"geoms/qwen3_vl_{key}"


def checkpoint_id(key: str, paradigm: str = "Instruct") -> str:
    """型号 key + 范式（Instruct / Thinking）→ 检查点名（HF/ModelScope 同名）。

    两个范式**几何相同**，只差后训练；训练用 BF16，FP8/GGUF 仅推理格式不在此列。
    """
    size = get(key)
    if paradigm not in ("Instruct", "Thinking"):
        raise SystemExit(f"[deeprecur] 范式只认 Instruct/Thinking：{paradigm}")
    return size.hf_id.rsplit("-", 1)[0] + f"-{paradigm}"


def get(key: str) -> VLSize:
    """按 key 取规模描述（找不到显式报错）。"""
    if key not in SIZES:
        raise SystemExit(f"[deeprecur] 不认识这个规模：{key}（可选：{sorted(SIZES)}）")
    return SIZES[key]


def main() -> int:
    print("[deeprecur] 可训规模（单卡 910B 口径；只登记几何，不下载权重）：")
    for size in SIZES.values():
        print(
            f"  {size.key:<3} {size.params:<6} {size.hf_id:<32}"
            f" text {size.text_layers}×{size.text_hidden} · vision {size.vision_depth}×{size.vision_hidden}"
            f" · deepstack {list(size.deepstack_indexes)} · recur_blocks={size.recur_blocks}"
        )
        print(f"      {size.tier}   档位名：{profile_name(size.key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
