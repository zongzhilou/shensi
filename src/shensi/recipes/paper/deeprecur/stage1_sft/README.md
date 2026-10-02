# stage1_sft · 指令微调（LLaVA-mixed-665k，解冻 LLM）

论文 §4.1：**"In SFT stage, we unfreeze LLM"**，视觉编码器冻结（"otherwise, we freeze our
vision encoder for a fair comparison"）。接 [stage0_pt](../stage0_pt/) 的 projector 产物，
在 LLaVA-mixed-665k 上做 1 epoch 指令微调；DeepStack-V/HD 变体见 `config/sft_v.yaml`。

## 超参（`config/default.yaml`，论文 Table 10 "DeepStack SFT" 列）

| 项 | 值 |
|---|---|
| 数据 | LLaVA-mixed-665k，1 epoch |
| 可训 | LLM + projector；vision 冻结 |
| global batch | 128（micro 4 × accum 32） |
| lr | 2e-5，cosine，warmup ratio 0.03 |
| 优化器 | AdamW（β₁ 0.9 / β₂ 0.999，wd 0，clip 1.0） |
| 精度 | bf16 + gradient checkpointing |
| loss | assistant-only（prompt / 图像 token / padding 全部 mask） |
| 占位模型 | `variant: unified`（去 deepstack + Gemma 4 式 token 预算 1120）；`native` 可对照 |

## 变体（`config/sft_v.yaml`，DeepStack-V / HD）

- `freeze: []`——视觉编码器也训，分组学习率 `vision_lr: 2e-6`（论文正文 1e-6 / Table 10
  2e-6 两处不一致，取 Table 10；要正文口径 `--set train.vision_lr=1e-6`）。
- 数据换 HD 的 748K 混合：`python data_prep.py --blend data_blend_hd.json --prepare`
  （论文 Table 9 原样，含 VQAv2/A-OKVQA/RefCOCO 的 task prompt 注入）。

## 数据

```bash
python data_prep.py --discover          # 报每个数据集的论文条数 / hf id / 本地落位
python data_prep.py --prepare           # 665k 配比（能自动拉的自动拉，缺的显式报错）
python data_prep.py --prepare --config tiny   # LLaVA-Instruct 64 条切片
```

LLaVA-Instruct 的图像（COCO + VG）要手动放
`<FS>/datasets/llm/post-training/images/llava-instruct`（保持 json 里的相对子路径）。

## 档位

| profile | 模型 | 数据 | 用途 |
|---|---|---|---|
| `default` | Qwen3-VL-8B 占位 | 665k 全量 | 论文口径 |
| `sft_v` | 同上 | 配 HD 混合 | V/HD 变体（vision 2e-6） |
| `debug` | tiny 随机几何 | 真实切片 | 管线验证 |
| `tiny`（`--smoke`） | tiny 随机几何 | 合成 | 冒烟，全程离线 |

## 已验证

`train.py --smoke`：5 步，loss 11.86 → 10.45；llm + projector 可训、视觉塔冻结，
assistant-only loss mask 区间在图像展开后不错位。

## 局限

- 多轮对话 loss 覆盖**每一段 assistant**（逐轮算展开后的 token 区间），与 LLaVA 口径一致。
- 全参 8B + AdamW 按 8×H100 规划；本机 16GB 只够 tiny/debug 真跑。
- 断点续训未接 `resume_from_checkpoint`（同 stage0_pt）。
