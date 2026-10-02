# stage0_pt · 预训练 / 对齐（LCS-558k，只训 projector）

论文 §4.1：**"We train our model with only the projection model tuned in the PT stage."**
用 LCS-558k（LAION/CC/SBU 经 BLIP 重写的 caption，558k 条）把视觉特征对齐到语言空间，
产出的 projector 交给 [stage1_sft](../stage1_sft/) 做指令微调。

## 超参（`config/default.yaml`，论文 Table 10 PT 列）

| 项 | 值 |
|---|---|
| 数据 | LCS-558k，1 epoch |
| 可训 | 仅 projector（占位模型 = `model.visual.merger.*`）；vision + llm 冻结 |
| global batch | 256（micro 8 × accum 32） |
| lr | 1e-3，cosine，warmup ratio 0.03 |
| 优化器 | AdamW（β₁ 0.9 / β₂ 0.999，wd 0，clip 1.0） |
| 精度 | bf16 |
| 占位模型 | `variant: unified`（去 deepstack + Gemma 4 式 token 预算 1120）；`native` 可对照；`SHENSI_DEEPRECUR_MODEL` 可指本地目录；tiny/debug 档不下载权重 |

模型变体与三镜像（transformers / vllm / megatron）见 [../README.md](../README.md) 的
「三个模型镜像」一节：unified = `vision_config.deepstack_visual_indexes=[]`（视觉特征只在
输入 embedding 单点进入 LLM）+ processor 像素预算，全部由 config 驱动，不重写建模代码。

## 数据

```bash
python data_prep.py --discover          # 报落位（<FS>/datasets/llm/pre-training/LLaVA-Pretrain）
python data_prep.py --prepare           # HF 拉取 + 图像落盘 + 转 messages jsonl
python data_prep.py --prepare --config tiny   # 64 条小切片（本地验证）
```

单轮对话：`user = [image] + caption 提示`，`assistant = caption`。

## 档位

| profile | 模型 | 数据 | 用途 |
|---|---|---|---|
| `default` | Qwen3-VL-8B 占位 | LCS-558k 全量 | 论文口径 |
| `debug` | tiny 随机几何 | 真实切片（limit） | 管线验证，不下载权重 |
| `tiny`（`--smoke`） | tiny 随机几何 | 合成 | 冒烟，全程离线 |

## 已验证

`train.py --smoke`：5 步，loss 12.1 → 11.31（≈ln vocab 起步、梯度非零——mask 与只训 merger
的口径都对）；可训 1.3M（merger），冻结 vision 3.1M + llm 80.7M。

## 局限

- 产物是 HF 格式目录（`ckpt/stage0_pt/<profile>/final`），接续用 `--load` 指给它。
- 早停默认看 `eval_loss`（val 切 2%），`--no-early-stop` 关。
- 断点续训未接 `resume_from_checkpoint`，重跑从零（正式档跑长训前先补）。
