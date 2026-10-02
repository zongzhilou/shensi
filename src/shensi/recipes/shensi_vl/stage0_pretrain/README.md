# Stage 0：VL 预训练（grounding 为主的多模态预训练）

论文 §2.3：在 DSV4F 之上做预训练，让模型获得**输出视觉原语**的基础能力
（box 定位物体、point 抽象引用），坐标统一归一到 0–999。论文用万亿级多模态 token，
本配方**按配比对齐、绝对量按本地预算缩**（`train_iters` 就是预算旋钮）。

## 数据（论文口径 → 开源替代）

| 论文数据 | 本配方落地（HF） | adapter |
| --- | --- | --- |
| 自建 40M 网络框标注（§2.3.3 两步过滤） | `detection-datasets/coco` + `surenreddy/object365` + `sshao0516/CrowdHuman` + `PaDT-MLLM/RefCOCO` | detection |
| PixMo-Points [4] | `allenai/pixmo-points` | points |
| 通用图文（网络爬取、不合成） | `lmms-lab/LLaVA-OneVision-Data` | caption |

落地：`$SHENSI_FS/datasets/llm/pre-training/<name>`（下载清单 `download_manifest.json`，
`huggingface-cli download <repo> --repo-type dataset --local-dir <dir>`）。

## 跑法

```bash
python data_prep.py --discover                 # 看落地数据与列名（对不上就改 blend 里的 key）
python data_prep.py --prepare                  # 全量；冒烟加 --limit 32
python train.py --profile debug                # 极小档：tiny LM + 真数据 20 步
python train.py --profile default              # 生产档（预算 = batch.train_iters）
python test_train.py                           # tiny 几何 + 假数据 3 步，PASS/FAIL
```

## 口径

- **tokenizer**：DSV4F（`$SHENSI_TOKENIZER`）+ 7 个增量特殊 token（6 原语 + `<|image_pad|>`），
  副本落在 `$SHENSI_FS/shensi/models/shensi-vl-tok`，基座词表不动；
- **图像侧**：Kimi K3 的 image_processor（patch 14 / merge 2×2 / mean·std 0.5），
  图像 token 数按 `make_image_prompt(w,h)` 用 DSV4F tokenizer 计数；
- **格式**：`<|ref|>目标<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>` 与
  `<|point|>[[x1,y1],...]<|/point|>`，多实例框从左到右；
- **优化器**：HF 循环用 AdamW（mcore 的 AdaMuon 双腿优化器吃不到 HF 模型，见配方 README 局限 2）。

## 已知限制

1. 论文的自建 40M 框数据是网络爬取 + MLLM 两步过滤的产物，开源集只覆盖其中一部分语义多样性；
   要再往上对齐，按 §2.3.3 的两步过滤流程自己扩爬（语义审查 → 几何质量审查）。
2. 视觉塔：论文的 DeepSeek-ViT 未开源，用与 Kimi image_processor 同 patch 口径的 HF ViT 权重
   （`SHENSI_VL_VISION`）替代；`CSA` KV 压缩来自 DSV4F 基座，HF 侧推理不带（见配方 README）。
3. `train.jsonl` 的图像按行落地成 jpg，磁盘占用 ≈ 图像总量；重跑 `--prepare` 会复用已有文件。
