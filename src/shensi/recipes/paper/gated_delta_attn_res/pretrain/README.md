# pretrain：GDAR 架构对比的预训练 stage

这是配方里唯一的训练 stage：**同一几何、同一数据、同一超参，只换深度连接模块**（`--spec`），
产出各变体的 val loss / 下游分数对比。方案出处是 gdar_package 的 `code/train/RECIPE.md`
（§0 评审硬约束 → §5 评测计划 → §6 消融矩阵）。

## 1. 档位

| profile | 是什么 | spec |
|---|---|---|
| `default` | base 对照臂（plain Qwen3-0.6B） | 无（上游 local 默认规格） |
| `gdar` | **论文主行**：objective 闭式更新 + 阶梯 64 + address=delta + 8 读头 + Softmax₁ + 全白化 + r64 + 正性投影 | `gdar_spec.gdar_layer_spec_paper` |
| `ar` / `dar` | 恒等锚定的 AR / DAR 对照臂 | `depth_spec.ar_layer_spec` / `dar_layer_spec` |
| `ablations/*` | 设计消融 A1–A16 与 E3 的对照行（每行只改一处） | 见各 yaml 头注 |
| `debug` | tiny 几何（4L/256h/seq512）+ 真实语料的极小档 | 继承所选 profile 的 spec |
| `tiny` | mock 冒烟档（`--smoke` 用的就是它） | `gdar_layer_spec` |

几何（default/gdar/ar/dar）：Qwen3-0.6B——`num_layers 28 / hidden 1024 / 16 heads /
8 KV groups / kv_channels 128 / ffn 3072 / seq 2048 / vocab 151936`。所有变体的主干参数
完全相同（751,632,384，含 untied 输出层），这就是"参数匹配"的依据；连接模块的增量参数
用 `python -m ...train.checks` 第 [4] 项实测，不引用语录。

## 2. 数据口径

- **Tokenizer**：Qwen3 同款（配方根 `tokenizer/Qwen3-0.6B/`；`$SHENSI_GDAR_TOKENIZER` 可覆盖）。
- **混合**（`config/data_prep/default.json`）：Ultra-FineWeb(en) 0.75 + UltraData-Code 0.10
  + UltraData-Math 0.05 + Ultra-FineWeb(zh) 0.10——对齐 RECIPE.md §1 与包里
  `anchor/build_mix.py` 的四个源。数据集落位 `$SHENSI_FS/datasets/llm/pre-training/<name>/`；
  没有 bin/idx 时先跑：

```bash
python data_prep.py --discover                # 面貌（有哪些文件、缺什么）
python data_prep.py --prepare                 # 全量 → bin/idx + blend.json
python data_prep.py --prepare --limit 1000    # 调试档
```

产物在 `$SHENSI_FS/shensi/data/gated_delta_attn_res/`（blend.json + `*_text_document.bin/.idx`），
`train.py` 自动拾取。语料拿不到时链路验证走 `--smoke`（mock，不碰语料）。

## 3. 优化器与调度

AdamW β=(0.9, 0.95)、wd 0.1、clip 1.0、BF16；0.6B 档峰值 LR 6e-4、warmup 2%、
cosine → 10% 峰值（RECIPE.md §2/§3）。两点如实说明：

1. **LR 要先 pilot**：正式主跑前用 {3e-4, 6e-4, 1e-3} 各跑 ~2000 步校准（评审 1 质疑点），
   `--set train.model.optimizer.lr_scheduler.lr=…` 即可覆写。
2. **WSD 是计划口径**：stable 90% → decay 10% 切高质量子集的课程在集群主跑执行；
   mcore 档用 cosine 近似（无内建 WSD），两臂同配所以对比仍干净，但与论文口径的差异要写。

## 4. 判据与验收

| 检查 | 命令 | 判据 |
|---|---|---|
| 集成测试 | `python test_train.py` | tiny 5 步：rc=0、到最后一 iter、`[after training is done]`、无 Traceback |
| 离线校验 | `python -m ...train.checks` | [1] 共享参数逐位相等；[2] eval logits `torch.equal`；[4] 增量参数表；[5] 门/投影梯度存活 |
| 冒烟 | `python train.py --smoke` | 同集成测试判据 |

训练中每个 eval 记录 val loss、门均值、`tau_spread`、`key_cond` 等路由统计（HF 侧由
`train/compare.py` + `hf/guarantee.py` 的下界守卫承担；mcore 侧的统计探针接 verl 时再搬，
包里 `code/flagscale_runs/depth_check.py` 是逐值对拍的出处）。

## 5. 局限

- 主表 / 消融的集群真跑未做；本 stage 的定位是**链路就绪 + 单卡/tiny 验证**。
- `--spec` 用 local 实现路径（`transformer_impl: local` 固定在 default.yaml，别切 TE），
  吞吐按未融合口径估（GDAR ≈ 2.19× base 的时间，见 RECIPE.md §3）。
- GDAR 的 packed hidden（`(1+N)*H` 宽）不支持 `fp32_residual_connection` 与
  `recompute_granularity='full'`（层内显式拦下）；PP>1 走动态 p2p（层内已开
  `variable_seq_lengths`），HC/mHC 不支持 PP。
