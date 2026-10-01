# 代码方向 RL teacher（stage2_code）

四方向之一。结构与其余三臂完全一致（脚本是薄封装，实现在 `../common/`）：

```bash
python data_prep.py --prepare --config {tiny,default}   # prompts → parquet
python train.py --{dry-run,profile <档>}                # 起训 / 只打印命令
```

| 项 | 值 |
|---|---|
| 起点 | SFT 的 HF 目录（`config/default.yaml` 的 `model.path`） |
| 奖励 | `reward.py`（单元测试执行（通过率）） |
| 算法档 | `default` / `dapo` / `drgrpo` / `token_baseline` / `critic` / `fsdp` / `gspo` / `cispo` |
| 配比 | `config/data_prep/data_blend_raw.json`（`datasets[].name` = UltraData-RL-2609 的对应子集） |
| 早停 | 默认开（`val/reward`，越大越好） |

判据：各方向验证集的可验证奖励（`test_freq` 出一次 val reward）；teacher 之间不比高低。

## 索引

- [RL 段总览](../README.md) —— 四臂分工与八个算法档
- [OPD](../../stage3_opd/README.md) —— 本臂产物的归宿
