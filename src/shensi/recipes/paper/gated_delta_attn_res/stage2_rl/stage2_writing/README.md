# 写作方向 RL teacher

四方向之一，从 SFT 检查点起训。结构与其余三臂完全一致：本目录只放该方向的奖励与配置，
启动与语料准备的实现在 `../../common/train_rl.py`、`../../common/prep_rl.py`。

| 项 | 值 |
|---|---|
| 起点 | SFT 的 HF 目录（`config/default.yaml` 的 `model.path`） |
| 奖励 | `reward.py`：篇幅 + 结构 + 与要点的词面覆盖，三项加权（0.4 / 0.3 / 0.3） |
| 算法档 | `default` / `dapo` / `drgrpo` / `token_baseline` / `critic` / `fsdp` / `gspo` / `cispo` |
| 配比 | `config/data_prep/data_blend_raw.json`（按数据集列 prompts 清单） |
| 早停 | 默认开（`val/reward`，越大越好） |

## 快速开始

```bash
python data_prep.py --prepare --config {tiny,default}   # prompts → parquet
python train.py --dry-run                               # 只打印 verl 命令
python train.py --profile gspo                          # 起训（八档之一）
```

## 判据

各方向验证集的可验证奖励（`test_freq` 出一次 val reward）；teacher 之间不比高低。

## 下一步

- [RL 段总览](../README.md) —— 四臂分工与八个算法档
- [OPD](../../stage3_opd/README.md) —— 本臂产物的归宿
