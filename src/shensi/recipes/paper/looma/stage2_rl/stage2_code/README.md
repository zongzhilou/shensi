# 代码方向 RL teacher

四方向之一，从 SFT 检查点起训。本目录只放该方向的奖励与配置，启动与语料准备的实现在
`../../common/launch_rl.py`、`../../common/prep_rl.py`。

| 项 | 值 |
|---|---|
| 起点 | SFT 的 HF 目录（`config/default.yaml` 的 `model.path`） |
| 奖励 | `reward.py`：抽取代码块并在临时目录跑随题测试，全部通过 1 分 |
| 算法档 | `default` / `dapo` / `drgrpo` / `token_baseline` / `critic` / `fsdp` / `gspo` / `cispo` / `tiny` |
| 配比 | `config/data_prep/data_blend_raw.json` |
| 早停 | 默认开（`val/reward`，越大越好） |

## 快速开始

```bash
python data_prep.py --prepare --config {tiny,default}   # prompts → parquet
python train.py --dry-run                               # 只打印 verl 命令
python train.py --profile gspo                          # 起训（档之一）
```

## 判据

验证集的可验证奖励（`test_freq` 出一次 val reward）；真起训的链路读数见
[RL 段总览](../README.md)。

## 下一步

- [RL 段总览](../README.md) —— 四臂分工与算法档
- [OPD](../../stage3_opd/README.md) —— 本臂产物的归宿
