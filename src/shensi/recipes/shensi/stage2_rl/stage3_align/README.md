# Stage 2.3: 偏好 / 指令 / 安全对齐

`stage2_rl` 的第三段：轨迹不再靠规则判分，接 GenRM 判分模型（verl 的 `reward_model` 通道）；
GRPO 与双侧截断口径不变，优化器继续用 AdaMuon（矩阵腿）+ AdEMAMix（标量腿）。

## 总览

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口（同一套 verl 命令） |
| `test_train.py` | 集成预检 |
| `data_prep.py` | 偏好 / 指令遵循 / 安全集 → parquet（`reward_model.ground_truth` 换成判分规格） |
| `config/` | `default.yaml` + `tiny.yaml` + `debug.yaml`（`base:` 继承 `../stage2_agentic/config`） |
| `config/data_prep/` | `data_blend_raw.json` + `data_blend_tiny.json` + 两个准备档 |

| 项 | 值 |
| --- | --- |
| 目标 | 指令遵循（结构化输出 / 日历 / 多轮）、安全性（红线）、偏好质量 |
| 数据 | 指令遵循、InverseIFEval、安全（5 个集） |
| 超参 | 继承 agentic 档，覆盖：`rollout.n: 8`、`max_response_length: 8192`、`lr: 5e-7`、`total_epochs: 50` |
| 优化器 | AdaMuon（矩阵腿）+ AdEMAMix（标量腿） |
| 判分 | 打开 verl 的 `reward_model` 通道，`reward_model.ground_truth` 换成偏好 / 评分规格；判分模型可以是自己的 ckpt 或托管端点 |

## 快速开始

```bash
python test_train.py --data-dir <parquet 目录>       # 集成预检
python data_prep.py --prepare && python train.py --dry-run && python train.py
```

判分端点用外部服务（GenRM）或本机的 CPU 判分服务：

```bash
python -m shensi.recipes.shensi.stage2_rl.stage4_world_model.local_judge \
    --model-dir $SHENSI_FS/models/SmolLM2-360M-Instruct --port 8000
export SHENSI_JUDGE_URL=http://127.0.0.1:8000/v1 SHENSI_JUDGE_MODEL=SmolLM2-360M-Instruct
```

## 验证

1. **集成预检 PASS**；
2. 安全类不退化（红线用例 0 命中）；
3. 指令遵循的结构化输出合法率上升；
4. `critic/score/mean` 不塌；
5. 早停：验证准确率超耐心即收尾。

本机实测：从导出的 SFT ckpt 起跑 19/19 步通过，权重同步 20 次。

## 局限

1. GenRM 判分模型的选型与规模未做消融；判分器自身的偏好会直接进入策略（同源风险），
   接托管端点或换更大判分器时先小规模对拍；
2. 判分端点属于外部依赖：本机可用 CPU 小模型（`local_judge.py`，逐维打分后组装官方五维 JSON，
   解析率会打印）或外部端点；
3. 优化器沿用 RLVR / agentic 口径，LR 与系数未单独扫描。

## 下一步

评测见 [Stage 3: 评测](../../stage3_eval/README.md)。
