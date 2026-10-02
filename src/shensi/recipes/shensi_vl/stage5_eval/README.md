# Stage 5：评测

对齐论文 §3.2/§3.3 的口径（本配方跑的是本地可复现子集）：

| 套件 | 论文基准 | 本配方 |
| --- | --- | --- |
| counting | CountQA / Pixmo-Count / DS_Finegrained_Counting | Pixmo-Count 落地集 EM + GQA 自建细粒度（`--suite count`） |
| spatial & VQA | SpatialMQA / CV-Bench / DS_Spatial_Reasoning | GQA 关系题（stage1 的 gqa 任务池换种子出题） |
| topological | DS_Maze_Navigation / DS_Path_Tracing（各 2000） | 合成器换种子现场生成（同分布），`--suite synth` |

模型走 HF generate（贪心），判分规则与 stage2_rl/reward.py 同源（答案提取一致）。

```bash
python eval.py --suite synth --n 200              # 迷宫 + 路径追踪
python eval.py --suite all --n 500 --model $SHENSI_FS/shensi/ckpt/shensi_vl/stage4_opd/final
```

产物 `$SHENSI_FS/shensi/data/shensi_vl/stage5_eval/summary.json`。
vLLM 路线：自定义 VL 架构注册进 vLLM 后可用基座 stage3_eval 的服务化评测（见配方 README 局限）。
