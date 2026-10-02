# Stage 1：Specialized SFT（box / point 两份专家）

论文 §2.5.1：预训练模型具备基础原语能力后，用冷启动数据分两份做 SFT——
**thinking with grounding**（box，`--profile default` → F_TwG）与
**thinking with pointing**（point，`--profile point` → F_TwP）。
数据量小时拆开训，防两类原语的模式互相打架。混合比例 70% 通用多模态 + 30% 专项。

## 冷启动数据（论文 §2.4 → 本配方口径）

| 任务族 | 论文做法 | 本配方 | 量级（论文 → blend 默认） |
| --- | --- | --- | --- |
| coarse 计数 | 密集检测集 + MLLM 合成三段思维链 + 严格校验 | 密集检测集（coco/crowdhuman/visdrone）+ **程序化**合成同样的"意图→批量grounding→求和"链（天然过校验） | 1 万 → blend 权重 |
| 细粒度计数 | GQA 场景图 + MLLM 出题/思维链 | GQA 场景图**程序化**出题（属性约束 + sequential scan + count=0 负样本） | — |
| 空间推理/通用 VQA | GQA + CLEVR 工具链 + 负样本 | GQA 场景图关系题（正/负样本）+ 通用图文 | 9 千 → blend 权重 |
| 迷宫导航 | DFS/Prim/Kruskal 合成 46 万（三拓扑、可解/不可解、四难度） | `common/synth_maze.py` 同口径（46 万按本地预算缩，blend `n_maze`） | 46 万 → 2 万 |
| 路径追踪 | Bézier 合成 12.5 万（交叉消歧、uniform 模式） | `common/synth_trace.py` 同口径 | 12.5 万 → 8 千 |

论文的 MLLM 合成思维链换成**程序化合成**（元数据直出、无需再调 MLLM）；
代价是思维链风格比 MLLM 生成的单一，换来的是每个框/点都严格对齐标注、零噪声。

## 跑法

```bash
python data_prep.py --discover            # 落地数据面貌
python data_prep.py --join-gqa            # GQA 图像×场景图合并（一次性）
python data_prep.py --prepare             # 全量；冒烟 --blend config/data_prep/data_blend_tiny.json --limit 16 --no-render
python train.py --profile debug           # tiny LM + 真数据
python train.py --profile default         # F_TwG（box）
python train.py --profile point           # F_TwP（point）
python test_train.py
```

## 产物

`$SHENSI_FS/shensi/ckpt/shensi_vl/stage1_sft/{box,point}/final` —— 下一 stage 的
RL 分别从这两个目录起（两族专家各 RL 各的）。
