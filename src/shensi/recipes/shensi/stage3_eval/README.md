# stage3_eval：评测（dsh 走 NeMo Gym）

默认让 **dsh（DeepSeek Harness）当 agent harness**、在沙箱里解 **NeMo Gym** 的基准任务：
任务内起 **vLLM**（我们的 Shensi 端点）→ 等端点健康 → 逐基准 `gym eval prepare` + `gym eval run --model-type vllm_model`
（带上 Gym 的 `harness_agent`，harness 命令就是 `dsh`）→ 收每个基准的 `*_aggregate_metrics.json` 成 `summary.json`。
结构对齐 Nemotron 3.5 Lightning 的 `stage3_eval`（`eval.py` + `config/{default,tiny}.yaml`）。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 入口 | `eval.py`（不是 `train.py`，也没有 `data_prep.py`：基准集由 Gym 自己拉） |
| 三个套件 | `--suite all`（默认，Gym + local）/ `--suite gym`（dsh 走 Gym）/ `--suite local`（离线可跑） |
| 环境 | `setup_env.sh` 一条命令备好（六步，幂等） |
| 世界模型 | 不在这一档：它按 AgentWorldBench 的五维判分，用 `../stage2_rl/stage4_world_model/bench.py` |
| 已验证 | `setup_env.sh` 六步全过（含自检）；`--dry-run` 的命令与覆盖项正确；`--profile tiny --suite local` 对桩端点跑完并落 `summary.json` |

> Nemotron 那边 instruct 基准已从 NeMo Evaluator 迁到 **Gym**；**base-model 任务**（lm-evaluation-harness
> 短上下文、RULER 长上下文）不是 Gym 环境，仍走 `nemo-evaluator-launcher`（config 里登记了 `launcher` 段）。

## 2. 环境（`setup_env.sh` 一条命令）

```bash
PROXY=http://127.0.0.1:7897 bash setup_env.sh
```

按顺序做六件事，每一步都可复跑（幂等）：

1. 建**独立 venv** `$SHENSI_ROOT/.venv_gym`（训练用的 `.venv` 不动）；
2. 装 `nemo-gym` 与 `deepseek-harness-sdk`（dsh 的 Python SDK，自带 runtime，不需要系统 Node）；
3. 备 Gym 检出到 `$SHENSI_ROOT/Gym`（benchmarks / environments 都在里面）；
4. 初始化 dsh profile：`DSH_HOME=$SHENSI_FS/shensi/dsh-home` + `dsh --profile sdk-minimal --dump-default-config`；
5. 给 profile 写补丁层 `cordis.patch.yml`，把 `llm-deepseek` 的 `baseUrl` 指到我们的 vLLM、`model` 指到 `shensi`；
6. **自检**：`dsh --profile sdk-minimal --dump-config` 里必须能看到那个端点，看不到就报错退出。

跑之前 export 两个环境变量（setup 的输出里有现成的）：

```bash
export DSH_HOME=$SHENSI_FS/shensi/dsh-home
export DEEPSEEK_API_KEY=dummy          # 本地 vLLM 不校验；托管端点填真 key
```

## 3. 运行

```bash
python eval.py --dry-run        # 先看命令：vllm serve + gym eval + harness_agent=dsh 的覆盖项
python eval.py                  # 默认：起服务 + Gym（dsh 当 harness）+ local 套件
python eval.py --suite gym      # 只跑 Gym
python eval.py --suite local    # 只跑不依赖 Gym 的 local 套件（离线可跑）
python eval.py --profile tiny   # 5 条冒烟，只验"起服务 → 打端点 → 出分数"
```

`gym` 段里 `workdir` 指向 Gym 检出、`command` 是调用方式（容器里常是 `uv run gym`）、`benchmarks` 默认只开
`gpqa`（唯一开箱可用的，需要 `HF_TOKEN` 取数据；其余如 hle / scicode / browsecomp / tau2 要 judge 模型、
API key 或资源文件，按 Nemotron 的写法加 `{name, overrides}`）。
`agent` 段会被翻成 Gym 的 `HarnessAgent` 覆盖项（字段对应 Gym 的 `harness_agent/app.py`）：

```text
++policy_model.responses_api_agents.harness_agent.agent=dsh
++policy_model.responses_api_agents.harness_agent.sandbox_model_base_url=<endpoint>/v1
++policy_model.responses_api_agents.harness_agent.sandbox_image=python:3.12-slim
++policy_model.responses_api_agents.harness_agent.setup_commands=["export DSH_HOME=…","export DEEPSEEK_API_KEY=dummy"]
```

## 4. local 套件（不依赖 Gym，离线可跑）

| 集 | 做法 | 判据 |
| --- | --- | --- |
| 能力集 | 从 post-training 抽 held-out prompt，与 RL 共用 verifier 打分 | 训练后 ckpt 在数学/代码上优于训练前 |
| 长上下文 | 长文档（`FinePDFs`）首尾拼到 `longctx_chars`，**中间埋一个只出现一次的编号**再问 | 命中率高于随机；换更长档不 OOM、不塌 |

## 5. 验收判据

1. `setup_env.sh` 第 ⑥ 步自检通过（profile 里能看到我们的端点）；
2. `--dry-run` 的命令与预期一致（换机器/换基准后先看一遍再跑）；
3. `--profile tiny` 出分数、`summary.json` 落盘；
4. Gym 侧：目标基准的 `*_aggregate_metrics.json` 齐全，`summary.json` 能看到每个基准的分数；
5. local 侧：能力集两组分数 + `longctx-*` 命中率记录在 `summary.json`。

## 6. 局限

**需要目标环境才能跑的部分**：Gym 基准本身要 `HF_TOKEN` 与各基准的数据/判分资产；dsh 在沙箱里的 harness 命令
要按实际容器镜像调（`agent.command` 是透传给 `HarnessAgent.agent_kwargs` 的）。这两步在有网、有资产的机器上
跑通后，把结果与 `summary.json` 收下来即可。
