# MindSpeed 移植清单（本 mcore revision 下 MindSpeed 无法加载的引用）

生成方式：对 `3rdparty/ascend/MindSpeed` 全部 1083 个 `.py` 做 AST 扫描，把每一处
`from megatron... import X` 逐个到 `3rdparty/common/Megatron-LM` 里验证；下面是**不存在**的引用。

## 为什么这台机器上 MindSpeed 是关闭的

`mindspeed.megatron_adaptor` 导入期会跑 `patch_features()` → `apply_features_patches()` →
逐个 feature 的 `register_patches()`，最后 `MindSpeedPatchesManager.apply_patches()`。任一环抛 ImportError，
`megatron_adaptor` 的 wrapper 会把异常吞成一行 warning，**但已经注册/应用的补丁不会回滚**。

实测两种状态：

| MindSpeed 失败位置 | 实际效果 |
|---|---|
| `register_patches()` 内（早失败） | `apply_patches()` 从未执行 → **一个补丁都没装** → 训练正常（已验证 9/9 步） |
| `apply_patches()` 内（晚失败） | 补丁**部分应用**，它自己的 `validate_features_args` 等钩子随后在 config 构建/前向里炸 → 冒烟失败 |

所以**不要**用"让它多加载一步"的垫片（我们试过 `expert_dist_ckpt_decorator` 恒等垫片 + 两个
`parallel_state` 符号 stub）：每修一个，它就多装一批钩子，然后炸在下一个缺失符号上
（`filesystem_async` → `create_hierarchical_parallel_groups` → `is_pipeline_stage_before_split` → …）。
**修法只能是完整移植**，让 MindSpeed 整体加载成功。

## 缺失引用（共 30 处）

- **MODULE** `megatron`
  - mindspeed/core/pipeline_parallel/flexible_schedules.py:21
- **MODULE** `megatron.core.datasets.megatron_tokenizer`
  - mindspeed/tokenizer/build_tokenizer/adaptor.py:6
  - mindspeed/tokenizer/tokenizer.py:21
- **MODULE** `megatron.core.distributed.custom_fsdp`
  - mindspeed/core/distributed/torch_fully_sharded_data_parallel/training.py:15
  - mindspeed/core/pipeline_parallel/dualpipev/dualpipev_chunks.py:14
- **MODULE** `megatron.core.distributed.custom_fsdp.param_and_grad_buffer`
  - mindspeed/core/distributed/custom_fsdp/param_and_grad_buffer.py:5
- **MODULE** `megatron.core.transformer.custom_layers.transformer_engine`
  - mindspeed/core/transformer/deepseek_v4/attention.py:84
- **MODULE** `megatron.core.transformer.moe.legacy_a2a_token_dispatcher`
  - mindspeed/core/transformer/moe/moe_feature/__init__.py:61
- **MODULE** `megatron.legacy.fp16_deprecated`
  - mindspeed/checkpointing.py:301
- **MODULE** `megatron.legacy.model`
  - tests_extend/unit_tests/features/tensor_parallel/test_rms_norm2d.py:31
- **MODULE** `megatron.legacy.model.module`
  - mindspeed/core/pipeline_parallel/dualpipev/dualpipev_chunks.py:24
- **MODULE** `megatron.legacy.model.rms_norm`
  - mindspeed/core/fusions/rms_norm.py:6
  - mindspeed/core/transformer/custom_layers/transformer_engine.py:6
  - tests_extend/unit_tests/ops/cann/test_npu_rms_norm.py:7
- **MODULE** `megatron.legacy.model.transformer`
  - mindspeed/core/data_parallel/distributed_data_parallel.py:13
  - mindspeed/features_manager/memory/swap_attention.py:34
  - tests_extend/unit_tests/features/unaligned_pp/test_unaligned_pp.py:18
  - tests_extend/unit_tests/mindspeed/core/context_parallel/test_hybrid_context_parallel.py:38
  - tests_extend/unit_tests/ops/cann/test_npu_swiglu.py:8
- **MODULE** `megatron.legacy.model.vision.knn_monitor`
  - mindspeed/core/pipeline_parallel/dualpipev/dualpipev_chunks.py:602
- **MODULE** `megatron.training.tokenizer.tokenizer`
  - mindspeed/tokenizer/build_tokenizer/adaptor.py:5
  - mindspeed/tokenizer/tokenizer.py:20
  - mindspeed/training.py:188
  - tools/preprocess_data.py:30
- **MODULE** `megatron_adaptor`
  - mindspeed/core/distributed/layerzero/__init__.py:3
- **SYMBOL** `megatron.core.datasets.indexed_dataset`
  - tools/data_handler.py:30
- **SYMBOL** `megatron.core.dist_checkpointing`
  - mindspeed/checkpointing.py:24
  - mindspeed/core/multi_modal/dist_train/dist_schedules.py:9
- **SYMBOL** `megatron.core.parallel_state.create_hierarchical_parallel_groups`
  - mindspeed/core/multi_modal/dist_train/dist_parallel_state.py:15
- **SYMBOL** `megatron.core.parallel_state.get_moe_layer_wise_logging_tracker`
  - mindspeed/core/pipeline_parallel/noop_layers/adaptor.py:14
- **SYMBOL** `megatron.core.parallel_state.is_pipeline_stage_after_split`
  - mindspeed/core/pipeline_parallel/multi_parameter/adaptor.py:13
- **SYMBOL** `megatron.core.parallel_state.is_pipeline_stage_before_split`
  - mindspeed/core/multi_modal/dist_train/dist_schedules.py:13
- **SYMBOL** `megatron.core.pipeline_parallel.p2p_communication`
  - mindspeed/core/memory/auto_pipeline/autopipeline.py:20
  - mindspeed/core/performance/auto_pipeline_perf/autopipeline_perf.py:18
  - mindspeed/core/performance/auto_pipeline_perf/schedules.py:23
  - mindspeed/core/pipeline_parallel/flexible_schedules.py:34
  - mindspeed/core/pipeline_parallel/multiparameter_schedules.py:25
  - mindspeed/core/pipeline_parallel/ripipe_schedules.py:26
  - …(共 9 处)
- **SYMBOL** `megatron.core.transformer.moe.experts.expert_dist_ckpt_decorator`
  - mindspeed/te/pytorch/module/grouped_linear.py:25
  - mindspeed/te/pytorch/module/performance_grouped_linear.py:23
- **SYMBOL** `megatron.core.transformer.moe.grouped_gemm_util`
  - mindspeed/core/transformer/moe/experts.py:9
  - mindspeed/core/transformer/moe/moe_feature/__init__.py:9
- **SYMBOL** `megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity`
  - mindspeed/core/transformer/moe/moe_feature/__init__.py:12
  - mindspeed/core/transformer/moe/router.py:5
- **SYMBOL** `megatron.training.checkpointing.fix_fp8_params_lose_precision_when_loading_dist_ckpt`
  - mindspeed/core/distributed/torch_fully_sharded_data_parallel/checkpointing.py:32
- **SYMBOL** `megatron.training.ft_integration`
  - mindspeed/core/distributed/torch_fully_sharded_data_parallel/checkpointing.py:25
  - mindspeed/core/pipeline_parallel/dualpipev/dualpipev_chunks.py:16
- **SYMBOL** `megatron.training.one_logger_utils`
  - mindspeed/core/pipeline_parallel/dualpipev/dualpipev_chunks.py:16
  - mindspeed/training.py:14
- **SYMBOL** `megatron.training.utils.get_args`
  - mindspeed/core/transformer/moe/moe_feature/balanced_moe/adaptor.py:9
- **SYMBOL** `megatron.training.utils.unwrap_model`
  - mindspeed/checkpointing.py:27
  - mindspeed/core/data_parallel/async_log_allreduce.py:29
  - mindspeed/core/distill/teacher_model_manager.py:208
  - mindspeed/core/distributed/layerzero/config.py:20
  - mindspeed/core/distributed/torch_fully_sharded_data_parallel/checkpointing.py:26
  - mindspeed/core/distributed/torch_fully_sharded_data_parallel/torch_fully_sharded_data_parallel_adaptor.py:44
  - …(共 14 处)
- **SYMBOL** `megatron.training.wandb_utils`
  - mindspeed/core/distributed/torch_fully_sharded_data_parallel/checkpointing.py:25
