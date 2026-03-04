# Offline Engine API

SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

- Offline Batch Inference
- Custom Server on Top of the Engine

This document focuses on the offline batch inference, demonstrating four different inference modes:

- Non-streaming synchronous generation
- Streaming synchronous generation
- Non-streaming asynchronous generation
- Streaming asynchronous generation

Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).



## Nest Asyncio
Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
```python
import nest_asyncio

nest_asyncio.apply()

```

## Advanced Usage

The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states). 

Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.

## Offline Batch Inference

SGLang offline engine supports batch inference with efficient scheduling.


```python
# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    [2026-03-04 06:40:00] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 06:40:00] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 06:40:00] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-04 06:40:03] INFO server_args.py:1975: Attention backend not specified. Use fa3 backend by default.


    [2026-03-04 06:40:03] INFO server_args.py:3066: Set soft_watchdog_timeout since in CI


    [2026-03-04 06:40:03] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=972630367, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.57it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.57it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=38.60 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=38.60 GB):   5%|▌         | 1/20 [00:00<00:04,  4.60it/s]Capturing batches (bs=120 avail_mem=37.79 GB):   5%|▌         | 1/20 [00:00<00:04,  4.60it/s]Capturing batches (bs=112 avail_mem=37.79 GB):   5%|▌         | 1/20 [00:00<00:04,  4.60it/s]Capturing batches (bs=112 avail_mem=37.79 GB):  15%|█▌        | 3/20 [00:00<00:01,  9.39it/s]Capturing batches (bs=104 avail_mem=37.79 GB):  15%|█▌        | 3/20 [00:00<00:01,  9.39it/s]

    Capturing batches (bs=96 avail_mem=37.79 GB):  15%|█▌        | 3/20 [00:00<00:01,  9.39it/s] Capturing batches (bs=96 avail_mem=37.79 GB):  25%|██▌       | 5/20 [00:00<00:01, 11.14it/s]Capturing batches (bs=88 avail_mem=38.51 GB):  25%|██▌       | 5/20 [00:00<00:01, 11.14it/s]Capturing batches (bs=80 avail_mem=38.51 GB):  25%|██▌       | 5/20 [00:00<00:01, 11.14it/s]Capturing batches (bs=72 avail_mem=38.50 GB):  25%|██▌       | 5/20 [00:00<00:01, 11.14it/s]

    Capturing batches (bs=72 avail_mem=38.50 GB):  40%|████      | 8/20 [00:00<00:00, 14.97it/s]Capturing batches (bs=64 avail_mem=38.50 GB):  40%|████      | 8/20 [00:00<00:00, 14.97it/s]Capturing batches (bs=56 avail_mem=37.84 GB):  40%|████      | 8/20 [00:00<00:00, 14.97it/s]Capturing batches (bs=56 avail_mem=37.84 GB):  50%|█████     | 10/20 [00:00<00:00, 16.17it/s]Capturing batches (bs=48 avail_mem=37.84 GB):  50%|█████     | 10/20 [00:00<00:00, 16.17it/s]Capturing batches (bs=40 avail_mem=37.84 GB):  50%|█████     | 10/20 [00:00<00:00, 16.17it/s]

    Capturing batches (bs=40 avail_mem=37.84 GB):  60%|██████    | 12/20 [00:00<00:00, 16.87it/s]Capturing batches (bs=32 avail_mem=37.84 GB):  60%|██████    | 12/20 [00:00<00:00, 16.87it/s]Capturing batches (bs=24 avail_mem=37.84 GB):  60%|██████    | 12/20 [00:00<00:00, 16.87it/s]Capturing batches (bs=24 avail_mem=37.84 GB):  70%|███████   | 14/20 [00:00<00:00, 17.56it/s]Capturing batches (bs=16 avail_mem=37.84 GB):  70%|███████   | 14/20 [00:00<00:00, 17.56it/s]Capturing batches (bs=12 avail_mem=37.84 GB):  70%|███████   | 14/20 [00:01<00:00, 17.56it/s]

    Capturing batches (bs=12 avail_mem=37.84 GB):  80%|████████  | 16/20 [00:01<00:00, 16.63it/s]Capturing batches (bs=8 avail_mem=38.50 GB):  80%|████████  | 16/20 [00:01<00:00, 16.63it/s] Capturing batches (bs=4 avail_mem=38.50 GB):  80%|████████  | 16/20 [00:01<00:00, 16.63it/s]Capturing batches (bs=2 avail_mem=38.50 GB):  80%|████████  | 16/20 [00:01<00:00, 16.63it/s]Capturing batches (bs=2 avail_mem=38.50 GB):  95%|█████████▌| 19/20 [00:01<00:00, 19.10it/s]Capturing batches (bs=1 avail_mem=38.49 GB):  95%|█████████▌| 19/20 [00:01<00:00, 19.10it/s]Capturing batches (bs=1 avail_mem=38.49 GB): 100%|██████████| 20/20 [00:01<00:00, 16.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:15,  2.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:15,  2.37s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:00,  1.07s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:00,  1.07s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:00,  1.07s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:14,  3.63it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:14,  3.63it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:14,  3.63it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:09,  5.34it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:09,  5.34it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.34it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:06,  7.11it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:06,  7.11it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:06,  7.11it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:06,  7.11it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04, 10.26it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04, 10.26it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04, 10.26it/s]

    Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04, 10.26it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.04it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.04it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.04it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.04it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 13.04it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 17.15it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 17.15it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 17.15it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 17.15it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 19.17it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:01, 19.17it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 24.78it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 24.78it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 24.78it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 24.78it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 25.22it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 25.22it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 25.22it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 25.22it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 25.22it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 25.22it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 30.96it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 30.96it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 30.96it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 30.96it/s]

    Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 30.96it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 38.62it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 38.62it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 38.62it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 38.62it/s]

    Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 38.62it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 38.62it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 40.65it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 40.65it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 40.65it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 40.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.10 GB):   2%|▏         | 1/58 [00:00<00:09,  5.86it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.09 GB):   2%|▏         | 1/58 [00:00<00:09,  5.86it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.09 GB):   3%|▎         | 2/58 [00:00<00:09,  6.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.14 GB):   3%|▎         | 2/58 [00:00<00:09,  6.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.14 GB):   5%|▌         | 3/58 [00:00<00:08,  6.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.13 GB):   5%|▌         | 3/58 [00:00<00:08,  6.24it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.13 GB):   7%|▋         | 4/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.13 GB):   7%|▋         | 4/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.13 GB):   9%|▊         | 5/58 [00:00<00:07,  6.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.12 GB):   9%|▊         | 5/58 [00:00<00:07,  6.69it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.12 GB):  10%|█         | 6/58 [00:00<00:08,  6.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.12 GB):  10%|█         | 6/58 [00:00<00:08,  6.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.12 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.12 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.53it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.12 GB):  14%|█▍        | 8/58 [00:01<00:07,  7.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.07 GB):  14%|█▍        | 8/58 [00:01<00:07,  7.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.07 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.10 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.54it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.10 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.09 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.09 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.09 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.06it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.09 GB):  21%|██        | 12/58 [00:01<00:05,  8.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.08 GB):  21%|██        | 12/58 [00:01<00:05,  8.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.06 GB):  21%|██        | 12/58 [00:01<00:05,  8.40it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=38.06 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.07 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.06 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.06 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.05 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.07 GB):  28%|██▊       | 16/58 [00:02<00:03, 11.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.04 GB):  28%|██▊       | 16/58 [00:02<00:03, 11.20it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=38.03 GB):  28%|██▊       | 16/58 [00:02<00:03, 11.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.03 GB):  34%|███▍      | 20/58 [00:02<00:02, 17.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.64 GB):  34%|███▍      | 20/58 [00:02<00:02, 17.06it/s]Capturing num tokens (num_tokens=960 avail_mem=56.63 GB):  34%|███▍      | 20/58 [00:02<00:02, 17.06it/s] Capturing num tokens (num_tokens=960 avail_mem=56.63 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.80it/s]Capturing num tokens (num_tokens=896 avail_mem=56.64 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.80it/s]

    Capturing num tokens (num_tokens=832 avail_mem=56.63 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.80it/s]Capturing num tokens (num_tokens=768 avail_mem=56.62 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.80it/s]Capturing num tokens (num_tokens=704 avail_mem=56.63 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.80it/s]Capturing num tokens (num_tokens=704 avail_mem=56.63 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.41it/s]Capturing num tokens (num_tokens=640 avail_mem=56.63 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.41it/s]Capturing num tokens (num_tokens=576 avail_mem=56.62 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.41it/s]Capturing num tokens (num_tokens=512 avail_mem=56.60 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.41it/s]Capturing num tokens (num_tokens=480 avail_mem=56.62 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.41it/s]

    Capturing num tokens (num_tokens=480 avail_mem=56.62 GB):  52%|█████▏    | 30/58 [00:02<00:01, 24.03it/s]Capturing num tokens (num_tokens=448 avail_mem=56.61 GB):  52%|█████▏    | 30/58 [00:02<00:01, 24.03it/s]Capturing num tokens (num_tokens=416 avail_mem=56.59 GB):  52%|█████▏    | 30/58 [00:02<00:01, 24.03it/s]Capturing num tokens (num_tokens=384 avail_mem=56.60 GB):  52%|█████▏    | 30/58 [00:02<00:01, 24.03it/s]Capturing num tokens (num_tokens=352 avail_mem=56.57 GB):  52%|█████▏    | 30/58 [00:02<00:01, 24.03it/s]Capturing num tokens (num_tokens=352 avail_mem=56.57 GB):  59%|█████▊    | 34/58 [00:02<00:00, 26.87it/s]Capturing num tokens (num_tokens=320 avail_mem=56.58 GB):  59%|█████▊    | 34/58 [00:02<00:00, 26.87it/s]Capturing num tokens (num_tokens=288 avail_mem=56.58 GB):  59%|█████▊    | 34/58 [00:02<00:00, 26.87it/s]Capturing num tokens (num_tokens=256 avail_mem=56.57 GB):  59%|█████▊    | 34/58 [00:02<00:00, 26.87it/s]

    Capturing num tokens (num_tokens=256 avail_mem=56.57 GB):  64%|██████▍   | 37/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=240 avail_mem=56.57 GB):  64%|██████▍   | 37/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=224 avail_mem=56.56 GB):  64%|██████▍   | 37/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=208 avail_mem=56.55 GB):  64%|██████▍   | 37/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=208 avail_mem=56.55 GB):  69%|██████▉   | 40/58 [00:02<00:00, 27.63it/s]Capturing num tokens (num_tokens=192 avail_mem=56.54 GB):  69%|██████▉   | 40/58 [00:02<00:00, 27.63it/s]Capturing num tokens (num_tokens=176 avail_mem=56.54 GB):  69%|██████▉   | 40/58 [00:02<00:00, 27.63it/s]Capturing num tokens (num_tokens=160 avail_mem=56.54 GB):  69%|██████▉   | 40/58 [00:02<00:00, 27.63it/s]Capturing num tokens (num_tokens=144 avail_mem=56.54 GB):  69%|██████▉   | 40/58 [00:02<00:00, 27.63it/s]

    Capturing num tokens (num_tokens=144 avail_mem=56.54 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.05it/s]Capturing num tokens (num_tokens=128 avail_mem=56.53 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.05it/s]Capturing num tokens (num_tokens=112 avail_mem=56.53 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.05it/s]Capturing num tokens (num_tokens=96 avail_mem=56.53 GB):  76%|███████▌  | 44/58 [00:03<00:00, 30.05it/s] Capturing num tokens (num_tokens=80 avail_mem=56.52 GB):  76%|███████▌  | 44/58 [00:03<00:00, 30.05it/s]Capturing num tokens (num_tokens=80 avail_mem=56.52 GB):  83%|████████▎ | 48/58 [00:03<00:00, 31.32it/s]Capturing num tokens (num_tokens=64 avail_mem=56.52 GB):  83%|████████▎ | 48/58 [00:03<00:00, 31.32it/s]Capturing num tokens (num_tokens=48 avail_mem=56.51 GB):  83%|████████▎ | 48/58 [00:03<00:00, 31.32it/s]Capturing num tokens (num_tokens=32 avail_mem=56.51 GB):  83%|████████▎ | 48/58 [00:03<00:00, 31.32it/s]

    Capturing num tokens (num_tokens=28 avail_mem=56.51 GB):  83%|████████▎ | 48/58 [00:03<00:00, 31.32it/s]Capturing num tokens (num_tokens=28 avail_mem=56.51 GB):  90%|████████▉ | 52/58 [00:03<00:00, 32.69it/s]Capturing num tokens (num_tokens=24 avail_mem=56.50 GB):  90%|████████▉ | 52/58 [00:03<00:00, 32.69it/s]Capturing num tokens (num_tokens=20 avail_mem=56.50 GB):  90%|████████▉ | 52/58 [00:03<00:00, 32.69it/s]Capturing num tokens (num_tokens=16 avail_mem=56.50 GB):  90%|████████▉ | 52/58 [00:03<00:00, 32.69it/s]Capturing num tokens (num_tokens=12 avail_mem=56.50 GB):  90%|████████▉ | 52/58 [00:03<00:00, 32.69it/s]Capturing num tokens (num_tokens=8 avail_mem=56.49 GB):  90%|████████▉ | 52/58 [00:03<00:00, 32.69it/s] Capturing num tokens (num_tokens=8 avail_mem=56.49 GB):  98%|█████████▊| 57/58 [00:03<00:00, 35.58it/s]Capturing num tokens (num_tokens=4 avail_mem=56.49 GB):  98%|█████████▊| 57/58 [00:03<00:00, 35.58it/s]Capturing num tokens (num_tokens=4 avail_mem=56.49 GB): 100%|██████████| 58/58 [00:03<00:00, 17.54it/s]


### Non-streaming Synchronous Generation


```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Hello, my name is
    Generated text:  Patricia. I'm from New York City. I have three children, Lucy, Andrew, and Jason. I've been a teacher for 25 years now. My favorite hobby is writing, and I write a lot of books and articles. I'm also a former world champion chess player. My husband is a professional basketball player. We have two sons, Andrew and Jason, and we are trying to have a baby. I'm really excited about having more children.
    Can we draw the following conclusion? Patricia is a chess player. The answer is yes.
    Yes, we can draw the conclusion that Patricia is a chess player based on the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man.
    A. 正确
    B. 错误
    Answer:
    
    A
    
    When the team leader conducts a safety inspection of the team, they should pay attention to ___.
    A. At least three times a month
    B. At least once a week
    C. At least once a month
    D. At least twice a week
    Answer:
    
    B
    
    Safety signs should be placed at the entrances and exits of production areas. ____
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    The main method for preparing and installing the DCS system is ____
    A. Fixed welding
    B. Manual welding
    C
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is the capital of which country?
    What is the answer?
    
    The capital of France is Paris, which is the capital of France and the country of its own name. France is a European country, and Paris is one of its most famous cities. Other famous cities in France include:
    
    1. Lyon (Alsace)
    2. Toulouse (Languedoc)
    3. Strasbourg (Alsace)
    4. Nice (Calvados)
    5. Bordeaux (Languedoc-Roussillon)
    6. Lille (Lorraine)
    7. Orleans (Loire-et-Orne)
    8. Montpellier
    ===============================
    Prompt: The future of AI is
    Generated text:  bound to be messy. At one end of the spectrum is the ambitious goal of creating completely autonomous and self-organizing systems, where AI will be able to function as a self-healing agent and adjust its structure to cope with the complexity of changing and unpredictable situations. On the other end of the spectrum are the already well-understood applications of AI, such as image recognition and natural language processing, for which AI is becoming increasingly crucial. But there is a whole spectrum in between. And for many AI experts, especially those in the field of robotics, the future of robotics lies in creating systems that are both self-healing and self-


### Streaming Synchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {
    "temperature": 0.2,
    "top_p": 0.9,
}

print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

for prompt in prompts:
    print(f"Prompt: {prompt}")
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print("Generated text:", merged_output)
    print()
```

    
    === Testing synchronous streaming generation with overlap removal ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic hub, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination, known for its rich history, art, and cuisine. The city is home to many famous French artists, writers, and musicians, and is a major center for the arts and culture industry. Paris is also known for its diverse and multicultural population, with a rich history of immigration and cultural exchange. The city is a major transportation hub, with many major highways and rail lines connecting
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are likely to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to become even more advanced in the future. AI-powered diagnostic tools will be able to analyze medical images and data more accurately and quickly than ever before, and AI-powered treatments will be able to personalize treatment plans based on individual patient data.
    
    2. Increased use of AI in finance: AI is already
    


### Non-streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous batch generation ===")


async def main():
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")


asyncio.run(main())
```

    
    === Testing asynchronous batch generation ===


    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text:  [Name], and I'm a software developer with a passion for [fill in the blank]. I love coding, problem-solving, and always looking for new ways to improve myself. I'm currently working on a project to create a [fill in the blank] using [fill in the blank]. And I'm always up for learning new technologies and languages, so I'm eager to expand my skillset. What's your name? [Name], and what's your profession? [Name], I hope you enjoy the company! [Name] - Meet [Name]!
    [Name] - Meet [Name]
    The team I'm part of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Despite its size, Paris is a bustling metropolis with a rich cultural history and an extensive array of attractions for visitors.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  currently transitioning from a purely computational to a more human-like approach. In the future, we can expect a wide range of technologies that aim to mimic human cognition and behavior, such as self-driving cars, virtual assistants, and even cognitive AI.
    
    Some potential areas for future AI include:
    
    1. Enhanced natural language processing: AI will continue to improve its ability to understand and respond to natural language, allowing for a more conversational and intuitive interface.
    
    2. Development of empathetic AI: AI will be able to understand and respond to emotions, emotions will become an essential part of AI systems, and emotional intelligence will be a key capability.
    
    3.


### Streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


asyncio.run(main())
```

    
    === Testing asynchronous streaming generation (no repeats) ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text: 

     [

    name

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    occupation

    ].

     Throughout

     my

     career

    ,

     I

    've

     always

     been

     [

    career

     goal

    ],

     and

     [

    accom

    pl

    ishment

    ]

     was

     the

     [

    achievement

    ].

     I

    'm

     a

     [

    person

    ality

     type

    ],

     and

     I

    'm

     [

    current

     status

    ].

     I

    'm

     passionate

     about

     [

    reason

     for

     passion

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    goal

    ].

     I

     enjoy

     [

    activity

    ],

     and

     I

    'm

     always

     eager

     to

     [

    future

     goal

    ].

     I

     believe

     [

    mot

    ivation

    ],

     and

     I

    'm

     always

     determined

     to

     [

    next

     challenge

    ].

     What

    's

     your

     name

    ,

     and

     what

    's

     your

     occupation

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

     and

     home

     to

     the

     headquarters

     of

     many

     international

     organizations

     such

     as

     the

     European

     Commission

    ,

     the

     European

     Central

     Bank

    ,

     the

     European

     Court

     of

     Justice

    ,

     and

     the

     European

     Union

    .

     It

     is

     also

     the

     heart

     of

     the

     French

     economy

    ,

     known

     for

     its

     bustling

     streets

    ,

     iconic

     landmarks

    ,

     and

     vibrant

     culture

    .

     With

     a

     population

     of

     over

     

    7

     million

     people

    ,

     Paris

     is

     a

     cultural

     and

     commercial

     center

     that

     plays

     a

     significant

     role

     in

     French

     identity

     and

     politics

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     world

    -ren

    owned

     museums

    ,

     art

     galleries

    ,

     and

     fashion

     scene

    .

     Paris

     is

     considered

     one

     of

     the

     most

     beautiful

     cities

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     influenced

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     responsible

     AI

    :

     As

     concerns

     about

     AI

    's

     potential

     to

     harm

     individuals

     and

     society

     grow

    ,

     there

     will

     be

     increased

     pressure

     on

     developers

     and

     researchers

     to

     focus

     on

     creating

     AI

     that

     is

     ethical

     and

     responsible

    .

     This

     includes

     creating

     AI

     systems

     that

     are

     transparent

    ,

     accountable

    ,

     and

     accountable

     for

     their

     actions

    .
    


    2

    .

     Emer

    gence

     of

     new

     AI

     technologies

    :

     As

     we

     learn

     more

     about

     the

     nature

     and

     workings

     of

     AI

    ,

     there

     will

     be

     new

     technologies

     that

     emerge

     that

     can

     help

     us create

     more

     powerful

    ,

     efficient

    ,

     and

     effective

     AI

    .

     These

     could

     include

     new

     algorithms

    ,

     machine

     learning

     techniques

    ,

     and other

     advanced

     technologies that

     can

    



```python
llm.shutdown()
```
