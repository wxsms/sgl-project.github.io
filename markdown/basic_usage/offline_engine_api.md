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
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    [2026-03-10 19:55:27] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-10 19:55:27] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-10 19:55:27] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-10 19:55:29] INFO server_args.py:2133: Attention backend not specified. Use fa3 backend by default.


    [2026-03-10 19:55:29] INFO server_args.py:3246: Set soft_watchdog_timeout since in CI


    [2026-03-10 19:55:29] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=813356137, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.25it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.24it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:05,  2.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:05,  2.20s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:54,  1.03it/s]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:54,  1.03it/s]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:54,  1.03it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:21,  2.51it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:21,  2.51it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:21,  2.51it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:21,  2.51it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:10,  5.06it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:10,  5.06it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:10,  5.06it/s]

    Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:10,  5.06it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:05,  8.05it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:05,  8.05it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:05,  8.05it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:05,  8.05it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:05,  8.05it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:02<00:03, 12.46it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:02<00:03, 12.46it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:02<00:03, 12.46it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:02<00:03, 12.46it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:02<00:03, 12.46it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:02<00:03, 12.46it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 18.07it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 18.07it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 18.07it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 18.07it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 18.07it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 18.07it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 23.37it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 23.37it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 23.37it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 23.37it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 23.37it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 26.66it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 26.66it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 26.66it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 26.66it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 26.66it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 26.66it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 31.59it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 31.59it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 31.59it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 31.59it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 31.59it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 31.59it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 38.04it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 38.04it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 41.36it/s]

    Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 41.36it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 41.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.59 GB):   2%|▏         | 1/58 [00:00<00:09,  6.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.06 GB):   2%|▏         | 1/58 [00:00<00:09,  6.22it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.06 GB):   3%|▎         | 2/58 [00:00<00:10,  5.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.05 GB):   3%|▎         | 2/58 [00:00<00:10,  5.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.05 GB):   5%|▌         | 3/58 [00:00<00:09,  5.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.04 GB):   5%|▌         | 3/58 [00:00<00:09,  5.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.04 GB):   7%|▋         | 4/58 [00:00<00:09,  5.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.05 GB):   7%|▋         | 4/58 [00:00<00:09,  5.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.05 GB):   9%|▊         | 5/58 [00:00<00:09,  5.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.04 GB):   9%|▊         | 5/58 [00:00<00:09,  5.79it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.04 GB):  10%|█         | 6/58 [00:01<00:08,  6.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.02 GB):  10%|█         | 6/58 [00:01<00:08,  6.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.02 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.02 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.09it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.02 GB):  14%|█▍        | 8/58 [00:01<00:08,  6.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.02 GB):  14%|█▍        | 8/58 [00:01<00:08,  6.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.02 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.01 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.28it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.01 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.00 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.00 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.00 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.65it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.00 GB):  21%|██        | 12/58 [00:01<00:06,  6.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.00 GB):  21%|██        | 12/58 [00:01<00:06,  6.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.00 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.00 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.95it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=38.00 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=37.99 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=37.99 GB):  26%|██▌       | 15/58 [00:02<00:06,  7.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=37.99 GB):  26%|██▌       | 15/58 [00:02<00:06,  7.09it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=37.99 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=37.98 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.14it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=37.98 GB):  29%|██▉       | 17/58 [00:02<00:06,  5.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=37.98 GB):  29%|██▉       | 17/58 [00:02<00:06,  5.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=37.98 GB):  31%|███       | 18/58 [00:02<00:06,  6.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=37.98 GB):  31%|███       | 18/58 [00:02<00:06,  6.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=37.97 GB):  31%|███       | 18/58 [00:02<00:06,  6.65it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=37.97 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=37.95 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.75it/s]Capturing num tokens (num_tokens=960 avail_mem=37.97 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.75it/s] Capturing num tokens (num_tokens=960 avail_mem=37.97 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.36it/s]Capturing num tokens (num_tokens=896 avail_mem=37.96 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.36it/s]

    Capturing num tokens (num_tokens=832 avail_mem=37.96 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.36it/s]Capturing num tokens (num_tokens=832 avail_mem=37.96 GB):  41%|████▏     | 24/58 [00:03<00:02, 11.53it/s]Capturing num tokens (num_tokens=768 avail_mem=37.96 GB):  41%|████▏     | 24/58 [00:03<00:02, 11.53it/s]Capturing num tokens (num_tokens=704 avail_mem=37.95 GB):  41%|████▏     | 24/58 [00:03<00:02, 11.53it/s]

    Capturing num tokens (num_tokens=704 avail_mem=37.95 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.00it/s]Capturing num tokens (num_tokens=640 avail_mem=37.95 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.00it/s]Capturing num tokens (num_tokens=576 avail_mem=37.95 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.00it/s]

    Capturing num tokens (num_tokens=576 avail_mem=37.95 GB):  48%|████▊     | 28/58 [00:03<00:03,  9.89it/s]Capturing num tokens (num_tokens=512 avail_mem=37.94 GB):  48%|████▊     | 28/58 [00:03<00:03,  9.89it/s]Capturing num tokens (num_tokens=480 avail_mem=37.95 GB):  48%|████▊     | 28/58 [00:03<00:03,  9.89it/s]

    Capturing num tokens (num_tokens=480 avail_mem=37.95 GB):  52%|█████▏    | 30/58 [00:03<00:02,  9.73it/s]Capturing num tokens (num_tokens=448 avail_mem=37.95 GB):  52%|█████▏    | 30/58 [00:03<00:02,  9.73it/s]Capturing num tokens (num_tokens=416 avail_mem=37.95 GB):  52%|█████▏    | 30/58 [00:03<00:02,  9.73it/s]

    Capturing num tokens (num_tokens=416 avail_mem=37.95 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.70it/s]Capturing num tokens (num_tokens=384 avail_mem=37.95 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.70it/s]Capturing num tokens (num_tokens=352 avail_mem=37.94 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.70it/s]Capturing num tokens (num_tokens=352 avail_mem=37.94 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.84it/s]Capturing num tokens (num_tokens=320 avail_mem=37.94 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.84it/s]

    Capturing num tokens (num_tokens=288 avail_mem=37.93 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.84it/s]Capturing num tokens (num_tokens=288 avail_mem=37.93 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.50it/s]Capturing num tokens (num_tokens=256 avail_mem=37.93 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.50it/s]Capturing num tokens (num_tokens=240 avail_mem=37.93 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.50it/s]

    Capturing num tokens (num_tokens=240 avail_mem=37.93 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.31it/s]Capturing num tokens (num_tokens=224 avail_mem=37.93 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.31it/s]Capturing num tokens (num_tokens=208 avail_mem=37.92 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.31it/s]Capturing num tokens (num_tokens=208 avail_mem=37.92 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.83it/s]Capturing num tokens (num_tokens=192 avail_mem=37.92 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.83it/s]

    Capturing num tokens (num_tokens=176 avail_mem=37.92 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.83it/s]Capturing num tokens (num_tokens=176 avail_mem=37.92 GB):  72%|███████▏  | 42/58 [00:04<00:01, 12.86it/s]Capturing num tokens (num_tokens=160 avail_mem=37.91 GB):  72%|███████▏  | 42/58 [00:04<00:01, 12.86it/s]Capturing num tokens (num_tokens=144 avail_mem=37.91 GB):  72%|███████▏  | 42/58 [00:04<00:01, 12.86it/s]

    Capturing num tokens (num_tokens=144 avail_mem=37.91 GB):  76%|███████▌  | 44/58 [00:05<00:01, 11.48it/s]Capturing num tokens (num_tokens=128 avail_mem=37.91 GB):  76%|███████▌  | 44/58 [00:05<00:01, 11.48it/s]Capturing num tokens (num_tokens=112 avail_mem=37.90 GB):  76%|███████▌  | 44/58 [00:05<00:01, 11.48it/s]Capturing num tokens (num_tokens=112 avail_mem=37.90 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.12it/s]Capturing num tokens (num_tokens=96 avail_mem=37.90 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.12it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=37.90 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.12it/s]

    Capturing num tokens (num_tokens=80 avail_mem=37.90 GB):  83%|████████▎ | 48/58 [00:05<00:01,  9.54it/s]Capturing num tokens (num_tokens=64 avail_mem=58.21 GB):  83%|████████▎ | 48/58 [00:05<00:01,  9.54it/s]Capturing num tokens (num_tokens=48 avail_mem=58.21 GB):  83%|████████▎ | 48/58 [00:05<00:01,  9.54it/s]Capturing num tokens (num_tokens=48 avail_mem=58.21 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.42it/s]Capturing num tokens (num_tokens=32 avail_mem=58.20 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.42it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.20 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.42it/s]Capturing num tokens (num_tokens=28 avail_mem=58.20 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.39it/s]Capturing num tokens (num_tokens=24 avail_mem=58.19 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.39it/s]Capturing num tokens (num_tokens=20 avail_mem=58.19 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.39it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.19 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.19it/s]Capturing num tokens (num_tokens=16 avail_mem=58.19 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.19it/s]Capturing num tokens (num_tokens=12 avail_mem=58.18 GB):  93%|█████████▎| 54/58 [00:06<00:00, 12.19it/s]Capturing num tokens (num_tokens=12 avail_mem=58.18 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.79it/s]Capturing num tokens (num_tokens=8 avail_mem=58.18 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.79it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.18 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.79it/s]Capturing num tokens (num_tokens=4 avail_mem=58.18 GB): 100%|██████████| 58/58 [00:06<00:00, 13.27it/s]Capturing num tokens (num_tokens=4 avail_mem=58.18 GB): 100%|██████████| 58/58 [00:06<00:00,  9.34it/s]


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
    Generated text:  Rongzhuo. I am from Shanghai, and I have been studying in this university since high school. My favorite subject is math, and I am very good at it. However, I have a tendency to procrastinate and get distracted during class, which makes my grades seem lower than expected. To change my poor study habits, I need to find a solution that involves both learning and practical skills. Can you suggest some methods to improve my study habits?
    Sure, I can help you with that! Here are some methods that you can try to improve your study habits:
    
    1. Set a regular study schedule: Create a study schedule
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. What does the word "president" mean in this sentence?
    A president is a person who holds the highest position in the government of a country. The president is the head of the executive branch of the government, which makes important decisions and manages the country's affairs. The word "president" in this sentence is used to refer to the person in charge of the country, or to the person who holds that position. It can also be used to refer to the person who is in charge of the country, but who may not be the president. In this case, the president is the person
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Brussels
    C. Lausanne
    D. Lyon
    Answer: A
    
    A company's capital structure includes common stock, preferred stock, and retained earnings, with the proportion of each component being 40%, 30%, and 30%, respectively. In the current financial year, the company needs to repurchase its own shares to maintain its capital structure. Which of the following practices by the company is incorrect? 
    A. First, it needs to calculate the par value of the shares and the par value of the preferred stock. 
    B. After calculating the par value of the
    ===============================
    Prompt: The future of AI is
    Generated text:  far from certain, and while it's clear that AI will play an ever more important role in shaping our world, we still have a lot to learn before we can fully harness its full power.
    As we look to the future, what are the most pressing challenges facing AI research and development, and how can we address them?
    To address the most pressing challenges facing AI research and development, here are some key areas to consider:
    1. Bias and Fairness: AI systems can perpetuate biases and discrimination if they are trained on biased data. To address this, researchers need to develop fair and unbiased algorithms that can be trained on diverse and representative


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I enjoy [reason for interest in the industry]. I'm a [reason for interest in the industry] and I'm always looking for [reason for interest in the industry]. I'm a [reason for interest in the industry] and I'm always looking for [reason for interest in the industry]. I'm a [reason for interest in the industry] and I'm always looking for [reason for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich cultural heritage and a vibrant nightlife. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and a major economic center, with a rich history and culture that continues to inspire
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in shaping society and the economy, with implications for everything from job creation and economic growth to social justice and ethical considerations. As AI continues to evolve, it is likely to have a profound impact on the way we live and work, and it is important to carefully consider the potential benefits and risks of these developments.
    


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
    Generated text:  [Name], and I'm a [Role] to the [Company]. I recently graduated from [University or School] and have [Summary of Achievements]. I enjoy [What I Do/What I Love/What I Want to Do/What I Wish to Do]. I am passionate about [Why I love this role]. I am a [Role] who can [Strengths or Traits]. I have been [What I've Done/What I've Achieved/What I Have Learned]. I am [What I Want to Be/What I Want to Do/What I Want to Work On/What I Want to Learn].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital of France and the largest city in the country. It is the seat of government, legislature, and executive branch of the French government. It is also known as "La Ville". Paris is a cosmopolitan and diverse city with a rich cultural heritage, including many museums, art galleries, and theaters. The city is also known for its delicious food, elegant fashion, and beautiful architecture. Paris is a popular tourist destination and attracts millions of visitors each year. It is considered a cultural and artistic capital of the world. The city has a rich history dating back to the Roman era, and it is home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but there are several potential trends that could shape the landscape of the field.
    
    One of the most significant trends is the increasing use of AI in autonomous vehicles. As autonomous vehicles become more common, they will continue to play a larger role in our daily lives. However, this will also create new challenges, such as the need to develop safe, efficient, and reliable AI systems that can operate in a wide range of environments.
    
    Another trend is the increased integration of AI in healthcare. AI can help doctors and researchers to better understand and predict the health outcomes of patients, as well as to develop new treatments and therapies. This could lead to


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

    Name

    ],

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

     My

     love

     for

     [

    job

     title

    ]

     has

     led

     me

     to

     travel

     the

     world

     and

     learn

     about

     various

     cultures

     and

     people

    .

     I

    'm

     passionate

     about

     sharing

     my

     experiences

     and

     insights

     with

     the

     world

    ,

     and

     I

    'm

     excited

     to

     contribute

     to

     the

     communities

     I

    'm

     a

     part

     of

    .

     I

    'm

     always

     open

     to

     new

     ideas

     and

     perspectives

    ,

     and

     I

     strive

     to

     make

     a

     positive

     impact

     on

     the

     world

     around

     me

    .

     What

    's

     your

     name

     and

     what

     kind

     of

     job

     do

     you

     have

    ?

     [

    Name

    ]

     is

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ],

     and

     I

    'm

     excited

     to

     share

     my

     experiences

     and

     insights

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     the

     country

    .

     Its

     architecture

    ,

     cuisine

    ,

     and

     festivals

     are

     among

     the

     world

    's

     most

     notable

    .

     Its

     nickname

     is

     "

    City

     of

     Light

    ."

     Located

     in

     the

     north

     of

     the

     country

    ,

     it

     is

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     home

     to

     the

     French

     Parliament

     and

     the

     University

     of

     Paris

    .

     Paris

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     hosts

     the

     most

     famous

     opera

     in

     the

     world

    ,

     the

     opera

     of

     the

     same

     name

    .

     It

     is

     also

     the

     "

    City

     of

     the

     Stars

    "

     and

     hosts

     the

     most

     famous

     night

    clubs

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     several

     trends

     that

     are

     currently

     being

     explored

    ,

     discussed

    ,

     and

     researched

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     increasing

     use

     of

     AI

     in

     healthcare

    ,

     we

     can

     expect

     to

     see

     a

     significant

     increase

     in

     the

     use

     of

     AI

     to

     diagnose

     and

     treat

     diseases

    ,

     as

     well

     as

     to

     improve

     patient

     outcomes

    .

     AI

     can

     be

     used

     to

     analyze

     large

     amounts

     of

     medical

     data

    ,

     identify

     patterns

    ,

     and

     make

     predictions

     about

     potential

     medical

     conditions

    .
    


    2

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     can

     expect

     to

     see

     a

     greater

     integration

     of

     AI

     into

     everyday

     life

    .

     This

     can

     include

     the

     use

     of

     AI

     in

     transportation

    ,

     agriculture

    ,

     manufacturing

    



```python
llm.shutdown()
```
