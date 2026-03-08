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

    [2026-03-08 13:38:12] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-08 13:38:12] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-08 13:38:12] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-08 13:38:14] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-08 13:38:14] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 13:38:14] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=680447460, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.64it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.64it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.51 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.51 GB):   5%|▌         | 1/20 [00:00<00:03,  5.27it/s]Capturing batches (bs=120 avail_mem=58.41 GB):   5%|▌         | 1/20 [00:00<00:03,  5.27it/s]

    Capturing batches (bs=112 avail_mem=58.41 GB):   5%|▌         | 1/20 [00:00<00:03,  5.27it/s]Capturing batches (bs=104 avail_mem=58.41 GB):   5%|▌         | 1/20 [00:00<00:03,  5.27it/s]Capturing batches (bs=104 avail_mem=58.41 GB):  20%|██        | 4/20 [00:00<00:01, 15.19it/s]Capturing batches (bs=96 avail_mem=58.41 GB):  20%|██        | 4/20 [00:00<00:01, 15.19it/s] Capturing batches (bs=88 avail_mem=58.41 GB):  20%|██        | 4/20 [00:00<00:01, 15.19it/s]Capturing batches (bs=80 avail_mem=58.41 GB):  20%|██        | 4/20 [00:00<00:01, 15.19it/s]Capturing batches (bs=80 avail_mem=58.41 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.41it/s]Capturing batches (bs=72 avail_mem=58.41 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.41it/s]

    Capturing batches (bs=64 avail_mem=58.41 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.41it/s]Capturing batches (bs=56 avail_mem=58.40 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.41it/s]Capturing batches (bs=56 avail_mem=58.40 GB):  50%|█████     | 10/20 [00:00<00:00, 22.71it/s]Capturing batches (bs=48 avail_mem=58.40 GB):  50%|█████     | 10/20 [00:00<00:00, 22.71it/s]Capturing batches (bs=40 avail_mem=58.40 GB):  50%|█████     | 10/20 [00:00<00:00, 22.71it/s]Capturing batches (bs=32 avail_mem=58.40 GB):  50%|█████     | 10/20 [00:00<00:00, 22.71it/s]Capturing batches (bs=32 avail_mem=58.40 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.93it/s]Capturing batches (bs=24 avail_mem=58.40 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.93it/s]

    Capturing batches (bs=16 avail_mem=58.40 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.93it/s]Capturing batches (bs=12 avail_mem=58.40 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.93it/s]Capturing batches (bs=12 avail_mem=58.40 GB):  80%|████████  | 16/20 [00:00<00:00, 22.60it/s]Capturing batches (bs=8 avail_mem=58.40 GB):  80%|████████  | 16/20 [00:00<00:00, 22.60it/s] Capturing batches (bs=4 avail_mem=58.39 GB):  80%|████████  | 16/20 [00:00<00:00, 22.60it/s]Capturing batches (bs=2 avail_mem=58.39 GB):  80%|████████  | 16/20 [00:00<00:00, 22.60it/s]Capturing batches (bs=1 avail_mem=58.39 GB):  80%|████████  | 16/20 [00:00<00:00, 22.60it/s]

    Capturing batches (bs=1 avail_mem=58.39 GB): 100%|██████████| 20/20 [00:00<00:00, 27.10it/s]Capturing batches (bs=1 avail_mem=58.39 GB): 100%|██████████| 20/20 [00:00<00:00, 22.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:56,  1.00s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:56,  1.00s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:56,  1.00s/it]

    Compiling num tokens (num_tokens=6144):   3%|▎         | 2/58 [00:02<00:56,  1.00s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:16,  3.18it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:16,  3.18it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:16,  3.18it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:10,  4.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:10,  4.78it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:10,  4.78it/s]

    Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:10,  4.78it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:10,  4.78it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:05,  8.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:05,  8.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:05,  8.98it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:05,  8.98it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:05,  8.98it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:02<00:03, 13.48it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:02<00:03, 13.48it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:02<00:03, 13.48it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:02<00:03, 13.48it/s]

    Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:02<00:03, 13.48it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:02<00:03, 13.48it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:01, 19.42it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:01, 19.42it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:01, 19.42it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:01, 19.42it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:01, 19.42it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:01, 19.42it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]

    Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:00, 31.42it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 38.30it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 44.70it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 54.94it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 54.94it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 54.94it/s]

    Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 54.94it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 54.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.03 GB):   3%|▎         | 2/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.03 GB):   3%|▎         | 2/58 [00:00<00:04, 11.89it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.02 GB):   3%|▎         | 2/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.02 GB):   7%|▋         | 4/58 [00:00<00:04, 13.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.02 GB):   7%|▋         | 4/58 [00:00<00:04, 13.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.97 GB):   7%|▋         | 4/58 [00:00<00:04, 13.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.97 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.95 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.95 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.99 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.99 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.99 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.98 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.97 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.20it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.97 GB):  21%|██        | 12/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.97 GB):  21%|██        | 12/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.96 GB):  21%|██        | 12/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.95 GB):  21%|██        | 12/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.95 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.95 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.94 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.93 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.64it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=57.91 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.91 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.92 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.90 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=960 avail_mem=57.91 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.77it/s] Capturing num tokens (num_tokens=896 avail_mem=57.90 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=896 avail_mem=57.90 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.31it/s]Capturing num tokens (num_tokens=832 avail_mem=57.89 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.31it/s]Capturing num tokens (num_tokens=768 avail_mem=57.88 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.31it/s]

    Capturing num tokens (num_tokens=704 avail_mem=57.88 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.31it/s]Capturing num tokens (num_tokens=640 avail_mem=57.87 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.31it/s]Capturing num tokens (num_tokens=640 avail_mem=57.87 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.65it/s]Capturing num tokens (num_tokens=576 avail_mem=57.88 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.65it/s]Capturing num tokens (num_tokens=512 avail_mem=57.86 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.65it/s]Capturing num tokens (num_tokens=480 avail_mem=57.88 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.65it/s]Capturing num tokens (num_tokens=448 avail_mem=57.87 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.65it/s]Capturing num tokens (num_tokens=448 avail_mem=57.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=416 avail_mem=57.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.84it/s]

    Capturing num tokens (num_tokens=384 avail_mem=57.86 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=352 avail_mem=57.85 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=320 avail_mem=57.84 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=320 avail_mem=57.84 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=288 avail_mem=57.83 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=256 avail_mem=57.81 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=240 avail_mem=57.82 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=224 avail_mem=57.81 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=224 avail_mem=57.81 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=208 avail_mem=57.81 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.72it/s]

    Capturing num tokens (num_tokens=192 avail_mem=57.80 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=176 avail_mem=57.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=160 avail_mem=57.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=144 avail_mem=57.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=144 avail_mem=57.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.38it/s]Capturing num tokens (num_tokens=128 avail_mem=57.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.38it/s]Capturing num tokens (num_tokens=112 avail_mem=57.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.38it/s]Capturing num tokens (num_tokens=96 avail_mem=57.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.38it/s] Capturing num tokens (num_tokens=80 avail_mem=57.77 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.38it/s]Capturing num tokens (num_tokens=64 avail_mem=57.77 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.38it/s]Capturing num tokens (num_tokens=64 avail_mem=57.77 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=48 avail_mem=57.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.80it/s]

    Capturing num tokens (num_tokens=32 avail_mem=57.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=28 avail_mem=57.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=24 avail_mem=57.75 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=20 avail_mem=57.75 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=20 avail_mem=57.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=16 avail_mem=57.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=12 avail_mem=57.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=8 avail_mem=57.74 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.30it/s] Capturing num tokens (num_tokens=4 avail_mem=57.74 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=4 avail_mem=57.74 GB): 100%|██████████| 58/58 [00:01<00:00, 30.29it/s]


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
    Generated text:  Kieran and I’m a creative writer and editor living in Cambridge, England. I write about 500 words a day, so I have lots of spare time to write about a lot of different subjects. I’m keen to help you learn more about the world around you, and to make your writing more accessible and enjoyable to read. As a professional writer, I focus on writing a wide variety of articles, both for print and online. I'm always on the lookout for new ideas and exciting stories to share, whether it's a new story to tell, a new sub genre, or a new style of writing. I believe
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular member of the media. His role is to represent the country and has a wide variety of responsibilities. He must attend press conferences, issue press statements, answer questions from reporters, and make pronouncements that reflect the image of the United States. The president has a wide range of powers and duties, and these responsibilities are not limited to just speaking. He is also responsible for managing the nation’s public relations and managing the media. How is the president responsible for his duties and responsibilities in the media? The president is responsible for his duties and responsibilities in the media through his role as a spokesperson for the country. As a public figure,
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which region?
    The capital of France is Paris. It is located in the south of the country, in the Île de France region.
    
    To make the task of creating an answer more complex, suppose you need to provide the coordinates of Paris, but you are not allowed to use any conversion functions (like `convert`, `round`, etc.) in your answer. Please provide the coordinates of Paris as a list of four floating-point numbers. The coordinates should be in the format [latitude, longitude]. To make the task more challenging, you should provide the coordinates of Paris as a string in the format "latitude,longitude". Additionally
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is complicated and challenging, especially for business leaders. But these days, businesses are increasingly adopting artificial intelligence solutions to overcome a myriad of challenges. For example, AI systems are used for fraud detection, chatbots, customer service, and customer retention. This article will provide a solid understanding of how AI can be beneficial to businesses.
    Today, businesses are utilizing AI to enhance their customers’ experience by providing personalized service. This enables businesses to close the loop on customer service, and it ensures that customers get the experience they deserve. For example, the customer service system helps businesses react to customer queries in real-time, reducing frustration and


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast. I'm passionate about [What interests you about your field of interest]. I'm always looking for new challenges and opportunities to learn and grow. I'm always eager to share my knowledge and experiences with others. I'm a [What do you enjoy doing in your free time]? I enjoy [What's something you're passionate about?]. I'm always looking for new ways to improve myself and expand my horizons. I'm a [What do you like to do in your free time?]. I enjoy [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Parliament building. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. It is a popular tourist destination, with millions of visitors annually. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we are likely to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for innovation and creativity.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy violations. This will require a more nuanced approach to
    


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
    Generated text:  [Name] and I am a [age] year old [occupation] who is [career goal]. I am confident in my abilities, always ready to step up and tackle challenges. I'm driven, hardworking, and always seeking to learn and grow in my field. I enjoy being creative, and my creativity often leads me to my best work. I believe that my unique perspective and skills can make a positive impact in my field, and I am excited to share my experiences with you. How can I help you today? And what can you tell me about your career goals? [Name] is a [career goal] who is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly known as "La Grande Bataille," and is one of the most visited cities in the world, attracting millions of tourists each year. Paris is famous for its museums, palaces, and fashion industry, and is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is known for its rich history and culture, as well as its world-renowned cuisine. Paris has a diverse population, with a population density of approximately 2,700 people per square mile. It is the largest city in France by population, and the seat of government for the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and will likely involve a range of potential trends, including:
    
    1. Increased autonomy and decision-making: AI systems will become more autonomous and able to make decisions on their own, without human intervention. This could lead to more efficient and cost-effective systems for tasks such as navigation and task assignment.
    
    2. Integration with human decision-making: AI systems will become more integrated with human decision-making processes, leading to a more balanced and informed decision-making approach. This could help to reduce bias and discrimination in decision-making.
    
    3. Greater focus on ethical and social implications: AI will likely become more focused on ethical and social implications, such as the


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

     am

     a

     [

    job

     title

    ]

     at

     [

    company

    ].

     I

     am

     enthusiastic

     about

     [

    objective

     of

     the

     role

    ],

     and

     I

     am

     looking

     for

     a

     job

     that

     fits

     my

     goals

     and

     values

    .

     Looking

     forward

     to

     hearing

     from

     you

    !

     [

    Job

     Title

    ]

     at

     [

    Company

    ]

     I

    'm

     excited

     to

     get

     started

    !

     Can

    't

     wait

     to

     meet

     you

    !

     Let

     me

     know

     if

     you

     have

     any

     questions

    .

     [

    Name

    ]

     [

    Phone

    ]

     [

    Email

    ]

     [

    LinkedIn

    ]

     [

    Company

     Website

    ]

     [

    Company

     Description

    ]

     [

    Company

     Job

     Description

    ]

     [

    Company

     Culture

    ]

     [

    Company

     Values

    ]

     [

    Company

     Mission

    ]

     [

    Company

     Vision

    ]

     [

    Company

     Philosophy

    ]

     [

    Company

     History

    ]

     [

    Company

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

     and

     the

     most

     visited

     tourist

     destination

     in

     Europe

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     culinary

     and

     cultural

     traditions

    .

     Paris

     also

     has

     a

     rich

     history

     and a

     vibrant

     culture that

     is

     celebrated

     globally

    .

     It

     is

     a city

     of

     contrasts

    ,

     combining

     traditional

     French

     architecture

     with

     modern

     urban

     design

    .

     Paris

     is

     one

     of

     the

     most

     important

     cities

     in

     the

     world

     and

     a

     symbol

     of

     France

    's

     cultural

     and

     political

     significance

    .

     It

     is

     a

     popular

     tourist

     destination

     for

     international

     visitors

     and

     a

     cultural

     hub

     for

     artists

    ,

     writers

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     trends

     and

     developments

    ,

     some

     of

     which

     are

     already

     beginning

     to

     take

     shape

    .

     Some

     of

     the

     most

     likely

     trends

     include

    :
    


    1

    .

     Increased

     efficiency

     and

     precision

     in

     AI

    :

     As

     AI

     becomes

     more

     powerful

     and

     capable

    ,

     we

     may

     see

     an

     increase

     in

     its

     ability

     to

     perform

     tasks

     more

     efficiently

     and

     accurately

    .

     This

     could

     lead

     to

     significant

     improvements

     in

     productivity

     and

     efficiency

     across

     a

     wide

     range

     of

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    .
    


    2

    .

     Greater

     reliance

     on

     AI

    :

     AI

     is

     becoming

     increasingly

     integrated

     into

     everyday

     life

    ,

     with

     applications

     ranging

     from

     self

    -driving

     cars

     to

     virtual

     assistants

     to

     medical

     imaging

    .

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

    



```python
llm.shutdown()
```
