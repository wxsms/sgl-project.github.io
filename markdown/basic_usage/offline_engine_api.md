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

    [2026-03-05 03:15:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-05 03:15:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-05 03:15:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-05 03:15:41] INFO server_args.py:1987: Attention backend not specified. Use fa3 backend by default.


    [2026-03-05 03:15:41] INFO server_args.py:3078: Set soft_watchdog_timeout since in CI


    [2026-03-05 03:15:41] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=166472015, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.47it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.47it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   5%|▌         | 1/20 [00:00<00:03,  5.11it/s]Capturing batches (bs=120 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.11it/s]

    Capturing batches (bs=112 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.11it/s]Capturing batches (bs=104 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.11it/s]Capturing batches (bs=104 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 14.97it/s]Capturing batches (bs=96 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 14.97it/s] Capturing batches (bs=88 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 14.97it/s]Capturing batches (bs=80 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 14.97it/s]Capturing batches (bs=80 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.85it/s]Capturing batches (bs=72 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.85it/s]

    Capturing batches (bs=64 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.85it/s]Capturing batches (bs=56 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.85it/s]Capturing batches (bs=56 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.41it/s]Capturing batches (bs=48 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.41it/s]Capturing batches (bs=40 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.41it/s]Capturing batches (bs=32 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.41it/s]Capturing batches (bs=32 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.92it/s]Capturing batches (bs=24 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.92it/s]

    Capturing batches (bs=16 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.92it/s]Capturing batches (bs=12 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.92it/s]Capturing batches (bs=12 avail_mem=58.49 GB):  80%|████████  | 16/20 [00:00<00:00, 20.76it/s]Capturing batches (bs=8 avail_mem=58.49 GB):  80%|████████  | 16/20 [00:00<00:00, 20.76it/s] Capturing batches (bs=4 avail_mem=58.48 GB):  80%|████████  | 16/20 [00:00<00:00, 20.76it/s]

    Capturing batches (bs=2 avail_mem=58.48 GB):  80%|████████  | 16/20 [00:00<00:00, 20.76it/s]Capturing batches (bs=1 avail_mem=58.48 GB):  80%|████████  | 16/20 [00:00<00:00, 20.76it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 20/20 [00:00<00:00, 24.21it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 20/20 [00:00<00:00, 21.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:13,  2.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:13,  2.35s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:58,  1.04s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:58,  1.04s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:58,  1.04s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.33it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.33it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.33it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.96it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.96it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.96it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.96it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.79it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.79it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.79it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.79it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:04,  9.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:04,  9.99it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:04,  9.99it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:02, 14.37it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:02, 14.37it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:02, 14.37it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:02, 14.37it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:02, 14.37it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.70it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.70it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.70it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.70it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.70it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 23.03it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 23.03it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 23.03it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 23.03it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 23.03it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 26.86it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 26.86it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 26.86it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 26.86it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 26.86it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 26.86it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 31.08it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 31.08it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 31.08it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 31.08it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 31.08it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 31.08it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 33.83it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 33.83it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 33.83it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 33.83it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 33.83it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 33.83it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 43.47it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 47.60it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 47.60it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 47.60it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 47.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.18 GB):   2%|▏         | 1/58 [00:00<00:08,  6.85it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.14 GB):   2%|▏         | 1/58 [00:00<00:08,  6.85it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.14 GB):   3%|▎         | 2/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.14 GB):   3%|▎         | 2/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.14 GB):   5%|▌         | 3/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.14 GB):   5%|▌         | 3/58 [00:00<00:07,  7.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.14 GB):   7%|▋         | 4/58 [00:00<00:07,  7.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.14 GB):   7%|▋         | 4/58 [00:00<00:07,  7.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.14 GB):   9%|▊         | 5/58 [00:00<00:06,  7.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.14 GB):   9%|▊         | 5/58 [00:00<00:06,  7.79it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.14 GB):  10%|█         | 6/58 [00:00<00:06,  8.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.14 GB):  10%|█         | 6/58 [00:00<00:06,  8.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.14 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.14 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.14 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.13 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.13 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.81it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.13 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.12 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.35it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.12 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.12 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.12 GB):  21%|██        | 12/58 [00:01<00:06,  6.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.12 GB):  21%|██        | 12/58 [00:01<00:06,  6.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.12 GB):  21%|██        | 12/58 [00:01<00:06,  6.96it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.12 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.11 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.11 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.11 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.10 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.31it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.10 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.10 GB):  31%|███       | 18/58 [00:02<00:03, 10.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.10 GB):  31%|███       | 18/58 [00:02<00:03, 10.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.09 GB):  31%|███       | 18/58 [00:02<00:03, 10.11it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.09 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.08 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.71it/s]Capturing num tokens (num_tokens=960 avail_mem=58.09 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.71it/s] Capturing num tokens (num_tokens=960 avail_mem=58.09 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.29it/s]Capturing num tokens (num_tokens=896 avail_mem=58.09 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.29it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.08 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.29it/s]Capturing num tokens (num_tokens=832 avail_mem=58.08 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.63it/s]Capturing num tokens (num_tokens=768 avail_mem=58.08 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.63it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.08 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.63it/s]Capturing num tokens (num_tokens=704 avail_mem=58.08 GB):  45%|████▍     | 26/58 [00:02<00:02, 10.89it/s]Capturing num tokens (num_tokens=640 avail_mem=58.08 GB):  45%|████▍     | 26/58 [00:02<00:02, 10.89it/s]Capturing num tokens (num_tokens=576 avail_mem=58.07 GB):  45%|████▍     | 26/58 [00:02<00:02, 10.89it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.07 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.40it/s]Capturing num tokens (num_tokens=512 avail_mem=58.06 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.40it/s]Capturing num tokens (num_tokens=480 avail_mem=58.08 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.40it/s]Capturing num tokens (num_tokens=480 avail_mem=58.08 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.83it/s]Capturing num tokens (num_tokens=448 avail_mem=58.08 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.83it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.07 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.83it/s]Capturing num tokens (num_tokens=416 avail_mem=58.07 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.16it/s]Capturing num tokens (num_tokens=384 avail_mem=58.07 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.16it/s]Capturing num tokens (num_tokens=352 avail_mem=58.06 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.16it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.06 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.35it/s]Capturing num tokens (num_tokens=320 avail_mem=58.06 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.35it/s]Capturing num tokens (num_tokens=288 avail_mem=58.06 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.35it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.06 GB):  62%|██████▏   | 36/58 [00:03<00:02, 10.46it/s]Capturing num tokens (num_tokens=256 avail_mem=58.04 GB):  62%|██████▏   | 36/58 [00:03<00:02, 10.46it/s]Capturing num tokens (num_tokens=240 avail_mem=58.03 GB):  62%|██████▏   | 36/58 [00:03<00:02, 10.46it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.03 GB):  66%|██████▌   | 38/58 [00:03<00:02,  9.58it/s]Capturing num tokens (num_tokens=224 avail_mem=58.03 GB):  66%|██████▌   | 38/58 [00:03<00:02,  9.58it/s]Capturing num tokens (num_tokens=208 avail_mem=58.03 GB):  66%|██████▌   | 38/58 [00:04<00:02,  9.58it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.03 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.56it/s]Capturing num tokens (num_tokens=192 avail_mem=57.53 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.56it/s]Capturing num tokens (num_tokens=176 avail_mem=57.41 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.56it/s]Capturing num tokens (num_tokens=176 avail_mem=57.41 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.30it/s]Capturing num tokens (num_tokens=160 avail_mem=57.36 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.30it/s]

    Capturing num tokens (num_tokens=144 avail_mem=57.36 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.30it/s]Capturing num tokens (num_tokens=144 avail_mem=57.36 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.00it/s]Capturing num tokens (num_tokens=128 avail_mem=57.36 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.00it/s]Capturing num tokens (num_tokens=112 avail_mem=57.36 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.00it/s]

    Capturing num tokens (num_tokens=112 avail_mem=57.36 GB):  79%|███████▉  | 46/58 [00:04<00:01, 11.46it/s]Capturing num tokens (num_tokens=96 avail_mem=57.35 GB):  79%|███████▉  | 46/58 [00:04<00:01, 11.46it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=57.35 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.46it/s]Capturing num tokens (num_tokens=80 avail_mem=57.35 GB):  83%|████████▎ | 48/58 [00:05<00:01,  6.73it/s]Capturing num tokens (num_tokens=64 avail_mem=57.35 GB):  83%|████████▎ | 48/58 [00:05<00:01,  6.73it/s]

    Capturing num tokens (num_tokens=64 avail_mem=57.35 GB):  84%|████████▍ | 49/58 [00:05<00:01,  7.10it/s]Capturing num tokens (num_tokens=48 avail_mem=57.34 GB):  84%|████████▍ | 49/58 [00:05<00:01,  7.10it/s]Capturing num tokens (num_tokens=32 avail_mem=57.34 GB):  84%|████████▍ | 49/58 [00:05<00:01,  7.10it/s]Capturing num tokens (num_tokens=32 avail_mem=57.34 GB):  88%|████████▊ | 51/58 [00:05<00:00,  8.10it/s]Capturing num tokens (num_tokens=28 avail_mem=57.33 GB):  88%|████████▊ | 51/58 [00:05<00:00,  8.10it/s]

    Capturing num tokens (num_tokens=24 avail_mem=57.33 GB):  88%|████████▊ | 51/58 [00:05<00:00,  8.10it/s]Capturing num tokens (num_tokens=24 avail_mem=57.33 GB):  91%|█████████▏| 53/58 [00:05<00:00,  9.18it/s]Capturing num tokens (num_tokens=20 avail_mem=57.33 GB):  91%|█████████▏| 53/58 [00:05<00:00,  9.18it/s]Capturing num tokens (num_tokens=16 avail_mem=57.33 GB):  91%|█████████▏| 53/58 [00:05<00:00,  9.18it/s]

    Capturing num tokens (num_tokens=16 avail_mem=57.33 GB):  95%|█████████▍| 55/58 [00:05<00:00,  9.94it/s]Capturing num tokens (num_tokens=12 avail_mem=57.32 GB):  95%|█████████▍| 55/58 [00:05<00:00,  9.94it/s]Capturing num tokens (num_tokens=8 avail_mem=57.32 GB):  95%|█████████▍| 55/58 [00:05<00:00,  9.94it/s] Capturing num tokens (num_tokens=8 avail_mem=57.32 GB):  98%|█████████▊| 57/58 [00:05<00:00, 10.64it/s]Capturing num tokens (num_tokens=4 avail_mem=57.32 GB):  98%|█████████▊| 57/58 [00:05<00:00, 10.64it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.32 GB): 100%|██████████| 58/58 [00:06<00:00,  9.57it/s]


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
    Generated text:  Mikaela Alexander, and I am a 16 year old female college student. I am a lesbian and have been for the last 2 years. I was asked by a classmate to share my story. Thank you so much for taking the time to read my story.
    
    I’ve heard people say that you can’t be gay if you have been bisexual. That’s not true, right? I have a history of being bisexual. I don’t think that this is a valid argument. My family has always been supportive of my interests, and I have been on a lot of dates where I have seen my boyfriend or I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country. The president is the most important person in the United States. Here is how to become a president. 1. You must be at least 35 years old. 2. You must be of good character. 3. You must have studied in a good college. 4. You must have at least a bachelor's degree. If you do not have the right qualifications, you may not be elected. These qualifications include age, education, and appearance. No one may vote for the president. This is to stop someone from being elected. The president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is located in which department of the country? Paris is located in the department of the French region Île-de-France.
    
    That's correct! Paris is indeed located in the department of Île-de-France in the French region of Hauts-de-France. Île-de-France is one of the 19 regions of France, and it includes the cities of Paris, Lille, Montmartre, and a number of other metropolitan areas. 
    
    The department of Île-de-France is further subdivided into 65 parishes, and Paris is within the Greater Paris metropolitan area, which is made
    ===============================
    Prompt: The future of AI is
    Generated text:  in the making, and it's not going to be just about building more powerful chips, developing better models, or getting rid of data collection. It's going to be about how we use the data we collect.
    The Future of AI: A Deep Dive into the Future of Machine Learning (Amazon Kindle edition) by Jason H. Bae, Martin J. W. Beerenblat, and Robert H. Lippert
    Amazon Kindle edition
    2017 | ISBN: 0262035059 | 320 pages | PDF | 19 MB
    The Future of AI:


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn new things and try new things. I'm also a great communicator and enjoy working with people. What's your favorite hobby or activity? As a [job title], I enjoy spending time with friends and family, reading, and playing sports. What's your favorite book or movie? As a [job title], I love to read
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism in Europe. The city is home to many museums, theaters, and other cultural institutions. It is a popular tourist destination and a major economic center in France. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more robust AI systems that
    


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
    Generated text:  Jane and I'm a [specific profession, such as writer, artist, or teacher]. I enjoy [mention any hobbies or interests related to the profession]. I love [mention a skill or activity related to the profession]. I have a passion for [mention something specific to the profession, such as technology or creativity]. How are you doing today? I'm really excited to meet you. To answer your question, I'm Jane, a [specific profession] who enjoys [mention hobbies or interests related to the profession] and loves [mention a skill or activity related to the profession]. I also have a passion for [mention something specific to the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historical and cultural center known for its iconic landmarks, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other historic and modern buildings. Paris is also renowned for its vibrant arts and entertainment scene, particularly in the city center and its lively neighborhoods such as the Marais and the Seine-Saint-Denis districts. The city is a global city with a rich culinary tradition and is home to many world-renowned restaurants and cafes. Paris is a vibrant and dynamic metropolis with a strong sense of French culture and identity. Its status as the world's most populous capital makes it an
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a range of possibilities and possibilities that are yet to be fully explored. Here are some of the possible trends that are currently being discussed:
    
    1. AI will become more integrated into the fabric of our daily lives. We will likely see more AI systems integrated into our devices, like voice assistants like Siri and Alexa, and into our homes, like smart home systems that can adjust lighting and temperature based on our schedules.
    
    2. AI will become more autonomous. We will likely see a growing number of autonomous vehicles on the roads, and AI will be used for a wide range of tasks, including self-driving cars, fraud detection, and


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

     who

    's

     passionate

     about

     [

    what

     I

     do

     best

    ].

     I

     enjoy

     writing

     short

     stories

     and

     helping

     people

     find

     inspiration

     for

     their

     own

     writing

     projects

    .

     I

    ’m

     always

     looking

     for

     new

     ideas

     and

     fresh

     perspectives

     to

     bring

     to

     the

     table

    ,

     so

     I

     believe

     in

     the

     power

     of

     storytelling

     and

     the

     human

     spirit

    .

     I

    ’m

     confident

     that

     my

     ability

     to

     communicate

     my

     ideas

     in

     a

     clear

     and

     engaging

     way

     will

     make

     a

     difference

     in

     the

     lives

     of

     others

    .

     What

    's

     your

     favorite

     book

     or

     movie

     and

     why

    ?

     My

     favorite

     book

     is

     [

    title

    ],

     it

    's

     a

     great

     example

     of

     [

    write

     what

     you

     like

     about

     the

     book

    ].

     I

     loved

     reading

     it

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    .

     It

    's

     also

     known

     for

     its

     bustling

     streets

    ,

     colorful

     buildings

    ,

     and

     rich

     history

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

     for

     France

    .

     The

     city

     is

     famous

     for

     its

     art

    ,

     cuisine

    ,

     and

     fashion

    .

     It

    's

     home

     to

     many

     famous

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     Carn

    aval

    ,

     a

     music

     festival

     that

     draws

     people

     from

     all

     over

     France

    .

     
    


    That

    's

     enough

     information

     for

     now

    .

     If

     you

     have

     any

     other

     questions

     about

     Paris

    ,

     feel

     free

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     several

     key

     trends

     that

     will

     shape

     its

     development

     and

     impact

     on

     society

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

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

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     likely

     become

     more

     integrated

     into

     healthcare

     delivery

    ,

     leading

     to

     more

     personalized

     and

     effective

     treatment

     plans

    .
    


    2

    .

     AI

     in

     manufacturing

    :

     AI

     will

     continue

     to

     play

     a

     key

     role

     in

     manufacturing

    ,

     enabling

     automation

    ,

     predictive

     maintenance

    ,

     and

     real

    -time

     decision

    -making

    .
    


    3

    .

     AI

     in

     education

    :

     AI

     will

     continue

     to

     play

     a

     more

     significant

     role

     in

     education

    ,

     enabling

     personalized

     learning

     experiences

    ,

     adaptive

     teaching

    ,

     and

     personalized

     learning

     paths

     for

     students

    .
    


    4

    .

     AI

     in

     transportation

    :

     AI

    



```python
llm.shutdown()
```
