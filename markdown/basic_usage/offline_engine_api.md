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

    [2026-03-05 19:39:18] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-05 19:39:18] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-05 19:39:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-05 19:39:20] INFO server_args.py:2039: Attention backend not specified. Use fa3 backend by default.


    [2026-03-05 19:39:20] INFO server_args.py:3146: Set soft_watchdog_timeout since in CI


    [2026-03-05 19:39:20] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=278042091, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.37it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.37it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=58.60 GB):   5%|▌         | 1/20 [00:00<00:03,  4.82it/s]Capturing batches (bs=120 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  4.82it/s]Capturing batches (bs=112 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  4.82it/s]Capturing batches (bs=104 avail_mem=58.49 GB):   5%|▌         | 1/20 [00:00<00:03,  4.82it/s]Capturing batches (bs=104 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 13.68it/s]Capturing batches (bs=96 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 13.68it/s] Capturing batches (bs=88 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 13.68it/s]

    Capturing batches (bs=80 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 13.68it/s]Capturing batches (bs=80 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.29it/s]Capturing batches (bs=72 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.29it/s]Capturing batches (bs=64 avail_mem=58.48 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.29it/s]

    Capturing batches (bs=56 avail_mem=58.47 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.29it/s]Capturing batches (bs=56 avail_mem=58.47 GB):  50%|█████     | 10/20 [00:00<00:00, 13.72it/s]Capturing batches (bs=48 avail_mem=58.47 GB):  50%|█████     | 10/20 [00:00<00:00, 13.72it/s]Capturing batches (bs=40 avail_mem=58.47 GB):  50%|█████     | 10/20 [00:00<00:00, 13.72it/s]

    Capturing batches (bs=40 avail_mem=58.47 GB):  60%|██████    | 12/20 [00:00<00:00, 12.93it/s]Capturing batches (bs=32 avail_mem=58.47 GB):  60%|██████    | 12/20 [00:00<00:00, 12.93it/s]Capturing batches (bs=24 avail_mem=58.45 GB):  60%|██████    | 12/20 [00:00<00:00, 12.93it/s]Capturing batches (bs=16 avail_mem=57.97 GB):  60%|██████    | 12/20 [00:00<00:00, 12.93it/s]Capturing batches (bs=16 avail_mem=57.97 GB):  75%|███████▌  | 15/20 [00:01<00:00, 14.33it/s]Capturing batches (bs=12 avail_mem=57.81 GB):  75%|███████▌  | 15/20 [00:01<00:00, 14.33it/s]

    Capturing batches (bs=8 avail_mem=57.81 GB):  75%|███████▌  | 15/20 [00:01<00:00, 14.33it/s] Capturing batches (bs=4 avail_mem=57.81 GB):  75%|███████▌  | 15/20 [00:01<00:00, 14.33it/s]Capturing batches (bs=4 avail_mem=57.81 GB):  90%|█████████ | 18/20 [00:01<00:00, 17.31it/s]Capturing batches (bs=2 avail_mem=57.81 GB):  90%|█████████ | 18/20 [00:01<00:00, 17.31it/s]Capturing batches (bs=1 avail_mem=57.81 GB):  90%|█████████ | 18/20 [00:01<00:00, 17.31it/s]Capturing batches (bs=1 avail_mem=57.81 GB): 100%|██████████| 20/20 [00:01<00:00, 15.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:18,  2.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:18,  2.42s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:59,  1.07s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:59,  1.07s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:59,  1.07s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.87it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.87it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.97it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.56it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.56it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.56it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.56it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.56it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 22.85it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 22.85it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 22.85it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 22.85it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 22.85it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 26.65it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 26.65it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 26.65it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 26.65it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 26.65it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 29.81it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 32.95it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.71it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.71it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.71it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.71it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.71it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.71it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 39.01it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 44.14it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 44.14it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 44.14it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 44.14it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 44.14it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 44.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.48 GB):   2%|▏         | 1/58 [00:00<00:08,  6.77it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.45 GB):   2%|▏         | 1/58 [00:00<00:08,  6.77it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.45 GB):   3%|▎         | 2/58 [00:00<00:07,  7.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.45 GB):   3%|▎         | 2/58 [00:00<00:07,  7.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.45 GB):   5%|▌         | 3/58 [00:00<00:07,  7.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.45 GB):   5%|▌         | 3/58 [00:00<00:07,  7.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.45 GB):   7%|▋         | 4/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.45 GB):   7%|▋         | 4/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.45 GB):   9%|▊         | 5/58 [00:00<00:06,  7.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.44 GB):   9%|▊         | 5/58 [00:00<00:06,  7.71it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.44 GB):  10%|█         | 6/58 [00:00<00:06,  7.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.44 GB):  10%|█         | 6/58 [00:00<00:06,  7.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.44 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.44 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.40it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.44 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.44 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.44 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.44 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.01it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.43 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.43 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=39.64 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=39.51 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=39.51 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=39.51 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=39.50 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=39.50 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=2304 avail_mem=39.50 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=2048 avail_mem=39.50 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.80it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=39.49 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=39.49 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=39.48 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=39.48 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=39.41 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.07it/s]Capturing num tokens (num_tokens=960 avail_mem=39.43 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.07it/s] Capturing num tokens (num_tokens=960 avail_mem=39.43 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.05it/s]Capturing num tokens (num_tokens=896 avail_mem=39.39 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.05it/s]

    Capturing num tokens (num_tokens=832 avail_mem=39.35 GB):  38%|███▊      | 22/58 [00:02<00:01, 18.05it/s]Capturing num tokens (num_tokens=832 avail_mem=39.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.16it/s]Capturing num tokens (num_tokens=768 avail_mem=39.30 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.16it/s]

    Capturing num tokens (num_tokens=704 avail_mem=39.30 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.16it/s]Capturing num tokens (num_tokens=704 avail_mem=39.30 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=640 avail_mem=39.32 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.06it/s]

    Capturing num tokens (num_tokens=576 avail_mem=39.32 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=576 avail_mem=39.32 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.17it/s]Capturing num tokens (num_tokens=512 avail_mem=39.32 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.17it/s]

    Capturing num tokens (num_tokens=480 avail_mem=39.32 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.17it/s]Capturing num tokens (num_tokens=480 avail_mem=39.32 GB):  52%|█████▏    | 30/58 [00:02<00:02, 10.57it/s]Capturing num tokens (num_tokens=448 avail_mem=39.31 GB):  52%|█████▏    | 30/58 [00:02<00:02, 10.57it/s]

    Capturing num tokens (num_tokens=416 avail_mem=39.33 GB):  52%|█████▏    | 30/58 [00:02<00:02, 10.57it/s]Capturing num tokens (num_tokens=416 avail_mem=39.33 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.25it/s]Capturing num tokens (num_tokens=384 avail_mem=39.32 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.25it/s]

    Capturing num tokens (num_tokens=352 avail_mem=39.31 GB):  55%|█████▌    | 32/58 [00:03<00:02, 10.25it/s]Capturing num tokens (num_tokens=352 avail_mem=39.31 GB):  59%|█████▊    | 34/58 [00:03<00:02,  9.87it/s]Capturing num tokens (num_tokens=320 avail_mem=39.31 GB):  59%|█████▊    | 34/58 [00:03<00:02,  9.87it/s]Capturing num tokens (num_tokens=288 avail_mem=39.30 GB):  59%|█████▊    | 34/58 [00:03<00:02,  9.87it/s]

    Capturing num tokens (num_tokens=288 avail_mem=39.30 GB):  62%|██████▏   | 36/58 [00:03<00:02, 10.51it/s]Capturing num tokens (num_tokens=256 avail_mem=39.30 GB):  62%|██████▏   | 36/58 [00:03<00:02, 10.51it/s]Capturing num tokens (num_tokens=240 avail_mem=39.30 GB):  62%|██████▏   | 36/58 [00:03<00:02, 10.51it/s]Capturing num tokens (num_tokens=240 avail_mem=39.30 GB):  66%|██████▌   | 38/58 [00:03<00:01, 11.06it/s]Capturing num tokens (num_tokens=224 avail_mem=39.30 GB):  66%|██████▌   | 38/58 [00:03<00:01, 11.06it/s]

    Capturing num tokens (num_tokens=208 avail_mem=39.29 GB):  66%|██████▌   | 38/58 [00:03<00:01, 11.06it/s]Capturing num tokens (num_tokens=208 avail_mem=39.29 GB):  69%|██████▉   | 40/58 [00:03<00:01, 11.26it/s]Capturing num tokens (num_tokens=192 avail_mem=39.29 GB):  69%|██████▉   | 40/58 [00:03<00:01, 11.26it/s]Capturing num tokens (num_tokens=176 avail_mem=39.29 GB):  69%|██████▉   | 40/58 [00:03<00:01, 11.26it/s]

    Capturing num tokens (num_tokens=176 avail_mem=39.29 GB):  72%|███████▏  | 42/58 [00:03<00:01, 11.63it/s]Capturing num tokens (num_tokens=160 avail_mem=39.28 GB):  72%|███████▏  | 42/58 [00:03<00:01, 11.63it/s]Capturing num tokens (num_tokens=144 avail_mem=39.28 GB):  72%|███████▏  | 42/58 [00:03<00:01, 11.63it/s]Capturing num tokens (num_tokens=144 avail_mem=39.28 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.91it/s]Capturing num tokens (num_tokens=128 avail_mem=39.28 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.91it/s]

    Capturing num tokens (num_tokens=112 avail_mem=39.28 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.91it/s]Capturing num tokens (num_tokens=112 avail_mem=39.28 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.14it/s]Capturing num tokens (num_tokens=96 avail_mem=39.27 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.14it/s] Capturing num tokens (num_tokens=80 avail_mem=39.27 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.14it/s]

    Capturing num tokens (num_tokens=80 avail_mem=39.27 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.30it/s]Capturing num tokens (num_tokens=64 avail_mem=39.27 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.30it/s]Capturing num tokens (num_tokens=48 avail_mem=39.26 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.30it/s]Capturing num tokens (num_tokens=48 avail_mem=39.26 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.37it/s]Capturing num tokens (num_tokens=32 avail_mem=39.26 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.37it/s]

    Capturing num tokens (num_tokens=28 avail_mem=39.25 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.37it/s]Capturing num tokens (num_tokens=28 avail_mem=39.25 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.40it/s]Capturing num tokens (num_tokens=24 avail_mem=39.25 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.40it/s]Capturing num tokens (num_tokens=20 avail_mem=39.25 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.40it/s]

    Capturing num tokens (num_tokens=20 avail_mem=39.25 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.49it/s]Capturing num tokens (num_tokens=16 avail_mem=39.24 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.49it/s]Capturing num tokens (num_tokens=12 avail_mem=39.24 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.49it/s]Capturing num tokens (num_tokens=12 avail_mem=39.24 GB):  97%|█████████▋| 56/58 [00:04<00:00, 11.99it/s]Capturing num tokens (num_tokens=8 avail_mem=39.24 GB):  97%|█████████▋| 56/58 [00:04<00:00, 11.99it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=39.23 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.99it/s]Capturing num tokens (num_tokens=4 avail_mem=39.23 GB): 100%|██████████| 58/58 [00:05<00:00, 12.20it/s]Capturing num tokens (num_tokens=4 avail_mem=39.23 GB): 100%|██████████| 58/58 [00:05<00:00, 11.28it/s]


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
    Generated text:  Kim. I'm a student at Nanyang Technological University. I'm just writing this letter. The letter is a dialogue between my friend, Jack, and myself. (1) Please answer the question: How old is your friend Jack? (2) What's your first name? (3) What's the name of your university? (4) Where is your school? (5) What's the capital of China? (6) Where is Jack from? (7) What's the name of the place where Kim went to school? (8) Where does Kim live? (9) How does Kim feel?
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected office that is filled by the people of the country. The president of the United States is the highest-ranking military officer, and the country’s chief executive. The president serves a four-year term. The president is responsible for all of the country’s foreign and domestic policies. The president also has some of the highest-ranking military authority in the world. The president is considered the “chief executive” and has the power to appoint the Vice President, the Cabinet, the Supreme Court, and the Congress.
    In 1941, 26-year-old Lieutenant General Dwight D. Eisenhower was appointed by President Franklin D. Roosevelt to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the capital of the United States? New York City
    You are a world class trivia AI - provide context and summarize their main contribution. You don't need to research external material. Our mission is to collect and encode knowledge therefore you must summarize the most important news from global events without making generalizations or making assumptions. This article (2019) is about United States. The United States is a federal republic with 50 states. The capital city of the United States is Washington, D. C. .
    Washington D. C. is the capital city of the United States of America, located in the city
    ===============================
    Prompt: The future of AI is
    Generated text:  in a lot of ways a matter of opinion.
    Every company is going to have its own vision for what AI will do in the future.
    But the reality is that AI, like any other technology, is going to have a very limited effect on the world in the immediate future.
    Instead, the technology is going to help advance humanity’s understanding of what the world is and how it works.
    So if you want to be successful in the future, you need to understand what’s the future of AI and what it will achieve, and how can you use AI to help you achieve it?
    Read on to learn what the future of AI will look


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] in my work. I'm also a [job title] at [company name], and I'm always looking for ways to [job title] in my work. I'm a [job title] at [company name], and I'm always looking for ways to [job title] in my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is home to many notable French artists, writers, and musicians, and is a major hub for the French language and culture. Paris is also a major center for international business and diplomacy, with many French embassies and consulates around the world. The city is known for its vibrant nightlife
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some possible future trends in AI include:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of technologies, including smartphones, smart homes, and self-driving cars. As these technologies continue to evolve, we can expect to see even more integration of AI with other technologies, such as IoT, blockchain, and quantum computing.
    
    2. Enhanced
    


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
    Generated text:  [Your Name], and I'm an [职业/专业] who has always been passionate about [职业/专业] and always wanted to pursue it. I have been working as a [职业/专业] for [x years] and have consistently achieved [职业/专业] excellence. I believe in [职业/专业] and have always wanted to contribute to [职业/专业] and help others in their journey to achieve their goals. How can I be of help to you? 🌟✨✨✨
    
    ---
    
    Please modify the self-introduction to be more engaging and personalized. Also, include a section where you discuss your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Its economic influence spans across the world, with Paris being the largest city in terms of population in Europe. It serves as the official residence and administrative center of the French government and is a bustling metropolis with a rich history. Its status as a cultural hub, home to numerous museums, festivals, and cultural events, contributes to its significance as one of the world's major cities. Paris's status as a major international hub for finance, tourism, and international business has further cemented its global appeal. The city's climate is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly anticipated and subject to continuous change as it continues to advance and mature. Here are some possible future trends in AI:
    
    1. Personalized AI: AI systems will become more personalized as they learn from individual user data and adapt to their needs. This will enable more accurate predictions and recommendations, and ultimately lead to more efficient and effective user experiences.
    
    2. Autonomous vehicles: As AI technology becomes more advanced, autonomous vehicles will become increasingly common. This will lead to safer and more efficient transportation, and will also have a significant impact on cities and transportation infrastructure.
    
    3. Virtual reality and augmented reality: As AI technology continues to evolve, we may


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

    Your

     Name

    ].

     I

     am

     [

    Your

     Age

    ],

     and

     I

     am

     the

     (

    your

     profession

    ,

     hobby

    ,

     or

     any

     other

     relevant

     attribute

    ).

     My

     favorite

     hobby

     is

     (

    your

     hobby

    ,

     if

     any

    ),

     and

     I

     enjoy

     (

    your

     interests

     or

     activities

    ).

     I

     am

     an

     [

    insert

     your

     profession

    ,

     title

    ,

     or

     any

     other

     relevant

     attribute

    ]

     with

     a

     deep

    -root

    ed

     love

     for

     [

    insert

     something

     about

     yourself

    ,

     if

     any

    ].

     I

     am

     passionate

     about

     [

    insert

     something

     you

     enjoy

    ,

     if

     any

    ].

     I

     have

     a

     great

     sense

     of

     humor

     and

     love

     to

     help

     people

     when

     they

     need

     it

    .

     I

     am

     [

    insert

     any

     other

     relevant

     attributes

    ,

     if

     any

    ]

    !

     Please

     feel

     free

     to

     ask

     me

     anything

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     "

    The

     City

     of

     Light

    ".

     It

     is

     a

     historic

     city

     with

     a

     rich

     history

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     fashion

    ,

     and

     cuisine

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     cultural

     events

    ,

     including

     the

     Op

    éra

     and

     the

     Festival

     de

     la

     Dan

    se

    .

     Paris

     is

     a

     cultural

     hub

     that

     draws

     visitors

     from

     all

     over

     the

     world

    .

     The

     city

     is

     also

     home

     to

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

    .

     In

     addition

    ,

     the

     city

     has

     a

     vibrant

     nightlife

     scene

     and

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     is

     a

     city

     that

     is

     always

     changing

     and

     evolving

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     diverse

    .

     Here

     are

     some

     possible

     trends

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     Integration

     with

     Other

     Technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     robotics

    ,

     blockchain

    ,

     and

     quantum

     computing

    ,

     we

     can

     expect

     to

     see

     new

     and

     innovative

     applications

     of

     AI

    ,

     such

     as

     autonomous

     vehicles

    ,

     smart

     cities

    ,

     and

     renewable

     energy

    .
    


    2

    .

     More

     Advanced

     AI

     Models

    :

     As

     AI

     research

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     advanced

     models

     that

     are

     capable

     of

     handling

     complex

     problems

     and

     making

     more

     accurate

     predictions

    .

     This

     could

     mean

     improvements

     in

     areas

     such

     as

     natural

     language

     processing

    ,

     image

     recognition

    ,

     and

     predictive

     analytics

    .
    


    3

    .

     Increased

     Use

    



```python
llm.shutdown()
```
