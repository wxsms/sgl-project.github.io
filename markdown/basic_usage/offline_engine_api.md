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

    [2026-03-08 15:25:51] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-08 15:25:51] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-08 15:25:51] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-08 15:25:53] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-08 15:25:53] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:25:53] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=891386191, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.75it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.75it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   5%|▌         | 1/20 [00:00<00:03,  5.17it/s]Capturing batches (bs=120 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.17it/s]

    Capturing batches (bs=112 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.17it/s]Capturing batches (bs=104 avail_mem=58.49 GB):   5%|▌         | 1/20 [00:00<00:03,  5.17it/s]Capturing batches (bs=104 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 15.11it/s]Capturing batches (bs=96 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 15.11it/s] Capturing batches (bs=88 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 15.11it/s]Capturing batches (bs=80 avail_mem=58.49 GB):  20%|██        | 4/20 [00:00<00:01, 15.11it/s]Capturing batches (bs=80 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.96it/s]Capturing batches (bs=72 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.96it/s]

    Capturing batches (bs=64 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.96it/s]Capturing batches (bs=56 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.96it/s]Capturing batches (bs=48 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.96it/s]Capturing batches (bs=48 avail_mem=58.49 GB):  55%|█████▌    | 11/20 [00:00<00:00, 24.82it/s]Capturing batches (bs=40 avail_mem=58.49 GB):  55%|█████▌    | 11/20 [00:00<00:00, 24.82it/s]Capturing batches (bs=32 avail_mem=58.49 GB):  55%|█████▌    | 11/20 [00:00<00:00, 24.82it/s]Capturing batches (bs=24 avail_mem=58.49 GB):  55%|█████▌    | 11/20 [00:00<00:00, 24.82it/s]Capturing batches (bs=24 avail_mem=58.49 GB):  70%|███████   | 14/20 [00:00<00:00, 25.45it/s]Capturing batches (bs=16 avail_mem=58.48 GB):  70%|███████   | 14/20 [00:00<00:00, 25.45it/s]

    Capturing batches (bs=12 avail_mem=58.48 GB):  70%|███████   | 14/20 [00:00<00:00, 25.45it/s]Capturing batches (bs=8 avail_mem=58.48 GB):  70%|███████   | 14/20 [00:00<00:00, 25.45it/s] Capturing batches (bs=8 avail_mem=58.48 GB):  85%|████████▌ | 17/20 [00:00<00:00, 23.44it/s]Capturing batches (bs=4 avail_mem=58.48 GB):  85%|████████▌ | 17/20 [00:00<00:00, 23.44it/s]Capturing batches (bs=2 avail_mem=58.48 GB):  85%|████████▌ | 17/20 [00:00<00:00, 23.44it/s]Capturing batches (bs=1 avail_mem=58.48 GB):  85%|████████▌ | 17/20 [00:00<00:00, 23.44it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 20/20 [00:00<00:00, 22.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:04,  2.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:04,  2.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:04,  2.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:04,  2.18s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:07,  6.85it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 16.23it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:02<00:01, 26.77it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=28):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=24):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]

    Compiling num tokens (num_tokens=20):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=16):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=12):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=8):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s] Compiling num tokens (num_tokens=4):  71%|███████   | 41/58 [00:02<00:00, 39.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 62.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 20.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.37 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.43it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=960 avail_mem=72.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s] Capturing num tokens (num_tokens=896 avail_mem=72.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=832 avail_mem=72.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=768 avail_mem=72.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=704 avail_mem=72.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=640 avail_mem=72.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=576 avail_mem=72.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=576 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=512 avail_mem=72.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=480 avail_mem=72.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.88it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=416 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=384 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=384 avail_mem=72.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=352 avail_mem=72.29 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=320 avail_mem=72.29 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=288 avail_mem=72.29 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=256 avail_mem=72.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=240 avail_mem=72.28 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.00it/s]Capturing num tokens (num_tokens=224 avail_mem=72.28 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.00it/s]Capturing num tokens (num_tokens=224 avail_mem=72.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=208 avail_mem=72.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.50it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=176 avail_mem=72.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=160 avail_mem=72.26 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=144 avail_mem=72.26 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=144 avail_mem=72.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=128 avail_mem=72.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=112 avail_mem=72.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=96 avail_mem=72.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s] Capturing num tokens (num_tokens=80 avail_mem=72.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=64 avail_mem=72.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=48 avail_mem=72.24 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=48 avail_mem=72.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=32 avail_mem=72.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.49it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.51 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=24 avail_mem=71.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=20 avail_mem=71.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=16 avail_mem=71.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=16 avail_mem=71.41 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=12 avail_mem=71.41 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=8 avail_mem=71.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.91it/s] Capturing num tokens (num_tokens=4 avail_mem=71.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=4 avail_mem=71.40 GB): 100%|██████████| 58/58 [00:01<00:00, 40.43it/s]


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
    Generated text:  Clara and I’m a 26-year-old nurse and writer. I spend most of my days providing care and advice to those who need it, and writing to myself and others for personal and professional development.
    My story began with my parents’ guidance when I was a young child. My father’s work required a lot of attention and was very stressful. My mother was a dedicated caregiver and a strong role model to me, and from the very beginning I felt the love and support she had for me. I went on to graduate from the University of Delaware with a B.S. in Nursing, and I fell in love with the concept of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country. That means the president works with other important people and helps make important decisions. The president is the boss of the United States. That means he or she is the boss of the whole country. President Bush is a very important president. He was the first one to be elected president. He was elected in 1988. He was the first African American to be elected president. But he did not like the way the country was run and he left office. The president is like a king. He has a lot of power. He can make his own decisions
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 错误
    B. 正确
    答案:
    
    B
    
    焊条电弧焊时，焊条与焊件之间的接触面积称为焊条的____。
    A. 焊接面积
    B. 熔化面积
    C. 接触面积
    D. 接触长度
    答案:
    
    C
    
    利用价值工程的方法进行产品改进，应注重产品的____。
    A. 市场价值
    B. 实用性
    C. 安全性
    D. 经济性
    答案:
    
    D
    
    干舷甲板上设置的梯道，在
    ===============================
    Prompt: The future of AI is
    Generated text:  bound to involve a continued evolution of our digital world. In the years to come, there will be a constant effort to innovate and develop new technologies that can improve our lives and enable us to reach new heights. As we continue to advance in this direction, it is important to consider the ethical and moral implications of these developments.
    One area that is likely to receive increased attention in the future is the use of AI in healthcare. With the advancement of AI, there is a potential to revolutionize the way we diagnose and treat diseases. For example, AI algorithms can be trained to analyze medical images and identify potential health issues more accurately than human doctors


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


    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm currently [Current Location] and I'm here to [Your Purpose]. I'm here to [Your Goal]. I'm here to [Your Achievements]. I'm here to [Your Passion]. I'm here to [Your Humor]. I'm here to [Your Personality]. I'm here to [Your Interests]. I'm here to [Your Skills]. I'm here to [Your Motivation]. I'm here to [Your Values]. I'm here to [Your Character]. I'm here to [Your Character]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, a historic neighborhood. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. Its history dates back to the Roman Empire and has been a major center of European culture for centuries. The city is known for its diverse cuisine, art, and fashion, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection,
    


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
    Generated text:  [Name]. I'm [Age] years old, [City, State] [Occupation], [Profession]. I enjoy [Favorite Hobby or Activity], and I have a deep passion for [Interest or Interest]. I'm always on the lookout for new experiences, whether it be in the form of travel, learning new skills, or simply exploring new places. I'm always eager to learn and grow, and I believe that learning is the key to success. So, if you have any questions or would like to discuss my experiences, feel free to reach out. I look forward to the opportunity to meet you. How can I assist
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.  
    
    Paris is a city in the Loire Valley region of central France and the country’s economic and cultural center. It is home to the President of the Republic and the highest court in France. It is also the world’s second-largest city and the eighth-largest in the world. 
    
    The city is famous for its architecture, especially the Eiffel Tower, which was built in 1889 and stands as a symbol of France. It is also home to the Notre-Dame Cathedral, the iconic Notre-Dame de Paris and the Pont des Arts. Paris is also a major center of the French language, the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  shaped by a complex interplay of technological, societal, and philosophical factors. Here are some potential trends in AI that are currently being explored and developed:
    
    1. Advancements in machine learning and deep learning: With the continued development of machine learning and deep learning algorithms, we are likely to see improvements in accuracy, speed, and versatility of AI systems. This could lead to more personalized and efficient healthcare, better natural language processing, and more intelligent self-driving vehicles.
    
    2. Integration of AI with human emotions: There is growing interest in AI systems that are designed to understand and respond to human emotions, such as empathy and emotional intelligence. This could


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

    insert

     name

     here

    ],

     and

     I

     am

     a

     [

    insert

     profession

     here

    ].

     I

     enjoy

     [

    insert

     why

     I

     enjoy

     my

     profession

     here

    ].

     My

     hobbies

     include

     [

    insert

     hobbies

     here

    ].

     My

     favorite

     book

     is

     [

    insert

     book

     title

     here

    ].

     I

     am

     passionate

     about

     [

    insert

     something

     related

     to

     the

     character

     or

     their

     profession

     here

    ].

     I

     have

     a

     deep

     respect

     for

     [

    insert

     a

     person

    's

     name

     here

    ].

     I

     believe

     that

     [

    insert

     something

     related

     to

     the

     character

     or

     their

     profession

     here

    ].

     I

     love

     to

     [

    insert

     what

     you

     like

     to

     do

     here

    ].

     I

     am

     always

     looking

     for

     new

     adventures

     and

     learning

     new

     things

    .

     If

     you

     ever

     meet

     me

    ,

     you

     will

     hear

     about

     all

     the

     adventures

     I

     have

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     a

     population

     of

     over

     

    2

     million

     and

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     The

     city

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     also

     home

     to

     many

     famous

     museums

    ,

     such

     as

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

    .

     The

     city

     is

     also

     famous

     for

     its

     music

    ,

     particularly

     the

     famous

     Paris

    ian

     jazz

     and

     pop

     scenes

    .

     Additionally

    ,

     Paris

     is

     a

     hub

     for

     business

     and

     commerce

    ,

     hosting

     numerous

     international

     events

     and

     attractions

    .

     With

     its

     rich

     history

     and

     beautiful

     architecture

    ,

     Paris

     continues

     to

     be

     a

     top

     destination

     for

     travelers

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     full

     of

     exciting

     developments

    ,

     but

     it

     is

     also

     challenging

     to

     predict

     exactly

     how

     it

     will

     evolve

    .

     Here

     are

     some

     of

     the

     potential

     trends

     we

     can

     expect

     to

     see

     in

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

    :

     As

     AI

     becomes

     more

     prevalent

     in

     everyday

     life

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     AI

     systems

     integrated

     into

     our

     daily

     routines

    .

     This

     could

     include

     devices

     like

     voice

     assistants

    ,

     smart

     homes

    ,

     and

     self

    -driving

     cars

    .
    


    2

    .

     Enhanced

     AI

     ethics

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     greater

     concern

     about

     the

     potential

     for

     AI

     to

     cause

     harm

     or

     harm

     to

     individuals

    .

     This

     is

     why

     it

    's

     important

     for

     developers

     to

     create

    



```python
llm.shutdown()
```
