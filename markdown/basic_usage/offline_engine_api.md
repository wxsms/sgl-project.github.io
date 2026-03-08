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

    [2026-03-08 12:24:43] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-08 12:24:43] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-08 12:24:43] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-08 12:24:45] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-08 12:24:45] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 12:24:45] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=365689173, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.47it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=77.01 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=77.01 GB):   5%|▌         | 1/20 [00:00<00:02,  6.38it/s]Capturing batches (bs=120 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:00<00:02,  6.38it/s]Capturing batches (bs=112 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:00<00:02,  6.38it/s]

    Capturing batches (bs=104 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:00<00:02,  6.38it/s]Capturing batches (bs=96 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:00<00:02,  6.38it/s] Capturing batches (bs=88 avail_mem=76.91 GB):   5%|▌         | 1/20 [00:00<00:02,  6.38it/s]Capturing batches (bs=88 avail_mem=76.91 GB):  30%|███       | 6/20 [00:00<00:00, 24.48it/s]Capturing batches (bs=80 avail_mem=76.91 GB):  30%|███       | 6/20 [00:00<00:00, 24.48it/s]Capturing batches (bs=72 avail_mem=76.91 GB):  30%|███       | 6/20 [00:00<00:00, 24.48it/s]Capturing batches (bs=64 avail_mem=76.91 GB):  30%|███       | 6/20 [00:00<00:00, 24.48it/s]Capturing batches (bs=56 avail_mem=76.90 GB):  30%|███       | 6/20 [00:00<00:00, 24.48it/s]Capturing batches (bs=48 avail_mem=76.90 GB):  30%|███       | 6/20 [00:00<00:00, 24.48it/s]Capturing batches (bs=48 avail_mem=76.90 GB):  55%|█████▌    | 11/20 [00:00<00:00, 31.32it/s]Capturing batches (bs=40 avail_mem=76.90 GB):  55%|█████▌    | 11/20 [00:00<00:00, 31.32it/s]

    Capturing batches (bs=32 avail_mem=76.90 GB):  55%|█████▌    | 11/20 [00:00<00:00, 31.32it/s]Capturing batches (bs=24 avail_mem=76.90 GB):  55%|█████▌    | 11/20 [00:00<00:00, 31.32it/s]Capturing batches (bs=16 avail_mem=76.90 GB):  55%|█████▌    | 11/20 [00:00<00:00, 31.32it/s]Capturing batches (bs=16 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 29.85it/s]Capturing batches (bs=12 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 29.85it/s]Capturing batches (bs=8 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 29.85it/s] Capturing batches (bs=4 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 29.85it/s]Capturing batches (bs=2 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 29.85it/s]

    Capturing batches (bs=1 avail_mem=76.89 GB):  75%|███████▌  | 15/20 [00:00<00:00, 29.85it/s]Capturing batches (bs=1 avail_mem=76.89 GB): 100%|██████████| 20/20 [00:00<00:00, 35.60it/s]Capturing batches (bs=1 avail_mem=76.89 GB): 100%|██████████| 20/20 [00:00<00:00, 30.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:07,  6.70it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 15.95it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:02<00:01, 26.29it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=64):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]

    Compiling num tokens (num_tokens=48):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=32):  66%|██████▌   | 38/58 [00:02<00:00, 33.87it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:02<00:00, 50.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 19.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.56 GB):   3%|▎         | 2/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.56 GB):   3%|▎         | 2/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.55 GB):   3%|▎         | 2/58 [00:00<00:03, 15.41it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.55 GB):   7%|▋         | 4/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.55 GB):   7%|▋         | 4/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.55 GB):   7%|▋         | 4/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.55 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.55 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.55 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.54 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.54 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.54 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.54 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.53 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.53 GB):  21%|██        | 12/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.53 GB):  21%|██        | 12/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.53 GB):  21%|██        | 12/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.52 GB):  21%|██        | 12/58 [00:00<00:01, 23.80it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=76.52 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.52 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.52 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.51 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.51 GB):  31%|███       | 18/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.51 GB):  31%|███       | 18/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.51 GB):  31%|███       | 18/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.49 GB):  31%|███       | 18/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=960 avail_mem=76.50 GB):  31%|███       | 18/58 [00:00<00:01, 26.07it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.50 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=896 avail_mem=76.50 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=832 avail_mem=76.50 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=768 avail_mem=76.49 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=768 avail_mem=76.49 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.53it/s]Capturing num tokens (num_tokens=704 avail_mem=76.49 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.53it/s]Capturing num tokens (num_tokens=640 avail_mem=76.49 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.53it/s]Capturing num tokens (num_tokens=576 avail_mem=76.48 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.53it/s]Capturing num tokens (num_tokens=512 avail_mem=76.47 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.53it/s]Capturing num tokens (num_tokens=480 avail_mem=76.49 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.53it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.49 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=448 avail_mem=76.49 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=416 avail_mem=76.49 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=384 avail_mem=76.48 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=352 avail_mem=76.48 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=320 avail_mem=76.47 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=288 avail_mem=76.47 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=288 avail_mem=76.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=256 avail_mem=76.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=240 avail_mem=76.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=224 avail_mem=76.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=208 avail_mem=76.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=192 avail_mem=76.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=176 avail_mem=76.45 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=160 avail_mem=76.45 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=144 avail_mem=76.45 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=128 avail_mem=76.44 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=112 avail_mem=76.44 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=96 avail_mem=76.44 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s] Capturing num tokens (num_tokens=80 avail_mem=76.43 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=80 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=64 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=48 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=32 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=28 avail_mem=76.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=20 avail_mem=76.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=20 avail_mem=76.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=16 avail_mem=76.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=12 avail_mem=76.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=8 avail_mem=76.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.27it/s] Capturing num tokens (num_tokens=4 avail_mem=76.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.27it/s]Capturing num tokens (num_tokens=4 avail_mem=76.40 GB): 100%|██████████| 58/58 [00:01<00:00, 35.15it/s]


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
    Generated text:  Gino and I'm a first year student at the University of British Columbia. I am an avid fan of both the band Weezer and the film, The Departed. I have a passion for the outdoors and am interested in learning more about sustainable practices and eco-friendly transportation. I am also an avid reader and enjoy exploring new and unique literary works.
    As a first year student at the University of British Columbia, I am passionate about pursuing a degree in Environmental Science and Sustainability. My goal is to gain a comprehensive understanding of the latest research, theories, and practices in the field. I am particularly interested in environmental justice, sustainability,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He has a lot of power. He is the leader of the country. The president can make important decisions. It is like the boss of the country. There are many different types of presidents. Some have more power than others. Presidents can make decisions that affect a lot of people. They can make decisions on many important things. Sometimes it is hard for people to understand how important a president is. The president is like the judge of the country. He is like the boss of the country. The president can make decisions that affect a lot of people. He can make important decisions on many important things. He can
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the capital of France and the largest city in France. The population of Paris is 2, 176, 000. The population of the city of Paris is 2, 176, 000. This city is located in the southeast of France. The population of Paris is 2, 176, 000. The city is located in the southeast of France.
    
    The headquarters of the French government is in the Eiffel Tower. The French government has its headquarters in the Eiffel Tower. The Eiffel Tower is in the middle
    ===============================
    Prompt: The future of AI is
    Generated text:  here – we’re on the cusp of a new era of automation. But before we can fully harness its potential, we need to learn how to use it.
    AI is a technology that has the potential to change the world in many ways, from new ways of managing and controlling our environment, to new ways of helping us make better decisions, to new ways of improving our lives.
    The automation of some tasks is already happening, but the full potential of AI is yet to be realized. In this blog post, we explore how AI is changing the world and the role that we can play in this new age of automation.
    One of the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [insert your profession] with [insert your experience] and [insert your education]. I'm passionate about [insert your passion or interest]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [insert your personality trait or characteristic] and I'm always ready to help others. I'm [insert your age] years old and I'm [insert your height or weight]. I'm [insert your hobbies or interests]. I'm [insert your favorite color] and I love [insert your favorite hobby or activity].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city is also home to many famous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée d'Orsay. Paris is a popular tourist destination and a major economic center in France. It is also known for its fashion industry, with many famous designers and boutiques. The city is home to many cultural institutions, including the Louvre Museum, the Mus
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into many aspects of our lives, from self-driving cars to smart home devices. As technology continues to advance, we can expect to see even more integration of AI into our daily routines.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    3. Development of AI that is more
    


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
    Generated text:  [Name], and I'm a [Occupation or Field] expert. I specialize in [explain a specific skill or expertise]. And I love [mention a hobby, activity, or interest that I enjoy]. Thank you for having me! 
    
    Remember, this is a neutral introduction and doesn't express any personal opinion or opinion on the field you specialize in. The goal is to establish a friendly and respectful tone. Let me know if there's anything else I can do for you! 
    Additionally, I would like to know more about your expertise or interests that you are not mentioned in the introduction. What are some things that you are passionate
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre Dame Cathedral. 
    
    (1 point) Yes
    (0 points) No
    You can enter a variable named `city_capital` which stores the name of the capital city of France, followed by a brief factual statement about it. The statement should be limited to 50 words.
    
    ```python
    city_capital("Paris")
    ``` ```python
    city_capital("Rome")
    ``` ```python
    city_capital("London")
    ``` ```python
    city_capital("Vienna")
    ``` ```python
    city_capital("
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and there are many potential areas of growth and development. Here are some possible trends that could occur in the coming years:
    
    1. Improved accuracy and efficiency: AI will continue to improve its ability to process large amounts of data and make accurate predictions. This could lead to greater efficiency and lower costs in many industries.
    
    2. Integration of AI with other technologies: AI is becoming increasingly integrated with other technologies, such as blockchain, cloud computing, and big data. This integration could lead to even greater benefits, such as increased security and better decision-making.
    
    3. More targeted AI: AI will continue to be used for tasks that are currently


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

    ],

     and

     I

     am

     an

     [

    insert

     your

     profession

     here

    ].

     I

     have

     always

     been

     fascinated

     by

     the

     world

     of

     technology

     and

     have

     always

     been

     interested

     in

     how

     it

     affects

     our

     lives

    .

     I

    've

     always

     had

     an

     eye

     for

     creativity

     and

     innovation

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     make

     the

     world

     a

     better

     place

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     spending

     time

     with

     friends

     and

     exploring

     new

     places

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?

     Can

     you

     tell

     me

     a

     little

     bit

     about

     your

     background

     or

     any

     relevant

     experience

     that

     might

     make

     you

     stand

     out

     in

     our

     conversation

    ?

     Please

     feel

     free

     to

     include

     any

     interesting

     facts

     or

     anecdotes

     that

     might

     help

     to

     bring

     you

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     major

     financial

     and

     cultural

     center

    ,

     home

     to

     many

     of

     the

     country

    's

     most

     prestigious

     institutions

    ,

     including

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

     It

     is

     also

     known

     for

     its

     extensive

     shopping

     and

     leisure

     industry

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Se

    ine

     River

    .

     The

     city

    's

     architecture

    ,

     particularly

     its

     Gothic

     and

     Bar

    oque

     buildings

    ,

     is

     also

     a

     significant

     feature

     of

     its

     cultural

     heritage

    .

     Paris

     is

     an

     important

     city

     in

     France

     and

     plays

     a

     central

     role

     in

     the

     country

    's

     history

    ,

     politics

    ,

     and

     economy

    .

     It

     is

     ranked

     as

     the

     most

     important

     city

     in

     the

     world

     for

     film

    ,

     music

    ,

     and

     publishing

    .

     Paris

     is

     also

     known

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     advancements

     in

     machine

     learning

    ,

     and

     continued

     advances

     in

     technology

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     is

     a

     growing

     awareness

     of

     its

     potential

     to

     be

     used

     for

     good

    .

     This

     includes

     efforts

     to

     ensure

     that

     AI

     is

     developed

     and

     used

     in

     a

     way

     that

     promotes

     positive

     outcomes

     for

     humans

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     has

     the

     potential

     to

     improve

     patient

     outcomes

     in

     many

     ways

    .

     For

     example

    ,

     AI

     can

     be

     used

     to

     analyze

     medical

     data

     to

     identify

     patterns

     that

     may

     be

     missed

     by

    



```python
llm.shutdown()
```
