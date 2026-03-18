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

    [2026-03-18 02:16:03] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-18 02:16:03] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-18 02:16:03] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-18 02:16:05] INFO server_args.py:2160: Attention backend not specified. Use fa3 backend by default.


    [2026-03-18 02:16:05] INFO server_args.py:3330: Set soft_watchdog_timeout since in CI


    [2026-03-18 02:16:05] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=10261743, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.96it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.96it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:34,  1.60it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:34,  1.60it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:34,  1.60it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:34,  1.60it/s]

    Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:34,  1.60it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=2816):  12%|█▏        | 7/58 [00:02<00:11,  4.52it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]

    Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=832):  24%|██▍       | 14/58 [00:02<00:04, 10.95it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:02<00:01, 21.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:02<00:00, 27.94it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:02<00:00, 41.61it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:03<00:00, 41.61it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 52.87it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 18.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.26 GB):   9%|▊         | 5/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.26 GB):   9%|▊         | 5/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.25 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.25 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.80it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=59.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.24 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.19 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=960 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s] Capturing num tokens (num_tokens=896 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=832 avail_mem=59.20 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s]

    Capturing num tokens (num_tokens=768 avail_mem=59.20 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=704 avail_mem=59.20 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=704 avail_mem=59.20 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=640 avail_mem=59.19 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=576 avail_mem=59.19 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=512 avail_mem=59.18 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=480 avail_mem=59.20 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=448 avail_mem=59.19 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=416 avail_mem=59.19 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.98it/s]Capturing num tokens (num_tokens=416 avail_mem=59.19 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=384 avail_mem=59.19 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=352 avail_mem=59.18 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=320 avail_mem=59.18 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]

    Capturing num tokens (num_tokens=288 avail_mem=59.18 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=256 avail_mem=59.17 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=240 avail_mem=59.17 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.68it/s]Capturing num tokens (num_tokens=240 avail_mem=59.17 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.23it/s]Capturing num tokens (num_tokens=224 avail_mem=59.17 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.23it/s]Capturing num tokens (num_tokens=208 avail_mem=59.16 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.23it/s]Capturing num tokens (num_tokens=192 avail_mem=59.16 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.23it/s]Capturing num tokens (num_tokens=176 avail_mem=59.16 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.23it/s]Capturing num tokens (num_tokens=160 avail_mem=59.16 GB):  66%|██████▌   | 38/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=144 avail_mem=59.15 GB):  66%|██████▌   | 38/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=144 avail_mem=59.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s]Capturing num tokens (num_tokens=128 avail_mem=59.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s]Capturing num tokens (num_tokens=112 avail_mem=59.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s]

    Capturing num tokens (num_tokens=96 avail_mem=59.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s] Capturing num tokens (num_tokens=80 avail_mem=59.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s]Capturing num tokens (num_tokens=64 avail_mem=59.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s]Capturing num tokens (num_tokens=48 avail_mem=59.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.50it/s]Capturing num tokens (num_tokens=48 avail_mem=59.14 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=32 avail_mem=59.13 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=28 avail_mem=59.13 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=24 avail_mem=59.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=20 avail_mem=59.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=16 avail_mem=59.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=12 avail_mem=59.11 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.27it/s]Capturing num tokens (num_tokens=12 avail_mem=59.11 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.70it/s]Capturing num tokens (num_tokens=8 avail_mem=59.11 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.70it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=59.11 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.70it/s]Capturing num tokens (num_tokens=4 avail_mem=59.11 GB): 100%|██████████| 58/58 [00:01<00:00, 44.28it/s]


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
    Generated text:  Alice. I'm a science student. I like sports very much and I love to play basketball. My favorite basketball team is the New York Knicks. They are the best basketball team in the world. I like to play basketball with my best friend, Tommy. I really like the New York Knicks, and I like to watch their basketball games. When I play basketball, I have fun. I usually play basketball at home on weekends. I have basketballs and basketballs for my family. I also have my favorite basketball player. I call him "Becky" in my game. I think her name sounds like "Becky
    ===============================
    Prompt: The president of the United States is
    Generated text:  78 years old now. In how many years will the president be 50 years old? If the president is currently 78 years old and wants to be 50 years old, we can calculate the number of years it will take by subtracting his current age from 50 years old and then dividing by 2 because the president will be half his current age when the year changes.
    
    50 years old - 78 years old = 12 years old
    
    Now, we divide that by 2 to find out how many years it will take for the president to become 50 years old:
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  located at the following coordinates: (48.8584, 2.3522).
    Capitalize the first letter of the first word of the first sentence and the first word of the second sentence.
    
    The capital of France is located at the following coordinates: (48.8584, 2.3522). The capital of France is located at the following coordinates: (48.8584, 2.3522). The capital of France is located at the following coordinates: (48.8584, 2.3522
    ===============================
    Prompt: The future of AI is
    Generated text:  ahead
    
    This is my 2013 Whitepaper on the future of AI, which will be available this week on my site.
    
    As we enter a new age of AI, the need to understand how AI will evolve over time is greater than ever. This is especially true in the case of complex and evolving human AI systems. How do we effectively understand AI systems’ behaviors? What impact do we want to have on them? What is the future of AI in the US, Europe, and elsewhere? This is the subject of this week’s post.
    
    As I explain in my paper, there are two kinds of artificial intelligence:
    
    


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a major tourist destination and a popular destination for business and leisure. The city is home to many famous French artists, writers, and musicians, and is known for its cuisine and wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating place. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare to transportation. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI ethics and privacy concerns: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy risks. This will require ongoing dialogue and regulation to ensure that AI is used in a responsible and ethical manner.
    
    3. AI for better health care: AI is already being
    


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
    Generated text:  Sarah. I'm a freelance writer and editor with a passion for helping people express themselves through language. I've been working in the industry for over a decade, and I love being able to connect with people on a personal level through my writing. I believe in the power of words to bring people together and I'm always looking for new ways to help people grow and express themselves creatively. If you're looking for someone to share your work, or someone to help you find your voice, feel free to reach out. How about you? Are there any specific projects or areas of focus that you're particularly interested in? It could be anything from
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which has a population of over 2.7 million people and is the city of the French President and the French national anthem is called "La Marseillaise".
    Paris is the largest city in France, with a population of over 2.7 million people and it is the seat of the French government. The city has a rich history and is famous for its art, cuisine, and architecture. The city has a vibrant nightlife and is known for its famous landmarks such as the Eiffel Tower and the Louvre Museum. It is also home to the historic streets of the Latin Quarter and the Seine River, which
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of technological advancements, changing societal and economic dynamics, and evolving human attitudes towards technology. Here are some possible future trends in artificial intelligence:
    
    1. Increased Use of AI in Healthcare: AI is likely to have a significant impact on healthcare in the coming years. AI-driven diagnostic tools, personalized treatment plans, and even automated diagnostic imaging are being developed. AI-powered healthcare robots are already being used to assist in surgeries, and AI algorithms are being used to analyze medical data to identify potential health risks and recommend preventive measures.
    
    2. AI in Education: AI is already being used in education to personalize learning experiences and assist


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

    'm

     a

     talented

     but

     ambitious

     software

     engineer

    .

     I

    've

     always

     been

     fascinated

     by

     technology

    ,

     and

     I

    'm

     constantly

     learning

     new

     things

     to

     keep

     up

     with

     the

     latest

     developments

     in

     the

     field

    .

     I

     thrive

     on

     problem

    -solving

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     my

     work

     and

     grow

     in

     my

     field

    .

     I

     enjoy

     mentoring

     others

     and

     helping

     them

     achieve

     their

     goals

    ,

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     learn

     and

     grow

    .

     Overall

    ,

     I

    'm

     a

     passionate

     and

     dedicated

     person

     who

     is

     always

     looking

     to

     take

     the

     next

     step

     in

     my

     career

    .

     Thanks

     for

     having

     me

    .

     I

    'm

     [

    Your

     Name

    ]

     and

     I

    'm

     excited

     to

     meet

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     and

     Lou

    vre

     Museum

    .

     It

     is

     also

     a

     historic

     center

     and

     a

     major

     cultural

     and

     economic

     hub

    .

     Known

     for

     its

     French

     cuisine

    ,

     fashion

    ,

     and

     art

     scene

    .

     Home

     to

     several

     world

    -ren

    owned

     museums

     and

     landmarks

    .

     Its

     rich

     history

     and

     cultural

     heritage

     are

     evident

     in

     its

     architecture

     and

     cuisine

    .

     The

     city

     is

     often

     referred

     to

     as

     the

     "

    Paris

     of

     the

     Paris

    "

     and

     is

     considered

     to

     be

     the

     world

    's

     most

     liv

    able

     city

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     and

     there

     are

     many

     potential

     areas

     where

     we

     can

     see

     advancements

     in

     the

     next

     decade

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     AI

     is

     becoming

     more

     accurate

     and

     more

     detailed

    ,

     leading

     to

     more

     precise

     predictions

     and

     more

     accurate

     outcomes

    .

     This

     will

     be

     particularly

     important

     in

     fields

     such

     as

     medicine

    ,

     where

     accurate

     diagnoses

     and

     treatments

     are

     crucial

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

     AI

     is

     already

     being

     integrated

     into

     our

     daily

     lives

    ,

     from

     the

     way

     we

     use

     smartphones

     to

     the

     way

     we

     design

     and

     build

     vehicles

    .

     As

     AI

     continues

     to

     advance

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     into

     our

     lives

     in

     ways

     that

     we

     may

     not

    



```python
llm.shutdown()
```
