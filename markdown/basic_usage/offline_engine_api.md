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

    [2026-03-09 03:17:14] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-09 03:17:14] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-09 03:17:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-09 03:17:16] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-09 03:17:16] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-09 03:17:16] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=269562124, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=131.91 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=131.91 GB):   5%|▌         | 1/20 [00:00<00:03,  5.51it/s]Capturing batches (bs=120 avail_mem=131.76 GB):   5%|▌         | 1/20 [00:00<00:03,  5.51it/s]

    Capturing batches (bs=112 avail_mem=131.73 GB):   5%|▌         | 1/20 [00:00<00:03,  5.51it/s]Capturing batches (bs=104 avail_mem=131.72 GB):   5%|▌         | 1/20 [00:00<00:03,  5.51it/s]Capturing batches (bs=96 avail_mem=131.71 GB):   5%|▌         | 1/20 [00:00<00:03,  5.51it/s] Capturing batches (bs=96 avail_mem=131.71 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.10it/s]Capturing batches (bs=88 avail_mem=131.70 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.10it/s]Capturing batches (bs=80 avail_mem=131.71 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.10it/s]Capturing batches (bs=72 avail_mem=131.70 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.10it/s]Capturing batches (bs=64 avail_mem=131.69 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.10it/s]

    Capturing batches (bs=64 avail_mem=131.69 GB):  45%|████▌     | 9/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=56 avail_mem=131.68 GB):  45%|████▌     | 9/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=48 avail_mem=131.67 GB):  45%|████▌     | 9/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=40 avail_mem=131.66 GB):  45%|████▌     | 9/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=32 avail_mem=131.65 GB):  45%|████▌     | 9/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=32 avail_mem=131.65 GB):  65%|██████▌   | 13/20 [00:00<00:00, 26.41it/s]Capturing batches (bs=24 avail_mem=131.64 GB):  65%|██████▌   | 13/20 [00:00<00:00, 26.41it/s]Capturing batches (bs=16 avail_mem=131.64 GB):  65%|██████▌   | 13/20 [00:00<00:00, 26.41it/s]

    Capturing batches (bs=12 avail_mem=131.63 GB):  65%|██████▌   | 13/20 [00:00<00:00, 26.41it/s]Capturing batches (bs=12 avail_mem=131.63 GB):  80%|████████  | 16/20 [00:00<00:00, 25.52it/s]Capturing batches (bs=8 avail_mem=131.63 GB):  80%|████████  | 16/20 [00:00<00:00, 25.52it/s] Capturing batches (bs=4 avail_mem=131.62 GB):  80%|████████  | 16/20 [00:00<00:00, 25.52it/s]Capturing batches (bs=2 avail_mem=131.62 GB):  80%|████████  | 16/20 [00:00<00:00, 25.52it/s]Capturing batches (bs=1 avail_mem=131.61 GB):  80%|████████  | 16/20 [00:00<00:00, 25.52it/s]Capturing batches (bs=1 avail_mem=131.61 GB): 100%|██████████| 20/20 [00:00<00:00, 29.34it/s]Capturing batches (bs=1 avail_mem=131.61 GB): 100%|██████████| 20/20 [00:00<00:00, 25.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]

    Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:03<00:03, 12.61it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:03<00:01, 20.65it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:03<00:00, 29.57it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:03<00:00, 40.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.27 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.26 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.25 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.24 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.24 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.23 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.23 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.12it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=131.23 GB):  21%|██        | 12/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.23 GB):  21%|██        | 12/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.22 GB):  21%|██        | 12/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.22 GB):  21%|██        | 12/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.21 GB):  21%|██        | 12/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.20 GB):  21%|██        | 12/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.19 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.20it/s]

    Capturing num tokens (num_tokens=960 avail_mem=131.18 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.20it/s] Capturing num tokens (num_tokens=960 avail_mem=131.18 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.01it/s]Capturing num tokens (num_tokens=896 avail_mem=131.18 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.01it/s]Capturing num tokens (num_tokens=832 avail_mem=131.17 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.01it/s]Capturing num tokens (num_tokens=768 avail_mem=131.17 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.01it/s]Capturing num tokens (num_tokens=704 avail_mem=131.16 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.01it/s]Capturing num tokens (num_tokens=640 avail_mem=131.16 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.01it/s]Capturing num tokens (num_tokens=640 avail_mem=131.16 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=576 avail_mem=131.16 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=512 avail_mem=131.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=480 avail_mem=131.16 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.67it/s]

    Capturing num tokens (num_tokens=448 avail_mem=131.16 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=416 avail_mem=131.15 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=416 avail_mem=131.15 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=384 avail_mem=131.15 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=352 avail_mem=131.14 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=320 avail_mem=131.14 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=288 avail_mem=131.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=256 avail_mem=131.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.78it/s]Capturing num tokens (num_tokens=256 avail_mem=131.13 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=240 avail_mem=131.12 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=224 avail_mem=131.12 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]

    Capturing num tokens (num_tokens=208 avail_mem=131.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=192 avail_mem=131.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=176 avail_mem=131.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=176 avail_mem=131.11 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=160 avail_mem=131.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=144 avail_mem=131.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=128 avail_mem=131.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=112 avail_mem=131.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=96 avail_mem=131.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.92it/s] Capturing num tokens (num_tokens=96 avail_mem=131.08 GB):  81%|████████  | 47/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=80 avail_mem=131.08 GB):  81%|████████  | 47/58 [00:01<00:00, 41.10it/s]

    Capturing num tokens (num_tokens=64 avail_mem=131.07 GB):  81%|████████  | 47/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=48 avail_mem=131.07 GB):  81%|████████  | 47/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=32 avail_mem=131.06 GB):  81%|████████  | 47/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=28 avail_mem=131.06 GB):  81%|████████  | 47/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=28 avail_mem=131.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=24 avail_mem=131.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=20 avail_mem=131.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=16 avail_mem=131.04 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=12 avail_mem=131.04 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=8 avail_mem=131.03 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.11it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=131.03 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=4 avail_mem=131.03 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=4 avail_mem=131.03 GB): 100%|██████████| 58/58 [00:01<00:00, 37.63it/s]


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
    Generated text:  Isabelle and I am an 18 year old high school student.
    I am from the United States and I am a fan of anime and manga. I'm excited to start reading for pleasure and exploring new stories. I hope to be able to use my reading skills to help others and make a positive impact on the world.
    I have a lot of free time and like to stay up to date on the latest news and trends. I'm also a big fan of science fiction and fantasy, and I enjoy reading about different types of worlds and characters. What kind of books are you currently reading? How do you think your reading experience has
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office that has been held by the same person for at least 40 years. True or False?
    A. True
    B. False
    Answer: A
    
    According to the "Railway Passenger Transport Service Quality Standards", when the train speed exceeds 120 kilometers per hour, the conductor should stand on the right side of the train, facing the direction of travel, with their body about 2 meters away from the train doors. Determine if this description is correct.
    A. Correct
    B. Incorrect
    Answer: A
    
    When a train is running in reverse, the driver must set up protection before departure. What
    ===============================
    Prompt: The capital of France is
    Generated text:  in the ____.
    A. Mississippi River
    B. St. Lawrence River
    C. River Seine
    D. Nile River
    Answer:
    
    C
    
    The capital of the United States is in the ____.
    A. Mississippi River
    B. St. Lawrence River
    C. River Seine
    D. Nile River
    Answer:
    
    A
    
    The capital of France is in the ______.
    A. Mississippi River
    B. St. Lawrence River
    C. River Seine
    D. Nile River
    Answer:
    
    B
    
    The capital of France is in the ______.
    A. Mississippi River
    B. St. Lawrence River
    C.
    ===============================
    Prompt: The future of AI is
    Generated text:  coming fast and furious. With the release of the AlphaGo program by Google, the technology is now coming into the mainstream. Its use has become more and more prevalent and has led to a massive increase in the number of applications and tasks that it can perform. It is now a very important component of the technology and is, therefore, becoming more and more important in our everyday lives. The problem with the technology is that it is doing so much damage to our society and economy.
    The potential risks of the technology are as follows:
    - It has the potential to cause social unrest and protests due to the unknown and unpredictable nature of the technology and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic hub, with a rich history dating back to the Roman Empire and being home to numerous museums, theaters, and art galleries. Paris is a major transportation hub, with the Eiffel Tower serving as a symbol of the city's importance in the world. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a popular tourist destination, known for its romantic architecture, vibrant nightlife, and cultural events. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. AI developers will need to prioritize ethical considerations in their designs and ensure that they are transparent and accountable.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as blockchain, IoT, and
    


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
    Generated text:  _______________. I have always been passionate about ______________. What's your favorite hobby or activity? What is something you enjoy doing in your free time? What makes you unique and special? What is your greatest accomplishment so far? What do you hope to achieve in the future? What's your favorite book or movie? What do you love about yourself? What are your strengths and weaknesses? How do you handle difficult situations? What's your dream job? What kind of people do you admire? What are your hobbies outside of work? What are you currently doing with your free time? What kind of person are you? What's your personal
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its historical significance and is home to many famous landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. The city also hosts the Eiffel Tower Parade and the annual Festival de Paris, which is one of the largest parades in the world. The city is home to many museums, galleries, and cultural institutions, including the Musée d'Orsay and the National Museum of Modern Art. Paris is also known for its vibrant cultural scene and is a popular destination for tourists from all over the world. As of 2021, Paris had a population of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and complex, with a number of potential trends that could shape the field. Here are some of the most likely developments:
    
    1. Increased integration with human decision-making: With AI increasingly becoming a part of everyday life, it is possible that there will be an increasing degree of integration with human decision-making. This could include the ability to incorporate human preferences and biases into AI algorithms, or the use of AI to make decisions that are influenced by human emotions or empathy.
    
    2. Greater reliance on AI for tasks that were previously done manually: AI is becoming more capable of performing tasks that were previously done manually, such as language


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

    ]

     and

     I

     am

     a

     [

    N

    oun

    ]

    .
    


    I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ]

     and

     I

     live

     in

     [

    Location

    ].

     I

     am

     currently

     pursuing

     my

     [

    Major

    /

    Interest

    ]

     degree

    .

     I

     am

     passionate

     about

     [

    Why

     you

     like

     [

    Major

    /

    Interest

    ]],

     and

     I

     believe

     that

     [

    Why

     it

     matters

     to

     you

    ].

     I

     am

     confident

     in

     my

     ability

     to

     [

    What

     you

     can

     do

    ],

     and

     I

     am

     excited

     to

     contribute

     to

     [

    Name

    's

     team

    ]

    .
    


    I

     believe

     that

     our

     community

     is

     vibrant

     and

     innovative

    ,

     and

     that

     we

     can

     create

     something

     truly

     amazing

     with

     our

     work

    .

     I

     look

     forward

     to

     [

    What

     you

     can

     look

     forward

     to

    ]

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     that

     has

     hosted

     several

     world

    -ren

    owned

     artistic

     and

     cultural

     events

    ,

     including

     the

     

    1

    9

    2

    0

    s

     Olympics

     and

     the

     

    2

    0

    0

    5

     FIFA

     World

     Cup

    .


    A

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .

      


    B

    .

     Paris

     is

     the

     capital

     of

     the

     United

     Kingdom

    .

      


    C

    .

     Paris

     is

     the

     capital

     of

     Belgium

    .

      


    D

    .

     Paris

     is

     the

     capital

     of

     the

     Netherlands

    .

      


    E

    .

     Paris

     is

     not

     the

     capital

     city

     of

     France

    .

      


    F

    .

     Paris

     is

     not

     a

     capital

     city

     of

     any

     country

    .

      


    G

    .

     Paris

     is

     the

     capital

     city

     of

     Germany

    .

      


    H

    .

     Paris

     is

     the

     capital

     city

     of

     Switzerland

    .

      


    I

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     promising

    ,

     and

     it

    's

     likely

     to

     continue

     to

     develop

     and

     evolve

     in

     interesting

     and

     innovative

     ways

    .

     Here

     are

     some

     of

     the

     key

     trends

     you

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

     more

     seamless

     integration

     between

     different

     AI

     systems

     and

     applications

    .

     For

     example

    ,

     AI

    -powered

     chat

    bots

     can

     be

     integrated

     with

     customer

     service

     platforms

    ,

     providing

     personalized

     support

     to

     customers

     and

     improving

     efficiency

    .
    


    2

    .

     Self

    -

    Driving

     Technology

    :

     Self

    -driving

     technology

     is

     likely

     to

     become

     more

     advanced

     and

     widespread

    .

     This

     will

     include

     autonomous

     vehicles

     that

     can

     drive

     themselves

    ,

     as

     well

     as

     other

     autonomous

     systems

     like

     drones

     and

     robotics

    .
    


    



```python
llm.shutdown()
```
