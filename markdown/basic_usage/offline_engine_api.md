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

    [2026-03-03 22:53:40] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 22:53:40] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 22:53:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-03 22:53:43] INFO server_args.py:1973: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 22:53:43] INFO server_args.py:3064: Set soft_watchdog_timeout since in CI


    [2026-03-03 22:53:43] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=239088120, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.56it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=60.64 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=60.64 GB):   5%|▌         | 1/20 [00:00<00:03,  5.99it/s]Capturing batches (bs=120 avail_mem=59.39 GB):   5%|▌         | 1/20 [00:00<00:03,  5.99it/s]

    Capturing batches (bs=112 avail_mem=59.39 GB):   5%|▌         | 1/20 [00:00<00:03,  5.99it/s]Capturing batches (bs=104 avail_mem=59.38 GB):   5%|▌         | 1/20 [00:00<00:03,  5.99it/s]Capturing batches (bs=104 avail_mem=59.38 GB):  20%|██        | 4/20 [00:00<00:01, 15.87it/s]Capturing batches (bs=96 avail_mem=59.38 GB):  20%|██        | 4/20 [00:00<00:01, 15.87it/s] Capturing batches (bs=88 avail_mem=59.38 GB):  20%|██        | 4/20 [00:00<00:01, 15.87it/s]Capturing batches (bs=80 avail_mem=59.38 GB):  20%|██        | 4/20 [00:00<00:01, 15.87it/s]Capturing batches (bs=80 avail_mem=59.38 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.09it/s]Capturing batches (bs=72 avail_mem=59.38 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.09it/s]

    Capturing batches (bs=64 avail_mem=59.38 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.09it/s]Capturing batches (bs=56 avail_mem=59.38 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.09it/s]Capturing batches (bs=56 avail_mem=59.38 GB):  50%|█████     | 10/20 [00:00<00:00, 21.00it/s]Capturing batches (bs=48 avail_mem=59.38 GB):  50%|█████     | 10/20 [00:00<00:00, 21.00it/s]Capturing batches (bs=40 avail_mem=59.37 GB):  50%|█████     | 10/20 [00:00<00:00, 21.00it/s]Capturing batches (bs=32 avail_mem=59.37 GB):  50%|█████     | 10/20 [00:00<00:00, 21.00it/s]

    Capturing batches (bs=32 avail_mem=59.37 GB):  65%|██████▌   | 13/20 [00:00<00:00, 17.38it/s]Capturing batches (bs=24 avail_mem=59.37 GB):  65%|██████▌   | 13/20 [00:00<00:00, 17.38it/s]Capturing batches (bs=16 avail_mem=59.37 GB):  65%|██████▌   | 13/20 [00:00<00:00, 17.38it/s]Capturing batches (bs=16 avail_mem=59.37 GB):  75%|███████▌  | 15/20 [00:00<00:00, 17.25it/s]Capturing batches (bs=12 avail_mem=59.37 GB):  75%|███████▌  | 15/20 [00:00<00:00, 17.25it/s]Capturing batches (bs=8 avail_mem=59.37 GB):  75%|███████▌  | 15/20 [00:00<00:00, 17.25it/s] 

    Capturing batches (bs=8 avail_mem=59.37 GB):  85%|████████▌ | 17/20 [00:01<00:00, 15.25it/s]Capturing batches (bs=4 avail_mem=59.37 GB):  85%|████████▌ | 17/20 [00:01<00:00, 15.25it/s]Capturing batches (bs=2 avail_mem=59.37 GB):  85%|████████▌ | 17/20 [00:01<00:00, 15.25it/s]Capturing batches (bs=1 avail_mem=59.37 GB):  85%|████████▌ | 17/20 [00:01<00:00, 15.25it/s]Capturing batches (bs=1 avail_mem=59.37 GB): 100%|██████████| 20/20 [00:01<00:00, 17.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:13,  2.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:13,  2.34s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:58,  1.04s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:58,  1.04s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:58,  1.04s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.35it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.35it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.35it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:05,  9.28it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:05,  9.28it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:05,  9.28it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:05,  9.28it/s]

    Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:05,  9.28it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 14.11it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 14.11it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 14.11it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 14.11it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 14.11it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:03, 14.11it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:01, 20.13it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:01, 20.13it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:01, 20.13it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:01, 20.13it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:01, 20.13it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 24.11it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 24.11it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 24.11it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 24.11it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 24.11it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 24.11it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:00, 29.66it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 33.62it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 33.62it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 33.62it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 33.62it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 33.62it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]

    Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 41.13it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s] 

    Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 45.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.52 GB):   2%|▏         | 1/58 [00:00<00:05,  9.59it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.52 GB):   2%|▏         | 1/58 [00:00<00:05,  9.59it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.52 GB):   3%|▎         | 2/58 [00:00<00:07,  7.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.55 GB):   3%|▎         | 2/58 [00:00<00:07,  7.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.55 GB):   3%|▎         | 2/58 [00:00<00:07,  7.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.55 GB):   7%|▋         | 4/58 [00:00<00:05,  9.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.02 GB):   7%|▋         | 4/58 [00:00<00:05,  9.20it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.58 GB):   7%|▋         | 4/58 [00:00<00:05,  9.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.58 GB):  10%|█         | 6/58 [00:00<00:05, 10.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.01 GB):  10%|█         | 6/58 [00:00<00:05, 10.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.61 GB):  10%|█         | 6/58 [00:00<00:05, 10.20it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.61 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.00 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.64 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.00 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.66 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.41it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.66 GB):  21%|██        | 12/58 [00:01<00:03, 13.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.99 GB):  21%|██        | 12/58 [00:01<00:03, 13.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.69 GB):  21%|██        | 12/58 [00:01<00:03, 13.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.69 GB):  24%|██▍       | 14/58 [00:01<00:02, 14.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.98 GB):  24%|██▍       | 14/58 [00:01<00:02, 14.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.98 GB):  24%|██▍       | 14/58 [00:01<00:02, 14.81it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=58.98 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.74 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.97 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.97 GB):  31%|███       | 18/58 [00:01<00:02, 16.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.76 GB):  31%|███       | 18/58 [00:01<00:02, 16.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.96 GB):  31%|███       | 18/58 [00:01<00:02, 16.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.94 GB):  31%|███       | 18/58 [00:01<00:02, 16.70it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.94 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.72it/s]Capturing num tokens (num_tokens=960 avail_mem=58.96 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.72it/s] Capturing num tokens (num_tokens=896 avail_mem=58.83 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.72it/s]Capturing num tokens (num_tokens=832 avail_mem=58.83 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.72it/s]Capturing num tokens (num_tokens=832 avail_mem=58.83 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.83it/s]Capturing num tokens (num_tokens=768 avail_mem=58.94 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.83it/s]Capturing num tokens (num_tokens=704 avail_mem=58.93 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.83it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.93 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.83it/s]Capturing num tokens (num_tokens=640 avail_mem=58.93 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.40it/s]Capturing num tokens (num_tokens=576 avail_mem=58.92 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.40it/s]Capturing num tokens (num_tokens=512 avail_mem=58.91 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.40it/s]Capturing num tokens (num_tokens=480 avail_mem=58.92 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.40it/s]Capturing num tokens (num_tokens=480 avail_mem=58.92 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=448 avail_mem=58.93 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=416 avail_mem=58.85 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.12it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.85 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=384 avail_mem=58.85 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=352 avail_mem=58.84 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=320 avail_mem=58.84 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=288 avail_mem=58.88 GB):  57%|█████▋    | 33/58 [00:02<00:01, 23.75it/s]Capturing num tokens (num_tokens=256 avail_mem=58.87 GB):  57%|█████▋    | 33/58 [00:02<00:01, 23.75it/s]Capturing num tokens (num_tokens=256 avail_mem=58.87 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.78it/s]Capturing num tokens (num_tokens=240 avail_mem=58.86 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.78it/s]Capturing num tokens (num_tokens=224 avail_mem=58.86 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.78it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.86 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.78it/s]Capturing num tokens (num_tokens=192 avail_mem=58.85 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.78it/s]Capturing num tokens (num_tokens=192 avail_mem=58.85 GB):  71%|███████   | 41/58 [00:02<00:00, 27.95it/s]Capturing num tokens (num_tokens=176 avail_mem=58.84 GB):  71%|███████   | 41/58 [00:02<00:00, 27.95it/s]Capturing num tokens (num_tokens=160 avail_mem=58.83 GB):  71%|███████   | 41/58 [00:02<00:00, 27.95it/s]Capturing num tokens (num_tokens=144 avail_mem=58.83 GB):  71%|███████   | 41/58 [00:02<00:00, 27.95it/s]Capturing num tokens (num_tokens=128 avail_mem=58.82 GB):  71%|███████   | 41/58 [00:02<00:00, 27.95it/s]Capturing num tokens (num_tokens=128 avail_mem=58.82 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.22it/s]Capturing num tokens (num_tokens=112 avail_mem=58.81 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.22it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.82 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.22it/s] Capturing num tokens (num_tokens=80 avail_mem=58.81 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.22it/s]Capturing num tokens (num_tokens=64 avail_mem=58.79 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.22it/s]Capturing num tokens (num_tokens=64 avail_mem=58.79 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=48 avail_mem=58.80 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=32 avail_mem=58.77 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=28 avail_mem=58.76 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=24 avail_mem=58.76 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.12it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.76 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=20 avail_mem=58.78 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=16 avail_mem=58.77 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=12 avail_mem=58.76 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.12it/s]Capturing num tokens (num_tokens=8 avail_mem=58.76 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.12it/s] Capturing num tokens (num_tokens=8 avail_mem=58.76 GB):  98%|█████████▊| 57/58 [00:02<00:00, 31.99it/s]Capturing num tokens (num_tokens=4 avail_mem=58.73 GB):  98%|█████████▊| 57/58 [00:02<00:00, 31.99it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.73 GB): 100%|██████████| 58/58 [00:02<00:00, 20.41it/s]


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
    Generated text:  Tim. I am a 17 year old boy from Beijing, China. I am in Grade Seven. I am very good at Maths and Science. I like sports like basketball and swimming. I like reading books. I am a good student. I have a problem now. When I have to take a test, I always get very nervous. I feel so sad. I always feel worried and I can't think of anything to do. I want to ask my teachers and friends for help. What should I do? Can you tell me what to do? 1. How to calm down when you feel nervous? 2.
    ===============================
    Prompt: The president of the United States is
    Generated text:  36 years old now. In 20 years, if the president is 36 years old again, how old would he be? To determine the president's age in 20 years if he is currently 36 years old, we can follow these steps:
    
    1. Identify the current age of the president: 36 years old.
    2. Determine the future age of the president in 20 years: \(36 + 20 = 56\) years old.
    3. Since we need to find out how old the president would be in 20 years if he is currently 3
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. London
    C. Moscow
    D. Tokyo
    
    To determine the capital of France, we need to understand the structure of the European Union and identify the country that is part of it. The European Union consists of 27 member countries, and each member country is a member of the Union. The capital of France is typically the capital of the country that was a member of the Union at the time it was established.
    
    Let's examine each option:
    
    A. Paris - Paris is the capital of France, but it is not part of the European Union.
    B. London - London is the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting, with applications in healthcare, finance, transportation, and so much more. One of the most promising areas of AI is machine learning, a subset of the broader field of artificial intelligence. But what is machine learning?
    What is Machine Learning?
    Machine learning is the process of building a model of data, using algorithms and statistical analysis to learn patterns in the data. The goal of machine learning is to make predictions or decisions based on the patterns and relationships that exist in the data.
    This can be useful for a wide variety of applications, including:
    - Medical diagnosis and treatment
    - Fraud detection and prevention
    - Personalized marketing and recommendations


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and challenges, and I'm always eager to learn and grow. What's your favorite hobby or activity? I'm always looking for new adventures and experiences, and I'm always eager to try new things. What's your favorite book or movie? I'm always looking for new ideas and inspiration
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and diverse culture. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a cultural hub for France and the world. It is home to many museums, theaters, and other cultural institutions. The city is also known for its annual festivals and events, such as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that could be expected in the future:
    
    1. Increased automation: As AI continues to improve, it is likely to become more and more integrated into our daily lives. This could lead to a significant increase in automation, with machines taking on many of the tasks that are currently performed by humans.
    
    2. Enhanced privacy: As AI becomes more integrated into our lives, there is a risk that it could lead to a loss of privacy. This is because AI systems are designed to learn from data
    


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
    Generated text:  [Name] and I'm a [Job Title] who has [X amount of years of experience]. I enjoy [some hobby or activity that I find interesting]. How can someone best describe me? What do you do for a living?
    Hello, my name is [Name] and I'm a [Job Title] who has [X amount of years of experience]. I enjoy [some hobby or activity that I find interesting]. How can someone best describe me? What do you do for a living? As an AI language model, I don't have a physical presence, so I can't have jobs or hobbies. However, I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Light" and the "City of Culture."
    
    This statement encapsulates the main facts about Paris, including its importance as the capital city of France, its designation as the "City of Light" and the "City of Culture," which are widely recognized as its prominent characteristics. It succinctly conveys the significance of Paris in terms of cultural, economic, and political influence within the country. To provide a more detailed and up-to-date version, Paris is often referred to as "la Ville Lumière" (the City of Light) due to its bright lights and warm, sunny climate, which have contributed
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undeniably rapidly evolving, with significant potential for both exciting developments and significant challenges. Here are some possible future trends in AI:
    
    1. Increased Use of AI in Healthcare: AI will continue to have a significant impact on healthcare, particularly in areas such as diagnosis, patient care, and treatment planning. AI will be used to analyze large amounts of medical data, identify patterns, and predict disease outcomes, ultimately leading to improved health outcomes.
    
    2. Deep Learning: Deep learning will continue to advance, with more advanced neural networks and models that can learn from large amounts of data. This will enable AI to better understand and interpret complex natural language and


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

    ].

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

    ].

     I

     have

     a

     [

    Number

    ]

     years

     of

     experience

     in

     [

    Field

     of

     Work

    ].

     I

    'm

     a

     [

    Status

    ]

     and

     I

     like

     to

     [

    Favorite

     Activity

    ].

     I

    'm

     [

    Gender

    ]

     and

     I

     have

     [

    Height

    /

    Weight

    ]

     [

    Country

    ]

     and

     [

    City

    ].

     I

    'm

     a

     [

    Friend

    ]

     to

     [

    Friend

    's

     Name

    ].

     I

     like

     to

     [

    Favorite

     Hobby

    /

    Activity

    ].

     I

    'm

     [

    Age

    ]

     years

     old

     and

     I

     believe

     in

     [

    L

    ifestyle

    ].

     I

    'm

     [

    Location

    ]

     and

     I

     love

     to

     [

    Exc

    use

    ]

     [

    M

    ood

    ].

     I

     am

     always

     [

    In

    spir

    ational

     Quote

    ]

     in

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

    ,

     renowned

     museums

    ,

     and

     vibrant

     artistic

     scene

    .

     It

    's

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

    ,

     including

     several

     UNESCO

     World

     Heritage

     sites

    .

     Paris

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

     It

    's

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     delicious

     cuisine

    .

     Paris is

     a

     city

     of

     contrasts

    ,

     and

     it

    's

     a

     favorite

     destination

     for

     tourists

     from

     all

     over

     the

     world

    .

     The

     city

     also

     boasts

     a

     thriving

     music

     scene

    ,

     with

     numerous

     bars

    ,

     clubs

    ,

     and

     festivals

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     offers

     something

     for

     everyone

    ,

     from

     its

     delicious

     food

     and

     art

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     combination

     of

     technological

     advancement

    ,

     integration

     with

     other

     areas

     of

     technology

    ,

     and

     the

     growth

     of

     ethical

     considerations

     and

     implications

     for

     society

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     More

     complex

     decision

    -making

    :

     AI

     systems

     are

     becoming

     more

     complex

    ,

     able

     to

     make

     decisions

     that

     were

     previously

     difficult

     for

     humans

     to

     accomplish

    .

     This

     will

     require

     more

     sophisticated

     algorithms

     and

     machine

     learning

     techniques

     to

     accurately

     understand

     and

     make

     decisions

     in

     a

     wide

     range

     of

     situations

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     with

     other

     technologies

     such

     as

     speech

     recognition

    ,

     facial

     recognition

    ,

     and

     natural

     language

     processing

    .

     As

     these

     technologies

     continue

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     even

     more

     integration

    



```python
llm.shutdown()
```
