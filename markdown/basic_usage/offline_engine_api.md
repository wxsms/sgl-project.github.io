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

    [2026-03-03 06:27:55] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 06:27:55] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 06:27:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-03 06:27:57] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 06:27:57] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 06:27:57] INFO engine.py:157: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=680521191, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.45it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=117.97 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=117.97 GB):   5%|▌         | 1/20 [00:00<00:04,  4.59it/s]Capturing batches (bs=120 avail_mem=117.86 GB):   5%|▌         | 1/20 [00:00<00:04,  4.59it/s]Capturing batches (bs=112 avail_mem=117.86 GB):   5%|▌         | 1/20 [00:00<00:04,  4.59it/s]Capturing batches (bs=104 avail_mem=117.85 GB):   5%|▌         | 1/20 [00:00<00:04,  4.59it/s]Capturing batches (bs=104 avail_mem=117.85 GB):  20%|██        | 4/20 [00:00<00:01, 12.92it/s]Capturing batches (bs=96 avail_mem=117.85 GB):  20%|██        | 4/20 [00:00<00:01, 12.92it/s] Capturing batches (bs=88 avail_mem=117.84 GB):  20%|██        | 4/20 [00:00<00:01, 12.92it/s]

    Capturing batches (bs=80 avail_mem=117.84 GB):  20%|██        | 4/20 [00:00<00:01, 12.92it/s]Capturing batches (bs=80 avail_mem=117.84 GB):  35%|███▌      | 7/20 [00:00<00:00, 16.69it/s]Capturing batches (bs=72 avail_mem=117.83 GB):  35%|███▌      | 7/20 [00:00<00:00, 16.69it/s]Capturing batches (bs=64 avail_mem=117.83 GB):  35%|███▌      | 7/20 [00:00<00:00, 16.69it/s]Capturing batches (bs=56 avail_mem=117.82 GB):  35%|███▌      | 7/20 [00:00<00:00, 16.69it/s]Capturing batches (bs=56 avail_mem=117.82 GB):  50%|█████     | 10/20 [00:00<00:00, 18.57it/s]Capturing batches (bs=48 avail_mem=117.82 GB):  50%|█████     | 10/20 [00:00<00:00, 18.57it/s]

    Capturing batches (bs=40 avail_mem=117.81 GB):  50%|█████     | 10/20 [00:00<00:00, 18.57it/s]Capturing batches (bs=32 avail_mem=117.81 GB):  50%|█████     | 10/20 [00:00<00:00, 18.57it/s]Capturing batches (bs=32 avail_mem=117.81 GB):  65%|██████▌   | 13/20 [00:00<00:00, 19.39it/s]Capturing batches (bs=24 avail_mem=117.51 GB):  65%|██████▌   | 13/20 [00:00<00:00, 19.39it/s]Capturing batches (bs=16 avail_mem=117.50 GB):  65%|██████▌   | 13/20 [00:00<00:00, 19.39it/s]

    Capturing batches (bs=12 avail_mem=117.50 GB):  65%|██████▌   | 13/20 [00:00<00:00, 19.39it/s]Capturing batches (bs=12 avail_mem=117.50 GB):  80%|████████  | 16/20 [00:00<00:00, 17.71it/s]Capturing batches (bs=8 avail_mem=117.49 GB):  80%|████████  | 16/20 [00:00<00:00, 17.71it/s] Capturing batches (bs=4 avail_mem=117.49 GB):  80%|████████  | 16/20 [00:01<00:00, 17.71it/s]Capturing batches (bs=4 avail_mem=117.49 GB):  90%|█████████ | 18/20 [00:01<00:00, 18.19it/s]Capturing batches (bs=2 avail_mem=117.48 GB):  90%|█████████ | 18/20 [00:01<00:00, 18.19it/s]Capturing batches (bs=1 avail_mem=117.48 GB):  90%|█████████ | 18/20 [00:01<00:00, 18.19it/s]

    Capturing batches (bs=1 avail_mem=117.48 GB): 100%|██████████| 20/20 [00:01<00:00, 17.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:08<08:17,  8.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:08<08:17,  8.72s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:08<03:24,  3.65s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:08<03:24,  3.65s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:08<03:24,  3.65s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:08<01:15,  1.41s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:08<01:15,  1.41s/it]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:09<01:15,  1.41s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:09<00:39,  1.30it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:09<00:39,  1.30it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:09<00:39,  1.30it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:09<00:39,  1.30it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:19,  2.49it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:19,  2.49it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:09<00:19,  2.49it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:09<00:19,  2.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:11,  4.02it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:11,  4.02it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:09<00:11,  4.02it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:09<00:11,  4.02it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:07,  5.92it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:09<00:07,  5.92it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:09<00:07,  5.92it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:09<00:07,  5.92it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:09<00:07,  5.92it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:03,  9.95it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:03,  9.95it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:03,  9.95it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:09<00:03,  9.95it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:09<00:03,  9.95it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 13.47it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 13.47it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 13.47it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:09<00:02, 13.47it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:09<00:02, 13.47it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:01, 17.28it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:01, 17.28it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:01, 17.28it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:01, 17.28it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:01, 17.28it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:09<00:01, 17.28it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:01, 22.82it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:01, 22.82it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:01, 22.82it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:09<00:01, 22.82it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:09<00:01, 22.82it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:09<00:00, 26.02it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:09<00:00, 26.02it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:09<00:00, 26.02it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:09<00:00, 26.02it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:09<00:00, 26.02it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:10<00:00, 26.02it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:10<00:00, 31.16it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:10<00:00, 31.16it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:10<00:00, 31.16it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:10<00:00, 31.16it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:10<00:00, 31.16it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:10<00:00, 31.16it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:10<00:00, 35.09it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:10<00:00, 42.69it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:10<00:00, 42.69it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:10<00:00, 42.69it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:10<00:00, 42.69it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:10<00:00, 42.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.90 GB):   2%|▏         | 1/58 [00:00<00:07,  7.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.43 GB):   2%|▏         | 1/58 [00:00<00:07,  7.22it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=116.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.45 GB):   5%|▌         | 3/58 [00:00<00:06,  8.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.84 GB):   5%|▌         | 3/58 [00:00<00:06,  8.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.46 GB):   5%|▌         | 3/58 [00:00<00:06,  8.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.46 GB):   9%|▊         | 5/58 [00:00<00:05, 10.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.82 GB):   9%|▊         | 5/58 [00:00<00:05, 10.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.81 GB):   9%|▊         | 5/58 [00:00<00:05, 10.04it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=116.81 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.50 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.79 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.79 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.57 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.88it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.78 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.88it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.78 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.77 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.57 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.76 GB):  19%|█▉        | 11/58 [00:01<00:03, 13.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.76 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.76 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.77 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.96it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=116.72 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.72 GB):  29%|██▉       | 17/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.73 GB):  29%|██▉       | 17/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.72 GB):  29%|██▉       | 17/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.72 GB):  29%|██▉       | 17/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.72 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.91it/s]

    Capturing num tokens (num_tokens=960 avail_mem=116.60 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.91it/s] Capturing num tokens (num_tokens=896 avail_mem=116.68 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.91it/s]Capturing num tokens (num_tokens=896 avail_mem=116.68 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=832 avail_mem=116.68 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=768 avail_mem=116.67 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=704 avail_mem=116.66 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=704 avail_mem=116.66 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.44it/s]Capturing num tokens (num_tokens=640 avail_mem=116.65 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.44it/s]

    Capturing num tokens (num_tokens=576 avail_mem=116.64 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.44it/s]Capturing num tokens (num_tokens=512 avail_mem=116.62 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.44it/s]Capturing num tokens (num_tokens=512 avail_mem=116.62 GB):  50%|█████     | 29/58 [00:01<00:01, 24.74it/s]Capturing num tokens (num_tokens=480 avail_mem=116.63 GB):  50%|█████     | 29/58 [00:01<00:01, 24.74it/s]Capturing num tokens (num_tokens=448 avail_mem=116.62 GB):  50%|█████     | 29/58 [00:01<00:01, 24.74it/s]Capturing num tokens (num_tokens=416 avail_mem=116.61 GB):  50%|█████     | 29/58 [00:01<00:01, 24.74it/s]Capturing num tokens (num_tokens=416 avail_mem=116.61 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=384 avail_mem=116.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]

    Capturing num tokens (num_tokens=352 avail_mem=116.59 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=320 avail_mem=116.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=288 avail_mem=116.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=288 avail_mem=116.57 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.88it/s]Capturing num tokens (num_tokens=256 avail_mem=116.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.88it/s]Capturing num tokens (num_tokens=240 avail_mem=116.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.88it/s]Capturing num tokens (num_tokens=224 avail_mem=116.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.88it/s]Capturing num tokens (num_tokens=208 avail_mem=116.53 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.88it/s]

    Capturing num tokens (num_tokens=208 avail_mem=116.53 GB):  69%|██████▉   | 40/58 [00:02<00:00, 29.46it/s]Capturing num tokens (num_tokens=192 avail_mem=116.54 GB):  69%|██████▉   | 40/58 [00:02<00:00, 29.46it/s]Capturing num tokens (num_tokens=176 avail_mem=116.50 GB):  69%|██████▉   | 40/58 [00:02<00:00, 29.46it/s]Capturing num tokens (num_tokens=160 avail_mem=116.49 GB):  69%|██████▉   | 40/58 [00:02<00:00, 29.46it/s]Capturing num tokens (num_tokens=144 avail_mem=116.50 GB):  69%|██████▉   | 40/58 [00:02<00:00, 29.46it/s]Capturing num tokens (num_tokens=144 avail_mem=116.50 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.94it/s]Capturing num tokens (num_tokens=128 avail_mem=116.49 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.94it/s]Capturing num tokens (num_tokens=112 avail_mem=116.48 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.94it/s]Capturing num tokens (num_tokens=96 avail_mem=116.47 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.94it/s] Capturing num tokens (num_tokens=80 avail_mem=116.47 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.94it/s]

    Capturing num tokens (num_tokens=80 avail_mem=116.47 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=64 avail_mem=116.46 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=48 avail_mem=116.45 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=32 avail_mem=116.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=28 avail_mem=116.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=28 avail_mem=116.43 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.19it/s]Capturing num tokens (num_tokens=24 avail_mem=116.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.19it/s]Capturing num tokens (num_tokens=20 avail_mem=116.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.19it/s]Capturing num tokens (num_tokens=16 avail_mem=116.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.19it/s]Capturing num tokens (num_tokens=12 avail_mem=116.38 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.19it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.38 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.27it/s]Capturing num tokens (num_tokens=8 avail_mem=116.37 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.27it/s] Capturing num tokens (num_tokens=4 avail_mem=116.37 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.27it/s]Capturing num tokens (num_tokens=4 avail_mem=116.37 GB): 100%|██████████| 58/58 [00:02<00:00, 23.21it/s]


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
    Generated text:  Nikita Ivanov. I'm a beginner in Python programming. I need help with some code related to the multiplication of two numbers in Python. Can you provide a solution to multiply two numbers without using any inbuilt multiplication function? I want to understand the logic behind this solution. Sure! You can solve this problem by using a loop. The idea is to multiply each digit of the first number by each digit of the second number and store the result in a new list. Then, you can sort the list in ascending order and return the product. Here's how you can do it:
    
    ```python
    def multiply(x, y):
       
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a new term of office at the end of the year. In order to complete the presidency, he will have to take a lot of decisions every day, which means that the number of his daily tasks will increase in every day. The president's schedule in the previous year was planned as follows: he worked 12 hours on Mondays, 15 hours on Tuesdays, 20 hours on Wednesdays, 25 hours on Thursdays, and 10 hours on Fridays. If the president's daily tasks increase by 10% each day, how many hours will he have to work on the last
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which is the largest city in France?
    A. London
    B. Amsterdam
    C. Paris
    D. Berlin
    
    To determine the largest city in France, we need to consider the following information:
    
    1. The capital of France is Paris.
    2. The other options (London, Amsterdam, and Berlin) are smaller cities located in different countries.
    
    Let's analyze each option:
    
    A. London: London is the capital of the United Kingdom and is the largest city by population in the United Kingdom. However, it is not the largest city in France.
    
    B. Amsterdam: Amsterdam is the capital of the Netherlands and is the largest
    ===============================
    Prompt: The future of AI is
    Generated text:  cloudy. On the one hand, we have incredible advancements, including machine learning, deep learning, and the proliferation of smaller, more affordable systems. On the other, there are growing concerns about the potential negative impacts of AI on people and society. As the technology becomes more advanced, so does the public debate about its impact. In this blog, we will explore the current state of AI and the questions that it raises about our society and our future.
    In this blog, we will start by discussing the history of AI. We will explore the concept of AI, its potential applications, and its challenges. We will also examine the current state of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also home to the Louvre Museum, the most famous art museum in the world, and the Notre-Dame Cathedral. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is also known for its fashion industry and its role in the French Revolution. The city is home to many famous landmarks and attractions, including the Palace of Versailles and the Champs-Élysées. Paris is a major transportation hub, with the Eiffel Tower serving as a symbol
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to healthcare. Automation will likely become more prevalent, with machines taking on tasks that were previously done by humans, such as data entry, customer service, and administrative tasks.
    
    2. AI ethics and privacy: As AI becomes more integrated into our lives, there will be increasing concerns about its impact on society. There will likely be a need for more ethical guidelines and regulations to ensure
    


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
    Generated text:  [Name]. I'm a friendly and energetic 24-year-old college student who loves to explore new things and try new things. I'm passionate about music, travel, and learning new things. I'm always ready to jump into a new challenge and be someone who inspires others to do the same. And, while I'm not a bookworm, I have a deep appreciation for literature and history. If you're looking for someone who can help with a specific project or have a question about a particular topic, I'm here to listen and offer my expertise. I look forward to talking to you soon! How can I get in touch
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is located in the south of the country and is the largest city in Europe by population. It is known for its rich history, art, and cuisine, as well as its significant role in French culture and politics. Paris has been a major economic and cultural center for over 2000 years, and is home to many renowned museums, theaters, and landmarks. Its location at the crossroads of Europe and Asia makes it an important transportation hub and gateway to the Mediterranean. Paris is also a popular tourist destination, offering numerous attractions, dining options, and cultural experiences for visitors. Overall, Paris is a vibrant, dynamic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of factors, including advancements in hardware, the development of new algorithms, and the integration of AI into existing systems. Here are some possible future trends in AI:
    
    1. Increased intelligence and learning abilities: AI is likely to become even more intelligent and capable over the coming years. Advances in deep learning, natural language processing, and machine learning will allow AI to learn from new data and adapt to new situations more effectively.
    
    2. Greater focus on ethical considerations: As AI becomes more integrated into everyday life, there will be a growing focus on ethical considerations such as bias, transparency, and fairness. Governments and organizations


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

     Jane

     Smith

    .

     I

     am

     a

     highly

     skilled

     and

     experienced

     professional

     in

     the

     field

     of

     software

     engineering

    .

     I

     have

     a

     passion

     for

     learning

     and

     always

     aim

     to

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     and

     technologies

     in

     the

     industry

    .

     My

     team

     and

     I

     are

     constantly

     improving

     our

     skills

     and

     knowledge

     to

     stay

     ahead

     of

     the

     competition

    .

     I

     am

     committed

     to

     helping

     others

     succeed

     and

     I

     am

     always

     eager

     to

     share

     my

     knowledge

     and

     expertise

    .

     Whether

     it

     is

     working

     on

     a

     complex

     project

     or

     just

     helping

     someone

     with

     a

     question

    ,

     I

     am

     always

     there

     to

     support

     you

    .

     I

     believe

     that

     hard

     work

     and

     perseverance

     are

     key

     to

     achieving

     success

     in

     any

     field

    ,

     and

     I

     am

     excited

     to

     help

     you

     too

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     the

     country

     of

     France

    ,

     located

     in

     the

     central

     region

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     both

     France

     and

     Europe

     and

     is

     the

     world

    's

     

    6

    th

     most

     populous

     city

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

    ,

     fashion

     industry

    ,

     and

     rich

     history

    ,

     and

     is

     a

     major

     tourist

     destination

    .

     The

     city

     is

     home

     to

     many

     important

     governmental

     offices

     and

     institutions

    ,

     including

     the

     French

     Parliament

     and

     the

     French

     Supreme

     Court

    .

     Paris

     is

     also

     known

     for

     its

     cuisine

    ,

     music

    ,

     and

     arts

     scene

    ,

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

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     increasingly

     focused

     on

     its

     ability

     to

     improve

     and

     evolve

    ,

     rather

     than

     to

     dominate

     or

     control

     it

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

     Integration

     with

     Human

     Intelligence

    :

     AI

     is

     expected

     to

     become

     more

     integrated

     with

     human

     intelligence

    ,

     enabling

     machines

     to

     understand

     and

     learn

     from

     human

     experiences

    .

     This

     could

     lead

     to

     more

     complex

     and

     nuanced

     AI

     systems

     that

     can

     interact

     with

     humans

     in

     novel

     and

     creative

     ways

    .
    


    2

    .

     Enhanced

     Data

    -

    Driven

     Decision

    -M

    aking

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     analyze

     vast

     amounts

     of

     data

     more

     quickly

     and

     accurately

     than

     humans

    .

     This

     will

     enable

     machines

     to

     make

     more

     informed

     and

     informed

     decisions

    ,

     improving

     their

     performance

     in

     various

     applications

    



```python
llm.shutdown()
```
