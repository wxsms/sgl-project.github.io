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

    [2026-03-09 02:46:50] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-09 02:46:50] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-09 02:46:50] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-09 02:46:52] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-09 02:46:52] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-09 02:46:52] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=19936709, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.67it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.66it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]Capturing batches (bs=120 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]

    Capturing batches (bs=112 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]Capturing batches (bs=104 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]Capturing batches (bs=104 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:00, 16.06it/s]Capturing batches (bs=96 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:00, 16.06it/s] Capturing batches (bs=88 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:00, 16.06it/s]Capturing batches (bs=80 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:00, 16.06it/s]Capturing batches (bs=80 avail_mem=58.50 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.35it/s]Capturing batches (bs=72 avail_mem=58.50 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.35it/s]

    Capturing batches (bs=64 avail_mem=58.50 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.35it/s]Capturing batches (bs=56 avail_mem=58.49 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.35it/s]Capturing batches (bs=56 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.69it/s]Capturing batches (bs=48 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.69it/s]Capturing batches (bs=40 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.69it/s]Capturing batches (bs=32 avail_mem=58.49 GB):  50%|█████     | 10/20 [00:00<00:00, 22.69it/s]Capturing batches (bs=32 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.64it/s]Capturing batches (bs=24 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.64it/s]

    Capturing batches (bs=16 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.64it/s]Capturing batches (bs=12 avail_mem=58.49 GB):  65%|██████▌   | 13/20 [00:00<00:00, 24.64it/s]Capturing batches (bs=12 avail_mem=58.49 GB):  80%|████████  | 16/20 [00:00<00:00, 22.96it/s]Capturing batches (bs=8 avail_mem=58.49 GB):  80%|████████  | 16/20 [00:00<00:00, 22.96it/s] Capturing batches (bs=4 avail_mem=58.49 GB):  80%|████████  | 16/20 [00:00<00:00, 22.96it/s]Capturing batches (bs=2 avail_mem=58.49 GB):  80%|████████  | 16/20 [00:00<00:00, 22.96it/s]

    Capturing batches (bs=1 avail_mem=58.48 GB):  80%|████████  | 16/20 [00:00<00:00, 22.96it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 20/20 [00:00<00:00, 26.10it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 20/20 [00:00<00:00, 22.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:08,  2.25s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:55,  1.01it/s]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:55,  1.01it/s]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:55,  1.01it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.45it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.45it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.45it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:06,  7.18it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:06,  7.18it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:06,  7.18it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:06,  7.18it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:04, 10.53it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:04, 10.53it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:04, 10.53it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:04, 10.53it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:04, 10.53it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:02<00:02, 15.09it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:02<00:02, 15.09it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:02<00:02, 15.09it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:02<00:02, 15.09it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:02, 15.09it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:01, 19.82it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:01, 19.82it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:01, 19.82it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:01, 19.82it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:01, 19.82it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:01, 19.82it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 25.47it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 25.47it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 25.47it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 25.47it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 25.47it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 25.47it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:00, 30.12it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 37.75it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 42.18it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 49.38it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.51 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.51 GB):   2%|▏         | 1/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=30.74 GB):   2%|▏         | 1/58 [00:00<00:07,  7.36it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=30.74 GB):   3%|▎         | 2/58 [00:00<00:07,  7.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=30.74 GB):   3%|▎         | 2/58 [00:00<00:07,  7.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=30.74 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=30.73 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=30.73 GB):   7%|▋         | 4/58 [00:00<00:06,  8.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=30.74 GB):   7%|▋         | 4/58 [00:00<00:06,  8.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=30.74 GB):   9%|▊         | 5/58 [00:00<00:06,  8.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=30.73 GB):   9%|▊         | 5/58 [00:00<00:06,  8.42it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=30.73 GB):  10%|█         | 6/58 [00:00<00:05,  8.84it/s]Capturing num tokens (num_tokens=5120 avail_mem=30.73 GB):  10%|█         | 6/58 [00:00<00:05,  8.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=30.73 GB):  10%|█         | 6/58 [00:00<00:05,  8.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=30.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=30.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.68it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=30.73 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=30.73 GB):  17%|█▋        | 10/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=30.72 GB):  17%|█▋        | 10/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=30.72 GB):  17%|█▋        | 10/58 [00:01<00:04, 10.37it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=30.72 GB):  21%|██        | 12/58 [00:01<00:04, 10.82it/s]Capturing num tokens (num_tokens=3072 avail_mem=30.72 GB):  21%|██        | 12/58 [00:01<00:04, 10.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=30.71 GB):  21%|██        | 12/58 [00:01<00:04, 10.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=30.71 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=30.71 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.04it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=30.70 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=30.70 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=30.70 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=30.70 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.51it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=30.70 GB):  31%|███       | 18/58 [00:01<00:03, 12.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=30.69 GB):  31%|███       | 18/58 [00:01<00:03, 12.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=30.69 GB):  31%|███       | 18/58 [00:01<00:03, 12.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=30.69 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=30.67 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=30.69 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.66it/s] Capturing num tokens (num_tokens=960 avail_mem=30.69 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.18it/s]Capturing num tokens (num_tokens=896 avail_mem=30.68 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.18it/s]Capturing num tokens (num_tokens=832 avail_mem=30.68 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.18it/s]

    Capturing num tokens (num_tokens=832 avail_mem=30.68 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.39it/s]Capturing num tokens (num_tokens=768 avail_mem=30.67 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.39it/s]Capturing num tokens (num_tokens=704 avail_mem=30.67 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.39it/s]Capturing num tokens (num_tokens=704 avail_mem=30.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.67it/s]Capturing num tokens (num_tokens=640 avail_mem=30.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.67it/s]

    Capturing num tokens (num_tokens=576 avail_mem=30.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.67it/s]Capturing num tokens (num_tokens=576 avail_mem=30.67 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=512 avail_mem=30.65 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=480 avail_mem=30.67 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.80it/s]

    Capturing num tokens (num_tokens=480 avail_mem=30.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.14it/s]Capturing num tokens (num_tokens=448 avail_mem=30.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.14it/s]Capturing num tokens (num_tokens=416 avail_mem=30.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.14it/s]Capturing num tokens (num_tokens=416 avail_mem=30.67 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.10it/s]Capturing num tokens (num_tokens=384 avail_mem=30.66 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=30.66 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.10it/s]Capturing num tokens (num_tokens=352 avail_mem=30.66 GB):  59%|█████▊    | 34/58 [00:02<00:01, 14.22it/s]Capturing num tokens (num_tokens=320 avail_mem=30.66 GB):  59%|█████▊    | 34/58 [00:02<00:01, 14.22it/s]Capturing num tokens (num_tokens=288 avail_mem=30.65 GB):  59%|█████▊    | 34/58 [00:02<00:01, 14.22it/s]

    Capturing num tokens (num_tokens=288 avail_mem=30.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.23it/s]Capturing num tokens (num_tokens=256 avail_mem=30.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.23it/s]Capturing num tokens (num_tokens=240 avail_mem=30.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.23it/s]Capturing num tokens (num_tokens=240 avail_mem=30.65 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.98it/s]Capturing num tokens (num_tokens=224 avail_mem=30.65 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.98it/s]

    Capturing num tokens (num_tokens=208 avail_mem=30.64 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.98it/s]Capturing num tokens (num_tokens=208 avail_mem=30.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.14it/s]Capturing num tokens (num_tokens=192 avail_mem=30.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.14it/s]Capturing num tokens (num_tokens=176 avail_mem=30.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.14it/s]

    Capturing num tokens (num_tokens=176 avail_mem=30.64 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=160 avail_mem=30.63 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=144 avail_mem=30.63 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=144 avail_mem=30.63 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.93it/s]Capturing num tokens (num_tokens=128 avail_mem=30.63 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.93it/s]

    Capturing num tokens (num_tokens=112 avail_mem=30.63 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.93it/s]Capturing num tokens (num_tokens=112 avail_mem=30.63 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.07it/s]Capturing num tokens (num_tokens=96 avail_mem=30.62 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.07it/s] Capturing num tokens (num_tokens=80 avail_mem=30.62 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.07it/s]

    Capturing num tokens (num_tokens=80 avail_mem=30.62 GB):  83%|████████▎ | 48/58 [00:03<00:00, 14.18it/s]Capturing num tokens (num_tokens=64 avail_mem=30.62 GB):  83%|████████▎ | 48/58 [00:03<00:00, 14.18it/s]Capturing num tokens (num_tokens=48 avail_mem=30.61 GB):  83%|████████▎ | 48/58 [00:03<00:00, 14.18it/s]Capturing num tokens (num_tokens=48 avail_mem=30.61 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.16it/s]Capturing num tokens (num_tokens=32 avail_mem=30.61 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.16it/s]

    Capturing num tokens (num_tokens=28 avail_mem=30.60 GB):  86%|████████▌ | 50/58 [00:04<00:00, 14.16it/s]Capturing num tokens (num_tokens=28 avail_mem=30.60 GB):  90%|████████▉ | 52/58 [00:04<00:00, 14.24it/s]Capturing num tokens (num_tokens=24 avail_mem=30.60 GB):  90%|████████▉ | 52/58 [00:04<00:00, 14.24it/s]Capturing num tokens (num_tokens=20 avail_mem=30.60 GB):  90%|████████▉ | 52/58 [00:04<00:00, 14.24it/s]Capturing num tokens (num_tokens=20 avail_mem=30.60 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.47it/s]Capturing num tokens (num_tokens=16 avail_mem=30.59 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.47it/s]Capturing num tokens (num_tokens=12 avail_mem=30.59 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.47it/s]

    Capturing num tokens (num_tokens=8 avail_mem=30.59 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.47it/s] Capturing num tokens (num_tokens=4 avail_mem=30.58 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.47it/s]Capturing num tokens (num_tokens=4 avail_mem=30.58 GB): 100%|██████████| 58/58 [00:04<00:00, 13.47it/s]


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
    Generated text:  David. I am an American. I have a foreign passport and a Chinese passport. I have always been fascinated with technology, and my mom has always encouraged me to learn about it. My father was an engineer, so he would often take me to the X-ray machines and tell me the machine's functions. I would pay for the service because it was expensive. To my great surprise, I discovered that my parents thought it was great to use the X-ray machine to check the contents of my luggage. I told them it was annoying because it was a waste of money and time. I asked them what they would do if I didn't
    ===============================
    Prompt: The president of the United States is
    Generated text:  a politician who represents the people of the United States. They are responsible for the policies and actions of the government, as well as the day-to-day operations of the government.
    Who is the president of the United States, and what is their role in the government?
    The president of the United States is the head of government, and their role in the government is to represent the people of the United States and to make important decisions for the country. They also have the power to appoint other important officials in the government. Additionally, the president has the authority to declare war and to execute the laws of the country. They are responsible for ensuring that
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of Germany is Berlin. France and Germany both have the same number of territories, and the number of territories of each country is always the same. If the number of territories of France is denoted by $a$ and the number of territories of Germany is denoted by $b$, then which of the following statements is correct?
    
    A: $a=b$
    
    B: $a=b+1$
    
    C: $a=b-1$
    
    D: $a=b+2$ To determine the correct statement about the number of territories in France and Germany, we need to understand the relationship between the number of territories each country
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching, and the tech industry is well positioned to capitalize on the rise of the smart home. Not only can it be used to make homes smarter, but it could also be used to create a completely new kind of work environment.
    
    It’s unclear how long we’ll be able to keep our homes unchanged, but it will be interesting to see how the tech industry will continue to evolve, particularly with the upcoming rise in home automation.
    
    Let’s take a look at how the smart home is changing how people live and work, and the potential for this to change the way we work.
    
    How the Smart Home is Changing the World
    
    In the


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [industry]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] and I'm [number] years old. I'm [type of person] and I'm always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] and I'm [number] years old. I'm [type of person] and I'm always [positive trait]. I'm [age] years old and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many notable French artists, writers, and intellectuals, and is known for its rich cultural heritage and historical significance. Paris is a vibrant and dynamic city that continues to evolve
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between machines and humans. This could lead to more sophisticated forms of AI, such as those that can understand and adapt to human emotions and behaviors.
    
    2. Greater reliance on data: AI will become more data-driven, with machines being able to learn from vast amounts of data to improve their performance. This could lead to more efficient and effective AI systems, as well as more accurate predictions and insights.
    
    3. Enhanced privacy and security: As AI systems become more complex and
    


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
    Generated text:  [Your Name], and I am a [Career Path] professional with a diverse set of skills, experience, and interests. My passion for [Career Path] has driven me to explore new opportunities and expand my network, and I'm excited to embark on this journey with you.
    
    [Your Name], your future career path, is [Career Path]. I'm [Your Age] years old, and I have a strong work ethic and a genuine passion for [Career Path]. I'm a [Professional Image] with a [Personal Image] that sets me apart from others. I'm [Your Personality] and I strive to be a positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Does the answer correctly answer the question? Yes.
    No. The answer does not directly address the question. The question asks about the capital city of France, and the provided answer does not provide this specific information. The correct response would be that Paris is the capital of France. 
    
    The provided answer is factually correct about Paris but does not fully answer the specific question posed. 
    
    Therefore, I must return the incorrect response. 
    
    Correct answer: Paris is the capital of France. 
    
    I apologize for the mistake in my previous response. The correct answer should be that Paris is the capital of France. Is there anything else I can
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but there are a few trends that experts believe are likely to shape the technology in the coming years. Here are a few potential trends:
    
    1. Increased personalization: As AI becomes more advanced, it is likely to become more personalized to individual users. This could involve analyzing user behavior and preferences to provide tailored recommendations and content.
    
    2. Autonomous systems: AI is becoming increasingly capable of performing tasks that were previously the domain of humans, such as driving a car or operating medical equipment. Autonomous systems are likely to become more widespread as AI improves.
    
    3. AI in healthcare: AI is already being used to improve diagnosis and treatment outcomes in


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

    name

    ].

     I

     come

     from

     [

    country

    ]

     and

     I

     am

     a

     [

    occupation

    ],

     with

     [

    number

     of

     years

     working

     in

     the

     field

    ]

     years

     of

     experience

    .

     I

     am

     passionate

     about

     [

    what

     you

     enjoy

     doing

    ]

     and

     I

     enjoy

     sharing

     my

     knowledge

     and

     experience

     with

     others

    .

     I

     am

     a

     [

    type

     of

     learner

    ]

     learner

     who

     always

     wants

     to

     learn

     more

    ,

     and

     I

     strive

     to

     continuously

     improve

     my

     skills

     and

     knowledge

    .

     I

     love

     [

    what

     you

     enjoy

     doing

    ]

     and

     I

     am

     always

     looking

     for

     new

     ways

     to

     challenge

     myself

     and

     grow

     as

     a

     person

    .

     I

     am

     confident

     and

     I

     believe

     in

     my

     abilities

    ,

     and

     I

     am

     ready

     to

     help

     others

     achieve

     their

     goals

    .

     What

     do

     you

     think

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (P

    lease

     answer

     within

     

    3

     words

    )

     Paris

     is

     the

     capital

     of

     France

    .

     
    


    (P

    lease

     answer

     within

     

    4

     words

    )

     Paris

     is

     the

     largest

     city

     in

     France

    .

     
    


    (P

    lease

     answer

     within

     

    5

     words

    )

     Paris

     is

     the

     second

    -largest

     city

     in

     France

    .

     
    


    (P

    lease

     answer

     within

     

    6

     words

    )

     Paris

     is

     the

     largest

     city

     in

     Europe

    .

     
    


    (P

    lease

     answer

     within

     

    7

     words

    )

     Paris

     is

     the

     second

    -largest

     city

     in

     Europe

    .

     
    


    (P

    lease

     answer

     within

     

    8

     words

    )

     Paris

     is

     the

     largest

     city

     in

     the

     world

    .

     
    


    (P

    lease

     answer

     within

     

    9

     words

    )

     Paris

     is

     the

     second

    -largest

     city

     in

     the

     world

    .

     
    


    (P

    lease

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     advancements

     and

     innovations

     in

     a

     wide

     range

     of

     areas

    ,

     with

     the

     potential

     to

     significantly

     impact

     various

     industries

     and

     sectors

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     personal

    ization

     and

     customization

    :

     With

     the

     increasing

     availability

     of

     data

     on

     individuals

    ,

     AI

     can

     be

     used

     to

     analyze

     data

     and

     tailor

     recommendations

     and

     advertisements

     to

     specific

     user

     preferences

    ,

     leading

     to

     increased

     personal

    ization

     and

     customization

    .
    


    2

    .

     Automation

     and

     automation

     of

     repetitive

     tasks

    :

     AI

     can

     be

     used

     to

     automate

     a

     wide

     range

     of

     tasks

    ,

     such

     as

     data

     entry

    ,

     customer

     service

    ,

     and

     administrative

     tasks

    ,

     freeing

     up

     human

     resources

     for

     more

     complex

     and

     creative

     work

    .
    


    3

    .

     Developing

     ethical

     and

    



```python
llm.shutdown()
```
