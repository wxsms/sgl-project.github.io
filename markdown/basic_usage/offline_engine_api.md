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

    [2026-03-06 22:07:04] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-06 22:07:04] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-06 22:07:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-06 22:07:07] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.


    [2026-03-06 22:07:07] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-06 22:07:07] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=1003945875, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.56it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.56it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=57.95 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=57.95 GB):   5%|▌         | 1/20 [00:00<00:03,  5.67it/s]Capturing batches (bs=120 avail_mem=57.85 GB):   5%|▌         | 1/20 [00:00<00:03,  5.67it/s]

    Capturing batches (bs=112 avail_mem=57.85 GB):   5%|▌         | 1/20 [00:00<00:03,  5.67it/s]Capturing batches (bs=104 avail_mem=57.84 GB):   5%|▌         | 1/20 [00:00<00:03,  5.67it/s]Capturing batches (bs=104 avail_mem=57.84 GB):  20%|██        | 4/20 [00:00<00:01, 13.47it/s]Capturing batches (bs=96 avail_mem=57.84 GB):  20%|██        | 4/20 [00:00<00:01, 13.47it/s] Capturing batches (bs=88 avail_mem=57.84 GB):  20%|██        | 4/20 [00:00<00:01, 13.47it/s]Capturing batches (bs=80 avail_mem=57.84 GB):  20%|██        | 4/20 [00:00<00:01, 13.47it/s]

    Capturing batches (bs=80 avail_mem=57.84 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.75it/s]Capturing batches (bs=72 avail_mem=57.84 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.75it/s]Capturing batches (bs=64 avail_mem=57.84 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.75it/s]Capturing batches (bs=56 avail_mem=57.84 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.75it/s]Capturing batches (bs=56 avail_mem=57.84 GB):  50%|█████     | 10/20 [00:00<00:00, 20.78it/s]Capturing batches (bs=48 avail_mem=56.76 GB):  50%|█████     | 10/20 [00:00<00:00, 20.78it/s]Capturing batches (bs=40 avail_mem=56.72 GB):  50%|█████     | 10/20 [00:00<00:00, 20.78it/s]Capturing batches (bs=32 avail_mem=43.50 GB):  50%|█████     | 10/20 [00:00<00:00, 20.78it/s]

    Capturing batches (bs=32 avail_mem=43.50 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.58it/s]Capturing batches (bs=24 avail_mem=43.50 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.58it/s]Capturing batches (bs=16 avail_mem=43.50 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.58it/s]Capturing batches (bs=12 avail_mem=43.50 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.58it/s]Capturing batches (bs=12 avail_mem=43.50 GB):  80%|████████  | 16/20 [00:00<00:00, 21.92it/s]Capturing batches (bs=8 avail_mem=43.50 GB):  80%|████████  | 16/20 [00:00<00:00, 21.92it/s] Capturing batches (bs=4 avail_mem=43.49 GB):  80%|████████  | 16/20 [00:00<00:00, 21.92it/s]

    Capturing batches (bs=2 avail_mem=43.49 GB):  80%|████████  | 16/20 [00:00<00:00, 21.92it/s]Capturing batches (bs=1 avail_mem=43.49 GB):  80%|████████  | 16/20 [00:00<00:00, 21.92it/s]Capturing batches (bs=1 avail_mem=43.49 GB): 100%|██████████| 20/20 [00:00<00:00, 26.35it/s]Capturing batches (bs=1 avail_mem=43.49 GB): 100%|██████████| 20/20 [00:00<00:00, 21.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:10,  2.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:10,  2.29s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:56,  1.01s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:56,  1.01s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:56,  1.01s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.38it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.38it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.38it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:12,  4.04it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:12,  4.04it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:12,  4.04it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:12,  4.04it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:04, 10.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:04, 10.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:04, 10.09it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:04, 10.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:04, 10.09it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:02, 14.45it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:02, 14.45it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:02, 14.45it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:02, 14.45it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:02, 14.45it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.72it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.72it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.72it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.72it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.72it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 18.72it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 23.99it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 23.99it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 23.99it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 23.99it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 23.99it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 27.18it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 27.18it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 27.18it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 27.18it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 27.18it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 27.18it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 30.53it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 31.01it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 31.01it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 31.01it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 31.01it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 31.01it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 31.01it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s] 

    Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 34.67it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.69it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 45.15it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 45.15it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 45.15it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 45.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.81 GB):   2%|▏         | 1/58 [00:00<00:08,  6.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.77 GB):   2%|▏         | 1/58 [00:00<00:08,  6.41it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=41.77 GB):   3%|▎         | 2/58 [00:00<00:08,  6.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.77 GB):   3%|▎         | 2/58 [00:00<00:08,  6.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.77 GB):   5%|▌         | 3/58 [00:00<00:08,  6.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.77 GB):   5%|▌         | 3/58 [00:00<00:08,  6.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.77 GB):   7%|▋         | 4/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.77 GB):   7%|▋         | 4/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.77 GB):   9%|▊         | 5/58 [00:00<00:06,  7.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.77 GB):   9%|▊         | 5/58 [00:00<00:06,  7.60it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.77 GB):  10%|█         | 6/58 [00:00<00:07,  7.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.77 GB):  10%|█         | 6/58 [00:00<00:07,  7.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.77 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.77 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.74it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.77 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.76 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.76 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.76 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.75 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.05it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.75 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.75 GB):  21%|██        | 12/58 [00:01<00:04,  9.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.75 GB):  21%|██        | 12/58 [00:01<00:04,  9.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.75 GB):  21%|██        | 12/58 [00:01<00:04,  9.67it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.75 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.74 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.74 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.74 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.73 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.61it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.73 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.73 GB):  31%|███       | 18/58 [00:01<00:03, 11.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.73 GB):  31%|███       | 18/58 [00:01<00:03, 11.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.73 GB):  31%|███       | 18/58 [00:02<00:03, 11.01it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=41.73 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.71 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.39it/s]Capturing num tokens (num_tokens=960 avail_mem=41.72 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.39it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=41.72 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.71it/s]Capturing num tokens (num_tokens=896 avail_mem=40.56 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.71it/s]Capturing num tokens (num_tokens=832 avail_mem=40.56 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.71it/s]

    Capturing num tokens (num_tokens=832 avail_mem=40.56 GB):  41%|████▏     | 24/58 [00:02<00:03,  8.93it/s]Capturing num tokens (num_tokens=768 avail_mem=40.55 GB):  41%|████▏     | 24/58 [00:02<00:03,  8.93it/s]Capturing num tokens (num_tokens=768 avail_mem=40.55 GB):  43%|████▎     | 25/58 [00:02<00:03,  8.34it/s]Capturing num tokens (num_tokens=704 avail_mem=40.55 GB):  43%|████▎     | 25/58 [00:02<00:03,  8.34it/s]

    Capturing num tokens (num_tokens=704 avail_mem=40.55 GB):  45%|████▍     | 26/58 [00:02<00:04,  7.92it/s]Capturing num tokens (num_tokens=640 avail_mem=40.55 GB):  45%|████▍     | 26/58 [00:02<00:04,  7.92it/s]Capturing num tokens (num_tokens=640 avail_mem=40.55 GB):  47%|████▋     | 27/58 [00:03<00:04,  7.54it/s]Capturing num tokens (num_tokens=576 avail_mem=40.55 GB):  47%|████▋     | 27/58 [00:03<00:04,  7.54it/s]

    Capturing num tokens (num_tokens=576 avail_mem=40.55 GB):  48%|████▊     | 28/58 [00:03<00:04,  7.44it/s]Capturing num tokens (num_tokens=512 avail_mem=40.53 GB):  48%|████▊     | 28/58 [00:03<00:04,  7.44it/s]Capturing num tokens (num_tokens=512 avail_mem=40.53 GB):  50%|█████     | 29/58 [00:03<00:04,  7.21it/s]Capturing num tokens (num_tokens=480 avail_mem=40.55 GB):  50%|█████     | 29/58 [00:03<00:04,  7.21it/s]

    Capturing num tokens (num_tokens=480 avail_mem=40.55 GB):  52%|█████▏    | 30/58 [00:03<00:04,  6.99it/s]Capturing num tokens (num_tokens=448 avail_mem=40.55 GB):  52%|█████▏    | 30/58 [00:03<00:04,  6.99it/s]Capturing num tokens (num_tokens=448 avail_mem=40.55 GB):  53%|█████▎    | 31/58 [00:03<00:04,  6.50it/s]Capturing num tokens (num_tokens=416 avail_mem=40.55 GB):  53%|█████▎    | 31/58 [00:03<00:04,  6.50it/s]

    Capturing num tokens (num_tokens=416 avail_mem=40.55 GB):  55%|█████▌    | 32/58 [00:03<00:04,  6.49it/s]Capturing num tokens (num_tokens=384 avail_mem=40.54 GB):  55%|█████▌    | 32/58 [00:03<00:04,  6.49it/s]Capturing num tokens (num_tokens=384 avail_mem=40.54 GB):  57%|█████▋    | 33/58 [00:04<00:03,  6.48it/s]Capturing num tokens (num_tokens=352 avail_mem=40.54 GB):  57%|█████▋    | 33/58 [00:04<00:03,  6.48it/s]

    Capturing num tokens (num_tokens=352 avail_mem=40.54 GB):  59%|█████▊    | 34/58 [00:04<00:03,  6.68it/s]Capturing num tokens (num_tokens=320 avail_mem=40.54 GB):  59%|█████▊    | 34/58 [00:04<00:03,  6.68it/s]Capturing num tokens (num_tokens=320 avail_mem=40.54 GB):  60%|██████    | 35/58 [00:04<00:03,  6.61it/s]Capturing num tokens (num_tokens=288 avail_mem=40.53 GB):  60%|██████    | 35/58 [00:04<00:03,  6.61it/s]

    Capturing num tokens (num_tokens=288 avail_mem=40.53 GB):  62%|██████▏   | 36/58 [00:04<00:03,  6.75it/s]Capturing num tokens (num_tokens=256 avail_mem=40.53 GB):  62%|██████▏   | 36/58 [00:04<00:03,  6.75it/s]Capturing num tokens (num_tokens=256 avail_mem=40.53 GB):  64%|██████▍   | 37/58 [00:04<00:03,  6.67it/s]Capturing num tokens (num_tokens=240 avail_mem=40.53 GB):  64%|██████▍   | 37/58 [00:04<00:03,  6.67it/s]

    Capturing num tokens (num_tokens=240 avail_mem=40.53 GB):  66%|██████▌   | 38/58 [00:04<00:02,  6.67it/s]Capturing num tokens (num_tokens=224 avail_mem=40.52 GB):  66%|██████▌   | 38/58 [00:04<00:02,  6.67it/s]Capturing num tokens (num_tokens=224 avail_mem=40.52 GB):  67%|██████▋   | 39/58 [00:04<00:02,  6.73it/s]Capturing num tokens (num_tokens=208 avail_mem=40.52 GB):  67%|██████▋   | 39/58 [00:04<00:02,  6.73it/s]

    Capturing num tokens (num_tokens=208 avail_mem=40.52 GB):  69%|██████▉   | 40/58 [00:05<00:02,  6.72it/s]Capturing num tokens (num_tokens=192 avail_mem=40.52 GB):  69%|██████▉   | 40/58 [00:05<00:02,  6.72it/s]Capturing num tokens (num_tokens=192 avail_mem=40.52 GB):  71%|███████   | 41/58 [00:05<00:02,  6.85it/s]Capturing num tokens (num_tokens=176 avail_mem=40.52 GB):  71%|███████   | 41/58 [00:05<00:02,  6.85it/s]

    Capturing num tokens (num_tokens=176 avail_mem=40.52 GB):  72%|███████▏  | 42/58 [00:05<00:02,  6.74it/s]Capturing num tokens (num_tokens=160 avail_mem=40.51 GB):  72%|███████▏  | 42/58 [00:05<00:02,  6.74it/s]Capturing num tokens (num_tokens=160 avail_mem=40.51 GB):  74%|███████▍  | 43/58 [00:05<00:02,  7.23it/s]Capturing num tokens (num_tokens=144 avail_mem=40.51 GB):  74%|███████▍  | 43/58 [00:05<00:02,  7.23it/s]

    Capturing num tokens (num_tokens=144 avail_mem=40.51 GB):  76%|███████▌  | 44/58 [00:05<00:01,  7.77it/s]Capturing num tokens (num_tokens=128 avail_mem=40.51 GB):  76%|███████▌  | 44/58 [00:05<00:01,  7.77it/s]Capturing num tokens (num_tokens=128 avail_mem=40.51 GB):  78%|███████▊  | 45/58 [00:05<00:01,  7.68it/s]Capturing num tokens (num_tokens=112 avail_mem=40.50 GB):  78%|███████▊  | 45/58 [00:05<00:01,  7.68it/s]

    Capturing num tokens (num_tokens=112 avail_mem=40.50 GB):  79%|███████▉  | 46/58 [00:05<00:01,  7.45it/s]Capturing num tokens (num_tokens=96 avail_mem=40.50 GB):  79%|███████▉  | 46/58 [00:05<00:01,  7.45it/s] Capturing num tokens (num_tokens=96 avail_mem=40.50 GB):  81%|████████  | 47/58 [00:06<00:01,  7.27it/s]Capturing num tokens (num_tokens=80 avail_mem=40.49 GB):  81%|████████  | 47/58 [00:06<00:01,  7.27it/s]

    Capturing num tokens (num_tokens=80 avail_mem=40.49 GB):  83%|████████▎ | 48/58 [00:06<00:01,  7.12it/s]Capturing num tokens (num_tokens=64 avail_mem=40.49 GB):  83%|████████▎ | 48/58 [00:06<00:01,  7.12it/s]Capturing num tokens (num_tokens=64 avail_mem=40.49 GB):  84%|████████▍ | 49/58 [00:06<00:01,  7.01it/s]Capturing num tokens (num_tokens=48 avail_mem=40.49 GB):  84%|████████▍ | 49/58 [00:06<00:01,  7.01it/s]

    Capturing num tokens (num_tokens=48 avail_mem=40.49 GB):  86%|████████▌ | 50/58 [00:06<00:01,  7.12it/s]Capturing num tokens (num_tokens=32 avail_mem=40.49 GB):  86%|████████▌ | 50/58 [00:06<00:01,  7.12it/s]Capturing num tokens (num_tokens=32 avail_mem=40.49 GB):  88%|████████▊ | 51/58 [00:06<00:00,  7.08it/s]Capturing num tokens (num_tokens=28 avail_mem=40.48 GB):  88%|████████▊ | 51/58 [00:06<00:00,  7.08it/s]

    Capturing num tokens (num_tokens=28 avail_mem=40.48 GB):  90%|████████▉ | 52/58 [00:06<00:00,  7.16it/s]Capturing num tokens (num_tokens=24 avail_mem=40.48 GB):  90%|████████▉ | 52/58 [00:06<00:00,  7.16it/s]Capturing num tokens (num_tokens=24 avail_mem=40.48 GB):  91%|█████████▏| 53/58 [00:06<00:00,  7.22it/s]Capturing num tokens (num_tokens=20 avail_mem=40.48 GB):  91%|█████████▏| 53/58 [00:06<00:00,  7.22it/s]

    Capturing num tokens (num_tokens=20 avail_mem=40.48 GB):  93%|█████████▎| 54/58 [00:07<00:00,  7.13it/s]Capturing num tokens (num_tokens=16 avail_mem=40.47 GB):  93%|█████████▎| 54/58 [00:07<00:00,  7.13it/s]Capturing num tokens (num_tokens=16 avail_mem=40.47 GB):  95%|█████████▍| 55/58 [00:07<00:00,  7.21it/s]Capturing num tokens (num_tokens=12 avail_mem=40.47 GB):  95%|█████████▍| 55/58 [00:07<00:00,  7.21it/s]

    Capturing num tokens (num_tokens=12 avail_mem=40.47 GB):  97%|█████████▋| 56/58 [00:07<00:00,  7.33it/s]Capturing num tokens (num_tokens=8 avail_mem=40.47 GB):  97%|█████████▋| 56/58 [00:07<00:00,  7.33it/s] Capturing num tokens (num_tokens=8 avail_mem=40.47 GB):  98%|█████████▊| 57/58 [00:07<00:00,  7.46it/s]Capturing num tokens (num_tokens=4 avail_mem=40.46 GB):  98%|█████████▊| 57/58 [00:07<00:00,  7.46it/s]

    Capturing num tokens (num_tokens=4 avail_mem=40.46 GB): 100%|██████████| 58/58 [00:07<00:00,  7.71it/s]Capturing num tokens (num_tokens=4 avail_mem=40.46 GB): 100%|██████████| 58/58 [00:07<00:00,  7.72it/s]


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
    Generated text:  Emily and I am a software developer. I’m currently working on a project that involves designing and implementing a system for tracking and analyzing customer behavior on a website. The project involves a lot of data analysis and statistical analysis, and I need a data analyst to help me with my work.
    
    I have experience with SQL and DAX, and I am also proficient in using Tableau. However, I am not confident in my ability to use Microsoft Office, specifically Excel. Can you please provide me with some tips on how to improve my Excel skills in order to make the most of my time working with data? Additionally, I would like some guidance
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking his 100th birthday. Assuming he is at least 35 years old, what is the minimum number of years between his birth and his birthday? To determine the minimum number of years between the president's birth and his 100th birthday, we need to consider the following information:
    
    1. The president is 35 years old.
    2. The president is seeking his 100th birthday.
    
    Since the president is seeking his 100th birthday, the years he has been alive are the sum of his age at the time of his birth and his age at his 100
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, a city situated on the Loire River, and located in which of the following countries? Paris is a historical city founded in 787 AD, and it was named after the city of the same name in ancient France. 
    
    Options:
    a) Germany
    b) Italy
    c) England
    d) France
    The correct answer is d) France. Paris is the capital of France, and it was founded in 787 AD. The other options are either not historical or not cities. England and Italy are located in Europe, not in France. Germany, being a country, is not the capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and AI is not the only thing to worry about. The power and potential of the Internet of Things (IoT) is also becoming ever more apparent. This future world is dependent on internet connectivity, and it requires the user to make a decision about which services to enable or disable. This technology is being used in many ways, but one of the most useful is using it to identify the best locations for place-based advertising. This can be done by analyzing the behavior of people and creating a list of the most popular places they visit.
    One of the most popular places that are visited is the city. Many people choose to live in


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I'm passionate about [Why I'm Passionate About My Field of Interest]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [What I'm Known For] who is always ready to help others. I'm a [What I'm Known For] who is always willing to lend a helping hand. I'm a [What I'm Known For] who is always ready to help others. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major center for art, music, and literature, and is home to many museums, theaters, and other cultural institutions. Paris is a popular tourist destination and a major economic center in France. The city is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater use of machine learning: Machine learning is likely to become more prevalent in AI, allowing machines to learn from data and improve their performance over time. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Increased focus on ethical AI: As AI becomes more integrated with human intelligence,
    


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
    Generated text:  [Your Name]. I am a [job title] at [company name], and I am excited to work with you. Let me know if there is anything specific you would like to know about me. I am currently working remotely from my home, and I have a [job title] who is [job title] at [company name]. I'm looking forward to working with you, and I'm excited to contribute my skills to your team. If you have any questions or concerns, please don't hesitate to reach out. Happy to chat! 
    Your response should be at least 40 words long. 
    
    Your response should be
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the River Seine in the region of Île-de-France. It is the largest city in Europe by population and is the second-largest city in the world by area. It has a rich history, including a long history of foreign rule, and is famous for its culture, cuisine, and iconic landmarks such as Notre-Dame Cathedral and the Arc de Triomphe. The city also has a diverse population, with over 3 million people in the metropolitan area. Paris is a popular tourist destination and is known for its museums, art galleries, and festivals. It is also home to the French Parliament, the Chamber
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and is constantly evolving. Some possible trends that are expected to shape the AI industry in the coming years include:
    
    1. Increased reliance on AI for decision-making and automation: As AI continues to become more sophisticated and capable, it is becoming more common for businesses and organizations to rely on AI for decision-making and automation. This trend is expected to continue as AI becomes more integrated into daily life, such as in healthcare, finance, and manufacturing.
    
    2. AI integration with human workers: As AI becomes more integrated into the workplace, it is expected that more human workers will be able to leverage AI to perform tasks that were previously done by humans


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

    Occup

    ation

    ]

     who

     has

     been

     working

     in

     [

    Field

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     am

     [

    Age

    ]

     years

     old

     and

     have

     a

     [

    Height

    /

    Weight

    ]

     of

     [

    Height

    /

    Weight

    ].

     I

     have

     always

     loved

     [

    Your

     Hobby

    /

    Interest

    /

    Activity

    ]

     and

     have

     been

     passionate

     about

     it

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     am

     always

     looking

     for

     new

     experiences

     and

     challenges

     and

     have

     always

     been

     driven

     to

     keep

     learning

     and

     growing

    .

     I

     am

     [

    Your

     Personality

    /O

    ld

     school

     teacher

    /etc

    .]

     and

     always

     strive

     to

     be

     the

     best

     version

     of

     myself

    .


    I

     would

     love

     to

     learn

     more

     about

     you

     and

     see

     what

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     a

     historic

     city

     located

     on

     the

     French

     Riv

    iera

    .


    Paris

     is

     a

     major

     European

     city

     with

     a

     rich

     history

    ,

     renowned

     for

     its

     iconic

     landmarks

    ,

     vibrant

     culture

    ,

     and

     cosm

    opolitan

     atmosphere

    .

     As

     the

     capital

     of

     France

    ,

     Paris

     is

     also

     a

     major

     tourist

     destination

    ,

     hosting

     annual

     events

    ,

     concerts

    ,

     and

     other

     cultural

     and

     artistic

     events

    .

     With

     its

     blend

     of

     historical

     and

     modern

     influences

    ,

     Paris

     continues

     to

     be

     a

     UNESCO

     World

     Heritage

     Site

     and

     a

     significant

     cultural

     and

     economic

     center

     in

     Europe

    .

     The

     city

     is

     also

     known

     for

     its

     French

     cuisine

     and

     wine

    ,

     and

     has

     been

     recognized

     as

     a

     UNESCO

     World

     Heritage

     site

     for

     its

     historical

     and

     cultural

     significance

    .

     Paris

     is

     an

     essential

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     very

     complex

     and

     multif

    ac

    eted

    ,

     and

     there

     are

     several

     possible

     trends

     that

     could

     shape

     its

     development

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     in

     AI

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     As

     AI

     continues

     to

     become

     more

     advanced

    ,

     we

     can

     expect

     to

     see

     significant

     improvements

     in

     its

     capabilities

    .

     For

     example

    ,

     we

     could

     see

     improvements

     in

     natural

     language

     processing

    ,

     image

     and

     speech

     recognition

    ,

     and

     decision

    -making

     capabilities

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

     likely

     to

     become

     even

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    ),

     blockchain

    ,

     and

     renewable

     energy

    .

     This

     integration

     could

     lead

     to

     new

     and

     innovative

     applications

     that

     we

     can

    't

    



```python
llm.shutdown()
```
