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

    [2026-03-08 06:34:09] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-08 06:34:09] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-08 06:34:09] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-08 06:34:11] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-08 06:34:11] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 06:34:11] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=405967246, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.67it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.67it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=117.96 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=117.96 GB):   5%|▌         | 1/20 [00:00<00:03,  5.53it/s]Capturing batches (bs=120 avail_mem=116.66 GB):   5%|▌         | 1/20 [00:00<00:03,  5.53it/s]

    Capturing batches (bs=112 avail_mem=116.66 GB):   5%|▌         | 1/20 [00:00<00:03,  5.53it/s]Capturing batches (bs=104 avail_mem=118.10 GB):   5%|▌         | 1/20 [00:00<00:03,  5.53it/s]Capturing batches (bs=104 avail_mem=118.10 GB):  20%|██        | 4/20 [00:00<00:01, 14.68it/s]Capturing batches (bs=96 avail_mem=117.81 GB):  20%|██        | 4/20 [00:00<00:01, 14.68it/s] Capturing batches (bs=88 avail_mem=117.80 GB):  20%|██        | 4/20 [00:00<00:01, 14.68it/s]Capturing batches (bs=80 avail_mem=117.08 GB):  20%|██        | 4/20 [00:00<00:01, 14.68it/s]Capturing batches (bs=80 avail_mem=117.08 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.04it/s]Capturing batches (bs=72 avail_mem=116.81 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.04it/s]

    Capturing batches (bs=64 avail_mem=116.80 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.04it/s]Capturing batches (bs=56 avail_mem=116.80 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.04it/s]Capturing batches (bs=56 avail_mem=116.80 GB):  50%|█████     | 10/20 [00:00<00:00, 20.56it/s]Capturing batches (bs=48 avail_mem=116.79 GB):  50%|█████     | 10/20 [00:00<00:00, 20.56it/s]Capturing batches (bs=40 avail_mem=116.79 GB):  50%|█████     | 10/20 [00:00<00:00, 20.56it/s]Capturing batches (bs=32 avail_mem=117.77 GB):  50%|█████     | 10/20 [00:00<00:00, 20.56it/s]

    Capturing batches (bs=32 avail_mem=117.77 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.66it/s]Capturing batches (bs=24 avail_mem=117.76 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.66it/s]Capturing batches (bs=16 avail_mem=116.84 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.66it/s]Capturing batches (bs=12 avail_mem=116.83 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.66it/s]Capturing batches (bs=12 avail_mem=116.83 GB):  80%|████████  | 16/20 [00:00<00:00, 20.35it/s]Capturing batches (bs=8 avail_mem=116.83 GB):  80%|████████  | 16/20 [00:00<00:00, 20.35it/s] 

    Capturing batches (bs=4 avail_mem=116.82 GB):  80%|████████  | 16/20 [00:00<00:00, 20.35it/s]Capturing batches (bs=2 avail_mem=117.74 GB):  80%|████████  | 16/20 [00:00<00:00, 20.35it/s]Capturing batches (bs=2 avail_mem=117.74 GB):  95%|█████████▌| 19/20 [00:00<00:00, 22.23it/s]Capturing batches (bs=1 avail_mem=117.73 GB):  95%|█████████▌| 19/20 [00:00<00:00, 22.23it/s]Capturing batches (bs=1 avail_mem=117.73 GB): 100%|██████████| 20/20 [00:00<00:00, 20.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:43,  1.28it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:43,  1.28it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:43,  1.28it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:03<00:43,  1.28it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.03it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.03it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.03it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.03it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.03it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:07,  6.03it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:03<00:04, 10.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s]

    Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 16.83it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 23.43it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]

    Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 35.78it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:03<00:00, 43.39it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 58.71it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 58.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.71 GB):   3%|▎         | 2/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=116.70 GB):   7%|▋         | 4/58 [00:00<00:03, 17.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.68 GB):   7%|▋         | 4/58 [00:00<00:03, 17.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.65 GB):   7%|▋         | 4/58 [00:00<00:03, 17.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=116.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.61 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.95it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=116.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.58 GB):  31%|███       | 18/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.57 GB):  31%|███       | 18/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.56 GB):  31%|███       | 18/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.53 GB):  31%|███       | 18/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=960 avail_mem=116.53 GB):  31%|███       | 18/58 [00:00<00:01, 29.03it/s] Capturing num tokens (num_tokens=960 avail_mem=116.53 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=896 avail_mem=116.53 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=832 avail_mem=116.52 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.66it/s]

    Capturing num tokens (num_tokens=768 avail_mem=116.51 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=704 avail_mem=116.50 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=704 avail_mem=116.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.05it/s]Capturing num tokens (num_tokens=640 avail_mem=116.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.05it/s]Capturing num tokens (num_tokens=576 avail_mem=116.49 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.05it/s]Capturing num tokens (num_tokens=512 avail_mem=116.48 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.05it/s]Capturing num tokens (num_tokens=480 avail_mem=116.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.05it/s]Capturing num tokens (num_tokens=448 avail_mem=116.49 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=448 avail_mem=116.49 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.51it/s]Capturing num tokens (num_tokens=416 avail_mem=116.49 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.51it/s]Capturing num tokens (num_tokens=384 avail_mem=116.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.51it/s]

    Capturing num tokens (num_tokens=352 avail_mem=116.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.51it/s]Capturing num tokens (num_tokens=320 avail_mem=116.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.51it/s]Capturing num tokens (num_tokens=288 avail_mem=116.47 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.51it/s]Capturing num tokens (num_tokens=288 avail_mem=116.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=256 avail_mem=116.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=240 avail_mem=116.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=224 avail_mem=116.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=208 avail_mem=116.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=192 avail_mem=116.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=192 avail_mem=116.45 GB):  71%|███████   | 41/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=176 avail_mem=116.45 GB):  71%|███████   | 41/58 [00:01<00:00, 39.17it/s]

    Capturing num tokens (num_tokens=160 avail_mem=116.44 GB):  71%|███████   | 41/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=144 avail_mem=116.43 GB):  71%|███████   | 41/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=128 avail_mem=116.43 GB):  71%|███████   | 41/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=112 avail_mem=116.43 GB):  71%|███████   | 41/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=112 avail_mem=116.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=96 avail_mem=116.42 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.73it/s] Capturing num tokens (num_tokens=80 avail_mem=116.42 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=64 avail_mem=116.41 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=48 avail_mem=116.41 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=32 avail_mem=116.40 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.73it/s]

    Capturing num tokens (num_tokens=32 avail_mem=116.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=28 avail_mem=116.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=24 avail_mem=116.39 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=20 avail_mem=116.39 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=16 avail_mem=116.38 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=12 avail_mem=116.38 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=12 avail_mem=116.38 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=8 avail_mem=116.37 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.48it/s] Capturing num tokens (num_tokens=4 avail_mem=116.36 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=4 avail_mem=116.36 GB): 100%|██████████| 58/58 [00:01<00:00, 34.12it/s]


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
    Generated text:  Samantha, and I am a neuroscientist with experience in learning and memory research. Currently, I am a PhD student at the University of Kentucky studying how individuals’ brains respond to the effects of depression and stress, and how these responses impact memory performance and cognitive functioning. I am interested in understanding how the brain responds to stressful or depressive experiences, and how this response can lead to memory difficulties and impaired cognitive performance. Additionally, I am working on developing a new treatment for depression and anxiety, and using neuroimaging to explore the effects of such treatments on brain function.
    How do you come up with your research questions?
    As a neuroscient
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the 2020 presidential election as a referendum on the existence of the Chinese government, to decide to increase funding for the Chinese consulate in the United States, or to increase funding for the National Economic Research Station in the United States. 
    
    Is it a valid strategy for the president to increase funding for the Chinese consulate or the National Economic Research Station in the United States?
    
    To determine whether it is a valid strategy for the president to increase funding for the Chinese consulate or the National Economic Research Station in the United States, we need to consider the historical context, the current situation, and the possible outcomes of different
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ].
    A. Paris
    B. London
    C. Rome
    D. Rome
    
    To determine the capital of France, we need to review the official list of countries and their capital cities. The official list is as follows:
    
    1. United Kingdom: London
    2. France: Paris
    3. Germany: Berlin
    4. Italy: Rome
    5. Spain: Madrid
    6. United States: Washington, D.C.
    7. Japan: Tokyo
    8. Switzerland: Zurich
    
    Based on this list, the capital of France is Paris.
    
    Therefore, the correct answer is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising. From small autonomous vehicles to super calculators, to fully automated cities. Here are some of the key trends that are expected to shape the future of AI:
      1. Augmented intelligence: Augmented intelligence is a type of AI that adds new capabilities to the human brain. In the future, this type of AI could be used to help people with memory loss, Alzheimer’s disease, and other chronic health conditions. Augmented AI could also be used to help people with speech and language problems.
      2. Brain-computer interface: Brain-computer interface (BCI) is a technology that allows individuals to control


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
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. The city is home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is also known for its vibrant nightlife and its role as a major transportation hub, with the iconic Eiffel Tower serving as a symbol of the city's importance. Overall, Paris is a city of art, culture, and history, and is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more robust AI systems that are designed to be fair, transparent, and accountable.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [Name], and I am a [Title] at [Company]. I enjoy [What I Do] and am always looking to learn more about [Why You Should Care]. I am passionate about [What Makes Me Special] and have a knack for [What I Do Best]. I am confident and always ready to help anyone who needs my expertise. How would you like to meet you, [Name]? You can expect me to be [Type of Person], [What I Do] and [How I Can Help]. I am excited to chat and learn more about you. [Name] look forward to meeting you. Good luck with your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    What is the capital of France? Paris. 
    
    This was the official capital of France from 1804 until 1963, when it was re-named Nice. It is the largest city in France and is known for its fashion industry and its iconic Eiffel Tower. Paris is also home to the Louvre Museum and the Arc de Triomphe, and is a popular tourist destination. It is home to the Eiffel Tower and the Champs-Élysées. It is also the location of the annual Eiffel Tower contest. France's capital is Paris. 
    
    Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements and significant changes in how it is used and applied. Some possible future trends in AI include:
    
    1. Increased integration with physical devices: As we continue to improve the computing power and processing speed of hardware, we will see a greater integration of AI technology into physical devices like smartphones, smart home devices, and industrial automation systems.
    
    2. AI-driven healthcare advancements: AI will be used in healthcare to improve patient outcomes, automate diagnostic procedures, and personalize treatment plans. For example, AI-powered medical imaging systems could detect diseases earlier and more accurately than humans.
    
    3. Autonomous vehicles: Autonomous vehicles are a potential future


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

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

     love

     [

    Favorite

     Hobby

     or

     Activity

    ].

     I

    'm

     currently

     [

    Current

     Location

    ]

     and

     my

     [

    Favorite

     Food

    ]

     is

     [

    Favorite

     Food

    ].

     I

     also

     enjoy

     [

    Favorite

     Music

    /

    Video

     Game

    /

    Activity

    ].

     I

     have

     [

    Number

     of

     Pets

    ]

     pets

     and

     [

    Number

     of

     Children

    /

    Grand

    children

    ]

     kids

    .

     [

    Name

    ]

     is

     a

     reliable

     and

     hard

    working

     person

     who

     always

     has

     a

     positive

     attitude

    .

     I

    'm

     always

     ready

     to

     help

     others

     and

     always

     strive

     to

     improve

     myself

    .

     I

     believe

     in

     [

    Rel

    igion

    /

    Value

    ]

     and

     I

     try

     to

     follow

     it

     strictly

    .

     How

     would

     you

     describe

     [

    Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     cultural

     attractions

    .

     It

     is

     also

     a

     historical

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     

    1

    3

    th

     century

    .

     Paris

     is

     a

     city

     with

     a

     unique

     blend

     of

     modern

     and

     traditional

     elements

     and

     is

     a

     major

     hub

     of

     culture

    ,

     commerce

    ,

     and

     diplomacy

    .

     It

     is

     known

     for

     its

     famous

     landmarks

     such

     as

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

     Dame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     food

     scene

    ,

     which

     includes

     several

     renowned

     restaurants

     and

     popular

     street

     food

    .

     The

     city

     has

     a

     large

     population

     and

     is

     home

     to

     many

     international

     companies

     and

     organizations

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     the

     possibilities

     are

     vast

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     development

     of

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     an

     increased

     reliance

     on

     automation

    .

     This

     means

     that

     AI

     systems

     will

     become

     more

     efficient

    ,

     precise

    ,

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     done

     by

     humans

    .

     This

     will

     likely

     lead

     to

     increased

     productivity

     and

     economic

     growth

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     will

     likely

     play

     a

     key

     role

     in

     medical

     diagnostics

     and

     treatment

     planning

    .

     AI

     algorithms

     will

     be

     able

     to

     analyze

     large

     amounts

     of

     medical

     data

    ,

     identify

     patterns

     and

     trends

    ,

     and

     provide

     insights

     that

     can

    



```python
llm.shutdown()
```
