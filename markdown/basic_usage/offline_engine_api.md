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

    [2026-03-13 01:48:48] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-13 01:48:48] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-13 01:48:48] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-13 01:48:51] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.


    [2026-03-13 01:48:51] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-13 01:48:51] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=407941261, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]Compiling num tokens (num_tokens=6144):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:20,  2.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:20,  2.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:20,  2.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:20,  2.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:20,  2.64it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:08,  5.58it/s]

    Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:03<00:04,  9.99it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 17.07it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 31.26it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 38.02it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 44.12it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 49.71it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 49.71it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 49.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.43 GB):   3%|▎         | 2/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.44 GB):   3%|▎         | 2/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.43 GB):   3%|▎         | 2/58 [00:00<00:03, 14.21it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=70.43 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.43 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.42 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.42 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.42 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.41 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.80it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=70.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.40 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.39 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.01it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=70.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.34 GB):  31%|███       | 18/58 [00:00<00:01, 28.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.34 GB):  31%|███       | 18/58 [00:00<00:01, 28.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.33 GB):  31%|███       | 18/58 [00:00<00:01, 28.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.31 GB):  31%|███       | 18/58 [00:00<00:01, 28.22it/s]Capturing num tokens (num_tokens=960 avail_mem=70.32 GB):  31%|███       | 18/58 [00:00<00:01, 28.22it/s] Capturing num tokens (num_tokens=896 avail_mem=70.31 GB):  31%|███       | 18/58 [00:00<00:01, 28.22it/s]Capturing num tokens (num_tokens=896 avail_mem=70.31 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=832 avail_mem=70.31 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=768 avail_mem=70.30 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.46it/s]

    Capturing num tokens (num_tokens=704 avail_mem=70.30 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=640 avail_mem=70.29 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=576 avail_mem=70.29 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=576 avail_mem=70.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.37it/s]Capturing num tokens (num_tokens=512 avail_mem=70.27 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.37it/s]Capturing num tokens (num_tokens=480 avail_mem=70.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=448 avail_mem=70.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=416 avail_mem=70.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=384 avail_mem=70.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=384 avail_mem=70.28 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=352 avail_mem=70.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=320 avail_mem=70.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]

    Capturing num tokens (num_tokens=288 avail_mem=70.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=256 avail_mem=70.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=240 avail_mem=70.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=224 avail_mem=70.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=224 avail_mem=70.26 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=208 avail_mem=70.25 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=192 avail_mem=70.25 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=176 avail_mem=70.25 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=160 avail_mem=70.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=144 avail_mem=70.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=128 avail_mem=70.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=128 avail_mem=70.24 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=112 avail_mem=70.24 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.24it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.24it/s] Capturing num tokens (num_tokens=80 avail_mem=70.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=64 avail_mem=70.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=48 avail_mem=70.22 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=48 avail_mem=70.22 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=32 avail_mem=70.22 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=24 avail_mem=69.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=20 avail_mem=69.93 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=16 avail_mem=69.93 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=16 avail_mem=69.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=12 avail_mem=69.47 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.71it/s]

    Capturing num tokens (num_tokens=8 avail_mem=69.22 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.71it/s] Capturing num tokens (num_tokens=4 avail_mem=69.22 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=4 avail_mem=69.22 GB): 100%|██████████| 58/58 [00:01<00:00, 36.36it/s]


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
    Generated text:  Liam. I'm an extremely adventurous boy from a small town. I have two cats named Triple and Submarine. I'm a keen outdoorsman, but I also enjoy art and music. I have a passion for building and programming. What's your favorite hobby? My favorite hobby is programming. I enjoy creating software applications that help people to improve their lives. What do you do on weekends? On weekends, I like to go for long walks, read books, and explore the park. What's your favorite book? My favorite book is "1984" by George Orwell. It's a dystopian novel that deals with themes
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking officer in the armed forces who is chosen by a democratically-elected head of state to hold a specific office of trust, which usually includes the office of the presidency. The president of the United States was first elected by the people in 1787, and every four years, the people vote to replace him or her.
    The role of the president is to serve as the head of state and government of the United States. The president is responsible for running the government and is responsible for making important decisions that affect the nation. The president is also responsible for representing the nation in the international community, and they are considered to
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Paris
    D. Tokyo
    Answer:
    
    A
    
    Which of the following is a correct basis for determining the depth of the crack? ____ 
    A. The length of the crack
    B. The width of the crack
    C. The diameter of the crack
    D. The distance between the cracks
    Answer:
    
    C
    
    To ensure that all levels of personnel have clear understanding of the overall situation and specific responsibilities of their respective positions, it is necessary to regularly convene a safety production meeting at least ____.
    A. Every quarter
    B. Every month
    C. Every half year
    ===============================
    Prompt: The future of AI is
    Generated text:  multi-faceted, and today’s development will form the backbone of tomorrow’s digital world. AI is a field of science, technology, engineering, and mathematics (STEM) that develops artificial intelligence and intelligent machines to replicate human intelligence and learn from data. AI is also a complex and vast field, with vast opportunities for growth and development. As we move into the future, it is important to understand how AI will shape our world.
    One of the key areas of AI development is in the field of machine learning. Machine learning is a type of artificial intelligence that allows machines to learn from data and make predictions or decisions based on that data.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. And what's your favorite hobby or activity? I love [insert a short, positive description of your hobby or activity]. And what's your favorite book or movie? I love [insert a short, positive description of your favorite book or movie]. And what's your favorite place to go? I love [insert a short, positive description of your favorite place]. And what's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major center for art, culture, and politics in Europe. Paris is a popular tourist destination and a cultural hub for France and the world. The city is home to many museums, theaters, and other cultural institutions, and is a major economic and financial center. Paris is a city of contrasts, with its rich history and modernity, and is a UNESCO World
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in various industries.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    3.
    


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
    Generated text:  ______ and I'm an AI language model. I'm here to assist you with any questions or concerns you may have, and I'm always here to help you improve your language skills. What can I do for you today? Let me know if you need help with anything. You can also ask me any questions or feedback you have, and I'll do my best to help. I'm here to make your experience with AI language models as smooth and enjoyable as possible. Let me know how I can assist you today. [Your name] [Your profession] [Your role] [Your responsibilities] [Your qualifications] [Your experience]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville Flottante" in French, which means "floating city". It is a UNESCO World Heritage site and the second-largest city in the European Union. Paris is famous for its architecture, such as the Eiffel Tower, and its culture, including its annual Carnaval festival. The city is also known for its love of art, food, and fashion. Paris is the heart of France and a popular tourist destination. According to the latest census, the population of Paris is 2. 2 million people. 
    
    The history of Paris dates back to the 6th century, and it is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be more complex and multifaceted than its current status. Here are some possible future trends in artificial intelligence:
    
    1. Increased awareness of ethical and responsible AI: As more people become aware of the negative consequences of AI, there will be a push towards developing AI that is more ethical and responsible. This may involve developing AI that is designed to minimize harm to individuals or society as a whole.
    
    2. Integration of AI into healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and streamline processes. As AI technology becomes more advanced and accessible, it is likely to be integrated into a wider range of healthcare


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

    ]

     and

     I

    'm

     a

     [

    Your

     Profession

    /

    Role

    ].

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Area

     of

     Expert

    ise

    /

    Interest

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     grow

     and

     learn

     from

     my

     mistakes

    ,

     and

     I

    'm

     always

     eager

     to

     hear

     from

     you

     about

     your

     own

     journey

     to

     becoming

     a

     more

     successful

     [

    Your

     Profession

    /

    Role

    ].

     What

    's

     your

     story

    ,

     and

     how

     can

     I

     help

     you

     in

     your

     journey

    ?

     Let

    's

     connect

    !

     [

    Your

     Name

    ]

     [

    Your

     Profession

    /

    Role

    ]

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ]

     and

     I

    'm

     a

     [

    Your

     Profession

    /

    Role

    ].

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Area

     of

     Expert

    ise

    /

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Lis

    ,"

     which

     is

     a

     French

     word

     for

     "

    the

     Lis

    ."

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     seat

     of

     the

     French

     government

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     with

     many

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     bustling

     street

     life

    ,

     delicious

     food

    ,

     and

     diverse

     cultural

     offerings

    .

     Paris

     is

     a

     popular

     tourist

     destination

     for

     people

     from

     all

     over

     the

     world

    ,

     and

     it

     is

     a

     must

    -

    visit

     for

     anyone

     who

     loves

     to

     explore

     new

     places

     and

     discover

     new

     cultures

    .

     Despite

     its

     size

     and

     complexity

    ,

     Paris

     remains

     a

     vibrant

     and

     welcoming

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     the

     following

     trends

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     The

     AI

     field

     is

     constantly

     advancing

    ,

     and

     as

     a

     result

    ,

     the

     accuracy

     of

     AI

     systems

     is

     expected

     to

     increase

    .

     This

     is

     expected

     to

     be

     particularly

     important

     for

     applications

     that

     require

     high

     levels

     of

     precision

    ,

     such

     as

     medical

     diagnosis

     and

     financial

     analysis

    .
    


    2

    .

     Enhanced

     collaboration

    :

     AI

     is

     becoming

     more

     capable

     of

     working

     collabor

    atively

     with

     humans

    ,

     especially

     in

     areas

     such

     as

     healthcare

     and

     natural

     language

     processing

    .

     This

     means

     that

     AI

     systems

     will

     be

     able

     to

     perform

     tasks

     more

     efficiently

     and

     effectively

    ,

     and

     this

     will

     require

     more

     collaboration

     between

     humans

     and

     machines

    .
    


    3

    .

     AI

     will

     play

     a

     more

     significant

     role

     in

     decision

    



```python
llm.shutdown()
```
