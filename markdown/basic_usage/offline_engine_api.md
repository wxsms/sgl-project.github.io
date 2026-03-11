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

    [2026-03-11 17:54:52] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-11 17:54:52] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-11 17:54:52] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-11 17:54:55] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.


    [2026-03-11 17:54:55] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-11 17:54:55] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=237314635, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.59it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.59it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.59it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.71it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 19.83it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 28.96it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 47.14it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 47.14it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 47.14it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 47.14it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 47.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=129.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=129.21 GB):   2%|▏         | 1/58 [00:00<00:06,  8.52it/s]Capturing num tokens (num_tokens=7680 avail_mem=129.18 GB):   2%|▏         | 1/58 [00:00<00:06,  8.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=129.17 GB):   2%|▏         | 1/58 [00:00<00:06,  8.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=129.17 GB):   5%|▌         | 3/58 [00:00<00:05, 10.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=129.17 GB):   5%|▌         | 3/58 [00:00<00:05, 10.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=129.16 GB):   5%|▌         | 3/58 [00:00<00:05, 10.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=129.16 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=129.16 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=129.16 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=129.16 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=129.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=129.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=129.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=129.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.75it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=129.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=129.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=129.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=129.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=129.12 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=129.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=129.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=129.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=129.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.65it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=129.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=129.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=129.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=129.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=960 avail_mem=129.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.16it/s] Capturing num tokens (num_tokens=896 avail_mem=129.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=896 avail_mem=129.08 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=832 avail_mem=129.07 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=768 avail_mem=129.07 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=704 avail_mem=129.07 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.68it/s]

    Capturing num tokens (num_tokens=640 avail_mem=129.06 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=576 avail_mem=129.06 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.68it/s]Capturing num tokens (num_tokens=576 avail_mem=129.06 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=512 avail_mem=129.05 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=480 avail_mem=129.06 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=448 avail_mem=129.06 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=416 avail_mem=129.05 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=384 avail_mem=129.05 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=384 avail_mem=129.05 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=352 avail_mem=129.04 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=320 avail_mem=129.04 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]

    Capturing num tokens (num_tokens=288 avail_mem=129.04 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=256 avail_mem=129.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=256 avail_mem=129.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=240 avail_mem=128.74 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=224 avail_mem=128.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=208 avail_mem=128.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.90it/s]

    Capturing num tokens (num_tokens=192 avail_mem=128.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.90it/s]Capturing num tokens (num_tokens=192 avail_mem=128.98 GB):  71%|███████   | 41/58 [00:01<00:00, 29.39it/s]Capturing num tokens (num_tokens=176 avail_mem=128.97 GB):  71%|███████   | 41/58 [00:01<00:00, 29.39it/s]Capturing num tokens (num_tokens=160 avail_mem=128.96 GB):  71%|███████   | 41/58 [00:01<00:00, 29.39it/s]Capturing num tokens (num_tokens=144 avail_mem=128.78 GB):  71%|███████   | 41/58 [00:01<00:00, 29.39it/s]Capturing num tokens (num_tokens=128 avail_mem=128.79 GB):  71%|███████   | 41/58 [00:01<00:00, 29.39it/s]Capturing num tokens (num_tokens=128 avail_mem=128.79 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.20it/s]Capturing num tokens (num_tokens=112 avail_mem=128.78 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.20it/s]

    Capturing num tokens (num_tokens=96 avail_mem=128.80 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.20it/s] Capturing num tokens (num_tokens=80 avail_mem=128.79 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.20it/s]Capturing num tokens (num_tokens=80 avail_mem=128.79 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.52it/s]Capturing num tokens (num_tokens=64 avail_mem=128.79 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.52it/s]Capturing num tokens (num_tokens=48 avail_mem=128.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.52it/s]Capturing num tokens (num_tokens=32 avail_mem=128.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.52it/s]Capturing num tokens (num_tokens=32 avail_mem=128.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=28 avail_mem=128.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=24 avail_mem=128.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.75it/s]

    Capturing num tokens (num_tokens=20 avail_mem=128.86 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=16 avail_mem=128.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=16 avail_mem=128.85 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s]Capturing num tokens (num_tokens=12 avail_mem=128.84 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s]Capturing num tokens (num_tokens=8 avail_mem=128.81 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s] Capturing num tokens (num_tokens=4 avail_mem=128.82 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s]Capturing num tokens (num_tokens=4 avail_mem=128.82 GB): 100%|██████████| 58/58 [00:02<00:00, 27.67it/s]


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
    Generated text:  Michael Jordan, and I'm an American professional basketball player. I've been playing for over 40 years and have represented my country and helped to make great strides for the United States in sports.
    I've been fortunate to have received numerous accolades in my career, including four NBA championships, six MVP awards, and three NBA Finals MVP awards. I've also won eight Olympic gold medals, one World Championship, and received seven NBA All-Star Games selections.
    In addition to basketball, I've worked as a successful businessman, a television personality, and have been involved in various charitable causes, including helping to establish the Michael Jordan Foundation and the
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. Which of the following is the best example of a situation where the Vice President would be in the best position to make an official statement to the public? A. If the President is in a heated argument with the Vice President; B. If the President is on a personal vacation; C. If the President is away on a trip overseas; D. If the President has a hospital visit and the Vice President is present; D. If the President has a hospital visit and the Vice President is present; D. If the President has a hospital visit and the Vice President is present; D. If the President has
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It's one of the most important cities in the world. There are lots of interesting things to see and do in Paris. Here are some of the most popular things to see and do in Paris. The Eiffel Tower is a symbol of Paris. It stands tall, and is worth visiting for its view of the city. The Louvre is the largest art museum in the world. It's famous for its beautiful paintings. The Louvre is also known as the "Museum of History and Civilization". The Louvre is open 24 hours a day, 7 days a week. It's very popular with visitors
    ===============================
    Prompt: The future of AI is
    Generated text:  human-ai hybridity
    
    That’s the theme of a recent workshop on the intersection of AI and human values led by the Institute of Ethics and Ethics in Inclusive Development (IEID), a non-profit organisation based in the UK. The workshop, organised by Professor Julia McCalman, the author of the book ‘AI and the Future of Humanity: Ethical Challenges’, aimed to explore the future of AI through a human-centric lens.
    
    Professor McCalman outlined that although the development of artificial intelligence has been hailed as a game-changer for mankind, it also raises a host of ethical challenges. While humans and AI have long been seen


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is a popular tourist destination and a major center for art, fashion, and cuisine. Its status as a major European capital has made it a significant player in the global economy. The city is also home to numerous museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city that continues to thrive as a major global city. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of devices and systems, from smart homes to self-driving cars. As more devices and systems become connected, we can expect to see even more integration of AI with other technologies, such as sensors, cameras, and machine learning algorithms.
    
    2. Enhanced capabilities: AI is likely to become even more capable in the future, with the ability to learn from data, adapt to new situations, and make decisions based on complex information. This could lead to more sophisticated and autonomous AI systems that can perform
    


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
    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm [Height] feet tall and [Weight] pounds. I was born in [Birthplace] and grew up in [Birthplace]. My [Profession] started [When] and ended [When]. I've always been [What's Your Favorite Thing to Do?]. I've always been [What's Your Ideal Job?]. I've always been [What's Your Life Goal?]. I'm always looking for [What's Your Problem?]. I'm always looking for [What's My Answer to Life's Question?].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city is known for its stunning architecture, rich history, and delicious cuisine. The city is also a major center for the arts and a leading political and cultural center in Europe. Additionally, Paris is a popular tourist destination and hosts numerous cultural and sporting events throughout the year. Paris has a population of approximately 2 million and is the largest city in Europe by population. As of 2020, Paris had a GDP of approximately $877 billion, making it one of the world’s largest economies. The city's status as a global financial hub, with the headquarters of many major corporations located there, has helped
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are several trends and possibilities that we can expect to see in the coming years. Here are some of the most promising future trends in artificial intelligence:
    
    1. Personalized AI: As AI continues to become more advanced, we will see an increase in personalized AI, where AI systems can learn from user data and adjust their behavior accordingly. This could result in more efficient and effective use of resources, as well as better customer service and experiences.
    
    2. Autonomous AI: Autonomous AI refers to machines that can operate without human intervention, such as robots, drones, and self-driving cars. While this technology is still in its early


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

     first

     name

     and

     last

     name

    ],

     and

     I

     am

     [

    insert

     age

    ].

     I

     love

     to

     explore

     new

     things

     and

     learn

     new

     things

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     grow

     and

     learn

    .

     I

     enjoy

     hanging

     out

     with

     friends

     and

     spending

     time

     in

     the

     outdoors

    ,

     and

     I

    'm

     always

     looking

     for

     fun

     activities

     to

     do

    .

     What

    's

     your

     favorite

     hobby

    ?

     And

     what

     do

     you

     like

     to

     do

     for

     entertainment

    ?


    [

    insert

     age

    ]


    I

    ’m

     a

     [

    insert

     occupation

    ].

     And

     I

     love

     [

    insert

     what

     I

     do

    ].

     I

    ’m

     always

     on

     the

     lookout

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     What

    's

     your

     favorite

     hobby

    ?

     And

     what

     do

     you

     like

     to

     do

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     historic

     Lou

    vre

     Museum

    ,

     and

     vibrant

     French

     culture

    .

     
    


    **

    Summary

     of

     Paris

    :

    **
    


    *

     **

    Location

    :**

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     North

     of

     the

     country

    .


    *

     **

    Population

    :**

     The

     city

    's

     population

     is

     around

     

    2

    .

     

    5

     million

    ,

     with

     the

     metropolitan

     area

     having

     over

     

    6

     million

     residents

    .


    *

     **

    Language

    :**

     French

     is

     the

     official

     language

     of

     the

     country

     and

     is

     the

     primary

     language

     of

     Paris

    .

     The

     French

     language

     is

     spoken

     by

     about

     

    6

    0

    %

     of

     the

     population

    .


    *

     **

    Culture

    :**

     Paris

     is

     known

     for

     its

     cultural

     diversity

    ,

     with

     a

     blend

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promising

    ,

     with

     a

     wide

     range

     of

     potential

     trends

     that

     could

     shape

     the

     technology

    's

     development

     and

     impact

     on

     society

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

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     become

     even

     more

     challenging

     to

     determine

     the

     appropriate

     limits

     and

     implications

     of

     its

     use

    .

     As

     a

     result

    ,

     there

     will

     be

     an

     increased

     focus

     on

     ethical

     considerations

     and

     the

     development

     of

     AI

     that

     is

     designed

     to

     be

     more

     ethical

     and

     transparent

    .
    


    2

    .

     Integration

     of

     AI

     into

     new

     areas

     of

     human

     activity

    :

     AI

     is

     already

     being

     used

     in

     a

     variety

     of

     new

     and

     innovative

     areas

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     manufacturing

    .

    



```python
llm.shutdown()
```
