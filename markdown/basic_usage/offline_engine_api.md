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

    [2026-03-10 05:48:18] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-10 05:48:18] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-10 05:48:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-10 05:48:21] INFO server_args.py:2133: Attention backend not specified. Use fa3 backend by default.


    [2026-03-10 05:48:21] INFO server_args.py:3246: Set soft_watchdog_timeout since in CI


    [2026-03-10 05:48:21] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=175796314, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:04,  2.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:04,  2.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:04,  2.19s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:04,  2.19s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:07,  6.83it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 16.16it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 26.45it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:02<00:00, 32.52it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:02<00:00, 43.30it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 43.30it/s]

    Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 43.30it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 42.24it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 42.24it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 42.24it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 42.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 18.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   2%|▏         | 1/58 [00:00<00:08,  6.73it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.26 GB):   2%|▏         | 1/58 [00:00<00:08,  6.73it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.26 GB):   3%|▎         | 2/58 [00:00<00:08,  6.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.26 GB):   3%|▎         | 2/58 [00:00<00:08,  6.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.26 GB):   5%|▌         | 3/58 [00:00<00:07,  7.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.26 GB):   5%|▌         | 3/58 [00:00<00:07,  7.02it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.26 GB):   7%|▋         | 4/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.26 GB):   7%|▋         | 4/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.26 GB):   9%|▊         | 5/58 [00:00<00:06,  7.58it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.25 GB):   9%|▊         | 5/58 [00:00<00:06,  7.58it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.25 GB):  10%|█         | 6/58 [00:00<00:06,  7.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.26 GB):  10%|█         | 6/58 [00:00<00:06,  7.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.26 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.25 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.25 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.25 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.25 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.01it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.25 GB):  16%|█▌        | 9/58 [00:04<00:05,  9.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.25 GB):  17%|█▋        | 10/58 [00:04<00:50,  1.05s/it]Capturing num tokens (num_tokens=3584 avail_mem=58.24 GB):  17%|█▋        | 10/58 [00:04<00:50,  1.05s/it]Capturing num tokens (num_tokens=3328 avail_mem=58.24 GB):  17%|█▋        | 10/58 [00:04<00:50,  1.05s/it]Capturing num tokens (num_tokens=3072 avail_mem=58.23 GB):  17%|█▋        | 10/58 [00:04<00:50,  1.05s/it]Capturing num tokens (num_tokens=3072 avail_mem=58.23 GB):  22%|██▏       | 13/58 [00:04<00:21,  2.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.23 GB):  22%|██▏       | 13/58 [00:04<00:21,  2.10it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.23 GB):  22%|██▏       | 13/58 [00:04<00:21,  2.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.22 GB):  22%|██▏       | 13/58 [00:04<00:21,  2.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.22 GB):  22%|██▏       | 13/58 [00:04<00:21,  2.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.22 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.22 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.21 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.21 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.19 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s]Capturing num tokens (num_tokens=960 avail_mem=58.20 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s] Capturing num tokens (num_tokens=896 avail_mem=58.20 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.11it/s]Capturing num tokens (num_tokens=896 avail_mem=58.20 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.96it/s]Capturing num tokens (num_tokens=832 avail_mem=58.20 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.96it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.19 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.96it/s]Capturing num tokens (num_tokens=704 avail_mem=58.19 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.96it/s]Capturing num tokens (num_tokens=640 avail_mem=58.19 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.96it/s]Capturing num tokens (num_tokens=576 avail_mem=58.19 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.96it/s]Capturing num tokens (num_tokens=576 avail_mem=58.19 GB):  48%|████▊     | 28/58 [00:04<00:02, 11.75it/s]Capturing num tokens (num_tokens=512 avail_mem=58.17 GB):  48%|████▊     | 28/58 [00:04<00:02, 11.75it/s]Capturing num tokens (num_tokens=480 avail_mem=58.19 GB):  48%|████▊     | 28/58 [00:04<00:02, 11.75it/s]Capturing num tokens (num_tokens=448 avail_mem=58.19 GB):  48%|████▊     | 28/58 [00:04<00:02, 11.75it/s]Capturing num tokens (num_tokens=416 avail_mem=58.19 GB):  48%|████▊     | 28/58 [00:04<00:02, 11.75it/s]Capturing num tokens (num_tokens=384 avail_mem=57.15 GB):  48%|████▊     | 28/58 [00:04<00:02, 11.75it/s]

    Capturing num tokens (num_tokens=384 avail_mem=57.15 GB):  57%|█████▋    | 33/58 [00:04<00:01, 14.94it/s]Capturing num tokens (num_tokens=352 avail_mem=57.15 GB):  57%|█████▋    | 33/58 [00:04<00:01, 14.94it/s]Capturing num tokens (num_tokens=320 avail_mem=57.14 GB):  57%|█████▋    | 33/58 [00:04<00:01, 14.94it/s]Capturing num tokens (num_tokens=288 avail_mem=57.14 GB):  57%|█████▋    | 33/58 [00:05<00:01, 14.94it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.14 GB):  57%|█████▋    | 33/58 [00:05<00:01, 14.94it/s]Capturing num tokens (num_tokens=256 avail_mem=58.14 GB):  64%|██████▍   | 37/58 [00:05<00:01, 14.48it/s]Capturing num tokens (num_tokens=240 avail_mem=58.14 GB):  64%|██████▍   | 37/58 [00:05<00:01, 14.48it/s]Capturing num tokens (num_tokens=224 avail_mem=57.31 GB):  64%|██████▍   | 37/58 [00:05<00:01, 14.48it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.31 GB):  64%|██████▍   | 37/58 [00:05<00:01, 14.48it/s]Capturing num tokens (num_tokens=208 avail_mem=57.31 GB):  69%|██████▉   | 40/58 [00:05<00:01, 14.22it/s]Capturing num tokens (num_tokens=192 avail_mem=57.31 GB):  69%|██████▉   | 40/58 [00:05<00:01, 14.22it/s]Capturing num tokens (num_tokens=176 avail_mem=57.30 GB):  69%|██████▉   | 40/58 [00:05<00:01, 14.22it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.12 GB):  69%|██████▉   | 40/58 [00:05<00:01, 14.22it/s]Capturing num tokens (num_tokens=160 avail_mem=58.12 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.62it/s]Capturing num tokens (num_tokens=144 avail_mem=58.12 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.62it/s]Capturing num tokens (num_tokens=128 avail_mem=57.35 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.62it/s]

    Capturing num tokens (num_tokens=112 avail_mem=57.35 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.62it/s]Capturing num tokens (num_tokens=112 avail_mem=57.35 GB):  79%|███████▉  | 46/58 [00:05<00:00, 13.95it/s]Capturing num tokens (num_tokens=96 avail_mem=57.35 GB):  79%|███████▉  | 46/58 [00:05<00:00, 13.95it/s] Capturing num tokens (num_tokens=80 avail_mem=58.11 GB):  79%|███████▉  | 46/58 [00:05<00:00, 13.95it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.11 GB):  83%|████████▎ | 48/58 [00:05<00:00, 14.30it/s]Capturing num tokens (num_tokens=64 avail_mem=57.40 GB):  83%|████████▎ | 48/58 [00:05<00:00, 14.30it/s]Capturing num tokens (num_tokens=48 avail_mem=57.40 GB):  83%|████████▎ | 48/58 [00:06<00:00, 14.30it/s]Capturing num tokens (num_tokens=48 avail_mem=57.40 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.75it/s]Capturing num tokens (num_tokens=32 avail_mem=57.39 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.75it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.10 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.75it/s]Capturing num tokens (num_tokens=28 avail_mem=58.10 GB):  90%|████████▉ | 52/58 [00:06<00:00, 14.22it/s]Capturing num tokens (num_tokens=24 avail_mem=57.44 GB):  90%|████████▉ | 52/58 [00:06<00:00, 14.22it/s]Capturing num tokens (num_tokens=20 avail_mem=57.44 GB):  90%|████████▉ | 52/58 [00:06<00:00, 14.22it/s]

    Capturing num tokens (num_tokens=20 avail_mem=57.44 GB):  93%|█████████▎| 54/58 [00:06<00:00, 13.62it/s]Capturing num tokens (num_tokens=16 avail_mem=57.44 GB):  93%|█████████▎| 54/58 [00:06<00:00, 13.62it/s]Capturing num tokens (num_tokens=12 avail_mem=58.09 GB):  93%|█████████▎| 54/58 [00:06<00:00, 13.62it/s]Capturing num tokens (num_tokens=12 avail_mem=58.09 GB):  97%|█████████▋| 56/58 [00:06<00:00, 14.40it/s]Capturing num tokens (num_tokens=8 avail_mem=57.49 GB):  97%|█████████▋| 56/58 [00:06<00:00, 14.40it/s] Capturing num tokens (num_tokens=4 avail_mem=57.48 GB):  97%|█████████▋| 56/58 [00:06<00:00, 14.40it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.48 GB): 100%|██████████| 58/58 [00:06<00:00, 13.71it/s]Capturing num tokens (num_tokens=4 avail_mem=57.48 GB): 100%|██████████| 58/58 [00:06<00:00,  8.64it/s]


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
    Generated text:  Tyktn and I am a digital artist. I am passionate about creating digital art that captures moments of beauty and emotion from everyday experiences. I specialize in creating surreal and whimsical art that is both visually stunning and emotionally evocative. My primary goal is to inspire and uplift people with my art, and to use my platform to make a positive impact on the world. I am currently a senior at the University of the Arts in London, England and I have been working in the creative arts for over a decade. I am currently in the final year of my degree and am working on a range of projects that I am excited to share
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what sport to play on the country's golf course. There are four sports to choose from: tennis, cricket, golf, and volleyball. The president is very enthusiastic about cricket and wants to play it on his country's golf course. He has to choose one sport from the four options. Can you determine how many possibilities are there for the president to choose between the four sports? Yes, there are two possibilities for the president to choose between the four sports.
    The first possibility is that he chooses to play cricket on the golf course, since he is very enthusiastic about it. The second possibility is that he chooses to play golf
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the western bank of the Seine river. The city is a popular place for tourists to visit, as well as a place of business and manufacturing, as well as a place for residents to live and work. The city is also an important place of government, and the president of France is elected here.
    The capital of France is Paris. It is situated at the western end of the French region of Paris.
    Paris is located on the western bank of the Seine river. It is the second largest city in France and the second largest in the world. The city is also the capital of France.
    The city is also one of
    ===============================
    Prompt: The future of AI is
    Generated text:  here. In the years to come, artificial intelligence will bring about a major change in every aspect of our lives. Whether it is the way we live, work, learn, or play, AI is going to have a profound impact on our society and our way of life.
    There are several ways in which AI is transforming the way we live and work:
    1. Smartphones: AI-powered smartphones have become an essential part of our lives. They can recognize and understand our emotions, learn about our habits, and even play music for us. They are also able to suggest movies, books, and other content that may interest us based on our


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center that plays a significant role in French politics and society. It is also known for its rich history, including the influence of the French Revolution and the influence of the French Renaissance. The city is home to many famous French artists, writers, and musicians, and is a popular tourist destination. Paris is a city that is both beautiful and bustling, and is a must-
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management,
    


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
    Generated text:  [insert your name here] and I am a [insert your occupation or profession] with a passion for [insert something you enjoy doing that you're passionate about]. I'm always looking to learn new things and try new experiences. Whether it's exploring new cultures, trying new hobbies, or simply spending time with friends and family, I'm always happy to make new connections and find new adventures. I'm a [insert something you can do that shows your personality and personality traits] and I'm always looking for ways to make my life more interesting and fun. How can I get to know you better? You can learn more about me by
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and one of the world’s most important cities. The city is home to many notable landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, the Louvre Museum, and the Palace of Versailles. It is a historic city with a rich history, including the ancient Roman city of Carthage, the ancient Arab city of Marrakech, and the medieval city of Paris. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid technological advancements, particularly in the areas of machine learning, deep learning, natural language processing, and robotics. Here are some of the potential future trends in AI:
    
    1. Increased efficiency and productivity: As AI technology continues to advance, it is expected to automate many tasks, reducing the need for human intervention and increasing efficiency in various industries. For example, AI-powered chatbots can handle customer service inquiries, while autonomous vehicles can reduce traffic congestion and increase safety.
    
    2. AI in healthcare: AI is already being used to develop new treatments for diseases and improve patient care. In the future, we may see a significant


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

     your

     name

     here

    ].

     I

    'm

     [

    insert

     your

     age

     here

    ],

     a

     [

    insert

     your

     occupation

     or

     profession

     here

    ]

     who

     loves

     to

     [

    insert

     one

     or

     two

     activities

     here

    ].

     I

    've

     always

     had

     a

     fascination

     with

     [

    insert

     your

     favorite

     hobby

     or

     interest

     here

    ],

     and

     I

    've

     decided

     to

     start

     following

     you

     as

     well

    .

     I

     hope

     we

     can

     connect

     and

     get

     to

     know

     each

     other

     better

    .

     [

    Insert

     your

     profession

     or

     background

     here

    ].

     I

     hope

     to

     always

     find

     joy

     in

     the

     simple

     things

     in

     life

    ,

     such

     as

     spending

     time

     with

     loved

     ones

    ,

     practicing

     mindfulness

    ,

     and

     pursuing

     my

     passion

    .

     I

     hope

     to

     make

     new

     friends

     and

     meet

     new

     people

     who

     inspire

     me

     to

     grow

     and

     change

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Key

     facts

     about

     Paris

    :
    


    -

     It

     is

     the

     capital

     of

     France

     and

     the

     largest

     city

     in

     Europe

     by

     population

    .


    -

     It

     is

     located

     in

     the

     northern

     part

     of

     the

     country

    ,

     near

     the

     Mediterranean

     Sea

    .


    -

     It

     is

     famous

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     various

     museums

     and

     attractions

    .


    -

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     cuisine

    ,

     and

     culture

    .

     
    


    The

     city

     is

     also

     home

     to

     many

     influential

     figures

     and

     events

     in

     French

     history

    .

     Its

     status

     as

     the

     capital

     is

     a

     key

     factor

     in

     its

     role

     as

     a

     major

     European

     city

    .

     However

    ,

     the

     city

     has

     faced

     challenges

     with

     gent

    r

    ification

     and

     rising

     housing

     prices

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     one

     of

     rapid

     change

     and

     transformation

    ,

     with

     many

     potential

     future

     trends

     shaping

     the

     direction

     of

     the

     industry

    .

     Some

     of

     the

     possible

     trends

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     will

     play

     a

     critical

     role

     in

     healthcare

    ,

     improving

     diagnostic

     accuracy

     and

     personal

    izing

     treatments

     for

     patients

     with

     complex

     medical

     conditions

    .
    


    2

    .

     Em

    phasis

     on

     automation

    :

     AI

     will

     continue

     to

     automate

     repetitive

     and

     time

    -consuming

     tasks

    ,

     freeing

     up

     human

     resources

     for

     more

     strategic

     and

     creative

     work

    .
    


    3

    .

     Development

     of

     ethical

     AI

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     increasing

     emphasis

     on

     ethical

     considerations

    ,

     including

     issues

     of

     privacy

    ,

     bias

    ,

     and

     accountability

    .
    


    4

    .

     Growth

     of

     AI

     for

     personal

    ization

    



```python
llm.shutdown()
```
