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

    [2026-03-19 07:13:07] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-19 07:13:07] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-19 07:13:07] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-19 07:13:10] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-19 07:13:10] INFO server_args.py:2183: Attention backend not specified. Use fa3 backend by default.


    [2026-03-19 07:13:10] INFO server_args.py:3410: Set soft_watchdog_timeout since in CI


    [2026-03-19 07:13:10] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=381754911, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.64it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.64it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:39,  1.39it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:39,  1.39it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:39,  1.39it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:39,  1.39it/s]

    Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:39,  1.39it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=2816):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=2560):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:02<00:04, 10.62it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:02<00:04, 10.62it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:02<00:04, 10.62it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:02<00:04, 10.62it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:02<00:04, 10.62it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:02<00:04, 10.62it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:03<00:04, 10.62it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:03<00:04, 10.62it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:03<00:04, 10.62it/s]Compiling num tokens (num_tokens=832):  26%|██▌       | 15/58 [00:03<00:04, 10.62it/s]Compiling num tokens (num_tokens=768):  26%|██▌       | 15/58 [00:03<00:04, 10.62it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=320):  43%|████▎     | 25/58 [00:03<00:01, 20.34it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]

    Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 30.78it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:03<00:00, 36.57it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 45.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=70.74 GB):   2%|▏         | 1/58 [00:00<00:12,  4.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.16 GB):   2%|▏         | 1/58 [00:00<00:12,  4.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.16 GB):   3%|▎         | 2/58 [00:00<00:10,  5.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.52 GB):   3%|▎         | 2/58 [00:00<00:10,  5.58it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=70.52 GB):   5%|▌         | 3/58 [00:00<00:08,  6.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.52 GB):   5%|▌         | 3/58 [00:00<00:08,  6.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.17 GB):   5%|▌         | 3/58 [00:00<00:08,  6.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.17 GB):   9%|▊         | 5/58 [00:00<00:06,  8.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.50 GB):   9%|▊         | 5/58 [00:00<00:06,  8.63it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=70.50 GB):  10%|█         | 6/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.51 GB):  10%|█         | 6/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.31 GB):  10%|█         | 6/58 [00:00<00:05,  8.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.31 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.51 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.50 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.10it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=70.50 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.49 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.50 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.50 GB):  21%|██        | 12/58 [00:01<00:03, 13.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.28 GB):  21%|██        | 12/58 [00:01<00:03, 13.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.48 GB):  21%|██        | 12/58 [00:01<00:03, 13.45it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=70.47 GB):  21%|██        | 12/58 [00:01<00:03, 13.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.47 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.47 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.46 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.46 GB):  31%|███       | 18/58 [00:01<00:02, 18.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.45 GB):  31%|███       | 18/58 [00:01<00:02, 18.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.44 GB):  31%|███       | 18/58 [00:01<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=70.42 GB):  31%|███       | 18/58 [00:01<00:02, 18.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.42 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=960 avail_mem=70.43 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.15it/s] Capturing num tokens (num_tokens=896 avail_mem=70.42 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=832 avail_mem=70.39 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=768 avail_mem=70.39 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=768 avail_mem=70.39 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.24it/s]Capturing num tokens (num_tokens=704 avail_mem=70.39 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.24it/s]Capturing num tokens (num_tokens=640 avail_mem=70.40 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.24it/s]

    Capturing num tokens (num_tokens=576 avail_mem=70.38 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.24it/s]Capturing num tokens (num_tokens=512 avail_mem=70.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.24it/s]Capturing num tokens (num_tokens=512 avail_mem=70.34 GB):  50%|█████     | 29/58 [00:01<00:01, 27.16it/s]Capturing num tokens (num_tokens=480 avail_mem=70.37 GB):  50%|█████     | 29/58 [00:01<00:01, 27.16it/s]Capturing num tokens (num_tokens=448 avail_mem=70.36 GB):  50%|█████     | 29/58 [00:01<00:01, 27.16it/s]Capturing num tokens (num_tokens=416 avail_mem=70.38 GB):  50%|█████     | 29/58 [00:01<00:01, 27.16it/s]Capturing num tokens (num_tokens=384 avail_mem=70.37 GB):  50%|█████     | 29/58 [00:01<00:01, 27.16it/s]Capturing num tokens (num_tokens=384 avail_mem=70.37 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=352 avail_mem=70.36 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=320 avail_mem=70.35 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]

    Capturing num tokens (num_tokens=288 avail_mem=70.35 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=256 avail_mem=70.34 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=256 avail_mem=70.34 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=240 avail_mem=70.34 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=224 avail_mem=70.35 GB):  64%|██████▍   | 37/58 [00:02<00:00, 32.04it/s]Capturing num tokens (num_tokens=208 avail_mem=70.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 32.04it/s]Capturing num tokens (num_tokens=192 avail_mem=70.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 32.04it/s]Capturing num tokens (num_tokens=192 avail_mem=70.31 GB):  71%|███████   | 41/58 [00:02<00:00, 32.78it/s]Capturing num tokens (num_tokens=176 avail_mem=70.30 GB):  71%|███████   | 41/58 [00:02<00:00, 32.78it/s]Capturing num tokens (num_tokens=160 avail_mem=70.30 GB):  71%|███████   | 41/58 [00:02<00:00, 32.78it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.31 GB):  71%|███████   | 41/58 [00:02<00:00, 32.78it/s]Capturing num tokens (num_tokens=128 avail_mem=70.31 GB):  71%|███████   | 41/58 [00:02<00:00, 32.78it/s]Capturing num tokens (num_tokens=112 avail_mem=70.30 GB):  71%|███████   | 41/58 [00:02<00:00, 32.78it/s]Capturing num tokens (num_tokens=112 avail_mem=70.30 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.28it/s]Capturing num tokens (num_tokens=96 avail_mem=70.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.28it/s] Capturing num tokens (num_tokens=80 avail_mem=70.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.28it/s]Capturing num tokens (num_tokens=64 avail_mem=70.28 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.28it/s]Capturing num tokens (num_tokens=48 avail_mem=70.25 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.28it/s]Capturing num tokens (num_tokens=32 avail_mem=70.25 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.28it/s]

    Capturing num tokens (num_tokens=32 avail_mem=70.25 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.54it/s]Capturing num tokens (num_tokens=28 avail_mem=70.24 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.54it/s]Capturing num tokens (num_tokens=24 avail_mem=70.25 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.54it/s]Capturing num tokens (num_tokens=20 avail_mem=70.24 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.54it/s]Capturing num tokens (num_tokens=16 avail_mem=70.24 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.54it/s]Capturing num tokens (num_tokens=16 avail_mem=70.24 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.36it/s]Capturing num tokens (num_tokens=12 avail_mem=70.23 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.36it/s]Capturing num tokens (num_tokens=8 avail_mem=70.23 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.36it/s] Capturing num tokens (num_tokens=4 avail_mem=70.22 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.36it/s]Capturing num tokens (num_tokens=4 avail_mem=70.22 GB): 100%|██████████| 58/58 [00:02<00:00, 22.89it/s]


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
    Generated text:  Wendy, I am a 14 year old and I am from England. I am a smart girl who loves to read, listen to music, and play games. I also love to cook and I am good at it. In addition, I have a pet dog named Bella. It is a great dog because it is very loyal to her master and it is also very good at playing with people. I have been playing with Bella for a few days now and I have learned a lot about her. My parents are both writers and they write in different genres. My father writes romance novels and my mother writes mystery novels. I am also
    ===============================
    Prompt: The president of the United States is
    Generated text:  inaugurated every four years. His successor is elected in a different year. If the two candidates have been competing for a combined total of 24 years, and both candidates have been in office for at least 2 years, what is the minimum number of years the two candidates have been competing for office?
    
    To determine the minimum number of years the two candidates have been competing for office, let's denote the number of years the two candidates have been competing as \( x \).
    
    Since the president is inaugurated every four years and the successor is elected every four years, the two candidates are competing for office in the following four-year cycles:
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Nice
    D. Barcelona
    Answer: A
    
    20. The most suitable temperature for storing protein powders is
    A. 20℃
    B. 15℃
    C. 10℃
    D. 25℃
    E. 30℃
    Answer: C
    
    The most suitable temperature for storing protein powders is
    A. 20℃
    B. 15℃
    C. 10℃
    D. 25℃
    E. 30℃
    Answer: C
    
    17.
    ===============================
    Prompt: The future of AI is
    Generated text:  dependent on the development of new, advanced technologies. The advent of AI is changing the way we do things, and is expected to continue to do so. As technology continues to advance, the role of human beings in the development and implementation of AI is likely to become increasingly important. AI is expected to have a significant impact on the way we live and work, and it is likely to play a role in shaping the future of our society.
    The development of AI is still in its early stages, and there are still many unknowns in this field. However, as we continue to develop and implement AI technologies, it is likely that we will


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is also known for its rich history, including the ancient Roman ruins of the Etruscan city of Rome, and its modern history, including the French Revolution and the French Revolution. The city is also home to many international organizations and events, including the World Cup and the Olympics. Paris is a vibrant and diverse city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we are likely to see more automation and artificial intelligence in our daily lives. This could include the automation of tasks such as customer service, transportation, and manufacturing, as well as the development of more advanced AI systems that can perform tasks that are currently performed by humans.
    
    2. Improved privacy and security: As AI systems become more advanced, there is a risk that they could be used to collect
    


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
    Generated text:  [Name] and I'm a [Job Title] who was born and raised in [Your Location] in [Year]. I'm [Age] years old. My favorite hobby is [Favorite Hobby] and I also love [Favorite Activity]. I'm [Position] at [Company Name], where I've been for [Number of Years]. I'm currently [Career Stage] in my professional career and I'm always looking for new challenges and opportunities to grow and learn. What's your [Most Common Word/Capital letter]?
    
    I'm a [Industry] expert with over [Number of Years] years of experience in the [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a bustling city with a rich history dating back over 2000 years. The city is famous for its iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. It is also known for its diverse food and wine culture, as well as its large population of around 2 million inhabitants. Additionally, Paris is a major international hub for finance, fashion, and art, and its vibrant nightlife and culture make it a popular tourist destination. The city is home to many of France's wealthiest individuals and has been the site of many important political and social movements. The French capital has been a major cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be dominated by the development of more advanced machine learning algorithms and the ability to process data at speeds and volumes never seen before. This will be realized through the use of more powerful computers and the development of new machine learning techniques that can recognize and understand human-like patterns in data. AI will also become increasingly integrated into everyday life, from self-driving cars to smart homes. It is also likely to become more collaborative, with more people working together to develop new AI technologies and tackle complex problems. Finally, the future of AI is likely to be shaped by regulatory developments, with governments and industry leaders working to establish clear guidelines and standards for the


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

     [

    Age

    ].

     I

     am

     a

     [

    Job

     Title

    ]

     with

     [

    Number

     of

     Years

     Experience

    ]

     years

     of

     experience

     in

     [

    Industry

    ].

     I

     graduated

     from

     [

    University

     Name

    ]

     in

     [

    Year

    ].

     I

     am

     currently

     working

     at

     [

    Company

     Name

    ]

     as

     a

     [

    Job

     Title

    ].

     I

     am

     passionate

     about

     [

    Inter

    ests

     or

     Expert

    ise

    ],

     and

     I

     enjoy

     [

    Favorite

     Activity

     or

     Hobby

    ].

     I

     am

     an

     [

    Ad

    jective

    ],

     and

     I

     am

     [

    In

     which

     Role

    ].

     I

     am

     [

    Position

    ]

     and

     I

     am

     [

    Employ

    er

    's

     Name

    ].

     I

     am

     [

    Name

    ]

     and

     I

     am

     excited

     to

     meet

     you

    .

     How

     can

     I

     help

     you

     today

    ?

     [

    Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     an

     iconic

     city

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     important

     historical

     landmarks

    ,

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

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     has

     been

     a

     major

     cultural

     and

     economic

     center

     for

     centuries

     and

     continues

     to

     be

     a

     dynamic

     city

     with

     a

     rich

     and

     diverse

     history

    .

     Visitors

     to

     Paris

     often

     come

     to

     experience

     its

     breathtaking

     beauty

     and

     take

     in

     its

     rich

     cultural

     heritage

    .

     The

     city

     is

     also

     famous

     for

     its

     food

     and

     drinks

    ,

     including

     its

     famous

     cro

    iss

    ants

    ,

     cheese

    ,

     and

     wine

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    ,

     with

     its

     unique

     charm

     and

     charm

    .

     As

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

    ,

     and

     it

     is

     difficult

     to

     predict

     exactly

     what

     trends

     will

     emerge

    .

     However

    ,

     there

     are

     several

     trends

     that

     are

     expected

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     already

     being

     developed

     and

     are

     expected

     to

     become

     more

     common

     in

     the

     coming

     years

    .

     The

     technology

     will

     improve

     as

     the

     need

     for

     human

     drivers

     increases

     and

     as

     the

     technology

     becomes

     more

     advanced

    .
    


    2

    .

     Natural

     language

     processing

    :

     As

     AI

     continues

     to

     improve

    ,

     so

     will

     the

     ability

     of

     AI

     systems

     to

     understand

     and

     interpret

     human

     language

    .

     This

     will

     lead

     to

     more

     natural

     and

     convers

    ational

     interactions

     between

     humans

     and

     AI

     systems

    .
    


    3

    .

     Cyber

    security

    :

     With

     the

     increasing

     use

     of

     AI

    ,

    



```python
llm.shutdown()
```
