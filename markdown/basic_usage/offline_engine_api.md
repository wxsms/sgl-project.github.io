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

    [2026-03-18 17:20:23] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-18 17:20:23] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-18 17:20:23] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-18 17:20:30] INFO server_args.py:2176: Attention backend not specified. Use fa3 backend by default.


    [2026-03-18 17:20:30] INFO server_args.py:3372: Set soft_watchdog_timeout since in CI


    [2026-03-18 17:20:30] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=426676628, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.89it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.89it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:13,  2.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:13,  2.35s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:57,  1.03s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:57,  1.03s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:57,  1.03s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.99it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.99it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.99it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.99it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.82it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.82it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.82it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.82it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:07,  6.82it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:02<00:04, 10.94it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:02<00:04, 10.94it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:02<00:04, 10.94it/s]

    Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:02, 15.32it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:02, 15.32it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:02, 15.32it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:02, 15.32it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:02, 15.32it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:01, 19.26it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:01, 19.26it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:01, 19.26it/s]

    Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:01, 19.26it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:01, 19.26it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:01, 19.26it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:00, 28.68it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 31.88it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 31.88it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 31.88it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 31.88it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 31.88it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 31.88it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:03<00:00, 35.61it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 48.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=960 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s] Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=768 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=768 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.88it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.88it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.88it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.88it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.88it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.88it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  60%|██████    | 35/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:00<00:00, 44.62it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  71%|███████   | 41/58 [00:01<00:00, 46.88it/s] Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  81%|████████  | 47/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.63it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.63it/s] Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 40.87it/s]


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
    Generated text:  Yu-Ting, and I’m a single mom who has come to the United States on a work-study program to pursue a master’s degree in educational technology. I am currently studying at New York University’s Graduate School of Education, and I’m studying to become a teacher. I’ve been living in New York City for the past few years, but I also have a strong interest in staying connected with my family back home in Taiwan. Since I was 12, I’ve been traveling around the country and helping my parents with their daily lives. While in Taiwan, I’ve always been deeply interested in using technology to help people.
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official, serving a four-year term. The president is the head of the executive branch of the U. S. government, and is the commander-in-chief of the armed forces. The president is chosen by the United States citizens through a process of a popular election.
    
    The first president of the United States was George Washington, who served from 1789 to 1797. The president also serves on the federal bench, which includes a position of "judge of the Supreme Court of the United States."
    
    The term of office for a president is 4 years. The president may be re-elected in a special election
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Lyon
    B. Paris
    C. Marseille
    D. Nice
    Answer:
    B
    
    Which of the following is NOT a quality management system document?
    A. Quality policy
    B. Work procedures
    C. Quality plan
    D. Quality objectives
    Answer:
    B
    
    To which category does a large organization's information system belong? ____ 
    A. Information System
    B. Information-Intensive System
    C. Information System
    D. Information-Intensive System
    Answer:
    A
    
    Which of the following statements is NOT true about the business environment?
    A. A business environment influences the choice of functional departments, human
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the hardware
    AI has the potential to transform how we live and work, especially in an age when the sheer volume of data is exploding and the complexity of the tasks that need to be done is growing exponentially.
    In this context, IBM has been investing heavily in the development of the next generation of artificial intelligence systems. One of the companies that have taken a lead in this regard is IBM Research.
    IBM has been working on a range of AI systems that are designed to tackle a range of different challenges. Their focus is on the development of a system that can understand the complex human emotions and the language of human beings.
    IBM Research has


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a few key points about yourself, such as your personality, skills, or experiences]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [insert a few key points about your job, such as your job title, company, or what you do for a living]. I'm always looking for ways to improve my skills and stay up-to-date with the latest
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its vibrant culture, including its annual festivals and events. Paris is a popular tourist destination and a major economic center in France. It is home to many famous museums, art galleries, and restaurants. The city is also known for its cuisine, with dishes such as croissants, boudin, and escargot being popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI systems that can better understand and respond to the needs of individuals.
    
    2. Greater use of machine learning: Machine learning is likely to become more prevalent, with more sophisticated algorithms and models being developed to improve AI systems' ability to learn and adapt to new situations. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Greater emphasis on ethical considerations: As AI
    


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
    Generated text:  [Name], and I'm a [job title] at [company]. I bring a fresh perspective, always looking for ways to improve efficiency and productivity in the workplace. What brings you here today? What's one of your favorite moments from work? How can I support you in your journey to success? What's the biggest challenge you've faced and how did you overcome it? How do you handle conflicts or disagreements in the workplace? What's your ultimate goal, and how are you going to achieve it? What's your favorite hobby, and what's your passion for? What are your hobbies or interests outside of work? What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with a population of over 2 million people.
    Paris is known for its historical architecture, world-renowned museums, and annual festivals such as the Eiffel Tower Parc du Joyeuse Fête and the Moulin Rouge. It's also known for its vibrant nightlife, art scene, and wine culture. Visitors can explore the city’s narrow streets, historical landmarks, and cultural attractions, including the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city also boasts a rich history of fashion, gastronomy, and cuisine. Paris is a city of contrasts, with the opulent
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a wide range of changes, including:
    
    1. Increased efficiency and accuracy: AI is becoming more capable of performing tasks faster and with greater precision than ever before. We can expect to see AI systems that are more efficient, accurate, and capable of handling complex problems.
    
    2. Integration with human AI: AI will become even more integrated with human AI, enabling more intelligent and complex interactions between humans and machines.
    
    3. Personalization: AI will allow machines to learn and adapt to our preferences and needs, providing more personalized experiences.
    
    4. Improved transparency: AI systems will become more transparent, allowing us to see how they


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

    ],

     and

     I

    'm

     a

     [

    career

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

    .

     My

     professional

     background

     includes

     [

    mention

     any

     relevant

     achievements

     or

     certifications

    ].

     How

     can

     I

     assist

     you

     today

    ?

     [

    Type

     in

     any

     additional

     information

     you

     feel

     comfortable

     sharing

    ]

     Start

     with

     an

     introduction

    ,

     and

     then

     explain

     what

     you

     can

     do

     for

     them

    .

     Let

    's

     get

     started

    !

     (

    Optional

    :

     Include

     a

     short

     quote

     or

     a

     personal

     anecd

    ote

     to

     set

     the

     tone

     for

     the

     conversation

    .)

     Start

     your

     introduction

    :

     [

    Name

    ]

     [

    Brief

     introduction

    ]

     Now

     that

     we

     know

     each

     other

    ,

     I

    'm

     excited

     to

     learn

     more

     about

     [

    career

    ],

     and

     I

     believe

     [

    career

    ]

     is

     the

     perfect

     match

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     home

     to

     iconic

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

    ,

     as

     well

     as

     a

     lively

     arts

     scene

     and

     a

     diverse

     population

    .

     Paris

     is

     also

     known

     for

     its

     international

     cuisine

    ,

     with

     numerous

     restaurants

     and

     bars

     serving

     delicious

     French

     dishes

    .

     Despite

     its

     famous

     landmarks

    ,

     Paris

     is

     also

     a

     bustling

     city

     with

     a

     vibrant

     nightlife

     and

     a

     growing

     economy

    ,

     attracting

     tourists

     and

     residents

     alike

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     several

     trends

     shaping

     its

     direction

     and

     impact

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     likely

     to

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     becoming

     increasingly

     integrated

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     IoT

    ,

     and

     edge

     computing

    .

     This

     integration

     will

     enable

     AI

     to

     work

     more

     seamlessly

     and

     co

    exist

     with

     other

     systems

    ,

     leading

     to

     new

     applications

     and

     opportunities

    .
    


    2

    .

     Personal

    ization

     and

     customization

    :

     AI

     will

     become

     more

     personalized

     and

     customized

    ,

     allowing

     machines

     to

     learn

     from

     user

     behavior

     and

     preferences

     to

     provide

     more

     relevant

     and

     effective

     results

    .

     This

     trend

     will

     enable

     AI

     to

     become

     more

     human

    -like

     and

     create

     more

     intuitive

     and

     seamless

    



```python
llm.shutdown()
```
