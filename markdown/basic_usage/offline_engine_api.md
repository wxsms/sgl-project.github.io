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

    [2026-03-13 21:33:35] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-13 21:33:35] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-13 21:33:35] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-13 21:33:38] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.


    [2026-03-13 21:33:38] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-13 21:33:38] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=528643866, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04,  9.58it/s]

    Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04,  9.58it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]

    Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 40.76it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.03 GB):   2%|▏         | 1/58 [00:00<00:11,  5.06it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.17 GB):   2%|▏         | 1/58 [00:00<00:11,  5.06it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=71.17 GB):   2%|▏         | 1/58 [00:00<00:11,  5.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.17 GB):   5%|▌         | 3/58 [00:00<00:06,  8.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.16 GB):   5%|▌         | 3/58 [00:00<00:06,  8.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.05 GB):   5%|▌         | 3/58 [00:00<00:06,  8.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.05 GB):   9%|▊         | 5/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.14 GB):   9%|▊         | 5/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.12 GB):   9%|▊         | 5/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.11 GB):   9%|▊         | 5/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.11 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.12 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.13 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.90it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=71.11 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.10 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.08 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.09 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.09 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.30it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=71.07 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.06 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.05 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.03 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.78it/s]Capturing num tokens (num_tokens=960 avail_mem=71.06 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.78it/s] Capturing num tokens (num_tokens=896 avail_mem=71.03 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.78it/s]Capturing num tokens (num_tokens=832 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.78it/s]Capturing num tokens (num_tokens=832 avail_mem=71.04 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.26it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.26it/s]

    Capturing num tokens (num_tokens=704 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.26it/s]Capturing num tokens (num_tokens=640 avail_mem=71.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.26it/s]Capturing num tokens (num_tokens=576 avail_mem=71.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.26it/s]Capturing num tokens (num_tokens=512 avail_mem=71.00 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.26it/s]Capturing num tokens (num_tokens=512 avail_mem=71.00 GB):  50%|█████     | 29/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=480 avail_mem=71.02 GB):  50%|█████     | 29/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=448 avail_mem=71.00 GB):  50%|█████     | 29/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=416 avail_mem=70.99 GB):  50%|█████     | 29/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=384 avail_mem=71.00 GB):  50%|█████     | 29/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=352 avail_mem=70.99 GB):  50%|█████     | 29/58 [00:01<00:00, 33.56it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=320 avail_mem=70.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=288 avail_mem=70.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=256 avail_mem=70.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=240 avail_mem=70.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=224 avail_mem=70.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=224 avail_mem=70.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=208 avail_mem=70.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.40it/s]

    Capturing num tokens (num_tokens=192 avail_mem=70.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=160 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=160 avail_mem=70.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=128 avail_mem=70.88 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.90it/s]

    Capturing num tokens (num_tokens=112 avail_mem=70.40 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=96 avail_mem=70.39 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.90it/s] Capturing num tokens (num_tokens=96 avail_mem=70.39 GB):  81%|████████  | 47/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=80 avail_mem=70.28 GB):  81%|████████  | 47/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=64 avail_mem=70.23 GB):  81%|████████  | 47/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=48 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=32 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  81%|████████  | 47/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=24 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=20 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.45it/s]

    Capturing num tokens (num_tokens=16 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=12 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:02<00:00, 34.45it/s]Capturing num tokens (num_tokens=8 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:02<00:00, 34.45it/s] Capturing num tokens (num_tokens=8 avail_mem=70.20 GB):  98%|█████████▊| 57/58 [00:02<00:00, 37.71it/s]Capturing num tokens (num_tokens=4 avail_mem=70.20 GB):  98%|█████████▊| 57/58 [00:02<00:00, 37.71it/s]Capturing num tokens (num_tokens=4 avail_mem=70.20 GB): 100%|██████████| 58/58 [00:02<00:00, 28.13it/s]


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
    Generated text:  Keli and I am a graduate student in the Department of Computer Science at the University of Oregon. I have been working on developing scalable, efficient, and robust machine learning algorithms for a variety of tasks, including natural language processing, computer vision, and speech recognition. My work is focused on improving the performance and efficiency of machine learning models, and I have published several papers and papers in peer-reviewed journals. I am a passionate about learning from the latest research and fostering collaboration between researchers and practitioners in the field of computer science. As a graduate student, I have a strong desire to gain a deep understanding of the latest developments in machine learning and
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He has to do a lot of work every day. He has to be at work for a long time. He has to meet people. He has to write letters to his friends. He has to watch TV. He has to spend time with his family. He has to do other things. He likes to go out with his family. He likes to read books with his children. He likes to watch movies. He likes to listen to music. He likes to watch television. He likes to read books. He likes to sit on the couch. He likes to eat hamburgers. He likes to go to the movies.
    ===============================
    Prompt: The capital of France is
    Generated text:  a city that has been in France since 1160, and is situated on the Seine River. It is located in the Île de France, one of the 56 regions of France. The city was founded by the Vikings in the 8th century. The name "Bordeaux" comes from the French word "borde," meaning "border" or "border area."
    Bordeaux is a large city, with a population of 300,000 residents. It is located on the river Seine, and has a population density of 5,335 people per square
    ===============================
    Prompt: The future of AI is
    Generated text:  not future-proof: Could Microsoft be the next big game changer?
    
    19 Apr 2018
    
    The world is in the throes of a shake-up. We are in the midst of a significant transformation in technology that is unprecedented in the history of our civilization. We are seeing the development of a wave of supercomputers, supercomputers and supercomputers, that can do more things with less power than we have ever done before. While the ability to process data at such a scale is a major breakthrough, the development of this technology is happening too quickly to keep up with the pace of the general public.
    
    We are


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and [job title]. I'm a [job title] who is always looking for ways to [job title] and [job title]. I'm a [job title] who is always looking for ways to [job title]. I'm a [job title] who is always looking for ways to [job title]. I'm a [job title] who
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower and numerous museums and art galleries. It is also a major center for finance, business, and politics. Paris is home to the Louvre Museum, the Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its annual Eiffel Tower Festival, which attracts millions of visitors each year. Paris is a cultural and economic hub of France and a major tourist destination. It is the capital of France and the largest city in the European Union. The city is also known for its rich history, including the Roman and medieval periods, and its influence on French culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management
    


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
    Generated text:  [Your Name], and I'm a [briefly describe your profession or role]. I'm an [age] year old with a [specific interest, hobby, or professional qualification]. I've been working in [your profession or role] for [your length of employment] years now. I've got a [specific skill or expertise] that I'm very passionate about. I also enjoy [a specific activity or hobby]. How can I be reached? Is there a way to share more details about you? [Your Name] [Your Profession or Role]
    Greeting: Hello! My name is [Your Name] and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich history and diverse culture.
    
    The capital of France is Paris, which is known for its rich history and diverse culture. Its famous landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, are some of the most popular tourist attractions in the world. Paris is also known for its romantic and artistic culture, with its iconic landmarks, such as the Palace of Versailles and the Champs-Élysées, as well as its world-famous fashion and food scenes. With its rich history, diverse culture, and stunning architecture, Paris is a bustling and vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  dynamic and can be influenced by a variety of factors, including advances in computing power, the development of new technologies, the integration of new data sources, and the evolution of human perspectives on AI. Here are some possible trends in the future of AI:
    
    1. Increased focus on ethical AI: As more people become aware of the potential negative consequences of AI systems, there will be increased focus on developing ethical AI that is designed to address这些问题. This could include prioritizing safety, privacy, and accountability, as well as designing AI systems that can be designed to interact with humans in a way that is respectful and effective.
    
    2. Enhanced accuracy and


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

    Character

    's

     Name

    ],

     and

     I

     am

     a

     [

    Brief

    ly

     describe

     your

     profession

     or

     expertise

     in

     your

     chosen

     field

    .

     For

     example

    ,

     I

     am

     a

     [

    Type

     of

     Project

    ],

     [

    Type

     of

     Project

    ],

     or

     [

    Type

     of

     Project

    ].

    ],

     and

     I

     am

     passionate

     about

     [

    What

     I

     love

     to

     do

    ].

     [

    Character

    's

     Name

    ]

     is

     a

     [

    Position

     or

     Title

    ]

     who

     [

    What

     role

     you

     play

     in

     the

     story

    ].

     I

     believe

     in

     [

    Why

     you

     are

     passionate

     about

     your

     profession

     or

     expertise

    ],

     and

     I

     strive

     to

     [

    What

     you

     believe

     your

     character

     should

     strive

     to

     achieve

    ,

     like

     being

     empath

    etic

    ,

     self

    less

    ,

     or

     hard

    working

    ].

     And

     [

    Character

    's

     Name

    ]

     is

     an

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     and

     most

     populous

     city

     of

     France

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     the

     cultural

     and

     political

     center

     of

     France

    .

     The

     city

     is

     known

     for

     its

     grand

     architecture

    ,

     rich

     history

    ,

     and

     bustling

     streets

    .

     Paris

     is

     also

     famous

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     music

    .

     It

     is

     a

     major

     international

     hub

     for

     business

    ,

     finance

    ,

     and

     tourism

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     attractions

    ,

     museums

    ,

     and

     cultural

     institutions

    .

     Paris

     is

     the

     country

    's

     second

    -largest

     city

    ,

     after

     Paris

    .

     It

     is

     also

     the

     birth

    place

     of

     many

     French

     writers

    ,

     artists

    ,

     and

     intellectuals

    .

     Paris

     is

     a

     UNESCO

     World

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     several

     trends

    ,

     including

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

     will

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     Internet

     of

     Things

     (

    Io

    T

    ),

     cloud

     computing

    ,

     big

     data

    ,

     and

     machine

     learning

    .

     These

     will

     allow

     for

     more

     efficient

     and

     personalized

     interactions

     between

     machines

     and

     humans

    .
    


    2

    .

     Growth

     of

     autonomous

     and

     semi

    -aut

    onomous

     machines

    :

     With

     advancements

     in

     artificial

     intelligence

     and

     robotics

    ,

     we

     can

     expect

     autonomous

     and

     semi

    -aut

    onomous

     machines

     that

     are

     able

     to

     make

     decisions

     on

     their

     own

    .

     This

     will

     enable

     robots

     and

     drones

     to

     work

     independently

    ,

     improving

     safety

     and

     efficiency

    .
    


    3

    .

     Em

    phasis

     on

     ethical

     and

     legal

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    



```python
llm.shutdown()
```
