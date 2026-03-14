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

    [2026-03-14 20:10:36] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-14 20:10:36] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-14 20:10:36] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-14 20:10:38] INFO server_args.py:2146: Attention backend not specified. Use fa3 backend by default.


    [2026-03-14 20:10:38] INFO server_args.py:3287: Set soft_watchdog_timeout since in CI


    [2026-03-14 20:10:38] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=394801347, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.61it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.61it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.18s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:06,  1.18s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.07it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:26,  2.07it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.56it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.56it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.19it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.19it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.21it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.21it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.21it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.21it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.36it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.36it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.36it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.36it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 13.36it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 17.49it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 17.49it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 22.80it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 22.80it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 22.80it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 22.80it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 22.80it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 26.27it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 26.27it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 26.27it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 26.27it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 26.27it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 26.27it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 30.92it/s]

    Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 37.29it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 37.29it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 37.29it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 37.29it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 37.29it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 37.29it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 38.44it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 38.44it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 38.44it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 38.44it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 38.44it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 38.44it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 40.77it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 40.77it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 40.77it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 40.77it/s] 

    Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 40.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.43 GB):   2%|▏         | 1/58 [00:00<00:08,  6.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:08,  6.83it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:08,  6.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.40 GB):   3%|▎         | 2/58 [00:00<00:08,  6.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.40 GB):   5%|▌         | 3/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.40 GB):   5%|▌         | 3/58 [00:00<00:07,  7.28it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.40 GB):   7%|▋         | 4/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.40 GB):   7%|▋         | 4/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:06,  7.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:06,  7.70it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.39 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.06it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.38 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.61it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.37 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.76it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.56it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.33 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.33 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.73it/s]Capturing num tokens (num_tokens=960 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.73it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.73it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 11.92it/s]Capturing num tokens (num_tokens=832 avail_mem=58.34 GB):  40%|███▉      | 23/58 [00:02<00:02, 11.92it/s]Capturing num tokens (num_tokens=768 avail_mem=58.34 GB):  40%|███▉      | 23/58 [00:02<00:02, 11.92it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.07it/s]Capturing num tokens (num_tokens=704 avail_mem=58.33 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.07it/s]Capturing num tokens (num_tokens=640 avail_mem=58.33 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.07it/s]Capturing num tokens (num_tokens=576 avail_mem=58.33 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.07it/s]Capturing num tokens (num_tokens=576 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.39it/s]Capturing num tokens (num_tokens=512 avail_mem=58.30 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.39it/s]Capturing num tokens (num_tokens=480 avail_mem=58.32 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.39it/s]

    Capturing num tokens (num_tokens=480 avail_mem=58.32 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.84it/s]Capturing num tokens (num_tokens=448 avail_mem=58.31 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.84it/s]Capturing num tokens (num_tokens=416 avail_mem=58.31 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.84it/s]Capturing num tokens (num_tokens=416 avail_mem=58.31 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.26it/s]Capturing num tokens (num_tokens=384 avail_mem=58.31 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.26it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.30 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.26it/s]Capturing num tokens (num_tokens=352 avail_mem=58.30 GB):  59%|█████▊    | 34/58 [00:03<00:01, 13.74it/s]Capturing num tokens (num_tokens=320 avail_mem=58.30 GB):  59%|█████▊    | 34/58 [00:03<00:01, 13.74it/s]Capturing num tokens (num_tokens=288 avail_mem=58.30 GB):  59%|█████▊    | 34/58 [00:03<00:01, 13.74it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.30 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.25it/s]Capturing num tokens (num_tokens=256 avail_mem=58.29 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.25it/s]Capturing num tokens (num_tokens=240 avail_mem=58.29 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.25it/s]Capturing num tokens (num_tokens=240 avail_mem=58.29 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.86it/s]Capturing num tokens (num_tokens=224 avail_mem=58.29 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.86it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.28 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.86it/s]Capturing num tokens (num_tokens=208 avail_mem=58.28 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.72it/s]Capturing num tokens (num_tokens=192 avail_mem=58.28 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.72it/s]Capturing num tokens (num_tokens=176 avail_mem=58.28 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.72it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.28 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.57it/s]Capturing num tokens (num_tokens=160 avail_mem=58.28 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.57it/s]Capturing num tokens (num_tokens=144 avail_mem=58.27 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.57it/s]Capturing num tokens (num_tokens=144 avail_mem=58.27 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.52it/s]Capturing num tokens (num_tokens=128 avail_mem=58.27 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.52it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.27 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.52it/s]Capturing num tokens (num_tokens=112 avail_mem=58.27 GB):  79%|███████▉  | 46/58 [00:03<00:00, 12.36it/s]Capturing num tokens (num_tokens=96 avail_mem=58.26 GB):  79%|███████▉  | 46/58 [00:03<00:00, 12.36it/s] Capturing num tokens (num_tokens=80 avail_mem=58.26 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.36it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.26 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.39it/s]Capturing num tokens (num_tokens=64 avail_mem=58.26 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.39it/s]Capturing num tokens (num_tokens=48 avail_mem=58.25 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.39it/s]Capturing num tokens (num_tokens=48 avail_mem=58.25 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.37it/s]Capturing num tokens (num_tokens=32 avail_mem=58.25 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.37it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.24 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.37it/s]Capturing num tokens (num_tokens=28 avail_mem=58.24 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.34it/s]Capturing num tokens (num_tokens=24 avail_mem=58.24 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.34it/s]Capturing num tokens (num_tokens=20 avail_mem=58.24 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.34it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.24 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.88it/s]Capturing num tokens (num_tokens=16 avail_mem=58.24 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.88it/s]Capturing num tokens (num_tokens=12 avail_mem=58.23 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.88it/s]Capturing num tokens (num_tokens=8 avail_mem=58.23 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.88it/s] Capturing num tokens (num_tokens=8 avail_mem=58.23 GB):  98%|█████████▊| 57/58 [00:04<00:00, 16.04it/s]Capturing num tokens (num_tokens=4 avail_mem=58.23 GB):  98%|█████████▊| 57/58 [00:04<00:00, 16.04it/s]Capturing num tokens (num_tokens=4 avail_mem=58.23 GB): 100%|██████████| 58/58 [00:04<00:00, 12.12it/s]


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
    Generated text:  Daniel and I am a visual artist. I have a passion for creating artwork that is inspired by the natural world and serves as a way to explore a sense of wonder and awe in an expressive manner. I have a degree in Art, Anthropology, and Archaeology from the University of Utah and have been teaching art classes at the University of Utah since 2015. I live in Salt Lake City and have spent my entire life exploring the natural world, and believe that art is a means to communicate and connect with the world around us.
    I have been making art since I was a child and have found my style to be a
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the vice president. In how many ways can the vice president be chosen if the vice president cannot be one of the original 10 candidates: John, Mary, and Ann. In how many ways can the vice president be chosen from the original 10 candidates, if the vice president cannot be one of the original 10 candidates?
    
    To determine the number of ways the vice president can be chosen from the original 10 candidates if the vice president cannot be one of the original 10 candidates, we need to follow these steps:
    
    1. Identify the original 10 candidates.
    2. Identify the 1
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the left bank of the Seine river. The city is surrounded by the hills of the Alps and the Pyrenees mountains.
    
    The oldest city in France, Paris is situated in the central part of the country. It is surrounded by the hills of the Seine River.
    
    The city was founded by the Romans in 733 B.C. and ruled by the kings of the Franks. Paris has been ruled by many kings and emperors over the centuries. From 1830 to 1914, it was the capital of France. In 1914,
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it is changing the way we do business. As a result, it is important for companies to develop a plan to ensure that their AI systems are secure and secure.
    Here are some of the most important factors to consider when developing an AI security plan:
    
      1. Identify the Threats: The first step in developing a security plan for AI is to identify the threats that it may encounter. These threats can include malicious actors, human error, and the security vulnerabilities inherent in the AI systems themselves. The first step in understanding these threats is to conduct a risk assessment, which will help you identify the potential vulnerabilities and weaknesses in


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


    Generated text:  Paris, also known as "La Ville-Marie" and "La Ville-Est". It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also known for its rich cultural heritage and is a major tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a city of people and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced interactions between the two.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased focus on privacy and security, with more stringent regulations and standards being put in place to protect user data.
    
    3. Greater automation and efficiency: AI is likely to become more automated and efficient, with more systems being able to perform tasks that were previously done by humans.
    
    4. Increased focus on ethical considerations: As AI systems
    


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
    Generated text:  [Your Name], and I'm [Your Age]. I'm an enthusiastic book lover and a passionate writer. I specialize in crafting original short stories and essays, and I love to explore new genres and explore new ideas. I've always been drawn to the thrill of the unknown and I love discovering new worlds and characters. I'm always looking for a fresh idea or a way to expand my knowledge and keep my mind active. If you're an aspiring writer or someone who loves storytelling, I can help you on your journey to becoming a great storyteller. I'm here to help you grow into a storyteller that can captivate readers,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Please answer the following question based on the information given:
    
    Where is the capital of France located? Paris is located in France. The capital of France is Paris. 
    
    To provide a more detailed response:
    
    1. Paris is the capital city of France, located in the east of the country.
    2. It is situated on the River Seine and the Mediterranean Sea.
    3. The city is surrounded by the Loire Valley and the Pyrénées mountains to the west and north.
    4. The city has a population of around 2 million and is the largest city in France.
    5. It is known for its rich history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be highly dynamic and constantly evolving, driven by the rapid advancements in computing power, data, and other technologies. Here are some possible trends in AI that are currently being explored and developed:
    
    1. Increased Robustness and Accuracy: One of the biggest challenges in AI is developing algorithms that can accurately reason, learn, and make decisions. As research continues, we can expect to see an increase in the use of machine learning algorithms that can improve the accuracy and robustness of AI systems, making them better able to handle a wider range of tasks and in a variety of contexts.
    
    2. Integration with Human Decision-making: One of the


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

    type

     of

     character

    ,

     e

    .g

    .,

     writer

    ,

     gamer

    ,

     athlete

    ,

     etc

    .]

     with

     an

     interest

     in

     [

    specific

     topic

    ,

     e

    .g

    .,

     technology

    ,

     politics

    ,

     etc

    .

    ].

     I

     enjoy

     exploring

     [

    interest

    ],

     and

     [

    related

     details

    ,

     e

    .g

    .,

     hobbies

    ,

     passions

    ,

     etc

    .

    ].

     I

     also

     like

     [

    related

     interests

    ,

     e

    .g

    .,

     travel

    ,

     hobbies

    ,

     etc

    .

    ].

     I

    'm

     confident

     in

     [

    interest

    s

     and

     passions

    ]

     and

     [

    related

     skills

     or

     abilities

    ],

     and

     I

    'm

     eager

     to

     [

    a

     goal

     or

     purpose

    ,

     e

    .g

    .,

     pursue

     my

     dream

    ,

     help

     someone

    ,

     learn

     something

     new

    ,

     etc

    .

    ].

     I

     love

     [

    related

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

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

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     population

    ,

     with

     a

     large

     Jewish

     community

     and

     a

     vibrant

     arts

     and

     culture

     scene

    .

     The

     city

     is

     famous

     for

     its

     annual

     May

     Day

     parade

     and

     its

     role

     as

     the

     world

    's

     oldest

     continuously

     operating

     public

     school

    .

     Paris

     is

     also

     a

     major

     economic

     hub

     and

     is

     home

     to

     many

     of

     France

    's

     most

     prestigious

     universities

     and

     research

     institutions

    .

     Its

     status

     as

     a

     cultural

     and

     educational

     center

     has

     helped

     make

     it

     a

     popular

     destination

     for

     international

     tourists

    .

     The

     French

     capital

     has

     a

     rich

     history

    ,

     and

     its

     many

     landmarks

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promising

    ,

     with

     many

     potential

     applications

     and

     developments

     to

     explore

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

     AI

     Ethics

    :

     As

     AI

     systems

     become

     more

     integrated

     into

     our

     lives

    ,

     there

     will

     be

     a

     growing

     awareness

     of

     ethical

     and

     moral

     issues

     surrounding

     AI

    .

     Developers

     will

     need

     to

     incorporate

     more

     ethical

     considerations

     into

     their

     systems

     and

     ensure

     that

     they

     are

     designed

     in

     a

     way

     that

     promotes

     fairness

    ,

     transparency

    ,

     and

     accountability

    .
    


    2

    .

     Increased

     AI

     Integration

    :

     AI

     is

     already

     playing

     an

     increasingly

     important

     role

     in

     our

     daily

     lives

    ,

     from

     self

    -driving

     cars

     to

     personalized

     search

     results

     on

     search

     engines

    .

     As

     AI

     continues

     to

     improve

    ,

     it

     is

     likely

     that

     we

     will

     see

     even

     more

    



```python
llm.shutdown()
```
