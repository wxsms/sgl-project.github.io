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

    [2026-03-15 21:33:37] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-15 21:33:37] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-15 21:33:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-15 21:33:39] INFO server_args.py:2156: Attention backend not specified. Use fa3 backend by default.


    [2026-03-15 21:33:39] INFO server_args.py:3297: Set soft_watchdog_timeout since in CI


    [2026-03-15 21:33:39] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=363250807, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.86it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 14.08it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 14.08it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 14.08it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 14.08it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 14.08it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 14.08it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 14.08it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 14.08it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 14.08it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 14.08it/s]

    Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:03<00:02, 14.08it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 23.40it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:03<00:00, 32.38it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 43.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.43 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.43 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.42 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.42 GB):   7%|▋         | 4/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.43 GB):   7%|▋         | 4/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.42 GB):   7%|▋         | 4/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.42 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.42 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.42 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.42 GB):  10%|█         | 6/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.42 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.41 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.40 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.39 GB):  21%|██        | 12/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.37 GB):  21%|██        | 12/58 [00:00<00:02, 21.79it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.36 GB):  21%|██        | 12/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.34 GB):  21%|██        | 12/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.34 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.32 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.29 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.88it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=56.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=960 avail_mem=55.77 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.88it/s] Capturing num tokens (num_tokens=960 avail_mem=55.77 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.75it/s]Capturing num tokens (num_tokens=896 avail_mem=55.76 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.75it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.75 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.75it/s]Capturing num tokens (num_tokens=768 avail_mem=55.75 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.75it/s]Capturing num tokens (num_tokens=768 avail_mem=55.75 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.31it/s]Capturing num tokens (num_tokens=704 avail_mem=55.75 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.31it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.74 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.31it/s]Capturing num tokens (num_tokens=576 avail_mem=55.74 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.31it/s]Capturing num tokens (num_tokens=576 avail_mem=55.74 GB):  48%|████▊     | 28/58 [00:01<00:01, 16.09it/s]Capturing num tokens (num_tokens=512 avail_mem=55.73 GB):  48%|████▊     | 28/58 [00:01<00:01, 16.09it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.74 GB):  48%|████▊     | 28/58 [00:01<00:01, 16.09it/s]Capturing num tokens (num_tokens=480 avail_mem=55.74 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.20it/s]Capturing num tokens (num_tokens=448 avail_mem=55.74 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.20it/s]Capturing num tokens (num_tokens=416 avail_mem=55.74 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.20it/s]

    Capturing num tokens (num_tokens=416 avail_mem=55.74 GB):  55%|█████▌    | 32/58 [00:01<00:01, 14.54it/s]Capturing num tokens (num_tokens=384 avail_mem=55.73 GB):  55%|█████▌    | 32/58 [00:01<00:01, 14.54it/s]Capturing num tokens (num_tokens=352 avail_mem=55.73 GB):  55%|█████▌    | 32/58 [00:01<00:01, 14.54it/s]Capturing num tokens (num_tokens=352 avail_mem=55.73 GB):  59%|█████▊    | 34/58 [00:01<00:01, 13.87it/s]Capturing num tokens (num_tokens=320 avail_mem=55.72 GB):  59%|█████▊    | 34/58 [00:01<00:01, 13.87it/s]

    Capturing num tokens (num_tokens=288 avail_mem=55.72 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.87it/s]Capturing num tokens (num_tokens=288 avail_mem=55.72 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.45it/s]Capturing num tokens (num_tokens=256 avail_mem=55.72 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.45it/s]Capturing num tokens (num_tokens=240 avail_mem=55.71 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.45it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.71 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.22it/s]Capturing num tokens (num_tokens=224 avail_mem=55.71 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.22it/s]Capturing num tokens (num_tokens=208 avail_mem=55.70 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.22it/s]Capturing num tokens (num_tokens=208 avail_mem=55.70 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.11it/s]Capturing num tokens (num_tokens=192 avail_mem=55.70 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.70 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.11it/s]Capturing num tokens (num_tokens=176 avail_mem=55.70 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.01it/s]Capturing num tokens (num_tokens=160 avail_mem=55.70 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.01it/s]Capturing num tokens (num_tokens=144 avail_mem=55.69 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.01it/s]

    Capturing num tokens (num_tokens=144 avail_mem=55.69 GB):  76%|███████▌  | 44/58 [00:02<00:01, 12.92it/s]Capturing num tokens (num_tokens=128 avail_mem=55.69 GB):  76%|███████▌  | 44/58 [00:02<00:01, 12.92it/s]Capturing num tokens (num_tokens=112 avail_mem=55.69 GB):  76%|███████▌  | 44/58 [00:02<00:01, 12.92it/s]Capturing num tokens (num_tokens=112 avail_mem=55.69 GB):  79%|███████▉  | 46/58 [00:02<00:00, 12.80it/s]Capturing num tokens (num_tokens=96 avail_mem=55.69 GB):  79%|███████▉  | 46/58 [00:02<00:00, 12.80it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=55.68 GB):  79%|███████▉  | 46/58 [00:02<00:00, 12.80it/s]Capturing num tokens (num_tokens=80 avail_mem=55.68 GB):  83%|████████▎ | 48/58 [00:03<00:00, 12.60it/s]Capturing num tokens (num_tokens=64 avail_mem=55.68 GB):  83%|████████▎ | 48/58 [00:03<00:00, 12.60it/s]Capturing num tokens (num_tokens=48 avail_mem=55.68 GB):  83%|████████▎ | 48/58 [00:03<00:00, 12.60it/s]

    Capturing num tokens (num_tokens=48 avail_mem=55.68 GB):  86%|████████▌ | 50/58 [00:03<00:00, 12.47it/s]Capturing num tokens (num_tokens=32 avail_mem=55.67 GB):  86%|████████▌ | 50/58 [00:03<00:00, 12.47it/s]Capturing num tokens (num_tokens=28 avail_mem=55.67 GB):  86%|████████▌ | 50/58 [00:03<00:00, 12.47it/s]Capturing num tokens (num_tokens=28 avail_mem=55.67 GB):  90%|████████▉ | 52/58 [00:03<00:00, 12.39it/s]Capturing num tokens (num_tokens=24 avail_mem=55.66 GB):  90%|████████▉ | 52/58 [00:03<00:00, 12.39it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.66 GB):  90%|████████▉ | 52/58 [00:03<00:00, 12.39it/s]Capturing num tokens (num_tokens=20 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:03<00:00, 12.57it/s]Capturing num tokens (num_tokens=16 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:03<00:00, 12.57it/s]Capturing num tokens (num_tokens=12 avail_mem=55.65 GB):  93%|█████████▎| 54/58 [00:03<00:00, 12.57it/s]

    Capturing num tokens (num_tokens=12 avail_mem=55.65 GB):  97%|█████████▋| 56/58 [00:03<00:00, 12.59it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 12.59it/s] Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  97%|█████████▋| 56/58 [00:03<00:00, 12.59it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 10.38it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 14.52it/s]


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
    Generated text:  Kai and I am a teacher at a community college in California. I teach at a community college, and I have been teaching for five years. What is the first thing that you would tell a new student at the college? I would tell a new student to be prepared to succeed. I would encourage my students to be open-minded, curious, and respectful of others. They must also be ready to ask questions and work hard to improve themselves. Ultimately, I would emphasize that education is a lifelong pursuit, and that success is built on the foundation of hard work, dedication, and self-improvement.
    What are some ways that you have
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president is in charge of the government of the United States. He or she has the power to make all important decisions. He or she can do things that make the people happy or upset. The president is elected by the people and that's the reason why the country is governed by him or her. 
    
    The United States is one of the oldest countries in the world. It has been in existence for over 200 years. The country is so important that it has its own capital city. Its capital city is called Washington D.C. Washington D.C. is where the president lives. 
    
    The president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the most populous city in the world, the country’s most important financial center, and the home of the Eiffel Tower. It is also the birthplace of the French Revolution and the King of the French, the Emperor of the French, the President of the French Republic, the President of the French Congress, and the President of the French Senate. Auberge La Défense is a historic cafe and restaurant located in Paris. It is also one of the few remaining historic cafes in Paris. The café is located on Rue de Fleurus, just one block from the Arc de Triomphe.
    It
    ===============================
    Prompt: The future of AI is
    Generated text:  dependent on how it is designed. This means that the field of AI is still very much in its infancy. This is because the human brain is more than a complex system that can be designed or programmed. It has a variety of features that enable it to process a huge amount of information in the time it takes a human to think. Many neuroscientists have developed theories about the basis of the brain, and the brain's ability to generate the complex functions of the human mind and body. These theories can help us understand how we process information, and how we use it to make decisions.
    If we can understand the brain and its role in


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm excited to learn and grow with you. What's your name? What's your job title? What's your company name? What's your industry? How many years of experience do you have? What's your job title? What's your company name? What's your industry? How many years of experience do you have?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known fact that Paris is the capital city of France, and this statement accurately captures that fact. 
    
    However, it's worth noting that the statement could be more precise if it included the city's other notable landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which are all located in Paris. Additionally, the statement could be expanded to include the city's role as a major economic and cultural center, as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we are likely to see
    


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
    Generated text:  [Your Name], and I am a [job title] at [Company Name]. I've been with the company for [number of years] years, and I've had the privilege of working on some of the most challenging projects, from [mention one specific project] to [mention another specific project]. I love spending time with my family, reading books, and volunteering in my community. What brings you to this company? I'm always looking for new challenges and opportunities for growth, and I'm excited to learn and grow with you. Have a great day! [Your Name] With gratitude [Your Name] [Company Name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its most famous landmarks include the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. French cuisine, including croissants, pastries, and baguettes, is renowned throughout the world. Paris is also home to many historical and cultural landmarks such as the Louvre Museum, the Notre Dame Cathedral, and the Arc de Triomphe. The city is known for its fashion, art, and music scene, attracting visitors from around the world with its vibrant culture and annual fashion week. Paris is an important city in French history, with its role as a major port and economic center. It's also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements in many areas, including:
    
    1. Enhanced computer hardware: Moore's Law is set to continue, allowing for faster processing speeds and more powerful processors. This will lead to a greater capacity for AI to process and analyze data.
    
    2. AI in healthcare: AI is already being used to diagnose diseases, develop personalized treatment plans, and improve patient outcomes. As AI becomes more sophisticated, it may be used to develop new medical technologies and treatments.
    
    3. AI in manufacturing: AI is already being used to optimize production processes, reduce costs, and improve product quality. As AI technology continues to improve, it may


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

    insert name

    ],

     and

     I

    'm

     a

     dedicated

     reader

     and

     writer

     of

     [

    insert

     genre

    ,

     such

     as

     science

     fiction

    ,

     romance

    ,

     or

     mystery

    ].

     I

     love

     to

     explore

     the

     world

     of

     technology

    ,

     love

     to

     explore

     the

     mysteries

     of

     the

     human

     mind

    ,

     and

     love

     to

     learn

     new

     things

     every

     day

    .

     I

    'm

     a

     passionate

     and

     enthusiastic

     writer

    ,

     always

     looking

     for

     fresh

     perspectives

     and

     fresh

     ideas

     to

     inspire

     my

     writing

    .


    My

     favorite

     book

     series

     is

     the

     "

    Ali

    ens

    "

     trilogy

     by

     [

    insert

     author

    's

     name

    ],

     and

     I

     love

     how

     the

     series

     has

     a

     unique

     style

     and

     engaging

     storyline

    .

     I

    'm

     a

     fan

     of

     the

     genre

    ,

     and

     I

    'm

     always

     eager

     to

     dive

     into

     new

     stories

     and

     explore

    
    
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

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     French

     cuisine

     is

     also

     a

     major

     draw

    ,

     and

     Paris

     is

     a

     popular

     tourist

     destination

    .

     It

     is

     also

     a

     cultural

     hub

     with

     many

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    .

     The

     city

     is

     known

     for

     its

     annual

     M

    ardi

     Gr

    as

    ,

     a

     vibrant

     celebration

     of

     traditional

     Louisiana

     culture

    .

     Paris

     is

     a

     hub

     of

     intellectual

     and

     artistic

     activity

    ,

     attracting

     many

     renowned

     artists

    ,

     writers

    ,

     and

     philosophers

    .

     The

     city

     is

     also

     home

     to

     the

     French

     Riv

    iera

    ,

     known

     for

     its

     beautiful

     beaches

     and

     Mediterranean

     cuisine

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     contrasts

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     the

     rapid

     development

     of

     new

     technologies

    ,

     the

     increasing

     complexity

     of

     data

    ,

     and

     the

     need

     for

     more

     advanced

     algorithms

     and

     models

     to

     extract

     meaningful

     insights

     from

     large

     amounts

     of

     data

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     More

     advanced

     machine

     learning

    :

     With

     the

     increasing

     availability

     of

     data

     and

     the

     complexity

     of

     the

     problem

    ,

     machine

     learning

     models

     are

     likely

     to

     become

     even

     more

     sophisticated

    .

     AI

     models

     that

     are

     able

     to

     learn

     from

     large

     amounts

     of

     data

     and

     adapt

     to

     new

     situations

     will

     become

     more

     common

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     With

     the

     growth

     of

     the

     internet

     and

     social

     media

    ,

     natural

     language

     processing

     (

    N

    LP

    )

     will

     become

     increasingly

     important

    .

    



```python
llm.shutdown()
```
