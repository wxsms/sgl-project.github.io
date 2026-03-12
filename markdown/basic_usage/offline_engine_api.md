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

    [2026-03-12 02:08:42] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-12 02:08:42] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-12 02:08:42] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-12 02:08:45] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.


    [2026-03-12 02:08:45] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-12 02:08:45] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=801578195, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]

    Compiling num tokens (num_tokens=4608):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=2304):  14%|█▍        | 8/58 [00:02<00:10,  4.60it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s] 

    Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=768):  28%|██▊       | 16/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:03<00:01, 19.52it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=144):  59%|█████▊    | 34/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=20):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=16):  76%|███████▌  | 44/58 [00:03<00:00, 39.60it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 51.81it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 51.81it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 51.81it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 51.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.76 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.84it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=61.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 30.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s]

    Capturing num tokens (num_tokens=960 avail_mem=61.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s] Capturing num tokens (num_tokens=896 avail_mem=61.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s]Capturing num tokens (num_tokens=832 avail_mem=61.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s]Capturing num tokens (num_tokens=768 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s]Capturing num tokens (num_tokens=704 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 37.18it/s]Capturing num tokens (num_tokens=704 avail_mem=61.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]Capturing num tokens (num_tokens=640 avail_mem=61.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]Capturing num tokens (num_tokens=576 avail_mem=61.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]Capturing num tokens (num_tokens=512 avail_mem=61.67 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]Capturing num tokens (num_tokens=480 avail_mem=61.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]Capturing num tokens (num_tokens=448 avail_mem=61.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]Capturing num tokens (num_tokens=416 avail_mem=61.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.05it/s]

    Capturing num tokens (num_tokens=416 avail_mem=61.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=384 avail_mem=61.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=352 avail_mem=61.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=320 avail_mem=61.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=288 avail_mem=61.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=256 avail_mem=61.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=240 avail_mem=61.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.22it/s]Capturing num tokens (num_tokens=240 avail_mem=61.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.39it/s]Capturing num tokens (num_tokens=224 avail_mem=61.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.39it/s]Capturing num tokens (num_tokens=208 avail_mem=61.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.39it/s]Capturing num tokens (num_tokens=192 avail_mem=61.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=176 avail_mem=61.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=160 avail_mem=61.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.39it/s]

    Capturing num tokens (num_tokens=144 avail_mem=61.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.39it/s]Capturing num tokens (num_tokens=144 avail_mem=61.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s]Capturing num tokens (num_tokens=128 avail_mem=61.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s]Capturing num tokens (num_tokens=112 avail_mem=61.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s]Capturing num tokens (num_tokens=96 avail_mem=61.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s] Capturing num tokens (num_tokens=80 avail_mem=61.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s]Capturing num tokens (num_tokens=64 avail_mem=61.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s]Capturing num tokens (num_tokens=48 avail_mem=61.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.31it/s]Capturing num tokens (num_tokens=48 avail_mem=61.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]Capturing num tokens (num_tokens=32 avail_mem=61.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]Capturing num tokens (num_tokens=28 avail_mem=61.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]Capturing num tokens (num_tokens=24 avail_mem=61.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]Capturing num tokens (num_tokens=20 avail_mem=61.61 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]

    Capturing num tokens (num_tokens=16 avail_mem=61.61 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]Capturing num tokens (num_tokens=12 avail_mem=61.61 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.23it/s]Capturing num tokens (num_tokens=12 avail_mem=61.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.71it/s]Capturing num tokens (num_tokens=8 avail_mem=61.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.71it/s] Capturing num tokens (num_tokens=4 avail_mem=61.60 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.71it/s]Capturing num tokens (num_tokens=4 avail_mem=61.60 GB): 100%|██████████| 58/58 [00:01<00:00, 42.61it/s]


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
    Generated text:  Sarah and I'm a freelancer. I specialize in writing content for various purposes and it's helping me a lot. I want to work with an affiliate program. I'm not sure which affiliate programs to choose. Could you assist me?
    
    Sure, I'd be happy to help! Let's start by exploring some basic information about affiliate programs and then we can discuss which ones would be suitable for you.
    
    ### What Are Affiliate Programs?
    Affiliate programs are programs where a business partner or affiliate sells a product or service, and the business partner earns a commission if someone else clicks on a link in the affiliate product or service. For example, if
    ===============================
    Prompt: The president of the United States is
    Generated text:  a) a federal elected official, b) a state representative, or c) a national legislator. A) a) President of the United States is a federal elected official. The President of the United States is elected by the people of the United States through a process of voting and drawing lots to determine their positions. This ensures that the President has the full authority of the United States government and represents the interests of all citizens. Therefore, the correct answer is B) a state representative. However, it's important to note that the President of the United States is not a federal elected official. The process for electing the President involves state
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the banks of the Seine River, in the Moselle Hills. The city was founded in the 6th century, during the Roman Empire. It was originally built as the capital of the Kingdom of France, until it was moved to Paris in 1960.
    So, how did Paris get its name? One of the most interesting stories is how a French traveler from the 13th century first used the name "Paris".
    The local inhabitants of a nearby village, who were protecting a large wild boar (which is very common in the area), discovered that they had a deer that
    ===============================
    Prompt: The future of AI is
    Generated text:  with us. You’ve probably seen the hype about it. The technology is advancing rapidly, and many companies are using it in an array of applications from smart city systems to personal assistants. But what does that mean for the future? That’s where we get in a discussion of the challenges, opportunities and potential of the field of AI.
    What is AI? In simple terms, AI is an artificial intelligence. It’s a computer program that mimics human intelligence and functions like a human. In this context, the term AI is used to describe an algorithm or system that makes decisions based on patterns, examples and data. The AI system can process


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, a historic district known for its French colonial architecture. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a city of contrasts, with its modern and historic aspects blending together to create a unique and fascinating city. The city is also home to many international organizations and events, making it a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is becoming more and more integrated into various industries, from manufacturing to healthcare to customer service. As AI becomes more capable of performing tasks that were previously done by humans, it is likely to automate more and more of these tasks, leading to increased efficiency and productivity.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. There will be a need for regulations and guidelines to ensure that AI is used ethically and that it does not harm individuals or the environment.
    
    3. AI in healthcare
    


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
    Generated text:  [Your Name], and I'm a [Your Profession] with over [Your Number] of years of experience in [Your Industry], specializing in [Your Speciality]. I'm an [Your Age], [Your Nationality], and I currently live and work in [Your Location]. I am a true believer in [Your Core Belief], and I strive to be a [Your Ideal Character]. In my free time, I enjoy [Your Passion] with friends and family. My [Your Favorite Genre] is [Your Favorite Genre], and I love [Your Favorite Hobby], [Your Favorite Sport], [Your Favorite Book]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the world-famous city, home to numerous historical landmarks such as Notre Dame Cathedral, the Louvre Museum, and the Arc de Triomphe. It is a popular tourist destination known for its rich culture, cuisine, and world-renowned fashion industry. Paris is also a significant economic hub and the seat of the French government, making it one of the most important cities in the world. It is also the birthplace of French literature and a cultural and artistic hub, known for its museums, galleries, and festivals. The city has a population of over one million and is a vibrant and bustling metropolis. Paris is often referred
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some potential trends that are likely to shape it:
    
    1. Increased AI integration with human society: As AI technology advances, we are likely to see more integration of AI with human society, such as through more sophisticated voice assistants, facial recognition systems, and machine learning algorithms.
    
    2. AI becomes more pervasive: AI will become more widespread across different industries and applications, from healthcare to finance to transportation, as more and more businesses and organizations incorporate AI into their operations.
    
    3. AI will become more autonomous: AI systems will become more capable of making decisions and taking actions without human intervention, leading to a more autonomous and


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

    occupation

    ]

     with

     [

    number

    ]

     years

     of

     experience

     in

     the

     industry

    .

     Throughout

     my

     career

    ,

     I

    've

     worked

     for

     [

    company

     name

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    career

     goal

    ].

     I

    'm

     always

     interested

     in

     learning

     new

     things

     and

     exploring

     new

     opportunities

    .

     I

    'm

     dedicated

     to

     [

    reason

     for

     career

     dedication

    ]

     and

     strive

     to

     be

     a

     [

    role

     model

    ]

     for

     those

     I

     mentor

    .

     I

    'm

     always

     open

     to

     feedback

     and

     I

    'm

     passionate

     about

     [

    purpose

     of

     my

     work

    ].

     I

    'm

     confident

     that

     I

     can

     make

     a

     real

     difference

     in

     the

     world

    .

     Thank

     you

     for

     considering

     my

     application

    .

     How

     can

     I

     best

     start

     the

     conversation

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     No

    ire

    "

     which

     means

     "

    The

     Dark

     City

    "

     due

     to

     its

     dim

     lighting

     and

     dark

     alle

    ys

    .

     The

     city

     is

     a

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    ,

     hosting

     world

    -f

    amous

     landmarks

     and

     attractions

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Despite

     its

     towering

     architecture

     and

     historic

     importance

    ,

     Paris

     is

     known

     for

     its

     diverse

     cultural

     scene

     and

     vibrant

     nightlife

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     museums

    ,

     including

     the

     Mus

    ée

     du

     Lou

    vre

     and

     the

     Mus

    ée

     Rod

    in

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

    ,

     with

     many

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     an

     exciting

     and

     rapidly

     evolving

     field

     with

     many

     potential

     directions

     and

     applications

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

     integration

     of

     AI

     into

     human

     decision

    -making

    :

     As

     AI

     becomes

     more

     advanced

     and

     integrated

     into

     various

     industries

    ,

     we

     can

     expect

     to

     see

     increased

     integration

     with

     human

     decision

    -making

    .

     AI

     systems

     will

     be

     able

     to

     analyze

     large

     amounts

     of

     data

     to

     make

     informed

     decisions

    ,

     and

     humans

     will

     be

     able

     to

     provide

     guidance

     and

     feedback

     to

     help

     AI

     improve

     its

     performance

    .
    


    2

    .

     Enhanced

     abilities

     in

     automation

    :

     AI

     is

     becoming

     increasingly

     powerful

     and

     capable

     of

     performing

     complex

     tasks

     that

     require

     human

     skills

    .

     This

     automation

     will

     likely

     lead

     to

     more

     efficient

     and

     cost

    -effective

     operations

    ,

     but

     it

    



```python
llm.shutdown()
```
