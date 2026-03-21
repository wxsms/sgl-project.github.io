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

    [2026-03-21 08:04:50] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-21 08:04:50] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-21 08:04:50] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-21 08:04:53] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-21 08:04:53] INFO server_args.py:2232: Attention backend not specified. Use fa3 backend by default.


    [2026-03-21 08:04:53] INFO server_args.py:3506: Set soft_watchdog_timeout since in CI


    [2026-03-21 08:04:53] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=409049167, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]

    Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:05<00:02, 11.74it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 17.09it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.85it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 27.24it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=136.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.04 GB):   3%|▎         | 2/58 [00:00<00:03, 14.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=136.04 GB):   3%|▎         | 2/58 [00:00<00:03, 14.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=136.04 GB):   3%|▎         | 2/58 [00:00<00:03, 14.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=136.04 GB):   7%|▋         | 4/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=136.04 GB):   7%|▋         | 4/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=136.03 GB):   7%|▋         | 4/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=136.04 GB):   7%|▋         | 4/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=136.04 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=136.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=136.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.60it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=136.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.60it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=136.03 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=136.02 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=136.02 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=136.02 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=136.02 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.01 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.01 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=136.01 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=136.01 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=136.00 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=136.00 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.55it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=135.99 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=135.99 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=135.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.68it/s]Capturing num tokens (num_tokens=960 avail_mem=135.99 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.68it/s] Capturing num tokens (num_tokens=896 avail_mem=135.99 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.68it/s]Capturing num tokens (num_tokens=832 avail_mem=135.98 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.68it/s]Capturing num tokens (num_tokens=768 avail_mem=135.98 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.68it/s]Capturing num tokens (num_tokens=768 avail_mem=135.98 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.75it/s]Capturing num tokens (num_tokens=704 avail_mem=134.93 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.75it/s]Capturing num tokens (num_tokens=640 avail_mem=134.90 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.75it/s]Capturing num tokens (num_tokens=576 avail_mem=134.90 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.75it/s]Capturing num tokens (num_tokens=512 avail_mem=134.83 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.75it/s]

    Capturing num tokens (num_tokens=480 avail_mem=127.47 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.75it/s]Capturing num tokens (num_tokens=480 avail_mem=127.47 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.30it/s]Capturing num tokens (num_tokens=448 avail_mem=121.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.30it/s]Capturing num tokens (num_tokens=416 avail_mem=121.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.30it/s]Capturing num tokens (num_tokens=384 avail_mem=121.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.30it/s]Capturing num tokens (num_tokens=352 avail_mem=121.30 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.30it/s]Capturing num tokens (num_tokens=320 avail_mem=120.65 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.30it/s]Capturing num tokens (num_tokens=320 avail_mem=120.65 GB):  60%|██████    | 35/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=288 avail_mem=120.64 GB):  60%|██████    | 35/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=256 avail_mem=120.64 GB):  60%|██████    | 35/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=240 avail_mem=120.64 GB):  60%|██████    | 35/58 [00:01<00:00, 37.05it/s]

    Capturing num tokens (num_tokens=224 avail_mem=120.64 GB):  60%|██████    | 35/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=208 avail_mem=120.63 GB):  60%|██████    | 35/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=208 avail_mem=120.63 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=192 avail_mem=120.63 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=176 avail_mem=120.63 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=160 avail_mem=120.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=144 avail_mem=120.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=128 avail_mem=120.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=128 avail_mem=120.62 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=112 avail_mem=120.62 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=96 avail_mem=120.61 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.10it/s] Capturing num tokens (num_tokens=80 avail_mem=120.61 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.10it/s]

    Capturing num tokens (num_tokens=64 avail_mem=120.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=48 avail_mem=120.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=48 avail_mem=120.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=32 avail_mem=120.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=28 avail_mem=120.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=24 avail_mem=120.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=20 avail_mem=120.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=16 avail_mem=120.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=16 avail_mem=120.59 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=12 avail_mem=120.58 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=8 avail_mem=120.58 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.07it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=120.58 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=4 avail_mem=120.58 GB): 100%|██████████| 58/58 [00:01<00:00, 32.09it/s]


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
    Generated text:  Andromeda, I'm a healthy, intelligent, and enthusiastic child. I have many talents and interests. I can sing, dance, play guitar, make movies, and even talk and think like an adult. I'm trying to teach myself advanced math and logic problems to improve my problem-solving skills. I also have a deep love for literature and have read many books on ancient civilizations and mythical creatures. I'm currently trying to learn and understand the principles of telekinesis, as I find it intriguing and awe-inspiring. Can you help me improve my problem-solving skills? Andromeda, what advice would you give to someone trying
    ===============================
    Prompt: The president of the United States is
    Generated text:  a highly elected and effective position that is vital to the political, economic, and social well-being of a country. At the same time, the position is not without controversy and challenges.
    The American president is a elected office that is filled through a competitive system. The president is nominated by the Democratic and Republican parties, and each party ultimately selects the candidate to run. In order to become a president, a person must be a natural-born citizen of the United States, have been 35 years of age and be over the age of 30 for 5 years and have a minimum of 10 years of military service. The
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) Berlin
    
    C) London
    
    D) Rome
    
    E) Moscow
    
    To determine the capital of France, we need to remember the official capital of France. France is a country, so its capital is a specific city that is widely recognized as the official seat of government for the country.
    
    Let's list the capital cities of different countries:
    - The capital of Germany is Berlin.
    - The capital of Italy is Rome.
    - The capital of Russia is Moscow.
    - The capital of Belgium is Brussels.
    - The capital of France is Paris.
    
    Based on this information, we can conclude that the capital of France is
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright. Our bodies will be controlled by artificial intelligence in the future. What are the key technologies that we will use to control our bodies? If artificial intelligence is not only used for controlling the body, but also for human interaction, such as communication, speech, language, etc., will there be some problems? So what are the major challenges of using AI to control human interaction? I would like to have a detailed analysis on this topic. Please discuss the technical, social, and ethical issues that need to be addressed by the future of AI. Finally, how can we ensure the safety and privacy of the human body while using AI to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to improve my skills. I'm always eager to learn and grow, and I'm always looking for new challenges and opportunities to grow. I'm a [job title] at [company name], and I'm always looking for ways to improve my skills. I'm always eager to learn and grow, and I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French Academy of Sciences. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to the French Parliament and the French Academy of Sciences. Paris is a major tourist destination and is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural hub and a major tourist
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate a wide range of tasks, from manufacturing and transportation to customer service and healthcare. This will lead to increased efficiency and productivity, as machines can perform tasks that would otherwise require human intervention.
    
    2. Enhanced human-computer interaction: AI will continue to improve the way we interact with machines, making it easier to control and use them. This will lead to more seamless and intuitive user experiences, as machines will be able to understand and respond to our needs more accurately.
    
    3. Greater integration of AI with other technologies: AI will continue
    


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
    Generated text: ... (insert name)
    I am a/an fictional character. Who am I? I am a/an fictional character. Who am I? I am a/an anthropomorphic creature. How did you come up with the character's name? I came up with the character's name because it's a fun way to create something new and unique. It's a mix of the word "anthropomorphize" and "creature." I chose it because it's something that I'm proud of, and it's something that I can use in a story. How did you come up with the character's personality? I came up with the character's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the heart of the heart of France and the largest city in the country. It is also the city with the most tourist attractions and is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Champs-Élysées. Paris is also known for its rich history and culture, including the historic Louvre Museum, the Arc de Triomphe, and the Notre-Dame Cathedral. The city is home to many international institutions and organizations, including the European Parliament and the French Academy of Sciences. Paris is a vibrant and dynamic city with a rich cultural and social life, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advances and transformation, with several key trends set to shape the industry in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be increased pressure on companies to prioritize ethical concerns. This will likely lead to increased regulation and oversight of AI development, with more emphasis on transparency and accountability.
    
    2. Greater reliance on machine learning: As AI becomes more capable of learning from data and making decisions, there will be a greater emphasis on machine learning algorithms. This will likely lead to more sophisticated models that can handle complex and ambiguous situations.
    
    3. AI in healthcare


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Occup

    ation

    ].

     I

     enjoy

     [

    What

     you

     like

     to

     do

    ].

     I

     like

     to

     [

    What

     you

     like

     to

     do

    ].

     I

     also

     love

     [

    What

     you

     like

     to

     do

    ].

     I

     also

     enjoy

     [

    What

     you

     like

     to

     do

    ].

     I

     like

     to

     [

    What

     you

     like

     to

     do

    ].

     I

    'm

     excited

     to

     meet

     everyone

    !

     


    Remember

    ,

     I

    'm

     a

     human

    ,

     not

     a

     fictional

     character

    .

     I

    'm

     just

     a

     regular

     person

     who

     loves

     to

     be

     social

    .

     I

     don

    't

     have

     any

     special

     abilities

     or

     knowledge

     that

     I

    'm

     not

     aware

     of

    .

     I

     just

     enjoy

     interacting

     with

     people

     and

     sharing

     my

     thoughts

     and

     experiences

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

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

     Paris

     is

     known

     for

     its

     architecture

    ,

     museums

    ,

     and

     gastr

    onomy

    .

     The

     city

     is

     also

     a

     major

     financial

     center

     and

     a

     world

    -ren

    owned

     art

     center

    .

     It

     is

     the

     official

     residence

     of

     the

     French

     President

    .

     
    


    Paris

     was

     founded

     in

     

    7

    8

    7

     by

     Clo

    vis

    ,

     the

     first

     French

     king

    ,

     and

     has

     been

     the

     capital

     of

     France

     since

     

    8

    4

    3

    .

     The

     city

     is

     home

     to

     many

     historical

     landmarks

     and

     attractions

    ,

     including

     the

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

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     famous

     for

     its

     fashion

     industry

     and

     its

     annual

     fashion

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     dominated

     by

     autonomous

     systems

    ,

     which

     will

     be

     able

     to

     operate

     without

     human

     intervention

    .

     These

     systems

     will

     be

     able

     to

     process

     data

    ,

     make

     decisions

    ,

     and

     perform

     tasks

     without

     being

     limited

     by

     human

     knowledge

     or

     experience

    .

     Autonomous

     systems

     will

     also

     be

     able

     to

     learn

     from

     experience

    ,

     adapt

     to

     new

     situations

    ,

     and

     improve

     over

     time

    .

     This

     will

     lead

     to

     increased

     efficiency

     and

     productivity

    ,

     as

     well

     as

     improved

     safety

     and

     security

    .

     However

    ,

     autonomous

     systems

     may

     also

     lead

     to

     job

     displacement

     as

     some

     tasks

     become

     automated

    .

     Additionally

    ,

     the

     ethical

     considerations

     of

     AI

     are

     expected

     to

     be

     a

     major

     factor

     in

     shaping

     the

     future

     of

     AI

    .

     AI

     researchers

     are

     also

     expected

     to

     continue

     to

     explore

     new

     applications

    



```python
llm.shutdown()
```
