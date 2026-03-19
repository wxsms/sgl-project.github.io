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

    [2026-03-19 20:25:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-19 20:25:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-19 20:25:31] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-19 20:25:33] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-19 20:25:33] INFO server_args.py:2221: Attention backend not specified. Use fa3 backend by default.


    [2026-03-19 20:25:33] INFO server_args.py:3448: Set soft_watchdog_timeout since in CI


    [2026-03-19 20:25:33] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=930782142, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.73it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:04,  1.14s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:04,  1.14s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:04,  1.14s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:14,  3.67it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:14,  3.67it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.67it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.67it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.38it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.38it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.64it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.64it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.64it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.64it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 13.64it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 17.80it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 17.80it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 17.80it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 17.80it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 17.80it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 17.80it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 23.19it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 23.19it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 23.19it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 23.19it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 23.19it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 23.19it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 27.59it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 27.59it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 27.59it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 27.59it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 27.59it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 27.59it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 32.03it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 32.03it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 32.03it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 32.03it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 32.03it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 32.03it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 35.13it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 35.13it/s]

    Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 35.13it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 35.13it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 35.13it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 35.13it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 38.81it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 38.81it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 38.81it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 38.81it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 38.81it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 38.81it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 40.48it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 40.48it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 40.48it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 40.48it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 40.48it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 40.48it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 42.84it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 42.84it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 42.84it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 42.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.43 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.40 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.40 GB):   3%|▎         | 2/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.40 GB):   3%|▎         | 2/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.40 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.40 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.40 GB):   7%|▋         | 4/58 [00:00<00:07,  7.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.40 GB):   7%|▋         | 4/58 [00:00<00:07,  7.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:06,  7.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.39 GB):   9%|▊         | 5/58 [00:00<00:06,  7.81it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.39 GB):  10%|█         | 6/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.39 GB):  10%|█         | 6/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.39 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.39 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.41it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.32it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.74it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:06,  7.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.37 GB):  21%|██        | 12/58 [00:01<00:06,  7.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.37 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.37 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.50it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:01<00:06,  6.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.35 GB):  24%|██▍       | 14/58 [00:01<00:06,  6.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.35 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.34 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.70it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=58.34 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.34 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.82 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.82 GB):  31%|███       | 18/58 [00:02<00:04,  8.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.73 GB):  31%|███       | 18/58 [00:02<00:04,  8.85it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=57.66 GB):  31%|███       | 18/58 [00:02<00:04,  8.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.66 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.64 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.19it/s]Capturing num tokens (num_tokens=960 avail_mem=57.65 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.19it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=57.65 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.07it/s]Capturing num tokens (num_tokens=896 avail_mem=57.65 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.07it/s]Capturing num tokens (num_tokens=832 avail_mem=57.65 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.07it/s]Capturing num tokens (num_tokens=832 avail_mem=57.65 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.72it/s]Capturing num tokens (num_tokens=768 avail_mem=57.64 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.72it/s]

    Capturing num tokens (num_tokens=704 avail_mem=57.64 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.72it/s]Capturing num tokens (num_tokens=704 avail_mem=57.64 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.18it/s]Capturing num tokens (num_tokens=640 avail_mem=57.64 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.18it/s]Capturing num tokens (num_tokens=576 avail_mem=57.64 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.18it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.64 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.52it/s]Capturing num tokens (num_tokens=512 avail_mem=57.62 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.52it/s]Capturing num tokens (num_tokens=480 avail_mem=57.64 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.52it/s]Capturing num tokens (num_tokens=480 avail_mem=57.64 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.88it/s]Capturing num tokens (num_tokens=448 avail_mem=57.64 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.88it/s]

    Capturing num tokens (num_tokens=416 avail_mem=57.64 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.88it/s]Capturing num tokens (num_tokens=416 avail_mem=57.64 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.04it/s]Capturing num tokens (num_tokens=384 avail_mem=57.63 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.04it/s]Capturing num tokens (num_tokens=352 avail_mem=57.63 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.04it/s]

    Capturing num tokens (num_tokens=352 avail_mem=57.63 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.11it/s]Capturing num tokens (num_tokens=320 avail_mem=57.62 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.11it/s]Capturing num tokens (num_tokens=288 avail_mem=57.62 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.11it/s]Capturing num tokens (num_tokens=288 avail_mem=57.62 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.17it/s]Capturing num tokens (num_tokens=256 avail_mem=57.62 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.17it/s]

    Capturing num tokens (num_tokens=240 avail_mem=57.62 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.17it/s]Capturing num tokens (num_tokens=240 avail_mem=57.62 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.23it/s]Capturing num tokens (num_tokens=224 avail_mem=57.61 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.23it/s]Capturing num tokens (num_tokens=208 avail_mem=57.61 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.23it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.61 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.31it/s]Capturing num tokens (num_tokens=192 avail_mem=57.61 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.31it/s]Capturing num tokens (num_tokens=176 avail_mem=57.60 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.31it/s]Capturing num tokens (num_tokens=176 avail_mem=57.60 GB):  72%|███████▏  | 42/58 [00:04<00:01, 12.39it/s]Capturing num tokens (num_tokens=160 avail_mem=57.60 GB):  72%|███████▏  | 42/58 [00:04<00:01, 12.39it/s]

    Capturing num tokens (num_tokens=144 avail_mem=56.54 GB):  72%|███████▏  | 42/58 [00:04<00:01, 12.39it/s]Capturing num tokens (num_tokens=144 avail_mem=56.54 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.36it/s]Capturing num tokens (num_tokens=128 avail_mem=42.57 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.36it/s]Capturing num tokens (num_tokens=112 avail_mem=42.57 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.36it/s]

    Capturing num tokens (num_tokens=112 avail_mem=42.57 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.34it/s]Capturing num tokens (num_tokens=96 avail_mem=42.57 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.34it/s] Capturing num tokens (num_tokens=80 avail_mem=42.57 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.34it/s]Capturing num tokens (num_tokens=80 avail_mem=42.57 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.38it/s]Capturing num tokens (num_tokens=64 avail_mem=42.56 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.38it/s]

    Capturing num tokens (num_tokens=48 avail_mem=42.56 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.38it/s]Capturing num tokens (num_tokens=48 avail_mem=42.56 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.47it/s]Capturing num tokens (num_tokens=32 avail_mem=42.56 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.47it/s]Capturing num tokens (num_tokens=28 avail_mem=42.55 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.47it/s]

    Capturing num tokens (num_tokens=28 avail_mem=42.55 GB):  90%|████████▉ | 52/58 [00:05<00:00, 12.42it/s]Capturing num tokens (num_tokens=24 avail_mem=42.55 GB):  90%|████████▉ | 52/58 [00:05<00:00, 12.42it/s]Capturing num tokens (num_tokens=20 avail_mem=42.54 GB):  90%|████████▉ | 52/58 [00:05<00:00, 12.42it/s]Capturing num tokens (num_tokens=20 avail_mem=42.54 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.44it/s]Capturing num tokens (num_tokens=16 avail_mem=42.54 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.44it/s]

    Capturing num tokens (num_tokens=12 avail_mem=42.54 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.44it/s]Capturing num tokens (num_tokens=12 avail_mem=42.54 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.64it/s]Capturing num tokens (num_tokens=8 avail_mem=42.53 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.64it/s] Capturing num tokens (num_tokens=4 avail_mem=42.53 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.64it/s]Capturing num tokens (num_tokens=4 avail_mem=42.53 GB): 100%|██████████| 58/58 [00:05<00:00, 10.69it/s]


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
    Generated text:  Oliver John Hart, a 12 year old boy. I like to play baseball. I have a brother and a sister. Our father is a doctor and our mother is a nurse. I have a younger sister called Mandy. We all live in the same house. Our house has two bedrooms, a living room and a kitchen. Mandy and I are very good friends. We often play soccer together and go to the park on the weekend. My younger sister Mandy and I like to play baseball. It is our favorite sport. I have a blue baseball bat, a red baseball, and a yellow ball. My brother Tommy
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. The United States has 50 states, and for each state, there is a maximum of 50 military bases. The president wants to have the minimum number of bases possible, but still not exceed the maximum number for that state. For example, the president should not have 50 bases in any one state, but should have at least 50 bases in any one state. How many bases should the president consider for each state? Let's think through this problem step by step.
    
    1. We have 50 states, and for each state, there can be 
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Tokyo
    D. Madrid
    Answer:
    A
    
    In 1934, the Soviet Union and the United States signed a treaty, which stated: "The United States will pay the Soviet Union the full amount of the payment it makes to the United States of America for the delivery of wheat and other agricultural products to the Soviet Union." This treaty's purpose was to ____
    A. Promote economic development
    B. Propose international cooperation
    C. Promote peaceful relations
    D. Implement international trade
    Answer:
    B
    
    In the 1950s and
    ===============================
    Prompt: The future of AI is
    Generated text:  here – and it’s already starting to make decisions that change the way we live our lives.
    As artificial intelligence and machine learning move toward ever greater sophistication, it’s becoming apparent that the decisions we make are not just about making sure the computer understands the language we use, but about how it decides.
    One of the big differences between humans and AI is the ability to understand context and intent, and how AI can use that information to make decisions. For example, when someone asks about the weather, if we’re talking to a human, we’re generally going to assume that they want to know the temperature and the weather forecast. In contrast,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills that you're passionate about]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a hobby or activity that you enjoy]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite book or movie? I love [insert a book or movie that you've read or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" in French. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, and is a popular destination for tourists and locals alike. The city is home to many museums, theaters, and other cultural institutions, and is a major center for French politics and society. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and evaluation of AI systems, as well as greater transparency and accountability
    


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
    Generated text:  [Your Name], and I am a [职业] from [Your location]. I have [number] of years of experience in [职业] and [number] of awards in the industry. I am always ready to learn and grow, and I am dedicated to [职业] and my goals. How can I be of assistance to you? [Your Name] [Your Contact Information] [Your Professional Background] [Your Skills and Qualifications] [Your Awards and Achievements]
    Hello, my name is [Your Name], and I am a [职业] from [Your location]. I have [number] of years of experience in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical architecture, iconic landmarks such as the Eiffel Tower, and a rich cultural heritage. It is also home to the Louvre Museum, the iconic place for witnessing world-renowned artworks like the Mona Lisa. The city is also known for its fashion and food scene, with its famous streets and bakeries serving classic French cuisine. Paris is one of the most populous cities in the world, making it a cosmopolitan hub that attracts visitors from around the world. It is a major center of learning, science, and art, and is recognized for its intellectual, cultural, and artistic achievements. Despite its size,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by exponential growth, the development of new technologies and applications, and the emergence of new ethical and regulatory challenges. Here are some potential trends that could shape the future of AI:
    
    1. AI will continue to become more integrated into human life in new and unexpected ways. For example, AI-powered devices such as smart homes, self-driving cars, and virtual assistants are already being developed and are set to become more widespread.
    
    2. AI will continue to become more capable and capable of performing a wider range of tasks than current human capabilities. As AI is able to process and analyze large amounts of data faster than humans, it will


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

     __

    ________

     and

     I

     am

     a

    (n

    )

     __

    ________

    .

     I

     am

     a

    /an

     ______

    ___

     (

    e

    .g

    .,

     teacher

    ,

     doctor

    ,

     artist

    ,

     etc

    .)

     who

     am

     passionate

     about

     __

    ________

     (

    e

    .g

    .,

     helping

     people

    ,

     making

     art

    ,

     studying

    ,

     etc

    .).

     I

     enjoy

     __

    ________

     (

    e

    .g

    .,

     traveling

    ,

     writing

    ,

     exploring

     new

     ideas

    ,

     etc

    .)

     and

     often

     get

     lost

     in

     the

     thrill

     of

     __

    ________

     (

    e

    .g

    .,

     unknown

     adventures

    ,

     mysterious

     landscapes

    ,

     etc

    .).

     I

     am

     always

     learning

     and

     growing

    ,

     and

     my

     journey

     is

     __

    ________

     (

    e

    .g

    .,

     discovery

    ,

     enlightenment

    ,

     continuous

     improvement

    ,

     etc

    .).

     I

     believe

     in

     the

     power

     of

     __

    ________

     (

    e

    .g

    .,

     teamwork

    
    
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

     most

     populous

     city

     in

     the

     European

     Union

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

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

     the

     Ch

    amps

    -

    É

    lys

    ées

    ,

     the

     Notre

    -D

    ame

     de

     Paris

    ,

     and

     many

     other

     historic

     sites

     and

     attractions

    .

     It

     is

     also

     a

     cultural

     and

     educational

     center

    ,

     with

     many

     universities

    ,

     museums

    ,

     and

     theaters

     located

     in

     the

     city

    .

     Paris

     has

     a

     rich

     history

     and

     has

     been

     a

     center

     of

     French

     culture

     and

     politics

     for

     centuries

    .

     Its

     status

     as

     the

     capital

     of

     France

     is

     a

     testament

     to

     its

     importance

     and

     significance

     in

     the

     country

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     will

     be

     shaped

     by

     a

     wide

     range

     of

     complex

     factors

    ,

     including

     advancements

     in

     technology

    ,

     shifts

     in

     societal

     values

    ,

     and

     evolving

     ethical

     considerations

    .

     Here

     are

     some

     of

     the

     most

     likely

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

     systems

     become

     more

     advanced

    ,

     there

     will

     likely

     be

     more

     emphasis

     on

     ensuring

     their

     ethical

     use

     and

     minimizing

     negative

     consequences

    .

     This

     includes

     things

     like

     privacy

    ,

     bias

    ,

     and

     fairness

    .
    


    2

    .

     Increasing

     reliance

     on

     AI

     for

     routine

     tasks

    :

     While

     AI

     may

     one

     day

     be

     able

     to

     perform

     tasks

     that

     require

     human

     intelligence

    ,

     it

    's

     unlikely

     that

     it

     will

     replace

     all

     human

     jobs

    .

     Instead

    ,

     it

    's

     likely

     that

     AI

     will

     be

    



```python
llm.shutdown()
```
