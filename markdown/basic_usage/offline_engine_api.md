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

    [2026-03-14 19:20:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-14 19:20:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-14 19:20:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-14 19:20:18] INFO server_args.py:2146: Attention backend not specified. Use fa3 backend by default.


    [2026-03-14 19:20:18] INFO server_args.py:3287: Set soft_watchdog_timeout since in CI


    [2026-03-14 19:20:18] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=813195000, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.39it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.62it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 30.79it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 36.95it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 37.38it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 37.38it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 37.38it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 37.38it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 37.38it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 37.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.34 GB):   2%|▏         | 1/58 [00:00<00:08,  6.52it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.31 GB):   2%|▏         | 1/58 [00:00<00:08,  6.52it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.31 GB):   3%|▎         | 2/58 [00:00<00:07,  7.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.31 GB):   3%|▎         | 2/58 [00:00<00:07,  7.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.31 GB):   5%|▌         | 3/58 [00:00<00:07,  7.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.31 GB):   5%|▌         | 3/58 [00:00<00:07,  7.18it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.31 GB):   7%|▋         | 4/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.31 GB):   7%|▋         | 4/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.31 GB):   9%|▊         | 5/58 [00:00<00:07,  7.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.30 GB):   9%|▊         | 5/58 [00:00<00:07,  7.20it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=59.30 GB):  10%|█         | 6/58 [00:00<00:06,  7.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.31 GB):  10%|█         | 6/58 [00:00<00:06,  7.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.31 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.30 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.00it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.30 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.30 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.30 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.26it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.30 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.26it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.30 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.29 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.29 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.29 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.63it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.29 GB):  21%|██        | 12/58 [00:01<00:05,  8.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.29 GB):  21%|██        | 12/58 [00:01<00:05,  8.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.29 GB):  21%|██        | 12/58 [00:01<00:05,  8.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.29 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.28 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.49it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=59.28 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.28 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.27 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.27 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.54it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=59.27 GB):  31%|███       | 18/58 [00:01<00:03, 11.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.27 GB):  31%|███       | 18/58 [00:01<00:03, 11.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.26 GB):  31%|███       | 18/58 [00:02<00:03, 11.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.26 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.24 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.11it/s]Capturing num tokens (num_tokens=960 avail_mem=59.26 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.11it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=59.26 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=896 avail_mem=59.25 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=832 avail_mem=59.25 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=832 avail_mem=59.25 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=768 avail_mem=59.25 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=704 avail_mem=59.25 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.83it/s]

    Capturing num tokens (num_tokens=640 avail_mem=59.24 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=640 avail_mem=59.24 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.55it/s]Capturing num tokens (num_tokens=576 avail_mem=59.24 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.55it/s]Capturing num tokens (num_tokens=512 avail_mem=59.23 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.55it/s]Capturing num tokens (num_tokens=480 avail_mem=59.25 GB):  47%|████▋     | 27/58 [00:02<00:01, 17.55it/s]Capturing num tokens (num_tokens=480 avail_mem=59.25 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.39it/s]Capturing num tokens (num_tokens=448 avail_mem=59.24 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.39it/s]Capturing num tokens (num_tokens=416 avail_mem=59.24 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.39it/s]

    Capturing num tokens (num_tokens=384 avail_mem=59.24 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.39it/s]Capturing num tokens (num_tokens=384 avail_mem=59.24 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.96it/s]Capturing num tokens (num_tokens=352 avail_mem=59.23 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.96it/s]Capturing num tokens (num_tokens=320 avail_mem=59.23 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.96it/s]Capturing num tokens (num_tokens=288 avail_mem=59.22 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.96it/s]Capturing num tokens (num_tokens=256 avail_mem=59.22 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.96it/s]Capturing num tokens (num_tokens=256 avail_mem=59.22 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.07it/s]Capturing num tokens (num_tokens=240 avail_mem=59.22 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.07it/s]Capturing num tokens (num_tokens=224 avail_mem=59.21 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.07it/s]

    Capturing num tokens (num_tokens=208 avail_mem=59.21 GB):  64%|██████▍   | 37/58 [00:02<00:00, 25.07it/s]Capturing num tokens (num_tokens=208 avail_mem=59.21 GB):  69%|██████▉   | 40/58 [00:02<00:00, 25.36it/s]Capturing num tokens (num_tokens=192 avail_mem=59.21 GB):  69%|██████▉   | 40/58 [00:02<00:00, 25.36it/s]Capturing num tokens (num_tokens=176 avail_mem=59.20 GB):  69%|██████▉   | 40/58 [00:02<00:00, 25.36it/s]Capturing num tokens (num_tokens=160 avail_mem=59.20 GB):  69%|██████▉   | 40/58 [00:02<00:00, 25.36it/s]Capturing num tokens (num_tokens=144 avail_mem=59.20 GB):  69%|██████▉   | 40/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=128 avail_mem=59.19 GB):  69%|██████▉   | 40/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=112 avail_mem=59.19 GB):  69%|██████▉   | 40/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=112 avail_mem=59.19 GB):  79%|███████▉  | 46/58 [00:03<00:00, 32.81it/s]Capturing num tokens (num_tokens=96 avail_mem=59.19 GB):  79%|███████▉  | 46/58 [00:03<00:00, 32.81it/s] Capturing num tokens (num_tokens=80 avail_mem=59.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 32.81it/s]

    Capturing num tokens (num_tokens=64 avail_mem=59.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 32.81it/s]Capturing num tokens (num_tokens=48 avail_mem=59.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 32.81it/s]Capturing num tokens (num_tokens=32 avail_mem=59.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 32.81it/s]Capturing num tokens (num_tokens=32 avail_mem=59.18 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Capturing num tokens (num_tokens=28 avail_mem=59.17 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Capturing num tokens (num_tokens=24 avail_mem=59.17 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Capturing num tokens (num_tokens=20 avail_mem=59.16 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Capturing num tokens (num_tokens=16 avail_mem=59.16 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Capturing num tokens (num_tokens=16 avail_mem=59.16 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s]Capturing num tokens (num_tokens=12 avail_mem=59.16 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s]

    Capturing num tokens (num_tokens=8 avail_mem=59.16 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s] Capturing num tokens (num_tokens=4 avail_mem=59.15 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s]Capturing num tokens (num_tokens=4 avail_mem=59.15 GB): 100%|██████████| 58/58 [00:03<00:00, 16.93it/s]


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
    Generated text:  Matthew and I have many friends. One of my best friends is Raj. I miss him terribly. I talk about him a lot and we would play games on the computer together. I used to have some very good conversations with him. I made him feel so happy. I played all my favorite games with him. I even took him to see a film. I miss him very much. What is the best way to write a letter to a friend? A) To show your good friendship. B) To tell him how sorry you are for not being here to play games with you. C) To tell him what you have done and
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to continue with the current round-robin tennis tournament format, where each team plays every other team exactly once, or to switch to a single-elimination format. In a single-elimination tournament, teams are paired off in rounds and eliminated one by one until only one team remains. 
    
    If the current format has 32 teams, how many rounds will it take to determine the winner? To determine the number of rounds required to determine the winner in a single-elimination format, we need to understand that each round eliminates one team. Let's break it down step by step.
    
    1. **Initial Number
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following statements is false?
    A. The French people all speak French.
    B. The French people all speak French.
    C. The French people all speak French.
    D. The French people all speak French.
    Answer:
    B
    
    The official language of the United States is ______.
    A. English
    B. French
    C. Spanish
    D. Russian
    Answer:
    A
    
    The official language of the United Kingdom is ______.
    A. English
    B. French
    C. Spanish
    D. Russian
    Answer:
    A
    
    The official language of the United States is ______.
    A. English
    B. French
    
    ===============================
    Prompt: The future of AI is
    Generated text:  solar
    
    I have been told by a colleague that he is going to do a PhD on something that requires a PhD, but he doesn’t know what that is.
    
    It may be that this is the start of a really interesting PhD. The University of Warwick just decided to launch a PhD in AI. In addition, the University is also conducting the first pilot of artificial intelligence at the £6 billion UK Solar Company.
    
    The Solar Company is developing a photovoltaic power system with a higher energy conversion efficiency than traditional solar panels. The photovoltaic panels are supposed to generate electricity from sunlight, but the existing technology is not able to


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm always eager to learn and grow, and I'm always looking for new challenges and opportunities to grow and succeed. What's your name? What's your job title? What's your company name? What's your passion? What's your current job title? What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 8 million people. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling and dynamic city with a rich cultural and artistic heritage, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and nuanced AI systems that can better understand and respond to human emotions and motivations.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a greater need for privacy and security measures to protect user data and prevent misuse of AI systems. This could lead to more stringent regulations and standards for AI development and deployment.
    
    3. Greater reliance on AI for decision-making: AI is
    


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
    Generated text:  [Name] and I am a [age] year old, [profession] type. I've got [number] years of experience in [industry or field]. I'm [description] and I'm currently [career stage] in [industry or field]. I enjoy [reason for interest]. I have a great sense of humor and love to [reason for interest] with [close relation]. I'm a [description] person. I like [reason for interest] with [close relation]. I'm an [age] year old, [profession] type and I have been [career stage] in [industry or field] for [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an enchanting city famous for its art, food, and world-renowned landmarks like the Eiffel Tower.
    
    That's correct! Paris is the capital of France, known for its beautiful architecture, rich history, and vibrant culture. Some famous landmarks include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and more. The city is also famous for its fashion scene and delicious food, such as croissants and patisseries. Paris is a bustling metropolis with a rich cultural heritage, and it's always a beautiful place to visit! 😍✨
    
    What are some unique food and drink experiences
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field that is poised for significant changes and innovations. Here are some potential future trends in AI:
    
    1. Increased Intelligence and Self-Driving Cars: With the development of deep learning and neural networks, autonomous driving systems are becoming more sophisticated. This is expected to lead to the development of self-driving cars that can navigate roads and traffic conditions on their own.
    
    2. Improved Chatbots: AI chatbots are becoming increasingly intelligent and capable of responding to a wide range of questions and tasks. This is expected to revolutionize customer service and support.
    
    3. AI in Healthcare: AI is already being used in healthcare to analyze medical images,


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

    Your

     Name

    ],

     and

     I

    'm

     currently

     working

     as

     a

     [

    Job

     Title

    ]

     at

     [

    Your

     Company

    ].

     I

    'm

     really

     excited

     about

     the

     opportunity

     to

     use

     my

     skills

     in

     [

    Your

     Field

    /

    Job

    ].

     What

     kind

     of

     projects

     do

     you

     work

     on

    ?

     What

     languages

     do

     you

     speak

    ?

     How

     many

     clients

     do

     you

     work

     with

    ?

     What

    's

     your

     favorite

     hobby

    ?


    Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ],

     and

     I

    'm

     currently

     working

     as

     a

     [

    Job

     Title

    ]

     at

     [

    Your

     Company

    ].

     I

    'm

     really

     excited

     about

     the

     opportunity

     to

     use

     my

     skills

     in

     [

    Your

     Field

    /

    Job

    ].

     What

     kind

     of

     projects

     do

     you

     work

     on

    ?

     What

     languages

     do

     you

     speak

    ?

     How

     many

     clients

    
    
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

     by

     population

     and

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    .

     Paris

     is

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    .

     It

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

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

     The

     city

     is

     known

     for

     its

     architecture

    ,

     art

    ,

     and

     cuisine

    .

     It

     is

     also

     a

     major

     tourist

     destination

    ,

     hosting

     many

     events

     and

     festivals

     throughout

     the

     year

    .

     Paris

     has

     a

     rich

     and

     diverse

     culture

     that

     reflects

     its

     history

     and

     contributions

     to

     European

     civilization

    .

     The

     city

     is

     also

     home

     to

     a

     large

     number

     of

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     As

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     blend

     of

     innovation

     and

     caution

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Enhanced

     intelligence

    :

     AI

     is

     expected

     to

     continue

     to

     advance

    ,

     with

     more

     complex

     and

     sophisticated

     models

     becoming

     possible

    .

     However

    ,

     it

     is

     also

     important

     to

     consider

     the

     ethical

     implications

     of

     AI

    ,

     such

     as

     bias

    ,

     privacy

    ,

     and

     control

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     likely

     to

     become

     more

     common

     in

     the

     future

    ,

     as

     AI

     becomes

     more

     integrated

     into

     transportation

     systems

    .

     However

    ,

     they

     may

     also

     raise

     safety

     concerns

    ,

     particularly

     in

     densely

     populated

     areas

    .
    


    3

    .

     Personal

    ized

     health

     care

    :

     AI

     is

     expected

     to

     play

     a

     greater

     role

     in

     personalized

     health

    



```python
llm.shutdown()
```
