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

    [2026-03-17 10:13:14] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-17 10:13:14] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-17 10:13:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-17 10:13:17] INFO server_args.py:2160: Attention backend not specified. Use fa3 backend by default.


    [2026-03-17 10:13:17] INFO server_args.py:3330: Set soft_watchdog_timeout since in CI


    [2026-03-17 10:13:17] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=712614297, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:16,  3.16it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:16,  3.16it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:16,  3.16it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:16,  3.16it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:09,  5.42it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:09,  5.42it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:09,  5.42it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:09,  5.42it/s]

    Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  5.42it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:09,  5.42it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:04, 10.14it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 16.31it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 24.18it/s]

    Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=28):  71%|███████   | 41/58 [00:03<00:00, 35.80it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 50.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.18 GB):   2%|▏         | 1/58 [00:00<00:08,  7.09it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.12 GB):   2%|▏         | 1/58 [00:00<00:08,  7.09it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=71.13 GB):   2%|▏         | 1/58 [00:00<00:08,  7.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.13 GB):   5%|▌         | 3/58 [00:00<00:05, 10.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.13 GB):   5%|▌         | 3/58 [00:00<00:05, 10.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.12 GB):   5%|▌         | 3/58 [00:00<00:05, 10.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.12 GB):   9%|▊         | 5/58 [00:00<00:03, 13.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.11 GB):   9%|▊         | 5/58 [00:00<00:03, 13.70it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=71.11 GB):   9%|▊         | 5/58 [00:00<00:03, 13.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.10 GB):   9%|▊         | 5/58 [00:00<00:03, 13.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.08 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.08 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.07 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.07 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.66it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=960 avail_mem=71.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.52it/s] Capturing num tokens (num_tokens=896 avail_mem=71.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=832 avail_mem=71.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=832 avail_mem=71.00 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.75it/s]Capturing num tokens (num_tokens=768 avail_mem=71.01 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.75it/s]Capturing num tokens (num_tokens=704 avail_mem=71.00 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.75it/s]Capturing num tokens (num_tokens=640 avail_mem=70.99 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.75it/s]Capturing num tokens (num_tokens=576 avail_mem=70.99 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.75it/s]

    Capturing num tokens (num_tokens=512 avail_mem=70.98 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.75it/s]Capturing num tokens (num_tokens=512 avail_mem=70.98 GB):  50%|█████     | 29/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=480 avail_mem=70.99 GB):  50%|█████     | 29/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=448 avail_mem=70.98 GB):  50%|█████     | 29/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=416 avail_mem=70.96 GB):  50%|█████     | 29/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=384 avail_mem=70.97 GB):  50%|█████     | 29/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=352 avail_mem=70.96 GB):  50%|█████     | 29/58 [00:01<00:00, 34.76it/s]Capturing num tokens (num_tokens=352 avail_mem=70.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=320 avail_mem=70.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=288 avail_mem=70.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=256 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.77it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=224 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=224 avail_mem=70.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.60it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.60it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.59it/s]

    Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.59it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 32.58it/s]


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
    Generated text:  Jiri. I'm 19 years old and I'm in the 9th grade. I'm a student at a private high school in Prague. I enjoy learning new languages, and I like to travel. I want to be a musician, and I'm currently playing the violin.
    I'm also a member of the "Minorities and Women" club at the school, and I'm an active member of the "Support for the Disabled" club. I'm also a member of the "Occupy School" club. I like to play video games on my tablet.
    I like to eat a lot of chocolate, and I
    ===============================
    Prompt: The president of the United States is
    Generated text:  getting paid $100,000 per year. If he earns 7 times this amount, how much money does he earn every year?
    
    To determine how much money the president of the United States earns every year, we need to follow these steps:
    
    1. Identify the annual salary of the president.
    2. Calculate the amount of money he earns each year by multiplying the annual salary by 7.
    
    First, we know that the president earns $100,000 per year. We need to find out how much he earns every year if he earns 7 times this amount. We do this by multiplying $
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. London C. Shanghai D. Rome
    Answer:
    
    A
    
    Which of the following statements is incorrect?
    A. Sodium hydroxide is not a metal.
    B. The pH of pure water is 7.
    C. Hydrochloric acid is not a basic solution.
    D. The freezing point of water is -25.5°C.
    Answer:
    
    B
    
    Given two points M(-2, 3) and N(4, -1) on the coordinate plane, what is the equation of the line containing these two points?
    A. 4x + 3y - 2 =
    ===============================
    Prompt: The future of AI is
    Generated text:  here now, and its potential is limitless. In the next decade, the technologies that will shape the future of AI include deep learning, machine learning, natural language processing, and robotics. Deep learning and machine learning are the most promising technologies that have the potential to revolutionize the way we interact with machines. Natural language processing is also a crucial field that will play a key role in the future of AI. Robotics is another important field that will be transformed by the emergence of AI. The use of AI in the healthcare industry is also a major trend, and it has the potential to revolutionize the way we treat patients.
    In conclusion, the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I'm a [insert a short description of your favorite hobby or activity]. I'm always looking for new experiences and adventures. What's your favorite book or movie? I'm a [insert a short description of your favorite book or movie]. I'm always on the lookout for new and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the country and is known for its rich history, art, and culture. Paris is also a major transportation hub and a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for business and finance, with many important financial institutions and companies headquartered there. Paris is a vibrant and diverse city with a rich cultural and artistic heritage. It is a popular tourist destination and a major economic center in France. The city is known for its beautiful architecture, including the iconic Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI systems become more complex and sophisticated, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and fairness.
    
    2. Greater integration with human decision-making: AI systems will become more integrated with human decision-making processes, allowing for more nuanced and context-aware decision-making. This will require a greater understanding of human emotions, motivations, and biases.
    
    3. Increased use of AI in healthcare:
    


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
    Generated text:  [Name], and I'm a [Age] year old, [Occupation] professional. I'm always ready to learn, so I'll be glad to help you, or you can call me [Name]. How can I assist you today? [Name], is an enthusiastic, well-spoken individual who can provide excellent customer service. [Name], is always looking for ways to improve and continue to grow as a professional. [Name], is always eager to learn, and I'm always looking for ways to help you achieve your goals. [Name], is always ready to help and is always willing to learn. [Name], is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with the iconic Eiffel Tower and the iconic 1856 Opéra-Bельgique. It's a sprawling, cosmopolitan metropolis with a rich history and a vibrant cultural scene. The city is home to 3 million people, making it the country's most populous city, and is known for its fashion, art, and gastronomy. Paris is a cosmopolitan city with a diverse range of cultural attractions and museums, including the Louvre and the Palace of Versailles. The city is also home to many world-renowned food and wine landmarks, including the Eiffel Tower and the R
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several trends that are expected to drive innovation and advancement in the field. Some potential trends include:
    
    1. Increased specialization and expertise: AI is expected to become increasingly specialized, with more researchers and developers focusing on specific applications and domains. This will require new skills and knowledge, such as machine learning, computer vision, natural language processing, and deep learning.
    
    2. Improved data availability and quality: AI will become more effective as it is able to process and analyze large amounts of data. This will require the development of better data storage and retrieval systems, as well as increased investment in data science and analytics.
    
    3. Automation


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

    ].

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     a

     wide

     range

     of

     skills

     and

     experiences

     that

     have

     enabled

     me

     to

     make

     significant

     contributions

     to

     society

    .

     My

     main

     areas

     of

     expertise

     include

     [

    mention

     any

     specific

     fields

     or

     areas

     of

     interest

    ],

     and

     I

     believe

     that

     my

     experience

     has

     allowed

     me

     to

     thrive

     in

     all

     aspects

     of

     life

    .
    


    I

     am

     a

     collaborative

     problem

     solver

     with

     a

     strong

     work

     ethic

     and

     a

     desire

     to

     make

     a

     positive

     impact

     on

     others

     and

     society

     as

     a

     whole

    .

     I

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     knowledge

     and

     am

     always

     eager

     to

     learn

     and

     grow

    .
    


    I

     am

     excited

     to

     meet

     you

     and

     see

     the

     potential

     of

     our

     partnership

    .

     
    


    Thank

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     in

     France

     and

     is

     known

     for

     its

     diverse

     culture

    ,

     historic

     architecture

    ,

     and

     world

    -ren

    owned

     museums

    .
    


    What

     is

     the

     population

     of

     Paris

    ?

     Paris

     has

     a

     population

     of

     approximately

     

    2

    .

    1

     million

     people

    ,

     with

     an

     average

     of

     around

     

    5

    0

    0

    ,

    0

    0

    0

     residents

    .

     The

     city

     is

     home

     to

     many

     museums

    ,

     such

     as

     the

     Lou

    vre

    ,

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     National

     Museum

     of

     France

    .

     It

    's

     also

     home

     to

     several

     world

    -ren

    owned

     restaurants

    ,

     shopping

     centers

    ,

     and

     entertainment

     venues

    .
    


    That

    's

     great

    !

     Can

     you

     tell

     me

     more

     about

     Paris

    's

     famous

     landmarks

     and

     attractions

    ?

     Paris

     is

     famous

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     and

     expand

     rapidly

    ,

     with

     many

     exciting

     possibilities

     and

     opportunities

     on

     the

     horizon

    .

     Here

     are

     some

     of

     the

     possible

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

     ethics

     and

     responsibility

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     lives

    ,

     it

    's

     becoming

     increasingly

     important

     to

     ensure

     that

     it

    's

     used

     eth

    ically

     and

     responsibly

    .

     This

     includes

     ensuring

     that

     AI

     systems

     are

     transparent

    ,

     fair

    ,

     and

     have

     a

     clear

     path

     to

     responsible

     and

     responsible

     use

    .
    


    2

    .

     Greater

     integration

     with

     human

     expertise

    :

     AI

     is

     being

     increasingly

     integrated

     with

     human

     expertise

     in

     various

     fields

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    .

     This

     integration

     will

     likely

     lead

     to

     greater

     efficiency

    ,

     accuracy

    ,

     and

     reliability

    



```python
llm.shutdown()
```
