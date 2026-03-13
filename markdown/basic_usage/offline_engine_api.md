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

    [2026-03-13 05:33:35] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-13 05:33:35] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-13 05:33:35] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-13 05:33:38] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.


    [2026-03-13 05:33:38] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-13 05:33:38] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=858044413, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.87it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.86it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.53it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 19.60it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 28.58it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 37.43it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 46.29it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 46.29it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 46.29it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 46.29it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 46.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=122.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=122.41 GB):   2%|▏         | 1/58 [00:00<00:13,  4.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=122.23 GB):   2%|▏         | 1/58 [00:00<00:13,  4.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.10 GB):   2%|▏         | 1/58 [00:00<00:13,  4.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.10 GB):   5%|▌         | 3/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.09 GB):   5%|▌         | 3/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.09 GB):   5%|▌         | 3/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.09 GB):   5%|▌         | 3/58 [00:00<00:05, 10.11it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=121.09 GB):  10%|█         | 6/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.09 GB):  10%|█         | 6/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.07 GB):  10%|█         | 6/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.07 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.07 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.06 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=121.06 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.51it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=121.05 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=121.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=121.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=121.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=121.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=121.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=121.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=121.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=121.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=960 avail_mem=121.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.75it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=121.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=832 avail_mem=121.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=832 avail_mem=121.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=768 avail_mem=121.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=704 avail_mem=121.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=640 avail_mem=120.99 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=576 avail_mem=120.99 GB):  41%|████▏     | 24/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=512 avail_mem=120.98 GB):  41%|████▏     | 24/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=512 avail_mem=120.98 GB):  50%|█████     | 29/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=480 avail_mem=120.99 GB):  50%|█████     | 29/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=448 avail_mem=120.99 GB):  50%|█████     | 29/58 [00:01<00:00, 37.01it/s]

    Capturing num tokens (num_tokens=416 avail_mem=120.98 GB):  50%|█████     | 29/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=384 avail_mem=120.98 GB):  50%|█████     | 29/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=352 avail_mem=120.97 GB):  50%|█████     | 29/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=352 avail_mem=120.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=320 avail_mem=120.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=288 avail_mem=120.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=256 avail_mem=120.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=240 avail_mem=120.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=224 avail_mem=120.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.44it/s]Capturing num tokens (num_tokens=224 avail_mem=120.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=208 avail_mem=120.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.42it/s]

    Capturing num tokens (num_tokens=192 avail_mem=120.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=176 avail_mem=120.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=160 avail_mem=120.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=144 avail_mem=120.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=144 avail_mem=120.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=128 avail_mem=120.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=112 avail_mem=120.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.03it/s]

    Capturing num tokens (num_tokens=96 avail_mem=120.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.03it/s] Capturing num tokens (num_tokens=80 avail_mem=120.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=64 avail_mem=120.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=64 avail_mem=120.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=48 avail_mem=120.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=32 avail_mem=120.38 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=28 avail_mem=120.29 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.96it/s]

    Capturing num tokens (num_tokens=24 avail_mem=120.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=24 avail_mem=120.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.79it/s]Capturing num tokens (num_tokens=20 avail_mem=120.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.79it/s]Capturing num tokens (num_tokens=16 avail_mem=120.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.79it/s]Capturing num tokens (num_tokens=12 avail_mem=120.20 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.79it/s]Capturing num tokens (num_tokens=8 avail_mem=120.20 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.79it/s] Capturing num tokens (num_tokens=4 avail_mem=120.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.79it/s]Capturing num tokens (num_tokens=4 avail_mem=120.19 GB): 100%|██████████| 58/58 [00:01<00:00, 35.08it/s]Capturing num tokens (num_tokens=4 avail_mem=120.19 GB): 100%|██████████| 58/58 [00:01<00:00, 30.93it/s]


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
    Generated text:  Karen and I'm 15. My birthday is on November 18th. I have a sister. Her name is Mary. She is 9 years old. We have different hobbies. My favorite game is playing computer games. It is fun and I like to play it every day. Mary likes to play tennis. She is good at tennis. She only plays it on weekends. At school, I like to draw pictures. It is my favorite activity. But my friend Sam likes to draw. We often draw pictures together at school. Sam thinks my picture is very beautiful. We are both 15. We like to
    ===============================
    Prompt: The president of the United States is
    Generated text:  36 years younger than his cousin. If the president is 56 years old, calculate the difference in age between the president and his cousin.
    To determine the difference in age between the president and his cousin, we need to follow these steps:
    
    1. Identify the age of the president.
    2. Identify the age of the cousin.
    3. Calculate the difference between the president's age and the cousin's age.
    
    The president is 36 years old. According to the problem, the president is 56 years old. Therefore, the difference in age between the president and his cousin is:
    
    \[ 56 - 
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. ____
    A. Lille
    B. Paris
    C. Lillehammer
    D. Lyon
    
    To determine the capital of France, let's go through the options step by step:
    
    A. Lille
       - Lille is a city in France but is not the capital. It is the seat of government.
    
    B. Paris
       - Paris is the capital of France. It is the largest city and the seat of the French government.
    
    C. Lillehammer
       - Lillehammer is a city in Norway, not France. It is not the capital.
    
    D. Lyon
       - Lyon is
    ===============================
    Prompt: The future of AI is
    Generated text:  coming fast and furious
    
    What is the future of AI, and how is it shaping up to change the way we live our lives?
    
    by March 17, 2016
    
    There has never been a better time for AI to change the world. Innovations like autonomous vehicles, smart home devices, predictive analytics, and AI chatbots are rapidly developing. Even more exciting, AI is transforming the business and financial industries.
    
    And there are still a lot more to come. In fact, in a recent analysis by the McKinsey Global Institute, 2016 AI trends will cover almost every industry, from government and healthcare


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


    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] [Ability] who has been [Experience] years in the [Field] industry. I'm passionate about [What I Love to Do], and I'm always looking for new opportunities to [What I Want to Do]. I'm a [Personality] person who is [What I Like to Do]. I'm [What I Do For a Living]. I'm [What I Do For a Living]. I'm [What I Do For a Living]. I'm [What I Do For a Living]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its fashion industry, art, and cuisine. Paris is a major cultural and economic center in Europe and is home to many world-renowned museums, theaters, and landmarks. It is a popular tourist destination and a major center for business and politics. The city is also known for its annual Eiffel Tower Festival and its annual fashion week
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical considerations: As AI systems become more complex and sophisticated, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes
    


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
    Generated text:  [name] and I'm [age]. I'm a [occupation/role] with [number of years of experience] years of experience in [field]. I'm a [personality trait] person, and I enjoy [action]. I'm a [role model or influencer] and I believe that [positive statement].
    Here's an example of a neutral self-introduction for a fictional character:
    "Hi there! My name is Sarah and I'm a [occupation/role] with [number of years of experience] years of experience in [field]. I'm a [personality trait] person, and I enjoy [action
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris, often referred to as "La Ville Fluviale" (The River City), is the capital city of France. The city is located on the River Seine and is home to many of the country's landmarks and cultural institutions, including Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is also known for its rich history, vibrant arts scene, and lively nightlife. Its status as a global cultural and political capital is further recognized by its high population density and annual tourism industry. Paris is a city of contrasts - from the lively night life of Montmartre to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and there are many possibilities. Here are some possible trends in AI:
    
    1. Machine learning: Machine learning algorithms will continue to improve and become more capable of learning from data to make more accurate predictions and decisions.
    
    2. Deep learning: Deep learning, a subset of machine learning, will continue to gain strength and power. Deep neural networks will become more complex and smarter, allowing for even more advanced tasks and applications.
    
    3. Cybersecurity: As AI becomes more integrated into our daily lives, cybersecurity will become an even more critical issue. AI systems will need to be more secure and have better protection against cyber threats.
    
    4. Autonomous vehicles


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

     a

     [

    Job

     Title

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

     in

     [

    Industry

     or

     field

    ].

     My

     goal

     is

     to

     contribute

     to

     the

     company

     in

     [

    specific

     area

    ]

     through

     [

    specific

     project

     or

     task

    ].

     I

     am

     a

     [

    positive

     or

     negative

    ]

     person

     who

     is

     always

     ready

     to

     learn

     and

     improve

    .

     I

     am

     confident

     in

     my

     abilities

     and

     strive

     to

     exceed

     expectations

     at

     all

     times

    .

     [

    Name

    ]

     will

     always

     be

     your

     guide

    ,

     mentor

    ,

     and

     friend

     in

     the

     company

    .

     I

     am

     dedicated

     to

     my

     work

     and

     always

     strive

     to

     excel

     in

     my

     role

    .

     I

     am

     passionate

     about

     my

     profession

     and

     committed

     to

     contributing

     to

     the

     company

    's

     success

    .

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     hosts

     the

     E

    iff

    el

     Tower

     and

     is

     the

     seat

     of

     the

     Government

     of

     France

    .

     It

     is

     located

     in

     the

     I

    le

     de

     la

     C

    ite

     region

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

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

     many

     world

    -ren

    owned

     landmarks

     and

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     is

     a

     cultural

     and

     economic

     hub

     of

     France

    .

     With

     its

     stunning

     views

     and

     bustling

     atmosphere

    ,

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     French

     culture

     and

     history

    .

     
    


    Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     historical

     importance

     and

     rich

     cultural

     heritage

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

    ,

     with

     many

     different

     areas

     of

     research

     and

     development

     underway

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

     accuracy

     and

     efficiency

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     expected

     to

     become

     even

     more

     accurate

     in

     its

     predictions

     and

     decisions

    .

     This

     could

     lead

     to

     applications

     such

     as

     self

    -driving

     cars

    ,

     personalized

     medicine

    ,

     and

     autonomous

     drones

    .
    


    2

    .

     Personal

    ized

     learning

    :

     With

     AI

    ,

     we

     can

     learn

     and

     adapt

     to

     our

     environment

    ,

     which

     means

     that

     our

     learning

     could

     become

     more

     personalized

    .

     This

     could

     lead

     to

     a

     more

     efficient

     learning

     process

     and

     better

     quality

     of

     life

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     could

     revolution

    ize

     healthcare

     by

     improving

     the

     accuracy

     of

     medical

     diagnoses

    



```python
llm.shutdown()
```
