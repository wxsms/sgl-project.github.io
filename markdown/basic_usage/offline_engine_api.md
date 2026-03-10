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
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    [2026-03-10 01:35:04] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-10 01:35:04] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-10 01:35:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-10 01:35:06] INFO server_args.py:2133: Attention backend not specified. Use fa3 backend by default.


    [2026-03-10 01:35:06] INFO server_args.py:3246: Set soft_watchdog_timeout since in CI


    [2026-03-10 01:35:06] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=1038175939, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=45.57 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=45.57 GB):   5%|▌         | 1/20 [00:00<00:02,  6.43it/s]Capturing batches (bs=120 avail_mem=45.47 GB):   5%|▌         | 1/20 [00:00<00:02,  6.43it/s]Capturing batches (bs=112 avail_mem=45.47 GB):   5%|▌         | 1/20 [00:00<00:02,  6.43it/s]

    Capturing batches (bs=104 avail_mem=45.47 GB):   5%|▌         | 1/20 [00:00<00:02,  6.43it/s]Capturing batches (bs=96 avail_mem=45.47 GB):   5%|▌         | 1/20 [00:00<00:02,  6.43it/s] Capturing batches (bs=96 avail_mem=45.47 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.30it/s]Capturing batches (bs=88 avail_mem=45.47 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.30it/s]Capturing batches (bs=80 avail_mem=45.47 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.30it/s]Capturing batches (bs=72 avail_mem=45.47 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.30it/s]Capturing batches (bs=64 avail_mem=45.47 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.30it/s]Capturing batches (bs=56 avail_mem=45.47 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.30it/s]Capturing batches (bs=56 avail_mem=45.47 GB):  50%|█████     | 10/20 [00:00<00:00, 29.84it/s]Capturing batches (bs=48 avail_mem=45.46 GB):  50%|█████     | 10/20 [00:00<00:00, 29.84it/s]

    Capturing batches (bs=40 avail_mem=45.46 GB):  50%|█████     | 10/20 [00:00<00:00, 29.84it/s]Capturing batches (bs=32 avail_mem=45.46 GB):  50%|█████     | 10/20 [00:00<00:00, 29.84it/s]Capturing batches (bs=24 avail_mem=45.46 GB):  50%|█████     | 10/20 [00:00<00:00, 29.84it/s]Capturing batches (bs=24 avail_mem=45.46 GB):  70%|███████   | 14/20 [00:00<00:00, 31.62it/s]Capturing batches (bs=16 avail_mem=45.46 GB):  70%|███████   | 14/20 [00:00<00:00, 31.62it/s]Capturing batches (bs=12 avail_mem=45.46 GB):  70%|███████   | 14/20 [00:00<00:00, 31.62it/s]Capturing batches (bs=8 avail_mem=45.46 GB):  70%|███████   | 14/20 [00:00<00:00, 31.62it/s] Capturing batches (bs=4 avail_mem=45.46 GB):  70%|███████   | 14/20 [00:00<00:00, 31.62it/s]

    Capturing batches (bs=4 avail_mem=45.46 GB):  90%|█████████ | 18/20 [00:00<00:00, 31.24it/s]Capturing batches (bs=2 avail_mem=45.46 GB):  90%|█████████ | 18/20 [00:00<00:00, 31.24it/s]Capturing batches (bs=1 avail_mem=45.45 GB):  90%|█████████ | 18/20 [00:00<00:00, 31.24it/s]Capturing batches (bs=1 avail_mem=45.45 GB): 100%|██████████| 20/20 [00:00<00:00, 29.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:12,  2.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:12,  2.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:12,  2.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:12,  2.32s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:07,  6.54it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 15.61it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:02<00:01, 25.82it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=28):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=24):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=20):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=16):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=12):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=8):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s] 

    Compiling num tokens (num_tokens=4):  71%|███████   | 41/58 [00:02<00:00, 37.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 60.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 19.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.12 GB):   2%|▏         | 1/58 [00:00<00:06,  9.11it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.08 GB):   2%|▏         | 1/58 [00:00<00:06,  9.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.08 GB):   2%|▏         | 1/58 [00:00<00:06,  9.11it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.08 GB):   5%|▌         | 3/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.08 GB):   5%|▌         | 3/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.08 GB):   5%|▌         | 3/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.08 GB):   5%|▌         | 3/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.08 GB):  10%|█         | 6/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.08 GB):  10%|█         | 6/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.08 GB):  10%|█         | 6/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.07 GB):  10%|█         | 6/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.07 GB):  10%|█         | 6/58 [00:00<00:02, 19.63it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=45.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 26.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.06 GB):  17%|█▋        | 10/58 [00:00<00:01, 26.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.06 GB):  17%|█▋        | 10/58 [00:00<00:01, 26.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.06 GB):  17%|█▋        | 10/58 [00:00<00:01, 26.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.06 GB):  17%|█▋        | 10/58 [00:00<00:01, 26.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.05 GB):  17%|█▋        | 10/58 [00:00<00:01, 26.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.03 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=45.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.01 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=960 avail_mem=45.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s] Capturing num tokens (num_tokens=896 avail_mem=45.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=832 avail_mem=45.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=768 avail_mem=45.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=704 avail_mem=45.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=640 avail_mem=45.01 GB):  36%|███▌      | 21/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=640 avail_mem=45.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=576 avail_mem=45.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=512 avail_mem=45.00 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=480 avail_mem=45.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=448 avail_mem=45.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.29it/s]

    Capturing num tokens (num_tokens=416 avail_mem=45.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=416 avail_mem=45.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=384 avail_mem=45.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=352 avail_mem=45.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=320 avail_mem=45.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=288 avail_mem=45.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.52it/s]Capturing num tokens (num_tokens=256 avail_mem=45.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=256 avail_mem=45.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=240 avail_mem=44.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=224 avail_mem=44.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=208 avail_mem=44.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]

    Capturing num tokens (num_tokens=192 avail_mem=44.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=176 avail_mem=44.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=160 avail_mem=44.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=160 avail_mem=44.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.76it/s]Capturing num tokens (num_tokens=144 avail_mem=44.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.76it/s]Capturing num tokens (num_tokens=128 avail_mem=44.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.76it/s]Capturing num tokens (num_tokens=112 avail_mem=44.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.76it/s]

    Capturing num tokens (num_tokens=96 avail_mem=44.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.76it/s] Capturing num tokens (num_tokens=80 avail_mem=44.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.76it/s]Capturing num tokens (num_tokens=80 avail_mem=44.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=64 avail_mem=44.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=48 avail_mem=44.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=32 avail_mem=44.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=28 avail_mem=44.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=24 avail_mem=44.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=20 avail_mem=44.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=20 avail_mem=44.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=16 avail_mem=44.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=12 avail_mem=44.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=8 avail_mem=44.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.93it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=44.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=4 avail_mem=44.93 GB): 100%|██████████| 58/58 [00:01<00:00, 37.28it/s]


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
    Generated text:  Claire and I am the CEO of Genius Consultancy, which has been helping businesses to grow and evolve their online presence for over 10 years. I am also the CEO of Bob's Business, a consultancy that specializes in marketing, branding and customer experience. My background is in retail and I have spent many years at the top of the market with marketing teams and in strategic business development roles. I am passionate about helping people be the best they can be and I believe that by sharing my own experiences and learning, I can help other businesses and individuals to do the same.
    I'm interested in talking about how you've used your background in
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many wars to fight. He found out from the Congress that there are currently 1000 wars in the world and that the number of wars doubled every year for the next 7 years. Additionally, he found out that in the first year, 10% of the wars were won by the opposing side, and in the second year, 5% of the remaining wars were won. How many wars does the president of the United States need to fight to win a war in the first year?
    
    To determine how many wars the president of the United States needs to fight to win a war in the first
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. London C. New York D. Moscow
    Answer:
    
    A
    
    Which of the following statements about large and small molecules in life activities is incorrect?
    A. Monosaccharides, disaccharides, and polysaccharides are all macromolecules.
    B. The amount of water absorbed by a plant is always greater than the amount of water lost.
    C. In the same cell, different metabolic activities may use different small molecules.
    D. In the same cell, different metabolic activities may use the same small molecule.
    Answer:
    
    D
    
    Which of the following options uses a metaphorical expression?
    
    ===============================
    Prompt: The future of AI is
    Generated text:  promising but also potentially dangerous. Here's a look at how we should handle the hype...
    
    The rise of artificial intelligence (AI) in the workforce is one of the most exciting developments that could fundamentally change the way we work and how we interact with each other. This artificial intelligence is being used for everything from personal assistants to software that can design and improve healthcare. The technology has the potential to revolutionize business and transform how we live, learn, and work. While there are many benefits to AI, it can also have unintended consequences.
    
    One of the most critical challenges with AI is its potential to replace human workers. As more jobs are automated


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also home to the Louvre Museum, the most famous art museum in the world, and the Notre-Dame Cathedral, a stunning Gothic structure. Paris is a bustling metropolis with a diverse population and a rich cultural heritage, making it a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest and most prestigious fashion events in the world. Overall, Paris is a city that has a unique blend of history, culture, and modernity.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is likely to become more prevalent in various industries, with automation becoming more prevalent in manufacturing, transportation, and customer service. This will lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. Enhanced human intelligence: AI is likely to continue to improve its ability to understand and interpret human language, leading to more sophisticated forms of artificial intelligence. This will enable AI to better understand and respond to human emotions and behaviors, leading to more empathetic and compassionate AI.
    
    3. AI ethics and privacy: As AI becomes more prevalent,
    


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
    Generated text:  [Name]. I have a passion for [Subject], and I'm constantly learning and growing. I love to explore new ideas and techniques, and I'm always looking for ways to improve my skills and knowledge. I enjoy collaborating with others, and I'm excited to learn and contribute to any project or discussion I get my hands on. And, of course, I'm always looking for new things to do, so there's always something to be learned and something to try. So, how would you describe yourself? [Name], feel free to share a bit more about yourself. [Name] [Introductory sentence] [Tell a funny
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A summary of Paris, the largest city in France and the capital of the country, is to be found in the city's name. The city is the administrative and cultural center of France, and is located on the Mediterranean coast. It has a long history and an important place in world culture.
    
    For example, many important works of literature, music, and art are created here. The city is also a hub for business and finance, and is home to major shopping centers and institutions such as the Louvre Museum and the Paris Opera. Despite its size, Paris remains a densely populated and vibrant city with a thriving arts and entertainment
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of emerging trends, including:
    
    1. Advances in machine learning and deep learning: AI will continue to be powered by advancements in machine learning and deep learning, which will allow it to learn and improve on complex tasks more quickly and accurately than ever before.
    
    2. The rise of AI in healthcare: AI will be increasingly used in healthcare to help diagnose and treat diseases, predict patient outcomes, and improve patient care. As AI becomes more integrated into healthcare, there may be a shift away from the traditional clinical models of patient care and towards more personalized and targeted approaches.
    
    3. The convergence of AI and robotics:


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

     am

     a

     [

    Job

     Title

    ]

     at

     [

    Company

    ].

     I

     have

     always

     been

     passionate

     about

     learning

     new

     things

     and

     taking

     on

     new

     challenges

    .

     My

     work

     ethic

     is

     unw

    av

    ering

    ,

     and

     I

     strive

     to

     always

     push

     myself

     to

     the

     limits

    .

     Whether

     it

    's

     delivering

     exceptional

     customer

     service

     or

     tackling

     complex

     technical

     problems

    ,

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    .

     I

    'm

     known

     for

     my

     strong

     communication

     skills

    ,

     and

     I

     enjoy

     mentoring

     others

     to

     help

     them

     grow

     too

    .

     I

    'm

     a

     confident

     and

     outgoing

     person

    ,

     always

     looking

     for

     ways

     to

     inspire

     and

     motivate

     others

    .

     Thank

     you

     for

     considering

     me

     for

     a

     job

    .

     [

    Name

    ]

     [

    Company

     Name

    ]

     You

     have

     been

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     sprawling

     and

     bustling

     met

    ropolis

     with

     a

     rich

     history

    ,

     architecture

    ,

     and

     vibrant

     culture

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

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Lou

    vre

     Museum

    .

     The

     city

     also

     hosts

     numerous

     international

     festivals

     and

     events

     throughout

     the

     year

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     drawing

     visitors

     from

     around

     the

     world

     to

     explore

     its

     rich

     cultural

     and

     historical

     heritage

    .

     Its

     status

     as

     the

     world

    's

     most

     populous

     city

     is

     reflected

     in

     its

     famous

     "

    T

    ango

    "

     and

     its

     annual

     "

    F

    ête

     de

     la

     musique

     française

    "

     (

    French

     Music

     Festival

    ).

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     that

     has

     played

     a

     crucial

     role

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     changing

    ,

     with

     new

     trends

     and

     advancements

     expected

     to

     emerge

     in

     the

     years

     ahead

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     transparency

    :

     As

     concerns

     about

     AI

    's

     impact

     on

     society

     and

     the

     environment

     continue

     to

     grow

    ,

     there

     is

     a

     growing

     focus

     on

     developing

     AI

     that

     is

     more

     transparent

     and

     accountable

    .

     This

     could

     mean

     increasing

     the

     amount

     of

     data

     that

     is

     collected

     and

     used

     in

     AI

     projects

    ,

     as

     well

     as

     implementing

     more

     ethical

     guidelines

     and

     standards

    .
    


    2

    .

     Greater

     integration

     with

     other

     technologies

    :

     AI

     is

     currently

     only

     able

     to

     be

     used

     in

     a

     few

     specific

     domains

    ,

     such

     as

     finance

     or

     healthcare

    .

     However

    ,

     as

     technology

     advances

    ,

     it

    



```python
llm.shutdown()
```
