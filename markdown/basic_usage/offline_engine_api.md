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

    [2026-03-05 06:06:13] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-05 06:06:13] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-05 06:06:13] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-05 06:06:15] INFO server_args.py:2038: Attention backend not specified. Use fa3 backend by default.


    [2026-03-05 06:06:15] INFO server_args.py:3129: Set soft_watchdog_timeout since in CI


    [2026-03-05 06:06:15] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=672178000, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.34it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.33it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=56.69 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=56.69 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]Capturing batches (bs=120 avail_mem=56.59 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]Capturing batches (bs=112 avail_mem=56.59 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]

    Capturing batches (bs=104 avail_mem=56.59 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]Capturing batches (bs=96 avail_mem=56.59 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s] Capturing batches (bs=96 avail_mem=56.59 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.49it/s]Capturing batches (bs=88 avail_mem=56.59 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.49it/s]Capturing batches (bs=80 avail_mem=56.59 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.49it/s]Capturing batches (bs=72 avail_mem=56.59 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.49it/s]Capturing batches (bs=64 avail_mem=56.59 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.49it/s]Capturing batches (bs=56 avail_mem=56.58 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.49it/s]Capturing batches (bs=56 avail_mem=56.58 GB):  50%|█████     | 10/20 [00:00<00:00, 30.73it/s]Capturing batches (bs=48 avail_mem=56.58 GB):  50%|█████     | 10/20 [00:00<00:00, 30.73it/s]Capturing batches (bs=40 avail_mem=56.58 GB):  50%|█████     | 10/20 [00:00<00:00, 30.73it/s]

    Capturing batches (bs=32 avail_mem=56.58 GB):  50%|█████     | 10/20 [00:00<00:00, 30.73it/s]Capturing batches (bs=24 avail_mem=56.58 GB):  50%|█████     | 10/20 [00:00<00:00, 30.73it/s]Capturing batches (bs=24 avail_mem=56.58 GB):  70%|███████   | 14/20 [00:00<00:00, 32.67it/s]Capturing batches (bs=16 avail_mem=56.58 GB):  70%|███████   | 14/20 [00:00<00:00, 32.67it/s]Capturing batches (bs=12 avail_mem=56.58 GB):  70%|███████   | 14/20 [00:00<00:00, 32.67it/s]Capturing batches (bs=8 avail_mem=56.58 GB):  70%|███████   | 14/20 [00:00<00:00, 32.67it/s] Capturing batches (bs=4 avail_mem=56.57 GB):  70%|███████   | 14/20 [00:00<00:00, 32.67it/s]

    Capturing batches (bs=4 avail_mem=56.57 GB):  90%|█████████ | 18/20 [00:00<00:00, 27.28it/s]Capturing batches (bs=2 avail_mem=56.57 GB):  90%|█████████ | 18/20 [00:00<00:00, 27.28it/s]Capturing batches (bs=1 avail_mem=56.57 GB):  90%|█████████ | 18/20 [00:00<00:00, 27.28it/s]Capturing batches (bs=1 avail_mem=56.57 GB): 100%|██████████| 20/20 [00:00<00:00, 27.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:07,  6.74it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 16.13it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:02<00:01, 26.65it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=28):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=24):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=20):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=16):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=12):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s]Compiling num tokens (num_tokens=8):  71%|███████   | 41/58 [00:02<00:00, 39.04it/s] 

    Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:02<00:00, 60.50it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:02<00:00, 60.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 20.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=50.90 GB):   9%|▊         | 5/58 [00:00<00:02, 22.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.89 GB):   9%|▊         | 5/58 [00:00<00:02, 22.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.89 GB):   9%|▊         | 5/58 [00:00<00:02, 22.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.89 GB):   9%|▊         | 5/58 [00:00<00:02, 22.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.89 GB):   9%|▊         | 5/58 [00:00<00:02, 22.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.89 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.89 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.12it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=50.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.87 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.87 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.85 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.83 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=960 avail_mem=50.85 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s] Capturing num tokens (num_tokens=896 avail_mem=50.84 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=832 avail_mem=50.84 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s]

    Capturing num tokens (num_tokens=768 avail_mem=50.83 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=704 avail_mem=50.83 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=704 avail_mem=50.83 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=640 avail_mem=50.83 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=576 avail_mem=50.83 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=512 avail_mem=50.82 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=480 avail_mem=50.83 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=448 avail_mem=50.83 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=416 avail_mem=50.83 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=416 avail_mem=50.83 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]Capturing num tokens (num_tokens=384 avail_mem=50.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]Capturing num tokens (num_tokens=352 avail_mem=50.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]Capturing num tokens (num_tokens=320 avail_mem=50.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]

    Capturing num tokens (num_tokens=288 avail_mem=50.81 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]Capturing num tokens (num_tokens=256 avail_mem=50.81 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]Capturing num tokens (num_tokens=240 avail_mem=50.81 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.51it/s]Capturing num tokens (num_tokens=240 avail_mem=50.81 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.27it/s]Capturing num tokens (num_tokens=224 avail_mem=50.80 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.27it/s]Capturing num tokens (num_tokens=208 avail_mem=50.80 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.27it/s]Capturing num tokens (num_tokens=192 avail_mem=50.80 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.27it/s]Capturing num tokens (num_tokens=176 avail_mem=50.79 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.27it/s]Capturing num tokens (num_tokens=160 avail_mem=50.79 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.27it/s]Capturing num tokens (num_tokens=144 avail_mem=50.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 49.27it/s]Capturing num tokens (num_tokens=144 avail_mem=50.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s]Capturing num tokens (num_tokens=128 avail_mem=50.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s]Capturing num tokens (num_tokens=112 avail_mem=50.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s]

    Capturing num tokens (num_tokens=96 avail_mem=50.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s] Capturing num tokens (num_tokens=80 avail_mem=50.74 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s]Capturing num tokens (num_tokens=64 avail_mem=50.74 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s]Capturing num tokens (num_tokens=48 avail_mem=50.73 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.66it/s]Capturing num tokens (num_tokens=48 avail_mem=50.73 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]Capturing num tokens (num_tokens=32 avail_mem=50.73 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]Capturing num tokens (num_tokens=28 avail_mem=50.73 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]Capturing num tokens (num_tokens=24 avail_mem=50.73 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]Capturing num tokens (num_tokens=20 avail_mem=50.72 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]

    Capturing num tokens (num_tokens=16 avail_mem=50.72 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]Capturing num tokens (num_tokens=12 avail_mem=50.72 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.58it/s]Capturing num tokens (num_tokens=12 avail_mem=50.72 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=8 avail_mem=50.71 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.37it/s] Capturing num tokens (num_tokens=4 avail_mem=50.71 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=4 avail_mem=50.71 GB): 100%|██████████| 58/58 [00:01<00:00, 43.34it/s]


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
    Generated text:  Tom. I have been a student at university for more than two years. There are many learning methods that I find most useful. I think it is essential to use a different learning method for each subject. I use different learning methods in different subjects, but I think that it is important to me that I learn as much as possible.
    
    There are some people who are used to the traditional way of learning. They are used to using textbooks, sitting in lectures and doing their homework. These people are much more likely to learn well if they use these methods.
    
    I prefer the traditional method of learning. I use textbooks, a computer, and the
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official and serves a four-year term. This is the first time that all U. S. presidents have served for four years. Under the 1787 Constitution, each state is required to elect a president.
    The president-elect is chosen by the people to replace the incumbent president who has been serving in that office for four years. The people choose the person they think is best suited to become president. The next person is selected by the Republican Party.
    The Republican Party nominates their candidate and they then propose the names of 23 people. The first choice is voted on by the people in their district and the second
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    
    B. London
    
    C. Rome
    
    D. Berlin
    
    To determine the capital of France, we need to consider the major cities and countries in Europe. The European Union capital, Brussels, is often mentioned as a reference point. However, the capital of France is typically referred to as Paris. 
    
    Let's verify this by checking the given options:
    
    A. Paris
    B. London
    C. Rome
    D. Berlin
    
    Based on the information provided, the capital of France is Paris. Therefore, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s one that will dramatically change the way we work and live. The ability to analyze vast amounts of data and make informed decisions is one of the greatest opportunities for humanity. But as we look to the future, there are also some important considerations that we must take into account.
    The first consideration is that AI should not be used for malicious purposes. This means that we must be careful to avoid any negative consequences of AI usage. For example, if an AI is used to deceive people or to cause harm, it is not appropriate for it to be used. We must also be mindful of the ethical implications of AI usage,


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short description of your character or personality]. And what's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity]. And what's your favorite book or movie? I love [insert a short description of your favorite book or movie]. And what's your favorite place to go? I love [insert a short description of your favorite place]. And what's your favorite color? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many famous museums, theaters, and restaurants. The city is known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is also a major center for science and technology, with many universities and research institutions. The city is also home to many cultural institutions, including the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial general intelligence: As AI continues to advance, we may see more automation in various industries, including manufacturing, transportation, and healthcare. This could lead to the development of AI that can perform tasks that were previously done by humans, such as driving cars, performing surgeries, and managing financial transactions.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. We may
    


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
    Generated text:  [Name] and I am a [career goal] enthusiast who has always been passionate about [something]. I enjoy [activities or hobbies] and [how I got into it], and I love [why I chose this path]. I have always been curious about the world, and I'm always eager to learn and explore new things. I believe that [why you believe this], and I strive to be a [role model or mentor] for others who want to pursue a similar path. I have a passion for [why you have this passion], and I am always looking for ways to contribute to society and make a positive impact. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Paris is the largest city in France, located on the western bank of the Seine river and serving as the administrative and cultural center of the country. It has a population of approximately 2.1 million people and is home to the European Parliament, the City of Paris Museum, and the Eiffel Tower. The city is famous for its iconic landmarks, such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. Paris is also a world-renowned center for art and culture, hosting numerous museums, galleries, and theaters. The French capital is known for its culinary and gastronomic delights,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve in many different directions, with a variety of potential trends shaping the technology's direction in the years to come.
    
    1. Increased Integration with Human-AI Interaction: AI is becoming more integrated with our daily lives, from interacting with voice-activated assistants to using AI to perform tasks like grocery shopping or scheduling appointments. AI is likely to become even more prevalent in the workplace, as companies look to automate processes and reduce costs.
    
    2. Enhanced Personalization: As AI becomes more advanced, it will become more effective at understanding and interpreting human behavior. This will allow for more personalized experiences, from personalized recommendations for products and services


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

     a

     [

    Occup

    ation

    ]

     [

    Your

     Occupation

    ].

     I

    'm

     happy

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Your

     Name

    ]

     [

    Your

     Name

    ].

     Hi

     there

    !

     Nice

     to

     meet

     you

    .

     I

    'm

     [

    Your

     Name

    ],

     a

     [

    Occup

    ation

    ]

     [

    Your

     Occupation

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     So

    ,

     what

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Your

     Name

    ]

     [

    Your

     Name

    ].

     Hi

     there

    !

     Nice

     to

     meet

     you

    .

     I

    'm

     [

    Your

     Name

    ],

     a

     [

    Occup

    ation

    ]

     [

    Your

     Occupation

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     iconic

     and

     historic

     city

     known

     for

     its

     intricate

     architecture

    ,

     diverse

     culture

    ,

     and

     significant

     role

     in

     European

     history

    .

     It

     is

     a

     vibrant

     met

    ropolis

     with

     a

     rich

     history

     and

     a

     reputation

     as

     a

     bustling

     hub

     of

     commerce

     and

     culture

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     light

    "

     due

     to

     its

     constant

     illumination

     by

     the

     Paris

    ian

     skyline

    ,

     and

     it

     is

     a

     world

    -ren

    owned

     tourist

     destination

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     museums

    ,

     and

     theaters

    ,

     and

     it

     is

     an

     important

     center

     for

     politics

    ,

     economics

    ,

     and

     the

     arts

    .

     Finally

    ,

     Paris

     is

     known

     for

     its

     fashion

     industry

    ,

     with

     a

     vibrant

     street

     culture

    ,

     fashion

     shows

    ,

     and

     fashion

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     potential

    .

     Here

     are

     some

     of

     the

     most

     common

     trends

     and

     developments

     that

     we

     can

     expect

     to

     see

     in

     the

     AI

     world

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     is

     being

     used

     more

     and

     more

     to

     automate

     repetitive

     and

     mundane

     tasks

    ,

     from

     manufacturing

     to

     customer

     service

    .

     This

     is

     expected

     to

     lead

     to

     increased

     efficiency

     and

     productivity

    ,

     while

     also

     creating

     new

     jobs

     and

     occupations

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     AI

    :

     There

     is

     growing

     recognition

     of

     the

     need

     for

     AI

     to

     be

     used

     responsibly

     and

     eth

    ically

    ,

     to

     ensure

     that

     it

     does

     not

     harm

     humans

     or

     the

     environment

    .

     This

     is

     expected

     to

     lead

     to

     more

     advanced

     ethical

     AI

    ,

     with

     the

     goal

     of

     creating

     AI

     that

    



```python
llm.shutdown()
```
