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

    [2026-03-05 04:32:42] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-05 04:32:42] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-05 04:32:42] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-05 04:32:44] INFO server_args.py:2038: Attention backend not specified. Use fa3 backend by default.


    [2026-03-05 04:32:44] INFO server_args.py:3129: Set soft_watchdog_timeout since in CI


    [2026-03-05 04:32:44] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=822286013, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=58.60 GB):   5%|▌         | 1/20 [00:00<00:03,  5.49it/s]Capturing batches (bs=120 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.49it/s]

    Capturing batches (bs=112 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.49it/s]Capturing batches (bs=104 avail_mem=58.50 GB):   5%|▌         | 1/20 [00:00<00:03,  5.49it/s]Capturing batches (bs=104 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 15.42it/s]Capturing batches (bs=96 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 15.42it/s] Capturing batches (bs=88 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 15.42it/s]Capturing batches (bs=80 avail_mem=58.50 GB):  20%|██        | 4/20 [00:00<00:01, 15.42it/s]

    Capturing batches (bs=80 avail_mem=58.50 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.48it/s]Capturing batches (bs=72 avail_mem=58.50 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.48it/s]Capturing batches (bs=64 avail_mem=58.50 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.48it/s]Capturing batches (bs=64 avail_mem=58.50 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.29it/s]Capturing batches (bs=56 avail_mem=58.49 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.29it/s]Capturing batches (bs=48 avail_mem=58.49 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.29it/s]

    Capturing batches (bs=40 avail_mem=58.49 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.29it/s]Capturing batches (bs=40 avail_mem=58.49 GB):  60%|██████    | 12/20 [00:00<00:00, 19.39it/s]Capturing batches (bs=32 avail_mem=58.49 GB):  60%|██████    | 12/20 [00:00<00:00, 19.39it/s]Capturing batches (bs=24 avail_mem=58.49 GB):  60%|██████    | 12/20 [00:00<00:00, 19.39it/s]Capturing batches (bs=16 avail_mem=58.49 GB):  60%|██████    | 12/20 [00:00<00:00, 19.39it/s]

    Capturing batches (bs=16 avail_mem=58.49 GB):  75%|███████▌  | 15/20 [00:00<00:00, 15.90it/s]Capturing batches (bs=12 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 15.90it/s]Capturing batches (bs=8 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:00<00:00, 15.90it/s] Capturing batches (bs=4 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:01<00:00, 15.90it/s]Capturing batches (bs=2 avail_mem=76.90 GB):  75%|███████▌  | 15/20 [00:01<00:00, 15.90it/s]Capturing batches (bs=2 avail_mem=76.90 GB):  95%|█████████▌| 19/20 [00:01<00:00, 20.65it/s]Capturing batches (bs=1 avail_mem=76.89 GB):  95%|█████████▌| 19/20 [00:01<00:00, 20.65it/s]Capturing batches (bs=1 avail_mem=76.89 GB): 100%|██████████| 20/20 [00:01<00:00, 18.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:14,  2.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:14,  2.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:14,  2.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:14,  2.37s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:02<00:02, 13.76it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]

    Compiling num tokens (num_tokens=640):  31%|███       | 18/58 [00:02<00:02, 13.76it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:02<00:01, 22.54it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]

    Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=96):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s] Compiling num tokens (num_tokens=80):  62%|██████▏   | 36/58 [00:02<00:00, 32.07it/s]Compiling num tokens (num_tokens=64):  62%|██████▏   | 36/58 [00:03<00:00, 32.07it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 48.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 18.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.38 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.38 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.37 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.37 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.37 GB):   9%|▊         | 5/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.37 GB):   9%|▊         | 5/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.37 GB):   9%|▊         | 5/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.37 GB):   9%|▊         | 5/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.32it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.36 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.31 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=960 avail_mem=72.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.73it/s] Capturing num tokens (num_tokens=960 avail_mem=72.32 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=896 avail_mem=72.32 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=832 avail_mem=72.32 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=768 avail_mem=72.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=704 avail_mem=72.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=640 avail_mem=72.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=576 avail_mem=72.30 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.33it/s]Capturing num tokens (num_tokens=576 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=512 avail_mem=72.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.45it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=448 avail_mem=72.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=416 avail_mem=72.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=384 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=352 avail_mem=72.30 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=352 avail_mem=72.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=320 avail_mem=72.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=288 avail_mem=72.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=256 avail_mem=72.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=240 avail_mem=72.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=224 avail_mem=72.28 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.17it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=208 avail_mem=72.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=192 avail_mem=72.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=176 avail_mem=72.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=160 avail_mem=72.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=144 avail_mem=72.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=144 avail_mem=72.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=128 avail_mem=72.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=112 avail_mem=72.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=96 avail_mem=72.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.94it/s] Capturing num tokens (num_tokens=80 avail_mem=72.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.94it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=64 avail_mem=72.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=48 avail_mem=72.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=32 avail_mem=72.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=28 avail_mem=72.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=24 avail_mem=72.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=20 avail_mem=72.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=16 avail_mem=72.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=16 avail_mem=72.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=12 avail_mem=72.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=8 avail_mem=72.22 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s] Capturing num tokens (num_tokens=4 avail_mem=72.22 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=4 avail_mem=72.22 GB): 100%|██████████| 58/58 [00:01<00:00, 36.64it/s]


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
    Generated text:  Roshan. I'm a big fan of film and anime. I also enjoy travel and reading. I like to travel to different locations to explore new cultures and immerse myself in the local customs. My interest in travel extends to understanding and appreciating the perspectives of different cultures.
    I often travel solo, but I also like to join groups like adventure groups, travel clubs, and photography groups. I love being alone, as it allows me to enjoy the beauty of nature and immerse myself in the cultural experiences that are unique to each location.
    In addition to travel, I enjoy reading and exploring my love of literature. I love reading
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate how many ice cream cones have been ordered. The president knows that during the last year, the number of ice cream cones ordered per person has been around 5, with a standard deviation of 3. He also knows that the total number of ice cream cones ordered in the last year was 500,000. He wants to find out how many ice cream cones on average were ordered per person.
    
    What is the president's estimated average number of ice cream cones ordered per person?
    
    To find the president's estimated average number of ice cream cones ordered per person, we need to divide the total number of ice
    ===============================
    Prompt: The capital of France is
    Generated text: : Paris
    
    Paris, the capital of France, is located on the River Seine, in the Loire Valley. It has a population of over 2 million people and is one of the most important cities in Europe. Paris is known for its beautiful architecture, iconic landmarks, and rich cultural heritage. The city is home to many world-renowned museums, theaters, and landmarks, including the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. Paris is also famous for its delicious cuisine, fashion, and art scene. It is a city that has made a significant impact on the world and continues to be
    ===============================
    Prompt: The future of AI is
    Generated text:  intrinsically linked to the future of health, yet it has not been adequately addressed in any framework. New and emerging technologies that could accelerate the pace of AI development and its deployment in health care are needed.
    Although there is increasing interest in AI for healthcare, it is difficult to identify those solutions that will reach clinical usefulness and scale. One cause is a lack of consensus on the technical requirements for effective AI. Another is the lack of experience with large-scale clinical trials. A third issue is the challenge of bridging research to clinical applications.
    AI is increasingly being deployed across the health care industry, and its deployment in the field is leading to


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
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also home to many famous French artists, writers, and musicians. It is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of contrasts, with its modern architecture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI becomes more integrated into our daily lives, we may see even more widespread adoption of AI in areas such as healthcare, finance, and transportation.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical and responsible AI. This may involve developing AI that
    


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
    Generated text:  [Your Name] and I am a 28-year-old software developer with over 5 years of experience in the field. My passion for technology and innovation has always driven me to pursue my career, and I have a knack for creating and improving systems that are both functional and aesthetically pleasing. I enjoy staying up-to-date with the latest trends and technologies in the field of software development, and I am always eager to learn from my colleagues and stay on top of the industry. Additionally, I am a strong communicator and enjoy working collaboratively with team members to achieve our goals, as well as taking on new challenges and learning from
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city in the south of the country.
    
    How many years has Paris had a monarchy? Paris has had a monarchy since 1792, when King Louis XVI ascended to the throne.
    
    How many years has Paris had a republic? Paris has had a republic since 1799, when the People’s Republic of France was established after the French Revolution.
    
    How many years has Paris had a republic since 1799? Paris has had a republic since 1799, when the People’s Republic of France was established after the French Revolution.
    
    How many years has Paris had a republic since
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be a combination of technology, innovation, and societal shifts. Here are some possible future trends in AI:
    
    1. Increased automation: As AI continues to advance, we can expect to see more automation in industries such as manufacturing, transportation, and healthcare. This will lead to increased efficiency, reduced labor costs, and improved quality of life.
    
    2. Improved personalization: AI will allow us to personalize our experiences more closely to our needs and preferences. This will create more personalized products and services, such as personalized healthcare and education.
    
    3. Autonomous vehicles: Autonomous vehicles will become more common, and we will see a significant reduction in traffic


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

    Job

     or

     profession

    ]

     in

     [

    Location

    ].

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Interest

     or

     Hobby

    ]

     and

     have

     always

     been

     motivated

     to

     achieve

     my

     goals

    .

     I

     enjoy

     making

     friends

     and

     helping

     others

    ,

     and

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    .

     What

     exc

    ites

     you

     the

     most

     about

     your

     profession

    ?

     What

     would

     you

     like

     to

     achieve

     in

     the

     next

     [

    year

    ,

     month

    ,

     week

    ,

     etc

    .]

    ?

     I

    'm

     [

    Your

     Age

    ],

     and

     I

     love

     [

    Your

     hobby

    ]

     as

     much

     as

     you

     do

    ,

     and

     I

     believe

     [

    Your

     Area

     of

     Expert

    ise

     or

     Interest

    ].

     I

    'm

     always

     here

     for

     you

    ,

     and

     I

     look

     forward

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     about

     the

     city

     of

     Paris

     is

     factual

    ,

     as

     it

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    ,

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    ,

     at

     the

     heart

     of

     the

     Paris

     region

     and

     the

     third

    -largest

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

     rich

     history

    ,

     art

    ,

     and

     culture

    ,

     and

     it

     is

     the

     birth

    place

     of

     many

     famous

     figures

     in

     European

     and

     global

     history

    .

     The

     city

     is

     also

     the

     seat

     of

     the

     President

     of

     the

     French

     Republic

     and

     a

     major

     financial

     and

     political

     center

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     automation

     and

     efficiency

    :

     AI

     is

     becoming

     increasingly

     capable

     of

     performing

     tasks

     that

     would

     typically

     require

     human

     labor

    ,

     leading

     to

     greater

     automation

     and

     efficiency

     in

     various

     industries

    .

     This

     could

     result

     in

     the

     displacement

     of

     some

     jobs

    ,

     but

     also

     the

     creation

     of

     new

     jobs

     that

     require

     AI

     skills

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     is

     becoming

     increasingly

     capable

     of

     personal

    izing

     the

     experience

     of

     users

    ,

     creating

     a

     more

     tailored

     and

     seamless

     user

     experience

    .

     This

     could

     lead

     to

     better

     customer

     satisfaction

     and

     loyalty

    .
    


    3

    .

     Improved

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     develop

     new

     treatments

     and

     improve

     existing

     ones

    ,

     leading

     to

     better

     outcomes

     for

     patients

    .

     AI

    



```python
llm.shutdown()
```
