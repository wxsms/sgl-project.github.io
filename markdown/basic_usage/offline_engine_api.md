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

    [2026-03-04 05:27:19] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 05:27:19] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 05:27:19] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-04 05:27:21] INFO server_args.py:1975: Attention backend not specified. Use fa3 backend by default.


    [2026-03-04 05:27:21] INFO server_args.py:3066: Set soft_watchdog_timeout since in CI


    [2026-03-04 05:27:21] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=781970694, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.52it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=56.81 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=56.81 GB):   5%|▌         | 1/20 [00:00<00:02,  6.55it/s]Capturing batches (bs=120 avail_mem=56.71 GB):   5%|▌         | 1/20 [00:00<00:02,  6.55it/s]Capturing batches (bs=112 avail_mem=56.71 GB):   5%|▌         | 1/20 [00:00<00:02,  6.55it/s]

    Capturing batches (bs=104 avail_mem=56.70 GB):   5%|▌         | 1/20 [00:00<00:02,  6.55it/s]Capturing batches (bs=96 avail_mem=56.70 GB):   5%|▌         | 1/20 [00:00<00:02,  6.55it/s] Capturing batches (bs=88 avail_mem=56.70 GB):   5%|▌         | 1/20 [00:00<00:02,  6.55it/s]Capturing batches (bs=88 avail_mem=56.70 GB):  30%|███       | 6/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=80 avail_mem=56.70 GB):  30%|███       | 6/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=72 avail_mem=56.70 GB):  30%|███       | 6/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=64 avail_mem=56.70 GB):  30%|███       | 6/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=56 avail_mem=56.70 GB):  30%|███       | 6/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=56 avail_mem=56.70 GB):  50%|█████     | 10/20 [00:00<00:00, 30.11it/s]Capturing batches (bs=48 avail_mem=56.70 GB):  50%|█████     | 10/20 [00:00<00:00, 30.11it/s]

    Capturing batches (bs=40 avail_mem=56.69 GB):  50%|█████     | 10/20 [00:00<00:00, 30.11it/s]Capturing batches (bs=32 avail_mem=56.69 GB):  50%|█████     | 10/20 [00:00<00:00, 30.11it/s]Capturing batches (bs=24 avail_mem=56.69 GB):  50%|█████     | 10/20 [00:00<00:00, 30.11it/s]Capturing batches (bs=24 avail_mem=56.69 GB):  70%|███████   | 14/20 [00:00<00:00, 15.45it/s]Capturing batches (bs=16 avail_mem=56.69 GB):  70%|███████   | 14/20 [00:00<00:00, 15.45it/s]Capturing batches (bs=12 avail_mem=56.69 GB):  70%|███████   | 14/20 [00:00<00:00, 15.45it/s]Capturing batches (bs=8 avail_mem=56.69 GB):  70%|███████   | 14/20 [00:00<00:00, 15.45it/s] Capturing batches (bs=8 avail_mem=56.69 GB):  85%|████████▌ | 17/20 [00:00<00:00, 17.78it/s]Capturing batches (bs=4 avail_mem=56.69 GB):  85%|████████▌ | 17/20 [00:00<00:00, 17.78it/s]

    Capturing batches (bs=2 avail_mem=56.69 GB):  85%|████████▌ | 17/20 [00:00<00:00, 17.78it/s]Capturing batches (bs=1 avail_mem=56.69 GB):  85%|████████▌ | 17/20 [00:00<00:00, 17.78it/s]Capturing batches (bs=1 avail_mem=56.69 GB): 100%|██████████| 20/20 [00:01<00:00, 19.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:14,  2.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:14,  2.36s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:57,  1.03s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:57,  1.03s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:57,  1.03s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:22,  2.39it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:10,  4.93it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:10,  4.93it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:10,  4.93it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:10,  4.93it/s]

    Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:10,  4.93it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:05,  8.93it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:05,  8.93it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:05,  8.93it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:05,  8.93it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:05,  8.93it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]

    Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:02<00:03, 13.27it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:03<00:03, 13.27it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:03<00:01, 22.50it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]

    Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 29.67it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]

    Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=20):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=16):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=12):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s]Compiling num tokens (num_tokens=8):  74%|███████▍  | 43/58 [00:03<00:00, 46.77it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 68.86it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 68.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=53.99 GB):   2%|▏         | 1/58 [00:00<00:13,  4.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.00 GB):   2%|▏         | 1/58 [00:00<00:13,  4.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.00 GB):   3%|▎         | 2/58 [00:00<00:08,  6.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.00 GB):   3%|▎         | 2/58 [00:00<00:08,  6.41it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=59.00 GB):   5%|▌         | 3/58 [00:00<00:07,  7.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.00 GB):   5%|▌         | 3/58 [00:00<00:07,  7.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.00 GB):   7%|▋         | 4/58 [00:00<00:07,  6.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.01 GB):   7%|▋         | 4/58 [00:00<00:07,  6.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.01 GB):   9%|▊         | 5/58 [00:00<00:07,  6.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.99 GB):   9%|▊         | 5/58 [00:00<00:07,  6.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.99 GB):  10%|█         | 6/58 [00:00<00:07,  6.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.07 GB):  10%|█         | 6/58 [00:00<00:07,  6.85it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.07 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.07 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.07 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.99 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.99 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.13 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.13 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.99 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.20it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.89 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.89 GB):  21%|██        | 12/58 [00:01<00:04,  9.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.19 GB):  21%|██        | 12/58 [00:01<00:04,  9.45it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.19 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.98 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.25 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.25 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.25 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.47it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.97 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.97 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.97 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.30 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.85it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.30 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.30 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.94 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.94 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.23it/s]Capturing num tokens (num_tokens=960 avail_mem=58.37 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.23it/s] Capturing num tokens (num_tokens=896 avail_mem=58.96 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.23it/s]

    Capturing num tokens (num_tokens=896 avail_mem=58.96 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.41it/s]Capturing num tokens (num_tokens=832 avail_mem=58.95 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.41it/s]Capturing num tokens (num_tokens=768 avail_mem=58.43 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.41it/s]Capturing num tokens (num_tokens=768 avail_mem=58.43 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.54it/s]Capturing num tokens (num_tokens=704 avail_mem=58.96 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.54it/s]Capturing num tokens (num_tokens=640 avail_mem=58.46 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.54it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.46 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.69it/s]Capturing num tokens (num_tokens=576 avail_mem=58.45 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.69it/s]Capturing num tokens (num_tokens=512 avail_mem=58.93 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.69it/s]Capturing num tokens (num_tokens=512 avail_mem=58.93 GB):  50%|█████     | 29/58 [00:02<00:01, 14.97it/s]Capturing num tokens (num_tokens=480 avail_mem=58.49 GB):  50%|█████     | 29/58 [00:02<00:01, 14.97it/s]Capturing num tokens (num_tokens=448 avail_mem=58.95 GB):  50%|█████     | 29/58 [00:02<00:01, 14.97it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.95 GB):  53%|█████▎    | 31/58 [00:02<00:01, 14.64it/s]Capturing num tokens (num_tokens=416 avail_mem=58.51 GB):  53%|█████▎    | 31/58 [00:02<00:01, 14.64it/s]Capturing num tokens (num_tokens=384 avail_mem=58.94 GB):  53%|█████▎    | 31/58 [00:02<00:01, 14.64it/s]Capturing num tokens (num_tokens=384 avail_mem=58.94 GB):  57%|█████▋    | 33/58 [00:02<00:01, 15.75it/s]Capturing num tokens (num_tokens=352 avail_mem=58.54 GB):  57%|█████▋    | 33/58 [00:02<00:01, 15.75it/s]Capturing num tokens (num_tokens=320 avail_mem=58.93 GB):  57%|█████▋    | 33/58 [00:03<00:01, 15.75it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.93 GB):  60%|██████    | 35/58 [00:03<00:01, 16.35it/s]Capturing num tokens (num_tokens=288 avail_mem=58.57 GB):  60%|██████    | 35/58 [00:03<00:01, 16.35it/s]Capturing num tokens (num_tokens=256 avail_mem=58.56 GB):  60%|██████    | 35/58 [00:03<00:01, 16.35it/s]Capturing num tokens (num_tokens=256 avail_mem=58.56 GB):  64%|██████▍   | 37/58 [00:03<00:01, 15.74it/s]Capturing num tokens (num_tokens=240 avail_mem=58.93 GB):  64%|██████▍   | 37/58 [00:03<00:01, 15.74it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  64%|██████▍   | 37/58 [00:03<00:01, 15.74it/s]Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=208 avail_mem=58.92 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=192 avail_mem=58.68 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=176 avail_mem=58.91 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=176 avail_mem=58.91 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.37it/s]Capturing num tokens (num_tokens=160 avail_mem=58.91 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.37it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.67 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.37it/s]Capturing num tokens (num_tokens=128 avail_mem=58.90 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.37it/s]Capturing num tokens (num_tokens=128 avail_mem=58.90 GB):  78%|███████▊  | 45/58 [00:03<00:00, 18.71it/s]Capturing num tokens (num_tokens=112 avail_mem=58.70 GB):  78%|███████▊  | 45/58 [00:03<00:00, 18.71it/s]Capturing num tokens (num_tokens=96 avail_mem=58.89 GB):  78%|███████▊  | 45/58 [00:03<00:00, 18.71it/s] Capturing num tokens (num_tokens=80 avail_mem=58.89 GB):  78%|███████▊  | 45/58 [00:03<00:00, 18.71it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.89 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.77it/s]Capturing num tokens (num_tokens=64 avail_mem=58.89 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.77it/s]Capturing num tokens (num_tokens=48 avail_mem=58.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.77it/s]Capturing num tokens (num_tokens=32 avail_mem=58.88 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.77it/s]Capturing num tokens (num_tokens=32 avail_mem=58.88 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.66it/s]Capturing num tokens (num_tokens=28 avail_mem=58.87 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.66it/s]Capturing num tokens (num_tokens=24 avail_mem=58.86 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.66it/s]Capturing num tokens (num_tokens=20 avail_mem=58.76 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.66it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.76 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.73it/s]Capturing num tokens (num_tokens=16 avail_mem=58.85 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.73it/s]Capturing num tokens (num_tokens=12 avail_mem=58.85 GB):  93%|█████████▎| 54/58 [00:04<00:00, 23.73it/s]Capturing num tokens (num_tokens=8 avail_mem=58.84 GB):  93%|█████████▎| 54/58 [00:04<00:00, 23.73it/s] Capturing num tokens (num_tokens=8 avail_mem=58.84 GB):  98%|█████████▊| 57/58 [00:04<00:00, 22.35it/s]Capturing num tokens (num_tokens=4 avail_mem=58.75 GB):  98%|█████████▊| 57/58 [00:04<00:00, 22.35it/s]Capturing num tokens (num_tokens=4 avail_mem=58.75 GB): 100%|██████████| 58/58 [00:04<00:00, 14.00it/s]


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
    Generated text:  Matt. I'm 17, and I live in Los Angeles. I'm a science enthusiast. I play video games, and I have a 3D printer. I also have a pet dog named Oreo.
    How do you plan your day?
    As a science enthusiast, my day would typically begin with doing homework. I like to try to do so creatively, often getting my hands dirty. Then, I could spend some time reading a good book or engaging in some serious debate about the future of science.
    After that, I might spend some time training my 3D printer and experimenting with some new science experiments. I also
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. 
    
    Does this mean that the president of the United States is a man? To determine if the president of the United States is a man, we need to consider the definition of the position. The position mentioned is that of President of the United States, which is a position held by the President of the United States. The President of the United States is the head of the executive branch of the federal government of the United States.
    
    A presidential candidate is a person who seeks to be elected to hold the office of President. In the United States, the President is elected through a process that includes the election of electors, who are
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 对
    B. 错
    答案:
    
    A
    
    A. exercise B. eat C. play D. watch
    答案:
    
    C
    
    The local government has made a lot of ________ to the city government in order to solve the traffic problem.
    A. contracts
    B. investments
    C. measures
    D. relationships
    答案:
    
    C
    
    According to the passage, the debate is about ________.
    A. the success of the civil rights movement
    B. the effectiveness of the civil rights movement
    C. the future of the civil rights movement
    D. the future of the civil rights movement
    答案:
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the individual. This is the finding of a new report released by the World Economic Forum. The report, titled "Generation AI", examines the future of artificial intelligence and its role in shaping the future of work.
    This report, based on a survey of 2, 343 people and 31 countries, concludes that as AI develops, it will increasingly require the participation of people who work in organizations, rather than just hiring AI professionals.
    The report asserts that when people join an organization, they will be responsible for developing and implementing AI. In the future, this will become even more important, as more and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or profession]. I enjoy [insert a brief description of your hobbies or interests]. I'm always looking for new experiences and learning new things. What's your favorite hobby or activity? I love [insert a brief description of your favorite activity or hobby]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or movie? I love [insert a brief description of your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. It is also known for its fashion industry, with many famous designers and boutiques located in the city. The city is home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is a popular destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in our daily lives. This could include the development of more efficient and cost-effective manufacturing processes, the automation of customer service and support, and the use of AI to assist with tasks that are currently performed by humans.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see increased concerns about privacy and security
    


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
    Generated text:  Emily Thompson, and I am a 30-year-old freelance writer and editor. I have a passion for crafting stories and helping people express themselves through words. I love to travel and explore new places, and I have always been fascinated by the stories that come from unexpected places. I am always looking for new ways to connect with people and help them find their voice. I am a strong believer in the power of the written word and believe that all stories have the power to shape our lives in some way. So, if you ever need a little help with your writing or need to start an exciting journey, I'm your guy! Have
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Chapelle-à-France".
    The statement is accurate and concise. The capital city of France is Paris, also known as "La Chapelle-à-France." Paris is often referred to as the "City of Light" and is one of the most populous cities in the world, with over 2.7 million inhabitants. It is a UNESCO World Heritage site and is home to the Louvre Museum, one of the largest and most iconic art museums in the world. Paris is also known for its distinctive architecture, including the Eiffel Tower and the Notre-Dame Cathedral. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly exciting, and there are several areas that are likely to see significant advancements and shifts in the next decade. Here are some of the most likely trends in AI that we can expect:
    
    1. Machine Learning: The most common type of AI currently used is machine learning, which involves training computer programs to perform tasks by analyzing data. The use of machine learning is expected to continue to grow, as more complex tasks can now be automated through algorithms that can learn from data.
    
    2. Deep Learning: Deep learning is a specific type of machine learning that involves using multiple layers of neural networks to extract features from data. This type of AI is expected


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

    name

    ].

     I

     am

     an

     intro

    verted

     personality

    ,

     a

     patient

    ,

     and

     someone

     who

     values

     simplicity

     and

     harmony

    .

     I

     am

     a

     writer

     and

     a

     philosopher

    .

     My

     main

     interests

     include

     nature

    ,

     books

    ,

     and

     meditation

    .

     I

     love

     to

     spend

     time

     in

     nature

    ,

     reading

     books

    ,

     and

     engaging

     in

     creative

     activities

    .

     I

    'm

     also

     interested

     in

     investing

     in

     sustainable

     products

     and

     traveling

     to

     different

     places

     to

     explore

     new

     cultures

    .

     How

     would

     you

     like

     to

     meet

     me

    ?

     I

     look

     forward

     to

     our

     conversation

    !

     [

    Name

    ]

    ...

     [

    Brief

    ly

     describe

     the

     role

     or

     purpose

     of

     the

     character

    ].

     The

     character

     is

     a

     [

    insert

     occupation

     or

     profession

    ]

     who

     is

     passionate

     about

     [

    insert

     their

     profession

     or

     hobby

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

    ,

     with

     a

     population

     of

     over

     

    2

     million

     people

    ,

     and

     is

     the

     capital

     of

     France

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     in

     the

     west

     of

     the

     country

     and

     is

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     cuisine

    .

     The

     city

     is

     home

     to

     many

     famous

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

     It

     is

     also

     one

     of

     the

     most

     tourist

    -friendly

     cities

     in

     the

     world

     and

     is

     a

     popular

     destination

     for

     visitors

     to

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     we

     are

     currently

     witnessing

     a

     number

     of

     exciting

     developments

     that

     promise

     to

     transform

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     the

     world

     around

     us

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     personal

    ization

    :

     With

     the

     increasing

     availability

     of

     data

    ,

     AI

     systems

     are

     becoming

     more

     adept

     at

     understanding

     and

     analyzing

     personal

     characteristics

    ,

     preferences

    ,

     and

     behaviors

    .

     This

     leads

     to more

     personalized

     experiences

    ,

     from

     personalized

     advertisements

     to

     tailored

     product

     recommendations

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     is

     already

     being

     used

     to

     create

     autonomous

     vehicles

     that

     can

     navigate

     roads

     and

     handle

     unexpected

     situations

     like

     accidents

     or

     road

    blocks

    .

     As

     the

     technology

     improves

    ,

     we

     may

     see

     more

     fully

    -aut

    onomous

     vehicles

    



```python
llm.shutdown()
```
