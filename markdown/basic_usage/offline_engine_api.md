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

    [2026-03-04 03:06:45] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 03:06:45] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 03:06:45] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-04 03:06:47] INFO server_args.py:1976: Attention backend not specified. Use fa3 backend by default.


    [2026-03-04 03:06:47] INFO server_args.py:3067: Set soft_watchdog_timeout since in CI


    [2026-03-04 03:06:47] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=734479509, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=55.07 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=55.07 GB):   5%|▌         | 1/20 [00:00<00:03,  6.22it/s]Capturing batches (bs=120 avail_mem=54.97 GB):   5%|▌         | 1/20 [00:00<00:03,  6.22it/s]Capturing batches (bs=112 avail_mem=54.97 GB):   5%|▌         | 1/20 [00:00<00:03,  6.22it/s]

    Capturing batches (bs=104 avail_mem=54.97 GB):   5%|▌         | 1/20 [00:00<00:03,  6.22it/s]Capturing batches (bs=96 avail_mem=54.97 GB):   5%|▌         | 1/20 [00:00<00:03,  6.22it/s] Capturing batches (bs=96 avail_mem=54.97 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=88 avail_mem=54.97 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=80 avail_mem=54.97 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=72 avail_mem=54.97 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=64 avail_mem=54.97 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=56 avail_mem=54.97 GB):  25%|██▌       | 5/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=56 avail_mem=54.97 GB):  50%|█████     | 10/20 [00:00<00:00, 30.57it/s]Capturing batches (bs=48 avail_mem=54.96 GB):  50%|█████     | 10/20 [00:00<00:00, 30.57it/s]Capturing batches (bs=40 avail_mem=54.96 GB):  50%|█████     | 10/20 [00:00<00:00, 30.57it/s]

    Capturing batches (bs=32 avail_mem=54.96 GB):  50%|█████     | 10/20 [00:00<00:00, 30.57it/s]Capturing batches (bs=24 avail_mem=54.96 GB):  50%|█████     | 10/20 [00:00<00:00, 30.57it/s]Capturing batches (bs=16 avail_mem=54.96 GB):  50%|█████     | 10/20 [00:00<00:00, 30.57it/s]Capturing batches (bs=16 avail_mem=54.96 GB):  75%|███████▌  | 15/20 [00:00<00:00, 30.62it/s]Capturing batches (bs=12 avail_mem=54.96 GB):  75%|███████▌  | 15/20 [00:00<00:00, 30.62it/s]Capturing batches (bs=8 avail_mem=54.96 GB):  75%|███████▌  | 15/20 [00:00<00:00, 30.62it/s] Capturing batches (bs=4 avail_mem=54.96 GB):  75%|███████▌  | 15/20 [00:00<00:00, 30.62it/s]Capturing batches (bs=2 avail_mem=54.96 GB):  75%|███████▌  | 15/20 [00:00<00:00, 30.62it/s]Capturing batches (bs=1 avail_mem=54.95 GB):  75%|███████▌  | 15/20 [00:00<00:00, 30.62it/s]

    Capturing batches (bs=1 avail_mem=54.95 GB): 100%|██████████| 20/20 [00:00<00:00, 35.96it/s]Capturing batches (bs=1 avail_mem=54.95 GB): 100%|██████████| 20/20 [00:00<00:00, 30.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:10,  2.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:10,  2.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:10,  2.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:10,  2.30s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s] 

    Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 14.61it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:02<00:01, 18.91it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:02<00:01, 18.91it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:02<00:01, 18.91it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:02<00:01, 18.91it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:02<00:01, 18.91it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:02<00:01, 18.91it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 22.92it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 22.92it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 22.92it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 22.92it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 22.92it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 22.92it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:02<00:00, 27.09it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:02<00:00, 27.09it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.09it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.09it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.09it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.09it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 30.63it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 30.63it/s]

    Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 30.63it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 30.63it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 30.63it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 30.63it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.99it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 42.56it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 42.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=52.57 GB):   2%|▏         | 1/58 [00:00<00:13,  4.21it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.54 GB):   2%|▏         | 1/58 [00:00<00:13,  4.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.54 GB):   2%|▏         | 1/58 [00:00<00:13,  4.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.54 GB):   5%|▌         | 3/58 [00:00<00:05,  9.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.53 GB):   5%|▌         | 3/58 [00:00<00:05,  9.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.54 GB):   5%|▌         | 3/58 [00:00<00:05,  9.53it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.53 GB):   5%|▌         | 3/58 [00:00<00:05,  9.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.53 GB):  10%|█         | 6/58 [00:00<00:04, 12.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.53 GB):  10%|█         | 6/58 [00:00<00:04, 12.12it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.53 GB):  10%|█         | 6/58 [00:00<00:04, 12.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.53 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.96 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.50 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.52it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=52.50 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.03 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.48 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.48 GB):  21%|██        | 12/58 [00:01<00:03, 13.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.53 GB):  21%|██        | 12/58 [00:01<00:03, 13.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.48 GB):  21%|██        | 12/58 [00:01<00:03, 13.01it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.48 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.48 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.10 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.10 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.47 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.12 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.14it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=52.12 GB):  31%|███       | 18/58 [00:01<00:02, 16.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.46 GB):  31%|███       | 18/58 [00:01<00:02, 16.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.46 GB):  31%|███       | 18/58 [00:01<00:02, 16.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.16 GB):  31%|███       | 18/58 [00:01<00:02, 16.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.16 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.08it/s]Capturing num tokens (num_tokens=960 avail_mem=52.45 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.08it/s] Capturing num tokens (num_tokens=896 avail_mem=52.20 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.08it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.44 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.08it/s]Capturing num tokens (num_tokens=832 avail_mem=52.44 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.00it/s]Capturing num tokens (num_tokens=768 avail_mem=52.44 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.00it/s]Capturing num tokens (num_tokens=704 avail_mem=52.46 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.00it/s]Capturing num tokens (num_tokens=640 avail_mem=52.43 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.00it/s]Capturing num tokens (num_tokens=640 avail_mem=52.43 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=576 avail_mem=52.42 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.65it/s]

    Capturing num tokens (num_tokens=512 avail_mem=52.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=480 avail_mem=52.42 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=480 avail_mem=52.42 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.79it/s]Capturing num tokens (num_tokens=448 avail_mem=52.26 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.79it/s]Capturing num tokens (num_tokens=416 avail_mem=52.42 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.79it/s]Capturing num tokens (num_tokens=384 avail_mem=52.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.79it/s]Capturing num tokens (num_tokens=352 avail_mem=52.40 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.79it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]Capturing num tokens (num_tokens=320 avail_mem=52.39 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]Capturing num tokens (num_tokens=288 avail_mem=52.37 GB):  59%|█████▊    | 34/58 [00:02<00:00, 25.16it/s]Capturing num tokens (num_tokens=256 avail_mem=52.32 GB):  59%|█████▊    | 34/58 [00:02<00:00, 25.16it/s]Capturing num tokens (num_tokens=240 avail_mem=52.29 GB):  59%|█████▊    | 34/58 [00:02<00:00, 25.16it/s]Capturing num tokens (num_tokens=240 avail_mem=52.29 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.26it/s]Capturing num tokens (num_tokens=224 avail_mem=52.28 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.26it/s]Capturing num tokens (num_tokens=208 avail_mem=52.32 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.26it/s]Capturing num tokens (num_tokens=192 avail_mem=52.32 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.26it/s]

    Capturing num tokens (num_tokens=176 avail_mem=52.25 GB):  66%|██████▌   | 38/58 [00:02<00:00, 27.26it/s]Capturing num tokens (num_tokens=176 avail_mem=52.25 GB):  72%|███████▏  | 42/58 [00:02<00:00, 29.77it/s]Capturing num tokens (num_tokens=160 avail_mem=52.24 GB):  72%|███████▏  | 42/58 [00:02<00:00, 29.77it/s]Capturing num tokens (num_tokens=144 avail_mem=52.25 GB):  72%|███████▏  | 42/58 [00:02<00:00, 29.77it/s]Capturing num tokens (num_tokens=128 avail_mem=52.25 GB):  72%|███████▏  | 42/58 [00:02<00:00, 29.77it/s]Capturing num tokens (num_tokens=112 avail_mem=52.26 GB):  72%|███████▏  | 42/58 [00:02<00:00, 29.77it/s]Capturing num tokens (num_tokens=112 avail_mem=52.26 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.11it/s]Capturing num tokens (num_tokens=96 avail_mem=52.26 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.11it/s] Capturing num tokens (num_tokens=80 avail_mem=52.25 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.11it/s]Capturing num tokens (num_tokens=64 avail_mem=52.23 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.11it/s]Capturing num tokens (num_tokens=48 avail_mem=52.24 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.11it/s]

    Capturing num tokens (num_tokens=32 avail_mem=52.23 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.11it/s]Capturing num tokens (num_tokens=32 avail_mem=52.23 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.05it/s]Capturing num tokens (num_tokens=28 avail_mem=52.22 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.05it/s]Capturing num tokens (num_tokens=24 avail_mem=52.21 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.05it/s]Capturing num tokens (num_tokens=20 avail_mem=52.22 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.05it/s]Capturing num tokens (num_tokens=16 avail_mem=52.19 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.05it/s]Capturing num tokens (num_tokens=16 avail_mem=52.19 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=12 avail_mem=52.21 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=8 avail_mem=52.20 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.18it/s] Capturing num tokens (num_tokens=4 avail_mem=52.20 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.18it/s]

    Capturing num tokens (num_tokens=4 avail_mem=52.20 GB): 100%|██████████| 58/58 [00:02<00:00, 22.07it/s]


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
    Generated text:  Mary Jane and I am a 28 year old mom. I am in college. I also have a 13 year old daughter. I am constantly trying to stop myself from starting a family with my 13 year old. I believe that our relationship is going well and that I am doing my best to support the needs of my daughter. How can I convince my husband to allow me to have children with my daughter?
    It sounds like you are dealing with a complex situation where your commitment to supporting your daughter is in conflict with your desire to have a family. Here are some steps you can take to try to convince your husband
    ===============================
    Prompt: The president of the United States is
    Generated text:  a government official. His or her primary responsibility is to lead the federal government. The president is usually the head of the executive branch of the federal government. The vice president also serves as the head of the executive branch. The president, vice president, and members of Congress together form the legislative branch. The president can pardon people who are convicted of serious crimes.
    The president has several key powers:
    - appointing federal judges
    - appointing ambassadors, ambassadors' associates, and other federal officials
    - declaring war
    - appointing federal judges who are not part of the executive branch
    - declaring a state of war
    - appointing military
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is located on the banks of the River Seine, a very important river in France. The river flows through many streets and buildings. There are many people in Paris. Many of them are young and some of them are old. They all like Paris very much. There are many parks in Paris. There are many bridges over the river. There are many tall buildings around Paris. There are many people in Paris.
    
    1. Where is the capital of France? (1 point)
    
    A. Italy
    
    B. England
    
    C. Spain
    
    D. Paris
    
    2. What is the river called? (1 point)
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of human beings. While we might be able to make decisions in areas such as healthcare and finance, we are currently far from being able to fully understand and make sense of the general trends and processes in the world. AI has the potential to help people make better decisions and improve the world around us, but it is also a complex field with many different subfields, some of which have already been developed and others that are still in the early stages of development. Here are some of the key areas of research in AI, as well as some of the challenges that it is currently facing:
    1. Machine Learning: Machine learning is


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for [Number of Years] years and I have been driving for [Number of Months] months. I have been driving for [Number of Days] days and I have been driving for [Number of Weeks] weeks. I have been driving for [Number of Months] months and I have been driving for [Number of Weeks] weeks. I have been driving for [Number of Months] months and I have been driving for [Number of Weeks] weeks. I have been driving
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French colonialism and its role in the French Revolution and the French Revolution. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role in shaping French culture and identity. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in our daily lives. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. AI-powered healthcare: AI is already being used to improve the accuracy and speed of medical diagnosis and treatment. As AI technology continues to advance, we can expect to see even more
    


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
    Generated text:  [Name], and I am a [Age] year old female. I currently work as a [Job Title], and I enjoy [Favorite Activity/Interest] with my [Favorite Person]. I am [Background]. I am a [Current Location] citizen, and I have a [Favorite Book/TV Show] that I enjoy reading or watching, and I am [Favorite Movie/Adventure]. I am [Favorite Color] and I am always [Favorite Clothing], and I enjoy [Favorite Hobby or Exercise]. I have [Favorite Food/Drink] and I have a [Favorite Travel Destination] that I have been to, and I have
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its rich history, beautiful architecture, and vibrant culture. It is the country's largest city and serves as its political, cultural, and economic center. Paris is a popular tourist destination and a major hub for international business and diplomacy. Its many landmarks, including the Eiffel Tower and Louvre Museum, are a testament to its rich history and artistic significance. Additionally, Paris is known for its food scene, including its famous croissants, cheese, and other gourmet foods. Overall, Paris is a city with a unique blend of modernity and tradition, making it a beloved destination for many visitors.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, and there are many potential areas where it could evolve and become even more advanced. Here are some potential future trends that could impact AI in significant ways:
    
    1. Increased accuracy and precision: As AI systems become more advanced, they are likely to become even more precise and accurate in their decisions and predictions. This could lead to breakthroughs in areas such as medical diagnosis, fraud detection, and autonomous driving.
    
    2. Increased transparency and explainability: As AI systems become more complex, it will be important to ensure that they are transparent and explainable to humans. This will require developers to create more human-like interfaces and to produce more


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

    'm

     a

     [

    profession

     or

     type

     of

     character

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?


    As

     an

     AI

     language

     model

    ,

     I

    'm

     here

     to

     provide

     help

     and

     answer

     your

     questions

    .

     How

     can

     I

     assist

     you

     today

    ?


    Great

    ,

     that

     sounds

     like

     a

     nice

     way

     to

     start

    .

     Can

     you

     tell

     me

     a

     little

     bit

     about

     yourself

    ?

     What

     are

     some

     of

     the

     things

     you

     enjoy

     doing

     outside

     of

     your

     profession

    ?

     


    As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     personal

     interests

     or

     hobbies

     like

     humans

     do

    ,

     but

     I

    'm

     here

     to

     assist

     you

     with

     any

     questions

     you

     might

     have

    .

     How

     can

     I

     help

     you

     today

    ?

     


    I

    'm

     happy

     to

     answer

     any

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

    .

     It

     is

     known

     for

     its

     rich

     history

     and

     diverse

     culture

    ,

     with

     attractions

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     a

     center

     of

     innovation

     and

     the

     largest

     city

     in

     France

     by

     population

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

     Pont

     des

     Arts

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     food

     and

     wine

    ,

     as

     well

     as

     its

     annual

     festivals

     such

     as

     the

     Carnival

     and

     the

     Notre

    -D

    ame

     du

     L

    oir

    .

     Paris

     is

     a

     cultural

     hub

     with

     many

     museums

    ,

     theaters

    ,

     and

     galleries

    ,

     and

     is

     often

     considered

     one

     of

     the

     most

     interesting

     cities

     in

     the

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     rapid

     increase

     in

     both

     the

     sophistication

     of

     the

     AI

     systems

     and

     the

     accessibility

     of

     AI

     applications

    .

     Some

     potential

     trends

     to

     watch

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     IoT

    ,

     and

     IoT

    -based

     AI

     systems

    ,

     to

     create

     more

     complex

     and

     diverse

     AI

     systems

    .
    


    2

    .

     Improved

     and

     more

     efficient

     use

     of

     AI

     for

     tasks

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    ,

     leading

     to

     increased

     efficiency

     and

     productivity

    .
    


    3

    .

     AI

     systems

     that

     can

     learn

     and

     adapt

     to

     new

     situations

     and

     environments

    ,

     making

     them

     more

     versatile

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     impossible

    .
    


    4

    .

     Increased

     use

     of

     AI

     to

     improve

     decision

    -making

     and

     problem

    -solving

    



```python
llm.shutdown()
```
