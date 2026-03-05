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

    [2026-03-04 23:56:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 23:56:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 23:56:38] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-04 23:56:40] INFO server_args.py:1978: Attention backend not specified. Use fa3 backend by default.


    [2026-03-04 23:56:40] INFO server_args.py:3069: Set soft_watchdog_timeout since in CI


    [2026-03-04 23:56:40] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=719053243, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


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
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=61.32 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=61.32 GB):   5%|▌         | 1/20 [00:00<00:03,  6.11it/s]Capturing batches (bs=120 avail_mem=61.22 GB):   5%|▌         | 1/20 [00:00<00:03,  6.11it/s]Capturing batches (bs=112 avail_mem=61.22 GB):   5%|▌         | 1/20 [00:00<00:03,  6.11it/s]

    Capturing batches (bs=104 avail_mem=61.22 GB):   5%|▌         | 1/20 [00:00<00:03,  6.11it/s]Capturing batches (bs=96 avail_mem=61.22 GB):   5%|▌         | 1/20 [00:00<00:03,  6.11it/s] Capturing batches (bs=96 avail_mem=61.22 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.67it/s]Capturing batches (bs=88 avail_mem=61.22 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.67it/s]Capturing batches (bs=80 avail_mem=61.22 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.67it/s]Capturing batches (bs=72 avail_mem=61.22 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.67it/s]Capturing batches (bs=64 avail_mem=61.22 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.67it/s]Capturing batches (bs=64 avail_mem=61.22 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.69it/s]Capturing batches (bs=56 avail_mem=61.21 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.69it/s]

    Capturing batches (bs=48 avail_mem=61.21 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.69it/s]Capturing batches (bs=40 avail_mem=61.21 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.69it/s]Capturing batches (bs=32 avail_mem=61.21 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.69it/s]Capturing batches (bs=32 avail_mem=61.21 GB):  65%|██████▌   | 13/20 [00:00<00:00, 28.64it/s]Capturing batches (bs=24 avail_mem=61.21 GB):  65%|██████▌   | 13/20 [00:00<00:00, 28.64it/s]Capturing batches (bs=16 avail_mem=61.21 GB):  65%|██████▌   | 13/20 [00:00<00:00, 28.64it/s]Capturing batches (bs=12 avail_mem=61.21 GB):  65%|██████▌   | 13/20 [00:00<00:00, 28.64it/s]Capturing batches (bs=8 avail_mem=61.21 GB):  65%|██████▌   | 13/20 [00:00<00:00, 28.64it/s] 

    Capturing batches (bs=8 avail_mem=61.21 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.94it/s]Capturing batches (bs=4 avail_mem=61.20 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.94it/s]Capturing batches (bs=2 avail_mem=61.20 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.94it/s]Capturing batches (bs=1 avail_mem=61.20 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.94it/s]Capturing batches (bs=1 avail_mem=61.20 GB): 100%|██████████| 20/20 [00:00<00:00, 28.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:11,  2.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:11,  2.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:11,  2.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:11,  2.31s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:07,  6.55it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 15.66it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:02<00:01, 25.91it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=28):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=24):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=20):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=16):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]Compiling num tokens (num_tokens=12):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s]

    Compiling num tokens (num_tokens=8):  71%|███████   | 41/58 [00:02<00:00, 37.98it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:02<00:00, 58.80it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:02<00:00, 58.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 19.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.80 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.80 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.80 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.80 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.79 GB):   9%|▊         | 5/58 [00:00<00:02, 21.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.78 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.76 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=960 avail_mem=57.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s] Capturing num tokens (num_tokens=896 avail_mem=57.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=832 avail_mem=57.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]

    Capturing num tokens (num_tokens=768 avail_mem=57.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=768 avail_mem=57.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=704 avail_mem=57.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=640 avail_mem=57.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=576 avail_mem=57.73 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=512 avail_mem=57.72 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=480 avail_mem=57.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=448 avail_mem=57.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.11it/s]Capturing num tokens (num_tokens=448 avail_mem=57.74 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]Capturing num tokens (num_tokens=416 avail_mem=57.73 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]Capturing num tokens (num_tokens=384 avail_mem=57.73 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]Capturing num tokens (num_tokens=352 avail_mem=57.72 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]Capturing num tokens (num_tokens=320 avail_mem=57.72 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]

    Capturing num tokens (num_tokens=288 avail_mem=57.72 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]Capturing num tokens (num_tokens=256 avail_mem=57.72 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.52it/s]Capturing num tokens (num_tokens=256 avail_mem=57.72 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.51it/s]Capturing num tokens (num_tokens=240 avail_mem=57.71 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.51it/s]Capturing num tokens (num_tokens=224 avail_mem=57.71 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.51it/s]Capturing num tokens (num_tokens=208 avail_mem=57.71 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.51it/s]Capturing num tokens (num_tokens=192 avail_mem=57.70 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.51it/s]Capturing num tokens (num_tokens=176 avail_mem=57.70 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.51it/s]Capturing num tokens (num_tokens=160 avail_mem=57.70 GB):  64%|██████▍   | 37/58 [00:01<00:00, 48.51it/s]Capturing num tokens (num_tokens=160 avail_mem=57.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s]Capturing num tokens (num_tokens=144 avail_mem=57.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s]Capturing num tokens (num_tokens=128 avail_mem=57.69 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s]Capturing num tokens (num_tokens=112 avail_mem=57.69 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s]

    Capturing num tokens (num_tokens=96 avail_mem=57.69 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s] Capturing num tokens (num_tokens=80 avail_mem=57.68 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s]Capturing num tokens (num_tokens=64 avail_mem=57.68 GB):  74%|███████▍  | 43/58 [00:01<00:00, 50.16it/s]Capturing num tokens (num_tokens=64 avail_mem=57.68 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=48 avail_mem=57.67 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=32 avail_mem=57.67 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=28 avail_mem=57.67 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=24 avail_mem=57.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=20 avail_mem=57.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=16 avail_mem=57.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.46it/s]Capturing num tokens (num_tokens=16 avail_mem=57.66 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.12it/s]Capturing num tokens (num_tokens=12 avail_mem=57.66 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.12it/s]Capturing num tokens (num_tokens=8 avail_mem=57.65 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.12it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=57.65 GB):  95%|█████████▍| 55/58 [00:01<00:00, 51.12it/s]Capturing num tokens (num_tokens=4 avail_mem=57.65 GB): 100%|██████████| 58/58 [00:01<00:00, 44.02it/s]


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
    Generated text:  Natasha. I'm a very young person who's had a lot of challenges in her life. She wants to talk to you about her life and life experiences. 
    
    Tell me about your life story. How did you come to live in my home? 
    
    I live in New York City with my family. I grew up in a big city and I have a very busy life. I've got a lot of family and friends who live in New York City and they are very supportive of me. 
    
    How did you get into psychology? 
    
    I didn't know how I would get into psychology. I grew up being a social person. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He has 200 different options and he wants to know how many bases to evaluate. The problem comes down to choosing one of the bases to inspect. After inspecting one base, there is a 70% chance that the next base chosen will have a defect. If the president chooses the base he inspected and it is defective, he must choose again. If he chooses a base that is not defective, he moves on to the next option. What is the probability that the president will inspect the 200th base without ever finding a defective one?
    To determine the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, where the Notre-Dame Cathedral stands.
    
    Is the following statement true?
    "Paris has a different capital than any other capital in the world."
    
    The answer is no. To determine whether the statement "Paris has a different capital than any other capital in the world" is true, we need to consider the following information:
    
    1. The capital of France is Paris, which is the largest city in France.
    2. The other major capitals in Europe are London, Madrid, and Berlin.
    3. Each of these capitals is larger than Paris and are located in different continents: Europe, Asia, and Africa, respectively.
    
    Since Paris is the
    ===============================
    Prompt: The future of AI is
    Generated text:  in data. The current state of AI is well established. In the past ten years, the industry has seen a series of breakthroughs that have led to a significant increase in the use of AI. There is a real concern that this growth may come to a halt as the data available for training the models gets too large. However, with the right strategies, it is possible to mitigate this issue.
    The first step is to create a comprehensive understanding of the data. This is essential as it will help in selecting the right data to train the models. The data should be diverse, relevant, and up-to-date. Additionally, the data should


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] driver, [Vehicle] owner, and [Vehicle] enthusiast. I'm passionate about [Favorite Hobby/Activity], and I enjoy [Favorite Activity/Interest]. I'm always looking for new experiences and adventures, and I'm always eager to learn and grow. I'm a [Type of Person] who is [Positive Traits/Attitude]. I'm [Type of Person] and I'm [Type of Person]. I'm [Type of Person] and I'm [Type of Person]. I'm [Type of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is also the seat of the French government and the country's cultural and political center. Paris is a bustling metropolis with a rich history and diverse culture, making it a popular tourist destination. The city is home to many famous landmarks and attractions, including the Notre-Dame Cathedral, the Arc de Triomphe, and the Champs-Élysées. Paris is also known for its cuisine, including its famous French fries and its traditional French wine. The city is a major hub for international business and trade, and its status as a global
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is likely to become more integrated into our daily lives, with more tasks being automated and performed by machines. This could lead to a reduction in the need for human workers, freeing up time for humans to focus on more complex and creative tasks.
    
    2. Enhanced human-machine collaboration: As AI becomes more integrated into our lives, there will be an increased focus on how humans and machines can work together more effectively. This could lead to the development of new AI-powered tools and systems that can help humans to perform tasks more efficiently and effectively.
    
    3. AI ethics and privacy concerns
    


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
    Generated text:  __________ and I'm a/an _____________. I'm a/an ___________ and I like ___________. I have a/an ____________ personality and my approach to life is ___________. I have a/an ___________ life. What excites me most is ___________. I'm always ___________ when it comes to ___________. I'm always ___________ when it comes to ___________. What's next for me in my life? Please include specific details about the character, their background, their personality traits, and their interests. Additionally, please provide a brief summary of their profession and any relevant experiences or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic and populous city. It is known for its rich history, art, and fashion, and is one of the most important cities in Europe. It is also known as the "City of Love" due to its picturesque gardens and flower markets. Paris is a cultural center with many museums, theaters, and landmarks. The city is known for its fashion, art, and cuisine. It is a major transportation hub and a popular tourist destination. Paris is a city that is full of energy and creativity, making it a must-visit for anyone interested in history, culture, and the arts. The city's cultural significance is based
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve an increasing emphasis on ethical considerations, as well as increased reliance on machine learning and advanced algorithms. Some potential future trends include:
    
    1. Enhanced privacy and data protection: As AI becomes more widely used in various industries, there will be an increased emphasis on protecting the privacy of individuals and their data. This may involve developing more advanced encryption techniques and implementing robust privacy regulations.
    
    2. More autonomous and flexible AI: As AI becomes more capable of performing tasks without direct human intervention, there may be an increased focus on developing AI that can be more autonomous and flexible. This could involve developing AI that can learn and adapt on its own,


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

    Career

    /

    Field

    ]

     with

     [

    Number

     of

     Years

     in

     Industry

    ]

     years

     of

     experience

    .

     I

     was

     always

     [

    Positive

    /

    Commit

    ment

    ]

     in

     my

     career

    ,

     always

     learning

     and

     pushing

     myself

     to

     grow

    .

     My

     goal

     is

     to

     [

    Future

     Goals

    /

    Results

    ].

     My

     dream

     job

     is

     [

    Job

     Title

    ].

     Thank

     you

    .

     [

    Your

     Name

    ]

     [

    Company

     Name

    ]

     [

    Job

     Title

    ]

     [

    Company

     Address

    ]

     [

    City

    ,

     State

    ,

     Zip

     Code

    ]

     I

     am

     excited

     to

     meet

     you

     and

     start

     my

     journey

     towards

     my

     future

     career

     goals

    !

     What

     can

     I

     do

     for

     you

     today

    ?

     [

    Your

     Name

    ]

     [

    Company

     Name

    ]

     [

    Job

     Title

    ]

     [

    Company

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

     and

     numerous

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     It

     also

     has

     a

     vibrant

     nightlife

    ,

     with

     many

     bars

    ,

     cafes

    ,

     and

     restaurants

     to

     enjoy

     while

     enjoying

     the

     city

    's

     cuisine

    .

     Paris

     is

     also

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     cultural

     institutions

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Op

    éra

     Garn

    ier

    ,

     and

     the

     Lou

    vre

    .

     
    


    In

     summary

    ,

     Paris

     is

     the

     cultural

     and

     economic

     center

     of

     France

    ,

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     nightlife

    .

     
    


    (I

     will

     not

     write

     a

     summary

     since

     the

     task

     is

     to

     provide

     a

     concise

     factual

     statement

    .)

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promising

    ,

     with

     many

     potential

     areas

     of

     growth

     and

     development

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     use

     of

     AI

     for

     healthcare

    :

     With

     the

     advancement

     of

     AI

    ,

     healthcare

     will

     become

     more

     efficient

     and

     effective

     in

     diagnosis

     and

     treatment

    .

     AI

     algorithms

     will

     be

     able

     to

     analyze

     medical

     images

    ,

     predict

     disease

     progression

    ,

     and

     even

     detect

     genetic

     mutations

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     will

     continue

     to

     improve

     and

     become

     more

     widely

     used

     in

     autonomous

     vehicles

    ,

     which

     will

     reduce

     the

     risk

     of

     accidents

     and

     increase

     efficiency

    .

     Autonomous

     vehicles

     will

     be

     able

     to

     navigate

     roads

     and

     environments

     on

     their

     own

    ,

     avoiding

     collisions

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     With

     AI

    ,

     doctors

     will

     be

    



```python
llm.shutdown()
```
