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

    [2026-03-03 07:50:55] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 07:50:55] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 07:50:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-03 07:51:03] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 07:51:03] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 07:51:03] INFO engine.py:157: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=41691900, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.75it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.74it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=74.88 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=74.88 GB):   5%|▌         | 1/20 [00:00<00:03,  5.90it/s]Capturing batches (bs=120 avail_mem=74.78 GB):   5%|▌         | 1/20 [00:00<00:03,  5.90it/s]Capturing batches (bs=112 avail_mem=74.78 GB):   5%|▌         | 1/20 [00:00<00:03,  5.90it/s]

    Capturing batches (bs=104 avail_mem=74.77 GB):   5%|▌         | 1/20 [00:00<00:03,  5.90it/s]Capturing batches (bs=96 avail_mem=74.77 GB):   5%|▌         | 1/20 [00:00<00:03,  5.90it/s] Capturing batches (bs=96 avail_mem=74.77 GB):  25%|██▌       | 5/20 [00:00<00:00, 19.55it/s]Capturing batches (bs=88 avail_mem=74.77 GB):  25%|██▌       | 5/20 [00:00<00:00, 19.55it/s]Capturing batches (bs=80 avail_mem=74.77 GB):  25%|██▌       | 5/20 [00:00<00:00, 19.55it/s]Capturing batches (bs=72 avail_mem=74.77 GB):  25%|██▌       | 5/20 [00:00<00:00, 19.55it/s]Capturing batches (bs=64 avail_mem=74.77 GB):  25%|██▌       | 5/20 [00:00<00:00, 19.55it/s]Capturing batches (bs=64 avail_mem=74.77 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.27it/s]Capturing batches (bs=56 avail_mem=74.77 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.27it/s]

    Capturing batches (bs=48 avail_mem=74.77 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.27it/s]Capturing batches (bs=40 avail_mem=74.76 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.27it/s]Capturing batches (bs=32 avail_mem=74.76 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.27it/s]Capturing batches (bs=32 avail_mem=74.76 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.71it/s]Capturing batches (bs=24 avail_mem=74.76 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.71it/s]Capturing batches (bs=16 avail_mem=74.76 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.71it/s]

    Capturing batches (bs=12 avail_mem=74.76 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.71it/s]Capturing batches (bs=12 avail_mem=74.76 GB):  80%|████████  | 16/20 [00:00<00:00, 25.12it/s]Capturing batches (bs=8 avail_mem=74.76 GB):  80%|████████  | 16/20 [00:00<00:00, 25.12it/s] Capturing batches (bs=4 avail_mem=74.76 GB):  80%|████████  | 16/20 [00:00<00:00, 25.12it/s]Capturing batches (bs=2 avail_mem=74.76 GB):  80%|████████  | 16/20 [00:00<00:00, 25.12it/s]Capturing batches (bs=1 avail_mem=74.76 GB):  80%|████████  | 16/20 [00:00<00:00, 25.12it/s]Capturing batches (bs=1 avail_mem=74.76 GB): 100%|██████████| 20/20 [00:00<00:00, 28.45it/s]Capturing batches (bs=1 avail_mem=74.76 GB): 100%|██████████| 20/20 [00:00<00:00, 25.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:17,  2.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:17,  2.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:17,  2.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:17,  2.41s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:26,  2.04it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:08,  5.48it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s] 

    Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=768):  28%|██▊       | 16/58 [00:02<00:03, 11.51it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:02<00:01, 20.56it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:02<00:00, 30.10it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:02<00:00, 30.10it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:02<00:00, 30.10it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:02<00:00, 30.10it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=144):  59%|█████▊    | 34/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=20):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=16):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=12):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=8):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s] Compiling num tokens (num_tokens=4):  76%|███████▌  | 44/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 18.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.93 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.93 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.92 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.93 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.93 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.92 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.90 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.88 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.86 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=960 avail_mem=73.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.12it/s] Capturing num tokens (num_tokens=896 avail_mem=73.87 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.12it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.87 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=832 avail_mem=73.87 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=768 avail_mem=73.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=704 avail_mem=73.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=640 avail_mem=73.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=576 avail_mem=73.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=512 avail_mem=73.84 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=480 avail_mem=73.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=480 avail_mem=73.86 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=448 avail_mem=73.86 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=416 avail_mem=73.86 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=384 avail_mem=73.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=352 avail_mem=73.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=288 avail_mem=73.84 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=288 avail_mem=73.84 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=256 avail_mem=73.84 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=240 avail_mem=73.84 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=224 avail_mem=73.83 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=208 avail_mem=73.83 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=192 avail_mem=73.83 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=176 avail_mem=73.83 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=176 avail_mem=73.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=160 avail_mem=73.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=144 avail_mem=73.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=128 avail_mem=73.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.57it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=96 avail_mem=73.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.57it/s] Capturing num tokens (num_tokens=96 avail_mem=73.81 GB):  81%|████████  | 47/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=80 avail_mem=73.81 GB):  81%|████████  | 47/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=64 avail_mem=73.81 GB):  81%|████████  | 47/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=48 avail_mem=73.80 GB):  81%|████████  | 47/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=32 avail_mem=73.80 GB):  81%|████████  | 47/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=28 avail_mem=73.79 GB):  81%|████████  | 47/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=28 avail_mem=73.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=24 avail_mem=73.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=20 avail_mem=73.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.47it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.77 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=12 avail_mem=73.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=8 avail_mem=73.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.47it/s] Capturing num tokens (num_tokens=8 avail_mem=73.76 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=4 avail_mem=73.76 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=4 avail_mem=73.76 GB): 100%|██████████| 58/58 [00:01<00:00, 40.77it/s]


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
    Generated text:  Xiaoyan. I'm 17 years old and I'm a junior in high school. I have never lived in a dorm room before. The dorm room is big and has lots of shelves for books and papers. The walls are white. The windows are big and bright. I like to have a quiet place to read books or do homework. I enjoy having a place where I can be alone and do things that I enjoy. I also like to do some housework and clean the room. I have a new friend named Xiaohui. He's a 15-year-old boy who's a bit more outgoing than me
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the country. There are also vice presidents. They are the second important person in the government. The vice president is a very important person. If the president is impeached, the vice president will be called the new president. The vice president is responsible for the day-to-day management of the country. The vice president is a very important person.
    How many duties does the vice president have? (If the question is unanswerable, say "unanswerable") unanswerable
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    
    B. London
    
    C. Berlin
    
    D. Moscow
    
    E. No capital
    
    To determine the capital of France, we need to recall which country is located in Europe. The capital of France is not one of the options given. Let's verify this:
    
    1. France is a country located in Western Europe.
    2. The European Union is a supranational organization that includes several countries, but France is not one of its member states.
    3. The European Capital of Culture is a European cultural event that takes place in different cities across the European Union. However, France does not participate in this event.
    
    Given
    ===============================
    Prompt: The future of AI is
    Generated text:  complex and constantly changing, with new technologies emerging regularly. AI has the potential to revolutionize many industries, but it is also facing significant challenges and regulatory hurdles. In this article, we will explore some of the key areas where AI is currently evolving, and how it might change the way we interact with technology.
    One area where AI is rapidly evolving is in the field of natural language processing (NLP). NLP is the ability of machines to understand and generate human language. This includes tasks such as sentiment analysis, text classification, and language translation. As AI technology continues to improve, it is becoming more accurate and capable of handling more complex


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason for interest]. I'm always looking for ways to [job title] and [job title]. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to [company name]. I'm a [job title] at [company name], and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also home to the Louvre Museum, the most famous art museum in the world, and the Notre-Dame Cathedral, a UNESCO World Heritage site. Paris is a bustling metropolis with a rich history and a diverse population of over 10 million people. It is a popular tourist destination and a cultural hub for France and the world. The city is known for its fashion, food, and music scenes, and is home to many famous landmarks and attractions. Paris is a city of contrasts, with its modern architecture and historical
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as
    


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
    Generated text:  [insert character's name]. I am a [insert character's profession or role]. I enjoy [insert something that describes my interests or hobbies, if applicable]. I am [insert age, height, weight, etc.]. I live in [insert location, if applicable]. I am [insert personality type, if applicable]. I have [insert accomplishments, if applicable]. I am [insert a creative and unique word that I use to describe myself, if applicable]. I am [insert a unique piece of technology or equipment that I have, if applicable]. I am [insert a unique skill or talent that I have, if applicable]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many cultural institutions and attractions, including the Palace of Versailles and the Aqueduct of Arc de Triomphe. It is a bustling metropolis with a rich history and culture, making it an important city for global trade and diplomacy. Paris is also known for its fashion industry and food scene, attracting millions of tourists each year. The city has a reputation for being elegant, sophisticated, and exciting, and it continues to be a major center of the French Republic. Based on
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by exponential growth in the availability and affordability of AI technologies. AI research will continue to push the boundaries of what is possible in terms of processing power, memory, and computation speed. This will lead to the development of more powerful and efficient AI systems that can perform a wider range of tasks. AI will also become more capable of handling increasingly complex and nuanced tasks, such as recognizing human emotions and understanding natural language. Finally, AI will also become more interconnected and collaborative, with more people and organizations collaborating to create and deploy AI technologies that can improve society as a whole.
    
    Another potential trend in AI is the increasing integration of AI


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

    job

     title

    ]

     with

     over

     [

    number

     of

     years

    ]

     of

     experience

    .

     I

     have

     a

     passion

     for

     [

    reason

     why

     you

     love

     this

     job

    ]

     and

     I

     am

     always

     ready

     to

     learn

     and

     grow

    .

     I

     enjoy

     [

    reason

     why

     you

     enjoy

     this

     job

    ]

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     help

     others

     succeed

    .

     Please

     let

     me

     know

     how

     I

     can

     assist

     you

     in

     your

     journey

    .

     [

    Name

    ]

     [

    Your

     name

    ]

     [

    Your

     profession

     or

     role

    ]


    [

    Name

    ]

     is

     a

     [

    job

     title

    ]

     with

     over

     [

    number

     of

     years

    ]

     of

     experience

    .

     I

     have

     a

     passion

     for

     [

    reason

     why

     you

     love

     this

     job

    ]

     and

     I

     am

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     most

     populous

     city

     in

     the

     European

     Union

     and

     is

     known

     for

     its

     iconic

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

     Ch

    amps

    -

    É

    lys

    ées

    .

     The

     city

     is

     also

     home

     to

     the

     Palace

     of

     Vers

    ailles

     and

     has

     been

     the

     seat

     of

     French

     government

     since

     the

     

    1

    3

    th

     century

    .

     Paris

     is

     a

     cultural

    ,

     political

    ,

     and

     economic

     hub

     of

     the

     country

    ,

     attracting

     millions

     of

     tourists

     and

     visitors

     annually

    .

     It

     is

     a

     major

     hub

     for

     business

    ,

     finance

    ,

     and

     media

     in

     the

     European

     Union

    .

     The

     city

     has

     a

     rich

     history

     and

     is

     known

     for

     its

     historical

     sites

    ,

     arts

    ,

     and

     architecture

    .

     Paris

     is

     the

     administrative

     capital

     of

     France

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     exciting

    .

     Here

     are

     some

     possible

     trends

     that

     are

     shaping

     the

     field

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     becomes

     more

     advanced

     and

     integrated

     with

     other

     technologies

    ,

     we

     can

     expect

     automation

     to

     become

     a

     major

     trend

    .

     This

     could

     lead

     to

     widespread

     automation

     in

     industries

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     healthcare

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     enable

     businesses

     to

     analyze

     vast

     amounts

     of

     data

     to

     understand

     customer

     behavior

     and

     preferences

    .

     This

     will

     lead

     to

     more

     personalized

     products

     and

     services

    ,

     as

     well

     as

     better

     customer

     experiences

    .
    


    3

    .

     Integration

     with

     other

     fields

    :

     AI

     will

     become

     more

     integrated

     with

     other

     fields

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     education

    .

     This

     will

     allow

     for

     more

    



```python
llm.shutdown()
```
