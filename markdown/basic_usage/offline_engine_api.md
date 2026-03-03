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

    [2026-03-03 23:10:40] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 23:10:40] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 23:10:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-03 23:10:42] INFO server_args.py:1974: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 23:10:42] INFO server_args.py:3065: Set soft_watchdog_timeout since in CI


    [2026-03-03 23:10:42] INFO engine.py:158: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=209681958, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.29it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.29it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=55.85 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=55.85 GB):   5%|▌         | 1/20 [00:00<00:03,  5.50it/s]Capturing batches (bs=120 avail_mem=55.75 GB):   5%|▌         | 1/20 [00:00<00:03,  5.50it/s]

    Capturing batches (bs=112 avail_mem=55.75 GB):   5%|▌         | 1/20 [00:00<00:03,  5.50it/s]Capturing batches (bs=104 avail_mem=55.73 GB):   5%|▌         | 1/20 [00:00<00:03,  5.50it/s]Capturing batches (bs=104 avail_mem=55.73 GB):  20%|██        | 4/20 [00:00<00:01, 15.23it/s]Capturing batches (bs=96 avail_mem=55.75 GB):  20%|██        | 4/20 [00:00<00:01, 15.23it/s] Capturing batches (bs=88 avail_mem=55.74 GB):  20%|██        | 4/20 [00:00<00:01, 15.23it/s]Capturing batches (bs=80 avail_mem=55.74 GB):  20%|██        | 4/20 [00:00<00:01, 15.23it/s]Capturing batches (bs=80 avail_mem=55.74 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.53it/s]Capturing batches (bs=72 avail_mem=55.74 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.53it/s]

    Capturing batches (bs=64 avail_mem=55.74 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.53it/s]Capturing batches (bs=56 avail_mem=55.73 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.53it/s]Capturing batches (bs=56 avail_mem=55.73 GB):  50%|█████     | 10/20 [00:00<00:00, 21.33it/s]Capturing batches (bs=48 avail_mem=55.73 GB):  50%|█████     | 10/20 [00:00<00:00, 21.33it/s]Capturing batches (bs=40 avail_mem=55.73 GB):  50%|█████     | 10/20 [00:00<00:00, 21.33it/s]Capturing batches (bs=32 avail_mem=55.73 GB):  50%|█████     | 10/20 [00:00<00:00, 21.33it/s]

    Capturing batches (bs=32 avail_mem=55.73 GB):  65%|██████▌   | 13/20 [00:00<00:00, 20.34it/s]Capturing batches (bs=24 avail_mem=55.72 GB):  65%|██████▌   | 13/20 [00:00<00:00, 20.34it/s]Capturing batches (bs=16 avail_mem=55.72 GB):  65%|██████▌   | 13/20 [00:00<00:00, 20.34it/s]Capturing batches (bs=12 avail_mem=55.72 GB):  65%|██████▌   | 13/20 [00:00<00:00, 20.34it/s]Capturing batches (bs=12 avail_mem=55.72 GB):  80%|████████  | 16/20 [00:00<00:00, 19.75it/s]Capturing batches (bs=8 avail_mem=55.72 GB):  80%|████████  | 16/20 [00:00<00:00, 19.75it/s] 

    Capturing batches (bs=4 avail_mem=55.71 GB):  80%|████████  | 16/20 [00:00<00:00, 19.75it/s]Capturing batches (bs=2 avail_mem=55.71 GB):  80%|████████  | 16/20 [00:00<00:00, 19.75it/s]Capturing batches (bs=1 avail_mem=55.71 GB):  80%|████████  | 16/20 [00:00<00:00, 19.75it/s]Capturing batches (bs=1 avail_mem=55.71 GB): 100%|██████████| 20/20 [00:00<00:00, 23.35it/s]Capturing batches (bs=1 avail_mem=55.71 GB): 100%|██████████| 20/20 [00:00<00:00, 20.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:11,  2.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:11,  2.31s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:57,  1.02s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:57,  1.02s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:57,  1.02s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  4.00it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  4.00it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  4.00it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  4.00it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.81it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.81it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.81it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.81it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:04, 10.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:04, 10.00it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:04, 10.00it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:04, 10.00it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 13.31it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 13.31it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 13.31it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 13.31it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 13.31it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 17.54it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 17.54it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 17.54it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 17.54it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 17.54it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 21.87it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 21.87it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 21.87it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 21.87it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 21.87it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:01, 21.87it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 26.82it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 26.82it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 26.82it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 26.82it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 26.82it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 26.82it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 30.87it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 30.87it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 30.87it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 30.87it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 30.87it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 30.87it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 37.42it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 43.05it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 48.72it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 48.72it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 48.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.30 GB):   2%|▏         | 1/58 [00:00<00:08,  6.77it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.27 GB):   2%|▏         | 1/58 [00:00<00:08,  6.77it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.27 GB):   3%|▎         | 2/58 [00:00<00:07,  7.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.27 GB):   3%|▎         | 2/58 [00:00<00:07,  7.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.27 GB):   5%|▌         | 3/58 [00:00<00:07,  7.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.26 GB):   5%|▌         | 3/58 [00:00<00:07,  7.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.26 GB):   7%|▋         | 4/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.27 GB):   7%|▋         | 4/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.27 GB):   9%|▊         | 5/58 [00:00<00:06,  7.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.26 GB):   9%|▊         | 5/58 [00:00<00:06,  7.68it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.26 GB):  10%|█         | 6/58 [00:00<00:06,  7.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.26 GB):  10%|█         | 6/58 [00:00<00:06,  7.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.26 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.26 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.31it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.26 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.26 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.26 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.25 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=55.25 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.25 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.25 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.24 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.83it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=55.24 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.24 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.24 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.24 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.23 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.09it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=55.23 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.23 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.23 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.22 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.52it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=55.22 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.22 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.20 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.20 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.22it/s]Capturing num tokens (num_tokens=960 avail_mem=55.22 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.22it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=55.21 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.22it/s]Capturing num tokens (num_tokens=896 avail_mem=55.21 GB):  40%|███▉      | 23/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=832 avail_mem=55.21 GB):  40%|███▉      | 23/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=768 avail_mem=55.20 GB):  40%|███▉      | 23/58 [00:02<00:03, 11.48it/s]

    Capturing num tokens (num_tokens=768 avail_mem=55.20 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.76it/s]Capturing num tokens (num_tokens=704 avail_mem=55.20 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.76it/s]Capturing num tokens (num_tokens=640 avail_mem=55.20 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.76it/s]Capturing num tokens (num_tokens=640 avail_mem=55.20 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.93it/s]Capturing num tokens (num_tokens=576 avail_mem=55.20 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.93it/s]

    Capturing num tokens (num_tokens=512 avail_mem=55.18 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.93it/s]Capturing num tokens (num_tokens=512 avail_mem=55.18 GB):  50%|█████     | 29/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=480 avail_mem=55.20 GB):  50%|█████     | 29/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=448 avail_mem=55.20 GB):  50%|█████     | 29/58 [00:02<00:02, 12.06it/s]

    Capturing num tokens (num_tokens=448 avail_mem=55.20 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.63it/s]Capturing num tokens (num_tokens=416 avail_mem=55.20 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.63it/s]Capturing num tokens (num_tokens=384 avail_mem=55.19 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.63it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.19 GB):  57%|█████▋    | 33/58 [00:03<00:02,  9.72it/s]Capturing num tokens (num_tokens=352 avail_mem=55.19 GB):  57%|█████▋    | 33/58 [00:03<00:02,  9.72it/s]Capturing num tokens (num_tokens=320 avail_mem=55.19 GB):  57%|█████▋    | 33/58 [00:03<00:02,  9.72it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.19 GB):  60%|██████    | 35/58 [00:03<00:02,  9.46it/s]Capturing num tokens (num_tokens=288 avail_mem=55.18 GB):  60%|██████    | 35/58 [00:03<00:02,  9.46it/s]Capturing num tokens (num_tokens=256 avail_mem=55.18 GB):  60%|██████    | 35/58 [00:03<00:02,  9.46it/s]Capturing num tokens (num_tokens=256 avail_mem=55.18 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.17it/s]Capturing num tokens (num_tokens=240 avail_mem=55.18 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.17it/s]

    Capturing num tokens (num_tokens=224 avail_mem=55.17 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.17it/s]Capturing num tokens (num_tokens=224 avail_mem=55.17 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.72it/s]Capturing num tokens (num_tokens=208 avail_mem=55.17 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.72it/s]Capturing num tokens (num_tokens=192 avail_mem=55.17 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.72it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.17 GB):  71%|███████   | 41/58 [00:04<00:01, 11.17it/s]Capturing num tokens (num_tokens=176 avail_mem=55.17 GB):  71%|███████   | 41/58 [00:04<00:01, 11.17it/s]Capturing num tokens (num_tokens=160 avail_mem=55.16 GB):  71%|███████   | 41/58 [00:04<00:01, 11.17it/s]Capturing num tokens (num_tokens=160 avail_mem=55.16 GB):  74%|███████▍  | 43/58 [00:04<00:01, 11.59it/s]Capturing num tokens (num_tokens=144 avail_mem=55.16 GB):  74%|███████▍  | 43/58 [00:04<00:01, 11.59it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.16 GB):  74%|███████▍  | 43/58 [00:04<00:01, 11.59it/s]Capturing num tokens (num_tokens=128 avail_mem=55.16 GB):  78%|███████▊  | 45/58 [00:04<00:01, 11.84it/s]Capturing num tokens (num_tokens=112 avail_mem=55.15 GB):  78%|███████▊  | 45/58 [00:04<00:01, 11.84it/s]Capturing num tokens (num_tokens=96 avail_mem=55.15 GB):  78%|███████▊  | 45/58 [00:04<00:01, 11.84it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=55.15 GB):  81%|████████  | 47/58 [00:04<00:00, 11.94it/s]Capturing num tokens (num_tokens=80 avail_mem=55.15 GB):  81%|████████  | 47/58 [00:04<00:00, 11.94it/s]Capturing num tokens (num_tokens=64 avail_mem=55.15 GB):  81%|████████  | 47/58 [00:04<00:00, 11.94it/s]Capturing num tokens (num_tokens=64 avail_mem=55.15 GB):  84%|████████▍ | 49/58 [00:04<00:00, 11.98it/s]Capturing num tokens (num_tokens=48 avail_mem=55.14 GB):  84%|████████▍ | 49/58 [00:04<00:00, 11.98it/s]

    Capturing num tokens (num_tokens=32 avail_mem=55.14 GB):  84%|████████▍ | 49/58 [00:04<00:00, 11.98it/s]Capturing num tokens (num_tokens=32 avail_mem=55.14 GB):  88%|████████▊ | 51/58 [00:04<00:00, 12.10it/s]Capturing num tokens (num_tokens=28 avail_mem=55.13 GB):  88%|████████▊ | 51/58 [00:04<00:00, 12.10it/s]Capturing num tokens (num_tokens=24 avail_mem=55.13 GB):  88%|████████▊ | 51/58 [00:04<00:00, 12.10it/s]

    Capturing num tokens (num_tokens=24 avail_mem=55.13 GB):  91%|█████████▏| 53/58 [00:05<00:00, 11.99it/s]Capturing num tokens (num_tokens=20 avail_mem=55.13 GB):  91%|█████████▏| 53/58 [00:05<00:00, 11.99it/s]Capturing num tokens (num_tokens=16 avail_mem=55.12 GB):  91%|█████████▏| 53/58 [00:05<00:00, 11.99it/s]Capturing num tokens (num_tokens=16 avail_mem=55.12 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.15it/s]Capturing num tokens (num_tokens=12 avail_mem=55.12 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.15it/s]

    Capturing num tokens (num_tokens=8 avail_mem=55.12 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.15it/s] Capturing num tokens (num_tokens=8 avail_mem=55.12 GB):  98%|█████████▊| 57/58 [00:05<00:00, 12.23it/s]Capturing num tokens (num_tokens=4 avail_mem=55.11 GB):  98%|█████████▊| 57/58 [00:05<00:00, 12.23it/s]Capturing num tokens (num_tokens=4 avail_mem=55.11 GB): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


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
    Generated text:  Robert and I am a full time college student majoring in Accounting. I have decided to develop a portfolio for my academic and professional career. Please create a portfolio using the following format and format requirements:
    
      * Define the purpose of your portfolio.
      * Define your key skills.
      * Provide a short summary of each key skill.
      * Provide an example of how you apply each skill to a current or current project.
      * Include a visual representation of your portfolio, such as a table or diagram, that shows the relationships between your skills and how they are linked to each other.
      * Include at least 5 different types of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country, the leader of the government and the leader of the United States. The president is elected by the people to have a term of 4 years. The president is the one who makes decisions on how to organize the country and also works to make sure that the government follows the rules. The president is always the leader of the country. The president's job is to help make sure that the country has a good government and that it works well. How does the president work? The president works by following the rules of the country and doing the things that they do. The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the Seine River. On the right side of the river, there is a bridge called the Arc de Triomphe. It is one of the most famous landmarks in Paris. The Arc de Triomphe has a rich history. It was built in 1790. It was named after the 10 Revolutionary Wars. The history of the building is as old as the city of Paris itself. The bridge across the river is called the Pont du Gard. It is the oldest bridge in Europe. It was built in 1263. It was used for boats for a long time
    ===============================
    Prompt: The future of AI is
    Generated text:  here now. With the world’s most advanced AI technologies, we have the power to not only recognize individuals, but to also predict how they will behave in future situations. This is a truly revolutionary advancement in terms of predicting human behavior. Not only do AI systems have the ability to predict a person’s behavior, but we also have the power to make adjustments to that behavior as needed.
    
    As these AI systems continue to evolve, one of the most significant challenges is ensuring that individuals are not vulnerable to exploitation. In order to avoid this, we need to ensure that AI systems are designed with security in mind.
    
    One of the most effective ways to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your character's personality or background]. And what do you do for a living? I'm a [insert a short description of your job or profession]. And what do you enjoy doing in your free time? I enjoy [insert a short description of your hobbies or interests]. And what do you like to do for fun? I like to [insert a short description of your hobbies or interests]. And what do you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its cuisine, fashion, and art, and is home to many world-renowned museums, theaters, and galleries. Paris is a vibrant and dynamic city with a strong sense of French identity and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced natural language processing: AI will become more capable of understanding and generating human language, enabling more sophisticated natural language processing and speech recognition.
    
    3. Improved predictive analytics: AI will become more capable of predicting future events and outcomes, enabling organizations to make more informed decisions and take proactive steps to mitigate risks.
    
    4. Enhanced machine learning: AI will become more capable of learning from large datasets and making more
    


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
    Generated text:  [insert name], and I'm an AI assistant designed to assist with various tasks such as answering questions, organizing files, and providing general information. My programming is based on the principles of AI and machine learning, and I'm constantly learning and improving my responses to better serve you. How can I assist you today? [insert name] [insert any relevant background or skills that you can provide]. I'm here to help and provide you with any assistance you need, so please feel free to ask me anything. How can I assist you today? [insert name] [insert any relevant background or skills that you can provide]. I'm here
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, located in the Seine valley in the south of the country. It is the world’s most populous city and one of the most visited tourist destinations, and is home to numerous notable landmarks such as the Eiffel Tower and the Louvre Museum. Paris is also a center for arts, culture, and politics, with numerous museums, theaters, and government buildings. The city is known for its beautiful architecture, vibrant nightlife, and delicious cuisine. Additionally, Paris hosts the Eiffel Tower Parc Monceau Cultural and the Festival de Cannes. Therefore, it is the largest city in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of factors, including the pace of technological development, the impact of new research and innovations, and the changing needs and priorities of individuals and organizations. Here are some potential trends that could shape the future of AI:
    
    1. Increased focus on ethical and responsible AI: As more individuals and organizations become aware of the potential risks and consequences of AI, there is likely to be increased focus on developing AI that is designed with ethical considerations in mind. This could involve developing more transparent algorithms and practices, and ensuring that AI is used to address real-world problems, rather than being used for malicious purposes.
    
    2. Greater integration


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

    Role

    ]

     at

     [

    Company

    ].

     I

     am

     [

    Age

    ]

     years

     old

    ,

     and

     I

     enjoy

     [

    Favorite

     Hobby

    /

    Activity

    ].

     I

     have

     [

    Number

     of

     years

     of

     experience

    ]

     years

     of

     experience

     in

     [

    Industry

    ].

     I

     have

     a

     passion

     for

     [

    Why

     I

     like

     what

     I

     do

    ].

     In

     my

     spare

     time

    ,

     I

     enjoy

     [

    What

     else

     I

     like

     to

     do

    ].

     Let

    's

     chat

    !

     [

    Int

    roduce

     yourself

     to

     potential

     customers

    ,

     colleagues

    ,

     or

     other

     potential

     partners

    ]

     Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

     am

     a

     [

    Role

    ]

     at

     [

    Company

    ].

     I

     am

     [

    Age

    ]

     years

     old

    ,

     and

     I

     enjoy

     [

    Favorite

     Hobby

    /

    
    
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

     and

     the

     fifth

    -largest

     city

     in

     the

     world

     by

     population

    .

     It

     is

     known

     for

     its

     stunning

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     famous

     for

     its

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

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     Carnival

     and

     its

     famous

     French

     cuisine

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     and

     economic

     hub

     in

     Europe

    .

     It

     is

     the

     capital

     of

     France

     and

     the

     largest

     city

     in

     the

     country

    .

     
    


    Therefore

    ,

     the

     answer

     is

    :
    


    Paris

     is

     the

     capital

     city

     of

     France

    .

     
    


    This

     statement

     provides

     a

     concise

     factual

     statement

     about

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     advancements

     in

     various

     areas

    ,

     including

    :
    


    1

    .

     Increased

     speed

     and

     accuracy

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     we

     may

     see

     faster

     and

     more

     accurate

     predictions

    ,

     analyses

    ,

     and

     solutions

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     AI

     systems

     that

     can

     understand

     and

     interpret

     natural

     language

     will

     become

     more

     sophisticated

    ,

     enabling

     more

     sophisticated

     and

     intelligent

     interactions

     with

     humans

    .
    


    3

    .

     Enhanced

     AI

     ethics

    :

     As

     more

     AI

     systems

     are

     used

     in

     decision

    -making

     processes

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ethical

     considerations

     and

     the

     development

     of

     more

     ethical

     AI

     systems

    .
    


    4

    .

     Rise

     of

     autonomous

     robots

    :

     The

     development

     of

     autonomous

     robots

     will

     likely

     become

     more

     advanced

    ,

     with

     the

     ability

     to

    



```python
llm.shutdown()
```
