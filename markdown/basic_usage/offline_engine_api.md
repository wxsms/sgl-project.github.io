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

    [2026-03-15 06:06:29] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-15 06:06:29] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-15 06:06:29] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-15 06:06:32] INFO server_args.py:2146: Attention backend not specified. Use fa3 backend by default.


    [2026-03-15 06:06:32] INFO server_args.py:3287: Set soft_watchdog_timeout since in CI


    [2026-03-15 06:06:32] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=304081193, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.66it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.65it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:24,  2.53s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.05it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:02<00:03, 12.79it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]Compiling num tokens (num_tokens=640):  31%|███       | 18/58 [00:02<00:03, 12.79it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 21.52it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 21.52it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 21.52it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 21.52it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 21.52it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 21.52it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 21.52it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 21.52it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 21.52it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 21.52it/s]Compiling num tokens (num_tokens=256):  47%|████▋     | 27/58 [00:03<00:01, 21.52it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s] Compiling num tokens (num_tokens=80):  64%|██████▍   | 37/58 [00:03<00:00, 32.36it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 44.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.68 GB):   2%|▏         | 1/58 [00:00<00:06,  8.42it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.65 GB):   2%|▏         | 1/58 [00:00<00:06,  8.42it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=68.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.64 GB):   3%|▎         | 2/58 [00:00<00:06,  8.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.64 GB):   3%|▎         | 2/58 [00:00<00:06,  8.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.64 GB):   7%|▋         | 4/58 [00:00<00:05, 10.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.64 GB):   7%|▋         | 4/58 [00:00<00:05, 10.57it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=68.64 GB):   7%|▋         | 4/58 [00:00<00:05, 10.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.63 GB):  10%|█         | 6/58 [00:00<00:03, 13.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.63 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.63 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.62 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.79it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=68.62 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.61 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.68it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=68.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=960 avail_mem=68.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.68it/s] Capturing num tokens (num_tokens=960 avail_mem=68.58 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=896 avail_mem=68.58 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=832 avail_mem=68.58 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=768 avail_mem=68.57 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=704 avail_mem=68.57 GB):  38%|███▊      | 22/58 [00:01<00:01, 32.96it/s]Capturing num tokens (num_tokens=640 avail_mem=68.57 GB):  38%|███▊      | 22/58 [00:01<00:01, 32.96it/s]Capturing num tokens (num_tokens=576 avail_mem=68.57 GB):  38%|███▊      | 22/58 [00:01<00:01, 32.96it/s]Capturing num tokens (num_tokens=576 avail_mem=68.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=512 avail_mem=68.56 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=480 avail_mem=68.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]

    Capturing num tokens (num_tokens=448 avail_mem=68.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=416 avail_mem=68.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=384 avail_mem=68.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=352 avail_mem=68.56 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=352 avail_mem=68.56 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=320 avail_mem=68.56 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=288 avail_mem=68.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=256 avail_mem=68.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=240 avail_mem=68.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=224 avail_mem=68.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]Capturing num tokens (num_tokens=208 avail_mem=68.54 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.30it/s]

    Capturing num tokens (num_tokens=208 avail_mem=68.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=192 avail_mem=68.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=176 avail_mem=68.25 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=160 avail_mem=68.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=144 avail_mem=68.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=128 avail_mem=68.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.39it/s]

    Capturing num tokens (num_tokens=128 avail_mem=68.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=112 avail_mem=68.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=96 avail_mem=68.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.16it/s] Capturing num tokens (num_tokens=80 avail_mem=68.31 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=64 avail_mem=68.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=64 avail_mem=68.32 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=48 avail_mem=68.32 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=32 avail_mem=68.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]

    Capturing num tokens (num_tokens=28 avail_mem=68.33 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=24 avail_mem=68.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=24 avail_mem=68.34 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=20 avail_mem=68.33 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=16 avail_mem=68.42 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=12 avail_mem=68.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=8 avail_mem=68.34 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.57it/s] Capturing num tokens (num_tokens=8 avail_mem=68.34 GB):  98%|█████████▊| 57/58 [00:01<00:00, 31.13it/s]Capturing num tokens (num_tokens=4 avail_mem=68.34 GB):  98%|█████████▊| 57/58 [00:01<00:00, 31.13it/s]

    Capturing num tokens (num_tokens=4 avail_mem=68.34 GB): 100%|██████████| 58/58 [00:01<00:00, 29.28it/s]


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
    Generated text:  Roland Voss, and I am a graduate of the University of California, Los Angeles, with a Master of Arts in Political Science, the University of Illinois at Chicago, and a Bachelor of Arts in English and American Studies, the University of California, Berkeley. I am an Area 50 member and an Adjunct Professor in the Department of Communication Studies at the University of Chicago.
    My research interests include political engagement, political action, and the social construction of the subject. I have also written articles on the use of virtual reality in the political arena and the role of social media in political discourse.
    My research has been supported by a National
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with the power to appoint and replace the president of the United States. What is this office?
    The office of the president of the United States is called the President of the United States. The President of the United States is a political office with the power to appoint and replace the president of the United States. The President of the United States serves a term of two years and is elected to a four-year term. The President is the head of state and government and is the leader of the United States. The President is also the Commander-in-Chief of the United States Armed Forces. The President is responsible for overseeing the federal government
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is famous for its landmarks and monuments. Among these, the Eiffel Tower is the most famous. How many digits are in the number of the Eiffel Tower?
    
    The Eiffel Tower has a number of digits in its name that is a clue to the number of digits in its number. The first digit is 1, the second digit is 5, the third digit is 3, the fourth digit is 7, the fifth digit is 0, and the sixth digit is 2. If we add these together, we get:
    
    1 + 5 + 3 + 7 + 
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    The future of AI is bright
    
    Image: Karen Bowler/Reuters
    
    If you’re not a fan of AI, you probably have a strong argument. You probably believe that it’s a scary, yet increasingly popular, field that has the potential to make our lives much easier. It could, for example, make speech recognition faster, speak to you more or even potentially make it more difficult to be a human being. However, the more we’ve learned about the technology, the more we know it has the potential to be a game-changer for us all.
    
    Let’s take a look at some of the exciting potential that’s


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its diverse cuisine, including French cuisine, and is home to many museums, theaters, and art galleries. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also known for its annual festivals and events, including the Eiffel Tower Parade and the Louvre Festival. Overall,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more complex and sophisticated AI systems.
    
    3. Development of new hardware: As AI technology continues to advance, there will be a
    


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
    Generated text:  [Your Name], and I'm a [job title] at [Your company's name], specializing in [specific area or expertise]. If you're in a similar role, I'm confident that you'd find me equally as capable. I can help with everything from project management to data analysis, and I enjoy working with a team. If you're looking for someone to help with [specific project or task], or if you have a question about [your field of expertise], I'm all ears. Looking forward to the opportunity to assist you. [Your Name] [Your Job Title] at [Your Company's Name], [Your Company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To break this down:
    - Paris is the largest city in France and is the capital of France.
    - It is located in the northern region of France.
    - It is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. 
    - Paris is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, which are all UNESCO World Heritage sites. 
    - The city is also the second-largest city in France by population. 
    - The French capital is also known as "La Ville Blanche," meaning "White City
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve rapidly, with a number of key trends shaping the field in the coming years. Here are some potential areas of development and innovation that are likely to occur:
    
    1. Increased automation: As AI technology continues to improve, it is likely to become more efficient and effective at performing a wide range of tasks, from manufacturing to customer service to data analysis. This will lead to increased automation in various industries, with more tasks being completed by machines rather than human workers.
    
    2. Biometric recognition: As more people become concerned about privacy and data security, biometric recognition is likely to become more widely adopted. This includes technologies such


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

    ].

     I

     am

     a

     [

    type

     of

     job

    ]

     with

     [

    number

     of

     years

     of

     experience

    ],

     and

     I

     love

     [

    job

     title

    ].

     I

    'm

     always

     looking

     to

     learn

     and

     improve

    ,

     and

     I

    'm

     always

     eager

     to

     explore

     new

     ideas

     and

     opportunities

     to

     grow

    .

     I

    'm

     an

     organized

     person

     and

     enjoy

     keeping

     my

     workspace

     clean

     and

     tidy

    .

     I

    'm

     passionate

     about

     [

    job

     title

    ]

     and

     I

     am

     always

     up

     for

     a

     challenge

    .

     I

    'm

     confident

     and

     driven

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     accomplish

     my

     goals

    .

     I

    'm

     not

     afraid

     of

     taking

     risks

     and

     I

    'm

     always

     willing

     to

     learn

     from

     failure

    .

     I

    'm

     a

     team

     player

     and

     enjoy

     collaborating

     with

     others

    .

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Par

    oi

    ."

     It

     is

     a

     historical

     and

     cultural

     city

     renowned

     for

     its

     rich

     history

    ,

     art

    ,

     cuisine

    ,

     and

     fashion

    .

     Paris

     is

     a

     major

     financial

     and

     cultural

     center

     and

     is

     the

     most

     visited

     city

     in

     the

     world

    .

     It

     is

     also

     a

     destination

     for

     world

    -class

     dining

    ,

     nightlife

    ,

     and

     shopping

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     a

     Hundred

     Eyes

    "

     due

     to

     its

     vast

     number

     of

     museums

    ,

     art

     galleries

    ,

     and

     historical

     buildings

    .

     It

     is

     also

     a

     popular

     tourist

     destination

     for

     tourists

     from

     all

     over

     the

     world

    .

     Paris

     is

     known

     for

     its

     annual

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     other

     landmarks

    .

     It

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

    ,

     and

     there

     are

     several

     possible

     trends

     that

     we

     can

     expect

     in

     the

     coming

     years

    .

     Here

     are

     a

     few

    :
    


    1

    .

     Increased

     emphasis

     on

     ethical

     AI

    :

     As

     more

     people

     become

     aware

     of

     the

     negative

     impacts

     of

     AI

    ,

     there

     will

     be

     an

     increased

     focus

     on

     ethical

     AI

    .

     This

     includes

     developing

     AI

     that

     is

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     fair

     to

     all

     participants

    .
    


    2

    .

     Larger

     data

     sets

    :

     As

     more

     data

     is

     collected

     and

     analyzed

    ,

     the

     volume

     and

     complexity

     of

     the

     data

     set

     will

     increase

    .

     This

     will

     require

     more

     powerful

     computing

     resources

     and

     more

     sophisticated

     algorithms

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     the

     healthcare

     industry

    ,

     with

     applications

    



```python
llm.shutdown()
```
