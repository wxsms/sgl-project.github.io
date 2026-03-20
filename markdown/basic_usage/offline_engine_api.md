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

    [2026-03-20 03:17:56] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-20 03:17:56] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-20 03:17:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 03:17:58] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 03:17:59] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.


    [2026-03-20 03:17:59] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    [2026-03-20 03:17:59] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=1062167997, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.89it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:07,  2.24s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:55,  1.01it/s]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:55,  1.01it/s]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:55,  1.01it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.45it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.45it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.45it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:12,  4.18it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:06,  7.16it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:06,  7.16it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:06,  7.16it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:06,  7.16it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:06,  7.16it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:02<00:03, 11.44it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:02<00:03, 11.44it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:02<00:03, 11.44it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:02<00:03, 11.44it/s]

    Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:02<00:03, 11.44it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:02<00:02, 16.15it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:02<00:02, 16.15it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:02<00:02, 16.15it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:02<00:02, 16.15it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:02, 16.15it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:01, 20.03it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:01, 20.03it/s]

    Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:00, 28.77it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:00, 28.77it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 33.01it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 33.01it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 33.01it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 33.01it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 33.01it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 34.47it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 34.47it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 34.47it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 34.47it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 34.47it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 34.47it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 37.80it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 37.80it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 37.80it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 37.80it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 37.80it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 37.80it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 39.57it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 39.57it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 39.57it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 39.57it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 39.57it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 39.57it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 41.91it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 41.91it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 41.91it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 41.91it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 41.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.65 GB):   2%|▏         | 1/58 [00:00<00:07,  7.81it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.56 GB):   2%|▏         | 1/58 [00:00<00:07,  7.81it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.56 GB):   3%|▎         | 2/58 [00:00<00:07,  7.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.11 GB):   3%|▎         | 2/58 [00:00<00:07,  7.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.11 GB):   5%|▌         | 3/58 [00:00<00:07,  7.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.10 GB):   5%|▌         | 3/58 [00:00<00:07,  7.84it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.10 GB):   7%|▋         | 4/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.10 GB):   7%|▋         | 4/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.10 GB):   9%|▊         | 5/58 [00:00<00:06,  8.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.10 GB):   9%|▊         | 5/58 [00:00<00:06,  8.46it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=54.10 GB):  10%|█         | 6/58 [00:00<00:05,  8.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.10 GB):  10%|█         | 6/58 [00:00<00:05,  8.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.10 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.10 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.87it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.10 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.10 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.09 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.09 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.73it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=54.09 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.09 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.08 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.08 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.08 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.00it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=54.08 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.08 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.07 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.07 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.49it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=54.07 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.07 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.06 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.06 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.06 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.04 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.45it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=54.04 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.86it/s]Capturing num tokens (num_tokens=960 avail_mem=54.05 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.86it/s] Capturing num tokens (num_tokens=896 avail_mem=54.05 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.86it/s]Capturing num tokens (num_tokens=832 avail_mem=54.05 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.86it/s]Capturing num tokens (num_tokens=832 avail_mem=54.05 GB):  41%|████▏     | 24/58 [00:02<00:02, 15.05it/s]Capturing num tokens (num_tokens=768 avail_mem=54.04 GB):  41%|████▏     | 24/58 [00:02<00:02, 15.05it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.04 GB):  41%|████▏     | 24/58 [00:02<00:02, 15.05it/s]Capturing num tokens (num_tokens=704 avail_mem=54.04 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.80it/s]Capturing num tokens (num_tokens=640 avail_mem=54.04 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.80it/s]Capturing num tokens (num_tokens=576 avail_mem=54.03 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.80it/s]

    Capturing num tokens (num_tokens=576 avail_mem=54.03 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=512 avail_mem=54.03 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=480 avail_mem=54.04 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=480 avail_mem=54.04 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.13it/s]Capturing num tokens (num_tokens=448 avail_mem=54.04 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.13it/s]

    Capturing num tokens (num_tokens=416 avail_mem=54.04 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.13it/s]Capturing num tokens (num_tokens=416 avail_mem=54.04 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.59it/s]Capturing num tokens (num_tokens=384 avail_mem=54.04 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.59it/s]Capturing num tokens (num_tokens=352 avail_mem=54.03 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.59it/s]Capturing num tokens (num_tokens=352 avail_mem=54.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 12.98it/s]Capturing num tokens (num_tokens=320 avail_mem=54.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 12.98it/s]

    Capturing num tokens (num_tokens=288 avail_mem=54.02 GB):  59%|█████▊    | 34/58 [00:02<00:01, 12.98it/s]Capturing num tokens (num_tokens=288 avail_mem=54.02 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.86it/s]Capturing num tokens (num_tokens=256 avail_mem=54.02 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.86it/s]Capturing num tokens (num_tokens=240 avail_mem=54.02 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.86it/s]Capturing num tokens (num_tokens=224 avail_mem=54.01 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.86it/s]

    Capturing num tokens (num_tokens=224 avail_mem=54.01 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.36it/s]Capturing num tokens (num_tokens=208 avail_mem=54.01 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.36it/s]Capturing num tokens (num_tokens=192 avail_mem=54.01 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.36it/s]Capturing num tokens (num_tokens=192 avail_mem=54.01 GB):  71%|███████   | 41/58 [00:03<00:01, 14.91it/s]Capturing num tokens (num_tokens=176 avail_mem=54.00 GB):  71%|███████   | 41/58 [00:03<00:01, 14.91it/s]

    Capturing num tokens (num_tokens=160 avail_mem=54.00 GB):  71%|███████   | 41/58 [00:03<00:01, 14.91it/s]Capturing num tokens (num_tokens=160 avail_mem=54.00 GB):  74%|███████▍  | 43/58 [00:03<00:01, 13.12it/s]Capturing num tokens (num_tokens=144 avail_mem=54.00 GB):  74%|███████▍  | 43/58 [00:03<00:01, 13.12it/s]

    Capturing num tokens (num_tokens=128 avail_mem=53.99 GB):  74%|███████▍  | 43/58 [00:03<00:01, 13.12it/s]Capturing num tokens (num_tokens=128 avail_mem=53.99 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.23it/s]Capturing num tokens (num_tokens=112 avail_mem=53.99 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.23it/s]Capturing num tokens (num_tokens=96 avail_mem=53.99 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.23it/s] Capturing num tokens (num_tokens=96 avail_mem=53.99 GB):  81%|████████  | 47/58 [00:03<00:00, 13.44it/s]Capturing num tokens (num_tokens=80 avail_mem=53.98 GB):  81%|████████  | 47/58 [00:03<00:00, 13.44it/s]

    Capturing num tokens (num_tokens=64 avail_mem=53.98 GB):  81%|████████  | 47/58 [00:03<00:00, 13.44it/s]Capturing num tokens (num_tokens=48 avail_mem=53.98 GB):  81%|████████  | 47/58 [00:03<00:00, 13.44it/s]Capturing num tokens (num_tokens=48 avail_mem=53.98 GB):  86%|████████▌ | 50/58 [00:03<00:00, 15.51it/s]Capturing num tokens (num_tokens=32 avail_mem=53.98 GB):  86%|████████▌ | 50/58 [00:03<00:00, 15.51it/s]Capturing num tokens (num_tokens=28 avail_mem=53.97 GB):  86%|████████▌ | 50/58 [00:04<00:00, 15.51it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.97 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.09it/s]Capturing num tokens (num_tokens=24 avail_mem=53.15 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.09it/s]Capturing num tokens (num_tokens=20 avail_mem=53.15 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.09it/s]Capturing num tokens (num_tokens=20 avail_mem=53.15 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.81it/s]Capturing num tokens (num_tokens=16 avail_mem=53.14 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.81it/s]Capturing num tokens (num_tokens=12 avail_mem=53.14 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.81it/s]

    Capturing num tokens (num_tokens=12 avail_mem=53.14 GB):  97%|█████████▋| 56/58 [00:04<00:00, 15.09it/s]Capturing num tokens (num_tokens=8 avail_mem=53.13 GB):  97%|█████████▋| 56/58 [00:04<00:00, 15.09it/s] Capturing num tokens (num_tokens=4 avail_mem=53.13 GB):  97%|█████████▋| 56/58 [00:04<00:00, 15.09it/s]Capturing num tokens (num_tokens=4 avail_mem=53.13 GB): 100%|██████████| 58/58 [00:04<00:00, 12.95it/s]


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
    Generated text:  Hayley, I have a 6 month old puppy. I was wondering if you could help me out with some advice. 
    
    Here is the question:
    
    I have a very young puppy and I'm not sure if she needs anything to be done. I have been having problems with me not being able to get her to eat and she will only eat when I give her food. I am not sure if she is an emergency or she just needs to be let out to eat. 
    
    I was wondering if you could give some advice. I'm very new to this whole puppy care and I really appreciate any help you can provide. Thanks!
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government officer, and he is the chief executive of the executive branch. The president usually takes several important decisions, such as issuing the budget, making appointments to important positions, and so on. The head of a government department is a junior-level government official. The president is not only the highest level of government, but also the highest-ranking government officer. The president is the head of the executive branch. The head of a department is a lower-level government officer. The president is in charge of the government. The president can be removed if he violates certain rules, such as breaking laws. The president is appointed by the president of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the banks of the River Seine, which is also the longest river in Europe. The capital of France is located on the Île de la Cité, which is one of the oldest and most famous buildings in Paris. The city is divided into the north, central and south areas, and each area is separated by rivers and the Seine.
    It is the capital of France and is the seat of the French government. The city is the largest of France’s three regions and is an important transportation hub in Europe, with access to the Mediterranean Sea via the Seine and the Seine Tunnel. It is
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but there are certain areas where it will likely make a significant impact on our lives. One area that has shown promising results in recent years is in the field of healthcare. AI has been used to develop computerized diagnostic tools, which can help doctors diagnose diseases more accurately and quickly. This has the potential to improve the accuracy of diagnoses and potentially save lives.
    There are also various AI applications in the field of medicine, including in the development of medical robots and virtual reality. AI can be used to help doctors train and provide more personalized care to their patients.
    In addition, AI has also been used to develop predictive analytics, which can


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few key points about your background, skills, or personality]. I'm looking forward to meeting you and discussing how I can contribute to your team. What do you think is the most important thing for a [job title] to do? I believe that the most important thing for a [job title] to do is to [insert a few key points about your role or responsibilities]. I'm looking forward to hearing your thoughts on this
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the seat of the French government and the country's cultural and political center. Paris is a bustling metropolis with a diverse population and a rich cultural heritage, making it a popular tourist destination. The city is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. Paris is also known for its cuisine, including its famous Parisian cuisine and its famous wine. The city is a major hub for business, finance, and entertainment, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI becomes more sophisticated, it is likely to be used in more complex and personalized ways, with the potential to revolutionize the way we treat and diagnose diseases.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce waste, and improve quality. As AI technology continues to improve, it is likely
    


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
    Generated text:  Jane. I am a 26-year-old journalist with a passion for investigative journalism. I write articles that shed light on issues affecting people's lives and that use data and research to inform readers. I've been in the field for over five years and I love the thrill of seeing my stories catch people's attention. If you have a question or want to learn more about me, feel free to reach out. I look forward to helping you! What a fascinating and intriguing self-introduction! What kind of stories do you typically write? Jane, you're a passionate journalist with a keen eye for detail. What makes your stories unique and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the 6th most populous city in the world and the largest city in Europe. It is a historic and cultural center of France and one of the world's most important cities. The city is renowned for its rich history, art, music, and cuisine. It has many landmarks such as the Eiffel Tower, Notre-Dame Cathedral, the Louvre Museum, and the Champs-Élysées. Paris is also the birthplace of numerous world-famous figures such as Louis XIV, Napoleon Bonaparte, and French playwright Jean-Paul Sartre. Its climate is temperate and is known for its mild
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and complex. However, several trends are likely to shape the way we use and develop AI in the coming years. Here are some possible future trends in AI:
    
    1. Autonomous vehicles: As more companies invest in AI, autonomous vehicles are becoming more common. These vehicles will be able to navigate the roads and handle traffic signals autonomously, which could lead to reduced accidents and improve traffic flow.
    
    2. Personalized AI: AI will be able to learn and adapt to the individual needs and preferences of people, leading to more personalized and efficient solutions.
    
    3. Biometrics: Biometric authentication will become increasingly important, as more people are


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

    Your

     Profession

    ]

     from

     [

    Your

     Country

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

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Purpose

     or

     Mission

    ].

     I

    've

     always

     wanted

     to

     help

     others

     and

     believe

     that

     I

     can

     make

     a

     positive

     impact

     on

     the

     world

    .

     And

     so

    ,

     I

    'm

     excited

     to

     dive

     into

     this

     role

     as

     a

     [

    Your

     Job

     Title

    ]

     and

     continue

     to

     make

     a

     difference

     in

     the

     lives

     of

     others

    .

     Thank

     you

     for

     the

     opportunity

     to

     introduce

     myself

    .

     What

     is

     your

     profession

     and

     what

     is

     your

     interest

     or

     hobby

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

     a

     profession

     or

     an

     interest

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     light

     and

     op

    ulence

    ,

     renowned

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

     Dame

     Cathedral

    .
    


    A

     sentence

     that

     compares

     the

     fashion

     industry

     in

     Paris

     to

     a

     jungle

     and

     the

     streets

     to

     a

     beach

    .

     


    French

     fashion

     is

     like

     a

     jungle

     with

     a

     rich

     tape

    stry

     of

     fabrics

    ,

     patterns

    ,

     and

     styles

     that

     have

     developed

     over

     centuries

    ,

     with

     Paris

     being

     the

     heart

     and

     soul

     of

     the

     fashion

     industry

    .

     The

     streets

     of

     Paris

     are

     a

     vast

     and

     lively

     space

    ,

     with

     a

     mix

     of

     high

    -end

     brands

     and

     street

    wear

     brands

     competing

     for

     attention

    .

     The

     streets

     are

     bustling

     with

     energy

     and

     energy

    ,

     with

     a

     mix

     of

     vendors

    ,

     shoppers

    ,

     and

     shoppers

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     dominated

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     reliance

     on

     machine

     learning

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

     rely

     more

     heavily

     on

     machine

     learning

     algorithms

     to

     automate

     tasks

    ,

     solve

     complex

     problems

     and

     make

     better

     decisions

    .
    


    2

    .

     Deep

     learning

    :

     Deep

     learning

     is

     a

     subset

     of

     AI

     that

     is

     focused

     on

     creating

     models

     with

     multiple

     layers

     that

     can

     learn

     complex

     patterns

     in

     data

    .

     This

     type

     of

     AI

     is

     expected

     to

     become

     more

     prevalent

     as

     AI

     is

     able

     to

     process

     and

     analyze

     large

     amounts

     of

     data

     at

     an

     unprecedented

     speed

    .
    


    3

    .

     Explain

    ability

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     it

     is

     expected

     to

     become

     more

     explain

    able

    .

     This

     means

     that

     AI

    



```python
llm.shutdown()
```
