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

    [2026-03-20 01:06:22] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-20 01:06:22] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-20 01:06:22] INFO utils.py:164: NumExpr defaulting to 16 threads.



    config.json:   0%|          | 0.00/659 [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/242 [00:00<?, ?B/s]


    [2026-03-20 01:06:25] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 01:06:25] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.


    [2026-03-20 01:06:25] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    [2026-03-20 01:06:25] INFO engine.py:177: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=694133173, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)



    tokenizer_config.json: 0.00B [00:00, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:18,  2.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:18,  2.42s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:59,  1.06s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:59,  1.06s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:59,  1.06s/it]

    Compiling num tokens (num_tokens=6144):   3%|▎         | 2/58 [00:02<00:59,  1.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:17,  2.99it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:17,  2.99it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:17,  2.99it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:17,  2.99it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:09,  5.41it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:09,  5.41it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:09,  5.41it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:02<00:09,  5.41it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:02<00:09,  5.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:04,  9.37it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]

    Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 16.14it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:02, 16.14it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:03<00:00, 43.81it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]

    Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 50.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 60.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=96.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=96.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=96.65 GB):   3%|▎         | 2/58 [00:00<00:04, 11.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=96.65 GB):   3%|▎         | 2/58 [00:00<00:04, 11.24it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=96.65 GB):   3%|▎         | 2/58 [00:00<00:04, 11.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=96.65 GB):   7%|▋         | 4/58 [00:00<00:04, 12.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=96.64 GB):   7%|▋         | 4/58 [00:00<00:04, 12.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=96.64 GB):   7%|▋         | 4/58 [00:00<00:04, 12.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=96.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=96.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.77it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=96.63 GB):  10%|█         | 6/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=96.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=96.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=96.50 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=96.61 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=96.61 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=96.61 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.29it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=96.60 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=96.52 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=96.52 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=96.50 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=96.50 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=96.58 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=96.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=96.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.38it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=96.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=96.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=96.53 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=96.53 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.47it/s]Capturing num tokens (num_tokens=960 avail_mem=96.54 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.47it/s] Capturing num tokens (num_tokens=896 avail_mem=96.53 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.47it/s]Capturing num tokens (num_tokens=832 avail_mem=96.53 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.47it/s]Capturing num tokens (num_tokens=832 avail_mem=96.53 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.68it/s]Capturing num tokens (num_tokens=768 avail_mem=96.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.68it/s]

    Capturing num tokens (num_tokens=704 avail_mem=96.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.68it/s]Capturing num tokens (num_tokens=640 avail_mem=96.51 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.68it/s]Capturing num tokens (num_tokens=576 avail_mem=96.50 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.68it/s]Capturing num tokens (num_tokens=576 avail_mem=96.50 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.77it/s]Capturing num tokens (num_tokens=512 avail_mem=96.50 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.77it/s]Capturing num tokens (num_tokens=480 avail_mem=96.52 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.77it/s]Capturing num tokens (num_tokens=448 avail_mem=96.51 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.77it/s]Capturing num tokens (num_tokens=416 avail_mem=96.51 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.77it/s]

    Capturing num tokens (num_tokens=416 avail_mem=96.51 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=384 avail_mem=96.50 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=352 avail_mem=96.49 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=320 avail_mem=96.48 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=288 avail_mem=96.46 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.09it/s]Capturing num tokens (num_tokens=288 avail_mem=96.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.60it/s]Capturing num tokens (num_tokens=256 avail_mem=96.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.60it/s]Capturing num tokens (num_tokens=240 avail_mem=96.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.60it/s]Capturing num tokens (num_tokens=224 avail_mem=96.44 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.60it/s]Capturing num tokens (num_tokens=208 avail_mem=96.44 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.60it/s]

    Capturing num tokens (num_tokens=208 avail_mem=96.44 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=192 avail_mem=96.43 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=176 avail_mem=96.45 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=160 avail_mem=96.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=144 avail_mem=96.41 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=144 avail_mem=96.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=128 avail_mem=96.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=112 avail_mem=96.42 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=96 avail_mem=96.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.30it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=96.39 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.30it/s]Capturing num tokens (num_tokens=80 avail_mem=96.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=64 avail_mem=96.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=48 avail_mem=96.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=32 avail_mem=96.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=28 avail_mem=96.38 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=28 avail_mem=96.38 GB):  90%|████████▉ | 52/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=24 avail_mem=96.37 GB):  90%|████████▉ | 52/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=20 avail_mem=96.37 GB):  90%|████████▉ | 52/58 [00:01<00:00, 33.41it/s]

    Capturing num tokens (num_tokens=16 avail_mem=96.36 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.41it/s]Capturing num tokens (num_tokens=12 avail_mem=96.35 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.41it/s]Capturing num tokens (num_tokens=12 avail_mem=96.35 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.15it/s]Capturing num tokens (num_tokens=8 avail_mem=96.35 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.15it/s] Capturing num tokens (num_tokens=4 avail_mem=96.34 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.15it/s]Capturing num tokens (num_tokens=4 avail_mem=96.34 GB): 100%|██████████| 58/58 [00:02<00:00, 27.19it/s]


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
    Generated text:  Xiaohui and I am a doctor. I have developed my own ideas on health care and therapy, which I want to share with you. To start, I'd like to discuss the relationship between heart disease and cancer. My personal experience tells me that heart disease, even if I do not have it, will potentially spread to the other organ if I have cancer. To put it another way, if you have cancer and have a heart disease, it is very likely that cancer will spread to the heart. Therefore, if you have cancer, it is better to do something about it. However, the opposite is not true: if you
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who governs the country of the United States. That person is the President of the United States. The president is the head of the executive branch of the government of the United States. The United States is a federal country, which means that the government of the country is divided into two branches: a legislature (legislative branch) and an executive branch (executive branch).
    
    How does the president of the United States work?
    
    The president of the United States works by appointing other government officials. The president must nominate an official for the position of president. The nomination is submitted to Congress, which is the legislative branch of the
    ===============================
    Prompt: The capital of France is
    Generated text:  in which country?
    The capital of France is Paris. France is an Alpine country, located on the Rhine river, at the crossroads of Europe and Asia. Paris is the capital of France, located in the north-central region of the country. Paris is the largest city in Europe by area and by population, making it the 9th largest city in the world by land area. The city is an important economic and cultural hub of France, and has a rich history and culture. Paris has a unique mix of old-world charm and modernity, making it a popular destination for tourists and locals alike. The city is home to numerous
    ===============================
    Prompt: The future of AI is
    Generated text:  set to grow at a breathtaking pace. But just like any other technological advance, the speed of development can have a significant impact on society. And it's important to understand what impact it can have, especially in the field of finance. While there are many benefits to having a well-functioning and secure financial system, there are also risks that need to be managed. In this article, we'll explore the potential impacts of AI on the finance sector and how it could impact the future of finance as a whole. Let's dive into the topic!
    AI has been a buzzword for the past few years, and it's quickly becoming a defining


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


    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm a [Occupation] who has always been [Positive Traits] and [Negative Traits]. I'm passionate about [My Passion], and I'm always looking for ways to [My Goals]. I'm always learning and growing, and I'm always looking for new experiences and opportunities to grow. I'm a [Positive Traits] person who is always [Positive Traits]. I'm a [Positive Traits] person who is always [Positive Traits]. I'm a [Positive Traits] person who is always [Positive Traits]. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is located on the Seine River and is the seat of government for the country. Paris is known for its rich history, art, and cuisine. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is known for its fashion industry and is home to many fashion houses. Paris is a popular tourist destination and is a major economic center in France. It is also a cultural center and is home to many museums, theaters, and other cultural institutions. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn and adapt to new situations. This will enable AI to perform tasks that are currently beyond human capability, such as playing chess or playing the stock market.
    
    2. Enhanced machine learning: Machine learning will become more sophisticated, allowing AI systems to learn from data and make more accurate predictions and decisions. This will enable AI to perform tasks that are currently beyond human capability, such as diagnosing diseases or predicting weather patterns.
    
    3. Improved privacy and security: AI systems will
    


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
    Generated text:  [Name] and I'm a [age] year old [gender]. I'm a self-proclaimed [occupation] and have a keen interest in [field of interest]. I enjoy [why I love this field of interest]. I'm passionate about [why I love this subject] and [why I like this subject] as a whole. I believe in [why I believe in this subject], and I am determined to [why I am determined to do this]. I'm always ready to learn and grow, and I am always looking for ways to improve myself. I'm a [career goal] and I plan to [how I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights, and it is the largest city in the world by population.
    The answer is 3. Paris is the largest city in the world by population. 
    Explain in detail:
    1. The answer is 3 because it is the largest city in the world by population, which is defined as the total number of people living in an urban area. Paris, being the capital of France, is the largest city in the European Union (EU) and the second-largest city in the world by population.
    2. To determine the population of Paris, one must refer to official sources such as the French government, World
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued growth and development, with the following key trends expected to emerge:
    
    1. Increased use of AI in healthcare: AI will continue to play a critical role in healthcare, with advancements in machine learning and artificial intelligence being applied to diagnose and treat diseases, predict outcomes, and optimize treatment plans.
    
    2. Enhanced cybersecurity: With the increasing number of cyber attacks on AI systems, there will be a greater focus on improving cybersecurity measures and implementing robust security protocols.
    
    3. Increased automation: AI will continue to automate routine tasks and reduce the need for human labor, leading to increased productivity and efficiency.
    
    4. Greater integration of AI with other


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

    occupation

     or

     profession

    ]

     who

     has

     [

    what

     you

     do

     best

    ].

     I

     enjoy

     [

    what

     I

     do

     for

     a

     living

    ],

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     improve

     myself

     and

     learn

     new

     things

    .

     What

     exc

    ites

     you

     most

     about

     your

     profession

     or

     occupation

    ?

     As

     an

     AI

     language

     model

    ,

     I

     do

     not

     have

     a

     physical

     presence

    ,

     but

     my

     goal

     is

     to

     assist

     and

     improve

     people

    's

     lives

     by

     providing

     knowledge

     and

     insights

     to

     those

     who

     interact

     with

     me

    .

     I

     am

     always

     here

     to

     help

    ,

     and

     I

     am

     eager

     to

     learn

     from

     others

     and

     grow

     as

     a

     person

    .

     What

     would

     you

     like

     to

     say

     about

     yourself

    ?

     You

     can

     say

     anything

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     is

     the

     seat

     of

     the

     government

     of

     France

     and

     the

     fifth

    -largest

     city

     in

     the

     European

     Union

    .

     The

     city

     is

     home

     to

     many

     cultural

     and

     historical

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

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

     known

     for

     its

     diverse

     food

     scene

    ,

     including

     gourmet

     cuisine

     and

     Paris

    -inspired

     cafes

    .

     French

     culture

     is

     deeply

     rooted

     in

     French

     and

     international

     styles

    ,

     and

     Paris

     is

     a

     popular

     tourist

     destination

     for

     its

     scenic

     views

    ,

     delicious

     food

    ,

     and

     vibrant

     culture

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     the

     ancient

     Greeks

     and

     Romans

    ,

     and

     has

     been

     an

     important

     center

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     some

     possible

     trends

     that

     are

     expected

     to

     shape

     the

     industry

     include

    :
    


    1

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     self

    -driving

     cars

     to

     virtual

     assistants

     that

     can

     understand

     and

     respond

     to

     our

     needs

    .
    


    2

    .

     Increased

     focus

     on

     ethics

     and

     safety

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     growing

     emphasis

     on

     ethical

     considerations

     and

     the

     safety

     of

     AI

     systems

    .
    


    3

    .

     Growth

     in

     AI

    -A

    I

     collaboration

    :

     AI

     and

     AI

    -A

    I

     collaboration

     will

     become

     more

     common

    ,

     with

     more

     companies

     and

     research

     institutions

     collaborating

     to

     develop

     new

     AI

     technologies

     and

     to

     share

     knowledge

     and

     resources

    .
    


    4

    .

     Adv

    ancements

     in

     AI

     for

    



```python
llm.shutdown()
```
