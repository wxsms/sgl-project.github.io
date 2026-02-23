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

    [2026-02-23 07:32:49] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-23 07:32:49] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-23 07:32:49] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-23 07:32:52] INFO server_args.py:1835: Attention backend not specified. Use fa3 backend by default.


    [2026-02-23 07:32:52] INFO server_args.py:2886: Set soft_watchdog_timeout since in CI


    [2026-02-23 07:32:52] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=582113763, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.69it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=4.53 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=4.53 GB):   5%|▌         | 1/20 [00:00<00:03,  5.61it/s]Capturing batches (bs=120 avail_mem=4.43 GB):   5%|▌         | 1/20 [00:00<00:03,  5.61it/s]

    Capturing batches (bs=112 avail_mem=4.42 GB):   5%|▌         | 1/20 [00:00<00:03,  5.61it/s]Capturing batches (bs=104 avail_mem=4.42 GB):   5%|▌         | 1/20 [00:00<00:03,  5.61it/s]Capturing batches (bs=104 avail_mem=4.42 GB):  20%|██        | 4/20 [00:00<00:00, 16.31it/s]Capturing batches (bs=96 avail_mem=4.42 GB):  20%|██        | 4/20 [00:00<00:00, 16.31it/s] Capturing batches (bs=88 avail_mem=4.41 GB):  20%|██        | 4/20 [00:00<00:00, 16.31it/s]Capturing batches (bs=80 avail_mem=4.41 GB):  20%|██        | 4/20 [00:00<00:00, 16.31it/s]Capturing batches (bs=80 avail_mem=4.41 GB):  35%|███▌      | 7/20 [00:00<00:00, 21.03it/s]Capturing batches (bs=72 avail_mem=4.40 GB):  35%|███▌      | 7/20 [00:00<00:00, 21.03it/s]

    Capturing batches (bs=64 avail_mem=4.40 GB):  35%|███▌      | 7/20 [00:00<00:00, 21.03it/s]Capturing batches (bs=56 avail_mem=4.39 GB):  35%|███▌      | 7/20 [00:00<00:00, 21.03it/s]Capturing batches (bs=56 avail_mem=4.39 GB):  50%|█████     | 10/20 [00:00<00:00, 24.05it/s]Capturing batches (bs=48 avail_mem=4.39 GB):  50%|█████     | 10/20 [00:00<00:00, 24.05it/s]Capturing batches (bs=40 avail_mem=4.38 GB):  50%|█████     | 10/20 [00:00<00:00, 24.05it/s]Capturing batches (bs=32 avail_mem=4.38 GB):  50%|█████     | 10/20 [00:00<00:00, 24.05it/s]Capturing batches (bs=32 avail_mem=4.38 GB):  65%|██████▌   | 13/20 [00:00<00:00, 25.82it/s]Capturing batches (bs=24 avail_mem=4.37 GB):  65%|██████▌   | 13/20 [00:00<00:00, 25.82it/s]

    Capturing batches (bs=16 avail_mem=4.37 GB):  65%|██████▌   | 13/20 [00:00<00:00, 25.82it/s]Capturing batches (bs=12 avail_mem=4.36 GB):  65%|██████▌   | 13/20 [00:00<00:00, 25.82it/s]Capturing batches (bs=12 avail_mem=4.36 GB):  80%|████████  | 16/20 [00:00<00:00, 24.30it/s]Capturing batches (bs=8 avail_mem=4.36 GB):  80%|████████  | 16/20 [00:00<00:00, 24.30it/s] Capturing batches (bs=4 avail_mem=4.35 GB):  80%|████████  | 16/20 [00:00<00:00, 24.30it/s]Capturing batches (bs=2 avail_mem=4.34 GB):  80%|████████  | 16/20 [00:00<00:00, 24.30it/s]Capturing batches (bs=1 avail_mem=4.34 GB):  80%|████████  | 16/20 [00:00<00:00, 24.30it/s]

    Capturing batches (bs=1 avail_mem=4.34 GB): 100%|██████████| 20/20 [00:00<00:00, 27.64it/s]Capturing batches (bs=1 avail_mem=4.34 GB): 100%|██████████| 20/20 [00:00<00:00, 23.72it/s]


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
    Generated text:  Alex. I’m from New York City. I have a lot of friends. I get up at 6:30 every morning and go to a coffee shop to get my coffee. I usually get a cup of coffee, some bread and some juice. I’m a vegetarian. I enjoy my meals. I usually get out of the coffee shop around 11:00 p. m. I like the way I sleep. I sleep for about 8 hours every night. I like to listen to my favorite songs. I like to watch movies and I like to read books. I take a long walk outside every morning.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a holder of the position of president of the United States, but the president does not have the power to appoint the chief executive officer of a government. If a president is removed from office, they are not automatically disqualified from running for another office. However, a president who loses an election is no longer eligible to hold office. The president cannot be replaced by another person.
    Based on these facts, what can be inferred about the president of the United States? Options:  *not enough information  *The president has the right to appoint the chief executive officer of a government.  *The president has the power to create laws.  *
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. London
    C. Rome
    D. Moscow
    Answer: A
    
    Which of the following statements about the powers of the National People's Congress is incorrect? ( )
    A. The National People's Congress has the power to make decisions on the establishment and supervision of courts
    B. The National People's Congress has the power to make decisions on the establishment and supervision of procuratorial organs
    C. The National People's Congress has the power to make decisions on the establishment and supervision of courts and procuratorial organs
    D. The National People's Congress has the power to make decisions on the establishment and supervision
    ===============================
    Prompt: The future of AI is
    Generated text:  here, so it’s important to be prepared for the challenges and opportunities that come with it.
    In the coming years, artificial intelligence (AI) will play a crucial role in various industries and applications. It will be used for tasks that are currently done manually or with limited automation, such as analyzing data, making decisions, and providing personalized experiences.
    However, AI will also bring about new challenges. The increasing availability and affordability of AI technology will lead to a rise in the number of jobs that require human-like capabilities. This will put pressure on the human workforce to adapt and change, which can be both challenging and rewarding.
    There are several challenges


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or achievement]. I'm [age] years old, and I'm [occupation] with a [number of years] of experience in the industry. I'm [reason for passion] and I'm always looking for ways to [action or achievement]. I'm [age] years old, and I'm [occupation] with a [number of years] of experience in the industry. I'm [reason for passion] and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history dating back to the Roman Empire. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest fashion events in the world. Paris is a cultural hub with a diverse population and a rich history, making it a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more efficient and effective at performing tasks that were previously done by humans. This could lead to increased automation in industries such as manufacturing, transportation, and healthcare.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing need for ethical guidelines and regulations to ensure that AI is used in a responsible and fair
    


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
    Generated text:  ____________. I'm a/an ____________ (such as sports star, artist, scientist, etc.). I'm a/an ____________. I'm ____________. I'm ____________. I'm ____________. I'm ____________. I'm ____________. I'm ____________. What do you think of me? How do you like my character? How do you think I should be written? What do you think about my character? What do you think of my personality? What do you think about my appearance? What do you think about my background? What do you think about my abilities? What do you think about my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its rich history, iconic landmarks, and French cuisine. The city is also home to many famous museums, art galleries, and theaters. Its famous landmarks such as the Eiffel Tower and Louvre Museum are well-known globally. Paris is a popular tourist destination, hosting many international events and festivals. The city is also known for its rich cultural heritage, including the Louvre Museum, Notre-Dame Cathedral, and various art and architecture sites. Overall, Paris is a culturally rich, cosmopolitan city with a vibrant and diverse community. 
    You are an AI assistant. You will be given a task. You
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising, with the potential to transform almost every aspect of our lives. Here are some potential future trends in AI:
    
    1. Increased privacy and security: As AI systems become more advanced, they will become even more personal and intimate. This means that users will have more control over the information they share and the AI systems will have to protect their data from breaches and cyber attacks.
    
    2. Autonomous vehicles: AI is already being used in self-driving cars, but there is a lot of work to be done before fully autonomous vehicles become a reality. As AI technology improves, autonomous vehicles will become more capable and affordable, potentially reducing the need for


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

     [

    Age

    ].

     I

     was

     born

     in

     [

    Birth

    place

    ],

     and

     I

     currently

     live

     in

     [

    Current

     Residence

    ].

     My

     love

     for

     literature

     has

     led

     me

     to

     explore

     various

     writing

     projects

    .

     My

     most

     recent

     venture

     was

     an

     experimental

     novel

    ,

     which

     was

     published

     online

     last

     month

    .

     I

    'm

     passionate

     about

     creating

     meaningful

     and

     engaging

     stories

     that

     inspire

     and

     entertain

    .

     What

    ’s

     your

     style

     of

     writing

    ?

     I

    'm

     not

     afraid

     to

     take

     risks

     and

     try

     new

     things

     in

     my

     creative

     pursuits

    .

     What

     would

     you

     like

     to

     share

     with

     me

     about

     yourself

    ?

     I

     love

     exploring

     the

     world

     and

     imm

    ers

    ing

     myself

     in

     different

     cultures

     and

     experiences

    .

     What

     brings

     you

     to

     this

     moment

    ,

     [

    Person

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     third

     most

     populous

     city

     in

     the

     world

     and

     is

     home

     to

     many

     of

     the

     country

    ’s

     cultural

     and

     political

     centers

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     vibrant

     streets

    ,

     and

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     home

     to

     the

     headquarters

     of

     many

     of

     France

    ’s

     largest

     and

     most

     influential

     companies

    ,

     and

     is

     a

     major

     center

     for

     art

    ,

     fashion

    ,

     and

     entertainment

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     major

     tourist

     destination

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     the

     Arc

     de

     Tri

    omp

    he

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     increased

     data

     availability

    ,

     and

     the

     development

     of

     new

     algorithms

    .

     Some

     possible

     trends

     that

     are

     likely

     to

     be

     seen

     in

     the

     future

     include

    :
    


    1

    .

     Increased

     AI

     transparency

    :

     With

     the

     use

     of

     machine

     learning

     and

     other

     AI

     technologies

    ,

     it

     is

     becoming

     increasingly

     important

     to

     ensure

     that

     AI

     systems

     are

     transparent

     and

     understandable

     to

     users

    .

     This

     means

     that

     we

     will

     see

     a

     greater

     emphasis

     on

     data

     privacy

     and

     security

    ,

     as

     well

     as

     efforts to

     ensure

     that

     AI

     systems

     are

     developed

     and

     used

     in

     a

     way

     that

     align

    s

     with

     ethical

     and

     moral

     principles

    .
    


    2

    .

     AI

    -driven

     automation

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

    



```python
llm.shutdown()
```
