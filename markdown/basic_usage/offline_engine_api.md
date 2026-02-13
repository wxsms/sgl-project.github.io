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

    [2026-02-13 21:33:08] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-13 21:33:08] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-13 21:33:08] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-13 21:33:11] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.


    [2026-02-13 21:33:11] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-13 21:33:11] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=436263255, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_enable_fake_auto=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.73it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=49.54 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=49.54 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=120 avail_mem=49.43 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=112 avail_mem=49.42 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=112 avail_mem=49.42 GB):  15%|█▌        | 3/20 [00:00<00:03,  4.64it/s]Capturing batches (bs=104 avail_mem=49.42 GB):  15%|█▌        | 3/20 [00:00<00:03,  4.64it/s]

    Capturing batches (bs=96 avail_mem=49.42 GB):  15%|█▌        | 3/20 [00:00<00:03,  4.64it/s] 

    Capturing batches (bs=96 avail_mem=49.42 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.30it/s]Capturing batches (bs=88 avail_mem=49.41 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.30it/s]Capturing batches (bs=80 avail_mem=49.41 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.30it/s]Capturing batches (bs=80 avail_mem=49.41 GB):  35%|███▌      | 7/20 [00:01<00:02,  6.32it/s]Capturing batches (bs=72 avail_mem=49.40 GB):  35%|███▌      | 7/20 [00:01<00:02,  6.32it/s]

    Capturing batches (bs=64 avail_mem=49.40 GB):  35%|███▌      | 7/20 [00:01<00:02,  6.32it/s]Capturing batches (bs=64 avail_mem=49.40 GB):  45%|████▌     | 9/20 [00:01<00:01,  6.55it/s]Capturing batches (bs=56 avail_mem=49.39 GB):  45%|████▌     | 9/20 [00:01<00:01,  6.55it/s]Capturing batches (bs=48 avail_mem=49.39 GB):  45%|████▌     | 9/20 [00:01<00:01,  6.55it/s]Capturing batches (bs=40 avail_mem=49.38 GB):  45%|████▌     | 9/20 [00:01<00:01,  6.55it/s]Capturing batches (bs=40 avail_mem=49.38 GB):  60%|██████    | 12/20 [00:01<00:00,  9.87it/s]Capturing batches (bs=32 avail_mem=49.38 GB):  60%|██████    | 12/20 [00:01<00:00,  9.87it/s]

    Capturing batches (bs=24 avail_mem=49.37 GB):  60%|██████    | 12/20 [00:01<00:00,  9.87it/s]Capturing batches (bs=16 avail_mem=49.37 GB):  60%|██████    | 12/20 [00:01<00:00,  9.87it/s]Capturing batches (bs=16 avail_mem=49.37 GB):  75%|███████▌  | 15/20 [00:01<00:00, 12.12it/s]Capturing batches (bs=12 avail_mem=49.36 GB):  75%|███████▌  | 15/20 [00:01<00:00, 12.12it/s]Capturing batches (bs=8 avail_mem=49.36 GB):  75%|███████▌  | 15/20 [00:01<00:00, 12.12it/s] Capturing batches (bs=4 avail_mem=49.35 GB):  75%|███████▌  | 15/20 [00:02<00:00, 12.12it/s]

    Capturing batches (bs=4 avail_mem=49.35 GB):  90%|█████████ | 18/20 [00:02<00:00, 15.25it/s]Capturing batches (bs=2 avail_mem=49.35 GB):  90%|█████████ | 18/20 [00:02<00:00, 15.25it/s]Capturing batches (bs=1 avail_mem=49.34 GB):  90%|█████████ | 18/20 [00:02<00:00, 15.25it/s]Capturing batches (bs=1 avail_mem=49.34 GB): 100%|██████████| 20/20 [00:02<00:00,  9.52it/s]


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
    Generated text:  Ted and I'm a health coach. I help people understand the science of health, diet, and exercise in order to lose weight, improve their health, and maintain a healthy lifestyle. Here's what I do: (a) I provide patient-centered care in the comfort of a private practice. (b) I have a background in nutritional science. (c) I teach workshops for weight management (e.g., weight loss, nutrition, exercise, personal finance). (d) I provide webinars to help people learn about health. (e) I have a background in exercise science. (f) I have a background in public health
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. ____
    A. 正确
    B. 错误
    答案:
    
    A
    
    八轴飞行器，从空中某一点以等速度竖直下落，第1、2、3轴随风向迎风的方向移动，第4、5、6、7、8轴随风向背风的方向移动，则相对于地面飞行的倾斜角度为
    A. 0°
    B. 90°
    C. 45°
    答案:
    
    A
    
    中国共产党第十九届中央委员会第四次全体会议提出，坚持和完善中国特色社会主义____，坚持和完善人民代表大会制度、
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is in the south of France, and the city is the main economic, cultural, and political center of France. With over one million inhabitants, Paris is the largest city in Europe.
    
    巴黎是法国的首都。它是法国南部的重要城市，是法国的主要经济、文化和政治中心。目前，巴黎的人口约为1000万，是法国人口最多的城市。 
    
    The capital of France is London. London is the second most populous city in the UK and the largest city in the European Union. It is located on the River Thames in the south of England. London is the political, cultural, economic
    ===============================
    Prompt: The future of AI is
    Generated text:  about creativity and innovation.
    Creating and implementing AI solutions is becoming one of the most popular and flexible tools for businesses to create a competitive advantage. The number of AI solutions used in the business world has steadily risen over the years. Companies that have been successful in implementing AI solutions are typically those that have invested in AI to create a competitive advantage.
    AI is one of the most powerful tools that a company can utilize to improve their business operations. The technology has become so sophisticated and capable that it is revolutionizing the way businesses work. The technology is also bringing about a number of changes in the way businesses operate. With the rise of AI, many


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. How can I help you today? I look forward to meeting you and discussing how I can assist you. [Name] [Company name] [Company website] [LinkedIn profile] [Twitter handle] [Facebook page] [Email address] [Phone number] [Optional: any other relevant information] [Optional: any additional information about your background or experience] [Optional: any specific
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the largest city in France and is known for its fashion, art, and cuisine. The city is also home to many important historical sites and landmarks, including the Louvre, the Notre-Dame Cathedral, and the Champs-Élysées. Paris is a city of contrasts, with its modern architecture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives through the use of voice assistants like Siri and Alexa, and in the development of self-driving cars. As AI becomes more integrated into our daily lives, we may see even more widespread adoption of AI in areas such as healthcare, finance, and transportation.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ensuring that AI is
    


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
    Generated text:  [name], and I'm a/an [occupation], with a keen interest in [project/field]. I'm a/an [age], [gender], [race] and [nationality] [occupation]. I currently work as a/an [job], but I've always been fascinated by the [challenge], and I'm [attitude] about it. I'm always looking for ways to [use my skills] and I enjoy [relate to my field], and I'm [attitude]. I'm a/an [age], [gender], [race] and [nationality] [occupation]. I'm a/an [att
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. The city is known for its historical and cultural landmarks, such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. Paris is also famous for its food, fashion, and music scenes. It is home to the European Parliament and many other important organizations. Paris is a major cultural and tourist center and attracts millions of visitors each year. It is also an important center for science, research, and education in France. Additionally, Paris is home to numerous museums, including the Louvre, Musée d'Orsay, and Musée d'Orsay. Lastly, Paris is known for its fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and diverse, with many potential directions for innovation. Here are some possible future trends in artificial intelligence:
    
    1. Automation and smart automation: AI is already being used to automate tasks and improve efficiency. We may see further automation of mundane tasks, such as data entry, logistics, and customer service. We may also see the development of "smart" robots and machines that can learn and adapt to their environment, without the need for human intervention.
    
    2. Advanced machine learning: Machine learning is being used to create increasingly sophisticated models that can learn from data on their own. This could lead to the creation of more sophisticated models that can learn to


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

    career

     field

    ]

     specialist

    .

     I

     specialize

     in

     [

    job

     title

    ]

     and

     have

     over

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     the

     field

    .

     I

     bring

     a

     [

    insert

     skills

    ,

     experience

    ,

     knowledge

    ]

     to

     the

     table

     and

     am

     committed

     to

     [

    reason

     for

     your

     experience

    ]

     in

     [

    career

     field

    ].

     I

     believe

     in

     [

    reason

     for

     your

     experience

    ].

     Whether

     it

    's

     problem

    -solving

    ,

     customer

     service

    ,

     or

     training

    ,

     I

     am

     here

     to

     help

     you

     succeed

    .

     What

     is

     your

     career

     field

     and

     what

     is

     your

     specialty

    ?

     


    [

    Name

    ]:

     [

    Career

     Field

    ]


    [

    Name

    ]:

     [

    Job

     Title

    ]


    [

    Name

    ]:

     [

    Field

    ]


    [

    Name

    ]:

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Love

    ".


    You

     are

     to

     answer

     this

     question

    :

     Where

     is

     the

     capital

     city

     of

     France

    ?

     Paris

     is

     the

     capital

     of

     France

    .


    Your

     answer

     should

     be

     limited

     to

     this

    :

     Paris

    ,

     France

    .

     Paris

     is

     the

     capital

     of

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     increased

     integration

     with

     natural

     language

     processing

    ,

     machine

     learning

    ,

     and

     deep

     learning

    .

     As

     a

     result

    ,

     we

     can

     expect

     more

     complex

     algorithms

     and

     tools

    ,

     which

     will

     enable

     AI

     to

     perform

     tasks

     that

     were

     previously

     thought

     impossible

    .

     We

     can

     also

     expect

     more

     advanced

     AI

     systems

     that

     can

     better

     understand

     and

     adapt

     to

     human

     behavior

    ,

     leading

     to

     more

     efficient

     and

     effective

     decision

    -making

    .

     Additionally

    ,

     we

     can

     expect

     AI

     to

     be

     more

     integrated

     with

     the

     physical

     world

    ,

     enabling

     more

     advanced

     robotic

     and

     autonomous

     systems

    .

     Finally

    ,

     we

     can

     expect

     AI

     to

     be

     more

     accessible

     to

     people

     of

     all

     ages

     and

     backgrounds

    ,

     making

     it

     more

     likely

     that

     everyone

     will

     benefit

     from

     its

     potential

     benefits

    .

     However

    ,

    



```python
llm.shutdown()
```
