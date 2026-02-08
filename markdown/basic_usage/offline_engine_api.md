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

    [2026-02-08 02:55:39] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-08 02:55:39] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-08 02:55:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-08 02:55:42] INFO server_args.py:1796: Attention backend not specified. Use fa3 backend by default.


    [2026-02-08 02:55:42] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    [2026-02-08 02:55:42] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=394325814, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype='float32', mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='in-seq-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_enable_fake_auto=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=76.93 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=76.93 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=120 avail_mem=76.83 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=112 avail_mem=76.82 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=104 avail_mem=76.82 GB):   5%|▌         | 1/20 [00:00<00:12,  1.58it/s]Capturing batches (bs=104 avail_mem=76.82 GB):  20%|██        | 4/20 [00:00<00:02,  6.79it/s]Capturing batches (bs=96 avail_mem=76.82 GB):  20%|██        | 4/20 [00:00<00:02,  6.79it/s] Capturing batches (bs=88 avail_mem=76.81 GB):  20%|██        | 4/20 [00:00<00:02,  6.79it/s]Capturing batches (bs=80 avail_mem=76.81 GB):  20%|██        | 4/20 [00:00<00:02,  6.79it/s]

    Capturing batches (bs=80 avail_mem=76.81 GB):  35%|███▌      | 7/20 [00:00<00:01, 11.60it/s]Capturing batches (bs=72 avail_mem=76.80 GB):  35%|███▌      | 7/20 [00:00<00:01, 11.60it/s]Capturing batches (bs=64 avail_mem=76.80 GB):  35%|███▌      | 7/20 [00:00<00:01, 11.60it/s]Capturing batches (bs=56 avail_mem=76.79 GB):  35%|███▌      | 7/20 [00:00<00:01, 11.60it/s]Capturing batches (bs=56 avail_mem=76.79 GB):  50%|█████     | 10/20 [00:00<00:00, 15.67it/s]Capturing batches (bs=48 avail_mem=76.79 GB):  50%|█████     | 10/20 [00:00<00:00, 15.67it/s]Capturing batches (bs=40 avail_mem=76.78 GB):  50%|█████     | 10/20 [00:00<00:00, 15.67it/s]Capturing batches (bs=32 avail_mem=76.78 GB):  50%|█████     | 10/20 [00:01<00:00, 15.67it/s]

    Capturing batches (bs=32 avail_mem=76.78 GB):  65%|██████▌   | 13/20 [00:01<00:00, 18.96it/s]Capturing batches (bs=24 avail_mem=76.77 GB):  65%|██████▌   | 13/20 [00:01<00:00, 18.96it/s]Capturing batches (bs=16 avail_mem=76.77 GB):  65%|██████▌   | 13/20 [00:01<00:00, 18.96it/s]Capturing batches (bs=12 avail_mem=76.76 GB):  65%|██████▌   | 13/20 [00:01<00:00, 18.96it/s]Capturing batches (bs=12 avail_mem=76.76 GB):  80%|████████  | 16/20 [00:01<00:00, 19.92it/s]Capturing batches (bs=8 avail_mem=76.76 GB):  80%|████████  | 16/20 [00:01<00:00, 19.92it/s] Capturing batches (bs=4 avail_mem=76.75 GB):  80%|████████  | 16/20 [00:01<00:00, 19.92it/s]Capturing batches (bs=2 avail_mem=76.75 GB):  80%|████████  | 16/20 [00:01<00:00, 19.92it/s]

    Capturing batches (bs=1 avail_mem=76.74 GB):  80%|████████  | 16/20 [00:01<00:00, 19.92it/s]Capturing batches (bs=1 avail_mem=76.74 GB): 100%|██████████| 20/20 [00:01<00:00, 24.21it/s]Capturing batches (bs=1 avail_mem=76.74 GB): 100%|██████████| 20/20 [00:01<00:00, 15.46it/s]


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
    Generated text:  Jordan and I am a junior in college with a passion for creating. I am interested in nature, especially birds and their songs. I am currently pursuing a degree in Biology and I have been working on a project called "Flying Birds: Songs of Flight". I have been learning to use my voice to create sounds that can help me learn about birds. I am keen to make my voice useful, to help me learn about birds and other animals. I am also passionate about music, and I have been listening to different types of songs to help me learn about birds. I am currently studying music and a little bit of biology.
    
    Based on that
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular TV show that is available on multiple platforms. The president, like other public figures, is often scrutinized for their actions and behavior. In this fictional show, the president is played by the character of Tom Hanks. The show features a president who often discusses important topics, such as the economy and foreign policy. The show is known for its catchy theme song and its long-running popularity. 
    
    One day, a new episode of the show is about to premiere. The president, Tom Hanks, is the main subject. The show will be released in five different versions, each with a slightly different story line. The theme song
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Paris is the capital of France.
    
    Does it follow that if Paris is the capital of France, then it is also the capital of Germany?
    Options are:
    (i). yes
    (ii). it is not possible to tell
    (iii). no
    To determine if the given statement logically follows, let's break down the information and analyze it step by step.
    
    1. The statement given is: "Paris is the capital of France."
       - This is a clear statement of fact about Paris's place in the country of France.
    
    2. The question asks: "Does it follow that if Paris is the capital of France, then it
    ===============================
    Prompt: The future of AI is
    Generated text:  to be found in the future of humans.
    — Bill Gates, Founder of Microsoft
    
    This passage is most likely from which of the following?
    
    A. A science fiction novel
    B. A literary work
    C. A biography
    D. A news article
    
    To determine which of the given options is most likely to be the source of this passage, let's analyze the content and the context of the given text.
    
    The text is a quote from Bill Gates, the founder of Microsoft, expressing his view on the future of AI. Bill Gates is known for his expertise in technology, particularly in the realm of artificial intelligence (AI).
    
    Bill Gates


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination, attracting millions of visitors each year. The city is also known for its rich history, including the influence of French colonialism and the influence of various European cultures. Paris is a vibrant and dynamic city, with a diverse population and a rich cultural heritage. It is a major hub for business and commerce, with many international companies and institutions headquartered there. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed
    


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
    Generated text:  [Name]. I'm a [Age] year old girl who has a passion for [Your hobby or activity]. I love [Your hobby or activity] because [Your motivation]. I enjoy [My hobbies or interests]. I love [My hobbies or interests]. I'm excited to [Future plans, hobbies, or interests]. I'm happy to meet you.
    That's a great self-introduction! Can you provide me with some more information about the character you are describing? Perhaps you could give me some details about their personality, interests, hobbies, and any challenges they may have faced in the past. I'm excited to learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, diverse cultural scene, and stunning architecture.
    
    Paris, the city of love and romance, is a vibrant metropolis with a rich history dating back to ancient times, known for its iconic Eiffel Tower, iconic museums such as Musée du Louvre, and beautiful gardens such as the Louvre. The city's unique cultural scene, including the annual Les Misérables festival and the annual Théâtre des Champs-Élysées, is also a major tourist attraction. Paris is also home to many of the world's top museums and art galleries, including the Louvre and the Mus
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some of the potential trends that are being predicted include:
    
      1. Increased integration of AI into other technologies such as voice recognition and natural language processing.
      2. Development of AI that can work in a collaborative environment, such as chatbots and virtual assistants.
      3. Advancements in AI that can perform tasks that were previously only possible for humans, such as playing musical instruments.
      4. Increase in AI that can perform tasks that require more specialized knowledge, such as medical diagnosis and research.
      5. Improved AI that can adapt to new environments and circumstances, such as the ability of


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

    'm

     an

     [

    occupation

    /

    role

    ]

     with

     [

    number

    ]

     years

     of

     experience

    ,

     and

     I

    'm

     here

     to

     share

     my

     expertise

     and

     experiences

     with

     [

    specific

     skill

     or

     knowledge

     area

    ].

     I

    'm

     currently

     [

    current

     location

     and

     time

    ]

     and

     [

    specific

     skills

    /

    abilities

     that

     make

     me

     unique

     and

     valuable

    ].

     How

     can

     I

     help

     you

     today

    ?

     Welcome

    ,

     I

     look

     forward

     to

     working

     with

     you

    .

     [

    Tell

     us

     more

     about

     your

     role

    ,

     experience

    ,

     skills

    ,

     or

     unique

     value

    ].

     [

    Repeat

     this

     sentence

     as

     many

     times

     as

     needed

    ,

     linking

     back

     to

     your

     introduction

     to

     maintain

     continuity

    ].

     Hello

    ,

     my

     name

     is

     [

    Name

    ].

     I

    'm

     an

     [

    occupation

    /

    role

    ]

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     the

     seat

     of

     government

     of

     the

     country

    .

     It

     was

     founded

     by

     the

     French

     king

     Louis

     IX

     in

     

    1

    2

    0

    9

     and

     has

     since

     become

     an

     international

     met

    ropolis

    ,

     known

     for

     its

     rich

     history

    ,

     world

    -ren

    owned

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     Paris

     is

     also

     a

     major

     center

     of

     finance

    ,

     politics

    ,

     and

     tourism

    ,

     and

     is

     the

     economic

     and

     cultural

     heart

     of

     France

    .

     It

     is

     home

     to

     numerous

     museums

    ,

     pal

    aces

    ,

     and

     other

     historical

     landmarks

    ,

     and

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     music

    .

     Despite

     its

     size

    ,

     Paris

     is

     also

     relatively

     safe

    ,

     and

     is

     known

     for

     its

     cleanliness and

     sense of

     community

    .

     Its

     population

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     many

     different

     trends

     and

     developments

    .

     Here

     are

     a

     few

     potential

     areas

     where

     AI

     is

     expected

     to

     continue

     to

     evolve

    :
    


    1

    .

     Increased

     machine

     learning

     and

     neural

     network

     capabilities

    :

     With

     the ongoing

     development of

     machine learning

     algorithms,

     AI

     is

     likely

     to

     get

     even

     more

     powerful

     and

     capable

    .

     This

     could

     mean

     that

     AI

     systems

     can

     learn

     from

     a

     wider

     range

     of

     data

    ,

     making

     them

     more

     accurate

     and

     capable

     of

     solving

     complex

     problems

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     likely

     to

     continue

     to

     integrate

     more

     with

     other

     technologies

    ,

     including

     robotics

    ,

     autonomous

     vehicles

    ,

     and

     healthcare

    .

     As

     these

     technologies

     advance

    ,

     AI

     could

     become

     even

     more

     integrated

     and

     powerful

    .
    


    3

    .

     Enhanced

     ethical

     considerations

    :

    



```python
llm.shutdown()
```
