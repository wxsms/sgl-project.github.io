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

    [2026-02-15 17:46:06] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-15 17:46:06] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-15 17:46:06] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-15 17:46:08] INFO server_args.py:1832: Attention backend not specified. Use fa3 backend by default.


    [2026-02-15 17:46:08] INFO server_args.py:2867: Set soft_watchdog_timeout since in CI


    [2026-02-15 17:46:08] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=168777625, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_enable_fake_auto=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.46it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.46it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=39.47 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=39.47 GB):   5%|▌         | 1/20 [00:00<00:03,  5.13it/s]Capturing batches (bs=120 avail_mem=39.36 GB):   5%|▌         | 1/20 [00:00<00:03,  5.13it/s]

    Capturing batches (bs=112 avail_mem=38.08 GB):   5%|▌         | 1/20 [00:00<00:03,  5.13it/s]Capturing batches (bs=104 avail_mem=37.98 GB):   5%|▌         | 1/20 [00:00<00:03,  5.13it/s]Capturing batches (bs=104 avail_mem=37.98 GB):  20%|██        | 4/20 [00:00<00:01, 15.07it/s]Capturing batches (bs=96 avail_mem=37.98 GB):  20%|██        | 4/20 [00:00<00:01, 15.07it/s] Capturing batches (bs=88 avail_mem=37.98 GB):  20%|██        | 4/20 [00:00<00:01, 15.07it/s]Capturing batches (bs=80 avail_mem=37.97 GB):  20%|██        | 4/20 [00:00<00:01, 15.07it/s]Capturing batches (bs=80 avail_mem=37.97 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=72 avail_mem=37.97 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.53it/s]

    Capturing batches (bs=64 avail_mem=37.96 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=56 avail_mem=37.96 GB):  35%|███▌      | 7/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=56 avail_mem=37.96 GB):  50%|█████     | 10/20 [00:00<00:00, 23.43it/s]Capturing batches (bs=48 avail_mem=37.95 GB):  50%|█████     | 10/20 [00:00<00:00, 23.43it/s]Capturing batches (bs=40 avail_mem=37.95 GB):  50%|█████     | 10/20 [00:00<00:00, 23.43it/s]Capturing batches (bs=32 avail_mem=37.94 GB):  50%|█████     | 10/20 [00:00<00:00, 23.43it/s]Capturing batches (bs=32 avail_mem=37.94 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.88it/s]Capturing batches (bs=24 avail_mem=37.94 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.88it/s]

    Capturing batches (bs=16 avail_mem=37.93 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.88it/s]Capturing batches (bs=12 avail_mem=37.90 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.88it/s]Capturing batches (bs=12 avail_mem=37.90 GB):  80%|████████  | 16/20 [00:00<00:00, 19.16it/s]Capturing batches (bs=8 avail_mem=37.89 GB):  80%|████████  | 16/20 [00:00<00:00, 19.16it/s] Capturing batches (bs=4 avail_mem=37.86 GB):  80%|████████  | 16/20 [00:00<00:00, 19.16it/s]

    Capturing batches (bs=2 avail_mem=37.85 GB):  80%|████████  | 16/20 [00:00<00:00, 19.16it/s]Capturing batches (bs=2 avail_mem=37.85 GB):  95%|█████████▌| 19/20 [00:00<00:00, 20.74it/s]Capturing batches (bs=1 avail_mem=37.85 GB):  95%|█████████▌| 19/20 [00:00<00:00, 20.74it/s]

    Capturing batches (bs=1 avail_mem=37.85 GB): 100%|██████████| 20/20 [00:01<00:00, 17.14it/s]


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
    Generated text:  Anna. I am a middle school student. My hobbies include going on school trips, reading books, and playing sports. I like to eat lots of vegetables and have healthy food. I also love to watch TV. I have a pet dog named Sandy. It is a white dog. Its name is Micky. Micky likes to play with Sandy. He gets along well with Sandy. He plays fetch every day with Sandy. Sometimes he plays hide and seek. I also like to do something special on New Year's Day. It is New Year's Day this year. I get up early on New Year's Day. I do many
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to hold a new presidential election. The likelihood of a major earthquake of magnitude 5.0 or greater hitting the continental United States in a particular year is 0.1. Let X represent the number of earthquakes that will occur in a particular year, where X follows a binomial distribution with n = 120. 
    
    a. Find the mean and standard deviation of X.
    
    b. Find the mean and standard deviation of the random variable X^2.
    
    c. Find the probability that the number of earthquakes in a year exceeds 10.
    
    d. Find the probability that the number of earthquakes in a
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a historic city in the south of France, and its population is over 2 million.
    The area is famous for its architecture and the famous Eiffel Tower. It is also a center of culture, with many museums and theaters. The weather is usually warm and pleasant in the summer, but can be cold and wet in the winter.
    Paris is the capital of France, which is the 2nd largest country in the world by area. It is the second largest city in the world in terms of population. The city has a rich and diverse history, with many historical sites and landmarks. It is also a hub
    ===============================
    Prompt: The future of AI is
    Generated text:  looking to a place where it will run like a laptop. In the first of two essays, Joe Gardner, AI Research Group, discusses the new age of artificial intelligence, and how it will impact everything from the future of work to our understanding of the world.
    It seems like the future of AI is arriving, and we’re still just beginning to understand how it will affect us. The recent announcement by Microsoft that it will build its own artificial intelligence framework, called HoloLens, is an excellent example of this. This is a virtual reality headset that will allow people to use AI to manipulate virtual environments and interact with them in real time,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. What do you like to do in your free time? I like to [insert a short, positive description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of French culture and the influence of the French Revolution. It is also home to many notable French artists, writers, and musicians. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. The city is known for its fashion, art, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis, treatment, and patient care.
    
    3. Greater use of AI in education: AI is already being used
    


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
    Generated text:  [Name], and I am an AI assistant designed to assist individuals in various tasks and provide them with accurate and helpful information. I'm here to help you with everything from answering questions to providing advice on how to navigate the world of technology and information. Whether you need help with a specific task or just want to learn something new, I'm here to help. So if you have any questions or need any guidance, don't hesitate to reach out. I'm here to help! 🚀👨‍💻💻
    
    Hey there, everyone! I'm [Name], your friendly virtual assistant here to assist with anything you need. �
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city with a rich history dating back to the Roman Empire.
    Paris is the capital city of France, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major transportation hub and a popular tourist destination, attracting millions of visitors each year. The city is known for its rich culture, unique architecture, and rich history, making it a fascinating city to explore and experience. Paris is a cultural melting pot of France, with a diverse population and a rich mix of traditions and customs. Its importance as a cultural and political center has made it a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating, and there are many exciting developments that could shape the landscape. Here are some possible future trends in AI:
    
    1. Increased machine learning and deep learning: As AI algorithms become more complex, the ability to learn and adapt to new data will become more efficient. We may see more sophisticated models that can recognize patterns and make decisions based on large datasets.
    
    2. Autonomous vehicles: With the increasing emphasis on autonomous transportation, we may see more autonomous vehicles on the roads, capable of safely navigating complex urban environments and maintaining safety standards.
    
    3. Increased ethical considerations: As AI systems become more advanced, there may be increasing ethical concerns about their use


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

    insert

     first

     name

     and

     last

     name

    ],

     and

     I

    'm

     an

     [

    insert

     occupation

     or

     hobby

    ]

     who

     has

     always

     been

     passionate

     about

     [

    insert

     something

    ,

     such

     as

     music

    ,

     literature

    ,

     or

     art

    ].

     I

     spend

     a

     lot

     of

     time

     [

    insert

     a

     hobby

     or

     activity

     that

     you

     enjoy

    ,

     such

     as

     reading

    ,

     writing

    ,

     playing

     music

    ,

     or

     practicing

     a

     particular

     craft

    ].

     I

     believe

     in

     the

     power

     of

     [

    insert

     something

    ,

     such

     as

     empathy

    ,

     kindness

    ,

     or

     dedication

     to

     one

    's

     craft

    ].

     I

    'm

     [

    insert

     age

    ,

     gender

    ,

     etc

    .

    ],

     and

     I

    'm

     a

     [

    insert

     any

     personal

     or

     professional

     qualities

    ,

     such

     as

     [

    insert

     a

     trait

    ,

     such

     as

     "

    out

    going

    ",

     "

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     provided

     is

     a

     brief

    ,

     one

    -s

    entence

     factual

     description

     of

     the

     capital

     city

     of

     France

    .

     It

     mentions

     that

     Paris

     is

     the

     capital

     of

     France

     and

     lists

     some

     of

     its

     key

     features

    ,

     including

     its

     location

    ,

     name

    ,

     historical

     significance

    ,

     and

     numerous

     landmarks

    .

     The

     statement

     is

     concise

     and

     informative

    ,

     highlighting

     the

     importance

     and

     uniqueness

     of

     the

     capital

     city

    .

     
    


    Therefore

    ,

     the

     answer

     is

    :

     The

     capital

     city

     of

     France

     is

     Paris

    .

     
    


    This

     answer

     provides

     a

     clear

    ,

     concise

     statement

     about

     the

     role

     of

     Paris

     in

     France

    's

     political

     and

     cultural

     life

    .

     However

    ,

     it

     doesn

    't

     include

     a

     comparison

     to

     other

     major

     cities

     or

     provide

     detailed

     information

     about

     other

     capitals

    .

     To

     include

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     rapidly

     advancing

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

     Personal

    ization

    :

     AI

     will

     enable

     machines

     to

     learn

     and

     adapt

     to

     users

    '

     preferences

     and

     behaviors

    ,

     leading

     to

     more

     personalized

     experiences

    .

     This

     will

     require

     more

     sophisticated

     algorithms

     and

     machine

     learning

     techniques

    .
    


    2

    .

     Enhanced

     Human

    -A

    I

     Interaction

    :

     AI

     will

     enable

     more

     human

    -like

     interactions

     between

     humans

     and

     machines

    ,

     such

     as

     through

     virtual

     assistants

     or

     chat

    bots

    .

     This

     will

     require

     better

     understanding

     of

     human

     psychology

     and

     behavior

    .
    


    3

    .

     Autonomous

     Vehicles

    :

     AI

     will

     enable

     autonomous

     vehicles

     that

     can

     navigate

     roads

    ,

     interact

     with

     pedestrians

     and

     other

     vehicles

    ,

     and

     make

     decisions

     on

     the

     fly

    .

     This

     will

     require

     significant

     advancements

     in

     autonomous

    



```python
llm.shutdown()
```
