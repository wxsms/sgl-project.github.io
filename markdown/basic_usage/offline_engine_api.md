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

    [2026-02-04 12:06:58] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-04 12:06:58] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-04 12:06:58] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-04 12:07:01] INFO server_args.py:1796: Attention backend not specified. Use fa3 backend by default.


    [2026-02-04 12:07:01] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    [2026-02-04 12:07:01] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=564527646, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype='float32', mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='in-seq-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_enable_fake_auto=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=76.93 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=76.93 GB):   5%|▌         | 1/20 [00:00<00:08,  2.32it/s]Capturing batches (bs=120 avail_mem=76.83 GB):   5%|▌         | 1/20 [00:00<00:08,  2.32it/s]Capturing batches (bs=112 avail_mem=76.82 GB):   5%|▌         | 1/20 [00:00<00:08,  2.32it/s]Capturing batches (bs=104 avail_mem=76.82 GB):   5%|▌         | 1/20 [00:00<00:08,  2.32it/s]Capturing batches (bs=104 avail_mem=76.82 GB):  20%|██        | 4/20 [00:00<00:01,  9.14it/s]Capturing batches (bs=96 avail_mem=76.82 GB):  20%|██        | 4/20 [00:00<00:01,  9.14it/s] Capturing batches (bs=88 avail_mem=76.81 GB):  20%|██        | 4/20 [00:00<00:01,  9.14it/s]Capturing batches (bs=80 avail_mem=76.81 GB):  20%|██        | 4/20 [00:00<00:01,  9.14it/s]

    Capturing batches (bs=80 avail_mem=76.81 GB):  35%|███▌      | 7/20 [00:00<00:00, 14.55it/s]Capturing batches (bs=72 avail_mem=76.80 GB):  35%|███▌      | 7/20 [00:00<00:00, 14.55it/s]Capturing batches (bs=64 avail_mem=76.80 GB):  35%|███▌      | 7/20 [00:00<00:00, 14.55it/s]Capturing batches (bs=56 avail_mem=76.79 GB):  35%|███▌      | 7/20 [00:00<00:00, 14.55it/s]Capturing batches (bs=56 avail_mem=76.79 GB):  50%|█████     | 10/20 [00:00<00:00, 15.84it/s]Capturing batches (bs=48 avail_mem=76.79 GB):  50%|█████     | 10/20 [00:00<00:00, 15.84it/s]

    Capturing batches (bs=40 avail_mem=76.78 GB):  50%|█████     | 10/20 [00:00<00:00, 15.84it/s]Capturing batches (bs=32 avail_mem=76.78 GB):  50%|█████     | 10/20 [00:00<00:00, 15.84it/s]Capturing batches (bs=32 avail_mem=76.78 GB):  65%|██████▌   | 13/20 [00:00<00:00, 16.30it/s]Capturing batches (bs=24 avail_mem=76.77 GB):  65%|██████▌   | 13/20 [00:00<00:00, 16.30it/s]Capturing batches (bs=16 avail_mem=76.77 GB):  65%|██████▌   | 13/20 [00:01<00:00, 16.30it/s]

    Capturing batches (bs=16 avail_mem=76.77 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.73it/s]Capturing batches (bs=12 avail_mem=76.76 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.73it/s]Capturing batches (bs=8 avail_mem=76.76 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.73it/s] Capturing batches (bs=4 avail_mem=76.75 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.73it/s]Capturing batches (bs=2 avail_mem=76.75 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.73it/s]Capturing batches (bs=2 avail_mem=76.75 GB):  95%|█████████▌| 19/20 [00:01<00:00, 21.55it/s]Capturing batches (bs=1 avail_mem=76.74 GB):  95%|█████████▌| 19/20 [00:01<00:00, 21.55it/s]Capturing batches (bs=1 avail_mem=76.74 GB): 100%|██████████| 20/20 [00:01<00:00, 16.27it/s]


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
    Generated text:  Emma. I'm a 16-year-old girl. My father and my mother are doctors. My grandparents are teachers. My sister is 12 years old. My parents are very kind and friendly. I have a big family. I like my family. We are very happy. We like to play together. Sometimes I play with my friends. Sometimes I stay at home and watch TV. I have two big sisters. They are very nice. I often help them. We all love our family. I often help my mother and my father with the housework. I like playing with my toys. Sometimes I play with my friends
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the average height of a certain species of tree. He randomly selects 50 trees and finds that the mean height is 150 feet with a standard deviation of 30 feet. Assuming the heights of the trees follow a normal distribution, what is the 95% confidence interval for the average height of the tree species? Round your answer to the nearest hundredth.
    
    To find the 95% confidence interval for the average height of the tree species, we need to use the formula for the confidence interval of the mean when the population standard deviation is unknown and the sample size is small (n < 
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. It is located in the Loire Valley region of the south of France, which is in the central part of the country. Paris is the largest city in the world, and its population of 2.2 million is the largest in the world. It is the capital of 5 regions, namely the North region, the North West region, the Center region, the North East region, and the South region. Paris is a metropolitan city, meaning it has the largest population in the area of metropolitan France.
    
    The capital of France was initially established in the 12th century, during the reign of King Philip
    ===============================
    Prompt: The future of AI is
    Generated text:  about being able to understand and interpret the context in which information is being presented, with the aim of using that context to create an accurate representation of the data. This is a very important goal because context is the building block of the information we learn from the internet and the internet itself is not always a clean or understandable place. Our purpose with this course is to help you learn the basics of natural language processing and how it can be used to analyze text and to accomplish tasks such as sentiment analysis, information retrieval, etc. As we go through this course, we will cover the basics of natural language processing, including a variety of topics such as


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast and I love to [Describe a hobby or activity]. I'm always looking for new experiences and adventures, and I'm always eager to learn new things. I'm a [Favorite Thing] person and I love to [Describe a favorite thing]. I'm always up for a challenge and I'm always looking for ways to grow and improve myself. I'm a [Favorite Book] person and I love to [Describe a favorite book]. I'm always looking for new ways to inspire and motivate others, and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the world. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous Parisian cuisine, and its fashion industry. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. As AI becomes more integrated into our daily lives, we may see a shift towards more ethical and responsible use of the technology, with a focus on ensuring that AI is used to benefit society as a whole rather than just for profit. Additionally, as AI becomes more integrated into our daily lives, we may see a shift towards more personalized and adaptive solutions, with AI systems being able to
    


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
    Generated text:  Sarah. I'm a software engineer with a strong passion for innovation and problem-solving. I've been working on this project for the past year and have been learning a lot along the way. I'm a confident and proactive individual who thrives in a fast-paced environment. I enjoy collaborating with people and taking on new challenges. I'm looking forward to bringing my skills to your team and contributing to your success. What's your name? My name is Sarah. Hello, my name is Sarah. I'm a software engineer with a strong passion for innovation and problem-solving. I've been working on this project for the past year and have been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and cultural center in France, and one of the world’s most populous cities. Paris is the seat of government, government agencies, the French government, the French High Court of Justice, and the European Parliament. It is also a major center of the arts, education, and nightlife. Paris has been a center for European art and architecture for centuries and is famous for its numerous museums and galleries. The city is also home to the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and many other historical landmarks. The city is known for its historical architecture, vibrant nightlife, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to expand and develop rapidly, with many potential trends that could shape the field. Here are some of the possible future trends in AI:
    
    1. Increased depth of learning: With the help of machine learning, AI is expected to get even more complex and perform better. Researchers are exploring deep learning, which involves training artificial neural networks to learn from multiple layers of data, and this could lead to new breakthroughs in areas like computer vision, natural language processing, and natural language generation.
    
    2. Increased focus on ethical AI: As more AI-powered technologies are developed, there is a growing awareness of the potential risks and challenges they could


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

     character

    's

     name

    ].

     I

    'm

     [

    insert

     character

    's

     age

    ]

     years

     old

    ,

     with

     [

    insert

     character

    's

     gender

    ]

     skin

     tone

     and

     [

    insert

     character

    's

     profession

    /

    occupation

    ].

     I

     grew

     up

     in

     [

    insert

     character

    's

     hometown

    ]

     and

     have

     always

     been

     [

    insert

     character

    's

     favorite

     hobby

    /

    interest

    ].

     I

     enjoy

     [insert

     character's

     favorite activity

    , hobby

    ,

     or

     hobby],

     and I

     believe that

     [insert

     character

    's

     personality trait

    ] makes

     me

     unique and

     special

    . I

    'm

     always eager

     to

     learn and

     grow

    ,

     and I

    'm always

     ready to

     share my

     experiences

     and thoughts

     with you

    . I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    !

     

    🌟

    ✨

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    .

     It

    's

     located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

     of

     the

     country

     and

     is

     a

     major

     cultural

    ,

     economic

    ,

     and

     political

     center

    .

     The

     city

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

     Paris

     is

     a

     popular

     tourist

     destination

     and

     has

     become

     a

     symbol

     of

     French

     culture

     and

     identity

    .

     The

     city

     is

     also

     home

     to

     a

     rich

     history

     and

     culture

     that

     is

     evident

     in

     its

     architecture

     and

     food

    .

     Paris

     is

     the

     second

    -largest

     city

     in

     the

     European

     Union

     by

     population

    ,

     and

     its

     importance

     as

     a

     major

     economic

     and

     political

     center

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continuous

     advancements

     in

     technologies

     and

     algorithms

    ,

     driven

     by

     advances

     in

     computing

     power

    ,

     data

     analysis

    ,

     and

     machine

     learning

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     AI

     is

     becoming

     increasingly

     complex

     and

     involves

     human

     emotions

    ,

     decision

    -making

    ,

     and

     decision

    -making

     processes

    .

     As

     such

    ,

     there

     is

     a

     growing

     emphasis

     on

     ethical

     considerations

     and

     ensuring

     that

     AI

     is

     used

     for

     the

     benefit

     of

     society rather

     than for

     personal

     gain

    .
    


    2

    .

     AI

     will

     become

     more

     specialized

    :

     As

     AI

     becomes

     more

     capable

    ,

     it

     will

     become

     more

     specialized

    .

     AI

     will

     be

     used

     for

     a

     wider

     range

     of

     applications

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

    



```python
llm.shutdown()
```
