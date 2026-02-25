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

    [2026-02-25 05:17:25] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-25 05:17:25] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-25 05:17:25] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-25 05:17:28] INFO server_args.py:1829: Attention backend not specified. Use fa3 backend by default.


    [2026-02-25 05:17:28] INFO server_args.py:2890: Set soft_watchdog_timeout since in CI


    [2026-02-25 05:17:28] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=683737012, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.15it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.14it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=76.27 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=76.27 GB):   5%|▌         | 1/20 [00:00<00:03,  5.64it/s]Capturing batches (bs=120 avail_mem=76.16 GB):   5%|▌         | 1/20 [00:00<00:03,  5.64it/s]

    Capturing batches (bs=112 avail_mem=76.16 GB):   5%|▌         | 1/20 [00:00<00:03,  5.64it/s]Capturing batches (bs=104 avail_mem=76.16 GB):   5%|▌         | 1/20 [00:00<00:03,  5.64it/s]Capturing batches (bs=96 avail_mem=75.70 GB):   5%|▌         | 1/20 [00:00<00:03,  5.64it/s] Capturing batches (bs=96 avail_mem=75.70 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.08it/s]Capturing batches (bs=88 avail_mem=75.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.08it/s]Capturing batches (bs=80 avail_mem=73.20 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.08it/s]Capturing batches (bs=72 avail_mem=72.63 GB):  25%|██▌       | 5/20 [00:00<00:00, 18.08it/s]

    Capturing batches (bs=72 avail_mem=72.63 GB):  40%|████      | 8/20 [00:00<00:00, 20.87it/s]Capturing batches (bs=64 avail_mem=72.62 GB):  40%|████      | 8/20 [00:00<00:00, 20.87it/s]Capturing batches (bs=56 avail_mem=72.62 GB):  40%|████      | 8/20 [00:00<00:00, 20.87it/s]Capturing batches (bs=48 avail_mem=72.61 GB):  40%|████      | 8/20 [00:00<00:00, 20.87it/s]Capturing batches (bs=48 avail_mem=72.61 GB):  55%|█████▌    | 11/20 [00:00<00:00, 21.42it/s]Capturing batches (bs=40 avail_mem=72.61 GB):  55%|█████▌    | 11/20 [00:00<00:00, 21.42it/s]Capturing batches (bs=32 avail_mem=72.60 GB):  55%|█████▌    | 11/20 [00:00<00:00, 21.42it/s]

    Capturing batches (bs=24 avail_mem=72.60 GB):  55%|█████▌    | 11/20 [00:00<00:00, 21.42it/s]Capturing batches (bs=24 avail_mem=72.60 GB):  70%|███████   | 14/20 [00:00<00:00, 23.68it/s]Capturing batches (bs=16 avail_mem=72.59 GB):  70%|███████   | 14/20 [00:00<00:00, 23.68it/s]Capturing batches (bs=12 avail_mem=72.59 GB):  70%|███████   | 14/20 [00:00<00:00, 23.68it/s]Capturing batches (bs=8 avail_mem=72.58 GB):  70%|███████   | 14/20 [00:00<00:00, 23.68it/s] Capturing batches (bs=8 avail_mem=72.58 GB):  85%|████████▌ | 17/20 [00:00<00:00, 22.76it/s]Capturing batches (bs=4 avail_mem=72.58 GB):  85%|████████▌ | 17/20 [00:00<00:00, 22.76it/s]

    Capturing batches (bs=2 avail_mem=72.57 GB):  85%|████████▌ | 17/20 [00:00<00:00, 22.76it/s]Capturing batches (bs=1 avail_mem=72.57 GB):  85%|████████▌ | 17/20 [00:00<00:00, 22.76it/s]Capturing batches (bs=1 avail_mem=72.57 GB): 100%|██████████| 20/20 [00:00<00:00, 22.49it/s]


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
    Generated text:  Angelique and I am a first year student at the University of Miami in the Department of Mechanical Engineering. My undergraduate degree is in Mechanical Engineering and I completed my Bachelor's in Electrical Engineering. Currently, I am working in the Mechanical Engineering Department, and I am currently a graduate student at the University of Miami. I am the Assistant Director of the Research Division.
    My research interests are in the areas of fluid dynamics and porous media flow. For instance, I have been interested in the study of the flow of fluids through porous media, with a particular emphasis on the transport of ions. I also have been working on the study of the transport of
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to choose between two candidates for his running mate. He has 5 different primary candidates for each candidate. If the president chooses the first candidate, he has a 1/6 chance of choosing the first candidate for the second candidate. What is the probability that the president will choose the first candidate for both candidates? Express your answer as a common fraction.
    
    To find the probability that the president will choose the first candidate for both candidates, we need to consider the following:
    
    1. The president has 5 different primary candidates for each candidate.
    2. The president has a 1/6 chance of choosing the first candidate for the second
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) London
    C) Rome
    D) Berlin
    E) Moscow
    
    To determine the capital of France, we need to review our knowledge of the world's capitals. The capital of France is Paris. Let's go through the options provided:
    
    A) Paris - This is correct.
    B) London - This is not the capital of France.
    C) Rome - This is not the capital of France.
    D) Berlin - This is not the capital of France.
    E) Moscow - This is not the capital of France.
    
    Therefore, the correct answer is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be cloud based. This is the consensus of experts, AI experts, and the general public. In the world of business and technology, this is the future, and it's here to stay. But, if you're not a business professional, what does that mean for you?
    With the increasing rise of the Internet of Things (IoT) and the increased use of technology, it is becoming more and more prevalent to have AI in place. AI, or artificial intelligence, is all about AI. It is not about computers, it is about human intelligence.
    AI is the ability of a computer to imitate human intelligence and the


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm confident in my abilities and I'm eager to [reason for eagerness]. I'm [age] years old, and I'm [gender] and [race]. I have a [number of hobbies] and [number of interests] that I enjoy. I'm [number of books] and [number of movies] away from home, and I'm always looking for new experiences. I'm [number of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political center. Paris is a bustling metropolis with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major economic hub for France. The city is known for its fashion industry, art scene, and cuisine, and is home to many famous landmarks and attractions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: As AI becomes more advanced, it is likely to automate many of the tasks that are currently done by humans. This could lead to a reduction in the need for human workers, which could have a significant impact on the job market.
    
    2. Improved privacy and security: As AI becomes more advanced, it is likely to require more data to function effectively. This could lead to concerns about privacy and security
    


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
    Generated text:  Sarah and I'm a 30-year-old software engineer with a degree in computer science. I've been working in the field for over a decade, and I love coding and learning new technologies. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in the industry. I'm also very passionate about sustainability and am committed to using technology to make a positive impact on the world. What are some of the hobbies or interests you have outside of work? As a software engineer, I enjoy working with coding languages and algorithms, exploring new technologies, and reading books on computer science and software development. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Grande-Bretagne" and "Ville de Paris". It is the largest city in Europe, with an estimated population of over 2. 8 million people. Paris is a highly influential and diverse city, known for its art, architecture, and culinary traditions. The city is home to many world-renowned landmarks such as the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Champs-Élysées. It is also the birthplace of many influential figures and the center of France's cultural and political life. As a result, Paris has been a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and difficult to predict, but there are several potential trends that are likely to shape the field in the coming years:
    
    1. Increased reliance on machine learning: As AI systems become more sophisticated, they will become increasingly adept at recognizing patterns and making predictions that are beyond human ability. This trend is likely to lead to more complex AI systems that are capable of performing tasks that were previously thought to be beyond human capability.
    
    2. AI that can communicate and share knowledge: AI systems that can communicate and share knowledge with other AI systems or humans will become increasingly important as the field of artificial intelligence continues to grow and evolve. This will likely lead to


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

     am

     a

     [

    Your

     Profession

     or

     Role

    ]

     who

     has

     been

     passionate

     about

     [

    Your

     Field

     of

     Interest

    ]

     for

     [

    Your

     Duration

     of

     Interest

    ].

     I

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

    .

     Let

    's

     get

     to

     know

     each

     other

    !

     [

    Your

     Name

    ]

     [

    Your

     Age

    ]

     [

    Your

     Gender

    ]
    


    ---
    


    **

    Name

    :**

     [

    Your

     Name

    ]

      


    **

    Age

    :**

     [

    Your

     Age

    ]

      


    **

    Gender

    :**

     [

    Your

     Gender

    ]

      


    **

    Prof

    ession

    :**

     [

    Your

     Profession

    ]

      
    


    **

    Appearance

    :

    **


    -

     Features

    :

     [

    Your

     Height

    ,

     Weight

    ,

     and

     Unique

     Physical

     Traits

    ]


    -

     Clothing

    :

     [

    Your

     Preferred

     Clothing

    ,

     If

     any

    ]


    -

    
    
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

     France

     by

     area

     and

     the

     

    1

    4

    th

     largest

     city

     by

     population

    ,

     and

     the

     world

    's

     

    6

    th

     largest

     by

     urban

     population

    .

     The

     city

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     historical

     significance

    .

     Its

     main

     landmarks

     include

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

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

     Love

    "

     due

     to

     its

     rich

     and

     diverse

     cultural

     scene

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     With

     its

     beautiful

     architecture

    ,

     vibrant

     city

     life

    ,

     and

     romantic

     atmosphere

    ,

     Paris

     is

     a

     truly

     unique

     and

     enchant

    ing

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


     

     

    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     are

     likely

     to

     become

     more

     integrated

     with

     human

     intelligence

    ,

     allowing

     them

     to

     process

     and

     analyze

     data

     in

     ways

     that

     are

     difficult

     for

     machines

     alone

    .


     

     

    2

    .

     Enhanced

     data

     privacy

     and

     security

    :

     AI

     systems

     are

     likely

     to

     require

     more

     and

     more

     data

     to

     function

     effectively

    ,

     and

     as

     such

    ,

     there

     will

     be

     increasing

     efforts

     to

     protect

     and

     secure

     this

     data

    .

     This

     could

     involve

     new

     laws

     and

     regulations

     to

     govern

     data

     privacy

     and

     cybersecurity

    .


     

     

    3

    .

     Improved

     transparency

     and

     explain

    ability

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     may

     become

     more

    



```python
llm.shutdown()
```
