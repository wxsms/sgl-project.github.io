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

    [2026-02-22 12:37:42] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-22 12:37:42] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-22 12:37:42] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-22 12:37:44] INFO server_args.py:1835: Attention backend not specified. Use fa3 backend by default.


    [2026-02-22 12:37:44] INFO server_args.py:2888: Set soft_watchdog_timeout since in CI


    [2026-02-22 12:37:44] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=440260705, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.66it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.65it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=5.30 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=5.30 GB):   5%|▌         | 1/20 [00:00<00:05,  3.67it/s]Capturing batches (bs=120 avail_mem=7.37 GB):   5%|▌         | 1/20 [00:00<00:05,  3.67it/s]Capturing batches (bs=112 avail_mem=7.37 GB):   5%|▌         | 1/20 [00:00<00:05,  3.67it/s]Capturing batches (bs=104 avail_mem=7.37 GB):   5%|▌         | 1/20 [00:00<00:05,  3.67it/s]Capturing batches (bs=96 avail_mem=7.37 GB):   5%|▌         | 1/20 [00:00<00:05,  3.67it/s] Capturing batches (bs=88 avail_mem=7.37 GB):   5%|▌         | 1/20 [00:00<00:05,  3.67it/s]Capturing batches (bs=88 avail_mem=7.37 GB):  30%|███       | 6/20 [00:00<00:00, 18.14it/s]Capturing batches (bs=80 avail_mem=7.37 GB):  30%|███       | 6/20 [00:00<00:00, 18.14it/s]Capturing batches (bs=72 avail_mem=7.37 GB):  30%|███       | 6/20 [00:00<00:00, 18.14it/s]Capturing batches (bs=64 avail_mem=7.37 GB):  30%|███       | 6/20 [00:00<00:00, 18.14it/s]Capturing batches (bs=56 avail_mem=7.36 GB):  30%|███       | 6/20 [00:00<00:00, 18.14it/s]

    Capturing batches (bs=56 avail_mem=7.36 GB):  50%|█████     | 10/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=48 avail_mem=7.36 GB):  50%|█████     | 10/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=40 avail_mem=7.36 GB):  50%|█████     | 10/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=32 avail_mem=7.36 GB):  50%|█████     | 10/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=24 avail_mem=7.36 GB):  50%|█████     | 10/20 [00:00<00:00, 24.79it/s]Capturing batches (bs=24 avail_mem=7.36 GB):  70%|███████   | 14/20 [00:00<00:00, 29.23it/s]Capturing batches (bs=16 avail_mem=7.36 GB):  70%|███████   | 14/20 [00:00<00:00, 29.23it/s]Capturing batches (bs=12 avail_mem=7.36 GB):  70%|███████   | 14/20 [00:00<00:00, 29.23it/s]Capturing batches (bs=8 avail_mem=7.36 GB):  70%|███████   | 14/20 [00:00<00:00, 29.23it/s] 

    Capturing batches (bs=4 avail_mem=7.35 GB):  70%|███████   | 14/20 [00:00<00:00, 29.23it/s]Capturing batches (bs=4 avail_mem=7.35 GB):  90%|█████████ | 18/20 [00:00<00:00, 29.43it/s]Capturing batches (bs=2 avail_mem=7.35 GB):  90%|█████████ | 18/20 [00:00<00:00, 29.43it/s]Capturing batches (bs=1 avail_mem=7.35 GB):  90%|█████████ | 18/20 [00:00<00:00, 29.43it/s]Capturing batches (bs=1 avail_mem=7.35 GB): 100%|██████████| 20/20 [00:00<00:00, 25.88it/s]


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
    Generated text:  Brian.
    
    I am a computer science student at Purdue University. My research interests include machine learning, natural language processing, computer vision, computer hardware, and data visualizations.
    
    I am currently working on a project to develop a new text classification algorithm using deep learning techniques. My goal is to improve the performance of existing algorithms by reducing their computational complexity.
    
    In addition to my research interests, I enjoy playing video games, listening to music, and reading books. I also have a passion for photography and want to learn more about the art of photography.
    
    What is your favorite hobby?
    
    As a language model AI, I don't have a personal hobby like
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a new term of office. The candidates are all well-known politicians. The president is going to have a fundraising event and he would like to raise as much money as possible for the candidates. The president is considering using a campaign bus to transport the candidates. The bus is about 30 feet long and has a capacity of 40 people. 
    
    1. How many cars would be needed to transport 1000 people?
    2. How many cars would be needed to transport 5000 people?
    3. If each car can hold 4 people, how many cars are needed to transport 100
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. New York
    D. Sydney
    Answer:
    
    A
    
    In the aftermath of the Third Plenary Session of the Eleventh Central Committee of the Communist Party of China, the focus of the Party's work was shifted to ____.
    A. Economic construction
    B. Reform and Opening Up
    C. Science and Technology
    D. Environmental Protection
    Answer:
    
    B
    
    The National Association of Insurance Agents and Dealers is an organization with the function of ____
    A. Insurance business
    B. Corporate governance
    C. Economic regulation
    D. Supervision and management of insurance intermediaries
    Answer
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and there’s no one surefire way to know what the future of AI will be. But as the scientific community continues to push the boundaries of what it is possible to do with artificial intelligence, we can see a number of new ways that AI is being applied to our daily lives.
    One example of this is in the area of self-driving cars. In the near future, the ability of self-driving cars to avoid collisions and accidents in real-time will likely play a huge role in making our transportation systems safer for everyone. As these cars become more widely available, we will see a lot of roadblocks to the development and adoption of


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm [age] years old, and I'm [gender] and [race]. I'm [occupation] and I'm [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm [age] years old, and I'm [gender] and [race]. I'm [occupation] and I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. It is also a major financial center and home to many world-renowned museums and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is home to many famous French artists, writers, and musicians, and is a cultural melting pot of diverse cultures. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate many tasks, from manufacturing to customer service, and will become more efficient and accurate.
    
    2. Improved privacy and security: As AI systems become more sophisticated, they will need to be designed with privacy and security in mind. This will require ongoing research and development to ensure that AI systems are safe and secure.
    
    3. Enhanced human-machine collaboration: AI will continue to play a more significant role in human-machine collaboration, allowing machines to assist humans in tasks that were previously too complex for humans to handle.
    
    4. AI ethics and governance: As AI systems
    


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
    Generated text:  [Name], and I am a/an [Role] in this game [Title].
    Hello! I'm [Name] and I'm a/an [Role] in this game [Title]. My name is [Name], and I'm a/an [Role] in this game [Title]. I'm [Name] and I'm a/an [Role] in this game [Title]. I'm [Name] and I'm a/an [Role] in this game [Title]. I'm [Name] and I'm a/an [Role] in this game [Title]. I'm [Name] and I'm a/an [Role]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of light and opulence, known for its rich history, beautiful architecture, and vibrant culture. Its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral are a must-visit for any visitor to the city. Additionally, Paris hosts numerous festivals and events throughout the year, making it a popular destination for tourists and locals alike. 
    
    Paris, the heart of French culture and history, is an essential destination for anyone seeking an authentic and immersive experience of the French way of life. Its beautiful architecture, delicious cuisine, and vibrant culture make it a must-visit destination for anyone who
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  not yet clear, but it is likely to continue to evolve and transform in the following ways:
    
    1. Increased accuracy: As AI improves in terms of accuracy and precision, it will become more widely used in various applications.
    
    2. Integration with other technologies: AI will continue to be integrated with other technologies such as machine learning, natural language processing, and blockchain, to create new and innovative applications.
    
    3. Personalization: AI will become more personalized, as it will learn and adapt to the user's behavior and preferences.
    
    4. Ethical concerns: AI will continue to be a subject of ethical concerns, as it raises questions about privacy,


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

     a

     [

    职

    名

    ]

     at

     [

    company

     name

    ].

     I

     have

     been

     at

     [

    company

     name

    ]

     for

     [

    x

     years

    ]

     and

     I

     have

     been

     working

     in

     [

    position

    ]

     for

     [

    company

     name

    ]

     since

     [

    start

     year

    ].

     I

     am

     a

     [

    field

    ]

     with

     [

    number

    ]

     years

     of

     experience

     in

     [

    field

    ].

     I

     am

     a

     [

    gender

    ]

     [

    race

    ]

     [

    national

    ity

    ]

     [

    age

    ]

     [

    height

    ]

     [

    weight

    ]

     [

    weight

     range

    ]

     [

    height

     range

    ]

     [

    b

    od

    ily

     type

    ]

     [

    gender

    ]

     [

    race

    ]

     [

    national

    ity

    ]

     [

    age

    ]

     [

    height

    ]

     [

    weight

    ]

     [

    weight

     range

    ]

     [

    height

     range

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     the

     French

     Republic

    ,

     located

     on

     the

     Î

    le

     de

     France

     in

     the

     Se

    ine

     River

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     second

    -largest

     in

     the

     European

     Union

    .

     It

     is

     known

     for

     its

     distinctive

     architecture

    ,

     fashion

    ,

     and

     cuisine

    ,

     and

     is

     one

     of

     the

     most

     visited

     tourist

     destinations

     in

     the

     world

    .

     Paris

     is

     also

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     historical

     landmarks

    ,

     including

     the

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

     the

     E

    iff

    el

     Tower

    .

     The

     city

     is

     known

     for

     its

     lively

     nightlife

    ,

     jazz

    ,

     and

     cultural

     events

     throughout

     the

     year

    .

     Paris

     is

     a

     multicultural

     and

     diverse

     city

     that

     is

     home

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     und

    eni

    ably

     promising

     and

     expanding

     rapidly

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     efficiency

    :

     With

     the

     advancements

     in

     machine

     learning

     and

     data

     analytics

    ,

     AI

     is

     becoming

     increasingly

     efficient

     at

     solving

     complex

     problems

    .

     This

     means

     that

     AI

     systems

     can

     now

     perform

     tasks

     at

     a

     faster

     rate

     than

     ever

     before

    ,

     and

     they

     can

     learn

     from

     data

     in

     ways

     that

     were

     previously

     impossible

    .
    


    2

    .

     Increased

     precision

    :

     AI

     is

     becoming

     more

     and

     more

     capable

     of

     making

     precise

     predictions

     and

     decisions

    .

     This

     means

     that

     we

     can

     now

     use

     AI

     to

     analyze

     and

     interpret

     complex

     data

     sets

     with

     greater

     accuracy

     than

     ever

     before

    .
    


    3

    .

     Greater

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     being

     used

     to

    



```python
llm.shutdown()
```
