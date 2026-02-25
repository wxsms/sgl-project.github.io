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

    [2026-02-25 01:53:01] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-25 01:53:01] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-25 01:53:01] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-25 01:53:03] INFO server_args.py:1828: Attention backend not specified. Use fa3 backend by default.


    [2026-02-25 01:53:03] INFO server_args.py:2889: Set soft_watchdog_timeout since in CI


    [2026-02-25 01:53:03] INFO engine.py:156: server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.835, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=121392278, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.57it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.56it/s]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=57.56 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=57.56 GB):   5%|▌         | 1/20 [00:00<00:03,  5.10it/s]Capturing batches (bs=120 avail_mem=57.46 GB):   5%|▌         | 1/20 [00:00<00:03,  5.10it/s]

    Capturing batches (bs=112 avail_mem=57.46 GB):   5%|▌         | 1/20 [00:00<00:03,  5.10it/s]Capturing batches (bs=104 avail_mem=57.46 GB):   5%|▌         | 1/20 [00:00<00:03,  5.10it/s]Capturing batches (bs=104 avail_mem=57.46 GB):  20%|██        | 4/20 [00:00<00:01, 14.76it/s]Capturing batches (bs=96 avail_mem=57.46 GB):  20%|██        | 4/20 [00:00<00:01, 14.76it/s] Capturing batches (bs=88 avail_mem=57.46 GB):  20%|██        | 4/20 [00:00<00:01, 14.76it/s]Capturing batches (bs=80 avail_mem=57.46 GB):  20%|██        | 4/20 [00:00<00:01, 14.76it/s]Capturing batches (bs=80 avail_mem=57.46 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.54it/s]Capturing batches (bs=72 avail_mem=57.46 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.54it/s]

    Capturing batches (bs=64 avail_mem=57.46 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.54it/s]Capturing batches (bs=56 avail_mem=57.45 GB):  35%|███▌      | 7/20 [00:00<00:00, 19.54it/s]Capturing batches (bs=56 avail_mem=57.45 GB):  50%|█████     | 10/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=48 avail_mem=57.45 GB):  50%|█████     | 10/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=40 avail_mem=57.45 GB):  50%|█████     | 10/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=32 avail_mem=57.45 GB):  50%|█████     | 10/20 [00:00<00:00, 22.02it/s]Capturing batches (bs=32 avail_mem=57.45 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=24 avail_mem=57.45 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.66it/s]

    Capturing batches (bs=16 avail_mem=57.45 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=12 avail_mem=57.45 GB):  65%|██████▌   | 13/20 [00:00<00:00, 23.66it/s]Capturing batches (bs=12 avail_mem=57.45 GB):  80%|████████  | 16/20 [00:00<00:00, 22.21it/s]Capturing batches (bs=8 avail_mem=57.45 GB):  80%|████████  | 16/20 [00:00<00:00, 22.21it/s] Capturing batches (bs=4 avail_mem=57.44 GB):  80%|████████  | 16/20 [00:00<00:00, 22.21it/s]Capturing batches (bs=2 avail_mem=57.44 GB):  80%|████████  | 16/20 [00:00<00:00, 22.21it/s]

    Capturing batches (bs=1 avail_mem=57.44 GB):  80%|████████  | 16/20 [00:00<00:00, 22.21it/s]Capturing batches (bs=1 avail_mem=57.44 GB): 100%|██████████| 20/20 [00:00<00:00, 25.56it/s]Capturing batches (bs=1 avail_mem=57.44 GB): 100%|██████████| 20/20 [00:00<00:00, 21.81it/s]


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
    Generated text:  Sophie and I'm a very enthusiastic and easy-going person. I'm a patient person and always ready to help others. Here's a story I heard: I have always been known as a gentle and warm-hearted person. I'm always ready to help others and offer my advice and guidance to those who need it. I'm always patient and never let anyone down. I have a good sense of humor and I always try to make people laugh. I'm a very kind and caring person. I'm very good at making friends with others and I can always be friendly to everyone. I have a heart of gold and always try my best to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a wealthy man who has inherited a large fortune. To ensure that the president can manage his wealth effectively, the government has decided to provide him with a balance between his wealth and a stipulated amount of money to be given to him each year. This amount is determined by the following formula:
    
    \[
    \text{Stipulated Amount} = \frac{1}{\sqrt[10]{(1 - r)^2}}
    \]
    
    where \( r \) is the interest rate of the investments being made in the government. If the president invests a portion of his wealth in the government's bonds, the interest rate \( r
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) London
    C) Madrid
    D) Berlin
    E) Athens
    The capital of France is Paris. Therefore, the correct answer is:
    
    A) Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  still in the future
    
    AI is rapidly changing the world. Technology is advancing at an unprecedented pace and new applications are emerging. The world is changing. But the pace of change is just beginning.
    
    That's a lot to digest in one sentence. Can you explain the concept of AI in a way that someone without technical knowledge could understand? Absolutely! Imagine you have a toy robot that can learn to do many things for you. This toy robot is like a smart computer that can understand and do things on its own without being told how to do them. It's like a magical tool that can help you do things better and faster. That's


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


    Generated text:  Paris, also known as the City of Light, a city renowned for its historical architecture, vibrant culture, and annual festivals. It is the largest city in France and the fifth-largest city in the world by population. Paris is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among other landmarks. The city is also known for its fashion industry, art scene, and annual cultural events. Paris is a major economic and cultural hub, and its influence extends beyond France to neighboring countries and the wider world. The city is also home to the French Parliament, the French National Library, and the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption.
    
    2. AI becoming more autonomous: As AI becomes more advanced, it is likely to become more autonomous, with machines being able to make decisions and take actions without human intervention.
    
    3. AI becoming more ethical: As AI becomes more advanced,
    


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
    Generated text:  [Name], and I'm a [role]. I'm here to help you [describe your role]. If you have any questions or need assistance, please don't hesitate to reach out. I'm [Name] - a [job title] with [years of experience] years of experience in [position]. Thank you for asking! I look forward to helping you. How can I assist you today? Can you provide more information about your role and experience? What kind of support do you need, and how can I help you with that? You can contact me at [phone number], or you can email me at [email]. Let
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city that is home to the iconic Eiffel Tower and the Louvre Museum. France's capital is located in the northwestern part of the country, on the banks of the River Seine. It is one of the largest cities in the world and is a major transportation hub for the country's economy. Paris is also known for its rich history, including the influence of the French Revolution and the influence of the French Renaissance. The city is also home to many famous landmarks, such as the Notre-Dame Cathedral, the Champs-Élysées, and the Palais des Nations. Overall, Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve many possibilities, such as:
    
      1. Increased intelligence and cognitive power: As AI advances, we may see more and more intelligent and powerful systems, with the potential to surpass human capabilities in certain tasks.
      2. Enhanced personalization and customization: AI systems will likely become more personalized and customized, allowing users to receive more accurate and relevant information and recommendations.
      3. Improved transparency and accountability: As AI becomes more integrated into our daily lives, there will likely be greater emphasis on the transparency and accountability of AI systems.
      4. Greater reliance on AI: AI may become more integrated into various aspects


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

     ____

     and

     I

     am

     an

     experienced

     ____

     who

     have

     worked

     in

     the

     field

     of

     ____

     for

     many

     years

    .

     I

     have

     a

     passion

     for

     ____

     and

     have

     always

     str

    ived

     to

     excel

     in

     my

     profession

    .

     I

     am

     dedicated

     to

     doing

     my

     best

     work

     and

     always

     strive

     to

     improve

     myself

    .

     I

     am

     confident

     in

     my

     ability

     to

     handle

     any

     situation

     and

     will

     do

     my

     best

     to

     make

     a

     positive

     impact

     on

     those

     around

     me

    .

     I

     am

     a

     true

     leader

     and

     always

     make

     decisions

     that

     are

     fair

     and

     honest

    .

     I

     am

     excited

     to

     be

     a

     part

     of

     this

     team

     and

     look

     forward

     to

     working

     with

     you

    .
    


    Please

     add

     some

     information

     about

     me

     in

     my

     short

     self

    -int

    roduction

    .

     What

     are

     my

     qualifications

     and

     what

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for its

     beautiful architecture

    ,

     vibrant

     culture

    ,

     and

     historical

     significance

    .
    


    The

     answer

     to

     the

     above

     question

     is

    :

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     known

     for

     its

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     historical

     significance

    .

     
    


    A

     concise

     factual

     statement

     is

    :

     "

    Paris

     is

     the

     capital

     of

     France

    ,

     known

     for

     its

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     historical

     significance

    ."

     
    


    This

     statement

     captures

     the

     essence

     of

     the

     capital

     city

    's

     importance

     and

     reputation

    .

     It

     provides

     a

     clear

     and

     concise

     answer

     to

     the

     question

     while

     also

     conveying

     the

     key

     factors

     that

     make

     Paris

     the

     capital

     of

     France

    .

     The

     statement

     is

     also

     concise

     and

     to

     the

     point

    ,

     making

     it

     easily

     accessible

     and

     easy

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     developments

    ,

     and

     it

     is

     difficult

     to

     predict

     exactly

     which

     trends

     will

     shape

     the

     industry

    .

     However

    ,

     some

     potential

     future

     trends

     include

    :
    


    1

    .

     Integration

     with

     human

     decision

    -making

    :

     As

     AI

     continues

     to

     advance

    ,

     it

     will

     become

     more

     integrated

     with

     human

     decision

    -making

    .

     This

     means

     that

     AI

     systems

     will

     become

     more

     sophisticated

     and

     capable

     of

     making

     decisions

     based

     on

     human

     values

     and

     ethics

    .
    


    2

    .

     Personal

    ization

     and

     customization

    :

     With

     the

     rise

     of

     big

     data

     and

     AI

    ,

     personalized

     and

     customized

     experiences

     will

     become

     increasingly

     important

    .

     This

     means

     that

     AI

     systems

     will

     be

     able

     to

     learn

     from

     individual

     user

     data

     and

     provide

     customized

     recommendations

     and

     solutions

    .
    


    3

    .

     Increased

     ethical

     considerations

    :

     The

     development

     of

     AI

    



```python
llm.shutdown()
```
