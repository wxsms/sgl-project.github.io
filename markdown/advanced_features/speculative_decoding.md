# Speculative Decoding

SGLang now provides an EAGLE-based (EAGLE-2/EAGLE-3) speculative decoding option. Our implementation aims to maximize speed and efficiency and is considered to be among the fastest in open-source LLM engines.

### Performance Highlights

Please see below for the huge improvements on throughput for LLaMA-Instruct 3.1 8B tested on MT bench that can be achieved via EAGLE3 decoding.
For further details please see the [EAGLE3 paper](https://arxiv.org/pdf/2503.01840).

| Method | Throughput (tokens/s) |
|--------|----------------|
| SGLang (w/o speculative, 1x H100) | 158.34 tokens/s |
| SGLang + EAGLE-2 (1x H100) | 244.10 tokens/s |
| SGLang + EAGLE-3 (1x H100) | 373.25 tokens/s |

## EAGLE Decoding

To enable EAGLE speculative decoding the following parameters are relevant:
* `speculative_draft_model_path`: Specifies draft model. This parameter is required.
* `speculative_num_steps`: Depth of autoregressive drafting. Increases speculation range but risks rejection cascades. Default is 5.
* `speculative_eagle_topk`: Branching factor per step. Improves candidate diversity, will lead to higher acceptance rate, but more lead to higher memory/compute consumption. Default is 4.
* `speculative_num_draft_tokens`: Maximum parallel verification capacity. Allows deeper tree evaluation but will lead to higher GPU memory usage. Default is 8.

These parameters are the same for EAGLE-2 and EAGLE-3.

You can find the best combinations of these parameters with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py).

In the documentation below, we set `--cuda-graph-max-bs` to be a small value for faster engine startup. For your own workloads, please tune the above parameters together with `--cuda-graph-max-bs`, `--max-running-requests`, `--mem-fraction-static` for the best performance. 

### EAGLE-2 decoding

You can enable EAGLE-2 decoding by setting `--speculative-algorithm EAGLE` and choosing an appropriate model.


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

import openai
```

    [2026-02-05 03:04:32] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-05 03:04:32] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-05 03:04:32] INFO utils.py:164: NumExpr defaulting to 16 threads.



```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 3 \
    --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 --cuda-graph-max-bs 8 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-05 03:04:38] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:04:38] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:04:38] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 03:04:40] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 03:04:40] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 03:04:40] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 03:04:46] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:04:46] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:04:46] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 03:04:46] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:04:46] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:04:46] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 03:04:52] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:04:52] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:04:52] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 03:04:52] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.81s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.32s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.39s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=24.52 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=24.52 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=3 avail_mem=24.44 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=2 avail_mem=24.43 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=2 avail_mem=24.43 GB):  75%|███████▌  | 3/4 [00:00<00:00,  7.96it/s]Capturing batches (bs=1 avail_mem=24.40 GB):  75%|███████▌  | 3/4 [00:00<00:00,  7.96it/s]Capturing batches (bs=1 avail_mem=24.40 GB): 100%|██████████| 4/4 [00:00<00:00,  7.48it/s]


    [2026-02-05 03:04:57] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 03:04:57] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.39s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.39s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=23.53 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=23.53 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.69s/it]Capturing batches (bs=3 avail_mem=22.00 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.69s/it]

    Capturing batches (bs=3 avail_mem=22.00 GB):  50%|█████     | 2/4 [00:05<00:04,  2.40s/it]Capturing batches (bs=2 avail_mem=21.93 GB):  50%|█████     | 2/4 [00:05<00:04,  2.40s/it]

    Capturing batches (bs=2 avail_mem=21.93 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.44s/it]Capturing batches (bs=1 avail_mem=19.40 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.44s/it]

    Capturing batches (bs=1 avail_mem=19.40 GB): 100%|██████████| 4/4 [00:09<00:00,  2.23s/it]Capturing batches (bs=1 avail_mem=19.40 GB): 100%|██████████| 4/4 [00:09<00:00,  2.30s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=18.53 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=18.47 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=18.47 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=18.45 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=18.45 GB): 100%|██████████| 4/4 [00:00<00:00, 79.08it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='d3bd94c2c1294d87a5a8e5dc8afaedd0', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770260716, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



```python
terminate_process(server_process)
```

### EAGLE-2 Decoding with `torch.compile`

You can also enable `torch.compile` for further optimizations and optionally set `--torch-compile-max-bs`:



```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 \
        --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --mem-fraction 0.6 \
            --enable-torch-compile --torch-compile-max-bs 2 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-05 03:05:22] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:05:22] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:05:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 03:05:24] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 03:05:24] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 03:05:24] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 03:05:30] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:05:30] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:05:30] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 03:05:30] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:05:30] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:05:30] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 03:05:35] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:05:35] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:05:35] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 03:05:35] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.68s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.23s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=35.96 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=35.96 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.70it/s]Capturing batches (bs=3 avail_mem=35.87 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.70it/s]Capturing batches (bs=2 avail_mem=35.85 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.70it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.04809600114822388, "best_triton_pos": 1, "best_triton_time": 0.04940799996256828, "best_triton_kernel": "triton_mm_18", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(128x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0481 ms 100.0% 
      triton_mm_18 0.0494 ms 97.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_12 0.0529 ms 91.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_8 0.0553 ms 87.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_7 0.0559 ms 86.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_11 0.0560 ms 85.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_17 0.0580 ms 82.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_10 0.0671 ms 71.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_14 0.0687 ms 70.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_4 0.0712 ms 67.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4387 seconds and 0.3060 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.02223999984562397, "best_triton_pos": 1, "best_triton_time": 0.02348800003528595, "best_triton_kernel": "triton_mm_27", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0222 ms 100.0% 
      triton_mm_27 0.0235 ms 94.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_31 0.0267 ms 83.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_23 0.0299 ms 74.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_37 0.0318 ms 69.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_26 0.0396 ms 56.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_30 0.0410 ms 54.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_20 0.0420 ms 52.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_22 0.0422 ms 52.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_36 0.0427 ms 52.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.2800 seconds and 0.3861 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_49", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.07574400305747986, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_49 0.0757 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.0764 ms 99.2% 
      triton_mm_55 0.0783 ms 96.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_50 0.0811 ms 93.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_56 0.0839 ms 90.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_45 0.0952 ms 79.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_46 0.1004 ms 75.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_47 0.1005 ms 75.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_48 0.1032 ms 73.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_54 0.1043 ms 72.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4693 seconds and 0.1148 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.04732799902558327, "best_triton_pos": 1, "best_triton_time": 0.04972799867391586, "best_triton_kernel": "triton_mm_65", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0473 ms 100.0% 
      triton_mm_65 0.0497 ms 95.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_69 0.0547 ms 86.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_61 0.0641 ms 73.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_75 0.0683 ms 69.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_64 0.0888 ms 53.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_68 0.0919 ms 51.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_74 0.0958 ms 49.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_60 0.0990 ms 47.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_58 0.1012 ms 46.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4251 seconds and 0.0002 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_93", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.1045759990811348, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_93 0.1046 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.1050 ms 99.6% 
      triton_mm_94 0.1058 ms 98.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_88 0.1114 ms 93.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_87 0.1200 ms 87.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_83 0.1235 ms 84.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_92 0.1331 ms 78.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_84 0.1384 ms 75.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_85 0.1410 ms 74.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_89 0.1463 ms 71.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.5160 seconds and 0.3254 seconds precompiling for 20 choices


    Capturing batches (bs=2 avail_mem=35.85 GB):  75%|███████▌  | 3/4 [00:20<00:07,  7.65s/it]Capturing batches (bs=1 avail_mem=55.05 GB):  75%|███████▌  | 3/4 [00:20<00:07,  7.65s/it]

    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04726399853825569, "best_triton_pos": 1, "best_triton_time": 0.04819199815392494, "best_triton_kernel": "triton_mm_107", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0473 ms 100.0% 
      triton_mm_107 0.0482 ms 98.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_111 0.0484 ms 97.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_103 0.0489 ms 96.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_99 0.0507 ms 93.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_102 0.0552 ms 85.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_106 0.0558 ms 84.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_96 0.0564 ms 83.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_98 0.0617 ms 76.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_97 0.0647 ms 73.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.3034 seconds and 0.2276 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.021536000072956085, "best_triton_pos": 1, "best_triton_time": 0.022304000332951546, "best_triton_kernel": "triton_mm_116", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0215 ms 100.0% 
      triton_mm_116 0.0223 ms 96.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_120 0.0234 ms 92.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_124 0.0265 ms 81.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_128 0.0301 ms 71.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_119 0.0389 ms 55.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_113 0.0395 ms 54.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_123 0.0404 ms 53.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_115 0.0406 ms 53.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_114 0.0425 ms 50.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.2545 seconds and 0.2933 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_137", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4", "best_time": 0.07478400319814682, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_137 0.0748 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_140 0.0749 ms 99.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_136 0.0750 ms 99.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      mm 0.0761 ms 98.3% 
      triton_mm_141 0.0764 ms 97.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_145 0.0794 ms 94.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_133 0.0836 ms 89.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_139 0.0915 ms 81.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_142 0.0945 ms 79.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_143 0.0955 ms 78.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.3591 seconds and 0.1420 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_150", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4", "best_time": 0.04745600000023842, "best_triton_pos": 0}
    AUTOTUNE mm(64x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      triton_mm_150 0.0475 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.0487 ms 97.5% 
      triton_mm_154 0.0493 ms 96.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_158 0.0563 ms 84.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_162 0.0654 ms 72.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_153 0.0894 ms 53.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_149 0.0896 ms 53.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_148 0.0938 ms 50.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_157 0.0938 ms 50.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_147 0.0947 ms 50.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3938 seconds and 0.0002 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_174", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.10016000270843506, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_174 0.1002 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_175 0.1004 ms 99.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_179 0.1007 ms 99.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_170 0.1014 ms 98.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_171 0.1014 ms 98.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.1028 ms 97.4% 
      triton_mm_167 0.1121 ms 89.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_172 0.1193 ms 83.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_176 0.1217 ms 82.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_173 0.1336 ms 75.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4537 seconds and 0.4588 seconds precompiling for 18 choices


    Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:38<00:00, 11.12s/it]Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:38<00:00,  9.61s/it]


    [2026-02-05 03:06:17] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 03:06:17] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.43s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.43s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=13.73 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=13.73 GB):  25%|██▌       | 1/4 [00:06<00:18,  6.01s/it]Capturing batches (bs=3 avail_mem=54.01 GB):  25%|██▌       | 1/4 [00:06<00:18,  6.01s/it]

    Capturing batches (bs=3 avail_mem=54.01 GB):  50%|█████     | 2/4 [00:06<00:05,  2.97s/it]Capturing batches (bs=2 avail_mem=53.99 GB):  50%|█████     | 2/4 [00:06<00:05,  2.97s/it]

    Capturing batches (bs=2 avail_mem=53.99 GB):  75%|███████▌  | 3/4 [00:07<00:01,  1.71s/it]Capturing batches (bs=1 avail_mem=53.95 GB):  75%|███████▌  | 3/4 [00:07<00:01,  1.71s/it]

    Capturing batches (bs=1 avail_mem=53.95 GB): 100%|██████████| 4/4 [00:11<00:00,  2.75s/it]Capturing batches (bs=1 avail_mem=53.95 GB): 100%|██████████| 4/4 [00:11<00:00,  2.85s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=53.23 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=53.16 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=53.16 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=53.14 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=53.14 GB): 100%|██████████| 4/4 [00:00<00:00, 71.86it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='836102d004a54229bd60667527087bdf', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770260799, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



```python
terminate_process(server_process)
```

### EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling

By employing a truncated high-frequency token vocabulary in the draft model, Eagle speculative decoding reduces `lm_head` computational overhead while accelerating the pipeline without quality degradation. For more details, checkout [the paper](https://arxiv.org/pdf/arXiv:2502.14856).

In our implementation, set `--speculative-token-map` to enable the optimization. You can get the high-frequency token in FR-Spec from [this model](https://huggingface.co/thunlp/LLaMA3-Instruct-8B-FR-Spec). Or you can obtain high-frequency token by directly downloading these token from [this repo](https://github.com/thunlp/FR-Spec/tree/main?tab=readme-ov-file#prepare-fr-spec-vocabulary-subset).

Thanks for the contribution from [Weilin Zhao](https://github.com/Achazwl) and [Zhousx](https://github.com/Zhou-sx). 


```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
    --mem-fraction 0.7 --cuda-graph-max-bs 2 --dtype float16  --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-05 03:06:44] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:06:44] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:06:44] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 03:06:46] WARNING model_config.py:1134: Casting torch.bfloat16 to torch.float16.
    [2026-02-05 03:06:46] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 03:06:46] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 03:06:46] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [2026-02-05 03:06:46] Casting torch.bfloat16 to torch.float16.


    [2026-02-05 03:06:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:06:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:06:53] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 03:06:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:06:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:06:53] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 03:06:55] Casting torch.bfloat16 to torch.float16.


    [2026-02-05 03:06:55] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 03:06:58] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:06:58] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:06:58] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 03:06:58] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.41s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.32s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.18s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.00s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.47s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=44.51 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=44.51 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.53it/s]Capturing batches (bs=3 avail_mem=44.33 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.53it/s]Capturing batches (bs=2 avail_mem=44.33 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.53it/s]Capturing batches (bs=2 avail_mem=44.33 GB):  75%|███████▌  | 3/4 [00:00<00:00,  7.05it/s]Capturing batches (bs=1 avail_mem=44.30 GB):  75%|███████▌  | 3/4 [00:00<00:00,  7.05it/s]Capturing batches (bs=1 avail_mem=44.30 GB): 100%|██████████| 4/4 [00:00<00:00,  7.29it/s]


    [2026-02-05 03:07:15] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 03:07:15] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-05 03:07:15] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-05 03:07:15] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.00s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.00s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=26.88 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=26.88 GB):  25%|██▌       | 1/4 [00:03<00:10,  3.46s/it]Capturing batches (bs=3 avail_mem=26.66 GB):  25%|██▌       | 1/4 [00:03<00:10,  3.46s/it]

    Capturing batches (bs=3 avail_mem=26.66 GB):  50%|█████     | 2/4 [00:04<00:03,  1.76s/it]Capturing batches (bs=2 avail_mem=26.64 GB):  50%|█████     | 2/4 [00:04<00:03,  1.76s/it]Capturing batches (bs=2 avail_mem=26.64 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.03s/it]Capturing batches (bs=1 avail_mem=26.59 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.03s/it]

    Capturing batches (bs=1 avail_mem=26.59 GB): 100%|██████████| 4/4 [00:06<00:00,  1.48s/it]Capturing batches (bs=1 avail_mem=26.59 GB): 100%|██████████| 4/4 [00:06<00:00,  1.59s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=20.60 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=20.53 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=20.53 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=20.51 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=20.51 GB): 100%|██████████| 4/4 [00:00<00:00, 54.91it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='a1594a2627604413be8b7604f9373ddb', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770260852, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



```python
terminate_process(server_process)
```

### EAGLE-3 Decoding

You can enable EAGLE-3 decoding by setting `--speculative-algorithm EAGLE3` and choosing an appropriate model.


```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct  --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B --speculative-num-steps 5 \
        --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 --mem-fraction 0.6 \
        --cuda-graph-max-bs 2 --dtype float16 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-05 03:07:37] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:07:37] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:07:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 03:07:39] WARNING model_config.py:1134: Casting torch.bfloat16 to torch.float16.
    [2026-02-05 03:07:39] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 03:07:39] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 03:07:39] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [2026-02-05 03:07:40] Casting torch.bfloat16 to torch.float16.


    [2026-02-05 03:07:46] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:07:46] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:07:46] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 03:07:46] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:07:46] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:07:46] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 03:07:48] Casting torch.bfloat16 to torch.float16.


    [2026-02-05 03:07:49] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 03:07:52] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:07:52] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:07:52] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 03:07:53] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:12,  4.00s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.08s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.06s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  2.92s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.34s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.06 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.06 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.31it/s]Capturing batches (bs=3 avail_mem=59.94 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.31it/s]Capturing batches (bs=2 avail_mem=59.93 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.31it/s]Capturing batches (bs=1 avail_mem=59.91 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.31it/s]Capturing batches (bs=1 avail_mem=59.91 GB): 100%|██████████| 4/4 [00:00<00:00,  8.21it/s]Capturing batches (bs=1 avail_mem=59.91 GB): 100%|██████████| 4/4 [00:00<00:00,  6.89it/s]


    [2026-02-05 03:08:09] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 03:08:09] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-05 03:08:09] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-05 03:08:09] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.84it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.84it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.75 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.75 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.72s/it]Capturing batches (bs=3 avail_mem=58.70 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.72s/it]

    Capturing batches (bs=3 avail_mem=58.70 GB):  50%|█████     | 2/4 [00:03<00:02,  1.34s/it]Capturing batches (bs=2 avail_mem=58.66 GB):  50%|█████     | 2/4 [00:03<00:02,  1.34s/it]Capturing batches (bs=1 avail_mem=58.62 GB):  50%|█████     | 2/4 [00:03<00:02,  1.34s/it]

    Capturing batches (bs=1 avail_mem=58.62 GB): 100%|██████████| 4/4 [00:05<00:00,  1.14s/it]Capturing batches (bs=1 avail_mem=58.62 GB): 100%|██████████| 4/4 [00:05<00:00,  1.28s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.57 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.49 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.49 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.47 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.47 GB): 100%|██████████| 4/4 [00:00<00:00, 110.49it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='124b95beb1164c8a995d76674203c53a', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770260903, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



```python
terminate_process(server_process)
```

## Multi Token Prediction

We support [MTP(Multi-Token Prediction)](https://arxiv.org/pdf/2404.19737) in SGLang by using speculative decoding. We use Xiaomi/MiMo-7B-RL model as example here (deepseek mtp usage refer to [deepseek doc](../basic_usage/deepseek.md#multi-token-prediction))


```python
server_process, port = launch_server_cmd(
    """
    python3 -m sglang.launch_server --model-path XiaomiMiMo/MiMo-7B-RL --host 0.0.0.0 --trust-remote-code \
    --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
    --mem-fraction 0.5 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-05 03:08:28] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:08:28] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:08:28] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 03:08:31] INFO server_args.py:1796: Attention backend not specified. Use fa3 backend by default.
    [2026-02-05 03:08:31] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 03:08:31] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 03:08:37] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:08:37] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:08:37] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 03:08:37] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 03:08:37] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 03:08:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 03:08:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:08:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 03:08:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 03:08:43] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:04,  1.60s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:04<00:04,  2.26s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:06<00:02,  2.43s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:09<00:00,  2.33s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:09<00:00,  2.28s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=28.73 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=28.73 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.85it/s]Capturing batches (bs=3 avail_mem=28.66 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.85it/s]Capturing batches (bs=2 avail_mem=28.65 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.85it/s]Capturing batches (bs=2 avail_mem=28.65 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.56it/s]Capturing batches (bs=1 avail_mem=28.65 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.56it/s]Capturing batches (bs=1 avail_mem=28.65 GB): 100%|██████████| 4/4 [00:00<00:00,  5.72it/s]


    [2026-02-05 03:08:54] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 03:08:54] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  5.15it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  5.81it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  5.76it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=26.88 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=26.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=26.74 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=26.74 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=26.74 GB): 100%|██████████| 4/4 [00:00<00:00, 36.15it/s]Capturing batches (bs=1 avail_mem=26.74 GB): 100%|██████████| 4/4 [00:00<00:00, 36.09it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
import requests

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "XiaomiMiMo/MiMo-7B-RL",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'id': '97ac07a0c6ec46808ac55582f263df7d', 'object': 'chat.completion', 'created': 1770260946, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<think>\nAlright, let\'s tackle the question: "What is the capital of France?" Okay, so I need to figure out the answer to that. Let me start by recalling what I know about France. France is a country in Europe, right? I remember that Paris is a major city there. But wait, is Paris actually the capital? I think it is, but maybe I should double-check to make sure I\'m not confusing it with another city.\n\nLet me think about other capitals. For example, London is the capital of the United Kingdom, Berlin for Germany, Rome for Italy. So following that pattern, Paris would make sense as the capital of France. But wait, have there been any changes recently? I don\'t recall any news about the capital of France changing. Maybe some cities have applied, but I don\'t think anything official happened. \n\nAnother angle: famous landmarks. The Eiffel Tower is in Paris. That\'s a strong indicator. Also, the Louvre Museum is there, which is a major tourist attraction. The President of France lives in the Elysée Palace, which is in Paris. So politically, Paris is definitely the seat of government. \n\nWait, could there be a trick here? Like, sometimes countries have multiple capitals, but France is a republic, and I think they only have one capital. Let me confirm that. Yes, France is a unitary state, so only one capital. Paris has been the capital since the French Revolution, right? Before that, maybe some other city, but not sure. \n\nI might also consider regional capitals. For example, some countries have regions with their own capitals, but France\'s national capital is definitely Paris. For instance, Lille is the capital of the Hauts-de-France region, but that\'s not the national capital. \n\nSo putting it all together: Paris is the capital of France. The evidence includes being home to major landmarks, the seat of government, and historical significance. I don\'t see any contradictions here. I think the answer is straightforward. Just to be thorough, maybe check if there\'s any official source that says otherwise, but I\'m pretty confident. No, the official websites like the government of France or tourists\' guides all list Paris as the capital. \n\nTherefore, the answer should be Paris. There\'s no confusion here. Unless the question is a trick, but it doesn\'t seem like it. The user is likely testing basic knowledge. Yeah, solid answer.\n</think>\n\nThe capital of France is **Paris**. This iconic city is home to major landmarks such as the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. It serves as the political, cultural, and economic center of the country, with institutions like the French government and the National Assembly headquartered there. Paris has been the capital since the French Revolution and remains a global hub for art, cuisine, and diplomacy.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 621, 'completion_tokens': 595, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



```python
terminate_process(server_process)
```

## References

EAGLE process is as follows:

- Within EAGLE the draft model predicts the next feature vector, i.e. the last hidden state of the original LLM, using the feature sequence $(f_1, ..., f_k)$ and the token sequence $(t_2, ..., t_{k+1})$. 
- The next token is then sampled from $p_{k+2}=\text{LMHead}(f_{k+1})$. Afterwards, the two sequences are extended in a tree style—branching out multiple potential continuations, with the branching factor per step controlled by the `speculative_eagle_topk` parameter—to ensure a more coherent connection of context, and are given as input again.
- EAGLE-2 additionally uses the draft model to evaluate how probable certain branches in the draft tree are, dynamically stopping the expansion of unlikely branches. After the expansion phase, reranking is employed to select only the top `speculative_num_draft_tokens` final nodes as draft tokens.
- EAGLE-3 removes the feature prediction objective, incorporates low and mid-layer features, and is trained in an on-policy manner.

This enhances drafting accuracy by operating on the features instead of tokens for more regular inputs and passing the tokens from the next timestep additionally to minimize randomness effects from sampling. Furthermore the dynamic adjustment of the draft tree and selection of reranked final nodes increases acceptance rate of draft tokens further. For more details see [EAGLE-2](https://arxiv.org/abs/2406.16858) and [EAGLE-3](https://arxiv.org/abs/2503.01840) paper.


For guidance how to train your own EAGLE model please see the [EAGLE repo](https://github.com/SafeAILab/EAGLE/tree/main?tab=readme-ov-file#train).
