# Speculative Decoding

SGLang provides several speculative decoding options, including EAGLE-2/EAGLE-3, MTP, classic draft-model decoding, and an NGRAM-based variant. Our implementation aims to maximize speed and efficiency and is considered to be among the fastest in open-source LLM engines.

## Summary

### Jump to sections

- [EAGLE Decoding](#eagle-decoding)
  - [EAGLE-2 decoding](#eagle-2-decoding)
  - [EAGLE-2 Decoding with torch.compile](#eagle-2-decoding-with-torchcompile)
  - [EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling](#eagle-2-decoding-via-frequency-ranked-speculative-sampling)
  - [EAGLE-3 Decoding](#eagle-3-decoding)
- [Multi Token Prediction](#multi-token-prediction)
- [Standalone Speculative Decoding (Small Draft Model)](#standalone-speculative-decoding-small-draft-model)
- [Speculative Decoding V2 (Overlap Scheduler)](#speculative-decoding-v2-overlap-scheduler)
- [Ngram Speculative Decoding](#ngram-speculative-decoding)

### Quick guidance

- **Best speed/quality (recommended)**: Use **EAGLE-3** with `--speculative-algorithm EAGLE3`.
- **Strong default / broad compatibility**: Use **EAGLE-2** with `--speculative-algorithm EAGLE`.
- **Lower `lm_head` overhead for EAGLE-2**: Enable **FR-Spec** with `--speculative-token-map`.
- **Model is MTP-enabled**: Use **MTP via speculative decoding** (often with small `speculative_num_steps/topk/num_draft_tokens`, see the example section).
- **You have a smaller draft LLM**: Use **STANDALONE** (`--speculative-algorithm STANDALONE`).
- **No extra model available**: Use **NGRAM** (`--speculative-algorithm NGRAM`, CUDA-only).
- **Want overlap scheduler (experimental)**: Enable **SpecV2** with `SGLANG_ENABLE_SPEC_V2=True` (requires `--speculative-eagle-topk 1`).

### Method comparison (mini table)

| Method | Draft source | Separate draft model? | How to enable | Notes / constraints |
|---|---|---:|---|---|
| EAGLE-2 | EAGLE draft model (feature drafting + tree) | Typically yes | `--speculative-algorithm EAGLE` + `--speculative-draft-model-path ...` | Tune `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` |
| EAGLE-2 + `torch.compile` | Same as EAGLE-2 | Typically yes | Add `--enable-torch-compile` (optionally `--torch-compile-max-bs`) | Further kernel-level optimizations |
| EAGLE-2 + FR-Spec | Same as EAGLE-2 + token subset | Typically yes | Add `--speculative-token-map ...` | Reduces `lm_head` overhead with high-frequency token vocab |
| EAGLE-3 | EAGLE3 draft model | Yes | `--speculative-algorithm EAGLE3` + `--speculative-draft-model-path ...` | Best throughput in the benchmark above |
| MTP | Built-in multi-token heads (model-specific) | Often no | See **Multi Token Prediction** section | Uses speculative workflow; draft path may be auto-handled for some models |
| STANDALONE | Smaller draft LLM (token-level) | Yes | `--speculative-algorithm STANDALONE` + `--speculative-draft-model-path ...` | Does **not** support `--enable-dp-attention` |
| SpecV2 (experimental) | V2 workers + overlap scheduler | N/A | `SGLANG_ENABLE_SPEC_V2=True` | Only supports `--speculative-eagle-topk 1`; applies to `EAGLE`, `EAGLE3`, `STANDALONE` |
| NGRAM | Ngram cache from previous tokens | No | `--speculative-algorithm NGRAM` | CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill |

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
* `speculative_draft_model_path`: Draft model path/weights. **Typically required** for EAGLE/EAGLE3 and STANDALONE. For some MTP-enabled models, this can be omitted (SGLang may auto-handle/auto-fill it).
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

    [2026-02-10 03:55:48] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-10 03:55:48] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-10 03:55:48] INFO utils.py:164: NumExpr defaulting to 16 threads.



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

    [2026-02-10 03:55:54] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:55:54] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:55:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 03:55:57] INFO server_args.py:1800: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-10 03:55:57] WARNING server_args.py:2308: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-10 03:55:57] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 03:56:04] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:56:04] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:56:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 03:56:05] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:56:05] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:56:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 03:56:10] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:56:10] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:56:10] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 03:56:10] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.65s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.16s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.24s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=24.52 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=24.52 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=3 avail_mem=24.44 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=2 avail_mem=24.43 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=2 avail_mem=24.43 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.40it/s]Capturing batches (bs=1 avail_mem=24.41 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.40it/s]

    Capturing batches (bs=1 avail_mem=24.41 GB): 100%|██████████| 4/4 [00:00<00:00,  4.28it/s]


    [2026-02-10 03:56:15] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-10 03:56:15] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.30s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.30s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=23.53 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=23.53 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.01s/it]Capturing batches (bs=3 avail_mem=23.44 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.01s/it]

    Capturing batches (bs=3 avail_mem=23.44 GB):  50%|█████     | 2/4 [00:05<00:04,  2.44s/it]Capturing batches (bs=2 avail_mem=23.44 GB):  50%|█████     | 2/4 [00:05<00:04,  2.44s/it]

    Capturing batches (bs=2 avail_mem=23.44 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.46s/it]Capturing batches (bs=1 avail_mem=23.41 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.46s/it]

    Capturing batches (bs=1 avail_mem=23.41 GB): 100%|██████████| 4/4 [00:09<00:00,  2.22s/it]Capturing batches (bs=1 avail_mem=23.41 GB): 100%|██████████| 4/4 [00:09<00:00,  2.33s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=23.39 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=23.33 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=23.33 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=23.31 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=23.31 GB): 100%|██████████| 4/4 [00:00<00:00, 83.92it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='059136b0de6a4846b11bd0975f09627a', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770695795, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-10 03:56:40] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:56:40] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:56:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 03:56:42] INFO server_args.py:1800: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-10 03:56:42] WARNING server_args.py:2308: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-10 03:56:42] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 03:56:49] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:56:49] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:56:49] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-10 03:56:49] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:56:49] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:56:49] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 03:56:54] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:56:54] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:56:54] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 03:56:54] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.60s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.09s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.17s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=40.17 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=40.17 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.53it/s]Capturing batches (bs=3 avail_mem=40.08 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.53it/s]Capturing batches (bs=2 avail_mem=40.07 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.53it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.048287998884916306, "best_triton_pos": 1, "best_triton_time": 0.049247998744249344, "best_triton_kernel": "triton_mm_18", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(128x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0483 ms 100.0% 
      triton_mm_18 0.0492 ms 98.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_12 0.0527 ms 91.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_8 0.0552 ms 87.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_11 0.0561 ms 86.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_7 0.0562 ms 85.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_17 0.0585 ms 82.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_10 0.0679 ms 71.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_14 0.0693 ms 69.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_4 0.0716 ms 67.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3589 seconds and 0.3284 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.022048000246286392, "best_triton_pos": 1, "best_triton_time": 0.023264000192284584, "best_triton_kernel": "triton_mm_27", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0220 ms 100.0% 
      triton_mm_27 0.0233 ms 94.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_31 0.0265 ms 83.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_23 0.0300 ms 73.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_37 0.0315 ms 70.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_26 0.0392 ms 56.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_30 0.0409 ms 53.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_20 0.0421 ms 52.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_22 0.0423 ms 52.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_36 0.0427 ms 51.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.2904 seconds and 0.3329 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_49", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.07507199794054031, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_49 0.0751 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.0762 ms 98.5% 
      triton_mm_55 0.0776 ms 96.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_50 0.0802 ms 93.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_56 0.0826 ms 90.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_45 0.0936 ms 80.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_46 0.0960 ms 78.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_47 0.0968 ms 77.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_54 0.1012 ms 74.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_48 0.1013 ms 74.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4640 seconds and 0.1620 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.0488319993019104, "best_triton_pos": 1, "best_triton_time": 0.05152000114321709, "best_triton_kernel": "triton_mm_65", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0488 ms 100.0% 
      triton_mm_65 0.0515 ms 94.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_69 0.0567 ms 86.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_61 0.0666 ms 73.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_75 0.0700 ms 69.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_64 0.0896 ms 54.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_68 0.0930 ms 52.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_74 0.0975 ms 50.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_58 0.0992 ms 49.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_60 0.0994 ms 49.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4324 seconds and 0.0003 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_93", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.10294400155544281, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_93 0.1029 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.1038 ms 99.2% 
      triton_mm_94 0.1047 ms 98.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_88 0.1083 ms 95.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_83 0.1155 ms 89.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_87 0.1158 ms 88.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_92 0.1268 ms 81.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_84 0.1292 ms 79.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_85 0.1358 ms 75.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_89 0.1440 ms 71.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.5099 seconds and 0.3224 seconds precompiling for 20 choices


    Capturing batches (bs=2 avail_mem=40.07 GB):  75%|███████▌  | 3/4 [00:21<00:07,  7.74s/it]Capturing batches (bs=1 avail_mem=35.76 GB):  75%|███████▌  | 3/4 [00:21<00:07,  7.74s/it]

    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04742399975657463, "best_triton_pos": 1, "best_triton_time": 0.04886399954557419, "best_triton_kernel": "triton_mm_107", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0474 ms 100.0% 
      triton_mm_107 0.0489 ms 97.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_111 0.0490 ms 96.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_103 0.0491 ms 96.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_99 0.0508 ms 93.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_102 0.0553 ms 85.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_106 0.0556 ms 85.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_96 0.0567 ms 83.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_98 0.0636 ms 74.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_97 0.0653 ms 72.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.2892 seconds and 0.2241 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.021888000890612602, "best_triton_pos": 1, "best_triton_time": 0.022336000576615334, "best_triton_kernel": "triton_mm_116", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0219 ms 100.0% 
      triton_mm_116 0.0223 ms 98.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_120 0.0233 ms 94.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_124 0.0267 ms 81.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_128 0.0302 ms 72.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_119 0.0389 ms 56.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_113 0.0393 ms 55.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_123 0.0406 ms 53.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_115 0.0407 ms 53.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_114 0.0423 ms 51.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.2593 seconds and 0.2922 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_136", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8", "best_time": 0.07385600358247757, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_136 0.0739 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_140 0.0744 ms 99.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_137 0.0753 ms 98.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.0757 ms 97.5% 
      triton_mm_141 0.0765 ms 96.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_145 0.0805 ms 91.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_133 0.0837 ms 88.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_139 0.0894 ms 82.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_143 0.0936 ms 78.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_142 0.0938 ms 78.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3449 seconds and 0.1423 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_150", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4", "best_time": 0.04825599864125252, "best_triton_pos": 0}
    AUTOTUNE mm(64x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      triton_mm_150 0.0483 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.0487 ms 99.1% 
      triton_mm_154 0.0496 ms 97.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_158 0.0564 ms 85.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_162 0.0648 ms 74.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_153 0.0889 ms 54.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_149 0.0900 ms 53.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_157 0.0930 ms 51.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_148 0.0931 ms 51.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_147 0.0933 ms 51.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3946 seconds and 0.0002 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_174", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.10019200295209885, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_174 0.1002 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_179 0.1009 ms 99.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_175 0.1011 ms 99.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_170 0.1012 ms 99.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_171 0.1020 ms 98.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.1038 ms 96.5% 
      triton_mm_167 0.1107 ms 90.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_172 0.1204 ms 83.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_176 0.1212 ms 82.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_164 0.1336 ms 75.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4164 seconds and 0.3191 seconds precompiling for 18 choices


    Capturing batches (bs=1 avail_mem=35.76 GB): 100%|██████████| 4/4 [00:39<00:00, 11.54s/it]Capturing batches (bs=1 avail_mem=35.76 GB): 100%|██████████| 4/4 [00:39<00:00,  9.95s/it]


    [2026-02-10 03:57:39] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-10 03:57:39] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.16s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.16s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=19.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=19.90 GB):  25%|██▌       | 1/4 [00:05<00:16,  5.64s/it]Capturing batches (bs=3 avail_mem=14.24 GB):  25%|██▌       | 1/4 [00:05<00:16,  5.64s/it]

    Capturing batches (bs=3 avail_mem=14.24 GB):  50%|█████     | 2/4 [00:06<00:05,  2.75s/it]Capturing batches (bs=2 avail_mem=13.72 GB):  50%|█████     | 2/4 [00:06<00:05,  2.75s/it]

    Capturing batches (bs=2 avail_mem=13.72 GB):  75%|███████▌  | 3/4 [00:06<00:01,  1.59s/it]Capturing batches (bs=1 avail_mem=13.63 GB):  75%|███████▌  | 3/4 [00:06<00:01,  1.59s/it]

    Capturing batches (bs=1 avail_mem=13.63 GB): 100%|██████████| 4/4 [00:10<00:00,  2.62s/it]Capturing batches (bs=1 avail_mem=13.63 GB): 100%|██████████| 4/4 [00:10<00:00,  2.70s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=32.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=32.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=32.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=32.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=32.70 GB): 100%|██████████| 4/4 [00:00<00:00, 68.30it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='292c5bc9071a4a599100df93100a89ab', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770695880, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-10 03:58:06] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:58:06] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:58:06] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 03:58:08] WARNING model_config.py:1141: Casting torch.bfloat16 to torch.float16.
    [2026-02-10 03:58:08] INFO server_args.py:1800: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-10 03:58:08] WARNING server_args.py:2308: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-10 03:58:08] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [2026-02-10 03:58:08] Casting torch.bfloat16 to torch.float16.


    [2026-02-10 03:58:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:58:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:58:14] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-10 03:58:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:58:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:58:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 03:58:16] Casting torch.bfloat16 to torch.float16.


    [2026-02-10 03:58:17] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 03:58:20] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:58:20] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:58:20] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 03:58:20] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.48s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.37s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:13<00:04,  4.36s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.21s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.64s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.19 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.19 GB):  25%|██▌       | 1/4 [00:04<00:12,  4.10s/it]Capturing batches (bs=3 avail_mem=60.02 GB):  25%|██▌       | 1/4 [00:04<00:12,  4.10s/it]Capturing batches (bs=2 avail_mem=60.01 GB):  25%|██▌       | 1/4 [00:04<00:12,  4.10s/it]Capturing batches (bs=2 avail_mem=60.01 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.12s/it]Capturing batches (bs=1 avail_mem=59.99 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.12s/it]

    Capturing batches (bs=1 avail_mem=59.99 GB): 100%|██████████| 4/4 [00:04<00:00,  1.08s/it]


    [2026-02-10 03:58:41] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-10 03:58:41] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-10 03:58:42] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-10 03:58:42] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.01it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.01it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.02 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.02 GB):  25%|██▌       | 1/4 [00:03<00:10,  3.64s/it]Capturing batches (bs=3 avail_mem=58.24 GB):  25%|██▌       | 1/4 [00:03<00:10,  3.64s/it]

    Capturing batches (bs=3 avail_mem=58.24 GB):  50%|█████     | 2/4 [00:04<00:03,  1.83s/it]Capturing batches (bs=2 avail_mem=58.22 GB):  50%|█████     | 2/4 [00:04<00:03,  1.83s/it]Capturing batches (bs=2 avail_mem=58.22 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.07s/it]Capturing batches (bs=1 avail_mem=58.17 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.07s/it]

    Capturing batches (bs=1 avail_mem=58.17 GB): 100%|██████████| 4/4 [00:06<00:00,  1.53s/it]Capturing batches (bs=1 avail_mem=58.17 GB): 100%|██████████| 4/4 [00:06<00:00,  1.65s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=43.78 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=43.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=43.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=43.68 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=43.68 GB): 100%|██████████| 4/4 [00:00<00:00, 104.47it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='89662657483b4388b5647698755b1335', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770695939, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-10 03:59:04] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:59:04] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:59:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 03:59:06] WARNING model_config.py:1141: Casting torch.bfloat16 to torch.float16.
    [2026-02-10 03:59:06] INFO server_args.py:1800: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-10 03:59:06] WARNING server_args.py:2308: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-10 03:59:06] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [2026-02-10 03:59:07] Casting torch.bfloat16 to torch.float16.


    [2026-02-10 03:59:12] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:59:12] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:59:12] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 03:59:12] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 03:59:12] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 03:59:12] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 03:59:15] Casting torch.bfloat16 to torch.float16.


    [2026-02-10 03:59:16] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 03:59:19] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:59:19] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 03:59:19] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 03:59:19] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.34s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.39s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:13<00:04,  4.34s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.07s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.54s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=21.75 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=21.75 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.13it/s]Capturing batches (bs=3 avail_mem=21.63 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.13it/s]Capturing batches (bs=2 avail_mem=21.63 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.13it/s]

    Capturing batches (bs=2 avail_mem=21.63 GB):  75%|███████▌  | 3/4 [00:01<00:00,  1.94it/s]Capturing batches (bs=1 avail_mem=42.75 GB):  75%|███████▌  | 3/4 [00:01<00:00,  1.94it/s]Capturing batches (bs=1 avail_mem=42.75 GB): 100%|██████████| 4/4 [00:01<00:00,  2.30it/s]


    [2026-02-10 03:59:38] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-10 03:59:38] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-10 03:59:38] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-10 03:59:38] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.39it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.39it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=41.59 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=41.59 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.93s/it]Capturing batches (bs=3 avail_mem=41.54 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.93s/it]

    Capturing batches (bs=3 avail_mem=41.54 GB):  50%|█████     | 2/4 [00:03<00:02,  1.45s/it]Capturing batches (bs=2 avail_mem=58.66 GB):  50%|█████     | 2/4 [00:03<00:02,  1.45s/it]

    Capturing batches (bs=2 avail_mem=58.66 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.13it/s]Capturing batches (bs=1 avail_mem=58.62 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.13it/s]

    Capturing batches (bs=1 avail_mem=58.62 GB): 100%|██████████| 4/4 [00:05<00:00,  1.45s/it]Capturing batches (bs=1 avail_mem=58.62 GB): 100%|██████████| 4/4 [00:05<00:00,  1.47s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.57 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.49 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.49 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.47 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.47 GB): 100%|██████████| 4/4 [00:00<00:00, 41.80it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='c6fe5e1bef8f428f94a451c8ddb8b474', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770695996, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-10 04:00:02] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:00:02] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:00:02] INFO utils.py:164: NumExpr defaulting to 16 threads.


    A new version of the following files was downloaded from https://huggingface.co/XiaomiMiMo/MiMo-7B-RL:
    - configuration_mimo.py
    . Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
    [2026-02-10 04:00:04] WARNING dynamic_module_utils.py:485: A new version of the following files was downloaded from https://huggingface.co/XiaomiMiMo/MiMo-7B-RL:
    - configuration_mimo.py
    . Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.


    [2026-02-10 04:00:04] INFO server_args.py:1800: Attention backend not specified. Use fa3 backend by default.
    [2026-02-10 04:00:04] WARNING server_args.py:2308: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-10 04:00:04] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 04:00:10] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:00:10] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:00:10] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-10 04:00:10] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:00:10] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:00:10] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 04:00:16] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:00:16] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:00:16] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 04:00:16] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Could not set the permissions on the file '/hf_home/hub/models--XiaomiMiMo--MiMo-7B-RL/blobs/1198f24370251bfcce3e8b3209ad80d4d17ce6743f36d29b469c0c0455bb0f66.incomplete'. Error: [Errno 2] No such file or directory: '/hf_home/hub/models--XiaomiMiMo--MiMo-7B-RL/blobs/1198f24370251bfcce3e8b3209ad80d4d17ce6743f36d29b469c0c0455bb0f66.incomplete'.
    Continuing without setting permissions.
    [2026-02-10 04:00:34] Could not set the permissions on the file '/hf_home/hub/models--XiaomiMiMo--MiMo-7B-RL/blobs/1198f24370251bfcce3e8b3209ad80d4d17ce6743f36d29b469c0c0455bb0f66.incomplete'. Error: [Errno 2] No such file or directory: '/hf_home/hub/models--XiaomiMiMo--MiMo-7B-RL/blobs/1198f24370251bfcce3e8b3209ad80d4d17ce6743f36d29b469c0c0455bb0f66.incomplete'.
    Continuing without setting permissions.
    [2026-02-10 04:00:34] [CI Download] Process 62726 hit download race condition (attempt 1/3) for XiaomiMiMo/MiMo-7B-RL: FileNotFoundError: [Errno 2] No such file or directory: '/hf_home/hub/models--XiaomiMiMo--MiMo-7B-RL/blobs/1198f24370251bfcce3e8b3209ad80d4d17ce6743f36d29b469c0c0455bb0f66.incomplete'


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.36it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.35it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.36it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.41it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.39it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.12 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.12 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.12it/s]Capturing batches (bs=3 avail_mem=61.06 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.12it/s]Capturing batches (bs=3 avail_mem=61.06 GB):  50%|█████     | 2/4 [00:01<00:00,  2.31it/s]Capturing batches (bs=2 avail_mem=61.05 GB):  50%|█████     | 2/4 [00:01<00:00,  2.31it/s]Capturing batches (bs=1 avail_mem=61.04 GB):  50%|█████     | 2/4 [00:01<00:00,  2.31it/s]

    Capturing batches (bs=1 avail_mem=61.04 GB): 100%|██████████| 4/4 [00:01<00:00,  3.65it/s]


    [2026-02-10 04:00:50] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-10 04:00:50] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  4.33it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  6.73it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  6.46it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.56 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=60.50 GB): 100%|██████████| 4/4 [00:00<00:00, 62.08it/s]



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


<strong style='color: #00008B;'>{'id': 'a03ca8a7bb52431ca16e7b05e5df2e34', 'object': 'chat.completion', 'created': 1770696060, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "<think>\nOkay, let's see. The user is asking for the capital of France. Hmm, I need to make sure I get this right. Let me think. France is a country in Europe, right? I remember that Paris is the capital. But wait, maybe there's some confusion. Let me double-check.\n\nWait, is there another major city in France? Like Lyon or maybe Marseille? No, those are big cities but not the capital. Paris is definitely the main one. The Eiffel Tower is in Paris, right? That's a major landmark. Also, the French government is based in Paris. So that must be the capital. \n\nWait, could there be any trick here? Like, sometimes countries have changed capitals over time. But France has been around for a long time, and Paris has been the capital since the French Republic was established. I don't think they moved it recently. Let me confirm in my mind. Yeah, Paris is the capital. \n\nSo the answer should be Paris straightforward. No need to overcomplicate. Just state Paris as the capital of France. Make sure there's no alternative correct answer. No, I can't think of any other possibilities. Yep, Paris is it.\n</think>\nThe capital of France is **Paris**. This iconic city is known for landmarks like the Eiffel Tower, Louvre Museum, and vibrant neighborhoods such as the Left Bank and Montmartre. As the political and cultural heart of France, it hosts the French government and many global institutions.", 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 340, 'completion_tokens': 314, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



```python
terminate_process(server_process)
```

## Standalone Speculative Decoding (Small Draft Model)

Besides EAGLE/MTP, SGLang also supports **token-level speculative decoding** using a smaller **draft model**. Enable it with `--speculative-algorithm STANDALONE` and provide a draft model via `--speculative-draft-model-path`.

Relevant parameters:
- `--speculative-draft-model-path`: Draft model weights (smaller than the target model).
- `--speculative-num-steps`: Draft depth (how many steps the draft model runs autoregressively).
- `--speculative-eagle-topk`: Branching factor (token candidates per step).
- `--speculative-num-draft-tokens`: Verification capacity.

Note:
- Standalone speculative decoding currently **does not support** `--enable-dp-attention`.



```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 --speculative-eagle-topk 2 --speculative-num-draft-tokens 7 \
    --cuda-graph-max-bs 8 --mem-fraction-static 0.7 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-10 04:01:05] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:01:05] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:01:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 04:01:08] INFO server_args.py:1800: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-10 04:01:08] WARNING server_args.py:2308: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-10 04:01:08] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 04:01:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:01:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:01:14] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-10 04:01:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:01:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:01:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 04:01:19] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:01:19] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:01:19] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 04:01:19] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.35it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.29it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.31it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.34it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.33it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=11.56 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=11.56 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=3 avail_mem=11.48 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=2 avail_mem=11.47 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=2 avail_mem=11.47 GB):  75%|███████▌  | 3/4 [00:00<00:00,  3.80it/s]Capturing batches (bs=1 avail_mem=11.45 GB):  75%|███████▌  | 3/4 [00:00<00:00,  3.80it/s]Capturing batches (bs=1 avail_mem=11.45 GB): 100%|██████████| 4/4 [00:01<00:00,  4.00it/s]


    [2026-02-10 04:01:27] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-10 04:01:27] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]
    
    [2026-02-10 04:01:35] max_total_tokens=20480 is larger than the profiled value -64993. Use the profiled value instead.


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=0.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=0.90 GB):  25%|██▌       | 1/4 [00:04<00:13,  4.61s/it]Capturing batches (bs=3 avail_mem=58.21 GB):  25%|██▌       | 1/4 [00:04<00:13,  4.61s/it]

    Capturing batches (bs=3 avail_mem=58.21 GB):  50%|█████     | 2/4 [00:05<00:04,  2.28s/it]Capturing batches (bs=2 avail_mem=58.20 GB):  50%|█████     | 2/4 [00:05<00:04,  2.28s/it]Capturing batches (bs=2 avail_mem=58.20 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.32s/it]Capturing batches (bs=1 avail_mem=58.15 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.32s/it]

    Capturing batches (bs=1 avail_mem=58.15 GB): 100%|██████████| 4/4 [00:08<00:00,  2.14s/it]Capturing batches (bs=1 avail_mem=58.15 GB): 100%|██████████| 4/4 [00:08<00:00,  2.21s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.11 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.03 GB):  75%|███████▌  | 3/4 [00:00<00:00, 25.38it/s]Capturing batches (bs=1 avail_mem=58.00 GB):  75%|███████▌  | 3/4 [00:00<00:00, 25.38it/s]Capturing batches (bs=1 avail_mem=58.00 GB): 100%|██████████| 4/4 [00:00<00:00, 21.61it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='271d57877743493dbb204eb9edcdedef', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770696113, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



```python
terminate_process(server_process)
```

## Speculative Decoding V2 (Overlap Scheduler)

SGLang provides an **experimental Speculative Decoding V2** implementation that enables an overlap scheduler and uses V2 speculative workers (e.g. `StandaloneWorkerV2`, `EAGLEWorkerV2`).

To enable it, set the environment variable:
- `SGLANG_ENABLE_SPEC_V2=True`

Notes:
- SpecV2 currently only supports `--speculative-eagle-topk 1`. When SpecV2 is enabled, **set `--speculative-eagle-topk 1` explicitly**.
- If you explicitly set `--speculative-eagle-topk > 1`, the server will error. If you omit `--speculative-eagle-topk`, auto-tuning may pick `topk > 1` for some models (e.g. Llama), which is not supported by SpecV2.
- This applies to `EAGLE`, `EAGLE3`, and `STANDALONE`.



```python
server_process, port = launch_server_cmd(
    """
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --cuda-graph-max-bs 8 --mem-fraction-static 0.7 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-10 04:01:58] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:01:58] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:01:58] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 04:02:00] INFO server_args.py:1800: Attention backend not specified. Use fa3 backend by default.
    [2026-02-10 04:02:00] WARNING server_args.py:2296: Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on.
    [2026-02-10 04:02:00] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 04:02:06] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:02:06] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:02:06] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-10 04:02:06] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:02:06] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:02:06] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 04:02:12] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:02:12] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:02:12] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 04:02:12] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.36it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.31it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.40it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.40it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.38it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.70 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.65it/s]Capturing batches (bs=3 avail_mem=60.64 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.65it/s]Capturing batches (bs=2 avail_mem=60.63 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.65it/s]Capturing batches (bs=2 avail_mem=60.63 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.84it/s]Capturing batches (bs=1 avail_mem=60.63 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.84it/s]Capturing batches (bs=1 avail_mem=60.63 GB): 100%|██████████| 4/4 [00:00<00:00,  5.12it/s]


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.07it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.07it/s]
    


    [2026-02-10 04:02:19] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=56.91 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=56.91 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.24s/it]Capturing batches (bs=3 avail_mem=56.79 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.24s/it]

    Capturing batches (bs=3 avail_mem=56.79 GB):  50%|█████     | 2/4 [00:02<00:02,  1.26s/it]Capturing batches (bs=2 avail_mem=56.77 GB):  50%|█████     | 2/4 [00:02<00:02,  1.26s/it]Capturing batches (bs=1 avail_mem=56.76 GB):  50%|█████     | 2/4 [00:02<00:02,  1.26s/it]

    Capturing batches (bs=1 avail_mem=56.76 GB): 100%|██████████| 4/4 [00:03<00:00,  1.30it/s]Capturing batches (bs=1 avail_mem=56.76 GB): 100%|██████████| 4/4 [00:03<00:00,  1.06it/s]


    [2026-02-10 04:02:25] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='caeaf33b010a45a8aa3f5df4d692a0f9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770696149, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



```python
terminate_process(server_process)
```

## Ngram Speculative Decoding

SGLang also supports **ngram-based speculative decoding** (no separate draft model). It retrieves draft tokens from an ngram cache built from previously generated tokens, and then verifies them with the target model.

Enable it with:
- `--speculative-algorithm NGRAM`

Common parameters:
- `--speculative-num-draft-tokens`: Number of draft tokens verified per step.
- `--speculative-ngram-min-match-window-size` / `--speculative-ngram-max-match-window-size`: Matching window range.
- `--speculative-ngram-min-bfs-breadth` / `--speculative-ngram-max-bfs-breadth`: BFS breadth range.
- `--speculative-ngram-branch-length`: How many recent tokens to insert into the cache.
- `--speculative-ngram-capacity`: Cache capacity.

Notes:
- Ngram speculative decoding **only supports CUDA**.
- It currently **does not support** `--enable-dp-attention`.
- It disables the overlap scheduler and mixed chunked prefill.
- Optional: set `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True` to force greedy verification.



```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-match-window-size 12 --speculative-ngram-max-bfs-breadth 10 \
    --cuda-graph-max-bs 8 --mem-fraction-static 0.8 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-10 04:02:35] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:02:35] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:02:35] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-10 04:02:38] INFO server_args.py:1800: Attention backend not specified. Use fa3 backend by default.
    [2026-02-10 04:02:38] WARNING server_args.py:2401: The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding.
    [2026-02-10 04:02:38] INFO server_args.py:2787: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-10 04:02:43] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:02:43] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:02:43] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-10 04:02:49] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:02:49] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-10 04:02:49] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-10 04:02:49] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.
    [2026-02-10 04:02:49] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-10 04:02:49] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-10 04:02:49] INFO utils.py:164: NumExpr defaulting to 16 threads.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.51it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.44it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.35it/s]
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.38it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.39it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=57.78 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=57.78 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.27it/s]Capturing batches (bs=3 avail_mem=57.67 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.27it/s]Capturing batches (bs=3 avail_mem=57.67 GB):  50%|█████     | 2/4 [00:00<00:00,  2.56it/s]Capturing batches (bs=2 avail_mem=57.67 GB):  50%|█████     | 2/4 [00:00<00:00,  2.56it/s]Capturing batches (bs=1 avail_mem=57.66 GB):  50%|█████     | 2/4 [00:00<00:00,  2.56it/s]

    Capturing batches (bs=1 avail_mem=57.66 GB): 100%|██████████| 4/4 [00:01<00:00,  5.12it/s]Capturing batches (bs=1 avail_mem=57.66 GB): 100%|██████████| 4/4 [00:01<00:00,  3.78it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>



```python
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='7f41cacbd29f4c19bec0a76bbd18d961', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770696193, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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
