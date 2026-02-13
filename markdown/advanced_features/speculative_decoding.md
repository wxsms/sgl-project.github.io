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

    [2026-02-13 02:50:11] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-13 02:50:11] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-13 02:50:11] INFO utils.py:164: NumExpr defaulting to 16 threads.



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

    [2026-02-13 02:50:19] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:50:19] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:50:19] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:50:23] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 02:50:23] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 02:50:23] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:50:29] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:50:29] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:50:29] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:50:29] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:50:29] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:50:29] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:50:35] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:50:35] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:50:35] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.64s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.13s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.21s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.22 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.22 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.89it/s]Capturing batches (bs=3 avail_mem=55.14 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.89it/s]Capturing batches (bs=2 avail_mem=55.13 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.89it/s]Capturing batches (bs=2 avail_mem=55.13 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.71it/s]Capturing batches (bs=1 avail_mem=55.11 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.71it/s]

    Capturing batches (bs=1 avail_mem=55.11 GB): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]


    [2026-02-13 02:50:40] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 02:50:40] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.20s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.20s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.23 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.23 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.69s/it]Capturing batches (bs=3 avail_mem=54.14 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.69s/it]

    Capturing batches (bs=3 avail_mem=54.14 GB):  50%|█████     | 2/4 [00:05<00:04,  2.33s/it]Capturing batches (bs=2 avail_mem=54.14 GB):  50%|█████     | 2/4 [00:05<00:04,  2.33s/it]

    Capturing batches (bs=2 avail_mem=54.14 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.40s/it]Capturing batches (bs=1 avail_mem=54.11 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.40s/it]

    Capturing batches (bs=1 avail_mem=54.11 GB): 100%|██████████| 4/4 [00:09<00:00,  2.18s/it]Capturing batches (bs=1 avail_mem=54.11 GB): 100%|██████████| 4/4 [00:09<00:00,  2.26s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.09 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=54.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=54.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=54.01 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=54.01 GB): 100%|██████████| 4/4 [00:00<00:00, 85.48it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='17a2f26c866d4f618586a64c8c385a39', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770951058, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 02:51:03] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:51:03] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:51:03] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:51:06] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 02:51:06] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 02:51:06] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:51:13] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:51:13] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:51:13] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:51:13] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:51:13] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:51:13] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:51:19] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:51:19] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:51:19] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.66s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.18s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.19 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.19 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.77it/s]Capturing batches (bs=3 avail_mem=55.10 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.77it/s]Capturing batches (bs=2 avail_mem=55.08 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.77it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.048767998814582825, "best_triton_pos": 1, "best_triton_time": 0.049695998430252075, "best_triton_kernel": "triton_mm_18", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(128x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0488 ms 100.0% 
      triton_mm_18 0.0497 ms 98.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_12 0.0531 ms 91.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_8 0.0547 ms 89.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_7 0.0564 ms 86.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_11 0.0564 ms 86.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_17 0.0587 ms 83.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_10 0.0675 ms 72.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_14 0.0689 ms 70.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_4 0.0710 ms 68.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3329 seconds and 0.2952 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.022592000663280487, "best_triton_pos": 1, "best_triton_time": 0.023615999147295952, "best_triton_kernel": "triton_mm_27", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0226 ms 100.0% 
      triton_mm_27 0.0236 ms 95.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_31 0.0270 ms 83.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_23 0.0303 ms 74.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_37 0.0321 ms 70.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_26 0.0400 ms 56.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_30 0.0413 ms 54.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_22 0.0420 ms 53.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_20 0.0425 ms 53.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_36 0.0429 ms 52.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.2868 seconds and 0.3586 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_49", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.07612799853086472, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_49 0.0761 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.0772 ms 98.7% 
      triton_mm_55 0.0780 ms 97.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_50 0.0802 ms 94.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_56 0.0832 ms 91.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_45 0.0943 ms 80.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_46 0.0963 ms 79.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_47 0.0977 ms 77.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_54 0.1020 ms 74.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_48 0.1026 ms 74.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4351 seconds and 0.1683 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.050016000866889954, "best_triton_pos": 1, "best_triton_time": 0.05299200117588043, "best_triton_kernel": "triton_mm_65", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0500 ms 100.0% 
      triton_mm_65 0.0530 ms 94.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_69 0.0579 ms 86.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_61 0.0657 ms 76.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_75 0.0719 ms 69.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_64 0.0914 ms 54.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_68 0.0950 ms 52.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_60 0.0967 ms 51.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_74 0.0991 ms 50.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_58 0.1012 ms 49.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4404 seconds and 0.0003 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_93", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.10396800190210342, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_93 0.1040 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_94 0.1042 ms 99.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      mm 0.1053 ms 98.7% 
      triton_mm_88 0.1084 ms 95.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_83 0.1156 ms 90.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_87 0.1159 ms 89.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_92 0.1242 ms 83.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_84 0.1289 ms 80.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_85 0.1331 ms 78.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_89 0.1390 ms 74.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.5042 seconds and 0.3223 seconds precompiling for 20 choices


    Capturing batches (bs=2 avail_mem=55.08 GB):  75%|███████▌  | 3/4 [00:20<00:07,  7.36s/it]Capturing batches (bs=1 avail_mem=55.05 GB):  75%|███████▌  | 3/4 [00:20<00:07,  7.36s/it]

    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04761600121855736, "best_triton_pos": 1, "best_triton_time": 0.04886399954557419, "best_triton_kernel": "triton_mm_111", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(64x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0476 ms 100.0% 
      triton_mm_111 0.0489 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_107 0.0489 ms 97.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_103 0.0496 ms 96.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_99 0.0518 ms 92.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_102 0.0551 ms 86.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_106 0.0556 ms 85.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_96 0.0565 ms 84.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_98 0.0619 ms 77.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_97 0.0652 ms 73.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.2979 seconds and 0.2297 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.021824000403285027, "best_triton_pos": 1, "best_triton_time": 0.02304000034928322, "best_triton_kernel": "triton_mm_116", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0218 ms 100.0% 
      triton_mm_116 0.0230 ms 94.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_120 0.0240 ms 91.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_124 0.0272 ms 80.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_128 0.0308 ms 71.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_119 0.0394 ms 55.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_113 0.0396 ms 55.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_123 0.0408 ms 53.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_115 0.0410 ms 53.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_114 0.0428 ms 51.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.2594 seconds and 0.2948 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_136", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8", "best_time": 0.07529599964618683, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_136 0.0753 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      mm 0.0754 ms 99.9% 
      triton_mm_137 0.0754 ms 99.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_140 0.0755 ms 99.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_141 0.0770 ms 97.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_145 0.0801 ms 94.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_133 0.0839 ms 89.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_139 0.0920 ms 81.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_142 0.0941 ms 80.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_130 0.0946 ms 79.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3467 seconds and 0.1341 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04927999898791313, "best_triton_pos": 1, "best_triton_time": 0.05110400170087814, "best_triton_kernel": "triton_mm_150", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0493 ms 100.0% 
      triton_mm_150 0.0511 ms 96.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_154 0.0523 ms 94.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_158 0.0587 ms 84.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_162 0.0685 ms 71.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_153 0.0917 ms 53.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_149 0.0928 ms 53.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_147 0.0947 ms 52.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_157 0.0952 ms 51.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_148 0.0964 ms 51.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4712 seconds and 0.0002 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_175", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4", "best_time": 0.10012800246477127, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_175 0.1001 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_174 0.1006 ms 99.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_179 0.1012 ms 99.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_170 0.1014 ms 98.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_171 0.1021 ms 98.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.1029 ms 97.3% 
      triton_mm_167 0.1107 ms 90.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_172 0.1162 ms 86.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_176 0.1185 ms 84.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_173 0.1313 ms 76.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4088 seconds and 0.2863 seconds precompiling for 18 choices


    Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:37<00:00, 10.86s/it]Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:37<00:00,  9.38s/it]


    [2026-02-13 02:52:02] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 02:52:02] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.26s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.26s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.13 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.13 GB):  25%|██▌       | 1/4 [00:05<00:16,  5.63s/it]Capturing batches (bs=3 avail_mem=54.01 GB):  25%|██▌       | 1/4 [00:05<00:16,  5.63s/it]

    Capturing batches (bs=3 avail_mem=54.01 GB):  50%|█████     | 2/4 [00:06<00:05,  2.81s/it]Capturing batches (bs=2 avail_mem=53.99 GB):  50%|█████     | 2/4 [00:06<00:05,  2.81s/it]

    Capturing batches (bs=2 avail_mem=53.99 GB):  75%|███████▌  | 3/4 [00:06<00:01,  1.62s/it]Capturing batches (bs=1 avail_mem=53.95 GB):  75%|███████▌  | 3/4 [00:06<00:01,  1.62s/it]

    Capturing batches (bs=1 avail_mem=53.95 GB): 100%|██████████| 4/4 [00:11<00:00,  3.06s/it]Capturing batches (bs=1 avail_mem=53.95 GB): 100%|██████████| 4/4 [00:11<00:00,  2.98s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=53.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=53.90 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.64it/s]Capturing batches (bs=3 avail_mem=53.83 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.64it/s]Capturing batches (bs=2 avail_mem=53.83 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.64it/s]Capturing batches (bs=1 avail_mem=53.80 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.64it/s]Capturing batches (bs=1 avail_mem=53.80 GB): 100%|██████████| 4/4 [00:00<00:00, 14.50it/s]Capturing batches (bs=1 avail_mem=53.80 GB): 100%|██████████| 4/4 [00:00<00:00, 12.50it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='a67f415b84f94ba4a96eab9900fde08e', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770951145, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 02:52:30] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:52:30] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:52:30] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:52:33] WARNING model_config.py:1158: Casting torch.bfloat16 to torch.float16.
    [2026-02-13 02:52:33] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 02:52:33] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 02:52:33] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:52:33] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 02:52:39] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:52:39] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:52:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:52:40] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:52:40] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:52:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:52:41] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 02:52:42] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:52:45] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:52:45] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:52:45] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.40s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.47s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:13<00:04,  4.44s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.17s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.64s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.19 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.19 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.05s/it]Capturing batches (bs=3 avail_mem=60.02 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.05s/it]Capturing batches (bs=2 avail_mem=60.01 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.05s/it]Capturing batches (bs=2 avail_mem=60.01 GB):  75%|███████▌  | 3/4 [00:01<00:00,  3.15it/s]Capturing batches (bs=1 avail_mem=59.98 GB):  75%|███████▌  | 3/4 [00:01<00:00,  3.15it/s]Capturing batches (bs=1 avail_mem=59.98 GB): 100%|██████████| 4/4 [00:01<00:00,  3.27it/s]


    [2026-02-13 02:53:03] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 02:53:03] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-13 02:53:03] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-13 02:53:03] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.02 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.02 GB):  25%|██▌       | 1/4 [00:03<00:11,  3.69s/it]Capturing batches (bs=3 avail_mem=58.90 GB):  25%|██▌       | 1/4 [00:03<00:11,  3.69s/it]

    Capturing batches (bs=3 avail_mem=58.90 GB):  50%|█████     | 2/4 [00:04<00:03,  1.91s/it]Capturing batches (bs=2 avail_mem=58.88 GB):  50%|█████     | 2/4 [00:04<00:03,  1.91s/it]

    Capturing batches (bs=2 avail_mem=58.88 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.14s/it]Capturing batches (bs=1 avail_mem=58.84 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.14s/it]

    Capturing batches (bs=1 avail_mem=58.84 GB): 100%|██████████| 4/4 [00:06<00:00,  1.60s/it]Capturing batches (bs=1 avail_mem=58.84 GB): 100%|██████████| 4/4 [00:06<00:00,  1.72s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.79 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.70 GB): 100%|██████████| 4/4 [00:00<00:00, 92.56it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='28a85c86a1364f56893d73c06fc75e73', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770951201, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 02:53:26] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:53:26] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:53:26] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:53:29] WARNING model_config.py:1158: Casting torch.bfloat16 to torch.float16.
    [2026-02-13 02:53:29] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 02:53:29] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 02:53:29] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:53:29] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 02:53:35] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:53:35] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:53:35] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:53:35] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:53:35] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:53:35] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:53:37] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 02:53:38] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:53:41] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:53:41] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:53:41] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:12,  4.22s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.23s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.23s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.02s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.46s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.07 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.07 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=3 avail_mem=59.95 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=2 avail_mem=59.94 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=1 avail_mem=59.92 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.23it/s]Capturing batches (bs=1 avail_mem=59.92 GB): 100%|██████████| 4/4 [00:00<00:00,  5.15it/s]Capturing batches (bs=1 avail_mem=59.92 GB): 100%|██████████| 4/4 [00:00<00:00,  4.16it/s]


    [2026-02-13 02:53:58] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 02:53:58] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-13 02:53:58] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-13 02:53:58] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.76 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.76 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.24s/it]Capturing batches (bs=3 avail_mem=58.71 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.24s/it]

    Capturing batches (bs=3 avail_mem=58.71 GB):  50%|█████     | 2/4 [00:03<00:03,  1.59s/it]Capturing batches (bs=2 avail_mem=58.67 GB):  50%|█████     | 2/4 [00:03<00:03,  1.59s/it]Capturing batches (bs=2 avail_mem=58.67 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.07it/s]Capturing batches (bs=1 avail_mem=58.63 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.07it/s]

    Capturing batches (bs=1 avail_mem=58.63 GB): 100%|██████████| 4/4 [00:05<00:00,  1.39s/it]Capturing batches (bs=1 avail_mem=58.63 GB): 100%|██████████| 4/4 [00:05<00:00,  1.48s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.57 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.48 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 4/4 [00:00<00:00, 108.34it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='78629acd4c984a07a208a6854c945455', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770951252, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 02:54:18] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:54:18] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:54:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:54:20] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 02:54:20] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 02:54:20] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:54:26] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:54:26] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:54:26] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:54:26] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:54:26] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:54:26] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:54:32] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:54:32] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:54:32] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.53it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.38it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.37it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.47it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.44it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.12 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.12 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.49it/s]Capturing batches (bs=3 avail_mem=61.06 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.49it/s]Capturing batches (bs=2 avail_mem=61.05 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.49it/s]Capturing batches (bs=1 avail_mem=61.04 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.49it/s]Capturing batches (bs=1 avail_mem=61.04 GB): 100%|██████████| 4/4 [00:00<00:00,  6.29it/s]Capturing batches (bs=1 avail_mem=61.04 GB): 100%|██████████| 4/4 [00:00<00:00,  5.07it/s]


    [2026-02-13 02:54:38] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 02:54:38] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  4.34it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  7.27it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  6.92it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.56 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=60.50 GB): 100%|██████████| 4/4 [00:00<00:00, 43.30it/s]



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


<strong style='color: #00008B;'>{'id': '8bb5c05d09ed46bcbea1440613884adc', 'object': 'chat.completion', 'created': 1770951288, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "<think>\nOkay, so the question is asking for the capital of France. Let me think. I remember that France is a country in Europe. The capital cities often have famous landmarks. Paris comes to mind immediately because of the Eiffel Tower. But wait, maybe I should verify that. Sometimes people confuse capitals with big cities. For example, maybe London is the capital of the UK, but that's not France. Let me think of other European capitals. Berlin is Germany's capital, Rome is Italy's. So yes, Paris should be France's. The official name of the city is Paris, and it's located in the Île-de-France region. There's also the Louvre Museum, which is in Paris. The French government and many companies are based there. I don't think there's any other major city that's the capital. Maybe some sources might say Lyon was a historical capital, but I believe that's not the case anymore. The current capital is definitely Paris. Let me confirm with some quick facts. Yes, all signs point to Paris being the capital. So the answer should be Paris.\n</think>\nThe capital of France is **Paris**. This vibrant city is known for iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Champs-Élysées. It serves as the country's political, cultural, and economic center.", 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 310, 'completion_tokens': 284, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



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

    [2026-02-13 02:54:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:54:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:54:53] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:54:56] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 02:54:56] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 02:54:56] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:55:02] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:55:02] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:55:02] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:55:02] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:55:02] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:55:02] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:55:07] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:55:07] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:55:07] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.44it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.35it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.33it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.31it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.33it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.25 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.25 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.24s/it]Capturing batches (bs=3 avail_mem=62.17 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.24s/it]Capturing batches (bs=2 avail_mem=62.17 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.24s/it]Capturing batches (bs=2 avail_mem=62.17 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.65it/s]Capturing batches (bs=1 avail_mem=62.14 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.65it/s]

    Capturing batches (bs=1 avail_mem=62.14 GB): 100%|██████████| 4/4 [00:01<00:00,  2.77it/s]


    [2026-02-13 02:55:14] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 02:55:14] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.58it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.58it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.33 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.33 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.79s/it]Capturing batches (bs=3 avail_mem=58.21 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.79s/it]

    Capturing batches (bs=3 avail_mem=58.21 GB):  50%|█████     | 2/4 [00:05<00:04,  2.36s/it]Capturing batches (bs=2 avail_mem=58.20 GB):  50%|█████     | 2/4 [00:05<00:04,  2.36s/it]

    Capturing batches (bs=2 avail_mem=58.20 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.39s/it]Capturing batches (bs=1 avail_mem=58.15 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.39s/it]

    Capturing batches (bs=1 avail_mem=58.15 GB): 100%|██████████| 4/4 [00:09<00:00,  2.20s/it]Capturing batches (bs=1 avail_mem=58.15 GB): 100%|██████████| 4/4 [00:09<00:00,  2.29s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.11 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.03 GB):  75%|███████▌  | 3/4 [00:00<00:00, 22.83it/s]Capturing batches (bs=1 avail_mem=58.01 GB):  75%|███████▌  | 3/4 [00:00<00:00, 22.83it/s]Capturing batches (bs=1 avail_mem=58.01 GB): 100%|██████████| 4/4 [00:00<00:00, 21.24it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='23627b9d9ef74948b1175573361c8ae7', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770951334, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 02:55:39] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:55:39] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:55:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:55:41] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 02:55:41] WARNING server_args.py:2309: Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on.
    [2026-02-13 02:55:41] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:55:47] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:55:47] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:55:47] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:55:47] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:55:47] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:55:47] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:55:53] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:55:53] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:55:53] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.38it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.26it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.20it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.24it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.25it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.80 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.99s/it]Capturing batches (bs=3 avail_mem=62.74 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.99s/it]Capturing batches (bs=2 avail_mem=62.73 GB):  25%|██▌       | 1/4 [00:02<00:05,  1.99s/it]Capturing batches (bs=2 avail_mem=62.73 GB):  75%|███████▌  | 3/4 [00:02<00:00,  1.77it/s]Capturing batches (bs=1 avail_mem=62.73 GB):  75%|███████▌  | 3/4 [00:02<00:00,  1.77it/s]Capturing batches (bs=1 avail_mem=62.73 GB): 100%|██████████| 4/4 [00:02<00:00,  1.85it/s]


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.74it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.74it/s]
    


    [2026-02-13 02:56:02] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.04 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.04 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.44s/it]Capturing batches (bs=3 avail_mem=58.96 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.44s/it]

    Capturing batches (bs=3 avail_mem=58.96 GB):  50%|█████     | 2/4 [00:02<00:02,  1.25s/it]Capturing batches (bs=2 avail_mem=58.95 GB):  50%|█████     | 2/4 [00:02<00:02,  1.25s/it]Capturing batches (bs=1 avail_mem=58.93 GB):  50%|█████     | 2/4 [00:02<00:02,  1.25s/it]

    Capturing batches (bs=1 avail_mem=58.93 GB): 100%|██████████| 4/4 [00:03<00:00,  1.32it/s]Capturing batches (bs=1 avail_mem=58.93 GB): 100%|██████████| 4/4 [00:03<00:00,  1.05it/s]


    [2026-02-13 02:56:08] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='d4861f0ebdcc4b198a82f4ffda61a436', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770951373, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 02:56:18] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:56:18] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:56:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 02:56:21] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 02:56:21] WARNING server_args.py:2415: The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding.
    [2026-02-13 02:56:21] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 02:56:27] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:56:27] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:56:27] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 02:56:27] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 02:56:27] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 02:56:27] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 02:56:32] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:56:32] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 02:56:32] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.50it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.33it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.32it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.35it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.36it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.77 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.77 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.31it/s]Capturing batches (bs=3 avail_mem=62.70 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.31it/s]Capturing batches (bs=2 avail_mem=62.69 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.31it/s]Capturing batches (bs=2 avail_mem=62.69 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.05it/s]Capturing batches (bs=1 avail_mem=62.68 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.05it/s]

    Capturing batches (bs=1 avail_mem=62.68 GB): 100%|██████████| 4/4 [00:01<00:00,  3.91it/s]Capturing batches (bs=1 avail_mem=62.68 GB): 100%|██████████| 4/4 [00:01<00:00,  3.43it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='bad37b5e7d0d434da90ec707f30c008d', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770951415, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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
