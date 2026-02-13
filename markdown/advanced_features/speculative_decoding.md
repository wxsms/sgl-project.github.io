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

    [2026-02-13 12:19:35] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-13 12:19:35] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-13 12:19:35] INFO utils.py:164: NumExpr defaulting to 16 threads.



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

    [2026-02-13 12:19:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:19:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:19:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:19:41] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 12:19:41] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 12:19:41] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:19:46] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:19:46] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:19:46] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:19:46] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:19:46] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:19:46] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:19:51] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:19:51] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:19:51] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.81it/s]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.24s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.13s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=23.12 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=23.12 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.18it/s]Capturing batches (bs=3 avail_mem=19.72 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.18it/s]Capturing batches (bs=2 avail_mem=19.72 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.18it/s]

    Capturing batches (bs=1 avail_mem=19.70 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.18it/s]Capturing batches (bs=1 avail_mem=19.70 GB): 100%|██████████| 4/4 [00:00<00:00, 13.79it/s]Capturing batches (bs=1 avail_mem=19.70 GB): 100%|██████████| 4/4 [00:00<00:00, 12.61it/s]


    [2026-02-13 12:19:56] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 12:19:56] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.03s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.03s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=18.75 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=18.75 GB):  25%|██▌       | 1/4 [00:03<00:11,  3.83s/it]Capturing batches (bs=3 avail_mem=18.62 GB):  25%|██▌       | 1/4 [00:03<00:11,  3.83s/it]

    Capturing batches (bs=3 avail_mem=18.62 GB):  50%|█████     | 2/4 [00:04<00:03,  1.89s/it]Capturing batches (bs=2 avail_mem=18.62 GB):  50%|█████     | 2/4 [00:04<00:03,  1.89s/it]

    Capturing batches (bs=2 avail_mem=18.62 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.15s/it]Capturing batches (bs=1 avail_mem=18.60 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.15s/it]

    Capturing batches (bs=1 avail_mem=18.60 GB): 100%|██████████| 4/4 [00:07<00:00,  1.88s/it]Capturing batches (bs=1 avail_mem=18.60 GB): 100%|██████████| 4/4 [00:07<00:00,  1.91s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=37.65 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=37.65 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=37.63 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=37.63 GB): 100%|██████████| 4/4 [00:00<00:00, 32.32it/s]Capturing batches (bs=1 avail_mem=37.63 GB): 100%|██████████| 4/4 [00:00<00:00, 32.28it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='7baa49b075a94a0186676244f8f1774b', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770985213, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 12:20:17] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:20:17] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:20:17] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:20:19] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 12:20:19] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 12:20:19] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:20:23] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:20:23] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:20:23] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:20:23] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:20:23] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:20:23] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:20:28] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:20:28] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:20:28] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.85it/s]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.22s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.12s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=35.97 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=35.97 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.75it/s]Capturing batches (bs=3 avail_mem=35.88 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.75it/s]Capturing batches (bs=2 avail_mem=35.87 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.75it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.04851200059056282, "best_triton_pos": 1, "best_triton_time": 0.05027199909090996, "best_triton_kernel": "triton_mm_18", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(128x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0485 ms 100.0% 
      triton_mm_18 0.0503 ms 96.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_12 0.0530 ms 91.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_8 0.0548 ms 88.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_7 0.0563 ms 86.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_11 0.0564 ms 86.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_17 0.0585 ms 82.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_10 0.0682 ms 71.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_14 0.0699 ms 69.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_4 0.0713 ms 68.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3405 seconds and 0.2645 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.022143999114632607, "best_triton_pos": 1, "best_triton_time": 0.023104000836610794, "best_triton_kernel": "triton_mm_27", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0221 ms 100.0% 
      triton_mm_27 0.0231 ms 95.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_31 0.0263 ms 84.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_23 0.0299 ms 74.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_37 0.0313 ms 70.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_26 0.0395 ms 56.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_30 0.0410 ms 54.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_22 0.0419 ms 52.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_20 0.0421 ms 52.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_36 0.0424 ms 52.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.2763 seconds and 0.3134 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_49", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.07468800246715546, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_49 0.0747 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.0762 ms 98.1% 
      triton_mm_55 0.0769 ms 97.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_50 0.0796 ms 93.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_56 0.0829 ms 90.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_45 0.0929 ms 80.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_46 0.0963 ms 77.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_47 0.0977 ms 76.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_48 0.1014 ms 73.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_54 0.1014 ms 73.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4140 seconds and 0.1532 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.04774399846792221, "best_triton_pos": 1, "best_triton_time": 0.050655998289585114, "best_triton_kernel": "triton_mm_65", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0477 ms 100.0% 
      triton_mm_65 0.0507 ms 94.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_69 0.0561 ms 85.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_61 0.0630 ms 75.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_75 0.0693 ms 68.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_64 0.0898 ms 53.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_68 0.0937 ms 51.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_74 0.0967 ms 49.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_60 0.0973 ms 49.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_58 0.1002 ms 47.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4203 seconds and 0.0002 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_93", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.10339199751615524, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_93 0.1034 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_94 0.1049 ms 98.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      mm 0.1067 ms 96.9% 
      triton_mm_88 0.1084 ms 95.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_87 0.1159 ms 89.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_83 0.1173 ms 88.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_84 0.1293 ms 80.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_92 0.1297 ms 79.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_85 0.1371 ms 75.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_89 0.1463 ms 70.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.5051 seconds and 0.2865 seconds precompiling for 20 choices


    Capturing batches (bs=2 avail_mem=35.87 GB):  75%|███████▌  | 3/4 [00:17<00:06,  6.53s/it]Capturing batches (bs=1 avail_mem=38.77 GB):  75%|███████▌  | 3/4 [00:17<00:06,  6.53s/it]

    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04739199951291084, "best_triton_pos": 1, "best_triton_time": 0.04864000156521797, "best_triton_kernel": "triton_mm_111", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(64x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0474 ms 100.0% 
      triton_mm_111 0.0486 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_107 0.0488 ms 97.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_103 0.0492 ms 96.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_99 0.0507 ms 93.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_106 0.0535 ms 88.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_102 0.0552 ms 85.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_96 0.0567 ms 83.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_98 0.0615 ms 77.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_97 0.0647 ms 73.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.3126 seconds and 0.2071 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.021536000072956085, "best_triton_pos": 1, "best_triton_time": 0.02252800017595291, "best_triton_kernel": "triton_mm_116", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0215 ms 100.0% 
      triton_mm_116 0.0225 ms 95.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_120 0.0238 ms 90.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_124 0.0266 ms 80.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_128 0.0301 ms 71.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_123 0.0355 ms 60.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_119 0.0389 ms 55.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_113 0.0395 ms 54.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_115 0.0410 ms 52.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_114 0.0424 ms 50.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.3618 seconds and 0.2445 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_137", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4", "best_time": 0.07491199672222137, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_137 0.0749 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_136 0.0757 ms 98.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_140 0.0757 ms 98.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_141 0.0764 ms 98.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      mm 0.0765 ms 97.9% 
      triton_mm_145 0.0802 ms 93.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_133 0.0836 ms 89.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_139 0.0921 ms 81.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_130 0.0937 ms 80.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_142 0.0948 ms 79.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3748 seconds and 0.1638 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04806400090456009, "best_triton_pos": 1, "best_triton_time": 0.049056001007556915, "best_triton_kernel": "triton_mm_150", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0481 ms 100.0% 
      triton_mm_150 0.0491 ms 98.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_154 0.0506 ms 95.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_158 0.0569 ms 84.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_162 0.0664 ms 72.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_153 0.0899 ms 53.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_149 0.0909 ms 52.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_147 0.0938 ms 51.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_157 0.0941 ms 51.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_148 0.0941 ms 51.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.3852 seconds and 0.0004 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_174", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.0992640033364296, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_174 0.0993 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_170 0.1003 ms 99.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_175 0.1006 ms 98.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_179 0.1006 ms 98.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_171 0.1014 ms 97.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.1029 ms 96.5% 
      triton_mm_167 0.1112 ms 89.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_172 0.1175 ms 84.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_176 0.1195 ms 83.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_173 0.1319 ms 75.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4769 seconds and 0.2458 seconds precompiling for 18 choices


    Capturing batches (bs=1 avail_mem=38.77 GB): 100%|██████████| 4/4 [00:33<00:00,  9.80s/it]Capturing batches (bs=1 avail_mem=38.77 GB): 100%|██████████| 4/4 [00:33<00:00,  8.42s/it]


    [2026-02-13 12:21:06] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 12:21:06] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.07s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.07s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.22 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.22 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.04s/it]Capturing batches (bs=3 avail_mem=54.10 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.04s/it]

    Capturing batches (bs=3 avail_mem=54.10 GB):  50%|█████     | 2/4 [00:05<00:04,  2.47s/it]Capturing batches (bs=2 avail_mem=53.41 GB):  50%|█████     | 2/4 [00:05<00:04,  2.47s/it]Capturing batches (bs=2 avail_mem=53.41 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.43s/it]Capturing batches (bs=1 avail_mem=53.38 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.43s/it]

    Capturing batches (bs=1 avail_mem=53.38 GB): 100%|██████████| 4/4 [00:09<00:00,  2.31s/it]Capturing batches (bs=1 avail_mem=53.38 GB): 100%|██████████| 4/4 [00:09<00:00,  2.39s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=19.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=19.76 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=19.76 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=19.73 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=19.73 GB): 100%|██████████| 4/4 [00:00<00:00, 80.24it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='81fa6efa0ee4462baaf6efe747f6ea96', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770985285, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 12:21:29] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:21:29] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:21:29] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:21:31] WARNING model_config.py:1178: Casting torch.bfloat16 to torch.float16.
    [2026-02-13 12:21:31] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 12:21:31] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 12:21:31] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:21:32] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 12:21:35] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:21:35] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:21:35] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:21:35] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:21:35] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:21:35] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:21:37] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 12:21:38] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:21:40] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:21:40] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:21:40] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:03<00:11,  3.83s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:07<00:07,  3.83s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:08<00:02,  2.55s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:12<00:00,  3.09s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:12<00:00,  3.15s/it]
    
    [2026-02-13 12:21:53] max_total_tokens=20480 is larger than the profiled value 20411. Use the profiled value instead.


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=22.89 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=22.89 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.47it/s]Capturing batches (bs=3 avail_mem=22.74 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.47it/s]Capturing batches (bs=2 avail_mem=22.74 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.47it/s]

    Capturing batches (bs=1 avail_mem=22.71 GB):  25%|██▌       | 1/4 [00:00<00:00,  6.47it/s]Capturing batches (bs=1 avail_mem=22.71 GB): 100%|██████████| 4/4 [00:00<00:00, 15.38it/s]Capturing batches (bs=1 avail_mem=22.71 GB): 100%|██████████| 4/4 [00:00<00:00, 13.93it/s]


    [2026-02-13 12:21:55] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 12:21:55] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-13 12:21:55] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-13 12:21:55] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=21.69 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=21.69 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.96s/it]Capturing batches (bs=3 avail_mem=20.93 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.96s/it]

    Capturing batches (bs=3 avail_mem=20.93 GB):  50%|█████     | 2/4 [00:03<00:03,  1.50s/it]Capturing batches (bs=2 avail_mem=20.91 GB):  50%|█████     | 2/4 [00:03<00:03,  1.50s/it]Capturing batches (bs=2 avail_mem=20.91 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.11it/s]Capturing batches (bs=1 avail_mem=20.87 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.11it/s]

    Capturing batches (bs=1 avail_mem=20.87 GB): 100%|██████████| 4/4 [00:05<00:00,  1.30s/it]Capturing batches (bs=1 avail_mem=20.87 GB): 100%|██████████| 4/4 [00:05<00:00,  1.38s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=20.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=20.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=20.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=20.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=20.70 GB): 100%|██████████| 4/4 [00:00<00:00, 36.58it/s]Capturing batches (bs=1 avail_mem=20.70 GB): 100%|██████████| 4/4 [00:00<00:00, 36.52it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='105b415c0e6649f387acf5f68febf8c1', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770985332, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 12:22:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:22:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:22:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:22:17] WARNING model_config.py:1178: Casting torch.bfloat16 to torch.float16.
    [2026-02-13 12:22:17] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 12:22:17] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 12:22:17] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:22:18] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 12:22:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:22:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:22:21] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:22:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:22:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:22:21] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:22:23] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 12:22:23] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:22:26] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:22:26] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:22:26] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:03<00:11,  3.71s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:07<00:07,  3.77s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:08<00:02,  2.48s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:12<00:00,  2.96s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:12<00:00,  3.04s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=42.98 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=42.98 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.03it/s]Capturing batches (bs=3 avail_mem=42.87 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.03it/s]

    Capturing batches (bs=2 avail_mem=42.86 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.03it/s]Capturing batches (bs=2 avail_mem=42.86 GB):  75%|███████▌  | 3/4 [00:00<00:00,  9.52it/s]Capturing batches (bs=1 avail_mem=42.84 GB):  75%|███████▌  | 3/4 [00:00<00:00,  9.52it/s]Capturing batches (bs=1 avail_mem=42.84 GB): 100%|██████████| 4/4 [00:00<00:00,  9.66it/s]


    [2026-02-13 12:22:40] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 12:22:40] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-13 12:22:40] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-13 12:22:40] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.96it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.95it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.18 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.18 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.15s/it]Capturing batches (bs=3 avail_mem=43.79 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.15s/it]

    Capturing batches (bs=3 avail_mem=43.79 GB):  50%|█████     | 2/4 [00:02<00:02,  1.05s/it]Capturing batches (bs=2 avail_mem=43.75 GB):  50%|█████     | 2/4 [00:02<00:02,  1.05s/it]Capturing batches (bs=1 avail_mem=43.71 GB):  50%|█████     | 2/4 [00:02<00:02,  1.05s/it]

    Capturing batches (bs=1 avail_mem=43.71 GB): 100%|██████████| 4/4 [00:04<00:00,  1.12it/s]Capturing batches (bs=1 avail_mem=43.71 GB): 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=42.36 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=42.29 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=42.28 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=42.26 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=42.26 GB): 100%|██████████| 4/4 [00:00<00:00, 115.46it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='0df2f345aa7c4942a852418067ceec38', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770985374, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 12:22:58] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:22:58] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:22:58] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:22:59] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 12:22:59] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 12:22:59] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:23:04] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:23:04] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:23:04] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:23:04] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:23:04] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:23:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:23:09] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:23:09] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:23:09] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.38it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.38it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.56it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.45it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.45it/s]
    
    [2026-02-13 12:23:12] max_total_tokens=20480 is larger than the profiled value 1137. Use the profiled value instead.


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=30.16 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=30.16 GB):  25%|██▌       | 1/4 [00:00<00:00,  3.44it/s]Capturing batches (bs=3 avail_mem=30.07 GB):  25%|██▌       | 1/4 [00:00<00:00,  3.44it/s]Capturing batches (bs=2 avail_mem=30.07 GB):  25%|██▌       | 1/4 [00:00<00:00,  3.44it/s]Capturing batches (bs=2 avail_mem=30.07 GB):  75%|███████▌  | 3/4 [00:00<00:00,  8.05it/s]Capturing batches (bs=1 avail_mem=30.10 GB):  75%|███████▌  | 3/4 [00:00<00:00,  8.05it/s]Capturing batches (bs=1 avail_mem=30.10 GB): 100%|██████████| 4/4 [00:00<00:00,  8.24it/s]


    [2026-02-13 12:23:14] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 12:23:14] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:00<00:00, 11.47it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  8.38it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  8.73it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=29.68 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=29.63 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=29.63 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=29.63 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=29.63 GB): 100%|██████████| 4/4 [00:00<00:00, 38.27it/s]Capturing batches (bs=1 avail_mem=29.63 GB): 100%|██████████| 4/4 [00:00<00:00, 38.20it/s]



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


<strong style='color: #00008B;'>{'id': '86f0f553c9d84b2da48cd776f4c65f12', 'object': 'chat.completion', 'created': 1770985410, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<think>\nOkay, so the user is asking, "What is the capital of France?" Let me start by recalling what I know about France\'s geography and politics. France is a country in Europe, right? I remember that Paris is a major city there. But wait, is Paris the actual capital?\n\nLet me think. The capital of a country is usually the seat of government, where the parliament, government buildings, and other political institutions are located. For France, I\'m pretty sure that\'s Paris. But maybe I should verify that. Sometimes people confuse the largest city with the capital, but in France, both the largest city and the capital are Paris. The Louvre Palace is there, and the French president\'s residence is the Elysée Palace. The national assembly is also in Paris. So yes, that makes sense.\n\nWait, could there be any confusion with other regions? For example, some countries have capitals different from their largest city. But in France, Paris is both the largest city and the capital. I don\'t think there\'s another city that serves as the capital. Maybe some regions have their own capitals, like Brittany\'s capital is Nantes, but that\'s not the national capital.\n\nAlso, considering the language, in French, the capital is called "Paris," which is spelled the same in French and English. So the answer should be straightforward. Unless there\'s some historical change I\'m not aware of. Did France ever have a different capital in the past? For instance, during the revolution, maybe they moved it? No, I think Paris remained the capital throughout the different periods of French history.\n\nAnother angle: sometimes countries have capitals that aren\'t the largest city, but in France\'s case, Paris is both. So the answer is definitely Paris. I can\'t think of any reason to doubt that. The user might be testing basic knowledge, or maybe they have a trick question, but no, the capital of France is Paris. Simple enough.\n</think>\n\nThe capital of France is **Paris**. This metropolis is not only the nation\'s largest city but also the seat of the French government, housing prominent institutions like the Elysée Palace (residence of the president) and the Louvre Palace (home to the national assembly). Its historical and cultural significance solidifies its role as the country\'s political and cultural heart.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 505, 'completion_tokens': 479, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



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

    [2026-02-13 12:23:34] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:23:34] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:23:34] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:23:36] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 12:23:36] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 12:23:36] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:23:40] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:23:40] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:23:40] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:23:40] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:23:40] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:23:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:23:45] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:23:45] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:23:45] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.88it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.64it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.70it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.65it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.67it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.33 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.33 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.70it/s]Capturing batches (bs=3 avail_mem=62.25 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.70it/s]Capturing batches (bs=2 avail_mem=62.25 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.70it/s]Capturing batches (bs=2 avail_mem=62.25 GB):  75%|███████▌  | 3/4 [00:00<00:00, 10.49it/s]Capturing batches (bs=1 avail_mem=62.23 GB):  75%|███████▌  | 3/4 [00:00<00:00, 10.49it/s]Capturing batches (bs=1 avail_mem=62.23 GB): 100%|██████████| 4/4 [00:00<00:00, 11.28it/s]


    [2026-02-13 12:23:50] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 12:23:50] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.10it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.10it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=57.75 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=57.75 GB):  25%|██▌       | 1/4 [00:03<00:10,  3.55s/it]Capturing batches (bs=3 avail_mem=12.83 GB):  25%|██▌       | 1/4 [00:03<00:10,  3.55s/it]

    Capturing batches (bs=3 avail_mem=12.83 GB):  50%|█████     | 2/4 [00:04<00:03,  1.78s/it]Capturing batches (bs=2 avail_mem=12.85 GB):  50%|█████     | 2/4 [00:04<00:03,  1.78s/it]Capturing batches (bs=2 avail_mem=12.85 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.04s/it]Capturing batches (bs=1 avail_mem=12.82 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.04s/it]

    Capturing batches (bs=1 avail_mem=12.82 GB): 100%|██████████| 4/4 [00:06<00:00,  1.70s/it]Capturing batches (bs=1 avail_mem=12.82 GB): 100%|██████████| 4/4 [00:06<00:00,  1.74s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=8.28 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=8.10 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=8.10 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=8.10 GB):  75%|███████▌  | 3/4 [00:00<00:00, 29.23it/s]Capturing batches (bs=1 avail_mem=8.08 GB):  75%|███████▌  | 3/4 [00:00<00:00, 29.23it/s]Capturing batches (bs=1 avail_mem=8.08 GB): 100%|██████████| 4/4 [00:00<00:00, 27.52it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='10852eec9d644e7db58df25ac34140c9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770985446, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 12:24:10] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:24:10] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:24:10] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:24:11] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 12:24:11] WARNING server_args.py:2309: Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on.
    [2026-02-13 12:24:11] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:24:16] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:24:16] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:24:16] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:24:16] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:24:16] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:24:16] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:24:20] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:24:20] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:24:20] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.83it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.66it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.76it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.69it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.71it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.87 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.87 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.15it/s]Capturing batches (bs=3 avail_mem=62.82 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.15it/s]

    Capturing batches (bs=2 avail_mem=62.82 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.15it/s]Capturing batches (bs=1 avail_mem=62.82 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.15it/s]Capturing batches (bs=1 avail_mem=62.82 GB): 100%|██████████| 4/4 [00:00<00:00, 14.22it/s]Capturing batches (bs=1 avail_mem=62.82 GB): 100%|██████████| 4/4 [00:00<00:00, 12.56it/s]


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.21it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.21it/s]
    


    [2026-02-13 12:24:25] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.14 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.14 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.74s/it]Capturing batches (bs=3 avail_mem=58.39 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.74s/it]

    Capturing batches (bs=3 avail_mem=58.39 GB):  50%|█████     | 2/4 [00:02<00:01,  1.12it/s]Capturing batches (bs=2 avail_mem=58.39 GB):  50%|█████     | 2/4 [00:02<00:01,  1.12it/s]Capturing batches (bs=1 avail_mem=58.38 GB):  50%|█████     | 2/4 [00:02<00:01,  1.12it/s]

    Capturing batches (bs=1 avail_mem=58.38 GB): 100%|██████████| 4/4 [00:02<00:00,  1.78it/s]Capturing batches (bs=1 avail_mem=58.38 GB): 100%|██████████| 4/4 [00:02<00:00,  1.44it/s]


    [2026-02-13 12:24:30] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='7ba8c199a5da42c18e5477ca945924ea', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770985475, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 12:24:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:24:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:24:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 12:24:41] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 12:24:41] WARNING server_args.py:2416: The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding.
    [2026-02-13 12:24:41] INFO server_args.py:2822: Set soft_watchdog_timeout since in CI


    [2026-02-13 12:24:45] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:24:45] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:24:45] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 12:24:45] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 12:24:45] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 12:24:45] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 12:24:49] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:24:49] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 12:24:49] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.95it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.58it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.65it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.63it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.64it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.85 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.85 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.04it/s]Capturing batches (bs=3 avail_mem=62.78 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.04it/s]

    Capturing batches (bs=2 avail_mem=62.78 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.04it/s]Capturing batches (bs=1 avail_mem=62.78 GB):  25%|██▌       | 1/4 [00:00<00:00,  5.04it/s]Capturing batches (bs=1 avail_mem=62.78 GB): 100%|██████████| 4/4 [00:00<00:00, 11.24it/s]Capturing batches (bs=1 avail_mem=62.78 GB): 100%|██████████| 4/4 [00:00<00:00, 10.28it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='40d01bbec4bb4c8eaec9f69c05101df6', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770985501, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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
