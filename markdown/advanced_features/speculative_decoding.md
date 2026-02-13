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

    [2026-02-13 00:27:34] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-13 00:27:34] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-13 00:27:34] INFO utils.py:164: NumExpr defaulting to 16 threads.



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

    [2026-02-13 00:27:41] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:27:41] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:27:41] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:27:44] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 00:27:44] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 00:27:44] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:27:55] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:27:55] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:27:55] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:27:55] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:27:55] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:27:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:28:01] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:28:01] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:28:01] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.44s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.02it/s]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.04s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=24.52 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=24.52 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=3 avail_mem=23.07 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=2 avail_mem=23.00 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=1 avail_mem=22.98 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s]Capturing batches (bs=1 avail_mem=22.98 GB): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]Capturing batches (bs=1 avail_mem=22.98 GB): 100%|██████████| 4/4 [00:00<00:00,  4.51it/s]


    [2026-02-13 00:28:06] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 00:28:06] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.08s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.08s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=18.67 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=18.67 GB):  25%|██▌       | 1/4 [00:04<00:13,  4.57s/it]Capturing batches (bs=3 avail_mem=18.51 GB):  25%|██▌       | 1/4 [00:04<00:13,  4.57s/it]

    Capturing batches (bs=3 avail_mem=18.51 GB):  50%|█████     | 2/4 [00:05<00:04,  2.26s/it]Capturing batches (bs=2 avail_mem=18.51 GB):  50%|█████     | 2/4 [00:05<00:04,  2.26s/it]

    Capturing batches (bs=2 avail_mem=18.51 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.36s/it]Capturing batches (bs=1 avail_mem=18.49 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.36s/it]

    Capturing batches (bs=1 avail_mem=18.49 GB): 100%|██████████| 4/4 [00:08<00:00,  2.16s/it]Capturing batches (bs=1 avail_mem=18.49 GB): 100%|██████████| 4/4 [00:08<00:00,  2.22s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.09 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=54.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=54.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=54.01 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=54.01 GB): 100%|██████████| 4/4 [00:00<00:00, 61.93it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='c1fff77ada2c41869abb10b0dff5bc71', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770942505, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 00:28:31] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:28:31] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:28:31] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:28:33] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 00:28:33] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 00:28:33] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:28:40] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:28:40] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:28:40] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:28:40] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:28:40] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:28:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:28:45] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:28:45] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:28:45] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.37s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.06it/s]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.01s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=35.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=35.90 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.67s/it]Capturing batches (bs=3 avail_mem=55.10 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.67s/it]Capturing batches (bs=2 avail_mem=55.08 GB):  25%|██▌       | 1/4 [00:01<00:05,  1.67s/it]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.04825599864125252, "best_triton_pos": 1, "best_triton_time": 0.04947200044989586, "best_triton_kernel": "triton_mm_18", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8"}
    AUTOTUNE mm(128x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0483 ms 100.0% 
      triton_mm_18 0.0495 ms 97.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_12 0.0531 ms 91.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_8 0.0550 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_7 0.0556 ms 86.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_11 0.0559 ms 86.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_17 0.0580 ms 83.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_10 0.0675 ms 71.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_14 0.0688 ms 70.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_4 0.0707 ms 68.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3248 seconds and 0.3333 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.022175999358296394, "best_triton_pos": 1, "best_triton_time": 0.023423999547958374, "best_triton_kernel": "triton_mm_27", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0222 ms 100.0% 
      triton_mm_27 0.0234 ms 94.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_31 0.0266 ms 83.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_23 0.0302 ms 73.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_37 0.0314 ms 70.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_26 0.0396 ms 56.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_30 0.0412 ms 53.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_22 0.0421 ms 52.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_20 0.0422 ms 52.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_36 0.0427 ms 52.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.2859 seconds and 0.3913 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_49", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.07487999647855759, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_49 0.0749 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.0758 ms 98.7% 
      triton_mm_55 0.0770 ms 97.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_50 0.0799 ms 93.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_56 0.0830 ms 90.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_45 0.0937 ms 79.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_46 0.0961 ms 77.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_47 0.0974 ms 76.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_48 0.1005 ms 74.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_54 0.1016 ms 73.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4842 seconds and 0.1759 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "mm", "best_time": 0.04873599857091904, "best_triton_pos": 1, "best_triton_time": 0.05161599814891815, "best_triton_kernel": "triton_mm_65", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(128x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0487 ms 100.0% 
      triton_mm_65 0.0516 ms 94.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_69 0.0567 ms 86.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_61 0.0660 ms 73.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_75 0.0705 ms 69.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_64 0.0900 ms 54.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_68 0.0933 ms 52.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_74 0.0976 ms 50.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_60 0.0976 ms 49.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_58 0.0996 ms 48.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.4304 seconds and 0.0002 seconds precompiling for 20 choices


    Autotune Choices Stats:
    {"num_choices": 20, "num_triton_choices": 19, "best_kernel": "triton_mm_93", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.10265599936246872, "best_triton_pos": 0}
    AUTOTUNE mm(128x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_93 0.1027 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      mm 0.1042 ms 98.5% 
      triton_mm_94 0.1045 ms 98.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_88 0.1084 ms 94.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_87 0.1160 ms 88.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_83 0.1164 ms 88.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_92 0.1244 ms 82.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_84 0.1286 ms 79.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_85 0.1355 ms 75.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_89 0.1431 ms 71.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.5302 seconds and 0.3255 seconds precompiling for 20 choices


    Capturing batches (bs=2 avail_mem=55.08 GB):  75%|███████▌  | 3/4 [00:22<00:08,  8.32s/it]Capturing batches (bs=1 avail_mem=38.70 GB):  75%|███████▌  | 3/4 [00:22<00:08,  8.32s/it]

    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04742399975657463, "best_triton_pos": 1, "best_triton_time": 0.0480320006608963, "best_triton_kernel": "triton_mm_107", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x12288)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0474 ms 100.0% 
      triton_mm_107 0.0480 ms 98.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_111 0.0481 ms 98.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_103 0.0492 ms 96.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_99 0.0508 ms 93.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_102 0.0551 ms 86.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_106 0.0554 ms 85.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_96 0.0565 ms 84.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_98 0.0617 ms 76.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_97 0.0642 ms 73.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.5523 seconds and 0.2296 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.02175999991595745, "best_triton_pos": 1, "best_triton_time": 0.0226879995316267, "best_triton_kernel": "triton_mm_116", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x4096, 4096x4096)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      mm 0.0218 ms 100.0% 
      triton_mm_116 0.0227 ms 95.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_120 0.0237 ms 91.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_124 0.0266 ms 81.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_128 0.0303 ms 71.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_119 0.0388 ms 56.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_113 0.0395 ms 55.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_123 0.0403 ms 54.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_115 0.0404 ms 53.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_114 0.0423 ms 51.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.2588 seconds and 0.2945 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_137", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4", "best_time": 0.07513599842786789, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x22016)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_137 0.0751 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.0752 ms 99.9% 
      triton_mm_140 0.0756 ms 99.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_136 0.0756 ms 99.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_141 0.0763 ms 98.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_145 0.0799 ms 94.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_133 0.0833 ms 90.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_139 0.0913 ms 82.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
      triton_mm_142 0.0943 ms 79.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_130 0.0945 ms 79.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 0.3501 seconds and 0.1412 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "mm", "best_time": 0.04848000034689903, "best_triton_pos": 1, "best_triton_time": 0.051072001457214355, "best_triton_kernel": "triton_mm_150", "best_triton_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4"}
    AUTOTUNE mm(64x11008, 11008x4096)
    strides: [11008, 1], [1, 11008]
    dtypes: torch.float16, torch.float16
      mm 0.0485 ms 100.0% 
      triton_mm_150 0.0511 ms 94.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_154 0.0524 ms 92.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_158 0.0589 ms 82.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_162 0.0682 ms 71.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_153 0.0917 ms 52.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_149 0.0926 ms 52.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_147 0.0952 ms 50.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=2, num_warps=4
      triton_mm_157 0.0956 ms 50.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_148 0.0958 ms 50.6% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4102 seconds and 0.0003 seconds precompiling for 18 choices


    Autotune Choices Stats:
    {"num_choices": 18, "num_triton_choices": 17, "best_kernel": "triton_mm_174", "best_kernel_desc": "ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4", "best_time": 0.09974399954080582, "best_triton_pos": 0}
    AUTOTUNE mm(64x4096, 4096x32000)
    strides: [4096, 1], [1, 4096]
    dtypes: torch.float16, torch.float16
      triton_mm_174 0.0997 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_170 0.1004 ms 99.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8
      triton_mm_175 0.1004 ms 99.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4
      triton_mm_179 0.1009 ms 98.9% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8
      triton_mm_171 0.1018 ms 98.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      mm 0.1037 ms 96.2% 
      triton_mm_167 0.1105 ms 90.3% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4
      triton_mm_172 0.1181 ms 84.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_176 0.1210 ms 82.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4
      triton_mm_173 0.1326 ms 75.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8
    SingleProcess AUTOTUNE benchmarking takes 0.4124 seconds and 0.2827 seconds precompiling for 18 choices


    Capturing batches (bs=1 avail_mem=38.70 GB): 100%|██████████| 4/4 [00:41<00:00, 11.75s/it]Capturing batches (bs=1 avail_mem=38.70 GB): 100%|██████████| 4/4 [00:41<00:00, 10.30s/it]


    [2026-02-13 00:29:31] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 00:29:31] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.28s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.28s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.13 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.13 GB):  25%|██▌       | 1/4 [00:06<00:18,  6.05s/it]Capturing batches (bs=3 avail_mem=53.35 GB):  25%|██▌       | 1/4 [00:06<00:18,  6.05s/it]

    Capturing batches (bs=3 avail_mem=53.35 GB):  50%|█████     | 2/4 [00:06<00:05,  2.88s/it]Capturing batches (bs=2 avail_mem=53.33 GB):  50%|█████     | 2/4 [00:06<00:05,  2.88s/it]

    Capturing batches (bs=2 avail_mem=53.33 GB):  75%|███████▌  | 3/4 [00:06<00:01,  1.67s/it]Capturing batches (bs=1 avail_mem=53.28 GB):  75%|███████▌  | 3/4 [00:06<00:01,  1.67s/it]

    Capturing batches (bs=1 avail_mem=53.28 GB): 100%|██████████| 4/4 [00:11<00:00,  2.88s/it]Capturing batches (bs=1 avail_mem=53.28 GB): 100%|██████████| 4/4 [00:11<00:00,  2.92s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=34.45 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=34.37 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=34.37 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=34.39 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=34.39 GB): 100%|██████████| 4/4 [00:00<00:00, 59.87it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='8bc0d7e2d58643ccbc0b500924cf9f69', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770942593, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 00:29:58] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:29:58] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:29:58] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:30:00] WARNING model_config.py:1158: Casting torch.bfloat16 to torch.float16.
    [2026-02-13 00:30:00] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 00:30:00] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 00:30:00] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:30:01] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 00:30:07] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:30:07] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:30:07] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:30:07] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:30:07] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:30:07] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:30:09] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 00:30:10] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:30:25] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:30:25] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:30:25] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.35s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.43s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:13<00:04,  4.50s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.27s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.70s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=27.99 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=27.99 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.47it/s]Capturing batches (bs=3 avail_mem=27.81 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.47it/s]Capturing batches (bs=2 avail_mem=27.80 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.47it/s]Capturing batches (bs=2 avail_mem=27.80 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.62it/s]Capturing batches (bs=1 avail_mem=27.78 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.62it/s]Capturing batches (bs=1 avail_mem=27.78 GB): 100%|██████████| 4/4 [00:00<00:00,  4.78it/s]


    [2026-02-13 00:30:43] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 00:30:43] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-13 00:30:43] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-13 00:30:43] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=21.46 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=21.46 GB):  25%|██▌       | 1/4 [00:03<00:11,  3.71s/it]Capturing batches (bs=3 avail_mem=20.69 GB):  25%|██▌       | 1/4 [00:03<00:11,  3.71s/it]

    Capturing batches (bs=3 avail_mem=20.69 GB):  50%|█████     | 2/4 [00:04<00:03,  1.98s/it]Capturing batches (bs=2 avail_mem=20.67 GB):  50%|█████     | 2/4 [00:04<00:03,  1.98s/it]

    Capturing batches (bs=2 avail_mem=20.67 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.20s/it]Capturing batches (bs=1 avail_mem=20.60 GB):  75%|███████▌  | 3/4 [00:04<00:01,  1.20s/it]

    Capturing batches (bs=1 avail_mem=20.60 GB): 100%|██████████| 4/4 [00:07<00:00,  1.84s/it]Capturing batches (bs=1 avail_mem=20.60 GB): 100%|██████████| 4/4 [00:07<00:00,  1.89s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=20.53 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=20.46 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=20.45 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=20.43 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=20.43 GB): 100%|██████████| 4/4 [00:00<00:00, 36.61it/s]Capturing batches (bs=1 avail_mem=20.43 GB): 100%|██████████| 4/4 [00:00<00:00, 36.49it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='610fdf4401a140559bfa0d28c543535e', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770942663, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 00:31:09] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:31:09] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:31:09] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:31:12] WARNING model_config.py:1158: Casting torch.bfloat16 to torch.float16.
    [2026-02-13 00:31:12] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 00:31:12] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 00:31:12] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:31:12] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 00:31:20] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:31:20] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:31:20] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:31:20] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:31:20] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:31:20] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:31:22] Casting torch.bfloat16 to torch.float16.


    [2026-02-13 00:31:22] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:31:26] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:31:26] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:31:26] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.51s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:09<00:09,  4.60s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:13<00:04,  4.66s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:15<00:00,  3.30s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:15<00:00,  3.78s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.06 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.06 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.16s/it]Capturing batches (bs=3 avail_mem=59.94 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.16s/it]Capturing batches (bs=2 avail_mem=59.94 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.16s/it]Capturing batches (bs=2 avail_mem=59.94 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.94it/s]Capturing batches (bs=1 avail_mem=59.92 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.94it/s]Capturing batches (bs=1 avail_mem=59.92 GB): 100%|██████████| 4/4 [00:01<00:00,  3.04it/s]


    [2026-02-13 00:31:45] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 00:31:45] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-13 00:31:45] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-13 00:31:45] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.84it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.84it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.74 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.74 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.33s/it]Capturing batches (bs=3 avail_mem=57.38 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.33s/it]

    Capturing batches (bs=3 avail_mem=57.38 GB):  50%|█████     | 2/4 [00:03<00:03,  1.64s/it]Capturing batches (bs=2 avail_mem=57.33 GB):  50%|█████     | 2/4 [00:03<00:03,  1.64s/it]Capturing batches (bs=1 avail_mem=57.29 GB):  50%|█████     | 2/4 [00:03<00:03,  1.64s/it]

    Capturing batches (bs=1 avail_mem=57.29 GB): 100%|██████████| 4/4 [00:05<00:00,  1.27s/it]Capturing batches (bs=1 avail_mem=57.29 GB): 100%|██████████| 4/4 [00:05<00:00,  1.47s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=28.55 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=28.48 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=28.47 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=28.45 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=28.45 GB): 100%|██████████| 4/4 [00:00<00:00, 102.03it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='ae3e46c719c24f90b26c907f68a65019', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770942721, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 00:32:06] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:32:06] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:32:06] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:32:09] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 00:32:09] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 00:32:09] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:32:15] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:32:15] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:32:15] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:32:15] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:32:15] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:32:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:32:22] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:32:22] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:32:22] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.16it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.10it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.12it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.22it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.18it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=43.06 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=43.06 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.38it/s]Capturing batches (bs=3 avail_mem=43.00 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.38it/s]Capturing batches (bs=2 avail_mem=42.99 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.38it/s]Capturing batches (bs=2 avail_mem=42.99 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.37it/s]Capturing batches (bs=1 avail_mem=42.98 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.37it/s]Capturing batches (bs=1 avail_mem=42.98 GB): 100%|██████████| 4/4 [00:00<00:00,  4.51it/s]


    [2026-02-13 00:32:28] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 00:32:28] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  4.42it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  7.38it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  7.02it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=28.16 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=28.11 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=28.11 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=28.11 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=28.11 GB): 100%|██████████| 4/4 [00:00<00:00, 33.45it/s]Capturing batches (bs=1 avail_mem=28.11 GB): 100%|██████████| 4/4 [00:00<00:00, 33.40it/s]



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


<strong style='color: #00008B;'>{'id': '01a6dabe65194f8baabe4ba572cc6147', 'object': 'chat.completion', 'created': 1770942762, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "<think>\nOkay, the user is asking for the capital of France. Let me start by recalling what I know about France's geography. France is a country in Europe, right? I remember that Paris is a major city there. But wait, maybe I should confirm if that's actually the capital. Sometimes people confuse the largest city with the capital. For example, in the US, New York City is the largest city, but Washington, D.C. is the capital.\n\nSo, in France, Paris is not only the largest city but also the capital. The Eiffel Tower is located in Paris, which is one of the most iconic landmarks. Also, government buildings like the French government's seat, which is the Réunion of the National Assembly, the Senate, and the National Council, all located in Paris. The president of France also resides in the Palace of the Enlightenment (Palais de l'Égalité) in巴黎. So that seems to check out. \n\nWait, maybe I should check if there's any other city that has claimed to be the capital historically. For instance, during the French Revolution, did they move the capital? No, I think Paris remained the capital throughout the revolutions. Also, historical cities like Lyon were important in the past, but now it's definitely Paris. \n\nAnother angle: administrative regions. France is divided into regions, and Paris is an autonomous region, the Île-de-France region. The capital city, Paris, is part of this region. So that's consistent. \n\nAlso, considering cultural references, when people talk about the French capital, they always mention Paris. The Louvre Museum is in Paris, which is a major cultural institution. The stock exchange of France is also in Paris, the Bourse de Paris. All these institutions point to Paris as the capital. \n\nI don't think there's any official change recently that moved the capital. Some countries have restructured their capitals over time, but France hasn't. For example, Berlin had East and West parts but was reunified. Paris remains unchanged. \n\nSo, putting it all together: Paris is the capital, largest city, and a global hub for politics, culture, and economy in France. The answer should be straightforward.\n\nWait, maybe check a reliable source mentally. If I recall, in geography tests, France's capital is always Paris. Yes, that's correct. No controversies anymore. Unless in some alternate history, but in reality, it's Paris. \n\nTherefore, the answer is Paris. Simple and straightforward.\n</think>\n\nThe capital of France is **Paris**. This historic city is known for iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Champs-Élysées. It serves as the political, cultural, and economic heart of the country. 🇫🇷 Paris", 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 604, 'completion_tokens': 578, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



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

    [2026-02-13 00:32:48] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:32:48] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:32:48] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:32:50] INFO server_args.py:1813: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-13 00:32:50] WARNING server_args.py:2321: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-13 00:32:50] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:32:57] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:32:57] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:32:57] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:32:57] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:32:57] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:32:57] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:33:03] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:33:03] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:33:03] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.32it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.25it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.23it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.25it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.25it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.97 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=37.97 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.42it/s]Capturing batches (bs=3 avail_mem=14.22 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.42it/s]Capturing batches (bs=2 avail_mem=14.07 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.42it/s]Capturing batches (bs=2 avail_mem=14.07 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.34it/s]Capturing batches (bs=1 avail_mem=14.04 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.34it/s]Capturing batches (bs=1 avail_mem=14.04 GB): 100%|██████████| 4/4 [00:00<00:00,  4.55it/s]


    [2026-02-13 00:33:09] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-13 00:33:09] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.65it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.65it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=9.88 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=9.88 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.92s/it]Capturing batches (bs=3 avail_mem=8.38 GB):  25%|██▌       | 1/4 [00:04<00:14,  4.92s/it]

    Capturing batches (bs=3 avail_mem=8.38 GB):  50%|█████     | 2/4 [00:05<00:05,  2.53s/it]Capturing batches (bs=2 avail_mem=25.42 GB):  50%|█████     | 2/4 [00:05<00:05,  2.53s/it]Capturing batches (bs=2 avail_mem=25.42 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.47s/it]Capturing batches (bs=1 avail_mem=25.36 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.47s/it]

    Capturing batches (bs=1 avail_mem=25.36 GB): 100%|██████████| 4/4 [00:09<00:00,  2.27s/it]Capturing batches (bs=1 avail_mem=25.36 GB): 100%|██████████| 4/4 [00:09<00:00,  2.37s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=25.28 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=25.28 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.25s/it]Capturing batches (bs=3 avail_mem=25.15 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.25s/it]Capturing batches (bs=2 avail_mem=25.07 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.25s/it]Capturing batches (bs=2 avail_mem=25.07 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.41s/it]Capturing batches (bs=1 avail_mem=24.99 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.41s/it]

    Capturing batches (bs=1 avail_mem=24.99 GB): 100%|██████████| 4/4 [00:05<00:00,  1.37s/it]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='49030f4e274b48d5b2c265a88e4fbdc2', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770942818, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 00:33:44] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:33:44] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:33:44] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:33:46] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 00:33:46] WARNING server_args.py:2309: Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on.
    [2026-02-13 00:33:46] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:33:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:33:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:33:53] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:33:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:33:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:33:53] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:33:59] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:33:59] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:33:59] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.50it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.40it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.34it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.38it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.39it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.80 GB):  25%|██▌       | 1/4 [00:01<00:04,  1.42s/it]Capturing batches (bs=3 avail_mem=62.74 GB):  25%|██▌       | 1/4 [00:01<00:04,  1.42s/it]Capturing batches (bs=2 avail_mem=62.73 GB):  25%|██▌       | 1/4 [00:01<00:04,  1.42s/it]Capturing batches (bs=2 avail_mem=62.73 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.42it/s]Capturing batches (bs=1 avail_mem=62.73 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.42it/s]Capturing batches (bs=1 avail_mem=62.73 GB): 100%|██████████| 4/4 [00:01<00:00,  2.53it/s]


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.88it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.88it/s]
    


    [2026-02-13 00:34:08] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.38 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.38 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.40s/it]Capturing batches (bs=3 avail_mem=56.86 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.40s/it]

    Capturing batches (bs=3 avail_mem=56.86 GB):  50%|█████     | 2/4 [00:02<00:02,  1.31s/it]Capturing batches (bs=2 avail_mem=56.82 GB):  50%|█████     | 2/4 [00:02<00:02,  1.31s/it]Capturing batches (bs=2 avail_mem=56.82 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.30it/s]Capturing batches (bs=1 avail_mem=56.80 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.30it/s]

    Capturing batches (bs=1 avail_mem=56.80 GB): 100%|██████████| 4/4 [00:04<00:00,  1.13it/s]Capturing batches (bs=1 avail_mem=56.80 GB): 100%|██████████| 4/4 [00:04<00:00,  1.03s/it]


    [2026-02-13 00:34:15] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='dad42e1ab7734ce492ac8d4aabc18e26', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770942859, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-13 00:34:24] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:34:24] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:34:24] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-13 00:34:26] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-13 00:34:26] WARNING server_args.py:2415: The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding.
    [2026-02-13 00:34:26] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-13 00:34:32] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:34:32] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:34:32] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-13 00:34:33] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-13 00:34:33] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-13 00:34:33] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-13 00:34:39] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:34:39] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-13 00:34:39] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.48it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.39it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.33it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.35it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.36it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.60 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.60 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.64it/s]Capturing batches (bs=3 avail_mem=58.53 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.64it/s]Capturing batches (bs=2 avail_mem=58.52 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.64it/s]Capturing batches (bs=2 avail_mem=58.52 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.10it/s]Capturing batches (bs=1 avail_mem=58.51 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.10it/s]Capturing batches (bs=1 avail_mem=58.51 GB): 100%|██████████| 4/4 [00:00<00:00,  4.97it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='7050a9ef08b140c4911ddfb5e365beb6', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1770942892, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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
