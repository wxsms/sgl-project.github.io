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

    [2026-02-14 15:12:59] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-14 15:12:59] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-14 15:12:59] INFO utils.py:164: NumExpr defaulting to 16 threads.



```python
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 3 \
    --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 --cuda-graph-max-bs 8 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:13:05] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:13:05] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:13:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:13:07] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 15:13:07] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 15:13:07] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:13:13] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:13:13] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:13:13] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:13:13] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:13:13] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:13:13] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:13:18] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:13:18] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:13:18] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.67s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.42s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=40.22 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=40.22 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.14it/s]Capturing batches (bs=3 avail_mem=40.14 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.14it/s]Capturing batches (bs=2 avail_mem=40.13 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.14it/s]Capturing batches (bs=2 avail_mem=40.13 GB):  75%|███████▌  | 3/4 [00:01<00:00,  3.66it/s]Capturing batches (bs=1 avail_mem=40.10 GB):  75%|███████▌  | 3/4 [00:01<00:00,  3.66it/s]

    Capturing batches (bs=1 avail_mem=40.10 GB): 100%|██████████| 4/4 [00:01<00:00,  3.52it/s]


    [2026-02-14 15:13:24] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 15:13:24] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.21s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.21s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.85 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=37.85 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.81s/it]Capturing batches (bs=3 avail_mem=37.75 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.81s/it]

    Capturing batches (bs=3 avail_mem=37.75 GB):  50%|█████     | 2/4 [00:03<00:02,  1.37s/it]Capturing batches (bs=2 avail_mem=37.73 GB):  50%|█████     | 2/4 [00:03<00:02,  1.37s/it]

    Capturing batches (bs=2 avail_mem=37.73 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.11it/s]Capturing batches (bs=1 avail_mem=37.70 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.11it/s]

    Capturing batches (bs=1 avail_mem=37.70 GB): 100%|██████████| 4/4 [00:05<00:00,  1.35s/it]Capturing batches (bs=1 avail_mem=37.70 GB): 100%|██████████| 4/4 [00:05<00:00,  1.39s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.64 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=37.58 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=37.58 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=37.56 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=37.56 GB): 100%|██████████| 4/4 [00:00<00:00, 46.87it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='231cf2dd09c8413c8aee28ab5e1eb23b', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1771082020, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:13:45] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:13:45] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:13:45] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:13:48] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 15:13:48] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 15:13:48] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:13:54] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:13:54] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:13:54] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:13:54] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:13:54] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:13:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:13:59] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:13:59] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:13:59] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.69s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.24s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.31s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.19 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.19 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.03it/s]Capturing batches (bs=3 avail_mem=55.10 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.03it/s]Capturing batches (bs=2 avail_mem=55.08 GB):  25%|██▌       | 1/4 [00:01<00:02,  1.03it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Capturing batches (bs=2 avail_mem=55.08 GB):  75%|███████▌  | 3/4 [00:07<00:02,  2.84s/it]Capturing batches (bs=1 avail_mem=52.31 GB):  75%|███████▌  | 3/4 [00:07<00:02,  2.84s/it]

    Capturing batches (bs=1 avail_mem=52.31 GB): 100%|██████████| 4/4 [00:15<00:00,  4.43s/it]Capturing batches (bs=1 avail_mem=52.31 GB): 100%|██████████| 4/4 [00:15<00:00,  3.85s/it]


    [2026-02-14 15:14:20] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 15:14:20] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.30s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.30s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.76 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=37.76 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.20s/it]Capturing batches (bs=3 avail_mem=37.61 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.20s/it]

    Capturing batches (bs=3 avail_mem=37.61 GB):  50%|█████     | 2/4 [00:03<00:03,  1.64s/it]Capturing batches (bs=2 avail_mem=37.59 GB):  50%|█████     | 2/4 [00:03<00:03,  1.64s/it]Capturing batches (bs=2 avail_mem=37.59 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.05it/s]Capturing batches (bs=1 avail_mem=37.54 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.05it/s]

    Capturing batches (bs=1 avail_mem=37.54 GB): 100%|██████████| 4/4 [00:06<00:00,  1.55s/it]Capturing batches (bs=1 avail_mem=37.54 GB): 100%|██████████| 4/4 [00:06<00:00,  1.59s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=53.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=53.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=53.83 GB):  50%|█████     | 2/4 [00:00<00:00, 11.51it/s]Capturing batches (bs=2 avail_mem=53.83 GB):  50%|█████     | 2/4 [00:00<00:00, 11.51it/s]

    Capturing batches (bs=1 avail_mem=53.81 GB):  50%|█████     | 2/4 [00:00<00:00, 11.51it/s]Capturing batches (bs=1 avail_mem=53.81 GB): 100%|██████████| 4/4 [00:00<00:00, 11.25it/s]Capturing batches (bs=1 avail_mem=53.81 GB): 100%|██████████| 4/4 [00:00<00:00, 11.28it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='f072f7ad70484fa0a86391bf54e59563', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1771082077, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:14:43] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:14:43] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:14:43] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:14:46] WARNING model_config.py:1178: Casting torch.bfloat16 to torch.float16.
    [2026-02-14 15:14:46] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 15:14:46] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 15:14:46] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:14:46] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 15:14:52] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:14:52] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:14:52] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:14:52] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:14:52] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:14:52] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:14:54] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 15:14:54] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:14:57] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:14:57] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:14:57] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.48s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.49s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:13<00:04,  4.43s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.16s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.64s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.05 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.05 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.07s/it]

    Capturing batches (bs=3 avail_mem=57.88 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.07s/it]Capturing batches (bs=3 avail_mem=57.88 GB):  50%|█████     | 2/4 [00:01<00:01,  1.64it/s]Capturing batches (bs=2 avail_mem=57.87 GB):  50%|█████     | 2/4 [00:01<00:01,  1.64it/s]

    Capturing batches (bs=2 avail_mem=57.87 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.10it/s]Capturing batches (bs=1 avail_mem=57.84 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.10it/s]Capturing batches (bs=1 avail_mem=57.84 GB): 100%|██████████| 4/4 [00:01<00:00,  2.87it/s]Capturing batches (bs=1 avail_mem=57.84 GB): 100%|██████████| 4/4 [00:01<00:00,  2.19it/s]


    [2026-02-14 15:15:16] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 15:15:16] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-14 15:15:16] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-14 15:15:16] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.02s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.02s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=56.84 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=56.84 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.82s/it]Capturing batches (bs=3 avail_mem=58.25 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.82s/it]

    Capturing batches (bs=3 avail_mem=58.25 GB):  50%|█████     | 2/4 [00:03<00:02,  1.38s/it]Capturing batches (bs=2 avail_mem=58.21 GB):  50%|█████     | 2/4 [00:03<00:02,  1.38s/it]

    Capturing batches (bs=2 avail_mem=58.21 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.06it/s]Capturing batches (bs=1 avail_mem=58.17 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.06it/s]

    Capturing batches (bs=1 avail_mem=58.17 GB): 100%|██████████| 4/4 [00:05<00:00,  1.42s/it]Capturing batches (bs=1 avail_mem=58.17 GB): 100%|██████████| 4/4 [00:05<00:00,  1.44s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=56.46 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=56.42 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=56.40 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=56.36 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=56.36 GB): 100%|██████████| 4/4 [00:00<00:00, 75.82it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='4979af9169a84c3896398f1f23609874', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1771082131, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:15:37] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:15:37] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:15:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:15:39] WARNING model_config.py:1178: Casting torch.bfloat16 to torch.float16.
    [2026-02-14 15:15:39] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 15:15:39] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 15:15:39] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:15:39] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 15:15:45] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:15:45] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:15:45] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:15:45] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:15:45] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:15:45] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:15:47] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 15:15:48] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:15:51] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:15:51] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:15:51] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:13,  4.40s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.37s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.26s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.07s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.53s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=57.89 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=57.89 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.75it/s]Capturing batches (bs=3 avail_mem=57.77 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.75it/s]Capturing batches (bs=2 avail_mem=57.76 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.75it/s]Capturing batches (bs=1 avail_mem=57.74 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.75it/s]Capturing batches (bs=1 avail_mem=57.74 GB): 100%|██████████| 4/4 [00:00<00:00,  6.78it/s]Capturing batches (bs=1 avail_mem=57.74 GB): 100%|██████████| 4/4 [00:00<00:00,  5.58it/s]


    [2026-02-14 15:16:08] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 15:16:08] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-14 15:16:08] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-14 15:16:08] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.97it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.97it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.09 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.09 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.75s/it]Capturing batches (bs=3 avail_mem=31.31 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.75s/it]

    Capturing batches (bs=3 avail_mem=31.31 GB):  50%|█████     | 2/4 [00:03<00:02,  1.35s/it]Capturing batches (bs=2 avail_mem=31.27 GB):  50%|█████     | 2/4 [00:03<00:02,  1.35s/it]

    Capturing batches (bs=2 avail_mem=31.27 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.15it/s]Capturing batches (bs=1 avail_mem=31.23 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.15it/s]

    Capturing batches (bs=1 avail_mem=31.23 GB): 100%|██████████| 4/4 [00:05<00:00,  1.31s/it]Capturing batches (bs=1 avail_mem=31.23 GB): 100%|██████████| 4/4 [00:05<00:00,  1.35s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=31.17 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=31.10 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=31.10 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=31.08 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=31.08 GB): 100%|██████████| 4/4 [00:00<00:00, 102.00it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='aa7a88167e79450885201aabeef3df89', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1771082182, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:16:28] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:16:28] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:16:28] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:16:30] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.
    [2026-02-14 15:16:30] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 15:16:30] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:16:38] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:16:38] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:16:38] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:16:38] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:16:38] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:16:38] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:16:44] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:16:44] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:16:44] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.48it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.41it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.44it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.57it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.52it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=56.09 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=56.09 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=3 avail_mem=56.03 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=2 avail_mem=56.02 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=1 avail_mem=56.01 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=1 avail_mem=56.01 GB): 100%|██████████| 4/4 [00:00<00:00,  5.94it/s]Capturing batches (bs=1 avail_mem=56.01 GB): 100%|██████████| 4/4 [00:00<00:00,  4.77it/s]


    [2026-02-14 15:16:49] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 15:16:49] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  5.13it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  8.67it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  8.24it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.56 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.56 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.27it/s]Capturing batches (bs=3 avail_mem=60.50 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.27it/s]Capturing batches (bs=2 avail_mem=60.50 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.27it/s]Capturing batches (bs=1 avail_mem=60.50 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.27it/s]Capturing batches (bs=1 avail_mem=60.50 GB): 100%|██████████| 4/4 [00:00<00:00, 13.68it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>{'id': '7f3cca73c9ff4e6b818bc3fa15df846a', 'object': 'chat.completion', 'created': 1771082219, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<think>\nOkay, so the user is asking, "What is the capital of France?" Let me start by recalling basic geography. I know France is a country in Europe. The capital... Hmm, I think it\'s Paris. But wait, let me make sure I\'m not mixing it up with another country. For example, Germany\'s capital is Berlin, and Italy\'s is Rome. So France\'s should be Paris.\n\nWait, maybe there\'s a trick here. Sometimes quizzes or tests might ask for capitals of regions instead of the country. But the question specifically says "the capital of France," which refers to the country as a whole. So Paris is definitely the answer. \n\nI should also think about major landmarks in Paris. The Eiffel Tower, Louvre Museum, Notre-Dame Cathedral—all are in Paris. The president of France resides in the Élysée Palace, which is in Paris. So all these points confirm that Paris is the capital. \n\nIs there any other city that could be confused as the capital? Maybe Lille or Lyon, but those are major cities but not the capital. I\'ve heard of some countries having capitals different from the largest city, but in France, Paris is both the largest city and the capital. \n\nSo I can confidently say the answer is Paris. No contradictions here. Just need to present it clearly and maybe add a bit more context if needed, but the question is straightforward.\n\nWait, just a moment. Let me check if there was any recent change. I don\'t recall any announcements about France moving its capital. Paris has been the capital since the French Revolution, right? Before that, some cities like Versailles had the royal residence, but the capital was still considered Paris. So yes, it\'s safe to say Paris is the capital.\n</think>\n\nThe capital of France is **Paris**. This historic city is known for iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French government and the official residence of the President, the Élysée Palace. Paris has long been the political, cultural, and economic center of France.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 465, 'completion_tokens': 439, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:17:05] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:17:05] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:17:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:17:07] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 15:17:07] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 15:17:07] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:17:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:17:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:17:14] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:17:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:17:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:17:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:17:19] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:17:19] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:17:19] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.63it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.59it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.43it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.46it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.48it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=45.89 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=45.89 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.48it/s]Capturing batches (bs=3 avail_mem=45.81 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.48it/s]Capturing batches (bs=2 avail_mem=45.80 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.48it/s]Capturing batches (bs=2 avail_mem=45.80 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.47it/s]Capturing batches (bs=1 avail_mem=45.78 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.47it/s]Capturing batches (bs=1 avail_mem=45.78 GB): 100%|██████████| 4/4 [00:00<00:00,  4.69it/s]


    [2026-02-14 15:17:25] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 15:17:25] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.78it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.78it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=40.57 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=40.57 GB):  25%|██▌       | 1/4 [00:04<00:13,  4.36s/it]Capturing batches (bs=3 avail_mem=39.12 GB):  25%|██▌       | 1/4 [00:04<00:13,  4.36s/it]

    Capturing batches (bs=3 avail_mem=39.12 GB):  50%|█████     | 2/4 [00:04<00:04,  2.17s/it]Capturing batches (bs=2 avail_mem=38.08 GB):  50%|█████     | 2/4 [00:04<00:04,  2.17s/it]Capturing batches (bs=2 avail_mem=38.08 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.26s/it]Capturing batches (bs=1 avail_mem=37.60 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.26s/it]

    Capturing batches (bs=1 avail_mem=37.60 GB): 100%|██████████| 4/4 [00:08<00:00,  2.14s/it]Capturing batches (bs=1 avail_mem=37.60 GB): 100%|██████████| 4/4 [00:08<00:00,  2.17s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.45 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=37.37 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=37.37 GB):  50%|█████     | 2/4 [00:00<00:00, 12.07it/s]Capturing batches (bs=2 avail_mem=37.37 GB):  50%|█████     | 2/4 [00:00<00:00, 12.07it/s]

    Capturing batches (bs=1 avail_mem=37.34 GB):  50%|█████     | 2/4 [00:00<00:00, 12.07it/s]Capturing batches (bs=1 avail_mem=37.34 GB): 100%|██████████| 4/4 [00:00<00:00, 12.61it/s]Capturing batches (bs=1 avail_mem=37.34 GB): 100%|██████████| 4/4 [00:00<00:00, 12.52it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='79864154846849b0be61ffcd64e8fbd2', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1771082263, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:17:48] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:17:48] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:17:48] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:17:50] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.
    [2026-02-14 15:17:50] WARNING server_args.py:2341: Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on.
    [2026-02-14 15:17:50] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:17:56] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:17:56] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:17:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:17:56] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:17:56] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:17:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:18:01] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:18:01] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:18:01] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.60it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.65it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.43it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.41it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.46it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.80 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.73it/s]Capturing batches (bs=3 avail_mem=62.74 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.73it/s]Capturing batches (bs=2 avail_mem=62.73 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.73it/s]Capturing batches (bs=2 avail_mem=62.73 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.12it/s]Capturing batches (bs=1 avail_mem=62.73 GB):  75%|███████▌  | 3/4 [00:00<00:00,  5.12it/s]Capturing batches (bs=1 avail_mem=62.73 GB): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.69it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.69it/s]
    


    [2026-02-14 15:18:09] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.38 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.38 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.32s/it]Capturing batches (bs=3 avail_mem=43.96 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.32s/it]

    Capturing batches (bs=3 avail_mem=43.96 GB):  50%|█████     | 2/4 [00:02<00:02,  1.34s/it]Capturing batches (bs=2 avail_mem=43.94 GB):  50%|█████     | 2/4 [00:02<00:02,  1.34s/it]Capturing batches (bs=1 avail_mem=43.93 GB):  50%|█████     | 2/4 [00:03<00:02,  1.34s/it]

    Capturing batches (bs=1 avail_mem=43.93 GB): 100%|██████████| 4/4 [00:03<00:00,  1.24it/s]Capturing batches (bs=1 avail_mem=43.93 GB): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]


    [2026-02-14 15:18:16] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='66f87b6cf1df45a6b9ecc5a6c1afe593', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1771082300, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-02-14 15:18:26] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:18:26] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:18:26] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 15:18:28] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.
    [2026-02-14 15:18:28] WARNING server_args.py:2448: The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding.
    [2026-02-14 15:18:28] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 15:18:36] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:18:36] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:18:36] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 15:18:36] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 15:18:36] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 15:18:36] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 15:18:41] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:18:41] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 15:18:41] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.83it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.76it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.48it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.52it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.56it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.77 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.77 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.51it/s]Capturing batches (bs=3 avail_mem=62.70 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.51it/s]Capturing batches (bs=2 avail_mem=62.69 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.51it/s]Capturing batches (bs=2 avail_mem=62.69 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.72it/s]Capturing batches (bs=1 avail_mem=62.68 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.72it/s]

    Capturing batches (bs=1 avail_mem=62.68 GB): 100%|██████████| 4/4 [00:00<00:00,  4.58it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='3cf161799a154aeda0952604f2b30383', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1771082334, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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
