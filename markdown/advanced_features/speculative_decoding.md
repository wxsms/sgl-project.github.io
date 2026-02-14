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

    [2026-02-14 14:25:08] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-14 14:25:08] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-14 14:25:08] INFO utils.py:164: NumExpr defaulting to 16 threads.



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

    [2026-02-14 14:25:14] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:25:14] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:25:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:25:17] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 14:25:17] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 14:25:17] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:25:23] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:25:23] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:25:23] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 14:25:23] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:25:23] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:25:23] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:25:31] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:25:31] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:25:31] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:02<00:02,  2.01s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.43s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.52s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.55 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.55 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.63it/s]Capturing batches (bs=3 avail_mem=54.47 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.63it/s]Capturing batches (bs=2 avail_mem=54.47 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.63it/s]Capturing batches (bs=1 avail_mem=54.44 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.63it/s]Capturing batches (bs=1 avail_mem=54.44 GB): 100%|██████████| 4/4 [00:00<00:00,  6.00it/s]Capturing batches (bs=1 avail_mem=54.44 GB): 100%|██████████| 4/4 [00:00<00:00,  5.00it/s]


    [2026-02-14 14:25:37] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 14:25:37] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.31s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.31s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=39.21 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=39.21 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.86s/it]Capturing batches (bs=3 avail_mem=37.70 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.86s/it]

    Capturing batches (bs=3 avail_mem=37.70 GB):  50%|█████     | 2/4 [00:03<00:02,  1.42s/it]Capturing batches (bs=2 avail_mem=37.68 GB):  50%|█████     | 2/4 [00:03<00:02,  1.42s/it]Capturing batches (bs=1 avail_mem=37.66 GB):  50%|█████     | 2/4 [00:03<00:02,  1.42s/it]

    Capturing batches (bs=1 avail_mem=37.66 GB): 100%|██████████| 4/4 [00:05<00:00,  1.17s/it]Capturing batches (bs=1 avail_mem=37.66 GB): 100%|██████████| 4/4 [00:05<00:00,  1.33s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=37.60 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=37.60 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.40it/s]Capturing batches (bs=3 avail_mem=37.54 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.40it/s]Capturing batches (bs=2 avail_mem=37.54 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.40it/s]Capturing batches (bs=1 avail_mem=37.52 GB):  25%|██▌       | 1/4 [00:00<00:00,  4.40it/s]Capturing batches (bs=1 avail_mem=37.52 GB): 100%|██████████| 4/4 [00:00<00:00, 15.13it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='b069cf6c59ed4334a11c63e58d1ed649', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1771079153, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-14 14:25:59] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:25:59] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:25:59] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:26:01] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 14:26:01] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 14:26:01] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:26:08] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:26:08] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:26:08] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 14:26:08] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:26:08] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:26:08] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:26:13] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:26:13] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:26:13] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.87s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.32s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.40s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=38.42 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=38.42 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.59it/s]Capturing batches (bs=3 avail_mem=38.33 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.59it/s]Capturing batches (bs=3 avail_mem=38.33 GB):  50%|█████     | 2/4 [00:00<00:00,  3.13it/s]Capturing batches (bs=2 avail_mem=38.32 GB):  50%|█████     | 2/4 [00:00<00:00,  3.13it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Capturing batches (bs=2 avail_mem=38.32 GB):  75%|███████▌  | 3/4 [00:08<00:03,  3.50s/it]Capturing batches (bs=1 avail_mem=55.05 GB):  75%|███████▌  | 3/4 [00:08<00:03,  3.50s/it]

    Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:14<00:00,  4.70s/it]Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:14<00:00,  3.64s/it]


    [2026-02-14 14:26:33] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 14:26:33] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.15s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.15s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=35.48 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=35.48 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.89s/it]Capturing batches (bs=3 avail_mem=35.19 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.89s/it]

    Capturing batches (bs=3 avail_mem=35.19 GB):  50%|█████     | 2/4 [00:03<00:02,  1.45s/it]Capturing batches (bs=2 avail_mem=35.17 GB):  50%|█████     | 2/4 [00:03<00:02,  1.45s/it]Capturing batches (bs=2 avail_mem=35.17 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.19it/s]Capturing batches (bs=1 avail_mem=35.13 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.19it/s]

    Capturing batches (bs=1 avail_mem=35.13 GB): 100%|██████████| 4/4 [00:06<00:00,  1.66s/it]Capturing batches (bs=1 avail_mem=35.13 GB): 100%|██████████| 4/4 [00:06<00:00,  1.59s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=53.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=53.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=53.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=53.81 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=53.81 GB): 100%|██████████| 4/4 [00:00<00:00, 65.80it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='b3ce2745f73b4c7a9fab8351bf1b2a5f', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1771079209, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-14 14:26:54] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:26:54] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:26:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:26:56] WARNING model_config.py:1178: Casting torch.bfloat16 to torch.float16.
    [2026-02-14 14:26:56] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 14:26:56] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 14:26:56] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:26:57] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 14:27:03] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:27:03] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:27:03] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 14:27:03] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:27:03] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:27:03] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:27:05] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 14:27:06] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:27:09] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:27:09] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:27:09] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:12,  4.23s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.26s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.20s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.00s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.45s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.96 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.96 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.31s/it]Capturing batches (bs=3 avail_mem=45.02 GB):  25%|██▌       | 1/4 [00:01<00:03,  1.31s/it]

    Capturing batches (bs=3 avail_mem=45.02 GB):  50%|█████     | 2/4 [00:01<00:01,  1.42it/s]Capturing batches (bs=2 avail_mem=45.01 GB):  50%|█████     | 2/4 [00:01<00:01,  1.42it/s]Capturing batches (bs=2 avail_mem=45.01 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.13it/s]Capturing batches (bs=1 avail_mem=44.98 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.13it/s]

    Capturing batches (bs=1 avail_mem=44.98 GB): 100%|██████████| 4/4 [00:01<00:00,  3.07it/s]Capturing batches (bs=1 avail_mem=44.98 GB): 100%|██████████| 4/4 [00:01<00:00,  2.12it/s]


    [2026-02-14 14:27:27] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 14:27:27] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-14 14:27:27] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-14 14:27:27] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.04it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.04it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=42.51 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=42.51 GB):  25%|██▌       | 1/4 [00:04<00:12,  4.03s/it]Capturing batches (bs=3 avail_mem=42.25 GB):  25%|██▌       | 1/4 [00:04<00:12,  4.03s/it]

    Capturing batches (bs=3 avail_mem=42.25 GB):  50%|█████     | 2/4 [00:04<00:04,  2.17s/it]Capturing batches (bs=2 avail_mem=58.88 GB):  50%|█████     | 2/4 [00:04<00:04,  2.17s/it]

    Capturing batches (bs=2 avail_mem=58.88 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.28s/it]Capturing batches (bs=1 avail_mem=58.84 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.28s/it]

    Capturing batches (bs=1 avail_mem=58.84 GB): 100%|██████████| 4/4 [00:07<00:00,  1.65s/it]Capturing batches (bs=1 avail_mem=58.84 GB): 100%|██████████| 4/4 [00:07<00:00,  1.84s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.70 GB): 100%|██████████| 4/4 [00:00<00:00, 103.38it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='730f30d5a64248d89a5fc1d849294f7d', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1771079265, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-14 14:27:50] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:27:50] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:27:50] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:27:53] WARNING model_config.py:1178: Casting torch.bfloat16 to torch.float16.
    [2026-02-14 14:27:53] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 14:27:53] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 14:27:53] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:27:53] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 14:27:59] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:27:59] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:27:59] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 14:27:59] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:27:59] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:27:59] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:28:02] Casting torch.bfloat16 to torch.float16.


    [2026-02-14 14:28:02] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:28:06] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:28:06] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:28:06] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:12,  4.23s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.32s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.30s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.09s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:14<00:00,  3.53s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=56.20 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=56.20 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.31s/it]Capturing batches (bs=3 avail_mem=59.93 GB):  25%|██▌       | 1/4 [00:02<00:06,  2.31s/it]

    Capturing batches (bs=3 avail_mem=59.93 GB):  50%|█████     | 2/4 [00:02<00:02,  1.08s/it]Capturing batches (bs=2 avail_mem=59.92 GB):  50%|█████     | 2/4 [00:02<00:02,  1.08s/it]Capturing batches (bs=2 avail_mem=59.92 GB):  75%|███████▌  | 3/4 [00:02<00:00,  1.49it/s]Capturing batches (bs=1 avail_mem=59.90 GB):  75%|███████▌  | 3/4 [00:02<00:00,  1.49it/s]

    Capturing batches (bs=1 avail_mem=59.90 GB): 100%|██████████| 4/4 [00:02<00:00,  1.43it/s]


    [2026-02-14 14:28:25] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 14:28:25] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-14 14:28:25] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-14 14:28:25] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.86it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.86it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.74 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.74 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.70s/it]Capturing batches (bs=3 avail_mem=54.80 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.70s/it]

    Capturing batches (bs=3 avail_mem=54.80 GB):  50%|█████     | 2/4 [00:03<00:02,  1.34s/it]Capturing batches (bs=2 avail_mem=54.76 GB):  50%|█████     | 2/4 [00:03<00:02,  1.34s/it]Capturing batches (bs=2 avail_mem=54.76 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.29it/s]Capturing batches (bs=1 avail_mem=54.72 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.29it/s]

    Capturing batches (bs=1 avail_mem=54.72 GB): 100%|██████████| 4/4 [00:05<00:00,  1.26s/it]Capturing batches (bs=1 avail_mem=54.72 GB): 100%|██████████| 4/4 [00:05<00:00,  1.30s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.63 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.63 GB):  25%|██▌       | 1/4 [00:00<00:00,  8.31it/s]Capturing batches (bs=3 avail_mem=54.55 GB):  25%|██▌       | 1/4 [00:00<00:00,  8.31it/s]Capturing batches (bs=2 avail_mem=54.55 GB):  25%|██▌       | 1/4 [00:00<00:00,  8.31it/s]

    Capturing batches (bs=1 avail_mem=54.53 GB):  25%|██▌       | 1/4 [00:00<00:00,  8.31it/s]Capturing batches (bs=1 avail_mem=54.53 GB): 100%|██████████| 4/4 [00:00<00:00, 19.65it/s]Capturing batches (bs=1 avail_mem=54.53 GB): 100%|██████████| 4/4 [00:00<00:00, 17.81it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='b28b7ec4ba494c86a2b0c1bbd37ba12c', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1771079320, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-14 14:28:46] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:28:46] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:28:46] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:28:49] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.
    [2026-02-14 14:28:49] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 14:28:49] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:28:56] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:28:56] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:28:56] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 14:28:56] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:28:56] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:28:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:29:02] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:29:02] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:29:02] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:02<00:07,  2.52s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:05<00:05,  2.58s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:07<00:02,  2.61s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:10<00:00,  2.45s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:10<00:00,  2.50s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.46 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.46 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=3 avail_mem=60.39 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=2 avail_mem=60.39 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=1 avail_mem=60.38 GB):  25%|██▌       | 1/4 [00:00<00:02,  1.39it/s]Capturing batches (bs=1 avail_mem=60.38 GB): 100%|██████████| 4/4 [00:00<00:00,  5.95it/s]Capturing batches (bs=1 avail_mem=60.38 GB): 100%|██████████| 4/4 [00:00<00:00,  4.77it/s]


    [2026-02-14 14:29:15] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 14:29:15] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  6.19it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  5.64it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  5.68it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=45.85 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=45.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=45.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=45.80 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=45.80 GB): 100%|██████████| 4/4 [00:00<00:00, 41.76it/s]



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


<strong style='color: #00008B;'>{'id': '51095b00b074450993385d5576ae6521', 'object': 'chat.completion', 'created': 1771079364, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "<think>\nOkay, let's see, the question is asking for the capital of France. Hmm, I remember that France is a country in Europe. The capital... I think it's Paris. Let me double-check that. Paris is the largest city in France as well, right? The Eiffel Tower is in Paris, which is a very famous landmark. So yes, that makes sense. The capital would probably be Paris. I don't think there's another major city that's the capital. Maybe some people might confuse it with other cities like Marseille or Lyon, but those are big cities but not the capital. Definitely, Paris is the answer here. No need to second-guess this one. The answer should be Paris.\n</think>\nThe capital of France is **Paris**. This vibrant city is not only the nation's political and cultural heart but also home to iconic landmarks like the Eiffel Tower, Louvre Museum, and the Champs-Élysées. 🇫🇷 Paris", 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 232, 'completion_tokens': 206, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



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

    [2026-02-14 14:29:29] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:29:29] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:29:29] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:29:33] INFO server_args.py:1819: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-14 14:29:33] WARNING server_args.py:2353: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-14 14:29:33] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:29:39] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:29:39] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:29:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:29:40] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:29:40] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:29:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:29:46] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:29:46] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:29:46] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.63it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.60it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.65it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.74it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.70it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.10 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.10 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.62it/s]Capturing batches (bs=3 avail_mem=59.37 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.62it/s]Capturing batches (bs=2 avail_mem=59.36 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.62it/s]Capturing batches (bs=2 avail_mem=59.36 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.77it/s]Capturing batches (bs=1 avail_mem=59.34 GB):  75%|███████▌  | 3/4 [00:00<00:00,  4.77it/s]Capturing batches (bs=1 avail_mem=59.34 GB): 100%|██████████| 4/4 [00:00<00:00,  5.02it/s]


    [2026-02-14 14:29:51] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-14 14:29:51] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:02<00:00,  2.05s/it]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:02<00:00,  2.05s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=28.76 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=28.76 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.12s/it]Capturing batches (bs=3 avail_mem=30.81 GB):  25%|██▌       | 1/4 [00:05<00:15,  5.12s/it]

    Capturing batches (bs=3 avail_mem=30.81 GB):  50%|█████     | 2/4 [00:05<00:04,  2.49s/it]Capturing batches (bs=2 avail_mem=30.80 GB):  50%|█████     | 2/4 [00:05<00:04,  2.49s/it]Capturing batches (bs=2 avail_mem=30.80 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.44s/it]Capturing batches (bs=1 avail_mem=30.75 GB):  75%|███████▌  | 3/4 [00:05<00:01,  1.44s/it]

    Capturing batches (bs=1 avail_mem=30.75 GB): 100%|██████████| 4/4 [00:09<00:00,  2.28s/it]Capturing batches (bs=1 avail_mem=30.75 GB): 100%|██████████| 4/4 [00:09<00:00,  2.38s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=30.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=30.64 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=30.63 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=30.63 GB):  75%|███████▌  | 3/4 [00:00<00:00, 23.49it/s]Capturing batches (bs=1 avail_mem=30.61 GB):  75%|███████▌  | 3/4 [00:00<00:00, 23.49it/s]

    Capturing batches (bs=1 avail_mem=30.61 GB): 100%|██████████| 4/4 [00:02<00:00,  1.97it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='95fc6a798cb045c2b4d01f727d229686', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1771079413, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-14 14:30:18] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:30:18] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:30:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:30:47] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.
    [2026-02-14 14:30:47] WARNING server_args.py:2341: Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on.
    [2026-02-14 14:30:47] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:30:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:30:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:30:53] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-14 14:30:53] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:30:53] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:30:53] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:31:01] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:31:01] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:31:01] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.67it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.59it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.58it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.63it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.62it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.78 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.78 GB):  25%|██▌       | 1/4 [00:01<00:04,  1.59s/it]Capturing batches (bs=3 avail_mem=62.07 GB):  25%|██▌       | 1/4 [00:01<00:04,  1.59s/it]Capturing batches (bs=2 avail_mem=62.07 GB):  25%|██▌       | 1/4 [00:01<00:04,  1.59s/it]Capturing batches (bs=2 avail_mem=62.07 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.18it/s]Capturing batches (bs=1 avail_mem=62.06 GB):  75%|███████▌  | 3/4 [00:01<00:00,  2.18it/s]

    Capturing batches (bs=1 avail_mem=62.06 GB): 100%|██████████| 4/4 [00:01<00:00,  2.21it/s]


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]
    


    [2026-02-14 14:31:09] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=56.94 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=56.94 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.41s/it]Capturing batches (bs=3 avail_mem=56.82 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.41s/it]

    Capturing batches (bs=3 avail_mem=56.82 GB):  50%|█████     | 2/4 [00:02<00:02,  1.32s/it]Capturing batches (bs=2 avail_mem=56.81 GB):  50%|█████     | 2/4 [00:02<00:02,  1.32s/it]Capturing batches (bs=1 avail_mem=56.80 GB):  50%|█████     | 2/4 [00:03<00:02,  1.32s/it]

    Capturing batches (bs=1 avail_mem=56.80 GB): 100%|██████████| 4/4 [00:03<00:00,  1.27it/s]Capturing batches (bs=1 avail_mem=56.80 GB): 100%|██████████| 4/4 [00:03<00:00,  1.03it/s]


    [2026-02-14 14:31:15] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='ad99555ef2ff4eb5a794f3c69191b6f0', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1771079480, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-14 14:31:25] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:31:25] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:31:25] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:31:28] INFO server_args.py:1819: Attention backend not specified. Use fa3 backend by default.
    [2026-02-14 14:31:28] WARNING server_args.py:2448: The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding.
    [2026-02-14 14:31:28] INFO server_args.py:2854: Set soft_watchdog_timeout since in CI


    [2026-02-14 14:31:33] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:31:33] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:31:33] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-14 14:31:34] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-14 14:31:34] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-14 14:31:34] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-14 14:31:40] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:31:40] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-14 14:31:40] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.65it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.52it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.51it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.53it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.53it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=44.33 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=44.33 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.54it/s]Capturing batches (bs=3 avail_mem=43.56 GB):  25%|██▌       | 1/4 [00:00<00:01,  1.54it/s]Capturing batches (bs=3 avail_mem=43.56 GB):  50%|█████     | 2/4 [00:00<00:00,  3.01it/s]Capturing batches (bs=2 avail_mem=43.55 GB):  50%|█████     | 2/4 [00:00<00:00,  3.01it/s]Capturing batches (bs=1 avail_mem=43.54 GB):  50%|█████     | 2/4 [00:00<00:00,  3.01it/s]

    Capturing batches (bs=1 avail_mem=43.54 GB): 100%|██████████| 4/4 [00:00<00:00,  5.74it/s]Capturing batches (bs=1 avail_mem=43.54 GB): 100%|██████████| 4/4 [00:00<00:00,  4.35it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='f28d2b2b22c94eec880d4038da4aa1ba', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries along with their capitals:\n\n1. France - Paris\n2. Japan - Tokyo\n3. Brazil - Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1771079525, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=30, prompt_tokens=37, total_tokens=67, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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
