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

    [2026-02-05 07:40:13] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-05 07:40:13] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-05 07:40:13] INFO utils.py:164: NumExpr defaulting to 16 threads.



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

    [2026-02-05 07:40:18] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:40:18] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:40:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 07:40:20] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 07:40:20] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 07:40:20] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:40:27] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:40:27] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:40:27] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 07:40:27] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:40:27] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:40:27] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 07:40:32] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:40:32] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:40:32] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 07:40:32] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.50s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.12s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.22 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.22 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=3 avail_mem=55.14 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=2 avail_mem=55.13 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=1 avail_mem=55.11 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.93it/s]Capturing batches (bs=1 avail_mem=55.11 GB): 100%|██████████| 4/4 [00:00<00:00,  8.79it/s]Capturing batches (bs=1 avail_mem=55.11 GB): 100%|██████████| 4/4 [00:00<00:00,  7.64it/s]


    [2026-02-05 07:40:37] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 07:40:37] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.10s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.10s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.23 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.23 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.72s/it]Capturing batches (bs=3 avail_mem=54.16 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.72s/it]

    Capturing batches (bs=3 avail_mem=54.16 GB):  50%|█████     | 2/4 [00:03<00:02,  1.33s/it]Capturing batches (bs=2 avail_mem=54.14 GB):  50%|█████     | 2/4 [00:03<00:02,  1.33s/it]Capturing batches (bs=2 avail_mem=54.14 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.30it/s]Capturing batches (bs=1 avail_mem=54.11 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.30it/s]

    Capturing batches (bs=1 avail_mem=54.11 GB): 100%|██████████| 4/4 [00:05<00:00,  1.28s/it]Capturing batches (bs=1 avail_mem=54.11 GB): 100%|██████████| 4/4 [00:05<00:00,  1.31s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.09 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=54.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=54.03 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=54.01 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=54.01 GB): 100%|██████████| 4/4 [00:00<00:00, 86.21it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='7f4928d7339240c08b13370c256501ed', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770277252, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-05 07:40:57] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:40:57] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:40:57] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 07:40:59] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 07:40:59] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 07:40:59] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:41:05] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:41:05] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:41:05] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 07:41:05] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:41:05] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:41:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 07:41:11] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:41:11] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:41:11] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 07:41:11] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.43s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.01s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.07s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=55.19 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=55.19 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.79it/s]Capturing batches (bs=3 avail_mem=55.10 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.79it/s]Capturing batches (bs=2 avail_mem=55.08 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.79it/s]

    /usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Capturing batches (bs=2 avail_mem=55.08 GB):  75%|███████▌  | 3/4 [00:07<00:02,  2.63s/it]Capturing batches (bs=1 avail_mem=55.05 GB):  75%|███████▌  | 3/4 [00:07<00:02,  2.63s/it]

    Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:13<00:00,  3.98s/it]Capturing batches (bs=1 avail_mem=55.05 GB): 100%|██████████| 4/4 [00:13<00:00,  3.43s/it]


    [2026-02-05 07:41:29] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 07:41:29] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=54.13 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=54.13 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.25s/it]Capturing batches (bs=3 avail_mem=54.02 GB):  25%|██▌       | 1/4 [00:03<00:09,  3.25s/it]

    Capturing batches (bs=3 avail_mem=54.02 GB):  50%|█████     | 2/4 [00:03<00:03,  1.62s/it]Capturing batches (bs=2 avail_mem=54.00 GB):  50%|█████     | 2/4 [00:03<00:03,  1.62s/it]Capturing batches (bs=2 avail_mem=54.00 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.07it/s]Capturing batches (bs=1 avail_mem=53.95 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.07it/s]

    Capturing batches (bs=1 avail_mem=53.95 GB): 100%|██████████| 4/4 [00:06<00:00,  1.58s/it]Capturing batches (bs=1 avail_mem=53.95 GB): 100%|██████████| 4/4 [00:06<00:00,  1.61s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=53.90 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=53.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=53.83 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=53.81 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=53.81 GB): 100%|██████████| 4/4 [00:00<00:00, 71.20it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='eab699d8a5e6436f9e091f44698cda38', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  Sure! Here are three countries and their capitals:\n\n1. Country: France\nCapital: Paris\n2. Country: Japan\nCapital: Tokyo\n3. Country: Brazil\nCapital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=2)], created=1770277305, model='meta-llama/Llama-2-7b-chat-hf', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=48, prompt_tokens=17, total_tokens=65, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-05 07:41:50] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:41:50] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:41:50] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 07:41:52] WARNING model_config.py:1134: Casting torch.bfloat16 to torch.float16.
    [2026-02-05 07:41:52] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 07:41:52] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 07:41:52] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [2026-02-05 07:41:52] Casting torch.bfloat16 to torch.float16.


    [2026-02-05 07:41:58] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:41:58] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:41:58] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 07:41:58] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:41:58] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:41:58] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:42:00] Casting torch.bfloat16 to torch.float16.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:42:01] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 07:42:03] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:42:03] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:42:03] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 07:42:03] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:04<00:12,  4.08s/it]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:08,  4.13s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:12<00:04,  4.11s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  2.92s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.35s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.19 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.19 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.69it/s]Capturing batches (bs=3 avail_mem=60.02 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.69it/s]Capturing batches (bs=2 avail_mem=60.01 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.69it/s]Capturing batches (bs=2 avail_mem=60.01 GB):  75%|███████▌  | 3/4 [00:00<00:00,  7.33it/s]Capturing batches (bs=1 avail_mem=59.99 GB):  75%|███████▌  | 3/4 [00:00<00:00,  7.33it/s]Capturing batches (bs=1 avail_mem=59.99 GB): 100%|██████████| 4/4 [00:00<00:00,  7.58it/s]


    [2026-02-05 07:42:19] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 07:42:19] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-05 07:42:19] Warning: Target model's context_length (8192) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-05 07:42:19] Overriding the draft model's max_position_embeddings to 8192.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.04it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.04it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.02 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.02 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.74s/it]Capturing batches (bs=3 avail_mem=58.92 GB):  25%|██▌       | 1/4 [00:02<00:08,  2.74s/it]

    Capturing batches (bs=3 avail_mem=58.92 GB):  50%|█████     | 2/4 [00:03<00:02,  1.35s/it]Capturing batches (bs=2 avail_mem=58.88 GB):  50%|█████     | 2/4 [00:03<00:02,  1.35s/it]Capturing batches (bs=2 avail_mem=58.88 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.23it/s]Capturing batches (bs=1 avail_mem=58.84 GB):  75%|███████▌  | 3/4 [00:03<00:00,  1.23it/s]

    Capturing batches (bs=1 avail_mem=58.84 GB): 100%|██████████| 4/4 [00:05<00:00,  1.30s/it]Capturing batches (bs=1 avail_mem=58.84 GB): 100%|██████████| 4/4 [00:05<00:00,  1.33s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.79 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.72 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.70 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.70 GB): 100%|██████████| 4/4 [00:00<00:00, 105.95it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='50c63d623cff4974b92b03c7d735ac5b', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. **France** - **Paris**\n2. **Japan** - **Tokyo**\n3. **Australia** - **Canberra**', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770277354, model='meta-llama/Meta-Llama-3-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=18, total_tokens=57, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-05 07:42:40] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:42:40] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:42:40] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 07:42:42] WARNING model_config.py:1134: Casting torch.bfloat16 to torch.float16.
    [2026-02-05 07:42:42] INFO server_args.py:1796: Attention backend not specified. Use flashinfer backend by default.
    [2026-02-05 07:42:42] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 07:42:42] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    [2026-02-05 07:42:42] Casting torch.bfloat16 to torch.float16.


    [2026-02-05 07:42:48] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:42:48] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:42:48] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 07:42:48] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:42:48] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:42:48] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:42:50] Casting torch.bfloat16 to torch.float16.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:42:50] Casting torch.bfloat16 to torch.float16.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 07:42:53] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:42:53] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:42:53] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 07:42:53] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.00it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:05<00:05,  2.92s/it]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:09<00:03,  3.54s/it]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.79s/it]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:13<00:00,  3.43s/it]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.07 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.07 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.66it/s]Capturing batches (bs=3 avail_mem=59.95 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.66it/s]Capturing batches (bs=2 avail_mem=59.94 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.66it/s]Capturing batches (bs=1 avail_mem=59.92 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.66it/s]Capturing batches (bs=1 avail_mem=59.92 GB): 100%|██████████| 4/4 [00:00<00:00,  9.08it/s]Capturing batches (bs=1 avail_mem=59.92 GB): 100%|██████████| 4/4 [00:00<00:00,  7.68it/s]


    [2026-02-05 07:43:09] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 07:43:09] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    [2026-02-05 07:43:09] Warning: Target model's context_length (131072) is greater than the derived context_length (2048). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
    [2026-02-05 07:43:09] Overriding the draft model's max_position_embeddings to 131072.


    Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.06it/s]
    Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.06it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.76 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.76 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.64s/it]Capturing batches (bs=3 avail_mem=58.71 GB):  25%|██▌       | 1/4 [00:02<00:07,  2.64s/it]

    Capturing batches (bs=3 avail_mem=58.71 GB):  50%|█████     | 2/4 [00:02<00:02,  1.30s/it]Capturing batches (bs=2 avail_mem=58.67 GB):  50%|█████     | 2/4 [00:02<00:02,  1.30s/it]Capturing batches (bs=1 avail_mem=58.63 GB):  50%|█████     | 2/4 [00:03<00:02,  1.30s/it]

    Capturing batches (bs=1 avail_mem=58.63 GB): 100%|██████████| 4/4 [00:05<00:00,  1.12s/it]Capturing batches (bs=1 avail_mem=58.63 GB): 100%|██████████| 4/4 [00:05<00:00,  1.26s/it]


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.57 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=58.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=58.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.48 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=58.48 GB): 100%|██████████| 4/4 [00:00<00:00, 108.15it/s]



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


<strong style='color: #00008B;'>Response: ChatCompletion(id='927d61309e88491294546ea830a9525c', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are 3 countries and their capitals:\n\n1. Country: Japan\n   Capital: Tokyo\n\n2. Country: Australia\n   Capital: Canberra\n\n3. Country: Brazil\n   Capital: Brasília', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=128009)], created=1770277404, model='meta-llama/Meta-Llama-3.1-8B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=43, prompt_tokens=43, total_tokens=86, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    [2026-02-05 07:43:29] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:43:29] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:43:29] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-05 07:43:31] INFO server_args.py:1796: Attention backend not specified. Use fa3 backend by default.
    [2026-02-05 07:43:31] WARNING server_args.py:2304: Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. 
    [2026-02-05 07:43:31] INFO server_args.py:2783: Set soft_watchdog_timeout since in CI


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-02-05 07:43:37] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:43:37] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:43:37] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-05 07:43:37] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-05 07:43:37] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-05 07:43:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-05 07:43:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:43:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-05 07:43:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-02-05 07:43:43] Ignore import error when loading sglang.srt.models.midashenglm: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA version 12.9. Please install the TorchAudio version that matches your PyTorch version.


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.69it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.43it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.37it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.41it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.43it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.12 GB):   0%|          | 0/4 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.12 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.21it/s]Capturing batches (bs=3 avail_mem=61.06 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.21it/s]Capturing batches (bs=2 avail_mem=61.05 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.21it/s]Capturing batches (bs=1 avail_mem=61.04 GB):  25%|██▌       | 1/4 [00:00<00:01,  2.21it/s]Capturing batches (bs=1 avail_mem=61.04 GB): 100%|██████████| 4/4 [00:00<00:00,  8.56it/s]Capturing batches (bs=1 avail_mem=61.04 GB): 100%|██████████| 4/4 [00:00<00:00,  7.04it/s]


    [2026-02-05 07:43:49] SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend
    [2026-02-05 07:43:49] SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend
    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  4.94it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  5.65it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:00<00:00,  5.59it/s]
    


      0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.56 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=3 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=2 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=60.50 GB):   0%|          | 0/4 [00:00<?, ?it/s]Capturing batches (bs=1 avail_mem=60.50 GB): 100%|██████████| 4/4 [00:00<00:00, 61.23it/s]



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


<strong style='color: #00008B;'>{'id': '91664706eafe4f82b23f11ff2855854b', 'object': 'chat.completion', 'created': 1770277440, 'model': 'XiaomiMiMo/MiMo-7B-RL', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<think>\nOkay, so the user is asking, "What is the capital of France?" Let me start by recalling what I know about France. France is a country in Europe, right? It\'s known for things like the Eiffel Tower, the Louvre Museum, and the French language. Now, the capital city... Hmm, I think it\'s Paris. But wait, let me make sure I\'m not mixing it up with any other countries. For instance, the capital of Germany is Berlin, and Italy is Rome. So France\'s capital should be Paris.\n\nWait, maybe I should verify this. Sometimes capitals can be tricky. For example, people might confuse the largest city with the capital. The largest city in France is Paris as well, so that lines up. The President of France resides in Paris, and so do many government buildings. The Louvre, which is a famous museum, is also in Paris. So all these points suggest that Paris is indeed the capital.\n\nAlternatively, could there be a different capital? Let me think. There\'s a region called Île-de-France, which includes Paris, and it\'s the administrative region. The capital of this region is Paris. So yes, that confirms it. I don\'t think there\'s another city that serves as the capital. Maybe some people might think Marseille or LYON, but those are major cities but not the capital. \n\nAnother angle: historical context. Since the French Revolution, Paris has been the political center. The events like the storming of the Bastille happened in Paris. So historically, it\'s been the capital. Also, international organizations like NATO have their headquarters in Brussels, but France\'s own capital is Paris. \n\nI can\'t think of any recent changes that would have moved the capital. Some countries have capitals that aren\'t their largest city, like Berlin and Paris, but in France\'s case, the capital is also the largest city. So all signs point to Paris being the correct answer. \n\nWait, just to be thorough, maybe check if there\'s any official source. The French government\'s website, perhaps. But since I can\'t browse the internet, I have to rely on existing knowledge. From what I know, all indicators confirm that Paris is the capital of France. No contradictions there. \n\nSo the answer should be straightforward. The capital of France is Paris. I don\'t see any ambiguity here. The user might be testing basic knowledge or perhaps trying to trick me, but there\'s no trick involved here. Paris is well-known as the capital. So yes, confident that\'s the correct answer.\n</think>\nThe capital of France is **Paris**. This iconic city is renowned for its cultural landmarks, such as the Eiffel Tower and the Louvre Museum, as well as its political and economic significance. As the heart of the French government and a global hub for art, fashion, and cuisine, Paris remains a symbol of both national and international heritage.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 26, 'total_tokens': 629, 'completion_tokens': 603, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>



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
