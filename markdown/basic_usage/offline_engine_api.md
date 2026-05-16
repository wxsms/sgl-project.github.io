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
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


    2026-05-16 15:55:35,344 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 15:55:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.62it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.39 GB):   3%|▎         | 2/58 [00:00<00:03, 14.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.02it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.38 GB):   7%|▋         | 4/58 [00:00<00:03, 14.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.38 GB):   7%|▋         | 4/58 [00:00<00:03, 14.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.38 GB):   7%|▋         | 4/58 [00:00<00:03, 14.14it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=59.38 GB):  10%|█         | 6/58 [00:00<00:04, 11.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.37 GB):  10%|█         | 6/58 [00:00<00:04, 11.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.36 GB):  10%|█         | 6/58 [00:00<00:04, 11.87it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.36 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.36 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.36 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.36 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.35 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.35 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.35 GB):  21%|██        | 12/58 [00:01<00:04, 10.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.35 GB):  21%|██        | 12/58 [00:01<00:04, 10.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.35 GB):  21%|██        | 12/58 [00:01<00:04, 10.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=59.35 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.34 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.34 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.34 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.33 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.13it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=59.33 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.33 GB):  31%|███       | 18/58 [00:01<00:03, 10.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.33 GB):  31%|███       | 18/58 [00:01<00:03, 10.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.33 GB):  31%|███       | 18/58 [00:01<00:03, 10.97it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=59.33 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.31 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.33it/s]Capturing num tokens (num_tokens=960 avail_mem=59.32 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.33it/s] Capturing num tokens (num_tokens=960 avail_mem=59.32 GB):  38%|███▊      | 22/58 [00:01<00:03, 11.67it/s]Capturing num tokens (num_tokens=896 avail_mem=59.32 GB):  38%|███▊      | 22/58 [00:01<00:03, 11.67it/s]

    Capturing num tokens (num_tokens=832 avail_mem=59.32 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.67it/s]Capturing num tokens (num_tokens=832 avail_mem=59.32 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.00it/s]Capturing num tokens (num_tokens=768 avail_mem=59.31 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.00it/s]Capturing num tokens (num_tokens=704 avail_mem=59.31 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.00it/s]

    Capturing num tokens (num_tokens=704 avail_mem=59.31 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.43it/s]Capturing num tokens (num_tokens=640 avail_mem=59.31 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.43it/s]Capturing num tokens (num_tokens=576 avail_mem=59.31 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.43it/s]Capturing num tokens (num_tokens=576 avail_mem=59.31 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.39it/s]Capturing num tokens (num_tokens=512 avail_mem=59.29 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.39it/s]

    Capturing num tokens (num_tokens=480 avail_mem=59.31 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.39it/s]Capturing num tokens (num_tokens=480 avail_mem=59.31 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.91it/s]Capturing num tokens (num_tokens=448 avail_mem=59.30 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.91it/s]Capturing num tokens (num_tokens=416 avail_mem=59.30 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.91it/s]

    Capturing num tokens (num_tokens=416 avail_mem=59.30 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.40it/s]Capturing num tokens (num_tokens=384 avail_mem=59.30 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.40it/s]Capturing num tokens (num_tokens=352 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.40it/s]Capturing num tokens (num_tokens=352 avail_mem=59.29 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.38it/s]Capturing num tokens (num_tokens=320 avail_mem=59.29 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.38it/s]Capturing num tokens (num_tokens=288 avail_mem=59.29 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.38it/s]

    Capturing num tokens (num_tokens=256 avail_mem=59.28 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.38it/s]Capturing num tokens (num_tokens=256 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:02<00:01, 15.30it/s]Capturing num tokens (num_tokens=240 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:02<00:01, 15.30it/s]Capturing num tokens (num_tokens=224 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:03<00:01, 15.30it/s]Capturing num tokens (num_tokens=224 avail_mem=59.28 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.11it/s]Capturing num tokens (num_tokens=208 avail_mem=59.27 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.11it/s]

    Capturing num tokens (num_tokens=192 avail_mem=59.27 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.11it/s]Capturing num tokens (num_tokens=176 avail_mem=59.27 GB):  67%|██████▋   | 39/58 [00:03<00:01, 16.11it/s]Capturing num tokens (num_tokens=176 avail_mem=59.27 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=160 avail_mem=59.27 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=144 avail_mem=59.26 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=128 avail_mem=59.26 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.18it/s]Capturing num tokens (num_tokens=128 avail_mem=59.26 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.58it/s]Capturing num tokens (num_tokens=112 avail_mem=59.26 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.58it/s]

    Capturing num tokens (num_tokens=96 avail_mem=59.25 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.58it/s] Capturing num tokens (num_tokens=80 avail_mem=59.25 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.58it/s]Capturing num tokens (num_tokens=80 avail_mem=59.25 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.27it/s]Capturing num tokens (num_tokens=64 avail_mem=59.25 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.27it/s]Capturing num tokens (num_tokens=48 avail_mem=59.24 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.27it/s]Capturing num tokens (num_tokens=32 avail_mem=59.24 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.27it/s]Capturing num tokens (num_tokens=28 avail_mem=59.23 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.27it/s]

    Capturing num tokens (num_tokens=28 avail_mem=59.23 GB):  90%|████████▉ | 52/58 [00:03<00:00, 24.63it/s]Capturing num tokens (num_tokens=24 avail_mem=59.23 GB):  90%|████████▉ | 52/58 [00:03<00:00, 24.63it/s]Capturing num tokens (num_tokens=20 avail_mem=59.23 GB):  90%|████████▉ | 52/58 [00:03<00:00, 24.63it/s]Capturing num tokens (num_tokens=16 avail_mem=59.23 GB):  90%|████████▉ | 52/58 [00:03<00:00, 24.63it/s]Capturing num tokens (num_tokens=12 avail_mem=59.22 GB):  90%|████████▉ | 52/58 [00:03<00:00, 24.63it/s]Capturing num tokens (num_tokens=12 avail_mem=59.22 GB):  97%|█████████▋| 56/58 [00:03<00:00, 28.33it/s]Capturing num tokens (num_tokens=8 avail_mem=59.22 GB):  97%|█████████▋| 56/58 [00:03<00:00, 28.33it/s] Capturing num tokens (num_tokens=4 avail_mem=59.21 GB):  97%|█████████▋| 56/58 [00:03<00:00, 28.33it/s]Capturing num tokens (num_tokens=4 avail_mem=59.21 GB): 100%|██████████| 58/58 [00:03<00:00, 15.63it/s]


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
    Generated text:  Kevin. I'm a Mexican-American, 5'10", 170 pounds. I have been working out and eating a healthy diet since childhood. I love trying out new food, trying to eat healthier, and being a good friend to others. I have also been very social and have a sense of humor. I love attending parties, singing, and playing music. I also like to read books and I like to eat ice cream.
    
    I love to keep up with the news and sports so I can stay current on what's happening in the world. I love to travel to places I've never been to before. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to host a press conference in the Johnson Space Center. If it is Friday, the president will go to a conference center and will be in the Johnson Space Center from 10:00 AM to 5:00 PM. Otherwise, he will be in the Johnson Space Center from 12:00 AM to 5:00 PM. On Wednesday, Friday, and Tuesday, he will be in the Johnson Space Center from 10:00 AM to 5:00 PM. If he attends the conference center, what is the probability that it is Tuesday? To determine the probability
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. London
    C. Rome
    D. Moscow
    Answer:
    
    A
    
    The capital of France is ____.
    A. Paris
    B. London
    C. Rome
    D. Moscow
    Answer:
    
    A
    
    The capital of France is: A. Paris B. London C. Rome D. Moscow
    Answer:
    
    A
    
    The capital of France is ( ). A. Paris B. London C. Rome D. Moscow
    Answer:
    
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Rome
    D. Moscow
    Answer:
    
    A
    
    Which of the following is NOT
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be the future
    
    Since the dawn of modern technology, the influence of AI has been profound. From the original computer languages that people created for the first electronic devices, to now, the influence of AI is beyond any doubt. With the rise of machine learning and artificial intelligence, people have been able to create systems that are incredibly intelligent and able to accomplish complex tasks that would have been difficult, if not impossible, for humans to do.
    
    In this post, we will examine the current state of AI, what it will be like in the future, and the questions that the field will have to answer in order to achieve the best


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the country and the seat of government and culture. It is also the world's most populous city, with an estimated population of over 2. 5 million people. Paris is known for its rich history, beautiful architecture, and vibrant culture, and is a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its cuisine, including its famous French fries and its famous cheese, brie. The city is a hub for business and commerce, with many international companies and institutions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed
    


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
    Generated text:  [name], and I’m a [career] specializing in [main activity]. I’m always looking for ways to [describe your main activity]. Whether it’s [job title], [occupation], or [field of expertise], I’m here to help you find what you need. If you have any questions, please don’t hesitate to reach out. And remember, I’m always here to assist you in your journey. Let’s connect. [Name]. Hello! My name is [name], and I’m a [career] specializing in [main activity]. I’m always looking for ways to [describe your main activity]. Whether it’s
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    This is a factual statement about the capital city of France. It is a common fact that Paris is the capital and largest city of France, known for its cultural, historical, and artistic richness. Its name translates to "Paris" in Latin, which means "Paris" in French. The city is located in the western part of France and is the most populous city in the country, with a population of over 20 million. Paris is also known for its famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Montmartre. Its beauty is often compared to that of the Italian city of Florence
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising, and there are many trends that are likely to shape its trajectory in the coming years. Here are some possible future trends in AI:
    
    1. Increased specialization and specialization: AI will become more specialized, with each AI system being better at a specific task than the one before it. This will lead to a more efficient and effective use of resources in various industries.
    
    2. Autonomous vehicles: With the development of AI, autonomous vehicles will become more common. These vehicles will be equipped with AI systems that can navigate traffic, detect obstacles, and communicate with other vehicles in real-time.
    
    3. Improved healthcare: AI will be used to improve


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

     Sarah

    .

     I

     am

     a

     writer

     and

     illustrator

    ,

     passionate

     about

     using

     art

     to

     tell

     stories

    .

     I

     have

     been

     working

     as

     a

     freelance

     artist

     for

     over

     a

     decade

    ,

     and

     my

     most

     recent

     project

     is

     a

     series

     of

     illustrations

     for

     a

     children

    's

     book

     about

     a

     magical

     forest

    .

     What

    's

     your

     favorite

     hobby

     or

     interest

     outside

     of

     art

     and

     writing

    ?

     Writing

    !

     I

     love

     exploring

     new

     ideas

     and

     trying

     out

     different

     genres

    .

     
    


    I

    'm

     always

     looking

     for

     opportunities

     to

     learn

     new

     things

     and

     inspire

     others

     with

     my

     work

    .

     What

    's

     your

     favorite

     book

     or

     movie

    ?

     "

    1

    0

     Things

     I

     Hate

     About

     You

    "

     is

     my

     all

    -time

     favorite

     movie

    .

     It

    's

     a

     classic

     comedy

     that

     keeps

     me

     laughing

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     western

     coast

     of

     the

     continent

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     and

     culture

    ,

     and

     is

     home

     to

     numerous

     world

    -ren

    owned

     attractions

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

    's

     unique

     atmosphere

     and

     food

     scene

     make

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     is

     also

     known

     for

     its

     distinctive

     music

     scene

     and

     festivals

    ,

     such

     as

     the

     Les

     Festival

     de

     Cannes

    .

     Its

     status

     as

     a

     global

     cultural

     hub

     has

     made

     it

     a

     major

     center

     for

     politics

    ,

     business

    ,

     and

     art

    ,

     making

     it

     a

     vital

     part

     of

     French

     society

    .

     The

     French

     government

     is

     responsible

     for

     the

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     growing

     convergence

     with

     other

     technologies

    ,

     including

     machine

     learning

     and

     quantum

     computing

    ,

     which

     will

     drive

     new

     advances

     in

     the

     area

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     emphasis

     on

     AI

     ethics

     and

     privacy

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     is

     a

     growing

     need

     for

     ethical

     guidelines

     and

     regulations

     to

     ensure

     that

     AI

     systems

     are

     used

     responsibly

     and

     not

     causing

     harm

     to

     individuals

     or

     society

    .

     This

     will

     likely

     lead

     to

     more

     focus

     on

     AI

     ethics

     and

     privacy

    .
    


    2

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     is

     likely

     to

     be

     a

     growing

     need

     for

     better

     natural

     language

    



```python
llm.shutdown()
```
