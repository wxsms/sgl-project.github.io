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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-19 07:26:40] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.10it/s]


    2026-04-19 07:26:44,903 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-19 07:26:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.93it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.02it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.86it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]

    Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.06it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.80it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 37.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.35 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=116.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.45it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.30 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]

    Capturing num tokens (num_tokens=960 avail_mem=116.29 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s] Capturing num tokens (num_tokens=960 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.21it/s]Capturing num tokens (num_tokens=896 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.21it/s]Capturing num tokens (num_tokens=832 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.21it/s]Capturing num tokens (num_tokens=768 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.21it/s]Capturing num tokens (num_tokens=704 avail_mem=116.28 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.21it/s]Capturing num tokens (num_tokens=704 avail_mem=116.28 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.54it/s]Capturing num tokens (num_tokens=640 avail_mem=116.28 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.54it/s]Capturing num tokens (num_tokens=576 avail_mem=116.28 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.54it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.27 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.54it/s]Capturing num tokens (num_tokens=480 avail_mem=116.28 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.54it/s]Capturing num tokens (num_tokens=480 avail_mem=116.28 GB):  52%|█████▏    | 30/58 [00:00<00:00, 33.73it/s]Capturing num tokens (num_tokens=448 avail_mem=116.28 GB):  52%|█████▏    | 30/58 [00:00<00:00, 33.73it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  52%|█████▏    | 30/58 [00:00<00:00, 33.73it/s]Capturing num tokens (num_tokens=384 avail_mem=116.28 GB):  52%|█████▏    | 30/58 [00:00<00:00, 33.73it/s]Capturing num tokens (num_tokens=352 avail_mem=116.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.73it/s]

    Capturing num tokens (num_tokens=352 avail_mem=116.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.79it/s]Capturing num tokens (num_tokens=320 avail_mem=116.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.79it/s]Capturing num tokens (num_tokens=288 avail_mem=116.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.79it/s]Capturing num tokens (num_tokens=256 avail_mem=116.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.79it/s]Capturing num tokens (num_tokens=240 avail_mem=116.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.79it/s]Capturing num tokens (num_tokens=240 avail_mem=116.19 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.07it/s]Capturing num tokens (num_tokens=224 avail_mem=116.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.07it/s]Capturing num tokens (num_tokens=208 avail_mem=116.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.07it/s]

    Capturing num tokens (num_tokens=192 avail_mem=116.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.07it/s]Capturing num tokens (num_tokens=176 avail_mem=116.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.07it/s]Capturing num tokens (num_tokens=176 avail_mem=116.18 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=160 avail_mem=116.17 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=144 avail_mem=116.17 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=128 avail_mem=116.17 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=112 avail_mem=116.16 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=112 avail_mem=116.16 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=96 avail_mem=116.16 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=116.16 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=64 avail_mem=116.15 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=48 avail_mem=116.15 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=48 avail_mem=116.15 GB):  86%|████████▌ | 50/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=32 avail_mem=116.14 GB):  86%|████████▌ | 50/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=28 avail_mem=116.14 GB):  86%|████████▌ | 50/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=24 avail_mem=116.14 GB):  86%|████████▌ | 50/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=20 avail_mem=116.13 GB):  86%|████████▌ | 50/58 [00:01<00:00, 32.00it/s]

    Capturing num tokens (num_tokens=20 avail_mem=116.13 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=16 avail_mem=116.13 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=12 avail_mem=116.08 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=8 avail_mem=116.08 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.33it/s] Capturing num tokens (num_tokens=4 avail_mem=116.07 GB):  93%|█████████▎| 54/58 [00:01<00:00, 31.33it/s]Capturing num tokens (num_tokens=4 avail_mem=116.07 GB): 100%|██████████| 58/58 [00:01<00:00, 32.52it/s]Capturing num tokens (num_tokens=4 avail_mem=116.07 GB): 100%|██████████| 58/58 [00:01<00:00, 31.56it/s]


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
    Generated text:  Dan and I'm in the meatpacking industry. I have been working in the industry for 30 years. I've been through many iterations of the industry, from small family owned operations to big box stores to malls to supermarkets. I've been through many seasons of hurricanes and flooding, but the industry has come a long way and I'm pleased to say that it's looking better now than it did 10 years ago. I have a degree in mechanical engineering from North Carolina State University. I have had a real job, real experience in the industry and I'm proud to be involved in this. I'm pleased that I am
    ===============================
    Prompt: The president of the United States is
    Generated text:  a three-term president who takes office on January 20, 2022. The president has been serving for 5 years. How many days have passed since the president took office?
    
    To determine the number of days that have passed since the president took office, we need to calculate the total number of days in the presidency of the United States from January 20, 2022, to December 31, 2022.
    
    1. **Calculate the number of days in each year:**
       - **2020:**
         - January has 31 days.
         -
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris B: London C: Rome D: Berlin
    
    The capital of France is:
    
    A: Paris
    
    Therefore, the correct answer is: A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and complex. As technology continues to advance, the future of AI is bright with the potential to revolutionize industries, improve our lives, and even shape the future of humanity. However, there are many challenges that must be addressed to ensure that AI is used responsibly and ethically. Here are some of the key challenges that need to be addressed to ensure the future of AI is bright:
    
    1. Ethical considerations: One of the biggest challenges of AI is its ethical considerations. There are concerns about the potential misuse of AI, such as bias, transparency, and privacy issues. The development of AI should be guided by ethical principles, and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What do you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many world-renowned museums and attractions. Paris is a bustling metropolis with a rich history and a diverse population, making it a popular tourist destination. The city is also known for its cuisine, with many famous French dishes and restaurants serving up delicious meals. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development
    


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
    Generated text:  [Your Name], and I'm [Your Age]. I'm a [Your Profession or Role] [Your Position]. I'm always trying to [Your Main Goal]. And I'm always up for [Your Challenge]. I enjoy [Your Hobby or Passion], and I think it's [Your Strength or Qualification]. I'm also a [Your Last Name] [Your Character Name]. And I'm a [Your Last Name] [Your Character Name]. My style is [Your Personality]. I'm [Your Skill Level]. I'm always looking for [Your Next Step] and always ready to learn. I'm [Your Enthus
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historical, cultural, and artistic center that is known for its landmarks such as the Eiffel Tower and the Louvre Museum. It is also a world-renowned center of business, finance, and education. Paris is also famous for its fashion and art scene. The city is a hub for world-class museums, including the Louvre, the Metropolitan Museum of Art, and the Centre Pompidou. Paris is a popular tourist destination and has a long and rich history. It is known for its diverse culture and cuisine. Paris has a vibrant nightlife scene and is a popular destination for tourists and locals alike. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and transformative, with many exciting developments on the horizon. Here are some possible trends we can expect:
    
    1. Enhanced accuracy and reliability: As AI technology improves, we can expect to see more accurate and reliable AI systems that perform tasks more accurately than human workers. This could lead to a more efficient and cost-effective way of performing complex tasks.
    
    2. AI will become more capable of complex decision-making: AI will become more capable of making complex decisions based on big data and analyzing large amounts of information. This could lead to more efficient and effective ways of decision-making.
    
    3. AI will be used in more areas of human life: AI


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

     [

    Name

    ],

     and

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Occup

    ation

    ].

     I

    've

     always

     had

     a

     passion

     for

     [

    occupation

    ],

     and

     I

    've

     dedicated

     many

     years

     to

     [

    occupation

    ]

     in

     my

     spare

     time

    .

     Despite

     not

     having

     a

     formal

     education

    ,

     I

    'm

     confident

     in

     my

     skills

     and

     have

     hon

    ed

     my

     abilities

     through

     hard

     work

    .

     I

    'm

     also

     very

     adaptable

     and

     can

     handle

     different

     situations

    ,

     whether

     they

    're

     professional

     or

     personal

    .

     I

     believe

     in

     the

     power

     of

     persistence

     and

     my

     unw

    av

    ering

     commitment

     to

     my

     goals

    .

     I

    'm

     excited

     to

     be

     a

     part

     of

     this

     community

     and

     contribute

     to

     its

     growth

     and

     success

    .

     Is

     there

     anything

     I

     can

     do

     for

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     known

     for

     its

     cultural

     richness

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     nightlife

    .

     The

     city

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     many

     other

     famous

     landmarks

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     romantic

     and

     historical

     atmosphere

    .

     The

     city

     is

     also

     home

     to

     many

     important

     museums

    ,

     such

     as

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

    ,

     and

     the

     Mus

    ée

     national

     d

    '

    art

     moderne

    .

     Paris

     is

     the

     heart

     of

     France

     and

     one

     of

     the

     most

     iconic

     cities

     in

     the

     world

    .

     Its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

     and

     involves

     a

     wide

     range

     of

     new

     technologies

    ,

     trends

    ,

     and

     possibilities

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

     Enhanced

     Artificial

     Intelligence

    :

     One

     of

     the

     most

     promising

     future

     trends

     in

     AI

     is

     the

     development

     of

     enhanced

     AI

    ,

     which

     can

     achieve

     human

    -level

     intelligence

     through

     deeper

     learning

     and

     neural

     networks

    .

     This

     could

     lead

     to

     applications

     such

     as

     self

    -driving

     cars

    ,

     medical

     diagnostics

    ,

     and

     more

     advanced

     forms

     of

     artificial

     general

     intelligence

    .
    


    2

    .

     Personal

    ization

    :

     Personal

    ization

     is

     becoming

     increasingly

     important

     in

     AI

    ,

     as

     it

     allows

     machines

     to

     tailor

     their

     responses

     and

     actions

     to

     the

     individual

     user

    .

     This

     could

     lead

     to

     more

     accurate

     and

     personalized

     healthcare

    ,

     more

     efficient

     business

     processes

    ,

     and

    



```python
llm.shutdown()
```
