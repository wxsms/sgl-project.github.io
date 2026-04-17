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


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 20:20:22] `torch_dtype` is deprecated! Use `dtype` instead!
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.68it/s]


    2026-04-17 20:20:27,246 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 20:20:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.16it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.16it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.16it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 13.16it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 13.16it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.16it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.16it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.16it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.16it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 20.55it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 29.93it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 39.20it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 48.37it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 48.37it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 48.37it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 48.37it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 48.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.42 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.41 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.41 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.41 GB):   3%|▎         | 2/58 [00:00<00:03, 16.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.91it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.39 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.39 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.39 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.93it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=118.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.93it/s]Capturing num tokens (num_tokens=960 avail_mem=118.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.93it/s] Capturing num tokens (num_tokens=960 avail_mem=118.36 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.07it/s]Capturing num tokens (num_tokens=896 avail_mem=118.36 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.07it/s]Capturing num tokens (num_tokens=832 avail_mem=118.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.07it/s]Capturing num tokens (num_tokens=768 avail_mem=118.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.07it/s]Capturing num tokens (num_tokens=704 avail_mem=118.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.07it/s]Capturing num tokens (num_tokens=640 avail_mem=118.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.07it/s]Capturing num tokens (num_tokens=640 avail_mem=118.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.21it/s]Capturing num tokens (num_tokens=576 avail_mem=118.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.21it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.21it/s]Capturing num tokens (num_tokens=480 avail_mem=118.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.21it/s]Capturing num tokens (num_tokens=448 avail_mem=118.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.21it/s]Capturing num tokens (num_tokens=416 avail_mem=118.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.21it/s]Capturing num tokens (num_tokens=416 avail_mem=118.34 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=384 avail_mem=117.08 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=352 avail_mem=116.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=320 avail_mem=116.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=288 avail_mem=116.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=256 avail_mem=116.96 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.38it/s]

    Capturing num tokens (num_tokens=256 avail_mem=116.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=240 avail_mem=116.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=224 avail_mem=116.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=208 avail_mem=116.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=192 avail_mem=116.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=176 avail_mem=116.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=176 avail_mem=116.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=160 avail_mem=116.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=144 avail_mem=116.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=128 avail_mem=116.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=112 avail_mem=116.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.07it/s]

    Capturing num tokens (num_tokens=96 avail_mem=116.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.07it/s] Capturing num tokens (num_tokens=96 avail_mem=116.93 GB):  81%|████████  | 47/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=80 avail_mem=116.93 GB):  81%|████████  | 47/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=64 avail_mem=116.93 GB):  81%|████████  | 47/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=48 avail_mem=116.93 GB):  81%|████████  | 47/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=32 avail_mem=116.92 GB):  81%|████████  | 47/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=28 avail_mem=116.92 GB):  81%|████████  | 47/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=28 avail_mem=116.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=24 avail_mem=116.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=20 avail_mem=116.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=16 avail_mem=116.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.66it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=8 avail_mem=116.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.66it/s] Capturing num tokens (num_tokens=8 avail_mem=116.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=4 avail_mem=116.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=4 avail_mem=116.90 GB): 100%|██████████| 58/58 [00:01<00:00, 38.03it/s]


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
    Generated text:  Fatma. I am a software developer and software designer. What are some tools and technologies I can use to create visually appealing and user-friendly websites and applications? I want to focus on using the latest technologies and also ensuring that the website or application is accessible to people with disabilities. Can you provide me with some insights on how to optimize the performance of a website or application and ensure it is compatible with different devices and screen sizes?
    Sure, I can definitely help you with that. When it comes to creating visually appealing and user-friendly websites and applications, there are several tools and technologies you can use. Here are some of the most popular ones
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. In how many ways can the Vice President be selected as the next candidate for president? To determine the number of ways the Vice President can be selected as the next candidate for president, we need to consider that there are only two candidates: the President and the Vice President. Each candidate can be chosen independently of the other. Therefore, the number of ways to choose the Vice President is simply the number of choices available, which is 1.
    
    Let's break it down step by step:
    
    1. Identify the number of choices available: There are only two candidates: the President and the Vice President.
    2. Determine
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Rome
    C. London
    D. Tokyo
    Answer: A
    
    The main type of construction project contract that appears in domestic and international construction projects is ____.
    A. Construction Contract
    B. Engineering Contract
    C. Labor Contract
    D. Engineering Entrustment Contract
    Answer: A
    
    The service industry is an important driving force for economic development. To meet the demand of the service industry, the government should ____. 
    A. Optimize the service industry
    B. Increase investment in technology
    C. Lower the value of the service industry
    D. Increase investment in labor
    Answer:
    ===============================
    Prompt: The future of AI is
    Generated text:  complex, as it spans multiple fields including computer science, software engineering, and data science. Here are some emerging areas that have shown great potential in AI development:
    
    1. NLP (Natural Language Processing): NLP involves developing systems that can understand and generate human language. This field is closely related to AI, as it requires a deep understanding of how natural language is processed and represented. NLP techniques include machine translation, text classification, sentiment analysis, and chatbots.
    
    2. Computer Vision: Computer vision is a field that involves processing and analyzing visual information. This technology is used in areas such as autonomous vehicles, facial recognition, and augmented


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have a [job title] at [company name]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm always eager to learn and grow, and I'm always willing to help others. What's your favorite hobby or activity? I love [hobby or activity], and I'm always looking for new ways
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many international organizations and is a major hub for business and commerce. It is also known for its cuisine, with dishes like croissants, escargot, and charcuterie being popular. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more efficient and effective AI systems that can make better decisions and solve complex problems.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a growing need for measures to protect user data and prevent unauthorized access. This could include measures such as encryption, access controls,
    


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
    Generated text:  [Name], and I'm a [Skill or Interest] who has been working for [Company Name] for [Number of Years]. I enjoy [What I enjoy doing] and strive to always be the best version of myself. I also have a love for [What I love doing], and am always looking for ways to improve and grow as a person. I'm dedicated to [What I'm dedicated to], and I'm passionate about [What I'm passionate about]. What are your interests or skills? I'm a [Skill or Interest] who has been working for [Company Name] for [Number of Years]. I enjoy [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower stands as a landmark of the world. It is the political, economic, and cultural center of France. The city has a rich history dating back to ancient times, and it is known for its stunning architecture, such as the Louvre and Notre-Dame Cathedral. The city is also home to important museums, theaters, and other cultural institutions. Paris is a popular tourist destination, and it attracts millions of visitors each year. It is the largest city in France by population. The Eiffel Tower is located in the Marais district of Paris, which is the heart of the city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and continues to evolve rapidly. Here are some possible trends that may occur in the AI field:
    
    1. Increased personalization: AI will continue to improve and become more personal, with the ability to tailor services and products to individual preferences and needs.
    
    2. Higher levels of automation: AI will become even more advanced, with the ability to perform tasks with greater efficiency and accuracy than humans.
    
    3. AI will be integrated into more industries: AI will become more widespread and integrated into various industries, from healthcare and finance to manufacturing and transportation.
    
    4. AI will be used for self-driving cars and drones: Self-driving cars and drones will


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

     am

     [

    Age

    ]

     years

     old

    .

     I

     come

     from

     [

    City

     or

     State

    ],

     and

     I

     specialize

     in

     [

    Favorite

     Hobby

    /

    Activity

    ].

     I

    'm

     here

     to

     help

     you

     with

     any

     questions

     you

     have

    ,

     and

     I

    'm

     always

     here

     to

     assist

     you

     in

     any

     way

     I

     can

    .

     What

     can

     I

     do

     for

     you

    ?

     Remember

    ,

     I

     am

     here

     to

     help

     you

     with

     any

     questions

     or

     concerns

     you

     may

     have

    .

     If

     you

     have

     any

     questions

     or

     need

     assistance

    ,

     feel

     free

     to

     ask

    ,

     and

     I

    'll

     do

     my

     best

     to

     assist

     you

    .

     Have

     a

     great

     day

    !

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historical

     and

     cultural

     center

     known

     for

     its

     rich

     history

    ,

     iconic

     landmarks

    ,

     and

     vibrant

     arts

     scene

    .

     The

     city

     is

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     numerous

     art

     galleries

     and

     museums

     that

     showcase

     the

     city

    's

     rich

     cultural

     heritage

    .

     Paris

     is

     also

     famous

     for

     its

     romantic

     romance

    ,

     its

     fashion

     and

     fashion

    -forward

     streets

    ,

     and

     its

     vibrant

     nightlife

    .

     As

     the

     largest

     city

     in

     France

    ,

     Paris

     is

     an

     important

     center

     for

     politics

    ,

     trade

    ,

     and

     culture

    .

     In

     recent

     years

    ,

     the

     city

     has

     undergone

     significant

     urban

     renewal

     and

     modern

    ization

     efforts

    ,

     which

     has

     brought

     about

     a

     new

     appreciation

     and

     respect

     for

     its

     historical

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     highly

     dynamic

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     field

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     other

     industries

    :

     As

     AI

     technologies

     continue

     to

     advance

    ,

     they

     will

     likely

     become

     more

     integrated

     into

     other

     industries

    .

     This

     could

     lead

     to

     a

     wide

     range

     of

     new

     opportunities

     for

     businesses

    ,

     such

     as

     the

     development

     of

     new

     applications

     for

     AI

     in

     healthcare

    ,

     manufacturing

    ,

     and

     transportation

    .
    


    2

    .

     Increased

     emphasis

     on

     ethical

     and

     legal

     considerations

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

     will

     be

     increased

     pressure

     on

     companies

     to

     take

     ethical

     and

     legal

     considerations

     into

     account

    .

     This

     could

     lead

     to

     a

     shift

     in

     the

     way

     we

     think

     about

     AI

     and

     how

     we

     use

    



```python
llm.shutdown()
```
