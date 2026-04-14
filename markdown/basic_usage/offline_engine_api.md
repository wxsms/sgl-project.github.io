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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.54it/s]


    2026-04-14 13:24:43,466 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 13:24:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:42,  2.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:42,  2.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:42,  2.84s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:42,  2.84s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:42,  2.84s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.34it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.22it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 28.80it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 46.55it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 46.55it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 46.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.35it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.03it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.01it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.01it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.01it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.01it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.73it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.73it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.73it/s]

    Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.73it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.73it/s]Capturing num tokens (num_tokens=512 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.73it/s]Capturing num tokens (num_tokens=512 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.71it/s]

    Capturing num tokens (num_tokens=288 avail_mem=120.25 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=256 avail_mem=120.25 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=240 avail_mem=120.25 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.44it/s]

    Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.95it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.23it/s]

    Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 29.00it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 29.00it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 29.00it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 29.00it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 29.00it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 30.75it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 30.75it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.75it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:02<00:00, 28.39it/s]


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
    Generated text:  Alex and I am a beginner in JavaScript. I am currently learning it and working on a project that involves manipulating HTML files. Can you provide some tips on how to optimize JavaScript code to improve performance and minimize memory usage in this scenario? Sure, I'd be happy to provide some tips on optimizing JavaScript code to improve performance and minimize memory usage. Here are a few suggestions:
    
    1. Use ES6 syntax: ES6 is a new JavaScript version that is widely supported and has some significant changes that can improve performance and reduce memory usage. ES6 introduces a new syntax for function declarations, arrow functions, and class definitions, among others, which
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to make the White House seriously overcrowded. He is considering two plans: plan A, where he installs 200 more people, or plan B, where he installs 300 people. If he installs more than 400 people, he decides to cut the number of people by 20. What is the minimum number of people he can have in the White House? To determine the minimum number of people in the White House under the given conditions, we need to analyze the two plans and consider the constraints provided.
    
    First, let's consider the two plans:
    1. Install 20
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the number of Paris?
    To determine the number of Paris, we need to understand the context provided. The question states that the capital of France is Paris. Therefore, the number of Paris is not specified in the given information.
    
    However, if we interpret the question as asking for the number of capitals in the world, then the number of capitals would be the same as the number of countries in the world.
    
    The number of countries in the world is approximately 194.
    
    Given this, the answer to the original question is:
    
    \[
    \boxed{194}
    \]
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, as depicted by the data from the latest O'Reilly AI Report.
    
    AI will grow at an incredible rate over the next few years. Currently, one in every 10 people will be connected to an AI device. With the increasing use of AI, and the advancing technology, the rate of growth of AI is expected to continue to rise, leading to the next wave of change in society.
    
    However, with the potential of AI, comes a huge potential for misinformation and harm. As AI becomes more advanced, it's easy for those in power to utilize AI to manipulate people and spread misinformation, and it's up to the people


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many museums, art galleries, and theaters, and is a major center for business and finance. It is also known for its cuisine, with many famous French dishes such as croissants, boudin, and escargot. Paris is a city of contrasts, with its modern architecture and historical landmarks blending
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence in the future, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to the needs of their users.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for
    


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
    Generated text:  [Name], and I am [Age]. I am a [Occupation] and I enjoy [What interests me or the hobbies you have]. I have been [How long you have been in the industry], and I have been [How many years of experience in the industry]. I am the [Industry of interest], and I have been [Experience in the industry] years. I am passionate about [Why you are passionate about the industry]. I have a desire to [What you want to achieve in your career]. As a [Optional: Educator, coach, mentor, etc.], I would like to be [What kind of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located in the south of the country. It is the largest city in terms of population, with a population of over 2 million people. Paris is known for its rich cultural heritage, landmarks such as the Eiffel Tower and the Louvre Museum, and its bustling nightlife. The city is also famous for its fashion industry, with many iconic fashion houses located in the city. Paris is a bustling metropolis with a rich history and a lively culture. Its status as the capital is an important part of French identity and is reflected in its architecture, cuisine, and fashion.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, and it will likely continue to evolve rapidly, influenced by a wide range of factors, including technological advancements, societal shifts, and changing priorities. Here are some possible future trends in artificial intelligence:
    
    1. Increased use of AI in healthcare: AI is expected to play a key role in improving the accuracy and speed of diagnoses and treatment. This could lead to more personalized and effective treatments for a wide range of conditions.
    
    2. AI in manufacturing: AI is expected to play a growing role in manufacturing, automating repetitive tasks and optimizing processes to increase efficiency and productivity.
    
    3. AI in education: AI is already being used in education to


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

    Your

     Name

    ],

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Education

    ]

     degree

    .

     I

     have

     always

     had

     a

     keen

     interest

     in

     the

     arts

     and

     have

     been

     passionate

     about

     exploring

     new

     ideas

     and

     creating

     innovative

     projects

    .

     I

     am

     always

     looking

     to

     learn

     and

     grow

    ,

     and

     I

     am

     always

     up

     for

     trying

     new

     things

    .

     If

     you

    're

     looking

     to

     connect

     and

     have

     a

     conversation

    ,

     let

    's

     chat

    !

     [

    Your

     Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     French

     capital

    ,

     is

     home

     to

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

    ,

     among

     many

     other

     world

    -ren

    owned

     landmarks

    .

     It

     has

     a

     rich

     history

     and

     is

     known

     for

     its

     vibrant

     culture

    ,

     fine

     art

    ,

     and

     historic

     architecture

    .

     It

     has

     been

     a

     major

     trading

     center

     for

     centuries

     and

     is

     considered

     one

     of

     the

     world

    's

     most

     important

     cities

    .

     Paris

     is

     also

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     major

     center

     of

     the

     French

     economy

    .

     Its

     unique

     blend

     of

     old

     and

     new

    ,

     rich

     and

     modern

    ,

     attracts

     millions

     of

     tourists

     each

     year

    .

     The

     city

     is

     an

     important

     center

     of

     learning

    ,

     science

    ,

     and

     philosophy

    ,

     and

     continues

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     rapid

     expansion

     of

     its

     application

     in

     various

     industries

     and

     fields

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     likely

     become

     more

     integrated

     into

     various

     industries

    ,

     such

     as

     manufacturing

    ,

     healthcare

    ,

     transportation

    ,

     and

     finance

    .

     Automation

     will

     lead

     to

     increased

     efficiency

    ,

     reduced

     costs

    ,

     and

     improved

     quality

     of

     service

    .
    


    2

    .

     AI

    -powered

     voice

     assistants

    :

     AI

    -powered

     voice

     assistants

     like

     Amazon

    's

     Alexa

    ,

     Google

     Assistant

    ,

     and

     Siri

     will

     continue

     to

     become

     more

     prevalent

    ,

     replacing

     humans

     in

     many

     areas

     of

     daily

     life

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     will

     be

     used

     to

     analyze

     medical

     data

    ,

     diagnose

     diseases

    ,

     and

     develop

     personalized

     treatment

    



```python
llm.shutdown()
```
