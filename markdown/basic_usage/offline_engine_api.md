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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]


    2026-05-13 08:54:40,374 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 08:54:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.03it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 15.03it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 15.03it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 15.03it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 15.03it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.95it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=51.30 GB):   2%|▏         | 1/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.27 GB):   2%|▏         | 1/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.22 GB):   2%|▏         | 1/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.22 GB):   2%|▏         | 1/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.22 GB):   7%|▋         | 4/58 [00:00<00:04, 11.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.22 GB):   7%|▋         | 4/58 [00:00<00:04, 11.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.21 GB):   7%|▋         | 4/58 [00:00<00:04, 11.34it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=51.20 GB):   7%|▋         | 4/58 [00:00<00:04, 11.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.20 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.20 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.20 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=51.20 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.19 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.18 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.03it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=51.18 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.18 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.18 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.18 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.17 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.17 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.17 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.16 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.16 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.15 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=960 avail_mem=51.16 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.37it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=51.16 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=832 avail_mem=51.15 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=832 avail_mem=51.15 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.28it/s]Capturing num tokens (num_tokens=768 avail_mem=51.15 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.28it/s]Capturing num tokens (num_tokens=704 avail_mem=51.15 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.28it/s]Capturing num tokens (num_tokens=640 avail_mem=51.14 GB):  41%|████▏     | 24/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=576 avail_mem=51.14 GB):  41%|████▏     | 24/58 [00:01<00:00, 34.28it/s]Capturing num tokens (num_tokens=576 avail_mem=51.14 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=512 avail_mem=51.13 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.10it/s]

    Capturing num tokens (num_tokens=480 avail_mem=51.14 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=448 avail_mem=51.14 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=416 avail_mem=51.14 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=416 avail_mem=51.14 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=384 avail_mem=51.14 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=352 avail_mem=51.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=320 avail_mem=51.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.79it/s]

    Capturing num tokens (num_tokens=288 avail_mem=51.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=288 avail_mem=51.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 24.50it/s]Capturing num tokens (num_tokens=256 avail_mem=51.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 24.50it/s]

    Capturing num tokens (num_tokens=240 avail_mem=51.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 24.50it/s]Capturing num tokens (num_tokens=224 avail_mem=51.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 24.50it/s]Capturing num tokens (num_tokens=224 avail_mem=51.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 20.30it/s]Capturing num tokens (num_tokens=208 avail_mem=51.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 20.30it/s]Capturing num tokens (num_tokens=192 avail_mem=51.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 20.30it/s]Capturing num tokens (num_tokens=176 avail_mem=51.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 20.30it/s]

    Capturing num tokens (num_tokens=176 avail_mem=51.11 GB):  72%|███████▏  | 42/58 [00:01<00:00, 21.01it/s]Capturing num tokens (num_tokens=160 avail_mem=51.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 21.01it/s]Capturing num tokens (num_tokens=144 avail_mem=51.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 21.01it/s]Capturing num tokens (num_tokens=128 avail_mem=51.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 21.01it/s]Capturing num tokens (num_tokens=112 avail_mem=51.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 21.01it/s]Capturing num tokens (num_tokens=112 avail_mem=51.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 24.74it/s]Capturing num tokens (num_tokens=96 avail_mem=51.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 24.74it/s] Capturing num tokens (num_tokens=80 avail_mem=51.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 24.74it/s]Capturing num tokens (num_tokens=64 avail_mem=51.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 24.74it/s]Capturing num tokens (num_tokens=48 avail_mem=51.08 GB):  79%|███████▉  | 46/58 [00:02<00:00, 24.74it/s]

    Capturing num tokens (num_tokens=32 avail_mem=51.08 GB):  79%|███████▉  | 46/58 [00:02<00:00, 24.74it/s]Capturing num tokens (num_tokens=32 avail_mem=51.08 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=28 avail_mem=51.07 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.90it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.90it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.90it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.90it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.90it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:02<00:00, 24.99it/s]


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
    Generated text:  A. You can call me Alex. I'm a computer science student in the U. S. A. I'm wondering if you've heard of The Big Bang Theory? I really like the show. I think it's a really funny show. And I hope to see it when it's on again. My name is a smart person. I'm smart enough to know that you're smart and that you have a better memory than I do. That's why I can say things that aren't even true. I mean, I'm smarter than most people. I know I'm right, but I'm just saying. I like you
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a vice president. If there are 10 vice presidents in the executive branch, and assuming each vice president has 4 terms to serve, how many total terms have been served by the vice presidents? To determine the total number of terms that have been served by the vice presidents, we need to follow these steps:
    
    1. Identify the number of vice presidents in the executive branch.
    2. Determine the number of terms each vice president serves.
    3. Multiply the number of vice presidents by the number of terms each serves.
    
    Given:
    - There are 10 vice presidents in the executive branch.
    - Each vice president has 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city is located in the north-central area of the country, at the foot of the Paris hill in the Seine valley. It is the largest city in France and the third-largest in the world. It is also the capital of the Île-de-France region. Paris is the third largest city by population after Beijing and Shanghai in the Chinese city.
    Is there an answer to this question (If it cannot be answered, return "Unanswerable"). Is there an answer to this question (If it cannot be answered, return "Unanswerable"). Is Paris the third largest city by population in the world? Unanswer
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is not all rosy. While it is hard to predict the end of the future of AI, some of the most promising areas are becoming more and more advanced. As a result, the need for AI developers has increased dramatically. With the right skills and knowledge, anyone can become a successful AI developer and create innovative AI systems that can benefit the world.
    In this article, we will explore the areas of AI that are most promising for the future, and provide some insights into what those areas might look like. We will also discuss the skills and qualifications that are required to become an AI developer, as well as some of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is also known for its cuisine, fashion, and art scene. Paris is a popular tourist destination and a cultural hub for France and the world. It is a city that has been a center of power and culture for centuries and continues to be a major economic and political center in the world. Paris is a city of contrasts, with its rich history and modernity blending together
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing AI that is designed to be ethical and responsible. This could include developing AI that is designed to minimize harm to individuals and society as a whole, and that is transparent and accountable.
    
    2. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used
    


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
    Generated text:  [Name], and I'm a [occupation] with [number of years in the industry]. What brings you to this scene?
    
    As a [occupation], my passion is [occupation]. I have always been fascinated by [occupation], and [mention a specific achievement, achievement, or reason for your success] has always inspired me to keep pursuing this field. In my free time, I enjoy [mention a hobby or activity].
    
    What brings you to this scene? How do you plan to make a difference in the world?
    Hello, my name is [Name], and I'm a [occupation] with [number of years in the industry].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a large and diverse city with a rich history and cultural heritage, known for its iconic landmarks, museums, and world-class attractions. The city is home to over 3.5 million people and is one of the most important cities in the world in terms of commerce, finance, and arts. Paris is also a center for education, research, and cultural institutions, and has been a major hub for politics, religion, and popular culture since the 13th century. It is an important center of global trade and diplomacy, and is home to the Eiffel Tower, Louvre Museum, and Notre-Dame
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and promising, with many exciting trends in the field. Here are some of the most promising trends:
    
    1. Increased AI Transparency: As AI systems become more complex, it's becoming increasingly important for developers to explain their decisions and operations to the public. This will be especially important in the context of AI ethics, where transparency is essential to ensure that AI systems are aligned with ethical principles.
    
    2. Enhanced AI Ethics: As AI systems become more integrated into our daily lives, there will be a greater need for ethical considerations. For example, AI systems that are used to predict crime patterns could be used to monitor and intervene in crime scenes,


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

    ]

     and

     I

    'm

     a

     [

    Position

    ]

     at

     [

    Company

    /

    Inc

    ].

     I

    'm

     a

     [

    Professional

     Title

    ]

     with

     a

     passion

     for

     [

    Skill

    /

    Field

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     [

    T

    alent

     Development

    ],

     to

     expand

     my

     knowledge

     and

     improve

     my

     skills

    .

     I

    'm

     eager

     to

     contribute

     to

     the

     company

    's

     growth

     and

     have

     a

     strong

     work

     ethic

    .

     I

    'm

     always

     ready

     to

     learn

     and

     improve

    ,

     and

     I

    'm

     excited

     to

     help

     make

     a

     positive

     impact

     on

     the

     world

    .

     
    


    [

    Name

    ]

     is

     a

     [

    Position

    ]

     at

     [

    Company

    /

    Inc

    ].

     I

    'm

     a

     [

    Professional

     Title

    ]

     with

     a

     passion

     for

     [

    Skill

    /

    Field

    ].

     I

    'm

     always

     looking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Lo

    ire

     Valley

     region

     in

     western

     France

    ,

     and

     is

     known

     for

     its

     rich

     history

     and

     cultural

     heritage

    ,

     including

     the

     famous

     E

    iff

    el

     Tower

    .

     It

     is

     also

     the

     seat

     of

     the

     government

     of

     France

     and

     the

     seat

     of

     the

     French

     parliament

    .

     Paris

     has

     a

     population

     of

     over

     

    2

    .

     

    5

     million

     people

     and

     is

     a

     major

     hub

     of

     French

     culture

    ,

     art

    ,

     and

     cuisine

    .

     The

     city

     is

     famous

     for

     its

     extravagant

     fashion

    ,

     op

    ulent

     architecture

    ,

     and

     iconic

     landmarks

     such

     as

     the

     Palace

     of

     Vers

    ailles

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     a

     center

     of

     many

     major

     sporting

     events

    ,

     including

     the

     French

     Open

     tennis

     tournament

    ,

     the

     Olympics

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     with

     a

     number

     of

     emerging

     trends

     that

     are

     likely

     to

     shape

     the

     technology

     and

     its

     applications

    .

     Here

     are

     some

     of

     the

     potential

     trends

     that

     are

     likely

     to

     play

     a

     role

     in

     shaping

     AI

     in

     the

     years

     to

     come

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     One

     of

     the

     main

     challenges

     facing

     AI

     is

     ensuring

     that

     it

     is

     used

     in

     a

     responsible

     and

     ethical

     manner

    .

     As

     such

    ,

     there

     is

     an

     increasing

     emphasis

     on

     developing

     AI

     that

     is

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     effective

     in

     meeting

     ethical

     standards

    .
    


    2

    .

     Adv

    ancements

     in

     AI

     technology

    :

     Continued

     technological

     advancements

     are

     likely

     to

     result

     in

     the

     development

     of

     even

     more

     advanced

     AI

     systems

     that

     are

     capable

     of

     handling

     complex

    



```python
llm.shutdown()
```
