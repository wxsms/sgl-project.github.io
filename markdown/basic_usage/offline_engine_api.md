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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.30it/s]


    2026-04-11 05:50:03,640 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 05:50:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.68s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.05it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.15it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.79it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  21%|██        | 12/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.50it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.50it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.78it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:00<00:00, 29.11it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:00<00:00, 29.11it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.52it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.52it/s]

    Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.52it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.52it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.52it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.52it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.70it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.60it/s]

    Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.60it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 35.83it/s]


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
    Generated text:  Mary Ann and I am a librarian at the Columbus Public Library. I have been working at the Columbus Public Library since 2002 and love the job. It is my job to help people find books, periodicals and other materials that they need to read or learn. My responsibilities include:
    
      1. Making sure that all the bookshelves and book areas in the library are well organized and have a complete collection of books, periodicals and other materials.
      2. Managing the circulation system, which includes making sure that the library's books are always in stock and that people are getting their materials.
      3
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many armed guards should be stationed at the capital city to keep it safe. He knows that there are $7$ armed guards working in different parts of the city, and there are $4$ different regions in the city. The president decides to assign a guard to each region in such a way that the number of guards assigned to any two regions is not equal. How many different ways can the president assign the guards to the regions? To solve this problem, we need to determine the number of ways to assign 7 guards to 4 regions such that no two guards are in the same region. This is a classic
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the River Seine in the center of the Île de France. It is surrounded by the Seine by two islands, the Île de la Cité and the Île de la Madeleine. The capital is also called the "city of love" and is the third largest city in the world by population.
    The city is home to some of the world's most important historical and cultural institutions, including the Louvre Museum, the Palace of Versailles, the Notre Dame Cathedral, and the Eiffel Tower.
    Paris is also famous for its food, including croissants, baguettes,
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but one thing is certain: A decade from now, robots will be capable of solving some of the most complex tasks that our minds are not yet capable of performing. In the meantime, robotics engineers are seeing a lot of opportunities to develop robotic systems that can communicate with humans in a more efficient manner, while still providing the same level of intelligence that our machines currently possess. By using advanced AI techniques, robotic systems can be designed to interact with us in a more natural way, allowing us to become more engaged and collaborative with our devices.
    Robotics can be divided into two main categories: the hardware side and the software side. Hardware


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


    Generated text:  [Name] and I'm a [occupation] with [number of years] years of experience in [industry]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] with [number of years] years of experience in [industry]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] with [number of years] years of experience in [industry]. I'm a [type of person] who is always [positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" or "La Ville Flottante de Paris" (The Floating City of Paris). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also the seat of the French government and the country's political, economic, and cultural capital. Paris is a popular tourist destination and a major financial center in Europe. It is home to many famous museums, theaters,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large amounts of data, allowing machines to make more accurate predictions and decisions.
    
    3. Improved natural language processing: AI is likely to become more capable of understanding and generating human language, allowing machines to communicate more effectively and efficiently.
    
    4. Increased use of AI in healthcare: AI is likely to be used more extensively in
    


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
    Generated text:  Sarah, and I'm a young woman who loves to read and write. I'm constantly exploring new ideas and experimenting with different writing techniques. Whether it's drawing or programming, I'm always trying to create something new. I'm passionate about sharing my creative thoughts and ideas with others. If you're looking for a writer or someone who enjoys exploring new ways of telling a story, I'm your person! Write a short, neutral self-introduction for a fictional character. Hello, my name is Sarah, and I'm a young woman who loves to read and write. I'm constantly exploring new ideas and experimenting with different writing techniques. Whether
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known as the “City of Light” and the “Crown City,” which is home to many art museums, opera houses, and landmarks such as the Eiffel Tower and Notre-Dame Cathedral. It is also an important center for science, culture, and entertainment, with its many museums, theaters, and restaurants catering to both locals and tourists alike. Paris has a rich history, including the historic Louvre Museum, the Notre-Dame Cathedral, and the Arc de Triomphe. The city is known for its exceptional architecture, including the Eiffel Tower, Montmartre, and the Louvre.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a multitude of factors, including advances in machine learning, the growth of data sets, and the increasing complexity of AI systems. Here are some possible future trends in AI:
    
    1. Increased focus on ethical AI: With the increasing concerns around privacy and bias in AI, there is likely to be increased focus on ethical considerations and guidelines for developing AI systems. This could include more transparency in how AI is used, data protection, and accountability for AI-based decisions.
    
    2. Advances in natural language processing: As AI systems become more capable of understanding and generating human language, there may be a focus on developing better natural language processing


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

    ]

     and

     I

     am

     [

    Your

     Age

    ]

     years

     old

    .

     I

    'm

     currently

     a

     [

    Your

     Profession

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Interest

     or

     Hobby

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     What

     is

     your

     favorite

     hobby

     or

     interest

    ,

     and

     why

     do

     you

     enjoy

     doing

     it

    ?


    [

    Your

     Name

    ]:

     Hello

    !

     My

     name

     is

     [

    Your

     Name

    ]

     and

     I

     am

     [

    Your

     Age

    ]

     years

     old

    .

     I

     am

     currently

     a

     [

    Your

     Profession

    ]

     and

     I

    've

     always

     been

     passionate

     about

     [

    Your

     Interest

     or

     Hobby

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     What

     is

     your

     favorite

     hobby

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    .

     It

    's

     the

     largest

     city

     in

     the

     European

     Union

     and

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    .

     The

     city

     is

     known

     for

     its

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     It

    's

     also

     home

     to

     many

     important

     museums

    ,

     fashion

     brands

    ,

     and

     cultural

     institutions

    .

     Paris

     is

     one

     of

     the

     most

     popular

     tourist

     destinations

     in

     the

     world

     and

     is

     known

     for

     its

     glamorous

     nightlife

     and

     stylish

     fashion

    .

     Despite

     its

     size

    ,

     Paris

     is

     a

     cultural

     and

     intellectual

     hub

    ,

     attracting

     visitors

     from

     around

     the

     world

    .

     Do

     you

     have

     any

     other

     questions

     about

     Paris

    ?

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     several

     key

     trends

     that

     will

     shape

     the

     technology

     and

     applications

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     likely

     trends

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     become

     more

     automated

     and

     integrated

     into

     everyday

     life

    ,

     replacing

     tasks

     that

     are

     currently

     performed

     by

     humans

    .

     This

     will

     lead

     to

     increased

     efficiency

    ,

     productivity

    ,

     and

     cost

     savings

     for

     businesses

    .
    


    2

    .

     Improved

     data

     analysis

    :

     AI

     will

     become

     more

     adept

     at

     analyzing

     vast

     amounts

     of

     data

     and

     identifying

     patterns

    ,

     which

     will

     enable

     organizations

     to

     make

     more

     informed

     decisions

     and

     improve

     their

     operations

    .
    


    3

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     be

     able

     to

     learn

     from

     individual

     user

     data

     and

     tailor

     its

     interactions

     to

     meet

     their

     specific

     needs

     and

     preferences

    



```python
llm.shutdown()
```
