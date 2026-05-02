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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]


    2026-05-02 11:35:30,039 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 11:35:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.02it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.81it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   2%|▏         | 1/58 [00:00<00:05,  9.80it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   2%|▏         | 1/58 [00:00<00:05,  9.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   2%|▏         | 1/58 [00:00<00:05,  9.80it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   5%|▌         | 3/58 [00:00<00:03, 13.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   5%|▌         | 3/58 [00:00<00:03, 13.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   5%|▌         | 3/58 [00:00<00:03, 13.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:03, 14.93it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.34it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.34it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.05it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.05it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.05it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.05it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.05it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.05it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.31it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.97it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.97it/s]

    Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.01it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.01it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 45.46it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 33.65it/s]


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
    Generated text:  Tricia. I'm 31 years old and have been married to my husband, Mike, for 10 years. We moved to the small town of Kennett Square a few years ago, and Mike is the one who raised us all. We moved from Kansas City, Missouri. Tricia has a passion for gardening and has started her own flower shop, which she opened in 2015. Her husband and I have an interest in history and we have been collecting items to keep us company and in memory of our parents.
    I have two children who are 10 years old and 8 years old. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very busy person. On a Monday he answers 120 phone calls. On a Tuesday, he answers 240 calls. On Wednesday, he answered 5/6 times as many calls as on Tuesday. On Thursday, he answered 1/4 as many calls as on Monday. How many calls did the president have total in the week?
    To determine the total number of phone calls the president answered during the week, we need to calculate the number of calls answered each day and then sum them up.
    
    First, let's find out how many calls the president answered on Monday:
    \[ \text{Monday} =
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the most populous city in the European Union (5, 587, 700 people in 2019), and also the third largest city in the world by population. It is located on the right bank of the Seine River, between the City of Paris and the city of the same name. Paris is situated in the region of the Center Region and is also located in the Centre-Val de Loire administrative region.
    
    Based on that paragraph can we conclude that this sentence is true?
    Paris is the capital of the United States.
    
    Options are:
    a). yes
    b). it is not possible
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting and diverse. The next generation of AI technologies will change the way we work, communicate and interact with the world. But, just like any new technology, there are also risks and challenges that the future of AI poses.
    It is essential to understand the potential risks and challenges that AI poses and how to mitigate them. In this post, we will explore the potential risks and challenges of AI, what they are and how they can be mitigated. We will also discuss the ethical implications of AI and how to address them.
    Artificial Intelligence (AI) can be defined as the development of computer systems that can perform tasks that normally require human


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! 🌟✨✨
    
    ---
    
    **[Name]**  
    [Company Name]  
    [Job Title]  
    [Company Website]  
    [LinkedIn Profile]  
    [Email Address]  
    [Phone Number]  
    [Company Address]  
    [City, State, ZIP Code]  
    [Country]  
    [Date of Birth]  
    [Gender]  
    [Age]  
    [Height]  
    [Weight]  
    [Hobbies]  
    [Interests]  
    [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art, and is a major tourist destination. Paris is a cultural and economic hub of France and plays a significant role in the country's political and social landscape. It is home to many of the country's most famous museums, including the Louvre and the Musée d'Orsay. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies such as blockchain, quantum computing, and quantum cryptography. This integration will enable new applications and services that were not possible before.
    
    2. Improved privacy and security: As AI systems become more complex and sophisticated, there will be a need to address privacy and security concerns. This will require advancements in encryption, data protection, and ethical considerations.
    
    3. Enhanced human-machine collaboration: AI will continue to play a more
    


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
    Generated text:  Jane. I am a friendly, helpful AI designed to assist people with a wide range of questions and tasks. I'm here to provide reliable information and answer any questions you might have. How can I help you today? I'm always here to assist you, just as you've always been here to help me. How can I help you today? I'm always here to assist you, just as you've always been here to help me. How can I help you today? I'm always here to assist you, just as you've always been here to help me. How can I help you today? I'm always here to assist
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and its capital. It is known for its stunning architecture, vibrant culture, and annual celebrations, including the Eiffel Tower and the World Cup. The city is also a major financial center and home to many famous museums and art galleries. Despite its size, Paris is relatively relaxed and has a rich history dating back over 2,000 years. With its unique blend of architecture and culture, Paris remains a city of contrasts and wonder. The French Republic is headquartered in Paris and is the largest city of the European Union. France’s capital, Paris, is a vibrant, historic city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic, and there are many potential trends that could shape the technology in the years to come. Here are some potential trends that could potentially impact AI:
    
    1. AI will become more ubiquitous: One of the biggest trends in AI is that it will become more ubiquitous, meaning that more people will interact with AI in their daily lives. This could be through our smartphones, smart home devices, virtual assistants, and more.
    
    2. AI will become more personalized: AI will also become more personalized, which means that machines will be able to understand and analyze patterns in human behavior, preferences, and experiences. This could lead to more targeted and personalized


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

     I

    'm

     [

    Age

    ],

     and

     I

    'm

     a

     [

    Career

     or

     hobby

    ]

    !

     I

    've

     always

     been

     curious

     about

     learning

     new

     things

    ,

     and

     I

    've

     always

     been

     fascinated

     by

     the

     mystery

     of

     the

     unknown

    .

     I

     like

     to

     learn

     on

     my

     own

    ,

     and

     I

     enjoy

     sharing

     my

     knowledge

     with

     others

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     learning

     opportunities

    ,

     and

     I

    'm

     always

     eager

     to

     expand

     my

     hor

    izons

    .

     I

     hope

     you

     enjoy

     learning

     more

     about

     me

    !

     How

     about

     you

    ,

     [

    Name

    ],

     what

    's

     your

     name

    ?

     I

    'm

     [

    Name

    ],

     and

     I

    'm

     [

    Age

    ].

     I

    'm

     also

     a

     [

    Career

     or

     hobby

    ].

     And

     I

    've

     always

     been

     fascinated

     by

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     on

     the

     Se

    ine

     river

     and

     is

     the

     political

    ,

     economic

    ,

     cultural

    ,

     and

     residential

     center

     of

     the

     country

    .

     Its

     famous

     landmarks

     include

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

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Sac

    ré

    -C

    œur

     Basil

    ica

    .

     Other

     famous

     French

     cities

     include

     Paris

    ,

     N

    antes

    ,

     and

     Lyon

    .

     Paris

     has

     a

     long

     and

     rich

     history

     dating

     back

     to

     the

     

    6

    th

     century

     and

     is

     known

     for

     its

     romantic

    ,

     historical

    ,

     and

     artistic

     atmosphere

    .

     It

     is

     also

     famous

     for

     its

     fashion

     industry

    ,

     art

     scene

    ,

     and

     cuisine

    .

     Paris

     is

     a

     city

     that

     is

     not

     only

     a

     place

     to

     live

     but

     also

     a

     place

     to

     explore

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     possible

     trends

     that

     are

     currently

     being

     studied

     and

     implemented

     in

     various

     industries

     and

     sectors

    .
    


    1

    .

     Increased

     Depend

    ence

     on

     AI

    :

     As

     AI

     continues

     to

     become

     more

     prevalent

     in

     our

     daily

     lives

    ,

     we

     are

     likely

     to

     see

     an

     increase

     in

     its

     dependence

     on

     us

    .

     This

     may

     lead

     to

     a

     situation

     where

     we

     are

     more

     reliant

     on

     AI

     in

     areas

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     customer

     service

    .

     It

     is

     also

     possible

     that

     we

     may

     have

     to

     rely

     on

     AI

     more

     heavily

     in

     the

     future

    .
    


    2

    .

     Increased

     Eth

    ical

     Concern

    s

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     daily

     lives

    ,

     there

     will

     be

     an

     increased

     focus

     on

     ethical

     concerns

     surrounding

     its

    



```python
llm.shutdown()
```
