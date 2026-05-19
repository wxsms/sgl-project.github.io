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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.73it/s]


    2026-05-19 23:10:43,383 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-19 23:10:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.83it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.83it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.83it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:17,  2.97it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:17,  2.97it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:17,  2.97it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:17,  2.97it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:05,  7.71it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:02, 14.35it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 25.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 44.02it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 44.02it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 44.02it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 44.02it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 44.02it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 44.02it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 44.02it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 44.02it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 44.02it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 44.02it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 44.02it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 44.02it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 57.65it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 57.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.46it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.71it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.64it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.64it/s] Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.64it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.64it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=640 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=448 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.47it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=352 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.04it/s]Capturing num tokens (num_tokens=320 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.04it/s]Capturing num tokens (num_tokens=288 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.04it/s]Capturing num tokens (num_tokens=256 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.04it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.04it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.04it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.34it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=96 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=64 avail_mem=71.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=64 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=48 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.58it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.20it/s] Capturing num tokens (num_tokens=4 avail_mem=71.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=4 avail_mem=71.60 GB): 100%|██████████| 58/58 [00:01<00:00, 40.28it/s]


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
    Generated text:  Cao Xin. I'm 20 years old, and my favorite color is blue. I have a dog named Zhihao and I like to eat spicy food. Here's a sentence about me: I'm 20 years old, my favorite color is blue, and I have a dog named Zhihao. Based on that, can you identify the type of sentence? The sentence is a:
    
    a) First person sentence
    b) Second person sentence
    c) First person plural sentence
    d) Second person plural sentence
    e) Compound sentence
    
    c) First person plural sentence
    
    The sentence is a first
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the income level of the people in his district. He randomly selects a sample of 100 people and finds that the average income is $50,000 with a sample standard deviation of $10,000. He wants to construct a 95% confidence interval estimate for the true average income of all the people in his district. What is the width of the 95% confidence interval for the true average income?
    
    To find the width of the 95% confidence interval for the true average income, we need to follow these steps:
    
    1. **Identify the given information
    ===============================
    Prompt: The capital of France is
    Generated text:  located in (　　) of the country.
    A: North
    B: South
    C: East
    D: West
    To determine the capital of France, we need to understand the general position of France and its major cities. France is a country located in Western Europe. The cities in France are distributed in a specific pattern along a specific line.
    
    France is divided into four main regions:
    1. The Atlantic coast, where the capital is typically located.
    2. The Alps, where the capital is not typically located.
    3. The French Alps, where the capital is located.
    4. The southern parts of the country, where the
    ===============================
    Prompt: The future of AI is
    Generated text:  brimming with incredible possibilities for improving the quality of life for everyone. In this article, we explore the various benefits of AI and how it can revolutionize the way we live, work, and communicate. We'll also look at the technical, regulatory, and ethical challenges that come with integrating AI into our digital world and exploring the potential for AI to impact all sectors of society. So, let's dive in and discover how AI is changing the way we live and work today!
    The benefits of AI in the future of work include:
    1. Increased efficiency: AI can help automate repetitive and time-consuming tasks, freeing up human workers to focus


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its rich history, including the French Revolution and the French Revolution Museum. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic hub in Europe. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: One of the most significant trends in AI is the increasing integration of AI with human intelligence. This could lead to more sophisticated and personalized AI systems that can learn from and adapt to human behavior and preferences.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and information that is generated and processed by AI systems.
    
    3. Increased focus on
    


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
    Generated text:  [Name] and I am [Age]. I am a [Age] year old [Gender] with [Profession], [Location], [Skills], and [Personal Traits] person. If you need any information about me, feel free to ask and I will be happy to share.
    
    I enjoy [What You Do], [How You Spend Your Time], and [What's Your Favorite Activity to Do]. I like to travel, read, and spend time with friends and family. I also enjoy [What You Do in Your Free Time]. If you want to know more about me, feel free to ask and I'll do my best
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the world's 21st most populous city, located in the northern region of the country. It is the largest city in the European Union, and is known for its rich history, art, fashion, cuisine, and architecture. Paris is home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. It is also the world's most-visited tourist destination, attracting millions of visitors each year. The city is also home to the French Parliament, the Supreme Court, and many of the country's cultural institutions. The French language is spoken by millions of people throughout the country, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities and potential applications. Here are some of the potential trends that may shape the development of AI in the next few years:
    
    1. More Advanced Self-Driving Cars: As AI continues to advance, we can expect self-driving cars to become more common in our daily lives. These vehicles will be equipped with a variety of sensors, cameras, and other AI technologies to improve their safety and efficiency.
    
    2. AI in Healthcare: AI will be used to improve the accuracy and efficiency of medical procedures. AI-powered tools will be able to analyze medical images, provide patient-specific treatment plans, and even predict disease outbreaks.
    
    3. AI in


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

    ].

     I

    'm

     a

     [

    occupation

     or

     hobby

    ]

     who

     have

     always

     been

     fascinated

     by

     the

     unknown

     and

     always

     wanted

     to

     explore

     the

     world

     around

     me

    .

     I

     enjoy

     learning

     new

     things

    ,

     trying

     new

     experiences

    ,

     and

     trying

     new

     foods

    .

     I

     have

     a

     friendly

     and

     outgoing

     personality

    ,

     and

     I

     love

     making

     friends

    .

     I

     am

     always

     looking

     for

     new

     adventures

     and

     exciting

     experiences

     to

     try

    .


    How

     would

     you

     describe

     your

     personality

     and

     interests

     as

     a

     whole

    ?

     My

     personality

     is

     friendly

    ,

     outgoing

    ,

     and

     adventurous

    .

     I

     love

     trying

     new

     foods

    ,

     trying

     new

     experiences

    ,

     and

     making

     friends

    .

     I

     enjoy

     learning

     new

     things

     and

     exploring

     the

     unknown

    .

     My

     interests

     include

     music

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Mont

    mart

    re

    .
    


    Paris

     is

     a

     vibrant

     and

     culturally

     rich

     city

     that

     has

     become

     synonymous

     with

     French

     values

     and

     culture

    ,

     with

     its

     iconic

     landmarks

     and

     stunning

     architecture

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

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Mont

    mart

    re

    ,

     which

     are

     all

     major

     attractions

     that

     draw

     millions

     of

     visitors

     each

     year

    .

     Paris

     is

     a

     city

     that

     has

     played

     a

     significant

     role

     in

     the

     development

     of

     French

     culture

     and

     has

     remained

     a

     popular

     tourist

     destination

     for

     generations

    .

     Its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     and

     change

     rapidly

    ,

     driven

     by

     advances

     in

     hardware

    ,

     software

    ,

     and

     data

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     More

     specialized

     AI

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     the

     focus

     will

     shift

     towards

     specific

     tasks

     and

     applications

    .

     This

     will

     likely

     lead

     to

     the

     development

     of

     more

     specialized

     AI

     systems

     that

     are

     tailored

     to

     solve

     particular

     problems

    .
    


    2

    .

     AI

     that

     is

     more

     human

    -like

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     may

     begin

     to

     exhibit

     more

     human

    -like

     behavior

     and

     decision

    -making

    .

     This

     could

     lead

     to

     more

     human

    -like

     AI

    ,

     where

     AI

     systems

     make

     decisions

     that

     feel

     natural

     and

     human

    -like

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
