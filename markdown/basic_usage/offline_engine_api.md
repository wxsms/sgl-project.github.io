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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]


    2026-04-12 07:44:01,606 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 07:44:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.58it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.58it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.46it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.95it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.92it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.22it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.62it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.21it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.21it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.30 GB):  31%|███       | 18/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.29 GB):  31%|███       | 18/58 [00:00<00:01, 34.80it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.27 GB):  31%|███       | 18/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=960 avail_mem=119.06 GB):  31%|███       | 18/58 [00:00<00:01, 34.80it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  31%|███       | 18/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.64it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.64it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.64it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.64it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.64it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.64it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 36.83it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 36.83it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.83it/s]

    Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.82it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.82it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.82it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.82it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.82it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.18it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.18it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.29it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 33.18it/s]


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
    Generated text:  X. I am a SQL Server 2012 user. I need to know what is the possible scenario of using xquery to transform a column into another column. 
    
    1. How can I do this?
    2. Are there any examples?
    3. How do I use it?
    4. What is the performance? 1. **How can I do this?**
    
       To transform a column in SQL Server 2012 using XQuery, you can use the `SELECT` statement to apply an XQuery expression to the column values. Here's a general approach:
    
       - **Target Column:** Identify the column
    ===============================
    Prompt: The president of the United States is
    Generated text:  a head of state. The current president of the United States is Joe Biden. This is a term of office for two years. In the event of death, resignation or withdrawal from the office, the vice president is elected in their place.
    
    If you were the president of the United States, what would you do?
    
    As the president of the United States, the primary responsibility would be to represent the country and its people in a way that reflects the values and interests of the people. This includes advocating for policies that are in the best interest of the country and ensuring that the government operates according to the constitution and laws.
    
    Some possible actions that the
    ===============================
    Prompt: The capital of France is
    Generated text: : A: Paris B: London C: New York D: Berlin
    The capital of France is A: Paris. Paris is the capital city of France, located on the northeastern coast of the country. It is the largest city in France and the seventh-largest city in the world by population. The other options are not capitals of France, they are cities in France: London is the capital city of the United Kingdom, New York is the capital city of the United States, and Berlin is the capital city of Germany.
    ===============================
    Prompt: The future of AI is
    Generated text:  defined by big data, which is a collection of data, usually in the form of structured or unstructured text, images, or videos, that can be used to extract useful information. This is the next big challenge that the AI industry is currently facing, as the amount of data available is currently overwhelming, and it's becoming more challenging to find meaningful patterns in it.
    As a result, the focus of AI research has shifted towards using advanced machine learning algorithms that can analyze the complex and diverse data available. However, it is important to note that AI is a dynamic field with new challenges emerging from time to time.
    One such challenge is the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the Louvre Museum. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. Its status as the world's most populous city is due to its large population and high density of buildings. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased pressure to address ethical concerns related to AI, such as bias, transparency, and accountability. This could lead to more rigorous
    


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
    Generated text:  [Name] and I’m a [Role] who specializes in [Title] at [Organization]. I bring a unique blend of [Attributes], [Skills], and [Specializations] to my work as a [Title] at [Organization]. If you have any questions or would like to learn more about my background and experience, feel free to reach out. I look forward to the opportunity to build a relationship with you! [Name]...
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A) True B) False
    B) False
    Paris, officially known as the "City of Love" and the "City of Fine Arts and Culture," is the largest city in France and the seat of the French government. It is located in the Île de France on the French Riviera, and its economy is heavily dependent on tourism, particularly the French Riviera. Paris is also known for its fashion industry, art, and music. It is a significant cultural and historical center in Europe and is one of the largest cities in the world. French cuisine, particularly in the city center, is renowned for its complexity
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several trends that are currently being explored and that could shape the technology's direction. Some of the potential future trends in AI include:
    
    1. AI Integration: AI is becoming more integrated with other technologies, such as cloud computing, IoT, and big data, which will create new opportunities for the AI industry.
    
    2. Artificial General Intelligence (AGI): AGI refers to the ability of an AI system to perform tasks that were previously thought to be impossible for humans. This technology could revolutionize many industries, such as healthcare, manufacturing, and transportation.
    
    3. Explainability and Interpretability: AI systems are often opaque


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

    Occup

    ation

    /

    Prof

    ession

    ]

     with

     [

    Number

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

     enjoy

     [

    Why

     do

     you

     like

     this

     profession

    ?

    ].

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

     so

     I

    'm

     always

     eager

     to

     grow

     and

     develop my

     skills

    .

     If

     you

    'd

     like

     to

     ask

     me

     a

     question

     or

     learn

     more

     about

     me

    ,

     feel

     free

     to

     let

     me

     know

    !

     #

    Self

    Introduction

     #

    Industry

    B

    logger

    


    How

     can

     I

     improve

     my

     writing

     skills

    ?
    


    Writing

     is

     a

     powerful

     tool

     for

     creating

     stories

     and

     conveying

     ideas

    .

     However

    ,

     it

     can

     also

     be

     a

     daunting

     experience

    .

     If

     you

    're

     looking

     to

     improve

     your

     writing

     skills

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     anticipated

     and

     currently

     involves

     a

     variety

     of

     possible

     trends

     and

     applications

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     already

     in

     use

     in

     various

     cities

     and

     countries

    ,

     and

     with

     the

     increasing

     use

     of

     AI

    ,

     we

     can

     expect

     more

     autonomous

     vehicles

     in

     the

     future

    .
    


    2

    .

     Super

    int

    elligent

     machines

    :

     AI

     is

     expected

     to

     become

     more

     powerful

     and

     capable

    ,

     reaching

     levels

     of

     intelligence

     that

     surpass

     human

     intelligence

    .

     This

     could

     lead

     to

     the

     creation

     of

     super

    int

    elligent

     machines

    ,

     which

     could

     potentially

     surpass

     human

     abilities

     in

     areas

     such

     as

     creativity

    ,

     problem

    -solving

    ,

     and

     decision

    -making

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

     can

     be

     used

     to

     personalize

     healthcare

    



```python
llm.shutdown()
```
