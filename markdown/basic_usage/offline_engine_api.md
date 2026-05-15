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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]


    2026-05-15 02:15:16,088 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 02:15:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:59,  5.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:59,  5.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:59,  5.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:59,  5.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:59,  5.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.20it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.08it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.95it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:06<00:00, 21.95it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]

    Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:06<00:00, 29.36it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:06<00:00, 36.94it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:06<00:00, 36.94it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:06<00:00, 36.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.71 GB):   2%|▏         | 1/58 [00:00<00:09,  6.18it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.59 GB):   2%|▏         | 1/58 [00:00<00:09,  6.18it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   2%|▏         | 1/58 [00:00<00:09,  6.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   5%|▌         | 3/58 [00:00<00:04, 11.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.18 GB):   5%|▌         | 3/58 [00:00<00:04, 11.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   5%|▌         | 3/58 [00:00<00:04, 11.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.43it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]

    Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s] Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.58it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.58it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.06it/s]

    Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.03it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.03it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.03it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.03it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.03it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=192 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.83it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.31it/s] Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=32 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.74it/s]

    Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.96it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 29.86it/s]


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
    Generated text:  David and I live in the GTA9 region, I can't speak with a voice so I am going to type it out. I am planning to take my search in a SQL database and want to write a query to find all the streets in the city that are taller than a certain height. 
    
    Here is the SQL query I have so far:
    
    ```
    SELECT city_name, ST_Centroid(Point) AS coordinate FROM city_list WHERE ST_GeometryField(T1.TownName, "height") > 10000
    
    ```
    
    I want to be able to see the list of city names in a table.
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a salary of $800,000 per year. If his salary increases by 4% every year, what will be his salary after 5 years?
    
    To determine the president's salary after 5 years, we need to apply the 4% annual increase to his initial salary of $800,000. This can be calculated using the formula for compound interest, which in this context is the formula for exponential growth:
    
    \[ \text{Future Salary} = \text{Initial Salary} \times (1 + \text{Interest Rate})^{\text{Number of Years}} \]
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Berlin
    C. Moscow
    D. Tokyo
    Answer:
    A
    
    What do I have to do before I can go to the party?
    A. Wake up early.
    B. Be well.
    C. Prepare some food.
    D. Buy a ticket.
    Answer:
    B
    
    The Chinese government and the international community are currently engaged in an effort to address climate change through the Paris Agreement, which was concluded in which year?
    A. 1987
    B. 2007
    C. 2009
    D. 2010
    Answer:
    C
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and advancing at an unprecedented pace, with a growing number of startups and large companies investing heavily in the field. As a student with a keen interest in AI and computer science, I often find myself drawn to the latest developments in the field, such as deep learning and neural networks. However, I also often feel intimidated by the complexity of the technology and the potential dangers associated with AI. What steps can I take to learn more about AI and stay safe while developing and deploying AI systems?
    
    One way to learn more about AI and stay safe while developing and deploying AI systems is to take the following steps:
    
    1. Start by reading up on


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about yourself]. I'm always looking for new opportunities to grow and learn, and I'm always eager to share my knowledge and experience with others. What's your favorite hobby or activity? I enjoy [insert a short, positive, enthusiastic statement about your favorite hobby or activity]. I'm always looking for new challenges and opportunities to grow and learn, and I'm always eager to share my knowledge and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is a major transportation hub and has a rich cultural heritage that continues to influence French society. Paris is a popular tourist destination and is home to many museums, theaters, and other cultural institutions. It is a major economic center and a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, transportation, and finance. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI becomes more advanced, there will be increasing concerns about its ethical implications and potential privacy violations. There will be a need for regulations and guidelines to ensure that
    


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
    Generated text:  [Your Name]. I am a [职业] who has been in the industry for [number of years]. I enjoy [what I enjoy doing]. And I'm always looking for [a new challenge or opportunity]! 
    
    Please provide feedback on the potential strengths and weaknesses of the self-introduction. While this is a fictional character, you can use similar traits as the author of the character. How would you modify the self-introduction to reflect the actual character's personality and abilities? Sure! Here's an example of a potential self-introduction for a fictional character:
    
    **Character Name:** Dr. Sophia Patel
    
    **Introduction:** "Hi
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Île de la Cité, the easternmost island in the French Riviera.
    Paris is the capital of France and is situated on the Île de la Cité, the easternmost island in the French Riviera. Its official name is "Paris" and it is the largest city in France. It is known for its rich history, art, cuisine, and fashion, and is also famous for its landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a significant cultural and political center in Europe. It is the 10th largest city in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some possible trends that could potentially come to fruition:
    
    1. Superintelligence: It's possible that AI could one day surpass human intelligence and become a superintelligence, with the ability to think and learn at a much faster rate than humans. This would have significant implications for society, including potential solutions to global problems like climate change, pandemics, and social inequality.
    
    2. Autonomous systems: AI is advancing rapidly and will likely become a major force in shaping our future. Autonomous systems are expected to take over many jobs and tasks that are currently done by humans, such as manufacturing, transportation, and disaster response.
    
    3


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

    'm

     a

     [

    Your

     Profession

    ].

     I

    'm

     passionate

     about

     [

    Your

     Passion

    ],

     so

     I

    'm

     dedicated

     to

     helping

     people

     find

     the

     solutions

     they

     need

     to

     succeed

    .

     Whether

     it

    's

     through

     teaching

    ,

     consulting

    ,

     or

     running

     a

     business

    ,

     I

    'm

     always

     willing

     to

     share

     my

     knowledge

     and

     experience

     to

     help

     others

     achieve

     their

     goals

    .

     Let

    's

     connect

    !

     

    📝

    ✨

    💡

     #

    Professional

     #

    Entre

    preneur

     #

    Life

    Coach

     #

    G

    rowth

    H

    acking

     #

    Business

    Leaders

    hip

    
    


    Remember

    ,

     the

     goal

     is

     to

     provide

     value

    ,

     and

     everyone

     deserves

     to

     succeed

    .

     Keep

     up

     the

     good

     work

    !

     

    💪

     #

    Insp

    iration

     #

    Success

     #

    Emp

    owering

    People

     #

    Career

    Advice

     #

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     answer

     can

     be

     limited

     to

     one

     or

     two

     words

    .

     The

     sentence

     should

     be

     gramm

    atically

     correct

     and

     easy

     to

     understand

    .

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     access

     to

     real

    -time

     information

    ,

     but

     you

     can

     easily

     find

     the

     answer

     by

     simply

     searching

     "

    Paris

     capital

    "

     on

     a

     search

     engine

    .

     The

     definition

     of

     Paris

     as

     a

     capital

     of

     France

     is

     that

     it

     is

     the

     most

     important

     city

     in

     the

     country

    ,

     hosting

     the

     seat

     of

     government

    ,

     the

     main

     commercial

     and

     cultural

     center

    ,

     and

     the

     seat

     of

     the

     French

     Parliament

    .

     
    


    Other

     cities

     in

     France

     have

     their

     own

     capitals

    ,

     and

     each

     one

     has

     its

     own

     unique

     features

     and

     histories

    .

     For

     example

    ,

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     one

     that

     promises

     to

     be

     exciting

     and

     transformative

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     development

     of

     artificial

     intelligence

     in

     the

     coming

     years

    :
    


    1

    .

     **

    Increased

     Integration

     of

     AI

     into

     Everyday

     Life

    **:

     AI

     is

     already

     becoming

     more

     integrated

     into

     our

     lives

     through

     things

     like

     smart

     home

     technology

    ,

     facial

     recognition

    ,

     and

     voice

     control

    .

     In

     the

     future

    ,

     we

     can

     expect

     even

     more

     seamless

     and

     efficient

     integration

     of

     AI

     into

     our

     daily

     routines

    ,

     from

     managing

     our

     energy

     consumption

     to

     ensuring

     our

     homes

     are

     secure

    .
    


    2

    .

     **

    Advanced

     Natural

     Language

     Processing

     (

    N

    LP

    )**

    :

     N

    LP

     is

     expected

     to

     continue

     advancing

     in

     terms

     of

     understanding

     and

     generating

     human

     language

    .

     This

     will

     enable

     machines

     to

     understand

     and

    



```python
llm.shutdown()
```
