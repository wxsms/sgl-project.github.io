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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.60it/s]


    2026-04-29 01:20:04,869 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 01:20:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]

    Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:05<00:06,  6.46it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:05<00:02, 11.45it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]

    Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:05<00:01, 18.39it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 26.10it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.97 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.97 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.97 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.97 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.97 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.96 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.95 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.10it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.93 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.93 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.92 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.92 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.92 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.91 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.91 GB):  31%|███       | 18/58 [00:00<00:01, 35.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.91 GB):  31%|███       | 18/58 [00:00<00:01, 35.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.91 GB):  31%|███       | 18/58 [00:00<00:01, 35.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.89 GB):  31%|███       | 18/58 [00:00<00:01, 35.18it/s]

    Capturing num tokens (num_tokens=960 avail_mem=118.90 GB):  31%|███       | 18/58 [00:00<00:01, 35.18it/s] Capturing num tokens (num_tokens=896 avail_mem=118.90 GB):  31%|███       | 18/58 [00:00<00:01, 35.18it/s]Capturing num tokens (num_tokens=896 avail_mem=118.90 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=832 avail_mem=118.90 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=768 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=704 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=640 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=576 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=576 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=512 avail_mem=118.87 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=480 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.95it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=416 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=352 avail_mem=118.88 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=320 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=288 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=256 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=240 avail_mem=118.86 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=240 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.13it/s]Capturing num tokens (num_tokens=224 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.13it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.13it/s]Capturing num tokens (num_tokens=192 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.13it/s]Capturing num tokens (num_tokens=176 avail_mem=118.85 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.13it/s]Capturing num tokens (num_tokens=160 avail_mem=118.78 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.13it/s]Capturing num tokens (num_tokens=160 avail_mem=118.78 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.95it/s]Capturing num tokens (num_tokens=144 avail_mem=118.77 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.95it/s]Capturing num tokens (num_tokens=128 avail_mem=118.77 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.95it/s]Capturing num tokens (num_tokens=112 avail_mem=118.77 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.95it/s]Capturing num tokens (num_tokens=96 avail_mem=118.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.95it/s] Capturing num tokens (num_tokens=80 avail_mem=118.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.95it/s]

    Capturing num tokens (num_tokens=80 avail_mem=118.76 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=64 avail_mem=118.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=48 avail_mem=118.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=32 avail_mem=118.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=28 avail_mem=118.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=24 avail_mem=118.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=24 avail_mem=118.74 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=20 avail_mem=118.74 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.39it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.74 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=12 avail_mem=118.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=8 avail_mem=118.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.39it/s] Capturing num tokens (num_tokens=4 avail_mem=118.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=4 avail_mem=118.73 GB): 100%|██████████| 58/58 [00:01<00:00, 21.26it/s]Capturing num tokens (num_tokens=4 avail_mem=118.73 GB): 100%|██████████| 58/58 [00:01<00:00, 30.50it/s]


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
    Generated text:  Emily and I have been working as a marketing manager for 2 years now. I have a strong desire to expand my career to other countries, and I am currently searching for a marketing manager position. Can you please share your job search experience and provide some insights on how to attract more clients and generate more leads?
    Certainly, I would be happy to share my experience and provide some insights for your job search. As a marketing manager with over 2 years of experience, I have worked with several different companies in different industries and have had the opportunity to work on a variety of marketing strategies, including social media, email marketing, and content marketing
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person, but which of the following are not people?  A. Trees  B. Dolphins  C. Lions  D. Planets  E. Apes
    The answer is D. Planets. Planets are celestial bodies, which means they are not people. Trees, dolphins, and lions are living organisms, which means they are not people. Planets, on the other hand, are not living beings but are the objects of study in astronomy and the bodies that orbit around the sun. The answer is not a living being and so cannot be classified as a person.
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A) Paris B) Brussels C) Tokyo D) Moscow
    Answer:
    A) Paris
    
    Paris is the capital of France. It has a rich history, beautiful architecture, and is home to many world-renowned attractions such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Other options like Brussels, Tokyo, and Moscow are not capitals of their respective countries. Tokyo is the capital of Japan, while Moscow is the capital of Russia. The capital of France is Paris, which is the seat of the French government and the largest city in France by population. The capital of some countries can be different
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be shaped by how it is deployed, not by its originator. That is what the Future of Life Institute, which has been working on the deployment of AI in its efforts to combat climate change, has to say in its recent research paper, published in the journal Science.
    The Future of Life Institute (FoLi) has been conducting research in areas such as AI for research, development, and deployment for decades. In its report, the institute says that today's AI can be deployed in a variety of ways, including for research, development, and deployment. But the future of AI is not only dependent on the deployment of AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your character, such as "a hardworking, detail-oriented, and creative individual" or "a passionate, innovative, and collaborative team player"]. I enjoy [insert a short description of your character's interests, such as "reading, cooking, and traveling" or "exploring new cultures, trying new foods, and learning new languages"]. I'm always looking for ways to [insert a short description of your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its cuisine, fashion, and music. Paris is a cultural and artistic center, and it is home to many world-renowned museums, theaters, and art galleries. The city is also known for its annual festivals and events, such as the Eiffel Tower Parade and the Carnaval de Paris. Paris is a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced capabilities in natural language processing: AI is likely to continue to improve its ability to understand and process natural language, leading to more sophisticated and accurate language models. This could lead to more natural and intuitive interactions with AI systems.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will
    


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
    Generated text:  ___________. I’m a(n) ___________. I work in __________. My ___________ is ___________. I’ve always been passionate about __________, and I’m very ___________ about it. That’s my __________. I like to __________ and __________. I’m always __________ and __________. I’m looking forward to __________. What’s your name? What’s your title? What’s your hobby? What’s your favorite hobby? What’s your favorite book? What’s your favorite movie? What’s your favorite TV show? What’s your favorite place to travel? What’s your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. It is also the birthplace of the French Revolution and the inspiration for many famous works of art and literature. With its rich history and diverse culture, Paris is a major center of learning, art, and commerce in the world. The city is a major tourist destination and a major influence on global culture. Its location at the crossroads of Europe and the Mediterranean has made it a vital hub for commerce and diplomacy. Its location near the North Sea has also made it an important sea trade center. The climate in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  going to be marked by several key trends that are likely to shape the development of this technology. One of the primary trends is the increasing integration of AI into various industries. AI is becoming increasingly integrated into our daily lives, from smart home devices and self-driving cars to healthcare and finance. As AI becomes more integrated into various industries, we are likely to see even more automation and efficiency in these sectors.
    
    Another trend is the growing importance of AI in governance. AI is being increasingly used in government to improve public services, reduce bureaucracy, and improve decision-making. As AI becomes more integrated into governance, it is likely that there will be greater emphasis


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

    职业

    ]

     specializing

     in

     [

    职

    責

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     discuss

     [

    your

     personal

     interest

     or

     professional

     goal

    ].

     I

    'm

     a

     [

    age

    ]

     year

     old

     and

     [

    gender

    ].

     And

     I

    'm

     currently

     [

    experience

     level

    ]

     in

     this

     industry

    .

     I

     enjoy

     [

    interest

     or

     hobby

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    make

     a

     difference

     in

     the

     world

    ].

     I

    'm

     eager

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     constantly

     seeking

     out

     new

     challenges

     and

     opportunities

    .

     I

    'm

     ready

     to

     dive

     in

     and

     take

     on

     whatever

     projects

     you

     throw

     at

     me

    .

     So

    ,

     what

    's

     your

     name

    ,

     and

     what

     do

     you

     do

    ?

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ro

    che

    -A

    jar

    ée

    "

     or

     "

    La

     Ro

    che

    -a

    ux

    -B

    acc

    al

    aur

    é

    ats

    ."

     It

     is

     one

     of

     the

     most

     important

     cities

     in

     France

     and

     is

     home

     to

     a

     large

     number

     of

     notable

     landmarks

    ,

     including

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

     many

     others

    .

     The

     city

     is

     also

     known

     for

     its

     food

    ,

     wine

    ,

     and

     fashion

     industry

    .

     Paris

     is

     a

     bustling

     and

     diverse

     city

     with

     a

     rich

     cultural

     heritage

     and

     a

     cosm

    opolitan

     population

    .

     Its

     official

     language

     is

     French

    ,

     but

     English

     is

     also

     widely

     spoken

    .

     
    


    Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     considered

     one

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     predicted

     to

     be

     one

     of

     rapid

     development

     and

     innovation

    ,

     with

     many

     potential

     areas

     of

     interest

    .

     Some

     of

     the

     most

     promising

     areas

     of

     future

     AI

     are

    :
    


    1

    .

     Natural

     Language

     Processing

    :

     As

     artificial

     intelligence

     continues

     to

     advance

    ,

     it

     is

     expected

     to

     become

     more

     adept

     at

     understanding

     and

     processing

     human

     language

    .

     This

     includes

     speech

     recognition

    ,

     text

     generation

    ,

     and

     the

     ability

     to

     interpret

     natural

     language

     context

     and

     intent

    .
    


    2

    .

     Computer

     Vision

    :

     Computer

     vision

     is

     another

     area

     of

     rapid

     development

    ,

     with

     the

     goal

     of

     making

     machines

     that

     can

     perceive

     and

     interpret

     the

     world

     around

     them

     in

     the

     same

     way

     humans

     do

    .

     This

     could

     include

     applications

     like

     self

    -driving

     cars

    ,

     facial

     recognition

    ,

     and

     augmented

     reality

    .
    


    3

    .

    



```python
llm.shutdown()
```
