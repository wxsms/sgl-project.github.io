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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.29it/s]


    2026-04-13 02:11:45,862 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 02:11:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.00it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.00it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.00it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.15it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.07it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.31 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.92 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=119.08 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.08 GB):  31%|███       | 18/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  31%|███       | 18/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  31%|███       | 18/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  31%|███       | 18/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  31%|███       | 18/58 [00:01<00:02, 17.50it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  31%|███       | 18/58 [00:01<00:02, 17.50it/s]Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.76it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.76it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.76it/s]

    Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.76it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.76it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.76it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.87it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.87it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.87it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.87it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.87it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.87it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.13it/s]

    Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.23it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.72it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.23it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.79it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 31.54it/s]


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
    Generated text:  Dierik and I am a Computer Vision Engineer at Tork, a Dutch startup that uses artificial intelligence to solve the toughest problems in the world. I have been working on natural language processing, Computer Vision, and machine learning for years and have been working in the industry for 15 years. My PhD is in Computer Vision, and I have been a professor of Computer Vision at the University of Twente. I love to travel, read, play piano, and knit.
    The latest research I am working on is called "Helmholtz Visual Detection", and my work will improve the quality and usability of medical imaging. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to appoint a new diplomat. There are 8 candidates available, each with a different level of experience in diplomatic work: 2 experienced, 3 moderately experienced, and 3 inexperienced. The candidate must be chosen such that the experience level ensures at least one diplomatic experience is selected. The president also has a preference for an experienced diplomat. If the president decides to prioritize the experience level over the diplomatic expertise, how many ways can he choose a diplomat with an experienced position, given that the experience level is the only factor considered? To determine the number of ways the president can choose a diplomat with an experienced position, given that the experience
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. Lyon C. Nancy D. Marseille
    Answer:
    
    A
    
    How many times does the sine function $y = \sin x$ intersect with the line $y = 1$?
    A. 0 times
    B. 1 time
    C. 2 times
    D. 3 times
    Answer:
    
    B
    
    In the following set of number lines, determine the value of $a$ such that the intersection points of the function $y = -a^2 + 2a + 3$ with the positive half-axis ($x > 0$) are exactly $x = 
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and companies are recognizing it as a significant threat to their survival. However, there are many ways to stay ahead of the curve.
    You might not know how to do it yet, but there are some techniques you can use to make your AI development process a little faster and more efficient. The key to successful AI is to have a plan and a strategy.
    Here are a few tips for making your AI development process faster and more efficient:
    1. Create a plan: Start by defining your goals and objectives for your AI development process. What are you trying to achieve? What are the key success metrics you want to measure? What is


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm [age] years old, [gender] and [nationality]. I have a [job title] at [company name] and I'm always looking for ways to [describe a new skill or accomplishment]. I'm [interests or hobbies] and I enjoy [describe a hobby or activity]. I'm [ability to do something]. I'm [ability to do something]. I'm [ability to do something]. I'm [ability to do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous croissants and its traditional French cuisine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has been a center of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be greater emphasis on ethical considerations. This could lead to more transparent and accountable AI systems that are designed to minimize harm and maximize benefits.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more integrated with human
    


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
    Generated text:  [Name]. I'm a [type] with [number] years of experience in [field]. I enjoy [reason for being in this role]. My goal is to [future goal or accomplishment]. I'm [ability to describe it in a positive or negative way]. I'm always [characteristic]. I'm [ability to describe it in a positive or negative way]. [Name] is a [type] with [number] years of experience in [field]. I enjoy [reason for being in this role]. My goal is to [future goal or accomplishment]. I'm [ability to describe it in a positive or negative way].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the seat of the French government. Paris is known for its iconic landmarks, including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its rich history, including the Romanesque and Gothic architecture, as well as its current influence on the arts and culture of France and the world. Paris is a popular tourist destination and is a major economic and cultural hub in France. The city is home to a diverse range of neighborhoods and districts, each with its own unique character and charm. The city has a rich history and a strong sense
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of key trends and developments. Here are some of the most likely areas of focus:
    
    1. Advancements in Machine Learning: Machine learning is the heart of AI and will continue to drive progress. With advancements in deep learning, reinforcement learning, and other machine learning techniques, we can expect to see even more sophisticated models that can learn from data and make more accurate predictions and decisions.
    
    2. Integration with Physical and Chemical Sciences: As AI becomes more integrated with physical and chemical sciences, we can expect to see more applications in fields like drug discovery, materials science, and climate modeling.
    
    3. Global Collaboration:


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

    insert

     name

     here

    ],

     and

     I

    'm

     [

    insert

     occupation

     here

    ].

     I

    'm

     a

     [

    insert

     age

     here

    ]

     year

     old

     [

    insert

     profession

     here

    ],

     [

    insert

     nationality

     here

    ],

     [

    insert

     location

     here

    ].

     I

     grew

     up

     in

     [

    insert

     your

     hometown

     or

     city

     here

    ]

     and

     have

     lived

     here

     for

     [

    insert

     number

     of

     years

     here

    ]

     years

     now

    .

     I

     have

     a

     passion

     for

     [

    insert

     hobby

     or

     activity

     here

    ],

     and

     I

     enjoy

     [

    insert

     hobby

     or

     activity

     here

    ].

     I

    'm

     [

    insert

     personality

     trait

     here

    ],

     and

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     learning

     new

     things

    .

     I

     love

     to

     [

    insert

     hobby

     here

    ],

     and

     I

     have

     a

     great

     sense

     of

     humor

    .

     I

    'm

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     light

    ,

     art

    ,

     and

     fashion

    .


    Paris

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     romantic

     cath

    ed

    r

    als

     like

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     world

    -ren

    owned

     Lou

    vre

     Museum

    ,

     which

     houses

     some

     of

     the

     world

    's

     most

     famous

     artworks

    .

     The

     city

    's

     vibrant

     nightlife

     and

     cultural

     scene

     are

     also

     highly

     regarded

    .

     In

     addition

    ,

     Paris

     is

     home

     to

     many

     other

     historic

     landmarks

     and

     museums

    ,

     such

     as

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     Rod

    in

    ,

     and

     the

     Place

     de

     la

     Con

    cor

    de

    .

     Paris

     is

     a

     city

     that

     has

     a

     strong

     sense

     of

     French

     culture

     and

     is

     a

     popular

     tourist

     destination

     for

     visitors

     from

     all

     over

     the

     world

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bound

     to

     be

     full

     of

     exciting

     developments

    ,

     and

     it

    ’s

     likely

     to

     continue

     to

     evolve

     at

     an

     unprecedented

     pace

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

     Increased

     privacy

     and

     security

     concerns

    :

     AI

     will

     continue

     to

     evolve

    ,

     leading

     to

     more

     sophisticated

     methods

     of

     data

     collection

    ,

     analysis

    ,

     and

     processing

    .

     As

     AI

     becomes

     more

     autonomous

    ,

     privacy

     and

     security

     concerns

     will

     become

     even

     more

     critical

    .
    


    2

    .

     Enhanced

     AI

     capabilities

    :

     AI

     will

     continue

     to

     advance

     at

     a

     faster

     pace

    ,

     leading

     to

     even

     more

     advanced

     models

     and

     systems

    .

     This

     could

     lead

     to

     a

     more

     sophisticated

     ability

     to

     understand

     and

     respond

     to

     human

     language

    .
    


    3

    .

     Autonomous

     vehicles

    :

     AI

     will

     continue

     to

     play

     a

     critical

    



```python
llm.shutdown()
```
