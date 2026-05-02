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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.79it/s]


    2026-05-02 09:49:49,462 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 09:49:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.28it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.93it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.73it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  31%|███       | 18/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.26it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.41it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.57it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.57it/s]

    Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.63it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 41.98it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 38.86it/s]


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
    Generated text:  Al Sweigart, and I'm an author of several popular computer programming and gaming articles. I'm not a technical guy, but I love games. One of my favorite games is "One Punch Man." In this article, I'll show you how to make a game that is fun and educational for children and teens. I'll explain how to make the game, how to design the controls, how to add in new levels, and then how to make it playable on a wide variety of devices. I hope you enjoy this project.
    Games can be fun for anyone, and this article will help you build a simple game that you can
    ===============================
    Prompt: The president of the United States is
    Generated text:  planning a visit to the world's largest solar panel installation to promote clean energy. The installation is located at the United States' largest natural gas pipeline that runs through the country. The president's flight plan includes driving through the installation, which is 300 miles long. If the president drives at a constant speed of 60 miles per hour, how many hours will it take to complete the 300-mile trip? If the president also plans to drive on a detour that is 200 miles long, how long will the detour take?
    To determine the time it takes for the president to complete the 
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. A．Paris B．Lyon C．Lille D．Lorient
    
    The correct answer is A. Paris.
    
    Paris is the capital city of France and is also the largest city in the country, with a population of over 2 million people. It is located in the Loire Valley region of France and is known for its beautiful architecture, stunning architecture, and historical landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    
    While Lyon and Lille are also cities in France, they are not the capital of the country. Lille, located in the Nord department of eastern France
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting, but we have to be careful with its development.
    Let’s take a look at the current AI landscape, its potential and the challenges that still lie ahead.
    1. Artificial Intelligence (AI) – The Future of the World
    When I say “Artificial Intelligence”, I mean a type of machine learning. It’s a technique of using statistical methods to teach machines to learn from data. In other words, it’s a way of programming computers to do what humans can do with their brains.
    With AI, machines can solve problems much faster than humans can, and they can also adapt to different situations and learn from experiences.
    AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its food, fashion, and music scene. Paris is a popular tourist destination and a major economic center in France. It is also home to many international organizations and institutions. The city is known for its annual Eiffel Tower Festival and the annual Parisian Carnival. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable of performing complex tasks that were previously impossible for humans. This could lead to more advanced AI systems that can perform tasks that were previously thought to be beyond the capabilities of human intelligence.
    
    3. Increased focus on ethical AI: As AI becomes more integrated with human intelligence
    


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
    Generated text:  [Name] and I'm a [career/position] in [company/industry]. I'm passionate about [motivational statement]. Let's chat! Let's get to know each other better. [Name]!
    
    Note: Replace the placeholders with your own names and the specifics of the company/industry with your own information. Remember to keep the introduction neutral and brief. Good luck! [Name]!
    
    Hello, my name is [Name] and I'm a [career/position] in [company/industry]. I'm passionate about [motivational statement]. Let's chat! Let's get to know each other better
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest and most populous city in the European Union, and is a major European city and cosmopolitan metropolis. It is located on the banks of the Seine River, in the southern part of the country. Paris is known for its rich history, art, music, and cuisine, and is home to many world-renowned institutions and landmarks. The city is home to important French cultural institutions, such as the Louvre Museum, the Centre Pompidou, and the Musée d'Orsay, and hosts numerous concerts, festivals, and exhibitions throughout the year. Paris is a vibrant and exciting city, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be more diverse, complex, and interconnected than ever before. Here are some possible future trends in AI:
    
    1. Advancements in machine learning: As AI systems become more complex, they are likely to require more powerful and diverse algorithms. This could lead to new forms of machine learning, such as deep reinforcement learning, which is a form of machine learning where a neural network learns to interact with the environment.
    
    2. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on making sure that AI systems are designed and used in ways that are ethical and fair. This


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

     Emily

    ,

     and

     I

     am

     a

     computer

     programmer

     with

     a

     strong

     interest

     in

     machine

     learning

     and

     AI

    .

     I

     have

     always

     been

     fascinated

     by

     how

     data

     and

     algorithms

     can

     be

     used

     to

     create

     complex

     and

     sophisticated

     systems

    .

     I

     have

     experience

     in

     building

     and

     deploying

     complex

     software

     systems

    ,

     as

     well

     as

     working

     with

     machine

     learning

     algorithms

    .

     I

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     advancements

     in

     the

     field

    .

     Thank

     you

    .

     [

    insert

     

    1

    -

    2

     sentences

     introducing

     yourself

     and

     your

     background

     in

     programming

     and

     machine

     learning

    ]

     Hi

     there

    ,

     Emily

    !

     I

    'm

     Emily

    ,

     a

     computer

     programmer

     with

     a

     passion

     for

     AI

     and

     machine

     learning

    .

     I

    've

     always

     been

     fascinated

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     should

     be

     clear

     and

     concise

    ,

     providing

     the

     key

     information

     that

     is

     expected

     to

     be

     known

     to

     most

     readers

    .

     It

     should

     not

     include

     any

     unnecessary

     details

     or

     technical

     language

    .

     
    


    Additionally

    ,

     please

     provide

     an

     alternative

     phrase

     that

     accurately

     describes

     Paris

     based

     on

     its

     historical

     and

     cultural

     significance

     as

     a

     major

     city

     in

     Europe

    .

     The

     alternative

     phrase

     should

     be

     distinct

     from

     the

     traditional

     "

    capital

    "

     and

     should

     capture

     the

     unique

     character

     and

     charm

     of

     Paris

    .
    


    Sure

    ,

     here

     is

     the

     requested

     statement

     and

     the

     alternative

     phrase

    :
    


    **

    F

    acts

    :**

     


    -

     The

     capital

     of

     France

     is

     Paris

    .


    -

     Alternative

     phrase

    :

     "

    Paris

    's

     Golden

     Age

    ".
    


    The

     statement

     "

    The

     capital

     of

     France

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     number

     of

     trends

    ,

     including

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     our

     daily

     lives

    ,

     from

     smart

     home

     devices

     to

     self

    -driving

     vehicles

    .

     This

     integration

     will

     likely

     create

     new

     opportunities

     and

     challenges

     for

     developers

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     become

     more

     capable

     of

     making

     decisions

     based

     on

     data

    ,

     potentially

     replacing

     human

     decision

    -makers

     in

     various

     industries

    .
    


    3

    .

     Increased

     reliance

     on

     AI

     for

     security

    :

     AI

     will

     continue

     to

     play

     a

     vital

     role

     in

     protecting

     our

     digital

     lives

     and

     information

    .

     As

     AI

     continues

     to

     improve

    ,

     it

     will

     become

     more

     capable

     of

    



```python
llm.shutdown()
```
