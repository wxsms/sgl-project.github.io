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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.24it/s]


    2026-05-20 21:24:05,658 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 21:24:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.58it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 22.07it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 31.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.67 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.67 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.38it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.38it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.39it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.39it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.39it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.39it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.39it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.39it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.94it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.19it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.56it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.71it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.61it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.61it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB): 100%|██████████| 58/58 [00:01<00:00, 41.05it/s]


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
    Generated text:  Robin. I am a college student and I have a writing problem. I wrote a short story about a young man named Jack and a woman named Mary. We both have a baby on the way. Jack lives with Mary. What do I do next? I'm so scared! I want to know the answers. I really need to know what I'm supposed to write next.
    Answering your concern about a short story about Jack and Mary (who are going to have a baby soon) and providing guidance on what you should write next is important. Given the limited information you have, the next step would likely be to consider the following:
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking the support of his constituents, who are mostly young adults. This prompts him to determine the age range that best reflects their demographic. He decides to use the standard deviation method for determining the percentage of people in different age groups. If the president selects 1000 people, how many people would you expect to be in the age group between 20 to 30 years old? To determine the percentage of the 1000 people who fall between 20 to 30 years old using the standard deviation method, we need to follow these steps:
    
    1. **Understand the standard deviation method**:
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. I doubt that you could call the capital of France, because I do not know it. I suppose that there is a capital in Germany. I am not sure. But the country itself is German. It is the largest in Europe, and the largest on the face of the globe. The capital is Berlin. It is a quite a small city. But it is in Germany. I know that there is a capital in Japan. I doubt whether there is. But the country itself is Japanese. It is the largest in Asia, and the largest on the face of the globe. The capital is Tokyo. The cities, cities,
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    AI will continue to evolve and change the way we work, learn, and live. Here are some of the most exciting AI developments and how they could change your life.
    
    The future of AI is bright
    
    The future of AI is bright
    
    There are many bright spots in the development of AI. Many researchers, innovators, and entrepreneurs are working hard to make it a reality and revolutionize the way we live, work, and learn. Here are some of the most exciting AI developments and how they could change your life.
    
    AI will continue to evolve and change the way we work, learn, and live. The possibilities for AI


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who has always been [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm a [Positive Trait] and I'm always [Positive Trait]. I'm a [Positive Trait] and I'm always [Positive Trait]. I'm a [Positive Trait] and I'm always [Positive Trait]. I'm a [Positive Trait] and I'm always [Positive Trait]. I'm a [Positive Trait] and I'm always [Positive Trait]. I'm a [Positive Trait]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the world. Paris is also home to many famous French artists, writers, and musicians. The city is a major center for the arts, with many museums, theaters, and galleries dedicated to the arts
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between machines and humans. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed to minimize harm and maximize benefits.
    
    3. Increased use of AI in healthcare: AI is already being used in
    


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
    Generated text:  ____. I'm a/an ____. I'm the ____. I'm from ____. I'm ___ years old. Please add a new line before the sentence "I'm the ____. " and a new line after the sentence "I'm the ____. " in your introduction, which character are you? Who are you writing about?
    Certainly! Here’s a neutral self-introduction for the fictional character:
    
    ---
    
    **Hello, my name is [Your Name], and I'm a/an [Your Profession]. I'm the [Your Profession] and I'm from [Your Location]. I'm [Your Age] years old. Please add
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the heart of the French countryside, and is the country's largest and most populous city. It is an important cultural and historical center, and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Palace of Versailles. Paris is also home to many world-renowned museums, opera houses, and gastronomic attractions. The French capital is a bustling and diverse urban area, with a population of over 7 million and a population density of about 20,000 people per square kilometer. The city's architecture and skyline is a testament to its rich history and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid development and increased adoption across multiple sectors. Here are some potential trends that are likely to shape the AI landscape in the coming years:
    
    1. Advanced Machine Learning and Deep Learning: One of the most promising trends in AI is the continued advancement of machine learning and deep learning technologies. These technologies are already enabling new forms of AI, such as image and speech recognition, natural language processing, and autonomous vehicles.
    
    2. Increased focus on ethical and legal issues: As AI becomes more integrated into our lives, there will be increasing scrutiny of how it is being used and regulated. This could lead to changes in policy and regulations


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

     am

     a

     [

    Main

     Character

    /

    Feature

    ]

     who

     is

     here

     to

     introduce

     myself

    .

     (

    Tell

     the

     reader

     who

     the

     character

     is

     and

     what

     they

     are

     good

     at

     or

     interested

     in

    .)

     After

     that

    ,

     you

     can

     continue

     to

     introduce

     yourself

     as

     long

     as

     you

     wish

    .

     It

    's

     important

     to

     give

     the

     reader

     a

     sense

     of

     who

     the

     character

     is

     and

     the

     reader

     will

     likely

     want

     to

     know

     more

     about

     them

    .
    


    I

    'm

     a

     [

    Main

     Character

    /

    Feature

    ],

     [

    name

    ].

     I

    'm

     [

    Name

    ]

     and

     I

    've

     always

     been

     a

     [

    Main

     Feature

    ]

    !

     I

    'm

     the

     kind

     of

     character

     who

     always

     gets

     lost

     and

     thinks

     no

     one

     sees

     them

    ,

     but

     then

     I

     find

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     "

    The

     Rose

     of

     the

     Enlightenment

    ,"

     which

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     France

    .

     It

     is

     one

     of

     the

     most

     popular

     and

     populous

     cities

     in

     the

     world

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    7

     million

     people

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     culture

    ,

     and

     cuisine

    ,

     as

     well

     as

     its

     status

     as

     one

     of

     the

     most

     influential

     and

     influential

     cities

     in

     the

     world

    .

     The

     city

     is

     home

     to

     some

     of

     the

     world

    's

     most

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

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

     the

     Arc

     de

     Tri

    omp

    he

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     exciting

     and

     multif

    ac

    eted

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

     direction

     of

     AI

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     As

     AI

     technology

     advances

    ,

     we

     can

     expect

     to

     see

     even

     more

     sophisticated

     models

     being

     developed

    .

     These

     models

     will

     be

     able

     to

     learn

     from

     large

     amounts

     of

     data

     and

     make

     increasingly

     accurate

     predictions

    .
    


    2

    .

     AI

     for

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     improve

     patient

     outcomes

    .

     However

    ,

     as

     more

     data

     is

     collected

     and

     analyzed

    ,

     the

     potential

     for

     AI

     to

     be

     used

     in

     the

     healthcare

     industry

     is

     expected

     to

     grow

    .
    


    3

    .

     AI

     for

     autonomous

     vehicles

    :

     The

     growth

     of

     self

    -driving

     cars

     is

     expected

     to

     continue

    ,

     and

    



```python
llm.shutdown()
```
