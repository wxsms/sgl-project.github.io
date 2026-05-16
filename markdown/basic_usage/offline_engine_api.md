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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.95it/s]


    2026-05-16 20:14:40,219 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 20:14:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.15it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.11it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.76it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.76it/s] Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.01it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.80it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.31it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.31it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.30it/s] Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 39.74it/s]


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
    Generated text:  Amina, a junior student at the University of Pennsylvania. I am a student of Political Science and I am currently in my first year. I enjoy reading fiction and watching movies. I try to find interesting books and movies every day. My first year has been very exciting and there are many things I want to do in the coming years. I would appreciate if you could write an essay on my life experience in your university.
    
    Sure! Please go ahead and share your life experience, and I'll provide an essay based on it. Let's get started! [Note: I'll need you to describe your university life in your first year,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have in the country. The cost of building a base is $10 billion plus 5% of total military expenditure for that year. The military expenditure for the year is projected to be 280 billion. How much would the total cost be for military bases?
    
    To determine the total cost for military bases, we need to follow these steps:
    
    1. Calculate the cost of building a single base.
    2. Multiply the cost of building a single base by the number of bases planned.
    
    First, let's calculate the cost of building a single base. The cost of building a base is given
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The national symbol of France is the guillemet, which means "lovely". It is a spiral that has two parts: a central spiral and the tip of the spiral. The central spiral is made up of two main parts: the core spiral and the outer spiral. The outer spiral is shaped like a spiral of 2835.62 points and the core spiral is shaped like a spiral of 456.86 points. The outer spiral has a length of 10.000 meters, while the core spiral has a length of 15.000 meters.
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, with companies like IBM, Facebook, and Google having a significant impact on the field. With this in mind, it’s essential to understand the various types of AI and what they can do. AI is the creation of intelligent machines that are capable of performing tasks that typically require human intelligence. In this article, we will explore the different types of AI and how they can be used.
    Types of AI
    1. Supervised Learning
    Supervised learning is a type of machine learning where the data is labeled. The algorithm learns the relationship between the inputs and outputs by providing training data. The algorithm can be used for tasks like


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


    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill or Hobby] enthusiast, and I love to [Describe an activity or hobby you enjoy]. I'm always looking for new experiences and challenges, and I'm always eager to learn and grow. I'm a [Favorite Food] lover, and I love to [Describe a favorite activity or hobby]. I'm a [Favorite Book] lover, and I love to [Describe a favorite activity or hobby]. I'm a [Favorite Movie] lover, and I love to [Describe a favorite activity or hobby]. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination, known for its rich history, art, and cuisine. The city is also home to many important political and military institutions, including the French Parliament and the French Navy. Paris is a major hub for international business and trade, with many international companies and organizations headquartered there. The city is also known for its fashion industry, with many famous designers and boutiques located in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with the goal of improving the quality of care and reducing the cost of treatment.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As
    


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
    Generated text:  [Name]. I am [Age] years old, and I grew up in [Place]. I'm a [Career] with [Company] in [Industry]. I'm a [Occupation] and I live in [City/Location]. I have a passion for [Interest/Skill]. I'm [Type of Person] and I'm [Positive or Negative]. I enjoy [How I'm Passionate About Something]. I'm [High or Low]. I'm excited to meet you! 
    How does your personality type influence your approach to problem-solving? As an AI language model, my personality type is not explicitly stated. However,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    (A) It is a city in the United States. 
    (B) It is a city in the United Kingdom. 
    (C) It is a city in Belgium. 
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D) It is a city in France. 
    
    (D
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid advancements, increasing complexity, and increasing reliance on artificial intelligence in many different areas of society. Here are some possible future trends in artificial intelligence:
    
    1. Increased autonomy: AI will continue to evolve and become more capable of making autonomous decisions, following complex instructions and making decisions on its own. This will likely lead to more advanced forms of self-driving vehicles and other autonomous machines.
    
    2. Improved emotional intelligence: AI will continue to be improved as developers develop more advanced algorithms for emotional intelligence. This will lead to more sophisticated chatbots and virtual assistants that can understand and respond to human emotions.
    
    3. Increased reliance on AI


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

     here

     to

     introduce

     myself

    .

     I

    've

     always

     been

     interested

     in

     learning

     about

     the

     world

     and

     discovering

     new

     things

    .

     I

     enjoy

     exploring

     different

     cultures

     and

     cuis

    ines

    ,

     trying

     new

     foods

    ,

     and

     reading

     books

    .

     My

     passion

     for

     learning

     and

     discovering

     new

     things

     has

     led

     me

     to

     become

     an

     avid

     reader

     and

     writer

    .

     I

     hope

     my

     introduction

     has

     been

     helpful

    ,

     and

     I

     appreciate

     the

     opportunity

     to

     meet

     you

    .

     Let

     me

     know

     if

     you

     have

     any

     questions

     or

     need

     any

     information

    .

     [

    Name

    ]

     [

    Your

     name

    ]

     [

    Your

     job

     or

     profession

    ]

     [

    Your

     hobbies

     and

     interests

    ]
    


    This

     self

    -int

    roduction

     sounds

     great

    !

     Can

     you

     give

     me

     some

     examples

     of

     cultural

     cuis

    ines

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     a

     population

     of

     over

     

    2

    .

     

    5

     million

     people

    .

     It

     is

     known

     for

     its

     unique

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     Visitors

     can

     experience

     the

     vibrant

     nightlife

     and

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     an

     international

     center

     of

     commerce

    ,

     finance

    ,

     and

     art

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     continues

     to

     be

     a

     major

     center

     for

     the

     arts

     and

     sciences

    .

     It

     is

     the

     political

    ,

     cultural

    ,

     and

     economic

     hub

     of

     France

     and

     hosts

     the

     most

     popular

     music

     festival

     in

     the

     world

    .

     The

     French

     capital

     of

     Paris

     is

     home

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

    ,

     with

     many

     possible

     trends

     that

     could

     significantly

     impact

     our

     lives

     and

     societies

    .

     Here

     are

     some

     of

     the

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

     have

     the

     potential

     to

     revolution

    ize

     transportation

     by

     reducing

     accidents

    ,

     speeding

     up

     traffic

    ,

     and

     increasing

     convenience

    .

     However

    ,

     they

     could

     also

     pose

     new

     safety

     concerns

     and

     job

     displacement

     for

     human

     drivers

    .
    


    2

    .

     Smart

     homes

    :

     With

     the

     increasing

     use

     of

     IoT

     devices

    ,

     smart

     homes

     could

     become

     more

     efficient

    ,

     energy

    -efficient

    ,

     and

     interconnected

    .

     They

     could

     also

     offer

     enhanced

     security

    ,

     weather

     forecasting

    ,

     and

     entertainment

    .
    


    3

    .

     Improved

     healthcare

    :

     AI

     could

     help

     doctors

     and

     researchers

     discover

     new

     treatments

     and

     diagnose

     illnesses

     more

     quickly

     and

    



```python
llm.shutdown()
```
