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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.48it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.85it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.78it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.60 GB):  21%|██        | 12/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]

    Capturing num tokens (num_tokens=960 avail_mem=53.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s] Capturing num tokens (num_tokens=960 avail_mem=53.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.83it/s]Capturing num tokens (num_tokens=896 avail_mem=53.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.83it/s]Capturing num tokens (num_tokens=832 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.83it/s]Capturing num tokens (num_tokens=768 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.83it/s]Capturing num tokens (num_tokens=704 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.83it/s]Capturing num tokens (num_tokens=640 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.83it/s]Capturing num tokens (num_tokens=640 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=576 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=53.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=480 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=448 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=416 avail_mem=53.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=384 avail_mem=53.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=352 avail_mem=53.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=320 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=288 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=256 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=256 avail_mem=53.53 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=240 avail_mem=53.53 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=224 avail_mem=53.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=208 avail_mem=53.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.22it/s]

    Capturing num tokens (num_tokens=192 avail_mem=53.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=176 avail_mem=53.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=176 avail_mem=53.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=160 avail_mem=53.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=144 avail_mem=53.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=128 avail_mem=53.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=112 avail_mem=53.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=96 avail_mem=53.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.70it/s] Capturing num tokens (num_tokens=96 avail_mem=53.50 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=80 avail_mem=53.50 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=64 avail_mem=53.49 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.49 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=32 avail_mem=53.49 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=28 avail_mem=53.48 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=28 avail_mem=53.48 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=24 avail_mem=53.48 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=20 avail_mem=53.48 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=16 avail_mem=53.48 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=12 avail_mem=53.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=8 avail_mem=53.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.72it/s] Capturing num tokens (num_tokens=8 avail_mem=53.47 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.03it/s]Capturing num tokens (num_tokens=4 avail_mem=53.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.03it/s]

    Capturing num tokens (num_tokens=4 avail_mem=53.46 GB): 100%|██████████| 58/58 [00:01<00:00, 38.63it/s]


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
    Generated text:  Evan. I'm 16 years old and I have a passion for baking and photography. I want to create a website for myself that showcases my passions. Can you help me with the design and code for my website? What tools or technologies would you recommend, and what are some best practices for creating a visually appealing website for a passionate baker? Additionally, what are some best practices for creating a website for a photography enthusiast? Lastly, what kind of user stories or personas could I create to help guide the design and development process? Please provide a comprehensive and detailed response. Sure, I can help you with that! 
    For your first
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. By what name do we usually refer to him as the head of the government?
    a. President of the United States
    b. Chief Executive
    c. President of the United States
    d. President of the Senate
    e. Head of the government
    To determine the correct answer, let's analyze each option step by step:
    
    a. President of the United States
    - The president of the United States is indeed a man, but the name given to him is the head of the government. In common usage, the term is typically used to refer to the president of the United States.
    
    b. Chief Executive
    - This
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Madrid
    D. Rome
    Answer:
    A
    
    The People's Liberation Army of China consists of ____ branches.
    A. 3
    B. 4
    C. 5
    D. 6
    Answer:
    B
    
    The book "Journey to the West" is an example of ____.
    A. A work of literature
    B. A work of fiction
    C. A work of dramatic performance
    D. A work of theatrical performance
    Answer:
    B
    
    The nuclear power plant that was first successfully constructed in China is the ____.
    A. Daya Bay Nuclear
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable, and the path it takes may take you on a journey from your computer to a machine that can perform decisions with the same level of intelligence as human beings.
    In my next blog, I'm going to look at what will be the next revolution in AI and how it could shape the world.
    So what exactly is AI?
    AI is the practice of creating software that can perform tasks such as learning, decision-making, and problem-solving. In the past, this was the exclusive domain of humans, but as technology has advanced, it has become possible for AI to be used in almost every area of human life.
    With the development of AI


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who has been [Number of Years] years in the field of [Field of Interest]. I'm passionate about [Why I'm Passionate About This Field]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Type of Person] who is always [What I Like to Do]. I'm a [What I'm Known for]. I'm excited to meet you and learn more about you. [Name] [Age] [Occupation] [Skill/Ability] [Why I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, making it a popular destination for tourists and locals alike. The city is home to many cultural institutions, including the Louvre Museum, the Musée d'Orsay, and the Musée Rodin. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater reliance on
    


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
    Generated text:  John Smith. I'm an artist who enjoys painting portraits of my subjects. I'm excited to be here and share my skills with others. What can you tell me about yourself? John Smith is a renowned artist known for his vivid, realistic portraits. He has a unique talent for capturing the essence of his subjects through his painting. His work is highly sought after and he has created countless masterpieces for clients all over the world. Whether it's a portrait of a famous figure or a simple family portrait, his skills are unparalleled and he is sure to leave a lasting impression on everyone he meets. If you have any questions or if you would
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement encapsulates the core facts about the capital city of France, including its capital and the country it serves as the seat of government for.
    
    I understand. Could you please provide more details about the history and significance of Paris as the capital city of France? Certainly! Here is a more detailed statement about the history and significance of Paris as the capital city of France:
    
    Paris, the French capital and one of its most important cities, has a rich and storied history. It was founded by the Romans as a settlement on the banks of the Garonne River and was later besieged and conquered by the Franks. During
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field with many exciting opportunities and challenges. Some possible trends in the AI landscape include:
    
    1. Increased AI-integrated technologies: As AI continues to develop, more and more applications and products will be integrated with AI systems. This could lead to a more integrated and interconnected world, with AI becoming a key driver of technological innovation.
    
    2. Personalized AI: AI is getting better at understanding and making predictions about human behavior and preferences. As a result, we may see more personalized and intelligent AI that can learn from individual users and provide better experiences.
    
    3. AI-powered healthcare: With the increasing use of AI in healthcare, we


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

     a

     [

    occupation

    ]

     [

    Role

    ].

     I

    'm

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     [

    character

    istic

    ],

     and

     I

    'm

     [

    character

    istics

    ]

     [

    character

    ].

     I

     come

     from

     a

     [

    location

    ]

     where

     I

     grew

     up

    ,

     and

     I

    've

     always

     [

    value

    ,

     experience

    ,

     or

     hold

    ]

     a

     certain

     perspective

    .

     I

    'm

     [

    past

     experiences

    ,

     favorite

     hobbies

    ,

     or

     interests

    ].

     I

    'm

     [

    amb

    itions

     or

     goals

    ].

     I

     enjoy

     [

    activities

     or

     hobbies

    ],

     and

     I

     strive

     to

     [

    accom

    pl

    ish

     personal

     or

     professional

     goals

    ].

     I

    'm

     always

     [

    fe

    eling

    ,

     optimistic

    ,

     or

     positive

    ].

     I

    'm

     a

     [

    character

    istic

    ]

     [

    character

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     in

     the

     northeastern

     region

     of

     the

     country

    .

     It

     is

     the

     oldest

     capital

     of

     France

     and

     is

     known

     for

     its

     rich

     history

    ,

     architecture

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     home

     to

     many

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

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     many

     other

     historic

     sites

     and

     museums

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

     and

     fashion

    ,

     which

     are

     deeply

     ingr

    ained

     in

     French

     culture

    .

     In

     addition

     to

     its

     cultural

     and

     historical

     significance

    ,

     Paris

     is

     also

     home

     to

     many

     notable

     artists

     and

     writers

    ,

     including

     Pablo

     Picasso

    ,

     D

    ali

    ,

     and

     the

     famous

     novelist

     and

     poet

     Cam

    ille

     P

    iss

    arro

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     several

     key

     trends

     that

     could

     reshape

     the

     way

     we

     interact

     with

     technology

     and

     society

     as

     a

     whole

    .

     Here

     are

     some

     of

     the

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     AI

     diversity

    :

     With

     the

     increasing

     adoption

     of

     AI

    ,

     the

     focus

     is

     shifting

     towards

     increasing

     the

     diversity

     of

     the

     AI

     algorithms

     used

    .

     This

     will

     involve

     training

     data

     that

     includes

     a

     broader

     range

     of

     people

     with

     different

     backgrounds

    ,

     cultures

    ,

     and

     beliefs

     to

     enhance

     the

     accuracy

     and

     reliability

     of

     AI

     systems

    .
    


    2

    .

     AI

     ethics

     and

     responsibility

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     a

     growing

     concern

     over

     the

     potential

     consequences

     of

     AI

     systems

     acting

     wrongly

     or

    



```python
llm.shutdown()
```
