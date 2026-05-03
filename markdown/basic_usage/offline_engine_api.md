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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]


    2026-05-03 15:12:01,084 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 15:12:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.23it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.63it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.01 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.00 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.00 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.00 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.99 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.98 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.98 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.98 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.98 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.96 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.96 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.40it/s]Capturing num tokens (num_tokens=960 avail_mem=74.94 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.40it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.94 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.40it/s]Capturing num tokens (num_tokens=832 avail_mem=74.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.40it/s]Capturing num tokens (num_tokens=832 avail_mem=74.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.03it/s]Capturing num tokens (num_tokens=768 avail_mem=74.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.03it/s]Capturing num tokens (num_tokens=704 avail_mem=74.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.03it/s]Capturing num tokens (num_tokens=640 avail_mem=74.92 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.03it/s]Capturing num tokens (num_tokens=576 avail_mem=74.92 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.91 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:00<00:00, 42.33it/s]Capturing num tokens (num_tokens=480 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.33it/s]Capturing num tokens (num_tokens=448 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.33it/s]Capturing num tokens (num_tokens=416 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.33it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.92 GB):  50%|█████     | 29/58 [00:00<00:00, 42.33it/s]Capturing num tokens (num_tokens=352 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:00<00:00, 42.33it/s]Capturing num tokens (num_tokens=352 avail_mem=74.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=320 avail_mem=74.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=288 avail_mem=74.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=256 avail_mem=74.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=240 avail_mem=74.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=224 avail_mem=74.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.18it/s]Capturing num tokens (num_tokens=224 avail_mem=74.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.78it/s]Capturing num tokens (num_tokens=208 avail_mem=74.89 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.78it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.78it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.78it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.78it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.78it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.82 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.59it/s] Capturing num tokens (num_tokens=80 avail_mem=72.81 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.88it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.88it/s]Capturing num tokens (num_tokens=32 avail_mem=72.80 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.88it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.88it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.88it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.88it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=8 avail_mem=72.78 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.45it/s] Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 40.91it/s]


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
    Generated text:  Marian Cialdini, I am a professor at the University of Pennsylvania’s School of Social Welfare, and I teach at the Caledon Center for Teaching and Learning. I have been awarded the prestigious MacArthur Research Fellowship in 2008, the Fulbright Scholar Award from Canada, the 2012 Loeb Award, the 2014 PEN Translation Award, and the 2014 John Jay Prize in Public Service. My research focuses on the design, use, and implications of persuasion, and has been published in the Journal of Personality and Social Psychology, the American Sociological Review, and
    ===============================
    Prompt: The president of the United States is
    Generated text:  worth $100,000 per year. If he has 5 branches, each with 4 employees, and the president has a say in how much money is paid to each employee, what is the average annual payment per employee for the president?
    
    To determine the average annual payment per employee for the president, we need to follow these steps:
    
    1. Calculate the total number of employees.
    2. Determine the total annual salary for the president.
    3. Find the average annual payment per employee.
    
    First, let's calculate the total number of employees. The president has 5 branches, and each branch has 4 employees.
    ===============================
    Prompt: The capital of France is
    Generated text:  ( )
    A: Paris
    B: London
    C: Brussels
    D: Moscow
    
    To determine the capital of France, we need to understand what the capital of France is. The capital of France is Paris, which is located in the northeastern region of France.
    
    Let's break it down step by step:
    
    1. Identify the capital of France: The capital of France is Paris.
    2. List the options provided:
       A: Paris
       B: London
       C: Brussels
       D: Moscow
    
    3. Compare the capital of France with the given options:
       - Paris is the capital of France, so it matches
    ===============================
    Prompt: The future of AI is
    Generated text:  real, and it’s here to stay, says Stephen Wolfram. On Wednesday, Wolfram introduced the world to his new book, “A New Kind of Science,” which explores how the shape of a set of parameters (a mathematical function) can be used to generate a set of images, and what this might mean for the future of artificial intelligence. It’s a way of thinking that makes sense of the world and could be used to develop more intelligent, precise, and responsible technology. It also has a practical application in teaching kids about the world of math and science.
    My question is: “What can you tell me about the author


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is accurate and brief, providing the essential information about the capital city of France. If you need any further assistance or have additional questions, feel free to ask! 
    
    Note: The capital city of France is Paris. It is the largest city in France and the seat of the French government. The city is located in the northwestern part of the country and is known for its rich history, beautiful architecture, and vibrant culture. Paris is also home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is known for its annual festivals, such
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well
    


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
    Generated text:  [Your Name], and I'm a [Your profession] with a passion for [Your career goal]. I have a Bachelor's degree in [Your subject of study] and have been working in [Your field of interest] for [Your number of years in this field]. I am always looking for opportunities to learn and grow, and I'm eager to share my knowledge with others. What's your story? What inspired you to pursue this field of study or interest? What can I expect to learn or gain from working with me? Also, what do you think my career path would look like? How can I best support my career growth
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the cultural, political, and economic center of France and the world. It is known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music. Paris is a bustling hub of activity and has become synonymous with luxury and extravagance, but it also has a rich history and cultural heritage that makes it a truly fascinating city. Paris is also home to many well-known museums, galleries, and theaters, including the Louvre Museum, the Musée d'Orsay, and the Théâtre de
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly speculative, but here are some possible trends that could happen:
    
    1. Increased automation: As AI becomes more capable, it could gradually replace human jobs in industries like manufacturing, healthcare, and finance. This could lead to job losses but could also create new opportunities for businesses to automate processes.
    
    2. Enhanced decision-making: AI can learn from vast amounts of data to make more accurate predictions and decisions, making it more difficult for human decision-makers to make errors.
    
    3. Improved privacy: As AI becomes more integrated into everyday life, there could be increased concerns about how data is collected, stored, and used. This could lead to the development


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

     am

     a

     [

    Your

     Job

     or

     Title

    ]

     at

     [

    Your

     Company

    ].

     I

     have

     been

     [

    Your

     Professional

     Experience

    ]

     for

     [

    Length

     of

     Time

    ]

     years

    ,

     and

     I

     love

     to

     [

    Your

     Passion

     or

     Hobby

    ].

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     I

     am

     a

     [

    Your

     Personality

     Type

    ]

     and

     I

     am

     always

     ready

     to

     help

     others

     in

     any

     way

     I

     can

    .

     I

     am

     [

    Your

     Character

     Traits

    ].

     I

     am

     excited

     to

     have

     the

     opportunity

     to

     meet

     you

     and

     learn

     more

     about

     your

     company

     and

     what

     you

     do

     for

     a

     living

    .

     Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .

     How

     does

     your

     personal

     background

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     most

     populous

     urban

     area

     in

     Europe

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    9

     million

     people

     according

     to

     the

     

    2

    0

    2

    0

     French

     census

    .

     Paris

     is

     located

     on

     the

     river

     Se

    ine

     and

     on

     the

     Î

    le

     de

     La

     C

    ité

    ,

     which

     is

     an

     island

    .

     It

     is

     an

     important

     political

    ,

     cultural

    ,

     and

     economic

     hub

     in

     the

     world

    .

     Paris

     was

     founded

     in

     the

     

    6

    th

     century

     and

     is

     the

     oldest

     city

     in

     the

     world

    .

     The

     city

     is

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

    .

     It

     is

     also

     home

     to

     many

     notable

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     many

     technological

     advancements

     and

     changes

     in

     the

     way

     that

     AI

     is

     used

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     emphasis

     on

     AI

     ethics

     and

     privacy

    :

     As

     more

     AI

     systems

     become

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increasing

     concerns

     about

     the

     impact

     of

     AI

     on

     people

    's

     privacy

     and

     personal

     data

    .

     AI

     will

     be

     required

     to

     address

     these

     ethical

     concerns

     in

     order

     to

     be

     used

     responsibly

    .
    


    2

    .

     Emer

    gence

     of

     new

     AI

     applications

     and

     industries

    :

     As

     AI

     technology

     advances

    ,

     there

     will

     be

     new

     applications

     and

     industries

     that

     require

     it

    .

     For

     example

    ,

     autonomous

     vehicles

    ,

     smart

     homes

    ,

     and

     virtual

     assistants

     will

     become

     more

     prevalent

     in

     our

     daily

     lives

    



```python
llm.shutdown()
```
