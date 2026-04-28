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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.69it/s]


    2026-04-28 20:18:00,948 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 20:18:00] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.80it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.60it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.53it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.07 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.06 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.06 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.06 GB):   3%|▎         | 2/58 [00:00<00:02, 19.22it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=117.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=117.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.03 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.03 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.02 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.02 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.02 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.01 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.01 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.01 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=116.99 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s]Capturing num tokens (num_tokens=960 avail_mem=117.00 GB):  31%|███       | 18/58 [00:00<00:01, 35.15it/s] Capturing num tokens (num_tokens=960 avail_mem=117.00 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=896 avail_mem=117.00 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=832 avail_mem=117.00 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=768 avail_mem=116.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=704 avail_mem=116.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=704 avail_mem=116.99 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.50it/s]Capturing num tokens (num_tokens=640 avail_mem=116.99 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.50it/s]Capturing num tokens (num_tokens=576 avail_mem=116.99 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.50it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.97 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.50it/s]Capturing num tokens (num_tokens=480 avail_mem=116.99 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.50it/s]Capturing num tokens (num_tokens=480 avail_mem=116.99 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=448 avail_mem=116.98 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=416 avail_mem=116.98 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=384 avail_mem=116.98 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=352 avail_mem=116.97 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=352 avail_mem=116.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=320 avail_mem=116.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=288 avail_mem=116.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.98it/s]

    Capturing num tokens (num_tokens=256 avail_mem=116.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=240 avail_mem=116.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=240 avail_mem=116.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=224 avail_mem=116.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=208 avail_mem=116.95 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=192 avail_mem=116.95 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=176 avail_mem=116.95 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=176 avail_mem=116.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=160 avail_mem=116.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=144 avail_mem=116.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.53it/s]

    Capturing num tokens (num_tokens=128 avail_mem=116.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=112 avail_mem=116.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=112 avail_mem=116.94 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=96 avail_mem=116.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.54it/s] Capturing num tokens (num_tokens=80 avail_mem=116.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=64 avail_mem=116.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=48 avail_mem=116.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.54it/s]Capturing num tokens (num_tokens=48 avail_mem=116.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.37it/s]Capturing num tokens (num_tokens=32 avail_mem=116.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.37it/s]Capturing num tokens (num_tokens=28 avail_mem=116.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.37it/s]

    Capturing num tokens (num_tokens=24 avail_mem=116.91 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.37it/s]Capturing num tokens (num_tokens=20 avail_mem=116.91 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.37it/s]Capturing num tokens (num_tokens=20 avail_mem=116.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=16 avail_mem=116.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=12 avail_mem=116.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=8 avail_mem=116.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.72it/s] Capturing num tokens (num_tokens=4 avail_mem=116.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=4 avail_mem=116.90 GB): 100%|██████████| 58/58 [00:01<00:00, 34.91it/s]


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
    Generated text:  Ryan and I am a software developer. I have been coding for as long as I can remember and I started learning programming when I was in 7th grade. I am now coding for a living and have been coding for many years. I have experience with HTML, CSS, and JavaScript. I am currently working on a project that involves creating an e-commerce website. Can you please give me some advice on how to get started in this field?
    Sure, I'd be happy to help you get started in the field of coding. Here are a few tips:
    1. Start with the basics: Take a course or read a book on
    ===============================
    Prompt: The president of the United States is
    Generated text:  a major U.S. political office, held once every four years, and is to be filled by election. The candidate with the most votes wins the election, and the president typically serves a four-year term.
    The president of the United States is elected by popular vote and is re-elected by a plurality of the vote. In a presidential election, voters rank the candidates in order of their choice. The winner of the popular vote is declared the victor. In a Senate election, the total number of votes received by each candidate is tallied, and the candidate with the most votes is declared the winner. In a presidential election, the candidate
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris
    
    B: Marseille
    
    C: Paris
    
    D: Lyon
    
    To determine the capital of France, let's break down the information given and analyze each option step by step.
    
    1. **Identify the capital of France:**
       - The capital of France is Paris. This is a well-known fact.
    
    2. **List the options:**
       - A: Paris
       - B: Marseille
       - C: Paris
       - D: Lyon
    
    3. **Match the capital with the options:**
       - We need to identify which of these options is the capital of France.
    
    4. **
    ===============================
    Prompt: The future of AI is
    Generated text:  in a crisis, and no one can predict the future more accurately than the AI that is developing it. Researchers at The University of Melbourne are focusing on making AI more ethical and responsible by introducing new strategies to create safer and more efficient AI.
    AI is making society safer by reducing the frequency of fraud and cyber threats. Yet, AI is also making the world a more dangerous place, by reducing human decision-making ability and leading to a breakdown in trust and cooperation.
    AI is responsible for a lot of security breaches, most notably the widely publicized hacking of the Equifax data breach. By focusing on building better AI, researchers at The University of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower and vibrant culture. It is the largest city in France and the seat of the French government. Paris is also a major tourist destination and a popular destination for French cuisine and fashion. The city is home to many historical landmarks and museums, including the Louvre and the Notre-Dame Cathedral. Paris is a cultural and economic hub of France and a major tourist destination. The city is also known for its rich history and diverse population. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is a major hub for international business and trade,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate tasks and processes, leading to increased efficiency and productivity. This will require significant changes in how we design and build AI systems, as well as how we interact with them.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be increased concerns about privacy and security. This will require significant investment in new technologies and practices to protect against these risks.
    
    3. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making processes, leading to more complex and nuanced decision-making.
    


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
    Generated text:  [Your Name], and I'm a [Occupation] with [Number of Years in Industry]. I've had a passion for [My Major or Area of Expertise] for [Number of Years] years, and I'm currently [Status] in this role. I'm always eager to learn and stay updated on the latest developments in my field, and I'm constantly striving to push the boundaries of what's possible in my work. I'm confident in my abilities and am always looking for the next great opportunity to contribute to my field. I'm excited to have the opportunity to work alongside you, [Person's Name]. Thank you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The answer is: Paris is the capital city of France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by several trends, including:
    
      1. Increased development of more advanced AI technologies: AI will continue to be developed and refined, with more advanced technologies being developed to enhance the performance of machines and make them more capable of performing tasks that previously required human intelligence.
      2. Integration of AI with other technologies: AI will continue to be integrated with other technologies, such as machine learning, natural language processing, and computer vision, to create more complex and advanced systems.
      3. Autonomous and intelligent machines: AI is expected to become more autonomous and intelligent, with machines being able to learn and adapt on their


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

    ...

     (

    Insert

     the

     character

    's

     name

    )

     and

     I

     specialize

     in

    ...

     (

    insert

     the

     main

     area

     or

     profession

     of

     the

     character

    )

     and

     my

     work

     revolves

     around

    ...

     (

    insert

     the

     main

     activity

     or

     task

     of

     the

     character

    )
    


    I

     am

     an

     experienced

     [

    insert

     the

     main

     area

     or

     profession

     of

     the

     character

    ].

     I

     am

     confident

     in

     my

     abilities

     and

     have

     developed

     a

     track

     record

     of

     excellence

     in

     [

    insert

     the

     main

     activity

     or

     task

     of

     the

     character

    ].

     I

     am

     eager

     to

     learn

     and

     expand

     my

     skills

    ,

     and

     I

     am

     always

     eager

     to

     contribute

     my

     knowledge

     and

     expertise

     to

     the

     world

     around

     me

    .

     I

     am

     a

     team

     player

    ,

     comfortable

     collaborating

     with

     others

     and

     always

     striving

     to

     work

     towards

     a

     common

     goal

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    ,

     with

     a

     population

     of

     over

     

    2

     million

     people

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     located

     on

     the

     banks

     of

     the

     River

     Se

    ine

     and

     the

     Se

    ine

    -B

    re

    ton

    ne

    ,

     and

     is

     home

     to

     the

     seat

     of

     the

     French

     government

    ,

     the

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

    ,

     among

     other

     landmarks

    .

     France

    ’s

     capital

     is

     considered

     a

     cultural

     and

     economic

     hub

    ,

     and

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

     vibrant

     art

     and

     architecture scene

    .

     It

     is

     a

     major

     tourist

     destination

    ,

     and

     is

     home

     to

     many

     of

     France

    ’s

     most

     famous

     landmarks

    ,

     including

     Notre

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     rapid

     evolution

     and

     convergence

     of

     technologies

    ,

     systems

    ,

     and

     applications

    .

     Some

     potential

     trends

     that

     could

     emerge

     include

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     data

     analysis

     and

     decision

    -making

     in

     various

     fields

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     marketing

    .
    


    2

    .

     Rise

     of

     AI

    -based

     models

     for

     personalized

     medicine

     and

     drug

     discovery

    .
    


    3

    .

     Expansion

     of

     AI

    -driven

     automation

     in

     manufacturing

    ,

     transportation

    ,

     and

     supply

     chain

     management

    .
    


    4

    .

     Emer

    gence

     of

     AI

    -powered

     chat

    bots

     and

     virtual

     assistants

     to

     improve

     customer

     service

     and

     customer

     engagement

    .
    


    5

    .

     Growth

     of

     AI

     for

     autonomous

     vehicles

     and

     robotics

     in

     transportation

     and

     industrial

     applications

    .
    


    6

    .

     Increasing

     use

     of

     AI

     for

     cybersecurity

     and

     threat

    



```python
llm.shutdown()
```
