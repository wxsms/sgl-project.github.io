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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.08it/s]


    2026-04-08 09:12:13,261 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 09:12:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.54it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.64it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.18it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.21it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]

    Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s] Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  50%|█████     | 29/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.36it/s]

    Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  50%|█████     | 29/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.73it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.50it/s]

    Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.57it/s]

    Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.57it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 40.70it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 37.59it/s]


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
    Generated text:  Adam. I have a summer job as a food delivery driver. I get up early in the morning, get my driver’s license, and start driving. I have a pickup station in my area. My driver’s license is good for all states in the United States.
    
    I deliver food for local restaurants, grocery stores, and customers who are driving cars with delivery carts. I don’t have to be in the same city as my customers. I can be on my way to their car and pick them up when they arrive.
    
    I have a good understanding of my job, but I’m not always sure how to explain my job to others.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public official who is the head of the executive branch of the federal government. This public office typically lasts for two years and is incumbent for a term that is renewable. There are currently 100 presidencies in the United States. 
    
    The governor of New Hampshire is the state's most powerful person, and the president of the United States is the most powerful person in the entire United States. 
    
    Who holds the most power in the United States?
    
    To determine who holds the most power in the United States, we need to evaluate the roles of the president and the governor of New Hampshire.
    
    1. **President of the United States
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. France is a country, not a city. What does it mean when it is said that Paris is the capital of France? 
    
    A) It is the capital of France only in French culture.  
    B) It is the capital of France only in French history.  
    C) It is the capital of France only in French culture.  
    D) It is the capital of France in French history.
    
    To determine the correct interpretation of what it means when it is said that Paris is the capital of France, let's analyze the statement step by step.
    
    1. **Identify the key phrase**: The phrase "Paris is the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  young and future-proof. With the latest advancements in AI, it is important for individuals to understand how it will change society and its applications. AI can help in a number of ways, such as improving efficiency in business operations and reducing human errors. It can also be used to help in the development of new technologies that will impact society in significant ways. AI is currently in the early stages of development, and it is expected to continue to evolve and improve over time. Ultimately, AI will play a significant role in shaping the future of society.
    AI is a rapidly growing field that has the potential to revolutionize many areas of human life. From


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a major economic center in France. It is home to many famous museums, theaters, and restaurants. The city is also known for its annual Eiffel Tower Festival, which attracts millions of visitors each year. Paris is a vibrant and diverse city with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a growing emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This will require developers to take a more thoughtful approach to designing and implementing AI systems that are fair and equitable.
    
    2. Greater integration with human intelligence: AI systems will
    


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
    Generated text:  [Name], and I'm [Age]. I'm a [major or minor] of the [major or minor] I serve in, and I've been with [role] for [x years or [x months] years]. I'm a [major or minor] with an [industry or personal interest]. I have an impressive [occupation-related qualification] and I love to [a personal interest or hobby]. I'm always ready to learn and I'm [what you'd say you're like]. I'm passionate about [my main interest], and I'm always looking for new ways to [a hobby or interest]. I have a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, the largest and most populous country in Europe. It is located in the north-central region of the country, on the Île de France, with the Seine River flowing through it. Paris is known for its beautiful architecture, vibrant culture, and rich history, including the Opéra Garnier, the Louvre Museum, and the Notre-Dame Cathedral. It is a major tourist destination, home to many world-renowned landmarks, museums, and cultural institutions. The city is also known for its fashion industry and its contributions to art, literature, and science. Despite its modernity and diversity
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of potential trends and developments, including:
    
    1. Increased automation: As AI continues to become more sophisticated, it's likely to become more integrated into many tasks, such as manufacturing, transportation, and service. This automation will likely lead to increased efficiency, cost savings, and overall productivity.
    
    2. Emotional intelligence: While AI is currently trained on large datasets and can perform tasks with high accuracy, it's important that these systems are trained on a wide range of emotions as well. This includes recognizing and responding to human emotions, such as sadness, anger, and happiness.
    
    3. Quantum computing: The advancement of quantum computing


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

    ].

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ].

     I

     was

     born

     in

     [

    insert

     date

     of

     birth

    ]

     in

     [

    insert

     location

    ].

     I

    'm

     [

    insert

     age

    ].

     I

    'm

     a

     [

    insert

     personality

     trait

    ]

     person

    .

     I

    'm

     a

     [

    insert

     personal

     characteristic

    ].

     I

    'm

     [

    insert

     occupation

     or

     profession

    ].

     I

     enjoy

     [

    insert

     hobby

     or

     activity

    ].

     I

     love

     [

    insert

     activity

    ],

     and

     it

    's

     my

     favorite

     thing

     to

     do

    .

     I

    'm

     [

    insert

     occupation

     or

     profession

    ]

     and

     I

     enjoy

     [

    insert

     hobby

     or

     activity

    ].

     I

     like

     [

    insert

     activity

    ]

     and

     it

    's

     my

     favorite

     thing

     to

     do

    .

     I

    'm

     [

    insert

     occupation

     or

     profession

    ],

     and

     I

     enjoy

     [

    insert

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     iconic

     city

     renowned

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     rich

     history

    .

     It

     was

     founded

     in

     the

     

    8

    th

     century

     and

     has

     remained

     a

     major

     European

     capital

     since

     the

     

    1

    2

    th

     century

    ,

     hosting

     numerous

     cultural

     and

     political

     events

     throughout

     the

     centuries

    .

     Paris

     is

     known

     for

     its

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

     Dame

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

     its

     role

     in

     French

     culture

     and

     history

    .

     The

     city

     is

     home

     to

     a

     diverse

     population

     of

     millions

     of

     people

    ,

     which

     contributes

     to

     its

     vibrant

     social

     and

     cultural

     life

    .

     France

    's

     capital

     city

    ,

     Paris

    ,

     is

     a

     major

     hub

     for

     international

     diplomacy

    ,

     media

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     a

     combination

     of

     emerging

     technologies

     and

     changes

     in

     the

     business

     landscape

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Deep

     learning

     and

     reinforcement

     learning

    :

     These

     are

     two

     key

     areas

     of

     AI

     that

     are

     expected

     to

     drive

     significant

     growth

     in

     the

     coming

     years

    .

     Deep

     learning

     is

     based

     on

     neural

     networks

     and

     can

     be

     used

     for

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    .

     Rein

    forcement

     learning

     is

     a

     type

     of

     AI

     that

     involves

     the

     development

     of

     algorithms

     that

     can

     learn

     from

     trial

     and

     error

     and

     improve

     over

     time

    .

     Both

     of

     these

     areas

     have

     the

     potential

     to

     revolution

    ize

     industries

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     Autonomous

     vehicles

    



```python
llm.shutdown()
```
