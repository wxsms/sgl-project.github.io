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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]


    2026-05-09 07:47:08,146 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 07:47:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=240):  48%|████▊     | 28/58 [00:04<00:02, 14.09it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]

    Compiling num tokens (num_tokens=64):  66%|██████▌   | 38/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.75it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.75it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.75it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.75it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.75it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 32.75it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 32.75it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 32.75it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 32.75it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 32.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  21%|██        | 12/58 [00:00<00:01, 29.69it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  21%|██        | 12/58 [00:00<00:01, 29.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.92it/s]Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.92it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=768 avail_mem=72.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=480 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=448 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.98it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=384 avail_mem=72.05 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=320 avail_mem=72.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=288 avail_mem=72.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.90it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=176 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.91it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.02 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=112 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.76it/s] Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  81%|████████  | 47/58 [00:01<00:00, 47.07it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.88it/s] Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:01<00:00, 41.41it/s]


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
    Generated text:  Anacara. I'm from the Philippines. I used to work for a company. One day, I had to do a public speaking at a town hall meeting. I was very nervous and I was trying to be calm. This is a first time I had to do public speaking at a town hall meeting. In the meeting, I was asked a question on why someone should eat a certain food. I was very surprised because I didn't know the answer and I didn't know what to say. So I tried my best to answer the question. It was very interesting.
    
    OPTIONS:
    [a]. No;
    [b]. Yes;
    Would the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man, and the vice president is a woman. Is this statement true or false?
    To determine whether the statement "The president of the United States is a man, and the vice president is a woman" is true or false, we need to verify the information provided and ensure that all parts of the statement are accurate.
    
    1. Identify the subject and the predicate:
       - The subject is "The president of the United States."
       - The predicate is "is a man."
       - The subject is "the vice president."
       - The predicate is "is a woman."
    
    2. Analyze the information:
       - The president
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the center of which region?
    
    a. Provence-Alpes-Côte d'Azur
    
    b. Île-de-France
    
    c. Atlantic coast
    
    d. Île-de-France
    
    e. North Africa
    
    To determine the capital of France, we need to identify the region where Paris is located. Paris is the capital of France, and it is situated in the center of the Île-de-France region.
    
    Let's analyze each option:
    
    a. Provence-Alpes-Côte d'Azur: This region is known for its wine production, not its capital.
    
    b. Île-de-France:
    ===============================
    Prompt: The future of AI is
    Generated text:  an exciting, but precarious one. While its use can certainly bring a lot of value, the benefits are not universally shared, and AI-driven solutions often come with significant ethical and social implications. As such, we must carefully consider the risks and opportunities that lie ahead as we navigate the challenges posed by AI. This means looking beyond the obvious, embracing more thoughtful, context-aware approaches to decision-making, and avoiding the pitfalls of over-reliance on AI. In this paper, we will explore the potential benefits and risks of AI, and suggest a framework for approaching the future of AI that takes into account the social, ethical, and technological dimensions


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third largest in the world. The city is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for the arts, music, and fashion. The city is known for its annual festivals and events, including the Eiffel Tower Parade and the Louvre Festival. Paris is a popular tourist destination and a cultural hub for the world. It is a major economic center and a major player in the French economy. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: As AI becomes more advanced, it is likely to automate many tasks that were previously done by humans, leading to increased efficiency and productivity.
    
    2. AI will become more integrated with other technologies: AI will continue to be integrated with other technologies, such as machine learning, robotics, and quantum computing, creating a more interconnected and integrated world.
    
    3. AI will become more personalized: As AI becomes more advanced, it is likely to become more personalized, with machines able to learn and adapt to individual users' needs and preferences.
    
    4. AI will become more ethical and
    


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
    Generated text:  [insert name]. I'm a [insert character's profession or ability] with a strong drive and a lot of energy. I'm always up for a challenge, and I'm excited to learn new things and take on new challenges. I'm a creative thinker and I believe in constantly pushing boundaries and trying new things. I'm always looking for new opportunities and I'm open to new experiences. I believe in making a difference and I'm passionate about using my skills to help others. I'm always eager to learn and grow, and I'm always willing to adapt to different situations. I'm confident in my abilities and I'm excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city and a major European cultural and political center.
    
    Paris is known for its rich history, art, and cuisine. It is also a major tourist destination, with a wide range of attractions such as the Louvre, Notre-Dame Cathedral, and the Arc de Triomphe.
    
    France's capital is Paris, a historic city and a major European cultural and political center. It is also a major tourist destination, with a wide range of attractions such as the Louvre, Notre-Dame Cathedral, and the Arc de Triomphe.
    
    This statement succinctly summarizes Paris's importance as a major European city and its role in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be exciting, with many potential developments and applications that could revolutionize our lives. Here are some of the possible future trends in AI:
    
    1. Autonomous vehicles: One of the most promising areas of AI is autonomous vehicles. As technology advances, self-driving cars are likely to become more advanced, with safer driving and reduced accidents.
    
    2. Smart homes: AI is being used in smart homes to control devices, automate tasks, and improve energy efficiency. This could lead to more energy-efficient homes and reduced energy bills.
    
    3. Personalized healthcare: AI is being used to develop algorithms that can analyze medical data and provide personalized treatment plans.


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

    'm

     a

     [

    Job

     Title

    ]

     with

     over

     [

    Number

     of

     Years

    ]

     years

     of

     experience

     in

     [

    Related

     Occupation

    /

    Field

    ].

     I

     have

     a

     passion

     for

     [

    Why

     I

     love

     my

     job

    ]

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     [

    How

     I

     stay

     up

    -to

    -date

     on

     industry

     trends

    ]

     and

     learn

     new

     skills

    .

     I

     thrive

     on

     [

    Why

     it

    's

     important

     to

     me

     to

     be

     a

     part

     of

     this

     organization

    ]

     and

     I

    'm

     excited

     to

     be

     a

     part

     of

     your

     team

    .

     I

    'm

     [

    Your

     Profession

    ].

     I

     look

     forward

     to

     [

    Future

     Goals

    /

    Next

     Steps

    ]

     with

     you

    .

     What

     other

     details

     do

     you

     need

    ?

     [

    Name

    ]

     Here

    's

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     capital

     of

     Australia

    ?
    


    The

     capital

     of

     Australia

     is

     Canberra

    .

     
    


    Please

     note

     that

     both

     cities

     are

     located

     in

     the

     country

     of

     Australia

     and

     are

     significant

     cities

     in

     the

     country

    's

     history

     and

     culture

    .

     The

     capital

     city

     of

     France

    ,

     Paris

    ,

     was

     the

     capital

     of

     France

     from

     

    1

    7

    9

    2

     to

     

    1

    7

    9

    5

     and

     is

     the

     seat

     of

     government

    ,

     parliament

    ,

     and

     the

     main

     city

     of

     the

     French

     Republic

    .

     Canberra

     is

     the

     capital

     city

     of

     Australia

     and

     is

     located

     in

     the

     country

    's

     southeastern

     region

    .

     The

     city

     is

     the

     capital

     of

     the

     Australian

     Capital

     Territory

    ,

     which

     is

     part

     of

     the

     Australian

     Capital

     Territory

    ,

     and

     is

     the

     country

    's

     largest

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     robotics

    ,

     and

     autonomous

     systems

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

     Increased

     automation

    :

     AI

     is

     expected

     to

     play

     an

     increasingly

     important

     role

     in

     autom

    ating

     repetitive

     and

     mundane

     tasks

    ,

     freeing

     up

     human

     beings

     to

     focus

     on

     more

     complex

     and

     creative

     work

    .

     This

     trend

     could

     lead

     to

     more

     widespread

     automation

     of

     jobs

    ,

     as

     machines

     can

     perform

     many

     tasks

     that

     are

     currently

     done

     by

     humans

    .
    


    2

    .

     Enhanced

     intelligence

    :

     AI

     is

     expected

     to

     continue

     to

     improve

     its

     ability

     to

     understand

     and

     interpret

     complex

     natural

     language

    ,

     to

     generate

     rich

     and

     diverse

     responses

     to

     user

     inputs

    ,

     and

     to

     learn

     from

    



```python
llm.shutdown()
```
