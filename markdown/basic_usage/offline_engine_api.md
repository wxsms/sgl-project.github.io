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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]


    2026-05-08 21:30:47,743 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 21:30:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.48it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.04it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s]

    Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.34it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.34it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.34it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.34it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.59it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.59it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.59it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.95it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.95it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.95it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.95it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.95it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.95it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.24it/s]

    Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.24it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 42.47it/s]


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
    Generated text:  Gail and I’m a retired artist, who makes hand-drawn portraits and stills, as well as other hand-painted work. I love making hand-drawn portraits of people and animals, and I’ve been working in the art industry since 1975. I’ve had the pleasure of exhibiting at numerous exhibitions in North America, as well as in Europe, where I’m hoping to get my work into the hands of collectors and art dealers. I'm also hoping to receive some kind of recognition for my work through an award program.
    For me, there are two main reasons why I’m interested in art: The art
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is in charge of the country. If the president is not in charge of the country, then the country will have no one in charge. That is why the president is very important.
    The president is in charge of a country, but he or she is not the only person in charge of the country. There are a number of other important people who run the country. These people are called the executive branch.
    The president and other executive branch members are called the cabinet. There are usually seven to nine cabinet members.
    When the president speaks, the other members of the executive branch are called
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of Spain is Madrid. One day, a French tourist wants to go to Madrid, which is 100 km away from Paris. He decides to take a taxi. If the taxi charges 15 euros for a distance of 1 km, how many euros will the tourist pay for his taxi ride? To determine how much the French tourist will pay for his taxi ride, we need to follow these steps:
    
    1. Identify the total distance to be traveled.
    2. Determine the cost per kilometer.
    3. Calculate the total cost by multiplying the total distance by the cost per kilometer.
    
    The total distance
    ===============================
    Prompt: The future of AI is
    Generated text:  set to impact how we live and work. But how will we navigate this new era of digital transformation? What are the key opportunities and risks? In this talk, we will discuss the trends, opportunities and challenges that lie ahead in AI. We will look at how AI is shaping the world and how we can prepare for this future. Presented by Pham Van Dong, CEO, AI Research and Strategy Institute, BrainMuse. The audience will learn the key ways to develop the capability and competencies of AI practitioners in order to be more competitive in the new era of AI. This is a virtual session. You can register at https://


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? I'm a [insert a few details about yourself, such as your age, gender, occupation, etc.]. I'm looking forward to meeting you and discussing how I can help you. What's your name? What's your job title? What's your company name? What's your experience? What's your favorite hobby? What's your favorite book? What's your favorite movie? What's your favorite food? What's your favorite color
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a cultural and economic hub of France and plays a significant role in the country's political and social life. It is also home to many international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve over time.
    
    3. Increased use of AI in healthcare: AI
    


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
    Generated text:  [insert name] and I am a [insert occupation] with [insert relevant experience] in [insert field]. In my free time, I enjoy [insert hobbies or interests]. Thank you for considering me as a potential character. Let me know if you'd like me to create a profile for you. [insert your name] [insert date] [insert information about your occupation, such as your job title, location, or the purpose of your work, if relevant. ] Hello, my name is [insert name] and I am a [insert occupation] with [insert relevant experience] in [insert field]. In my free time
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as “la Ville de Paris”.
    
    Wow, Paris sounds like a huge city with lots of history! Can you tell me more about the famous landmarks in Paris? Certainly! Paris is home to many iconic landmarks that are world-famous, including the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, Montmartre, Eiffel Tower, Notre Dame, and the Arc de Triomphe. Some other notable landmarks in Paris include the Champs-Élysées, Montmartre, Notre Dame Cathedral, the Basilica of the Sacré-Cœur, the Seine, and the Louvre
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and unpredictable, driven by advances in technology, changes in societal norms, and evolving human values. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and more advanced algorithms will likely be developed that can analyze large datasets more effectively and accurately.
    
    2. Increased integration of AI into consumer electronics: AI is already being integrated into consumer electronics, such as smartphones and smart home devices, but we may see even more integration in the coming years. AI will likely become an essential component of consumer electronics, enabling smarter devices that can


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

    role

    ]

     who

     has

     been

     around

     for

     [

    number

    ]

     years

    .

     I

     have

     a

     [

    specific

     skill

     or

     ability

    ]

     that

     has

     helped

     me

     achieve

     success

     in

     my

     career

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Name

    ]

     is

     a

     [

    mention

     a

     profession

     or

     occupation

    ],

     [

    describe

     your

     role

     or

     contribution

    ].

     I

     enjoy

     [

    mention

     any

     hobbies

    ,

     interests

    ,

     or

     passions

    ],

     and

     I

    'm

     excited

     to

     learn

     more

     about

     your

     experience

     and

     experiences

    .

     What

    's

     the

     most

     interesting

     or

     surprising

     part

     of

     your

     background

     that

     you

     want

     to

     share

    ?

     [

    Name

    ]

     is

     a

     [

    mention

     a

     profession

     or

     occupation

    ],

     [

    describe

     your

     role

     or

     contribution

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historic

     landmarks

    ,

     vibrant

     culture

    ,

     and

     rich

     culinary

     traditions

    .

     It

     is

     also

     the

     seat

     of

     France

    's

     government

    ,

     economy

    ,

     and

     international

     diplomacy

    .

     Visitors

     can

     explore

     the

     city

    ’s

     various

     neighborhoods

    ,

     museums

    ,

     and

     iconic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     Gothic

     architecture

    ,

     museums

    ,

     and

     vibrant

     nightlife

    ,

     making

     it

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     French

     culture

     and

     history

    .

     Can

     you

     provide

     more

     information

     about

     the

     E

    iff

    el

     Tower

     and

     its

     significance

     to

     Paris

    ?

     Sure

    ,

     the

     E

    iff

    el

     Tower

     is

     a

     iconic

     landmark

     in

     Paris

    ,

     France

    .

     It

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     significant

     advancements

     in

     multiple

     areas

    ,

     including

     but

     not

     limited

     to

    :
    


    1

    .

     Enhanced

     machine

     learning

     algorithms

     that

     can

     learn

     from

     vast

     amounts

     of

     data

     and

     improve

     their

     ability

     to

     identify

     patterns

     and

     make

     predictions

    .
    


    2

    .

     Increased

     use

     of

     machine

     learning

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

     to

     improve

     efficiency

    ,

     accuracy

    ,

     and

     decision

    -making

    .
    


    3

    .

     Improved

     algorithms

     for

     natural

     language

     processing

    ,

     translation

    ,

     and

     generation

     that

     can

     handle

     complex

     tasks

     such

     as

     language

     translation

    ,

     text

     summar

    ization

    ,

     and

     sentiment

     analysis

    .
    


    4

    .

     Increased

     use

     of

     AI

     in

     industries

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     energy

     to

     reduce

     costs

    ,

     increase

     efficiency

    ,

     and

     improve

     decision

    -making

    .
    


    



```python
llm.shutdown()
```
