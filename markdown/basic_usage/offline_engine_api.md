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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.90it/s]


    2026-04-08 14:20:22,283 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 14:20:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.62it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.62it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.62it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.62it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.62it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.62it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.62it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.62it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.62it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.91it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.91it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.96it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.60it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.61it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.02it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=132.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=132.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=132.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=132.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=132.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=132.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=132.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=132.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=132.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=132.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=132.36 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=132.36 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=132.36 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=132.35 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=132.35 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=132.35 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=132.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=132.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=132.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=132.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=132.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.24it/s]

    Capturing num tokens (num_tokens=960 avail_mem=132.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.24it/s] Capturing num tokens (num_tokens=960 avail_mem=132.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.73it/s]Capturing num tokens (num_tokens=896 avail_mem=132.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.73it/s]Capturing num tokens (num_tokens=832 avail_mem=132.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.73it/s]Capturing num tokens (num_tokens=768 avail_mem=132.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.73it/s]Capturing num tokens (num_tokens=704 avail_mem=132.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.73it/s]Capturing num tokens (num_tokens=640 avail_mem=132.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.73it/s]Capturing num tokens (num_tokens=640 avail_mem=132.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=576 avail_mem=132.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=512 avail_mem=132.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=480 avail_mem=132.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.57it/s]

    Capturing num tokens (num_tokens=448 avail_mem=132.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=416 avail_mem=132.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=416 avail_mem=132.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.19it/s]Capturing num tokens (num_tokens=384 avail_mem=132.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.19it/s]Capturing num tokens (num_tokens=352 avail_mem=132.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.19it/s]Capturing num tokens (num_tokens=320 avail_mem=132.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.19it/s]Capturing num tokens (num_tokens=288 avail_mem=132.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.19it/s]Capturing num tokens (num_tokens=256 avail_mem=132.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.19it/s]Capturing num tokens (num_tokens=256 avail_mem=132.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=240 avail_mem=132.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=224 avail_mem=132.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.25it/s]

    Capturing num tokens (num_tokens=208 avail_mem=132.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=192 avail_mem=132.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=176 avail_mem=132.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=176 avail_mem=132.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=160 avail_mem=132.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=144 avail_mem=132.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=128 avail_mem=132.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=112 avail_mem=132.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=96 avail_mem=132.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.09it/s] Capturing num tokens (num_tokens=96 avail_mem=132.27 GB):  81%|████████  | 47/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=80 avail_mem=132.26 GB):  81%|████████  | 47/58 [00:01<00:00, 43.52it/s]

    Capturing num tokens (num_tokens=64 avail_mem=132.26 GB):  81%|████████  | 47/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=48 avail_mem=132.26 GB):  81%|████████  | 47/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=32 avail_mem=132.25 GB):  81%|████████  | 47/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=28 avail_mem=132.25 GB):  81%|████████  | 47/58 [00:01<00:00, 43.52it/s]Capturing num tokens (num_tokens=28 avail_mem=132.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=24 avail_mem=132.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=20 avail_mem=132.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=16 avail_mem=132.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=12 avail_mem=132.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=8 avail_mem=132.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.97it/s] Capturing num tokens (num_tokens=8 avail_mem=132.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=4 avail_mem=132.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.51it/s]

    Capturing num tokens (num_tokens=4 avail_mem=132.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.99it/s]


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
    Generated text:  Olivia and I am writing for the National Geographic Society and am trying to find some cool facts and tidbits for the magazine. I'm working on a feature story on the weather, but I'm not sure where to start. Can you provide some interesting facts about the weather? 
    Let's dive into some fascinating facts about the weather! Here are some interesting tidbits to keep you in the loop:
    1. The average temperature in the United States is around 72°F (22°C).
    2. The coldest temperature recorded in the United States was -70°F (-53°C) on January 7, 1
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 43 years old. How many years ago was the president the age of a 25 year old? We can set up a ratio to find the number of years ago when the president was 25 years old. We know that 43 - 25 = 18 years ago, and the president was 25 years old. Therefore, the president was 18 years old 18 years ago.
    
    Let's break it down into a more detailed explanation:
    
    1. We know the current age of the president is 43 years.
    2. We need to find out how many years ago
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. A: Paris B: Vienna C: London D: Moscow
    A: Paris
    
    The capital of France is Paris. It is known for its beautiful architecture, rich history, and vibrant culture. Paris is also a popular tourist destination, with its famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is located in the northern part of the country and is the largest city in France by population. 
    
    Let's briefly go through each option:
    
    B: Vienna - Capital of Austria, not France
    C: London - Capital of England, not France
    D: Moscow - Capital
    ===============================
    Prompt: The future of AI is
    Generated text:  moving from data to applications, with applications being used for various purposes such as medical diagnosis, weather forecasting, or personalized learning. The early success of AI has led to a rapid development of the field, with many promising technologies emerging. However, the field is still in its infancy and many questions remain unanswered. In this paper, we discuss the future of AI and the key challenges we face. We believe that these challenges will play a significant role in shaping the future of AI, and we urge researchers to continue to explore and develop AI technologies that can help us solve real-world problems. The challenge of providing context, including the development and applications of


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name], and I've been driving for [Number of Years] years. I'm [Favorite Hobby] and I enjoy [Favorite Activity]. I'm [Favorite Food] and I love [Favorite Book]. I'm [Favorite Movie] and I've been watching [Favorite TV Show] for [Number of Seasons] years. I'm [Favorite Sport] and I play [Favorite Game]. I'm [Favorite Music] and I love [Favorite Album]. I'm [Favorite Book] and I've been reading
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. It is also known for its cuisine, including French cuisine, which is famous for its rich flavors and use of fresh ingredients. Paris is a city that is a melting pot of different cultures and it is a city that is always on the move, with many new developments taking place every day. The city is also known for its fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could be expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with the goal of improving the accuracy and efficiency of diagnoses and treatments.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI
    


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
    Generated text:  [Your Name], and I am a [age] year old [job title] with a degree in [relevant subject]. I enjoy [majority of personal interests, such as hobbies, travel, music, sports, etc.]. I am a [career goal or significant achievement] and am currently [status, such as unemployed, employed, etc.]. I have [number of years of experience, if any, in the industry or field you are working in]. In your view, what are your strengths and weaknesses?
    You are the [job title] at [company name], where you have been working since [date]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights, where its historical significance and cultural importance are evident in its architecture, art, and cuisine. Paris is also known as the "City of Love" due to its romantic history and the presence of some of the world's most famous landmarks. It is a popular tourist destination with many popular attractions, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is a vibrant and diverse city with a rich cultural heritage and a strong economy. Its status as the capital of France is one of the most important in the world, and it continues to attract visitors from around the globe
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of rapid growth and innovation, driven by the increasing complexity and availability of data, as well as advances in computing power and machine learning techniques. Here are some potential future trends in AI:
    
    1. Increased use of AI for tasks that were previously done by humans, such as customer service, healthcare, and manufacturing.
    
    2. AI will become more capable of performing tasks that require higher levels of expertise and creativity, such as visual and natural language processing.
    
    3. AI will become more widely integrated into everyday life, such as in devices like smart homes, self-driving cars, and virtual assistants.
    
    4. AI will continue to develop


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

    job

     title

    ]

     in

     the

     [

    industry

    ]

     field

    .

     I

    'm

     passionate

     about

     [

    what

     interests

     you

     about

     your

     job

    /

    field

    ]

     and

     always

     strive

     to

     [

    how

     you

     aim

     to

     grow

     in

     this

     field

    ].

     I

     enjoy

     [

    what

     you

     do

     for

     fun

    /

    activities

    ],

     and

     I

     believe

     that

     [

    why

     you

     excel

     at

     your

     work

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     [

    what

     you

     believe

     your

     role

     should

     be

    ]

     and

     I

    'm

     constantly

     trying

     to

     [

    how

     you

     believe

     you

     can

     contribute

     to

     the

     team

    ].

     Whether

     it

    's

     [

    what

     you

     do

    ],

     [

    what

     you

    're

     passionate

     about

    ],

     [

    what

     you

     do

     for

     fun

    ],

     or

     [

    what

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     city

     of

     light

     and

     the

     city

     of

     love

    ,

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     the

     largest

     city

     in

     the

     world

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

     million

     people

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     a

     popular

     tourist

     destination

    ,

     with

     many

     visitors

     from

     around

     the

     world

    .

     The

     city

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     home

     to

     many

     historical

     and

     cultural

     landmarks

    .

     In

     addition

     to

     its

     landmarks

    ,

     Paris

     is

     also

     known

     for

     its

     cuisine

    ,

     art

    ,

     and

     music

    .

     The

     city

     is

     also

     home

     to

     the

     famous

     French

     Quarter

    ,

     known

     for

     its

     cob

    ble

    stone

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increasing

     automation

     and

     self

    -learning

    :

     AI

     is

     expected

     to

     become

     increasingly

     capable

     of

     performing

     tasks

     that

     were

     once

     done

     by

     humans

    ,

     such

     as

     language

     translation

    ,

     image

     recognition

    ,

     and

     autonomous

     vehicles

    .

     Self

    -learning

     algorithms

     will

     be

     able

     to

     continuously

     improve

     and

     adapt

     to

     new

     data

    .
    


    2

    .

     Improved

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     there

     will

     be

     an

     increasing

     emphasis

     on

     protecting

     user

     privacy

     and

     ensuring

     that

     data

     is

     not

     being

     mis

    used

     or

     mis

    used

    .

     This

     will

     include

     measures

     to

     prevent

     data

     breaches

     and

     ensure

     that

     AI

     systems

     are

     secure

    .
    


    3

    .

     Greater

     integration

     with

     human

     intelligence

    :

     AI

     is

     expected

    



```python
llm.shutdown()
```
