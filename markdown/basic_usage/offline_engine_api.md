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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.19it/s]


    2026-04-09 10:04:37,655 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 10:04:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.68it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.68it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.68it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.97it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.06it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.70it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.11it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  21%|██        | 12/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.28it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.28it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.28it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.03it/s]

    Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.03it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.43it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.43it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]

    Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=96 avail_mem=120.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.30it/s] Capturing num tokens (num_tokens=96 avail_mem=120.22 GB):  81%|████████  | 47/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=80 avail_mem=120.22 GB):  81%|████████  | 47/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=64 avail_mem=120.19 GB):  81%|████████  | 47/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=48 avail_mem=118.99 GB):  81%|████████  | 47/58 [00:01<00:00, 44.62it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  81%|████████  | 47/58 [00:01<00:00, 44.62it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.59it/s] Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 39.80it/s]


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
    Generated text:  Gerald and I’m the founder of the Oasis Institute. It is the first school in the world dedicated to the development of emotional intelligence (EQ) and learning to navigate the turbulent world of the modern world. I am passionate about how the human experience can be impacted by our emotional well-being and the positive changes in our lives can be a catalyst for us to change the world for the better. I believe in giving people the tools to make choices, they need to be made with understanding and compassion. I will continue to do my part in helping you get what you need. My name is Gerald and I’m the founder of the Oasis Institute.
    ===============================
    Prompt: The president of the United States is
    Generated text:  scheduled to visit the United Kingdom for a two-week tour of the country. In order to provide context, the president would like to know the current population of the United Kingdom. One of the key U. S. cities in the United Kingdom is London. How many people are there in London? To determine the population of London, we need to follow these steps:
    
    1. Identify the capital city of the United Kingdom, which is London.
    2. Research the population of London.
    
    Step 1: Identify the capital city of the United Kingdom.
    The capital city of the United Kingdom is London.
    
    Step 2: Research the population of London
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. The capital of France is Paris.
    
    The capital of France is Paris. Paris is the capital city of France. It is located in the northern part of the country, near the English Channel. Paris is known for its rich history, famous landmarks, and beautiful architecture, including the Eiffel Tower. It is the second-largest city in France by population, after Paris, and the 17th-largest by area. The city has a diverse culture, including iconic landmarks such as the Louvre Museum and Notre-Dame Cathedral. Paris is also home to many international institutions and organizations, including UNESCO and the French Academy of Sciences.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the data. The ability to access and analyse large amounts of data is one of the key factors driving the development of AI. However, without a good understanding of the data, we are unlikely to achieve the optimal solution.
    Data is complex, and a lot of it can be unstructured. For example, if you want to develop an AI system, you must access a variety of data sources to be able to learn from the real world. This data may be numeric, structured, unstructured or semi-structured.
    To understand the data better, you can use some methods like:
    - Data visualisation
    - Data cleaning
    - Data


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few key points about your background, education, or experience that you can share with me]. I'm looking forward to meeting you and discussing how I can help you. What's your name? What's your job title? What's your company name? What's your job title? What's your company name? What's your job title? What's your company name? What's your job title? What's your company name?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. It is also a popular tourist destination, with millions of visitors each year. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to customer service. This could lead to the automation of many jobs, freeing up time for humans to focus on more complex tasks.
    
    2. AI ethics and privacy: As AI becomes more integrated into our lives, there will be a growing concern about its impact on society. This includes issues such as bias in AI algorithms, privacy concerns, and
    


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
    Generated text:  [Name], and I'm a [age] year old [gender] [race]. I'm an ambitious and determined individual who is passionate about [insert something you do or are passionate about]. I'm also an educator, and I love teaching and helping others. I believe in the power of knowledge and the importance of empathy. I'm always looking for new ways to improve myself and make a difference in the world. My mission is to inspire and motivate others to do the same. I am constantly seeking new opportunities to learn and grow, and I'm always looking for the next challenge to overcome. Thank you for considering me for a potential
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks like the Eiffel Tower and Notre-Dame Cathedral, and for its rich history and cultural influences. French cuisine, including its famous pastries and regional specialties, is also a prominent aspect of the city's culinary scene. Additionally, Paris is a major tourist destination known for its world-renowned museums, art galleries, and fashion shows. As the largest city in France, Paris plays a significant role in the country's economy and culture. The city is home to several important cultural institutions, including the Louvre Museum and the Notre-Dame Cathedral, and has become a hub for various forms of art,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some possible trends that are currently being explored and discussed include:
    
    1. Increased reliance on AI for autonomous driving: As autonomous vehicles become more prevalent, the demand for AI algorithms and technologies to control and manage these vehicles will grow. This will lead to more complex AI systems with more sophisticated decision-making capabilities.
    
    2. Greater integration of AI into everyday life: As AI becomes more advanced and accessible, more people will be able to interact with AI-powered systems in a more seamless and automated way. This could lead to significant changes in how we work, communicate, and access information.
    
    3. Enhanced AI for healthcare and medicine: AI


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

    ],

     and

     I

    'm

     a

     professional

     software

     engineer

     with

     over

     

    1

    0

     years

     of

     experience

     in

     programming

     and

     software

     development

    .

     I

     have

     a

     diverse

     range

     of

     skills

     and

     expertise

    ,

     including

     Python

    ,

     JavaScript

    ,

     Java

    ,

     C

    ++,

     and

     SQL

    ,

     and

     I

     enjoy

     working

     on

     projects

     that

     challenge

     me

     to

     innovate

     and

     solve

     complex

     problems

    .

     I

     am

     a

     strong

     communicator

    ,

     able

     to

     collaborate

     with

     people

     from

     different

     backgrounds

     and

     cultures

    ,

     and

     I

     am

     eager

     to

     learn

     and

     grow

     in

     my

     field

    .

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

     develop

     my

     skills

     and

     stay

     up

     to

     date

     with

     the

     latest

     trends

     and

     technologies

     in

     the

     industry

    .

     Thank

     you

     for

     considering

     me

     as

     a

     potential

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     which

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    .

     
    


    To

     elaborate

    ,

     Paris

     is

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

     and

     is

     home

     to

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

     many

     other

     famous

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     vibrant

     nightlife

    ,

     fashion

     scene

    ,

     and

     rich

     history

    .

     
    


    Paris

     was

     founded

     in

     

    7

    9

    3

     and

     became

     the

     capital

     of

     France

     in

     

    8

    6

    4

    ,

     and

     is

     considered

     one

     of

     the

     most

     historic

     and

     significant

     cities

     in

     the

     world

    .

     The

     city

     is

     also

     known

     for

     its

     education

    ,

     research

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     several

     trends

     that

     will

     shape

     how

     it

     is

     used

     and

     developed

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Increased

     focus

     on

     ethics

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     an

     increased

     emphasis

     on

     ethical

     considerations

     and

     principles

    .

     This

     may

     lead

     to

     the

     development

     of

     new

     ethical

     guidelines

     and

     standards

     for

     AI

    .
    


    2

    .

     Greater

     integration

     with

     human

     decision

    -making

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     human

     decision

    -making

     in

     many

     applications

    ,

     such

     as

     in

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     This

     may

     lead

     to

     the

     development

     of

     new

     AI

     systems

     that

     can

     make

     more

     accurate

     and

     timely

     decisions

     based

     on

     human

     input

    .
    


    3

    .

     Automation

     and

     self

    -learning

    :

     One

     of

    



```python
llm.shutdown()
```
