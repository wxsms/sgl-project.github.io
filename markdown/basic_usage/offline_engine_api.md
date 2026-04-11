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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.41it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.39it/s]


    2026-04-11 02:11:09,524 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 02:11:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.01it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.12it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.55it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.37it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.34 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.12 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=119.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.99 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.99 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]

    Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s] Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.35it/s]Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.35it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.35it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.35it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.35it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.35it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.73it/s]

    Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=384 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.13it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.13it/s]

    Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.01it/s] Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 44.15it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  81%|████████  | 47/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  81%|████████  | 47/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.55it/s] Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 39.74it/s]


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
    Generated text:  Vladimir and I am an engineering student from China. I'm planning a trip to Barcelona, Spain for an academic year. I was wondering if you could provide some detailed information about the best time of the year to visit Barcelona, and what the main attractions and activities are in the city? Additionally, could you share any unique experiences you might have while traveling there, and what would be the most memorable experience you have? Finally, please let me know if you have any recommendations for local restaurants or bars in Barcelona that are worth visiting during your trip.
    Certainly! Barcelona is a beautiful city with a rich history and vibrant culture. The best time to
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military ships to keep in the ocean to support his country. He has an unlimited supply of ships, each with a capacity of at least 5000 square feet and a maximum weight capacity of 10000 pounds. Each ship is capable of transporting a certain amount of people, and the president wants to ensure that the total weight of the people transported is less than or equal to 500000 pounds. If the president has 24000 square feet of area available for storage and 80000 pounds of weight capacity, what is the maximum number
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is famous for its landmarks. The city is also known for its art, literature, food, and music. Paris is a city with a long history. During the Middle Ages, Paris was the capital of France. In 1800, Napoleon Bonaparte took the city to become the capital of France. In the 19th century, the city became a famous place for art and music. Today, Paris is a very busy city with lots of different restaurants, shops, and theaters.
    
    Write a short passage about Paris from your point of view. As I stepped out of my car and onto the narrow
    ===============================
    Prompt: The future of AI is
    Generated text:  here. But it’s not just about machines taking over human jobs. It’s about how machines are used to solve problems and how we use AI in our homes and businesses. So, what are the key elements that define the future of AI?
    The future of AI is not just about machines taking over human jobs. It’s about how machines are used to solve problems and how we use AI in our homes and businesses. The key elements that define the future of AI are:
    1. Robust Data: Machine learning requires massive amounts of data to learn and improve. With the growth of data, the amount of data that can be used to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity], and I'm always looking for ways to expand my skills and knowledge. What's your favorite book or movie? I love [book or movie], and I'm always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many famous museums and theaters. The city is also known for its cuisine, with its famous dishes such as croissants, beignets, and escargot. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is both
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation: AI is likely to become more integrated into various industries, leading to increased automation of tasks and processes. This could result in the creation of new jobs, but also potentially leading to job displacement.
    
    2. Enhanced human AI: As AI becomes more advanced, it is likely to become more human-like, with the ability to learn and adapt to new situations. This could lead to the development of more sophisticated and nuanced AI that can better understand and respond to human emotions and behaviors.
    
    3. AI ethics and governance: As AI becomes more integrated into our daily lives, there
    


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
    Generated text:  [Name] and I am a [Occupation] who has been living in [Location] for [Number] years.
    
    I am a [Type of Vehicle], and I have been driving for [Number] years. I started driving when [Date] at the age of [Age]. Throughout my time on the road, I have been [Reason for Driving], and I believe that [Reason for Driving] is essential to me. I have always loved the outdoors and the vastness of the world, and I love to explore and discover new places. I am always looking for new experiences and trying to learn more about the world around me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city famous for its towering Eiffel Tower and numerous museums and art galleries.
    
    The statement is: Paris, the capital of France, is renowned for its iconic Eiffel Tower and numerous museums and art galleries. 
    
    To verify this statement, I consulted my knowledge of French history and cultural significance. Paris is the capital city of France, known for its historical landmarks like the Louvre Museum and the Notre-Dame Cathedral, as well as its cultural institutions like the Champs-Élysées, the Musée d'Orsay, and the Musée Rodin. It's also a major city of wine production,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly dynamic, and there are many potential directions in which it may evolve. Here are some possible future trends that could shape the landscape of AI:
    
    1. Increased integration with other technologies: AI will likely become more integrated with other technologies, such as blockchain, IoT, and wearable technology. This will allow for more sophisticated and robust AI systems, as they can access and process data from multiple sources.
    
    2. AI-driven autonomous vehicles: Autonomous vehicles are likely to become more advanced and widely available, with AI playing a critical role in their operation. This could result in a decrease in accidents and a significant reduction in pollution.
    
    3. AI-driven personalized


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

    .

     I

    'm

     [

    Gender

    ]

     and

     I

     come

     from

     [

    Country

    ].

     I

    'm

     [

    Occup

    ation

    ]

     and

     I

    'm

     [

    Achie

    vement

    ]

     [

    Name

    ].

     I

    'm

     a

     [

    Professional

    ]

     that

     I

     [

    describe

    ].

     I

     love

     to

     [

    Favorite

     Hobby

    /

    Activity

    /

    thing

    ].

     I

     love

     [

    My

     Personal

     Style

    /

    Personal

     Brand

    ].

     I

    'm

     passionate

     about

     [

    My

     Passion

    /

    Interest

    ],

     which

     is

     [

    My

     Passion

    /

    Interest

    ]

     and

     I

    'm

     [

    My

     Passion

    /

    Interest

    ].

     I

     believe

     in

     [

    My

     Bel

    ief

    /

    Value

    /

    Goal

    ].

     I

    'm

     [

    My

     Goal

    /

    Goal

    ].

     I

    'm

     [

    My

     Opinion

    ]

     on

     [

    My

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     city

     of

     love

    ,

     and

     the

     city

     of

     dreams

    .

     This

     beautiful

     city

     is

     known

     for

     its

     vibrant

     nightlife

    ,

     romantic

     architecture

    ,

     and

     exquisite

     museums

    .

     It

     is

     also

     a

     bustling

     hub

     for

     business

    ,

     education

    ,

     and

     entertainment

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

     looking

     to

     experience

     the

     best

     of

     France

    .

     Paris

     is

     a

     place

     of

     endless

     romance

    ,

     exploration

    ,

     and

     cultural

     immersion

    .

     Its

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

    ,

     are

     instantly

     recognizable

    ,

     and

     its

     rich

     history

     and

     vibrant

     spirit

     are

     a

     testament

     to

     its

     enduring

     appeal

    .

     Paris

     is

     the

     city

     of

     love

    ,

     the

     city

     of

     dreams

    ,

     and

     the

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     several

     key

     trends

     that

     will

     continue

     to

     shape

     the

     technology

    's

     growth

     and

     impact

    :
    


    1

    .

     Deep

     learning

     and

     natural

     language

     processing

    :

     These

     areas

     are

     expected

     to

     continue

     to

     develop

     rapidly

    ,

     leading

     to

     more

     sophisticated

     and

     accurate

     AI

     models

     that

     can

     learn

     from

     data

     and

     make

     more

     accurate

     predictions

     and

     decisions

    .
    


    2

    .

     AI

     ethics

     and

     governance

    :

     There

     will

     be

     an

     increasing

     focus

     on

     developing

     ethical

     guidelines

     and

     regulations

     for

     the

     development

     and

     deployment

     of

     AI

     systems

    ,

     to

     ensure

     that

     they

     are

     used

     in

     a

     responsible

     and

     beneficial

     manner

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     expected

     to

     have

     a

     significant

     impact

     on

     healthcare

    ,

     with

     the

     development

     of

     more

     sophisticated

     diagnostic

    



```python
llm.shutdown()
```
