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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.28it/s]


    2026-04-06 02:22:33,388 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 02:22:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.48it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.62it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.62it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.57it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 20.58it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 26.06it/s]

    Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 31.44it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 35.36it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 43.92it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 43.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.27it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.27it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.27it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.27it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=704 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=640 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=576 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.18it/s]

    Capturing num tokens (num_tokens=576 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.46it/s]Capturing num tokens (num_tokens=512 avail_mem=137.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.46it/s]Capturing num tokens (num_tokens=480 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.46it/s]Capturing num tokens (num_tokens=448 avail_mem=137.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.46it/s]Capturing num tokens (num_tokens=416 avail_mem=136.80 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.46it/s]Capturing num tokens (num_tokens=416 avail_mem=136.80 GB):  55%|█████▌    | 32/58 [00:00<00:00, 34.78it/s]Capturing num tokens (num_tokens=384 avail_mem=136.80 GB):  55%|█████▌    | 32/58 [00:00<00:00, 34.78it/s]Capturing num tokens (num_tokens=352 avail_mem=136.64 GB):  55%|█████▌    | 32/58 [00:00<00:00, 34.78it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.78it/s]

    Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.17it/s]Capturing num tokens (num_tokens=256 avail_mem=136.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.17it/s]Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.17it/s]Capturing num tokens (num_tokens=224 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.17it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.17it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.17it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  71%|███████   | 41/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=176 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=144 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  71%|███████   | 41/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  71%|███████   | 41/58 [00:01<00:00, 39.02it/s]

    Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=96 avail_mem=136.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.79it/s] Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=64 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.87it/s]

    Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.20it/s]Capturing num tokens (num_tokens=8 avail_mem=136.56 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.20it/s] Capturing num tokens (num_tokens=4 avail_mem=136.56 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.20it/s]Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:01<00:00, 37.27it/s]


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
    Generated text:  Amin. I was born in 1983 and I want to write my own story. 
    
    I have always been interested in writing stories. I did a lot of writing, but I was never really satisfied with what I wrote. So, I decided to try writing my own story. 
    
    I decided to focus on a specific character, a specific time period, and a specific setting. I wanted to write a story that was not only a realistic portrayal of the specific time period, but also a story that explored the psychological and emotional aspects of the character.
    
    I wrote the story, and I was pleased with the result. But
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting three different cities, A, B, and C. He must choose one city to visit first, two cities to visit second and third, and on the last day, he will not visit any city. The president must choose a city that was not visited in the first two days and another city that was not visited in the second and third days. How many different ways can the president choose cities for his three days?
    To solve the problem, we need to carefully consider the constraints given:
    
    1. The president must choose a city that was not visited in the first two days.
    2. The president must choose a city that was not
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ).
    
    A: Paris  
    B: London  
    C: Rome  
    D: Berlin
    
    To determine the capital of France, we need to recall the correct capital cities of the countries that border France. The bordering countries are:
    
    - Germany: Berlin
    - Belgium: Brussels
    - Luxembourg: Luxembourg City
    - Switzerland: Geneva
    
    Since France borders Germany, Belgium, and Luxembourg, the capital of France is not one of these neighboring countries. The correct capital of France is not Berlin, because Berlin is in Germany.
    
    Therefore, the correct answer is:
    
    \boxed{C}
    ===============================
    Prompt: The future of AI is
    Generated text:  not about technology but about the skills that make it possible.
    
    ## Machine Learning and Deep Learning
    
    ### Introduction
    
    Machine learning and deep learning are the two dominant paradigms of AI. They were originally developed in the 1940s and 1950s, with the first major developments occurring in the 1990s and 2000s. Now, they are applied in many areas of science and engineering, and have become a major part of many areas of research.
    
    ### Overview
    
    Machine learning and deep learning are two broad approaches to artificial intelligence (AI). Machine learning is the process of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to many famous museums and art galleries, including the Louvre and the Musée d'Orsay. Paris is a vibrant and diverse city with a rich history and culture. It is a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust AI systems that are designed to be transparent, accountable, and responsible.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes
    


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
    Generated text:  [Name] and I am a [Professional Title] with [Number of years of experience] years of experience in [Current field of work or career]. I have a passion for [Your Passion], [Hobby], [Interest], and I am [Your Skill Level].
    
    I am organized, passionate about my work, and always strive to improve myself. I enjoy working on projects and collaborating with others to achieve success. What excites you most about your profession? I love the opportunity to make a difference, to help people, and to use my skills to solve complex problems. How do you stay motivated and focused on your work? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city with a rich cultural heritage. It is known for its many historic landmarks, including the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. Paris is also famous for its cuisine, including its famous Parisian croissants and its seafood. Additionally, Paris is known for its French language, as it is one of the most widely spoken languages in the world. Overall, Paris is a vibrant, dynamic city that is an important cultural and economic center of France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting, with potential applications across various industries and fields. Here are some possible trends that are likely to shape AI in the coming years:
    
    1. Increased Use of AI in Healthcare: AI will continue to be used in healthcare to improve patient outcomes and reduce costs. AI will be used to analyze medical images, detect diseases, and predict patient outcomes. It will also be used to develop personalized treatments for patients.
    
    2. Automation and Machine Learning: AI will continue to automate tasks in industries such as manufacturing, finance, and transportation. Machine learning algorithms will be used to improve efficiency and reduce costs.
    
    3. AI will become more integrated with the


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

     am

     a

     [

    Occup

    ation

    ]

     and

     [

    Position

    ]

     who

     has

     a

     passion

     for

     [

    Project

    /

    Goal

    ].

     I

     love

     to

     explore

     new

     ideas

     and

     techniques

    ,

     and

     I

    'm

     always

     looking

     to

     learn

     and

     grow

    .

     I

    'm

     excited

     to

     dive

     into

     any

     new

     challenge

     and

     see

     what

     surprises

     me

    .

     What

    's

     your

     current

     project

     or

     goal

    ?


    [

    Your

     Name

    ]

     -

     [

    Your

     Occupation

    /

    Position

    ]

     and

     [

    Your

     Position

    ]

     [

    Your

     Name

    ]

     is

     a

     tech

     entrepreneur

     with

     a

     passion

     for

     innovation

     and

     technology

    .

     [

    Your

     Name

    ]

     has

     a

     deep

     understanding

     of

     the

     latest

     trends

     and

     technologies

     in

     the

     field

    ,

     and

     is

     always

     seeking

     out

     new

     ideas

     and

     techniques

     to

    
    
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

     Europe

     and

     the

     largest

     city

     in

     the

     world

     by

     population

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     beautiful

     architecture

    ,

     and

     is

     the

     seat

     of

     the

     Government

     of

     France

    .

     The

     city

     is

     home

     to

     many

     iconic

     landmarks

    ,

     including

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

     The

     French

     Riv

    iera

     is

     also

     a

     popular

     tourist

     destination

     in

     Paris

    .

     Paris

     has

     a

     diverse

     range

     of

     cultures

     and

     food

     options

    ,

     and

     is

     considered

     the

     "

    city

     of

     love

    "

     by

     many

    .

     Despite

     its

     fame

    ,

     Paris

     remains

     a

     peaceful

     and

     welcoming

     city

     for

     visitors

    .

     Today

    ,

     it

     is

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

    ,

     with

     many

     possible

     developments

     shaping

     the

     technology

    's

     direction

     in

     the

     next

     decade

    .

     Here

     are

     some

     possible

     trends

     that

     we

     can

     expect

     in

     AI

    :
    


    1

    .

     Increased

     AI

     awareness

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     a

     greater

     understanding

     and

     control

     of

     AI

    's

     abilities

    .

     This

     could

     lead

     to

     more

     responsible

     AI

     development

    ,

     with

     developers

     and

     users

     working

     together

     to

     create

     AI

     that

     is

     ethical

     and

     responsible

    .
    


    2

    .

     AI

     ethics

     and

     morality

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     a

     greater

     emphasis

     on

     ethical

     and

     moral

     considerations

     when

     developing

     AI

    .

     This

     could

     lead

     to

     more

     transparent

     and

     ethical

     design

     practices

    ,

     and

     a

     greater

     focus

     on

     the

     potential

     impact

     of

     AI

     on

    



```python
llm.shutdown()
```
