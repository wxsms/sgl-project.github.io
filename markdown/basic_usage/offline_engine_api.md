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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.65it/s]


    2026-04-06 05:23:26,983 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 05:23:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.29it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.23it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.33it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:06,  8.40it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   5%|▌         | 3/58 [00:00<00:12,  4.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   5%|▌         | 3/58 [00:00<00:12,  4.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   5%|▌         | 3/58 [00:00<00:12,  4.41it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:08,  6.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:08,  6.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:08,  6.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.55it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.14it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.14it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.91it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.91it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.12it/s]

    Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=144 avail_mem=120.23 GB):  71%|███████   | 41/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=128 avail_mem=120.23 GB):  71%|███████   | 41/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=112 avail_mem=119.39 GB):  71%|███████   | 41/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=112 avail_mem=119.39 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.25it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.25it/s]

    Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.98it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.98it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.98it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.98it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:02<00:00, 41.98it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:02<00:00, 41.98it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:02<00:00, 43.36it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:02<00:00, 43.36it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:02<00:00, 43.36it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:02<00:00, 27.58it/s]


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
    Generated text:  Mary and I'm from Denmark. I have a dream that I want to be a doctor. In my dreams, when I wake up, I'm in a nice hospital. I'm a doctor helping sick people. It's a very funny dream because doctors are very important and very nice. I would like to help sick people because I think sick people can make people happy. In my dream, I have two brothers. My brother has a sick brother. I always try to help them. I think sick people are happy because of what they have. 
    
    Do you like the dream? What kind of advice would you give to someone who has
    ===============================
    Prompt: The president of the United States is
    Generated text:  32 years older than the president of Central America. The president of Central America is 2 times older than the president of South America. The president of South America is half as old as the president of America. How old would the president of America be if the president of Central America were 10 years younger? Let's start by defining the variables for the ages of the presidents in each region. Let \( A \) be the age of the president of America, \( C \) be the age of the president of Central America, and \( S \) be the age of the president of South America.
    
    From the problem
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lille
    C. Strasbourg
    D. Marseille
    Answer:
    A
    
    For the following scenarios, which should be recognized as employee benefits under accounting? ____.
    A. Corporate officers receiving a monthly bonus
    B. Bank employees receiving a salary from the bank
    C. Maintenance workers at a kindergarten receiving a monthly salary
    D. Tourist service personnel receiving a monthly salary
    Answer:
    C
    
    The characteristic of the defect is that it is ___.
    A. In the form of a bubble
    B. In the form of a crater
    C. Formed by the film plate grinding process
    
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it is already happening. From the powerful models of Amazon, Google, and Apple to the smaller, more affordable models that are available to the masses, the technology is advancing at an incredibly rapid pace. We are now seeing more and more AI-powered products and services as the technology has advanced. However, just like any other technology, there are also some challenges that come with using AI. One of these challenges is the privacy and security concerns associated with using AI.
    
    AI has the potential to transform the way we live and work, but it also has the potential to raise ethical concerns. It is important to address these concerns and ensure that


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? I'm a [insert a short description of your personality or skills that you're passionate about]. I enjoy [insert a short description of your hobbies or interests that you're passionate about]. I'm always looking for new experiences and learning new things, so I'm always eager to learn more about the world around me. What's your favorite hobby or activity? I'm always looking for new experiences and learning new things, so I'm always eager to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and culture. It is located in the south of France and is the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also famous for its cuisine, fashion, and art. Paris is a major tourist destination and is home to many museums, theaters, and other cultural institutions. It is a popular destination for business and leisure, and is considered one of the most beautiful cities in the world. Paris is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and standards for AI development and deployment, as well as
    


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
    Generated text:  [Name] and I'm a [occupation] by day, and a [career] by night. I love exploring new places and trying new things, and have a great sense of humor that I use to make people laugh. I also love reading and writing, and have a passion for using my creativity to create unique and meaningful stories. I hope to continue my journey of learning and growth, and I am always looking for new adventures to explore. 
    
    What is your job or occupation? How do you earn a living? What are some of your hobbies and interests? What are you most proud of in your career so far? 
    
    [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the Loire Valley region. It was founded as the city of Arras in the 5th century and became the capital in 1806, when Napoleon Bonaparte made it the new capital of the French Empire. Paris is a UNESCO World Heritage Site and is known for its many famous landmarks, including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. It is also home to the iconic Eiffel Tower, which stands as the city's symbol and is considered one of the most recognizable buildings in the world. In addition to its rich history and culture, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and is likely to continue to evolve in various directions. Some of the possible trends in AI include:
    
    1. Machine learning: This is the foundation of modern AI and is a core technology that allows machines to learn and improve on their own. With the development of machine learning, AI systems can learn to recognize patterns, identify trends, and make predictions on their own, without human intervention.
    
    2. Deep learning: This involves the use of neural networks to recognize complex patterns in data. Deep learning has the potential to significantly improve the accuracy of machine learning models and make them more efficient.
    
    3. Neural networks: These are the core components


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

    type

     of

     character

    ]

     [

    person

    ality

    ].

     I

     am

     currently

     living

     in

     [

    your

     hometown

    ]

     and

     I

     am

     passionate

     about

     [

    why

     you

     like

     living

     in

     this

     location

    ].

     I

     am

     a

     [

    what

     is

     your

     profession

    ]

     [

    field

    ]

     [

    or

    ,

     if

     you

     are

     in

     a

     position

     of

     authority

    ,

     "

    leader

    "]

     [

    title

    ].

     I

     have

     always

     been

     drawn

     to

     [

    why

     you

     are

     drawn

     to

     this

     field

     or

     position

    ],

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    what

     you

     plan

     to

     do

     next

    ]

     [

    or

    ,

     if

     you

     are

     a

     leader

    ,

     "

    implement

     a

     vision

    "]

     [

    what

     you

     hope

     to

     achieve

     in

     the

     future

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     North

     Bank

     of

     the

     Se

    ine

    ,

     the

     river

     that

     flows

     through

     the

     city

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

     and

     the

     center

     of

     France

    's

     political

    ,

     cultural

    ,

     and

     economic

     life

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

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     a

     significant

     center

     for

     art

    ,

     literature

    ,

     and

     music

    .

     Paris

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     nightlife

    ,

     and

     cultural

     attractions

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     people

     around

     the

     world

    .

     It

     is

     the

     seat

     of

     the

     French

     government

     and

     the

     largest

     city

     in

     the

     European

     Union

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     with

     new

     technologies

     and

     applications

     emerging

     at

     an

     unprecedented

     rate

    .

     Some

     of

     the

     key

     trends

     in

     AI

     are

    :
    


    1

    .

     Enhanced

     Machine

     Learning

    :

     Machine

     learning

     is

     one

     of

     the

     most

     widely

     used

     AI

     technologies

    ,

     and

     it

     is

     becoming

     more

     powerful

     and

     adaptable

    .

     As

     data

     sets

     continue

     to

     grow

     and

     become

     more

     diverse

    ,

     machine

     learning

     models

     will

     become

     even

     more

     accurate

     and

     capable

    .
    


    2

    .

     Autonomous

     Vehicles

    :

     Autonomous

     vehicles

     are

     already

     being

     tested

     and

     developed

    ,

     with

     companies

     like

     Tesla

     and

     Uber

     already

     offering

     self

    -driving

     cars

    .

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     autonomous

     vehicles

     to

     become

     more

     widespread

     and

     affordable

    ,

     allowing

     for

     safer

     and

     more

     efficient

     transportation

    .
    


    3

    .

     Cyber

    security

    :

    



```python
llm.shutdown()
```
