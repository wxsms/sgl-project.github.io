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


    2026-04-10 16:42:14,170 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 16:42:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.38it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.93it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.88it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.43 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.43 GB):   9%|▊         | 5/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.28 GB):   9%|▊         | 5/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.93 GB):   9%|▊         | 5/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.93 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.72 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.71 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.71 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.38 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.49it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.77it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.77it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.77it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.77it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.71it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.71it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.71it/s]

    Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.71it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.71it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.71it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.78it/s]

    Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.19it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.11it/s]

    Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.11it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.51it/s]Capturing num tokens (num_tokens=48 avail_mem=120.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.51it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.51it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.51it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.51it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.51it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.54it/s]

    Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.54it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 30.71it/s]


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
    Generated text:  Lucy. I am a junior high school student. I have a good friend. His name is Mr. Chen. He is a nice man. He speaks Chinese and also he speaks English. He is very smart and he can solve many math problems. He has two big eyes. He likes to wear a new t-shirt and shorts. He is always in his study. He is a teacher. I have a new friend today. His name is Tony. He is a middle school student. He studies very hard. He speaks English very well. He likes to play basketball. He likes to eat a lot. He has a big mouth.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very busy person who likes to work on important policies. One day, he decided to organize a meeting with his team to discuss a proposal. They have two options: Option A and Option B. Option A has a probability of 0.6 of being approved, and Option B has a probability of 0.7 of being approved. The president wants to know the probability that the proposal will be approved by at least one of the teams.
    
    To calculate this, you need to first find the probability that at least one of the teams will approve the proposal. This can be found by subtracting the probability that neither team will approve the
    ===============================
    Prompt: The capital of France is
    Generated text:  ___________.
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    To determine the capital of France, let's analyze the options provided:
    
    A. Paris: This is the capital of France, located on the Île de la Cité in the French Riviera. Paris is known for its historical sites, museums, and art, making it a popular tourist destination.
    
    B. London: This is the capital of England, located in the eastern part of the United Kingdom. London is famous for its architecture, history, and cultural heritage.
    
    C. Rome: This is the capital of Italy, located in the
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain, as it has the potential to change the world in unpredictable ways. The rapid pace of technological advancement and the increasing availability of data mean that AI is likely to become an integral part of daily life. While it has the potential to improve efficiency and productivity, there are also concerns about job displacement, privacy and ethical implications.
    AI has the potential to revolutionize the way we work, communicate and interact with each other. With the help of AI, we can automate tasks and make decisions faster, more accurately and with less error. This can lead to increased productivity and efficiency, but it can also create job displacement as machines take over some


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I Love About My Profession]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality Trait] person and I'm always ready to learn from others. I'm [What I Do Best]. I'm [What I Hope to Achieve]. I'm excited to meet you and learn more about you. [Name] [Age] [Occupation] [Skill] [Number] [Field] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and diverse culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 8 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to many world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major economic center in France. It is also home to many cultural institutions, including the Louvre Museum
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to improve, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve risk management, fraud detection, and investment decision
    


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
    Generated text:  [insert name here] and I am [insert age here]. I am a [insert profession here] who enjoys [insert hobby or activity here]. I have [insert number of years of experience here] years of experience in [insert relevant field or industry here]. I am [insert personality traits here, such as a friendly, curious, or adventurous one]. I am always looking for new challenges and opportunities to grow and learn. What's your name and how do you feel about yourself? [insert your name here]. I am excited to meet you! Let's get to know each other! What's your name? [insert your name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, also known as the City of Light, is the largest city in France and its capital. It is known for its iconic architecture, museums, and cultural attractions such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also a major commercial and financial center, hosting major events such as the Eiffel Tower Fashion Week and the World Cup. Paris has a rich cultural heritage and is home to many famous artists, writers, and architects. It is a vibrant and cosmopolitan city with a population of over 2.5 million people. The city is also home to several world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly complex and evolving, with many new trends and developments emerging. Here are some of the possible future trends in AI:
    
    1. Enhanced machine learning algorithms: Advances in machine learning algorithms, such as deep learning and neural networks, are expected to lead to even more accurate and sophisticated AI systems. This will enable AI to understand and analyze complex data more effectively, leading to improved decision-making and prediction capabilities.
    
    2. Integration with other technologies: AI systems are becoming increasingly integrated with other technologies, such as sensors, blockchain, and IoT, which will enable them to gather and process more data and information. This will allow AI systems to


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

    ].

     I

     am

     a

     [

    Your

     occupation

     or

     profession

    ]

     who

     has

     always

     been

     passionate

     about

     [

    What

     they

     enjoy

     doing

    ].

     I

     love

     spending

     time

     with

     my

     family

     and

     friends

    ,

     and

     I

     enjoy

     traveling

     to

     new

     and

     exciting

     places

    .

     I

     am

     always

     looking

     for

     new

     experiences

     and

     new

     adventures

    ,

     and

     I

     hope

     to

     continue

     doing

     that

     for

     as

     long

     as

     I

     can

    .

     What

     is

     your

     profession

    ,

     hobbies

    ,

     and

     travel

     plans

    ?

     You

    ,

     my

     friend

    ,

     are

     [

    Your

     name

    ].

     You

    're

     a

     [

    Your

     occupation

     or

     profession

    ].

     I

     love

     [

    What

     they

     enjoy

     doing

    ].

     I

     spend

     time

     with

     my

     family

     and

     friends

    ,

     and

     I

     enjoy

     [

    What

     they

     enjoy

     doing

    ].

     I

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    Paris

     is

     known

     for

     its

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


    Paris

     is

     also

     famous

     for

     its

     cuisine

    ,

     including

     seafood

     dishes

     such

     as

     cro

    iss

    ants

     and

     esc

    arg

    ot

    .

     


    The

     city

    's

     architecture

     is

     also

     noteworthy

    ,

     featuring

     examples

     of

     Gothic

    ,

     Renaissance

    ,

     and

     Modern

    ist

     styles

    .


    Overall

    ,

     Paris

     is

     a

     culturally

     rich

     and

     diverse

     city

     with

     a

     long

     history

     dating

     back

     to

     the

     

    1

    2

    th

     century

    .

     


    The

     city

     has

     a

     population

     of

     over

     

    7

     million

     and

     is

     known

     for

     its

     nightlife

    ,

     museums

    ,

     and

     other

     cultural

     attractions

    .

     


    It

     is

     also

     home

     to

     the

     French

     Parliament

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     bright

     and

     diverse

    ,

     and

     there

     are

     many

     potential

     trends

     that

     will

     shape

     the

     technology

     and

     applications

     in

     the

     years

     to

     come

    .

     Some

     of

     the

     most

     notable

     trends

     include

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     neural

     networks

    :

     As

     the

     complexity

     of

     AI

     systems

     continues

     to

     increase

    ,

     there

     will

     be

     an

     increasing

     need

     for

     more

     powerful

     machine

     learning

     and

     neural

     network

     models

    .

     This

     means

     that

     we

     will

     see

     continued

     improvements

     in

     algorithms

    ,

     training

     techniques

    ,

     and

     data

     processing

     capabilities

    .
    


    2

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     into

     many

     different

     areas

    ,

     including

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     entertainment

    .

     As

     more

     technology

     and

     data

     is

     collected

     and

     shared

    ,

    



```python
llm.shutdown()
```
