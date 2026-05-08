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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.87it/s]


    2026-05-08 20:28:29,529 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 20:28:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.27it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.27it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.27it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.27it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.27it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.09it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.09it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.09it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.09it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.09it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 14.87it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 14.87it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 14.87it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.87it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.87it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 19.09it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 22.86it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 22.86it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 22.86it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 22.86it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 22.86it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 22.86it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 27.60it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 30.28it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 47.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.17 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=960 avail_mem=72.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s] Capturing num tokens (num_tokens=896 avail_mem=72.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.07it/s]Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=768 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=704 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=640 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=576 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=512 avail_mem=72.16 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=512 avail_mem=72.16 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=480 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=448 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=416 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=384 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.16 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=352 avail_mem=72.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.56it/s]Capturing num tokens (num_tokens=320 avail_mem=72.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.56it/s]Capturing num tokens (num_tokens=288 avail_mem=72.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.56it/s]Capturing num tokens (num_tokens=256 avail_mem=72.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.56it/s]Capturing num tokens (num_tokens=240 avail_mem=72.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.56it/s]Capturing num tokens (num_tokens=224 avail_mem=72.14 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.56it/s]Capturing num tokens (num_tokens=224 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.67it/s]Capturing num tokens (num_tokens=208 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.67it/s]Capturing num tokens (num_tokens=192 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=176 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=160 avail_mem=72.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.67it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=128 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=112 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=96 avail_mem=72.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.55it/s] Capturing num tokens (num_tokens=80 avail_mem=72.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=64 avail_mem=72.11 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=64 avail_mem=72.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=48 avail_mem=72.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=32 avail_mem=72.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=24 avail_mem=72.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.34it/s]Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=16 avail_mem=72.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=12 avail_mem=72.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=8 avail_mem=72.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.72it/s] Capturing num tokens (num_tokens=4 avail_mem=72.08 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB): 100%|██████████| 58/58 [00:01<00:00, 42.17it/s]


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
    Generated text:  NIMSK, I was born in the year 1995, the year I was born was 04/09/1995.
    Is there anything that is the same for me and for you? 
    
    As an AI, I don't have personal experiences or memories, but I can share some general information about the year 1995.
    
    In 1995, the year I was born, the United States was experiencing a significant political and social upheaval. The country was divided into two main parties, the Democratic Party and the Republican Party, each with their own candidates for the presidency
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small island nation. The island nation has a population of 4 million. The president wants to ensure that the gift they are presenting is as specific as possible. They decide to give the gift to 30% of the population. How many people will receive the gift?
    To determine how many people will receive the gift, we need to calculate 30% of the island nation's population. The population of the island nation is 4 million. We can find 30% of 4 million by following these steps:
    
    1. Convert the percentage to a decimal. Since 30% is the same as 
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) London
    B) Paris
    C) Rome
    D) New York
    B) Paris
    
    The capital of France is Paris. It is the largest city in France and the seat of the French government and major political, commercial, and cultural centers. The other options are not capital cities of France: London is the capital of the United Kingdom, and Rome is the capital of Italy. New York is the capital of the United States.
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright, and there are many different types of AI applications. The most common ones are artificial intelligence (AI) which is used to detect spam, translate text, interpret visual content, and more.
    On the other hand, there are also many other types of AI applications such as healthcare, finance, education, and more. Some of these AI applications are more focused on specific fields such as finance, where a person uses an AI chatbot to help them with their financial problems, or health care, where a person uses an AI chatbot to help them with their medical problems.
    AI applications can also be used for personalization. For example


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich cultural heritage and a vibrant nightlife. It is located in the south of France and is the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its diverse food scene and fashion scene. It is also a major center for business, politics, and culture, and is a popular tourist destination. Paris is a city of contrasts, with its historical architecture and modern fashion, and is a UNESCO World Heritage site. It is a city of art,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement for some workers.
    
    2. AI-powered healthcare: AI is already being used to improve the accuracy and efficiency of medical diagnoses and treatments. As AI technology continues to advance, we may see even more
    


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
    Generated text:  [Your Name] and I am a [Your Profession]. I have been working in marketing for [Your Company] for [X] years. I am dedicated to [Your Primary Goal], and have always been passionate about [Your Hobby/Interest]. I have always been a strong team player, and I am always looking for ways to improve my skills and knowledge. I am always looking for new opportunities to learn, grow, and contribute to the success of my team. In short, I am a [Your Profession], dedicated to [Your Goal], and always looking for ways to improve. Thank you for having me! What is your professional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. It is also a cultural and economic center, known for its rich history and diverse cuisine. Its population is over 6 million, and it is one of the largest cities in the world by population. Paris is a popular tourist destination, with over 50 million visitors annually. Despite being one of the oldest cities in the world, Paris has a vibrant and dynamic culture, with a rich history and a mix of historical and contemporary elements. The city is home to many renowned artists, writers, and musicians, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and it's likely to bring about a range of exciting and transformative changes. Here are some possible trends in AI over the next few decades:
    
    1. Improved Human-AI collaboration: As AI gets more advanced, it will become even more capable of understanding and responding to human emotions and motivations. This will lead to more effective and intelligent human-AI collaboration, where machines can better communicate and work together in complex environments.
    
    2. Increased AI ethics and transparency: As AI becomes more sophisticated, it will become increasingly important to address ethical concerns such as bias, privacy, and accountability. AI experts will need to develop new ethical frameworks and


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

    ],

     and

     I

     am

     a

     [

    insert

     age

    ]

     year

     old

     [

    insert

     occupation

    ].

     I

     am

     currently

     living

     in

     [

    insert

     city

    ,

     state

    ,

     country

    ]

     and

     I

     am

     a

     [

    insert

     hobby

     or

     interest

    ]

     at

     heart

    .

     If

     you

     have

     a

     question

    ,

     I

     will

     do

     my

     best

     to

     answer

     it

    .

     I

     am

     a

     [

    insert

     skill

     or

     talent

    ],

     which

     makes

     me

     uniquely

     [

    insert

     why

    ].

     I

     am

     [

    insert

     nationality

    ],

     and

     I

     am

     [

    insert

     ethnicity

    ].

     I

     enjoy

     [

    insert

     hobbies

    ,

     interests

    ,

     or

     passions

    ]

     and

     I

     believe

     that

     everyone

     deserves

     [

    insert

     why

    ].

     I

     am

     [

    insert

     personality

     trait

     or

     background

    ].

     I

     am

     [

    insert

     appearance

    ]

     and

     I

     love

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     island

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     most

     populous

     city

     in

     the

     European

     Union

    ,

     known

     for

     its

     medieval

     architecture

    ,

     rich

     history

    ,

     and

     numerous

     museums

     and

     festivals

    .

     Paris

     is

     a

     hub

     for

     culture

    ,

     fashion

    ,

     and

     cuisine

     and

     is

     a

     major

     financial

     hub

     as

     well

    .

     It

     is

     famous

     for

     its

     E

    iff

    el

     Tower

    ,

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

     its

     annual

     Spring

     Festival

    .

     
    


    Paris

    ,

     like

     most

     cities

    ,

     is

     a

     complex

     city

     with

     a

     mix

     of

     modern

     and

     traditional

     elements

    .

     It

     has

     a

     high

     density

     of

     people

    ,

     leading

     to

     a

     high

     pressure

     and

     low

     pressure

     urban

     environment

    ,

     as

     well

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     marked

     by

     further

     advances

     in

     machine

     learning

    ,

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     deep

     learning

    .

     As

     these

     fields

     continue

     to

     evolve

    ,

     we

     may

     see

     more

     complex

    ,

     intelligent

     systems

     that

     can

     learn

     and

     adapt

     on

     their

     own

    ,

     with

     the

     ability

     to

     solve

     increasingly

     complex

     problems

     and

     make

     decisions

     that

     affect

     our

     daily

     lives

    .
    


    One

     potential

     trend

     is

     the

     increased

     use

     of

     AI

     in

     healthcare

    ,

     with

     the

     development

     of

     more

     advanced

     diagnostic

     tools

    ,

     treatment

     algorithms

    ,

     and

     predictive

     analytics

    .

     AI

     may

     also

     play

     an

     increasing

     role

     in

     manufacturing

     and

     supply

     chain

     management

    ,

     with

     the

     ability

     to

     optimize

     production

     processes

     and

     reduce

     waste

    .
    


    In

     addition

     to

     these

     applications

    ,

     AI

     is

     likely

     to

     have

     a

    



```python
llm.shutdown()
```
