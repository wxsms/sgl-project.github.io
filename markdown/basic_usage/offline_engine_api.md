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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]


    2026-05-03 17:22:25,816 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 17:22:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:07,  5.89it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:07,  5.89it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:07,  5.89it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:07,  5.89it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:07,  5.89it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:07,  5.89it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.07it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  9.07it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  9.07it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  9.07it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]

    Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 20.14it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 28.45it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 37.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 46.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:15,  3.69it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   2%|▏         | 1/58 [00:00<00:15,  3.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   2%|▏         | 1/58 [00:00<00:15,  3.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   5%|▌         | 3/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.37 GB):   5%|▌         | 3/58 [00:00<00:07,  7.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.37 GB):   5%|▌         | 3/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.37 GB):   9%|▊         | 5/58 [00:00<00:05,  9.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.36 GB):   9%|▊         | 5/58 [00:00<00:05,  9.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.34 GB):   9%|▊         | 5/58 [00:00<00:05,  9.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.34 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.33 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.84it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.33 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.33 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.32 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.24it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.30 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  22%|██▏       | 13/58 [00:01<00:02, 18.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.29 GB):  22%|██▏       | 13/58 [00:01<00:02, 18.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.29 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.28 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.78it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.53it/s]Capturing num tokens (num_tokens=960 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.53it/s] Capturing num tokens (num_tokens=896 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.53it/s]Capturing num tokens (num_tokens=896 avail_mem=74.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=576 avail_mem=74.21 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.23it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.39it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.39it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.39it/s]Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.39it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.39it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.98it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=192 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=176 avail_mem=74.13 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.11it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.02it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.02it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.02it/s] Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.02it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.02it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.70it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.70it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.70it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.70it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.91it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.91it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.91it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.91it/s] Capturing num tokens (num_tokens=4 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.91it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:02<00:00, 38.43it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:02<00:00, 26.19it/s]


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
    Generated text:  Minka and I am a 14 year old female. I was in a car accident 5 days ago that resulted in a severe injury to my head. There was a large area of concussion and swelling to my skull. The doctors who treated me said that I need to get to a neurologist and have a brain scan.
    
    My question is this: should I call my parents or my doctor first? I feel a little panicked that I am not telling them yet and I really need to know what's going on. If I tell them about the brain scan first, should I be worried about it? I am just scared and
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He works very hard. He works hard to _______________ the people. A. make the people happy B. make people to be happy C. make people happy D. make people to work well C. make people happy
    
    Question: The president of the United States is very busy. He works very hard. He works hard to make people happy. What is the most logical completion for the blank? A. make the people happy B. make people to be happy C. make people happy D. make people to work well C. make people happy
    
    In this sentence, the president of the United States works hard to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was founded by the French Bourgeois Revolution in 1667 and remains the capital of the European Union. The city of Paris is a popular tourist destination, with a rich and diverse culture, fine art galleries, historic landmarks, and numerous museums. The city is known for its food culture, which is highly regarded around the world. The city is also popular for its fashion, music, and film industry. Paris is a city that attracts visitors from all over the world with its beauty, history, and culture. What makes Paris so special? The answer is the people of the city. Paris is the capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  artificial intelligence. It will be the main driver of future technology and the method of control and operation of today's technology. It will be the main basis for the development of today's technology. AI will be the key to future technology. Its core is artificial intelligence, the core is human consciousness, the core is human intelligence, and the core is the development of humans.
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    In the 2019 personal income tax law, the preferential tax rate for individual income tax is ____.
    A. 10%
    B. 15%
    C. 20%
    


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [Describe your personality traits here]. I'm [Describe your hobbies or interests here]. I'm [Describe your strengths and weaknesses here]. I'm [Describe your values and beliefs here]. I'm [Describe your goals and aspirations here]. I'm [Describe your future plans here]. I'm [Describe your future career goals here]. I'm [Describe your future education here]. I'm [Describe your future location here]. I'm [Describe your future family here]. I'm [Describe your future pets here].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is also known for its cuisine, including French cuisine, and its annual Eiffel Tower Festival. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is the largest city in France by population
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, we can expect it to be used in even more areas, such as personalized medicine, drug discovery, and image analysis.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve efficiency and reduce costs. As AI becomes more advanced, we can expect it to be used in even
    


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
    Generated text:  [Name], and I'm a [type of fictional character] in the [genre of literature, film, game, etc.]. I'm passionate about [interest or hobby of the fictional character] and I've always been motivated to [motivational statement about the fictional character's passion]. I've been working on my character for [number of years] years now, and I've achieved [positive accomplishment]. I'm always ready to [statement about the fictional character's personality or nature], and I enjoy [positive statement about the fictional character's skills or abilities]. I'm a [charisma, kind, smart, etc.],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light." The city is known for its iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination, renowned for its fashion, cuisine, and artistic scene. Paris has a rich cultural heritage and is home to many important museums and historical sites. It is considered a major global city and a symbol of France's rich history and culture. It is home to many important institutions, including the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. The city is also known for its rich culinary traditions,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in both hardware and software. Here are some potential trends that could shape the future of AI:
    
    1. Increased accuracy and precision: As AI continues to get better at recognizing patterns and making predictions, we can expect to see more accurate and precise decisions being made in various domains, including healthcare, finance, and logistics.
    
    2. Faster and more efficient data analysis: With the increased availability of big data, we can expect AI to be able to process and analyze data at an unprecedented rate, which could lead to new opportunities for innovation and business.
    
    3. Improved security and privacy: AI will continue to play an increasingly important


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

    name

    ]

     and

     I

    'm

     a

     [

    occupation

    ].

     I

    'm

     [

    age

    ],

     and

     I

     have

     [

    work

     experience

    ].

     I

    'm

     [

    degree

    ],

     and

     I

     hold

     a

     [

    professional

     designation

    ].

     I

    'm

     [

    favorite

     hobby

     or

     activity

    ],

     and

     I

     enjoy

     [

    reason

     for

     liking

     this

     hobby

     or

     activity

    ].

     I

    'm

     [

    rel

    igion

     or

     political

     affiliation

    ],

     and

     I

     believe

     in

     [

    value

     or

     belief

    ].

     I

    'm

     [

    interest

    s

     or

     hobbies

    ],

     and

     I

     love

     [

    reason

     for

     liking

     these

     activities

    ].

     I

    'm

     [

    person

    ality

     trait

    ],

     and

     I

    'm

     [

    character

     traits

    ].

     Thank

     you

     for

     considering

     me

     as

     your

     fictional

     character

    .

     Please

     feel

     free

     to

     write

     about

     your

     character

    's

     life

    ,

     interests

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     on

     the

     Se

    ine

     River

     in

     the

     center

     of

     the

     country

     and

     is

     one

     of

     the

     oldest

     cities

     in

     the

     world

    .

     It

     is

     home

     to

     numerous

     famous

     landmarks

    ,

     including

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

     Paris

     is

     also

     known

     as

     "

    the

     City

     of

     Light

    "

     for

     its

     vibrant

     and

     diverse

     culture

     and

     has

     been

     a

     hub

     of

     art

     and

     culture

     for

     centuries

    .

     The

     city

     is

     also

     home

     to

     many

     important

     institutions

     and

     organizations

    ,

     including

     the

     French

     Academy

     of

     Sciences

     and

     the

     French

     Parliament

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     has

     an

     estimated

     population

     of

     over

     

    2

    .

    1

     million

     people

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     unpredictable

    ,

     but

     there

     are

     several

     potential

     trends

     that

     could

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

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

     already

     being

     used

     to

     improve

     medical

     outcomes

    ,

     but

     it

     has

     the

     potential

     to

     transform

     the

     field

     in

     ways

     that

     are

     only

     beginning

     to

     be

     explored

    .

     AI

    -powered

     diagnostic

     tools

    ,

     chat

    bots

    ,

     and

     personalized

     medicine

     are

     all

     promising

     areas

     of

     development

    .
    


    2

    .

     Enhanced

     AI

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

     more

     room

     for

     debate

     about

     the

     ethical

     implications

     of

     their

     use

    .

     There

     are

     already

     some

     concerns

     about

     the

     potential

     for

     AI

     to

     be

     used

     to

     perpet

    uate

     bias

    ,

     discrimination

    ,

     and

     even

     harm

    .

    



```python
llm.shutdown()
```
