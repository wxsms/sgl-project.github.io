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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 10:23:31] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]


    2026-04-16 10:23:36,716 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 10:23:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.31it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.31it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.31it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:02<00:03, 12.31it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.31it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.31it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.31it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.31it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 26.60it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 33.08it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 40.11it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 40.11it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 40.11it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 40.11it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 40.11it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 40.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.21 GB):   9%|▊         | 5/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.20 GB):   9%|▊         | 5/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.20 GB):   9%|▊         | 5/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.20 GB):   9%|▊         | 5/58 [00:00<00:02, 20.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.20 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.20 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.19 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.19 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.19 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.19 GB):  21%|██        | 12/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.18 GB):  21%|██        | 12/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.18 GB):  21%|██        | 12/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.18 GB):  21%|██        | 12/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.17 GB):  21%|██        | 12/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.17 GB):  21%|██        | 12/58 [00:00<00:02, 22.44it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=55.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=960 avail_mem=55.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.66it/s] Capturing num tokens (num_tokens=960 avail_mem=55.15 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=896 avail_mem=55.15 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=832 avail_mem=55.15 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=768 avail_mem=55.14 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=704 avail_mem=55.14 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=640 avail_mem=55.14 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.97it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=576 avail_mem=55.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=512 avail_mem=55.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=480 avail_mem=55.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=448 avail_mem=55.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=416 avail_mem=55.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=384 avail_mem=55.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=384 avail_mem=55.14 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=352 avail_mem=55.13 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=320 avail_mem=55.13 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=288 avail_mem=55.12 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.02it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.12 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=240 avail_mem=55.12 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=240 avail_mem=55.12 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=224 avail_mem=55.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=208 avail_mem=55.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=192 avail_mem=55.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=176 avail_mem=55.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=160 avail_mem=55.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=144 avail_mem=55.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.97it/s]Capturing num tokens (num_tokens=144 avail_mem=55.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=128 avail_mem=55.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=96 avail_mem=55.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.70it/s] Capturing num tokens (num_tokens=80 avail_mem=55.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=64 avail_mem=55.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=64 avail_mem=55.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=48 avail_mem=55.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=32 avail_mem=55.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=28 avail_mem=55.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=24 avail_mem=55.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=20 avail_mem=55.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.72it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.07 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=16 avail_mem=55.07 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=12 avail_mem=55.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=8 avail_mem=55.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.92it/s] Capturing num tokens (num_tokens=4 avail_mem=55.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=4 avail_mem=55.05 GB): 100%|██████████| 58/58 [00:01<00:00, 34.65it/s]


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
    Generated text:  Tracy. I have always been passionate about helping others. After graduating from college, I decided to pursue a career in counseling. My first job was in my hometown, and I have continued to work with those who are struggling with personal, emotional, and relationship issues.
    My clients are often looking for help for a wide range of issues, including anxiety, depression, relationship problems, trauma, grief, and coping with trauma. My main approach is a client-centered approach. I work with clients to help them become better able to understand and address their own mental health, to seek information, and to develop healthy coping strategies.
    I am licensed in Michigan
    ===============================
    Prompt: The president of the United States is
    Generated text:  a post where the president holds a ceremonial position and represents the country's government and its policies. Elections are held to elect the president, who then holds office until his successor is elected and sworn in.
    Does this next sentence follow, given the above sentence. "The president is the highest-ranking officer in the United States government."?
    Pick your answer from:
     (A). yes
     (B). it is not possible to tell
     (C). no
    (A). Yes
    
    The sentence "The president is the highest-ranking officer in the United States government" does follow from the given sentence. 
    
    In the United States government, the president holds
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Tokyo
    D. Sydney
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Tokyo
    D. Sydney
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Tokyo
    D. Sydney
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Tokyo
    D. Sydney
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Tokyo
    D. Sydney
    Answer:
    ===============================
    Prompt: The future of AI is
    Generated text:  digital, and AI is the leading driver of growth in the digital space. The digital economy will drive the adoption of AI, and while the market will be fragmented, an ecosystem of AI is already forming. The digital economy will also drive AI adoption in the design of software. In the same vein, we see that AI will be the key driver of growth in the cloud computing market. The market will be fragmented, but an ecosystem of cloud computing will be forming, as cloud providers increasingly see AI as a key competitive advantage and service.
    Please paraphrase the text to make it easier to understand for a non-technical audience.
    
    The future of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower and vibrant cultural scene. It is also the seat of the French government and home to many of France's most famous landmarks and museums. Paris is a bustling metropolis with a rich history and a diverse population of over 2 million people. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the French National Library. Overall, Paris is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable of performing tasks that were previously impossible for humans. This could lead to more efficient and effective use of resources, as well as more accurate and reliable predictions.
    
    3. Increased use of AI in healthcare: AI is likely to play a more significant role in healthcare, with
    


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
    Generated text:  [Your Name], and I am a [职业/身份] who was born on [Date of Birth] in [Location]. I am here today to [Your Profession/Role] and I am excited to share my experiences and knowledge. What can you tell me about yourself? Let me know if you would like me to share any personal information or experiences. I am always happy to answer any questions or provide any information that you might need. Good day, and thank you for the opportunity to speak with you today. [Your Name] 🌍🌍✨
    
    That's a great self-introduction! Can you tell me more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the capital of the country. It is renowned for its rich history, art, and vibrant culture. The city is home to iconic landmarks such as Notre-Dame Cathedral and the Eiffel Tower, and is a popular tourist destination. The capital city plays a significant role in French politics and culture, and it is known for its annual Les Misérables Festival. Paris is also a major economic and financial center, with many multinational companies headquartered in the city. Overall, Paris is an important cultural and political center of France. 
    
    However, it is important to note that there are many other
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to expand and develop, with a number of exciting new trends and possibilities on the horizon. Here are some potential areas of focus:
    
    1. Improved Emotional Intelligence: AI is already being used to analyze speech and natural language, but there's much more to learn about how to make AI systems more empathetic and understanding of emotional states. This could lead to more personalized and intelligent interactions, as well as more accurate diagnoses of mental health conditions.
    
    2. Augmented Reality: With the rise of VR and AR, we could see a greater integration of AI in augmented reality applications. AI algorithms could be used to create more immersive and realistic


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

    ],

     and

     I

    'm

     a

    /an

     [

    character

     type

    ]

     who

     has

     always

     loved

     [

    reason

     for

     love

    ]

     since

     childhood

    .

     I

    'm

     [

    age

    ],

     and

     I

     have

     a

     diverse

     background

    ,

     from

     [

    country

    /

    region

    ]

     to

     [

    city

    /

    area

    ].

     I

    'm

     an

     [

    occupation

    ]

     and

     I

     believe

     in [

    phil

    osoph

    y

    /

    cur

    iosity

    ].

     I

    'm

     passionate

     about

     [

    career

     goal

    ]

     and

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     as

     a

     person

    .

     I

    'm

     [

    character

     trait

    ],

     and

     I

     believe in

     [

    bel

    iefs

    /

    phil

    osoph

    y

    ].

     I

    'm

     [

    character

     summary

    ].

     I

    'm

     a

    /an

     [

    character

     type

    ]

     who

     has

     always

     loved

     [

    reason

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

     and

     was

     established

     in

     the

     

    1

    2

    th

     century

    .

     It

     is

     the

     largest

     city

     in

     both

     Europe

     and

     the

     world

     and

     is

     home

     to

     the

     E

    iff

    el

     Tower

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     cultural

     attractions

    .

     Paris

     is

     also

     famous

     for

     its

     fashion

     and

     food

     scene

    ,

     including

     the

     famous

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

     Paris

    ian

     cuisine

    .

     The

     city

     is

     located

     on

     the

     Mediterranean

     Sea

     and

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     a

     Thousand

     Forms

    ".

     Paris

     has

     a

     rich

     culture

     and

     history

     that

     dates

     back

     over

     

    2

    ,

    5

    0

    0

     years

     and

     continues

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     landscape

     with

     many

     potential

     trends

     that

     are

     shaping

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     proficiency

     in

     AI

    :

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     AI

     systems

     become

     more

     capable

     and

     efficient

     at

     solving

     complex

     problems

    .

     This

     could

     involve

     developments

     in

     natural

     language

     processing

    ,

     machine

     learning

    ,

     and

     other

     areas

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     there

     will

     likely

     be

     a

     growing

     concern

     about

     privacy

     and

     security

    .

     This

     could

     include

     new

     regulations

     and

     technologies

     for

     protecting

     user

     data

     and

     preventing

     cyber

     attacks

    .
    


    3

    .

     AI

     in

    



```python
llm.shutdown()
```
