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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 06:31:47] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.57it/s]


    2026-04-26 06:31:52,024 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 06:31:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  6.21it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]

    Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 24.56it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]

    Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 31.94it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 41.37it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 41.37it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 41.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.20 GB):   3%|▎         | 2/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.18 GB):   3%|▎         | 2/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.17 GB):   3%|▎         | 2/58 [00:00<00:03, 16.11it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=116.17 GB):   7%|▋         | 4/58 [00:00<00:03, 16.33it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.17 GB):   7%|▋         | 4/58 [00:00<00:03, 16.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.17 GB):   7%|▋         | 4/58 [00:00<00:03, 16.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.17 GB):  10%|█         | 6/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.16 GB):  10%|█         | 6/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  10%|█         | 6/58 [00:00<00:03, 16.06it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.53it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=960 avail_mem=116.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.43it/s] Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=832 avail_mem=116.11 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=768 avail_mem=116.10 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=704 avail_mem=116.10 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.43it/s]Capturing num tokens (num_tokens=704 avail_mem=116.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.32it/s]Capturing num tokens (num_tokens=640 avail_mem=116.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.32it/s]

    Capturing num tokens (num_tokens=576 avail_mem=116.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.32it/s]Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.32it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.32it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.55it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.26it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.26it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.26it/s]Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.26it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.26it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.68it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.68it/s]Capturing num tokens (num_tokens=208 avail_mem=116.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.68it/s]Capturing num tokens (num_tokens=192 avail_mem=116.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.68it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.68it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.31it/s]

    Capturing num tokens (num_tokens=144 avail_mem=116.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=128 avail_mem=116.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=96 avail_mem=116.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.32it/s] Capturing num tokens (num_tokens=80 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=64 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=48 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=48 avail_mem=116.04 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=32 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.48it/s]

    Capturing num tokens (num_tokens=28 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=24 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=16 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=12 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=8 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.83it/s] Capturing num tokens (num_tokens=4 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.83it/s]

    Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:01<00:00, 34.50it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:01<00:00, 29.32it/s]


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
    Generated text:  Mark. I'm an engineer working in the automotive industry, specializing in electric vehicle technology.
    I’m a freelance writer, writer, and the author of the popular science fiction novel, "The Electric Mind."
    What is the difference between a "high school diploma" and a "college degree"?
    What is the difference between a "bachelor's degree" and a "master's degree"?
    Do you have any recommendations for a good book that would educate someone about a subject that I would be interested in? That is, a subject that I find interesting and keep up to date with.
    Hello Mark, thank you for your time and for
    ===============================
    Prompt: The president of the United States is
    Generated text:  considering implementing a new policy to combat climate change. The policy involves purchasing a large number of carbon credits from a company that has been involved in significant environmental and carbon emission reduction efforts. The company has a base price of $10 per credit and can sell each credit for $15.
    
    The president decides to buy 100 credits. After purchasing, the carbon credits are sold for $3 per credit. Calculate the total profit or loss the president would have made from this transaction.
    
    To calculate the total profit or loss the president would have made from this transaction, we need to consider both the revenue from selling the carbon credits and the
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Marseille
    C. Nice
    D. Lyon
    Answer: A
    
    Which of the following scenarios would lead to a positive autocorrelation coefficient?
    A. The correlation between variables in a regression analysis is 0.00
    B. The correlation between variables in a regression analysis is -0.80
    C. The correlation between variables in a regression analysis is 0.80
    D. The correlation between variables in a regression analysis is -0.00
    Answer: C
    
    Which of the following items would be the best choice for long-term investment?
    A. A piece
    ===============================
    Prompt: The future of AI is
    Generated text:  digital and computational
    This may sound like an amusing observation, but it is a very serious one. When it comes to AI, the future is digital and computational. This means that AI is going to continue to evolve and grow in complexity and sophistication. It is going to be more and more integrated into the digital world, and it is going to be used to solve problems that were previously impossible to solve.
    The integration of AI into the digital world will not only allow for new possibilities, but it will also pose new challenges. With AI, we are seeing a shift from the traditional way of thinking about technology, towards a more digital and computational


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting the headquarters of many major French companies and institutions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and literature of the 19th century. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among other landmarks. Paris is a popular tourist destination, with millions of visitors annually. It is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk assessment, and portfolio management
    


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
    Generated text:  [Your Name], and I'm a [Occupation] who has been studying [Occupation] for [Number] years now. I'm always [What You Do Best], and I love [Favorite Thing to Do]. What are some of your most memorable experiences? I'm always looking to learn new things, so if you have any questions or need help, don't hesitate to ask. I'm looking forward to the day when we can work together. Welcome to my world, and I hope you enjoy your time here. [Your Name] [Occupation] [Your profession] [Your name] [Your occupation] Hello,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, and is located in the heart of the city of Paris and serves as the political, cultural, and economic center of France.
    
    Please provide a detailed description of the city's architecture and landmarks:
    
    The city of Paris is known for its elegant and opulent architecture. The most famous landmark is the Eiffel Tower, a wrought iron structure that rises 324 meters (1,063 feet) above the city's skyline. Other notable structures include the Louvre Museum, which houses the famous Mona Lisa and other world-renowned art collections, and the Notre-Dame Cathedral,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with many possible developments and applications. Some possible trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to assist with diagnoses, drug development, and patient care. We may see even more applications in the future, with AI used to predict and prevent disease, develop personalized treatment plans, and enhance healthcare accessibility.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes and reduce costs. We may see even more applications in the future, with AI used to predict equipment failures, optimize supply chain management, and improve product quality.
    
    3. AI in finance:


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

    ]

     and

     I

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     passionate

     about

     [

    Reason

     for

     passion

    ],

     and

     I

     believe

     in

     [

    Unique

     Experience

     or

     Achievement

    ].

     I

     am

     constantly

     learning

     and

     growing

    ,

     and

     I

     strive

     to

     be

     the

     best

     version

     of

     myself

     every

     day

    .

     I

     enjoy

     [

    Side

     Note

    ],

     and

     I

     believe

     in

     [

    Goal

     or

     Promise

    ].

     My

     goal

     is

     to

     make

     a

     positive

     impact

     in

     [

    Cause

    /

    Field

    ],

     and

     I

     am

     committed

     to

     [

    Action

    ]

     to

     achieve

     this

     goal

    .

     How

     do

     you

     feel

     about

     yourself

    ?

     I

     am

     confident

     and

     strong

    ,

     and

     I

     am

     proud

     of

     what

     I

     have

     achieved

     so

     far

    .

     I

     am

     always

     eager

     to

     learn

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .
    


    That

    's

     correct

    !

     Paris

    ,

     the

     capital

     city

     of

     France

    ,

     is

     famous

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     annual

     annual

     festival

     known

     as

     the

     E

    iff

    el

     Tower

     World

     Heritage

     Site

    .

     The

     city

     is

     also

     known

     for

     its

     diverse

     culinary

     scene

     and

     global

     influence

    .

     It

    's

     a

     popular

     tourist

     destination

    ,

     with

     many

     tourists

     choosing

     to

     visit

     during

     the

     summer

     months

     for

     the

     warm

     weather

     and

     vibrant

     culture

    .

     Paris

     has

     been

     a

     UNESCO

     World

     Heritage

     site

     since

     

    1

    9

    8

    5

    .

     

    😊

    ✨

    ✨

    
    


    I

    'm

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     not

     yet

     clear

    ,

     but

     several

     trends

     are

     likely

     to

     shape

     the

     technology

    ’s

     development

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

     With

     the

     growing

     demand

     for

     personalized

     healthcare

    ,

     AI

     will

     become

     more

     prevalent

     in

     the

     field

    .

     This

     will

     lead

     to

     the

     development

     of

     more

     accurate

     and

     efficient

     diagnostic

     tools

    ,

     treatment

     plans

    ,

     and

     patient

     care

    .
    


    2

    .

     Integration

     of

     AI

     in

     consumer

     products

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     consumer

     products

     like

     smart

     home

     devices

    ,

     wearable

     technology

    ,

     and

     intelligent

     transportation

     systems

    .

     This

     will

     make

     the

     lives

     of

     consumers

     more

     convenient

     and

     efficient

    .
    


    3

    .

     Automation

     of

     jobs

    :

     AI

     will

     automate

     many

     jobs

     in

     industries

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    .

    



```python
llm.shutdown()
```
