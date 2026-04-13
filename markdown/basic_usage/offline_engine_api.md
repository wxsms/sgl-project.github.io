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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.39it/s]


    2026-04-13 14:59:28,426 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 14:59:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.80it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.80it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.80it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.80it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.52it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.52it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.52it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.52it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.59it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.59it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.59it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 12.78it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 12.78it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 12.78it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 12.78it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 12.78it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 17.21it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 17.21it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 17.21it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 17.21it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 17.21it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 24.17it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 26.54it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 26.54it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 26.54it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 26.54it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 26.54it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.49it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 32.41it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 32.41it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 32.41it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 32.41it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 32.41it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 32.31it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 32.31it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.51it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.51it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.51it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.51it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.51it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.51it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:04<00:00, 36.69it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:04<00:00, 36.69it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:04<00:00, 36.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:08,  7.06it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:08,  7.06it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.38it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:07,  7.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:07,  7.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:07,  6.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.38 GB):   9%|▊         | 5/58 [00:00<00:07,  6.94it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.38 GB):  10%|█         | 6/58 [00:00<00:07,  6.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.39 GB):  10%|█         | 6/58 [00:00<00:07,  6.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.39 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.89 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.53it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.89 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.89 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.89 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.89 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.91it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=57.88 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.88 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.88 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.87 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.88it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=57.87 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.87 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.87 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.87 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.86 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.86 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.86 GB):  31%|███       | 18/58 [00:01<00:02, 17.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.86 GB):  31%|███       | 18/58 [00:01<00:02, 17.02it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=57.85 GB):  31%|███       | 18/58 [00:01<00:02, 17.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.85 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.83 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=960 avail_mem=57.84 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.41it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=57.84 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.40it/s]Capturing num tokens (num_tokens=896 avail_mem=57.84 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.40it/s]Capturing num tokens (num_tokens=832 avail_mem=57.84 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.40it/s]Capturing num tokens (num_tokens=832 avail_mem=57.84 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=768 avail_mem=57.67 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.72it/s]

    Capturing num tokens (num_tokens=704 avail_mem=57.67 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=704 avail_mem=57.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.35it/s]Capturing num tokens (num_tokens=640 avail_mem=57.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.35it/s]Capturing num tokens (num_tokens=576 avail_mem=57.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.35it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.67 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.07it/s]Capturing num tokens (num_tokens=512 avail_mem=57.66 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.07it/s]Capturing num tokens (num_tokens=480 avail_mem=57.67 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.07it/s]Capturing num tokens (num_tokens=480 avail_mem=57.67 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.88it/s]Capturing num tokens (num_tokens=448 avail_mem=57.67 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.88it/s]

    Capturing num tokens (num_tokens=416 avail_mem=57.67 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.88it/s]Capturing num tokens (num_tokens=416 avail_mem=57.67 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.78it/s]Capturing num tokens (num_tokens=384 avail_mem=57.67 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.78it/s]Capturing num tokens (num_tokens=352 avail_mem=57.66 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.78it/s]

    Capturing num tokens (num_tokens=352 avail_mem=57.66 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.64it/s]Capturing num tokens (num_tokens=320 avail_mem=57.66 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.64it/s]Capturing num tokens (num_tokens=288 avail_mem=57.65 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.64it/s]Capturing num tokens (num_tokens=288 avail_mem=57.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.63it/s]Capturing num tokens (num_tokens=256 avail_mem=57.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.63it/s]

    Capturing num tokens (num_tokens=240 avail_mem=57.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.63it/s]Capturing num tokens (num_tokens=240 avail_mem=57.65 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=224 avail_mem=57.65 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=208 avail_mem=57.64 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.55it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.54it/s]Capturing num tokens (num_tokens=192 avail_mem=57.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.54it/s]Capturing num tokens (num_tokens=176 avail_mem=57.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.54it/s]Capturing num tokens (num_tokens=176 avail_mem=57.64 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.53it/s]Capturing num tokens (num_tokens=160 avail_mem=57.63 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.53it/s]

    Capturing num tokens (num_tokens=144 avail_mem=57.63 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.53it/s]Capturing num tokens (num_tokens=144 avail_mem=57.63 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=128 avail_mem=57.63 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=112 avail_mem=57.63 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.55it/s]

    Capturing num tokens (num_tokens=112 avail_mem=57.63 GB):  79%|███████▉  | 46/58 [00:03<00:00, 12.53it/s]Capturing num tokens (num_tokens=96 avail_mem=57.17 GB):  79%|███████▉  | 46/58 [00:03<00:00, 12.53it/s] Capturing num tokens (num_tokens=80 avail_mem=54.11 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.53it/s]Capturing num tokens (num_tokens=80 avail_mem=54.11 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.50it/s]Capturing num tokens (num_tokens=64 avail_mem=54.11 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.50it/s]

    Capturing num tokens (num_tokens=48 avail_mem=54.10 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.50it/s]Capturing num tokens (num_tokens=48 avail_mem=54.10 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.56it/s]Capturing num tokens (num_tokens=32 avail_mem=54.10 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.56it/s]Capturing num tokens (num_tokens=28 avail_mem=54.09 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.56it/s]

    Capturing num tokens (num_tokens=28 avail_mem=54.09 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.46it/s]Capturing num tokens (num_tokens=24 avail_mem=54.09 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.46it/s]Capturing num tokens (num_tokens=20 avail_mem=54.09 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.46it/s]Capturing num tokens (num_tokens=20 avail_mem=54.09 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.45it/s]Capturing num tokens (num_tokens=16 avail_mem=54.09 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.45it/s]

    Capturing num tokens (num_tokens=12 avail_mem=54.08 GB):  93%|█████████▎| 54/58 [00:04<00:00, 12.45it/s]Capturing num tokens (num_tokens=12 avail_mem=54.08 GB):  97%|█████████▋| 56/58 [00:04<00:00, 12.47it/s]Capturing num tokens (num_tokens=8 avail_mem=54.08 GB):  97%|█████████▋| 56/58 [00:04<00:00, 12.47it/s] Capturing num tokens (num_tokens=4 avail_mem=54.08 GB):  97%|█████████▋| 56/58 [00:04<00:00, 12.47it/s]

    Capturing num tokens (num_tokens=4 avail_mem=54.08 GB): 100%|██████████| 58/58 [00:04<00:00, 12.57it/s]Capturing num tokens (num_tokens=4 avail_mem=54.08 GB): 100%|██████████| 58/58 [00:04<00:00, 11.73it/s]


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
    Generated text:  Rhea and I have a certificate in chemistry from the University of South Carolina, South Carolina, and I have more than 20 years of experience with chemicals and industrial processes in the chemical industry. I specialize in the accurate analysis of moisture and contaminants in petroleum, non-oil (i.e. coal), and other industrial materials. We're also a part of the ASME qualified Environmental Testing Laboratory (ETL).
    I'm also certified to perform binary and ternary equilibria and titrations. I have extensive training and experience with nitrogen analysis, solvent analysis, gas analysis and dissolved gas analysis. I also have extensive training and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political party that is in power in the United States. The United States has two major political parties: the Democratic Party and the Republican Party. These parties each have several factions within their ranks. One faction of the Democratic Party is the National Republican Party. This party was formed in 1968. The National Republican Party has a membership of about 500,000 people. It is a member of the U.S. Constitution and is a component of the American Political System. This faction is responsible for representing the rural and agricultural constituencies in the U.S. Congress. The National Republican Party is a subgroup of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is the capital of France. It has the following two streets: Boulevard Saint Germain and Avenue Montaigne.
    The two streets have a total length of 1300 meters. They are situated at the same horizontal distance from each other, and they share a single point of intersection.
    Find the distance between the two streets in meters.
    To find the distance between the two streets in Paris, we need to determine the lengths of the two streets and then use the Pythagorean theorem. Let's denote the length of the two streets as follows:
    
    - \( AB = 1300 \) meters.
    -
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. It's an interesting and exciting time to be alive. As AI evolves, it may change the way we live, work, and interact with the world around us. However, it's essential to understand the potential risks and challenges that come with this technology. AI has the power to automate many aspects of our lives and make them more efficient and convenient. However, it can also lead to new forms of automation, leading to job displacement, and exacerbating income inequality. We must weigh the benefits and drawbacks of AI before we adopt it in our daily lives.
    
    ## What is AI and How Does It Work?
    
    Artificial Intelligence (AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. What can I help you with today? [Name] is looking for a [job title] at [company name]. I'm excited to meet you and learn more about your skills and experience. What can I help you with today? [Name] is looking for a [job title] at [company name]. I'm excited to meet you and learn more about your skills and experience. What can I help you with today? [Name] is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a popular tourist destination and a major economic center in France. It is home to many world-renowned museums, theaters, and art galleries. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a cultural and artistic hub, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to continue to be used for a wide range of applications, from healthcare to finance to transportation, and will likely become an increasingly important part of our society. However, there are also potential risks and challenges associated with AI, including issues such as bias, privacy, and security, and it is important that we continue to work to address these concerns. Overall, the future of AI is
    


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
    Generated text:  [Name], and I'm a [Job Title] in the [Industry] field. I'm passionate about [My Love Interest's Name], and I believe we can create something amazing with our collaboration. Let's get started on this project together!
    
    ---
    
    Feel free to adjust any details to better match your own character or situation! I'm looking forward to our conversation. 🌐✨
    
    ---
    
    Any other suggestions for how to make it even more personalized?
    
    ---
    
    Please include some **specific details** related to the character and their profession to make the introduction stand out. 🎉
    
    ---
    
    Looking forward to your response and staying in touch
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is located on the Île de la Cité. It is the largest city in Europe and has a population of over 3 million. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower and the Louvre Museum. The city is also home to many famous universities and is a major center for business, tourism, and science. Paris is a UNESCO World Heritage site and is considered a cultural and political center of France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in areas such as machine learning, natural language processing, robotics, and autonomous systems. Some potential future trends include:
    
    1. Improved accuracy of AI in areas such as facial recognition, speech recognition, and image analysis. This will likely lead to applications in areas such as law enforcement, healthcare, and customer service.
    
    2. The development of more sophisticated AI that can interact with humans in more natural and empathetic ways. This could lead to more effective emotional intelligence and empathy in AI systems.
    
    3. The increasing use of AI in industries such as finance, manufacturing, and transportation. This could lead to more efficient and cost


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

     creative

     graphic

     designer

     with

     experience

     in

     marketing

     and

     branding

    .

     My

     work

     has

     been

     featured

     in

     numerous

     publications

     and

     digital

     spaces

    .

     I

     thrive

     on

     discovering

     new

     ideas

     and

     bringing

     them

     to

     life

    .

     Let

     me

     know

     if

     you

    'd

     like

     to

     learn

     more

     about

     me

     or

     if

     I

     can

     help

     you

     with

     something

    !

     

    📖

    🎨

     #

    Creative

     #

    Graphic

    Designer

     #

    Marketing

    Professional

    
    


    What

     are

     some

     ways

     to

     showcase

     your

     skills

     as

     a

     graphic

     designer

    ?

     As

     a

     creative

     graphic

     designer

     with

     experience

     in

     marketing

     and

     branding

    ,

     showcasing

     your

     skills

     can

     be

     approached

     in

     several

     ways

    .

     Here

     are

     some

     strategies

     to

     help

     you

     stand

     out

     and

     effectively

     communicate

     your

     work

    :
    


    1

    .

     **

    Work

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     River

     Se

    ine

    ,

     near

     the

     city

     of

     Paris

    ,

     which

     is

     one

     of

     the

     largest

     cities

     in

     Europe

    .

     It

     is

     also

     the

     seat

     of

     the

     French

     government

     and

     capital

     of

     the

     French

     Republic

    .

     Paris

     is

     a

     world

    -f

    amous

     cultural

    ,

     artistic

    ,

     and

     financial

     center

    .

     The

     city

     is

     famous

     for

     its

     distinctive

     architecture

    ,

     museums

    ,

     and

     opera

    .

     It

     is

     also

     a

     major

     tourist

     destination

     and

     home

     to

     several

     museums

     and

     historical

     sites

    .

     Paris

     has

     a

     rich

     and

     diverse

     cultural

     scene

    ,

     and

     is

     known

     for

     its

     gastr

    onomy

    ,

     art

    ,

     and

     nightlife

    .

     The

     French

     capital

     is

     home

     to

     numerous

     museums

    ,

     monuments

    ,

     and

     historical

     sites

    ,

     as

     well

     as

     famous

     landmarks

     such

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     number

     of

     trends

     that

     will

     shape

     the

     direction

     of

     the

     technology

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     AI

     is

     already

     becoming

     increasingly

     integrated

     into

     our

     daily

     lives

    ,

     from

     smart

     home

     appliances

     to

     virtual

     assistants

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

     to

     see

     even

     more

     seamless

     integration

     of

     AI

     into

     our

     daily

     routines

    .
    


    2

    .

     AI

     for

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     help

     diagnose

     diseases

    ,

     predict

     the

     spread

     of

     infections

    ,

     and

     improve

     treatment

     outcomes

    .

     As

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     AI

     used

     for

     even

     more

     critical

     applications

     in

     healthcare

    



```python
llm.shutdown()
```
