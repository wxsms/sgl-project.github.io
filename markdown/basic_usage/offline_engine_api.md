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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.41it/s]


    2026-04-09 08:34:06,511 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 08:34:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:03,  1.13s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:03,  1.13s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:03,  1.13s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:14,  3.53it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:14,  3.53it/s]

    Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.53it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.15it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.15it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.15it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:06,  6.98it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:06,  6.98it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:06,  6.98it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:06,  6.98it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04,  9.97it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.12it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.12it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.12it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.12it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 15.58it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 15.58it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 15.58it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 15.58it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 18.51it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 20.18it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 20.18it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 20.18it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 20.18it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 22.29it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 22.29it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 22.29it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 22.29it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 23.02it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 23.02it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 23.02it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 23.02it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 23.02it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 25.98it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 26.03it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 26.03it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 26.03it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 26.03it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 26.03it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 27.54it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 27.54it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 27.54it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 27.54it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 28.14it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 28.14it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 28.14it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 28.14it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 28.35it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 28.35it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 28.35it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 28.35it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 28.35it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 29.64it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 29.64it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 29.64it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 29.64it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 29.64it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 29.64it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 33.00it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 33.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.10 GB):   2%|▏         | 1/58 [00:00<00:09,  5.90it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.07 GB):   2%|▏         | 1/58 [00:00<00:09,  5.90it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.07 GB):   3%|▎         | 2/58 [00:00<00:09,  6.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.07 GB):   3%|▎         | 2/58 [00:00<00:09,  6.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.07 GB):   5%|▌         | 3/58 [00:00<00:09,  5.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.06 GB):   5%|▌         | 3/58 [00:00<00:09,  5.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=50.06 GB):   7%|▋         | 4/58 [00:00<00:08,  6.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.06 GB):   7%|▋         | 4/58 [00:00<00:08,  6.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.06 GB):   9%|▊         | 5/58 [00:00<00:08,  6.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.05 GB):   9%|▊         | 5/58 [00:00<00:08,  6.55it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=50.05 GB):  10%|█         | 6/58 [00:00<00:07,  6.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.05 GB):  10%|█         | 6/58 [00:00<00:07,  6.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.05 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.04 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.53it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=50.04 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.03 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.03 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.00 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.45it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=50.00 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.99 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.00 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.00 GB):  21%|██        | 12/58 [00:01<00:04,  9.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.00 GB):  21%|██        | 12/58 [00:01<00:04,  9.29it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=50.00 GB):  21%|██        | 12/58 [00:01<00:04,  9.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.00 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.99 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.99 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.13it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=49.99 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.98 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.97 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.97 GB):  31%|███       | 18/58 [00:02<00:03, 12.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.97 GB):  31%|███       | 18/58 [00:02<00:03, 12.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.96 GB):  31%|███       | 18/58 [00:02<00:03, 12.19it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=49.96 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.93 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.08it/s]Capturing num tokens (num_tokens=960 avail_mem=49.94 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.08it/s] Capturing num tokens (num_tokens=960 avail_mem=49.94 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.81it/s]Capturing num tokens (num_tokens=896 avail_mem=49.94 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.81it/s]Capturing num tokens (num_tokens=832 avail_mem=49.93 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.81it/s]

    Capturing num tokens (num_tokens=832 avail_mem=49.93 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.52it/s]Capturing num tokens (num_tokens=768 avail_mem=49.94 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.52it/s]Capturing num tokens (num_tokens=704 avail_mem=49.93 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.52it/s]Capturing num tokens (num_tokens=704 avail_mem=49.93 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.21it/s]Capturing num tokens (num_tokens=640 avail_mem=49.90 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.21it/s]Capturing num tokens (num_tokens=576 avail_mem=49.92 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.21it/s]

    Capturing num tokens (num_tokens=576 avail_mem=49.92 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.70it/s]Capturing num tokens (num_tokens=512 avail_mem=49.91 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.70it/s]Capturing num tokens (num_tokens=480 avail_mem=49.92 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.70it/s]Capturing num tokens (num_tokens=480 avail_mem=49.92 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.36it/s]Capturing num tokens (num_tokens=448 avail_mem=49.91 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.36it/s]Capturing num tokens (num_tokens=416 avail_mem=49.91 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.36it/s]

    Capturing num tokens (num_tokens=416 avail_mem=49.91 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.58it/s]Capturing num tokens (num_tokens=384 avail_mem=49.90 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.58it/s]Capturing num tokens (num_tokens=352 avail_mem=49.90 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.58it/s]Capturing num tokens (num_tokens=352 avail_mem=49.90 GB):  59%|█████▊    | 34/58 [00:02<00:01, 17.15it/s]Capturing num tokens (num_tokens=320 avail_mem=49.87 GB):  59%|█████▊    | 34/58 [00:02<00:01, 17.15it/s]Capturing num tokens (num_tokens=288 avail_mem=49.88 GB):  59%|█████▊    | 34/58 [00:03<00:01, 17.15it/s]

    Capturing num tokens (num_tokens=288 avail_mem=49.88 GB):  62%|██████▏   | 36/58 [00:03<00:01, 17.25it/s]Capturing num tokens (num_tokens=256 avail_mem=49.88 GB):  62%|██████▏   | 36/58 [00:03<00:01, 17.25it/s]Capturing num tokens (num_tokens=240 avail_mem=49.87 GB):  62%|██████▏   | 36/58 [00:03<00:01, 17.25it/s]Capturing num tokens (num_tokens=240 avail_mem=49.87 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.02it/s]Capturing num tokens (num_tokens=224 avail_mem=49.87 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.02it/s]

    Capturing num tokens (num_tokens=208 avail_mem=49.86 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.02it/s]Capturing num tokens (num_tokens=208 avail_mem=49.86 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.73it/s]Capturing num tokens (num_tokens=192 avail_mem=49.85 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.73it/s]Capturing num tokens (num_tokens=176 avail_mem=49.85 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.73it/s]

    Capturing num tokens (num_tokens=176 avail_mem=49.85 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.13it/s]Capturing num tokens (num_tokens=160 avail_mem=49.84 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.13it/s]Capturing num tokens (num_tokens=144 avail_mem=49.83 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.13it/s]Capturing num tokens (num_tokens=144 avail_mem=49.83 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.12it/s]Capturing num tokens (num_tokens=128 avail_mem=49.82 GB):  76%|███████▌  | 44/58 [00:03<00:01, 12.12it/s]

    Capturing num tokens (num_tokens=112 avail_mem=49.82 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.12it/s]Capturing num tokens (num_tokens=112 avail_mem=49.82 GB):  79%|███████▉  | 46/58 [00:04<00:01,  8.18it/s]Capturing num tokens (num_tokens=96 avail_mem=49.81 GB):  79%|███████▉  | 46/58 [00:04<00:01,  8.18it/s] Capturing num tokens (num_tokens=80 avail_mem=49.80 GB):  79%|███████▉  | 46/58 [00:04<00:01,  8.18it/s]

    Capturing num tokens (num_tokens=80 avail_mem=49.80 GB):  83%|████████▎ | 48/58 [00:04<00:01,  8.74it/s]Capturing num tokens (num_tokens=64 avail_mem=49.79 GB):  83%|████████▎ | 48/58 [00:04<00:01,  8.74it/s]Capturing num tokens (num_tokens=48 avail_mem=49.79 GB):  83%|████████▎ | 48/58 [00:04<00:01,  8.74it/s]Capturing num tokens (num_tokens=32 avail_mem=49.79 GB):  83%|████████▎ | 48/58 [00:04<00:01,  8.74it/s]Capturing num tokens (num_tokens=32 avail_mem=49.79 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.86it/s]Capturing num tokens (num_tokens=28 avail_mem=49.78 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.86it/s]Capturing num tokens (num_tokens=24 avail_mem=49.78 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.86it/s]Capturing num tokens (num_tokens=20 avail_mem=49.77 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.86it/s]

    Capturing num tokens (num_tokens=20 avail_mem=49.77 GB):  93%|█████████▎| 54/58 [00:04<00:00, 14.80it/s]Capturing num tokens (num_tokens=16 avail_mem=49.77 GB):  93%|█████████▎| 54/58 [00:04<00:00, 14.80it/s]Capturing num tokens (num_tokens=12 avail_mem=49.77 GB):  93%|█████████▎| 54/58 [00:04<00:00, 14.80it/s]Capturing num tokens (num_tokens=8 avail_mem=49.77 GB):  93%|█████████▎| 54/58 [00:04<00:00, 14.80it/s] Capturing num tokens (num_tokens=4 avail_mem=49.76 GB):  93%|█████████▎| 54/58 [00:04<00:00, 14.80it/s]Capturing num tokens (num_tokens=4 avail_mem=49.76 GB): 100%|██████████| 58/58 [00:04<00:00, 19.18it/s]Capturing num tokens (num_tokens=4 avail_mem=49.76 GB): 100%|██████████| 58/58 [00:04<00:00, 12.20it/s]


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
    Generated text:  Kenneth and I'm the editor of this website.
    Where can I find a list of schools I can apply to?
    You can search the website for a list of schools you can apply to in the "Schools Available" section at the bottom of the "Profile" page. Here, you can sort the schools by category, field of study, location, and more.
    What are some key factors to consider when choosing a school?
    There are several factors you should consider when choosing a school, including the school's reputation, the quality of its education, the student body, the school's reputation, the school's accreditation, and the school's
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. Every day he has to answer many questions from the people. One day, he is busy with some important questions, but at one point, he starts to think about how he answered them. He is very sad to hear that. He was afraid that he would never know if the questions were right. What can the president do to make sure he knows what he answered? He can think about all the questions again. He can ask other people who have seen him answer those questions. He can write down the answers carefully, so he can use them again later. He can talk to the people who are supposed to answer his questions
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    The capital of France is Paris. So, the correct answer is:
    
    A. Paris
    
    Paris, also known as "La Garde," is the capital city of France and is located in the region of Loire. It is known for its medieval architecture, iconic landmarks such as Notre-Dame Cathedral, and its historical importance in French culture. Paris is also the seat of the French government and is home to many important museums and cultural institutions. The city has a rich history dating back to the Roman Empire and the early medieval period, and it is considered
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, and it's here to stay. This is the background for our next discussion on the topic. AI refers to computer systems that can simulate human-like intelligence. It is commonly used in various fields such as medicine, finance, and transportation. However, there are still some issues that must be addressed before AI can be fully utilized. One of these is the ethical and moral implications of AI. The ethical and moral implications of AI are complex and multifaceted, and they depend on the context in which they are used. Some ethical and moral issues that must be addressed include the use of AI for malicious purposes, the potential for AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a cultural and economic center of France and a major tourist destination. It is home to many world-renowned museums, theaters, and landmarks. The city is also known for its annual festivals and events, including the Eiffel Tower Festival and the Paris Fashion Week. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations more effectively.
    
    2. Enhanced natural language processing: AI will continue to improve its ability to understand and interpret human language, leading to more natural and intuitive interactions with machines.
    
    3. Increased use of machine learning: AI will become more widely used in areas such as image and speech recognition, natural language processing, and predictive analytics, leading to more efficient and effective decision-making.
    
    4. Greater reliance on AI
    


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
    Generated text:  [Your Name], and I am a [Position] at [Your Company]. I love [Your Profession] because [Your Personal Trait or Passion]. I am always ready to [Why You Are Ready to be here]. I am a [Your Major], and I have been [Your Job Description]. I am a [Your Last Name] with over [Number] years of experience in [Your Profession], and I am always looking for ways to [Your Goal]. I am a [Your Major], and I am constantly learning and growing. I am a [Your Last Name], and I am always ready to learn and grow. Thank you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its magnificent architecture, rich history, and vibrant culture. 
    
    Paris, officially known as the French Capital of Culture, is the seat of France's government and is located in the Seine-Marne region, near Paris. The city is famous for its iconic Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the famous Montmartre neighborhood. Paris is also home to many world-renowned museums, theaters, and art galleries, making it a major tourist destination. 
    
    Despite its significance, Paris has faced challenges in recent years, including issues related to crime, pollution, and gentrification. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with numerous possibilities emerging in the coming years. Here are some potential trends:
    
    1. Increased accessibility: With advancements in AI technology, we may see more affordable AI solutions available to more people, making it easier and more accessible for individuals to use AI for a wide range of applications.
    
    2. More personalized experiences: AI is becoming increasingly capable of understanding and responding to personal preferences and needs, leading to more personalized experiences for users.
    
    3. Advanced autonomy: Autonomous AI systems are becoming more advanced, with the ability to make decisions and take actions on their own. This will likely lead to more autonomy in AI applications.
    
    4. Improved security


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

    occupation

     or

     profession

    ]

     who

     has

     always

     had

     an

     interest

     in

     [

    specific

     hobby

     or

     activity

    ].

     I

     like

     to

     [

    describe

     the

     hobby

     or

     activity

    ].

     
    


    I

     enjoy

     [

    activities

     that

     make

     me

     happy

     and

     energ

    ized

    ].

     
    


    I

     also

     love

     [

    int

    roduce

     any

     other

     hobbies

    ,

     talents

    ,

     or

     interests

     that

     you

     may

     have

    ].

     
    


    Please

     tell

     me

     about

     yourself

    .

     What

    's

     your

     favorite

     thing

     to

     do

    ?

     What

     hobbies

     do

     you

     have

    ?

     What

    's

     your

     best

     trait

    ?

     
    


    Thank

     you

     for

     taking

     the

     time

     to

     learn

     more

     about

     me

    .
    


    [

    Name

    ]

     is

     a

     [

    occupation

     or

     profession

    ]

     who

     enjoys

     [

    activities

     that

     make

     them

     happy

     and

     energ

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

     in

     the

     Mar

    ne

     Valley

    ,

     making

     it

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    .


    That

     statement

     is

     true

    ,

     but

     it

     doesn

    't

     fully

     capture

     the

     complexity

     and

     unique

     characteristics

     of

     Paris

    .

     Could

     you

     please

     clarify

     which

     aspect

     you

    're

     interested

     in

     learning

     about

     about

     Paris

    ?

     For

     example

    ,

     are

     you

     asking

     about

     its

     historical

     significance

    ,

     art

     and

     culture

    ,

     cuisine

    ,

     or

     something

     else

    ?

     Additionally

    ,

     could

     you

     specify

     the

     level

     of

     detail

     (

    e

    .g

    .,

     broad

     overview

    ,

     detailed

     description

    )

     you

     would

     like

     about

     Paris

    ?

     Paris

    ,

     also

     known

     as

     "

    La

     Gar

    de

    ,"

     is

     a

     city

     with

     a

     rich

     and

     diverse

     history

    .

     Its

     impressive

     architecture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     rapidly

    ,

     and

     there

     are

     several

     key

     trends

     that

     are

     likely

     to

     shape

     its

     direction

    .

     Some

     of

     the

     most

     likely

     trends

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     are

     likely

     to

     see

     even

     more

     integration

     with

     other

     technologies

     such

     as

     IoT

    ,

     blockchain

    ,

     and

     virtual

     reality

    .

     This

     will

     likely

     lead

     to

     new

     ways

     of

     communicating

    ,

     shopping

    ,

     and

     working

    ,

     as

     well

     as

     new

     possibilities

     for

     self

    -driving

     cars

     and

     other

     applications

    .
    


    2

    .

     Greater

     emphasis

     on

     ethical

     AI

    :

     As

     we

     become

     more

     aware

     of

     the

     potential

     impact

     of

     AI

     on

     society

    ,

     we

     are

     likely

     to

     see

     an

     increase

     in

    



```python
llm.shutdown()
```
