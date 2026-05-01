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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.59it/s]


    2026-05-01 19:59:17,530 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 19:59:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  3.07it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  3.07it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  3.07it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:16,  3.07it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:06,  7.28it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:06,  7.28it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:06,  7.28it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:06,  7.28it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:06,  7.28it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 10.96it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 10.96it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 10.96it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 10.96it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 10.96it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 14.84it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 14.84it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 14.84it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 14.84it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 14.84it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 14.84it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 24.79it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 24.79it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 24.79it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 24.79it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 24.79it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 24.79it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 33.53it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 33.53it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 36.39it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 36.39it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 39.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.33it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.53it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:06,  7.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:06,  7.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.96it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.51it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.45it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.92it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.36 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.75it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.21it/s]Capturing num tokens (num_tokens=960 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.21it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:02, 15.69it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:02, 15.69it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.69it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.79it/s]Capturing num tokens (num_tokens=768 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.79it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.79it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.15it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.15it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.15it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=512 avail_mem=58.32 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.60it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.60it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.33 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.60it/s]Capturing num tokens (num_tokens=416 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.41it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.41it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.41it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.31it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.31it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 13.31it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.38it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.38it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.31 GB):  62%|██████▏   | 36/58 [00:03<00:01, 13.38it/s]Capturing num tokens (num_tokens=240 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.47it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.47it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.47it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.64it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.64it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.64it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 13.65it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 13.65it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 13.65it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:01, 11.74it/s]Capturing num tokens (num_tokens=128 avail_mem=58.29 GB):  76%|███████▌  | 44/58 [00:03<00:01, 11.74it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  76%|███████▌  | 44/58 [00:03<00:01, 11.74it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:01, 11.56it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:01, 11.56it/s] Capturing num tokens (num_tokens=80 avail_mem=58.28 GB):  79%|███████▉  | 46/58 [00:03<00:01, 11.56it/s]Capturing num tokens (num_tokens=80 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.30it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.30it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.30it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.95it/s]Capturing num tokens (num_tokens=32 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.95it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.95it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.43it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.43it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.43it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.43it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.83it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.83it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.83it/s] Capturing num tokens (num_tokens=4 avail_mem=76.60 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.83it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.60 GB): 100%|██████████| 58/58 [00:04<00:00, 12.72it/s]


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
    Generated text:  Sarah. I'm an 11-year-old girl who loves to eat lots of junk food. I'm too shy to talk to people outside of my family. Every day, I eat a huge pizza and have a big lunch. I do my homework and play video games in the evening. I'm really happy to have a lot of time to play video games. I have a good friend who is in the same class as me. He has a very small bedroom, but he spends all his time playing video games. I often talk to him on the phone. We chat a lot, and he helps me understand what I'm doing
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is the leader of the country and must make important decisions. Everyone in the country must trust the president. If the president makes a mistake, the whole country will lose its trust in him or her. In America, the president must answer to the Congress, which is the highest legislative body in the country.
    The president is responsible for the day-to-day running of the country. He or she must make decisions about the military, the economy, and the laws that govern the country. The president has a lot of power and must act within the limits of the law. If he or she
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the northwest of the country, near the river Seine and the 10th century castle of the castle Trianon, which has been built at the end of the 18th century.
    The city of Paris is a very large city and is home to a high density of population. Some of the oldest buildings in the city were built in the 12th century.
    The city is divided into 3 parts, called "L'Est", "Le Centre" and "Le Vieux-Paris". The 2 outer parts, Le Centre and Le Vieux-Paris, are not the main parts of
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly evolving and it’s clear that AI will play a vital role in the transformation of how we live and work. But what’s AI and what does it do? In this article, we will explore what AI is, how it works, and the potential applications of AI in different industries. We will also discuss the ethical considerations that need to be taken into account when using AI.
    What is AI?
    AI stands for Artificial Intelligence, which is a type of machine learning that allows a computer to perform tasks that normally require human intelligence. AI can be used in a variety of fields, including healthcare, finance, transportation, and more.
    How does


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm [age] years old, [gender] and I have [number] years of experience in [job title]. I'm [job title] and I'm always looking for ways to [describe a goal or activity]. I'm [job title] and I'm always looking for ways to [describe a goal or activity]. I'm [job title] and I'm always looking for ways to [describe a goal or activity]. I'm [job
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many international institutions and organizations, including the French Academy of Sciences and the European Parliament. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has played a significant role in French history and continues to be a major economic and cultural center in the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more personalized and accurate diagnoses and treatments.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can
    


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
    Generated text:  Alex, a friendly, quick-witted, and adventurous young man. I'm a young man with a playful spirit and a knack for finding new and exciting things. Whether I'm hiking in the mountains, traveling the world, or exploring new culinary adventures, I'm always looking for the next adventure and a fresh perspective on life. I'm a good listener, and a great friend, and I love to explore the world and make new friends. I'm excited to see where my imagination takes me and what kind of adventures I'll have on my next journey. I hope I can meet you in the next chapter of my life and join you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the city of love and a cultural melting pot of art and sophistication. 
    
    Additionally, provide the following information:
    - The historical significance of Paris, dating back to the Roman period.
    - The role of the French monarchy in shaping Paris' cultural identity.
    - The 2017 Paris climate change agreement.
    - The current population of Paris, which is approximately 1,385,887.
    
    Finally, provide the French version of the sentence you created, making sure to use the correct grammar and spelling. 
    
    Paris, known as the city of love, is a city of art, culture, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to be characterized by increasing complexity and sophistication, as well as growing integration with other technologies and industries. Here are some potential trends that could emerge in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more advanced, there will be increased pressure to ensure that it is used ethically and responsibly. This will likely involve developing new ethical frameworks and guidelines to govern the use of AI, as well as more transparent and accountable decision-making processes.
    
    2. AI-powered autonomous vehicles: Autonomous vehicles could become more prevalent as AI continues to improve in terms of accuracy and range. This could have a significant impact on transportation and


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

     __

    ________

    __

     and

     I

    'm

     a

    /an

     __

    ________

    __

     (

    profession

    /

    academic

     area

    /

    occupation

    ).

     I

    'm

     __

    ________

    __

     (

    born

     date

    /m

    ar

    riage

     date

    /

    other

    )

     and

     I

    'm

     __

    ________

    __

     (

    number

     of

     children

    ).

     I

    'm

     from

     __

    ________

    __

     (

    town

    /c

    ity

    /place

    ).

     I

    've

     been

     married

     to

     __

    ________

    __

     (

    配偶

    的

    姓名

    )

     since

     __

    ________

    __

     (

    mar

    riage

     date

    ).

     We

     have

     been

     together

     for

     __

    ________

    __

     (

    years

    ).

     We

     have

     two

     children

    ,

     __

    ________

    __

     and

     __

    ________

    __.

     We

     love

     our

     family

    ,

     our

     home

    ,

     and

     our

     jobs

    .

     I

    'm

     passionate

     about

     __

    ________

    __

     (

    life

     interests

    )

     and

     I

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     is

     fact

    ually

     correct

     and

     concise

    .

     
    


    For

     the

     Paris

     location

    ,

     the

     full

     address

     is

     

    7

    5

    0

    0

    8

    ,

     Par

    ís

    ,

     France

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     most

     populous

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    8

     million

     as

     of

     

    2

    0

    2

    1

    .

     
    


    I

     will

     not

     use

     the

     word

     "

    city

    "

     in

     the

     statement

     as

     per

     the

     instructions

    .

     
    


    I

     will

     use

     the

     following

     code

     in

     my

     response

    :

     
    


    ```

    python

    


    def

     concise

    _f

    acts

    (city

    ):


       

     return

     f

    "The

     capital

     of

     {

    city

    }

     is

     {

    city

    }.

    "


    ``

    `
    


    This

     code

     defines

     a

     function

     that

     takes

     the

     name

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     it

     is

     likely

     to

     continue

     to

     evolve

     at

     an

     unprecedented

     pace

    .

     Here

     are

     some

     of

     the

     most

     promising

     trends

     in

     artificial

     intelligence

     that

     are

     expected

     to

     shape

     the

     industry

     in

     the

     next

     decade

    :
    


    1

    .

     Self

    -driving

     cars

    :

     Self

    -driving

     cars

     are

     likely

     to

     become

     more

     common

     in

     the

     next

     decade

    ,

     and

     they

     will

     be

     driven

     by

     AI

     algorithms

     that

     can

     detect

     and

     avoid

     obstacles

     and

     navigate

     complex

     roads

    .
    


    2

    .

     Chat

    bots

     and

     virtual

     assistants

    :

     AI

    -powered

     chat

    bots

     and

     virtual

     assistants

     will

     become

     more

     sophisticated

    ,

     enabling

     them

     to

     provide

     personalized

     and

     convenient

     services

     to

     customers

    .
    


    3

    .

     Medical

     diagnosis

     and

     treatment

    :

     AI

     algorithms

     will

     be

     used

     to

     improve

     the

     accuracy

     and

     speed

     of

     medical

    



```python
llm.shutdown()
```
