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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.34it/s]


    2026-05-01 07:08:19,158 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 07:08:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:18,  1.43s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:18,  1.43s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:18,  1.43s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:09,  4.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:09,  4.95it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:09,  4.95it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:09,  4.95it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:09,  4.95it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  7.65it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:06<00:05,  7.65it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:06<00:05,  7.65it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:03, 10.68it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:03, 10.68it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:03, 10.68it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:03, 10.68it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:03, 10.68it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:02, 14.31it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:02, 14.31it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:02, 14.31it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:02, 14.31it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:06<00:02, 14.31it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:06<00:01, 18.07it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:06<00:01, 18.07it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:01, 18.07it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:06<00:01, 18.07it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:06<00:01, 18.07it/s]

    Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:06<00:01, 18.07it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:00, 27.20it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:00, 27.20it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:00, 27.20it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:00, 27.20it/s]

    Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:00, 27.20it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:06<00:00, 27.20it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 31.54it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 31.54it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 31.54it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 31.54it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 31.54it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 31.54it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 34.06it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 34.06it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 34.06it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 34.06it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 34.06it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 34.06it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 35.99it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 35.99it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 35.99it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 35.99it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 35.99it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 35.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 39.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   2%|▏         | 1/58 [00:00<00:08,  7.04it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:08,  7.04it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:08,  6.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:08,  6.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.91it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.81it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:07,  7.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:07,  7.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.60it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.99it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  21%|██        | 12/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.52it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.60it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  31%|███       | 18/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  31%|███       | 18/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  31%|███       | 18/58 [00:02<00:03, 10.88it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.26it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.26it/s] Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.55it/s]Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.55it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.55it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.83it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.83it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.83it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.33it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.33it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.33it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.73it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.73it/s]

    Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.73it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.18it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.18it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.18it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.89it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.89it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.89it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.05it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.05it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.05it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.16it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.16it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.16it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.42it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.42it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.42it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.69it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.69it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.69it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.12it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.12it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.12it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.10it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.10it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.10it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.15it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.15it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.15it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 14.96it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 14.96it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:04<00:00, 14.96it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.19it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.19it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.19it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.61it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.61it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:04<00:00, 15.61it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 15.77it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 15.77it/s] Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  97%|█████████▋| 56/58 [00:04<00:00, 15.77it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:04<00:00, 16.13it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:04<00:00, 12.35it/s]


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
    Generated text:  Jim and I’m a college student. I'm from China and I'm going to study in the United States. I can’t wait to start here. But now I can’t understand the American spelling of some words. What do you think is the best way to learn the American English? It seems that I can't find the right way. I wonder what you can tell me?
    The problem you are facing is quite common, especially for students like yourself who are trying to learn American English. Here are some tips that might help you:
    
    1. Start with the basics: It's important to learn the basic rules of American English before you
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ___ position.
    
    A) executive  
    B) legislative  
    C) judicial  
    D) executive-legislative
    
    To determine the correct answer, we need to understand the roles of the President of the United States. The President is the head of the executive branch of the government, responsible for carrying out the policies and executing the laws passed by Congress. The other options provided (legislative and judicial) are not typically associated with the president.
    
    Let's analyze each option:
    
    A) Executive: This refers to the role of the President, but not the main office where decisions are made.
    B) Legislative: This refers to the role of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is one of the most important cities in the world and has a rich history. Here are some interesting facts about Paris that you might find interesting:
    
    1. Paris is the capital of France and is home to the French Parliament, the Senate, and the President of the Council of Ministers. It is also home to the French National Museum, the Louvre Museum, and many other important cultural institutions.
    
    2. The city of Paris is home to many historic landmarks such as the Eiffel Tower, the Palace of Versailles, and the Arc de Triomphe. The Eiffel Tower is one of the most famous landmarks
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the AI community. After decades of research, innovations, and collaborations, the technology has reached a point where it can perform tasks that would typically require human intelligence. From speech recognition to computer vision, to natural language understanding, these technologies have opened up a whole new world of possibilities. But how can you make sure that your organization is prepared to take advantage of these technological advancements? Well, there are several steps you can take to ensure your organization is on the cutting edge of AI. Here are some ideas:
    First, it’s important to invest in the right technology. If your organization is using a tool like TensorFlow, you


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been working here for [number of years] years. I'm passionate about [job title] and I love [job title] because [reason for passion]. I'm always looking for ways to [action or goal], and I'm always eager to learn and grow. I'm a [job title] at [company name], and I'm always looking for ways to [action or goal], and I'm always eager to learn and grow. I'm a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. Its status as the capital has made it a major hub for international diplomacy and commerce. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination, with millions of visitors annually. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability, as well as the potential for AI to be used for harmful purposes.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be
    


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
    Generated text:  ______ and I'm a young professional with a diverse range of interests and skills. I have a strong work ethic, a love for learning and always strive to grow and improve. I enjoy working in teams and am a strong advocate for diversity and inclusion in my work. I'm always eager to learn and to help others in my field. What's your background? What brings you to this position? What's the most important skill you bring to the table? What do you think makes you a great fit for the position? What's your passion for the job? What are some of the biggest challenges you've faced and how have you overcome them
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historic landmarks, vibrant arts scene, and rich cultural diversity.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased sophistication: As AI systems become more sophisticated, they will be able to perform increasingly complex tasks, including decision-making, learning, and problem-solving.
    
    2. Personalization: AI will become more personalized, enabling machines to learn and adapt to individual users, improving their usefulness and efficiency.
    
    3. Ubiquity: AI will become more ubiquitous, permeating every aspect of our lives, from transportation to entertainment to healthcare.
    
    4. Ethical considerations: AI will face ethical challenges, such as the use of AI to perpetuate or cause harm, leading to debates


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

    ],

     and

     I

     am

     a

     [

    job

     title

     or

     profession

    ]

     with

     a

     background

     in

     [

    relevant

     field

     or

     area

     of

     expertise

    ].

     I

     have

     a

     passion

     for

     [

    specific

     hobby

     or

     interest

    ],

     and

     my

     journey

     as

     a

     storyt

    eller

     has

     taken

     me

     on

     exciting

     and

     adventurous

     journeys

    ,

     from

     [

    previous

     experiences

    ]

     to

     [

    current

     projects

     or

     projects

     in

     progress

    ].

     I

     am

     constantly

     exploring

     new

     ideas

     and

     learning

     new

     things

    ,

     and

     I

     am

     looking

     forward

     to

     exploring

     new

     possibilities

     with

     you

    .

     What

    's

     your

     name

    ?

     What

    's

     your

     job

     title

     or

     profession

    ?

     What

    's

     your

     background

     in

     [

    relevant

     field

     or

     area

     of

     expertise

    ]?

     What

    's

     your

     passion

     for

     [

    specific

     hobby

     or

     interest

    ]?

     How

     have

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     largest

     metropolitan

     area

     in

     the

     European

     Union

     and

     the

     third

    -largest

     city

     in

     the

     world

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

     art

    ,

     cuisine

    ,

     and

     music

    .

     Paris

     is

     home

     to

     many

     notable

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

     known

     for

     its

     fashion

     and

     haute

     cuisine

    .

     Paris

     is

     a

     major

     tourist

     destination

     and

     a

     major

     economic

     power

    .

     According

     to

     a

     

    2

    0

    1

    9

     report

     by

     the

     United

     Nations

    ,

     the

     population

     of

     Paris

     is

     approximately

     

    2

    .

     

    4

     million

     people

    .

     
    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     renowned

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     exciting

     possibilities

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     may

     shape

     how

     we

     live

    ,

     work

    ,

     and

     interact

     with

     each

     other

    :
    


    1

    .

     Increased

     intelligence

     and

     adapt

    ability

    :

     With

     advancements

     in

     machine

     learning

    ,

     AI

     is

     becoming

     more

     capable

     of

     learning

     from

     data

     and

     adapting

     to

     new

     situations

    .

     This

     could

     lead

     to

     machines

     that

     can

     improve

     their

     performance

     over

     time

    ,

     as

     they

     become

     more

     familiar

     with

     the

     world

     they

    're

     in

    .
    


    2

    .

     Human

    -com

    puter

     interaction

    :

     As

     AI

     becomes

     more

     intelligent

    ,

     it

    's

     becoming

     easier

     and

     more

     natural

     for

     humans

     to

     interact

     with

     it

    .

     For

     example

    ,

     voice

     recognition

     technology

     could

     enable

     us

     to

     control

     our

     smart

     home

     appliances

     with

     our

     voice

    



```python
llm.shutdown()
```
