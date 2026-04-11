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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.05it/s]


    2026-04-11 02:41:13,467 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 02:41:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:50,  2.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:50,  2.99s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:08,  5.91it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:08,  5.91it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:08,  5.91it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:08,  5.91it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:08,  5.91it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04,  9.77it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04,  9.77it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04,  9.77it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04,  9.77it/s]

    Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04,  9.77it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:02, 14.06it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:02, 14.06it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:02, 14.06it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:02, 14.06it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 16.55it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 16.55it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 16.55it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 16.55it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 16.55it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 20.82it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 24.08it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 24.08it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 24.08it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 24.08it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 24.08it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:00, 26.95it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:00, 26.95it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:00, 26.95it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:00, 26.95it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:00, 26.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 29.28it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 30.56it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 30.56it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 30.56it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 30.56it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 30.56it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 32.26it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 32.26it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 34.62it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 34.62it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 34.62it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 34.62it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 34.62it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 35.52it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 35.52it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 39.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:28,  2.00it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:28,  2.00it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:15,  3.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:15,  3.58it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:11,  4.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:11,  4.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:09,  5.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:09,  5.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:07,  6.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:07,  6.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:01<00:07,  6.81it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:06,  8.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:06,  8.08it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:06,  8.08it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  21%|██        | 12/58 [00:02<00:08,  5.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  21%|██        | 12/58 [00:02<00:08,  5.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  21%|██        | 12/58 [00:02<00:08,  5.40it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.07it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  31%|███       | 18/58 [00:02<00:04,  9.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  31%|███       | 18/58 [00:02<00:04,  9.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  31%|███       | 18/58 [00:03<00:04,  9.18it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.10it/s]Capturing num tokens (num_tokens=960 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.10it/s] Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.10it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.96it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.96it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.34 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.96it/s]Capturing num tokens (num_tokens=768 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:03<00:02, 14.15it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:03<00:02, 14.15it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:03<00:02, 14.15it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:03<00:02, 14.15it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:03<00:01, 15.11it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:03<00:01, 15.11it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:03<00:01, 15.11it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.85it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.85it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.85it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.62it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.62it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.62it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.33it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.33it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:04<00:01, 14.33it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:04<00:01, 14.06it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:04<00:01, 14.06it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:04<00:01, 14.06it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:04<00:01, 13.85it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:04<00:01, 13.85it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:04<00:01, 13.85it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:04<00:01, 13.56it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:04<00:01, 13.56it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:04<00:01, 13.56it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.49it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.49it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.49it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 10.30it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 10.30it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 10.30it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.92it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.92it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.92it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:05<00:01,  9.81it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:05<00:01,  9.81it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:05<00:01,  9.81it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.78it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.78it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.78it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.47it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.47it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.47it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:05<00:00, 11.92it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:05<00:00, 11.92it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:05<00:00, 11.92it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.96it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.96it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.96it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:06<00:00, 13.07it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:06<00:00,  9.61it/s]


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
    Generated text:  Liza and I am 25 years old and I am a student at the University of Pennsylvania. I have been working for a few years now on my senior thesis. I have been working 40 hours a week on it and have been attending a small graduate school program.
    I'm a very serious person who really wants to learn and have a lot of work. I am not afraid of failure. I have a lot of information and can apply it in my thesis. I do my research and I have a lot of critical thinking skills.
    Do you think I can go to graduate school with my current schedule? If I go,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ____
    A. chief executive officer
    B. senator
    C. senator or vice president
    D. president
    Answer:
    A
    
    The civil servant who implements national and social public affairs, and performs administrative management functions in various fields of social economy and public welfare is called a(n) ____
    A. President
    B. Minister
    C. Clerk
    D. Civil Servant
    Answer:
    D
    
    Which of the following cannot be an administrative subject?
    A. Judicial organ
    B. Public institution
    C. Government agency
    D. Public institution
    E. State-owned enterprise
    Answer:
    A
    
    Patient, female, 4
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris  
    B: Berlin  
    C: Moscow  
    D: Rome
    
    To determine the capital of France, we need to consider the typical layout of major European cities. France, being a country, has a complex urban and administrative structure, and cities like Paris, Berlin, and Rome are significant but do not fit the general layout of a major European country.
    
    Let's analyze the options:
    
    A: Paris
    - Paris is one of the most famous and significant cities in Europe, known for its art, culture, and its role in shaping the French identity.
    
    B: Berlin
    - Berlin is the capital of Germany and is known
    ===============================
    Prompt: The future of AI is
    Generated text:  in action - what it will mean for you and your business
    
    The future of AI is in action - what it will mean for you and your business
    
    What will be the impact of the future of AI on your organization? We talk to the experts on the future of AI to help you understand the key drivers, pitfalls and potential impact of this innovative technology on your business. Learn more about the future of AI.
    
    The future of AI is in action – and what does it mean for your business? We spoke with the experts to see how the future of AI is transforming our world, and what it means for your business.
    
    AI in action


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] new things. I'm a [job title] and I'm always looking for ways to [job title] new things. I'm a [job title] and I'm always looking for ways to [job title] new things. I'm a [job title] and I'm always looking for ways to [job title] new things. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third largest in the world. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art. Paris is a cultural and historical center that attracts millions of visitors each year. It is a major transportation hub and a major economic center in Europe. The city is also home to many international organizations and institutions. Paris is a vibrant and dynamic city that continues to thrive and grow. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see more widespread use of AI in healthcare, including in areas such
    


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
    Generated text:  [Your Name], and I am a professional software developer with over [number] years of experience. I have a strong background in [specific technology or software] and have consistently excelled in my role due to my attention to detail, problem-solving skills, and ability to quickly adapt to new technologies. I have a passion for helping others, and I am constantly learning and growing within my field. I enjoy working in a team environment and am always looking for new challenges to overcome. I am a reliable, organized, and driven professional who is dedicated to delivering exceptional results on time and within budget. I am excited to work with [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is accurate, factual, and comprehensive. Paris, the city of light, is the capital of France, known for its historical, cultural, and artistic significance. It is a UNESCO World Heritage site and the birthplace of French literature, film, and architecture. The city is also home to the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a dynamic and vibrant metropolis with a rich history and a vibrant culture that continues to inspire and captivate visitors. Paris is a vibrant and dynamic metropolis with a rich history and a vibrant culture that continues to inspire and captivate visitors
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  predicted to be one of exponential growth. As technology continues to advance, we are likely to see more advanced AI systems and applications that will be even more sophisticated than what we have today. Here are some possible future trends in AI:
    
    1. Increased Personalization: As AI learns to understand and understand more about human behavior and preferences, we may see more personalized recommendations and experiences. This could include personalized recommendations for products and services, as well as personalized communication and support.
    
    2. Autonomous and Self-Driving Vehicles: Autonomous and self-driving vehicles could revolutionize transportation by reducing traffic congestion, improving safety, and increasing efficiency. AI systems could be used


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

    _,

     and

     I

     am

     a

    /an

     ______

    __

    _.

     I

     have

     always

     been

     a

    /an

     __

    ________

    _

     with

     a

    /an

     __

    ________

    _

     personality

    .

     I

     am

     __

    ________

    _,

     and

     I

     enjoy

     __

    ________

    _

     (

    fill

     in

     with

     any

     subject

     you

     like

    ).
    


    I

     hope

     you

     enjoy

     this

     character

     introduction

     and

     find

     it

     interesting

    .

     How

     can

     I

     assist

     you

     further

    ?

     Let

     me

     know

     if

     you

     have

     any

     questions

     or

     need

     further

     details

     on

     this

     character

    's

     background

    .

     I

     am

     here

     to

     help

    .

     Let

    's

     get

     started

    !

     

    😊

    👍

    ✨

    
    


    Did

     you

     have

     a

     good

     time

     helping

     me

    ?

     The

     character

     introduction

     was

     clear

     and

     concise

    ,

     and

     I

     appreciated

     the

     attention

     to

     detail.

     The

     personality

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     economic

     and

     cultural

     center

     in

     the

     country

    .
    


    Paris

     is

     known

     for

     its

     historical

     significance

    ,

     vibrant

     culture

    ,

     and

     iconic

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

     It

     also

     hosts

     several

     major

     events

     annually

    ,

     including

     the

     Winter

     Olympics

     and

     the

     Par

    aly

    mp

    ics

    ,

     making

     it

     a

     popular

     destination

     for

     tourists

     from

     around

     the

     world

    .

     Additionally

    ,

     Paris

     is

     known

     for

     its

     urban

     design

    ,

     architecture

    ,

     and

     cuisine

    ,

     which

     are

     highly

     regarded

     internationally

    .

     The

     city

    's

     reputation

     as

     one

     of

     the

     world

    's

     most

     vibrant

     and

     cosm

    opolitan

     places

     is

     well

    -d

    ocumented

    ,

     and

     it

     continues

     to

     attract

     visitors

     from

     all

     over

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     has

     the

     potential

     to

     transform

     nearly

     every

     aspect

     of

     society

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     precision

     and

     efficiency

    :

     AI

     is

     becoming

     more

     accurate

     and

     efficient

     in

     a

     wide

     range

     of

     applications

    ,

     from

     healthcare

     diagnosis

     to

     financial

     fraud

     detection

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     enable

     more

     accurate

     and

     personalized

     recommendations

     and

     experiences

     for

     individuals

    ,

     from

     personalized

     news

     feeds

     to

     personalized

     product

     recommendations

    .
    


    3

    .

     Autonomous

     vehicles

    :

     The

     future

     of

     AI

     will

     likely

     see

     the

     widespread

     adoption

     of

     autonomous

     vehicles

    ,

     which

     will

     be

     able

     to

     safely

     navigate

     roads

    ,

     avoid

     collisions

     and

     handle

     various

     tasks

     autonom

    ously

    .
    


    4

    .

     Improved

     quality

     and

     accuracy

    :

     AI

     will

     continue

     to

    



```python
llm.shutdown()
```
