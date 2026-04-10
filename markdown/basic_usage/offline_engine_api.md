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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.18it/s]


    2026-04-10 01:27:11,695 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 01:27:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:10,  1.26s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:10,  1.26s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:03<01:10,  1.26s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:27,  1.95it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:27,  1.95it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:27,  1.95it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:15,  3.35it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.01it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.01it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.01it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.01it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.00it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.00it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.00it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.00it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 11.08it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 11.08it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 11.08it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 11.08it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 11.08it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 15.38it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 15.38it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 15.38it/s]

    Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 15.38it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 17.92it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 17.92it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 17.92it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 17.92it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 17.92it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 21.91it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 21.91it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 21.91it/s]

    Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 21.91it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 21.91it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 24.68it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 24.68it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 24.68it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 24.68it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 24.68it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:00, 27.65it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:00, 27.65it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:00, 27.65it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:00, 27.65it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:00, 27.65it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 28.38it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 28.38it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 28.38it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 28.38it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 28.38it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 29.54it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 29.54it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 29.54it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 29.54it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 29.54it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 29.98it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 29.98it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 29.98it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 29.98it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 29.98it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 30.73it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 30.73it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 30.73it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 30.73it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 30.73it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 32.45it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 32.45it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 32.45it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 32.45it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 32.45it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 32.45it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 36.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.92 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=53.92 GB):   2%|▏         | 1/58 [00:00<00:14,  4.06it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.89 GB):   2%|▏         | 1/58 [00:00<00:14,  4.06it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.89 GB):   3%|▎         | 2/58 [00:00<00:09,  5.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.89 GB):   3%|▎         | 2/58 [00:00<00:09,  5.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=53.89 GB):   5%|▌         | 3/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.88 GB):   5%|▌         | 3/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.88 GB):   7%|▋         | 4/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.89 GB):   7%|▋         | 4/58 [00:00<00:07,  7.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.89 GB):   9%|▊         | 5/58 [00:00<00:06,  7.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.88 GB):   9%|▊         | 5/58 [00:00<00:06,  7.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.88 GB):  10%|█         | 6/58 [00:00<00:06,  8.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.88 GB):  10%|█         | 6/58 [00:00<00:06,  8.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.88 GB):  10%|█         | 6/58 [00:00<00:06,  8.25it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.88 GB):  14%|█▍        | 8/58 [00:01<00:05,  9.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.88 GB):  14%|█▍        | 8/58 [00:01<00:05,  9.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.87 GB):  14%|█▍        | 8/58 [00:01<00:05,  9.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.87 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.87 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.85it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.87 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.87 GB):  21%|██        | 12/58 [00:01<00:04, 10.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.86 GB):  21%|██        | 12/58 [00:01<00:04, 10.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.86 GB):  21%|██        | 12/58 [00:01<00:04, 10.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.86 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.86 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.85 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.34it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=53.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.84 GB):  31%|███       | 18/58 [00:01<00:03, 11.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.84 GB):  31%|███       | 18/58 [00:01<00:03, 11.61it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.84 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.82 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.97it/s]Capturing num tokens (num_tokens=960 avail_mem=53.83 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.97it/s] Capturing num tokens (num_tokens=960 avail_mem=53.83 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.22it/s]Capturing num tokens (num_tokens=896 avail_mem=53.83 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.22it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.83 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.22it/s]Capturing num tokens (num_tokens=832 avail_mem=53.83 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.28it/s]Capturing num tokens (num_tokens=768 avail_mem=53.82 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.28it/s]Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.28it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.25it/s]Capturing num tokens (num_tokens=640 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.25it/s]Capturing num tokens (num_tokens=576 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.25it/s]Capturing num tokens (num_tokens=576 avail_mem=53.82 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.21it/s]Capturing num tokens (num_tokens=512 avail_mem=53.81 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.21it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.21it/s]Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.20it/s]Capturing num tokens (num_tokens=448 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.20it/s]Capturing num tokens (num_tokens=416 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.20it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.43it/s]Capturing num tokens (num_tokens=384 avail_mem=53.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 12.43it/s]Capturing num tokens (num_tokens=352 avail_mem=53.81 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.43it/s]Capturing num tokens (num_tokens=352 avail_mem=53.81 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.41it/s]Capturing num tokens (num_tokens=320 avail_mem=53.81 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.41it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.41it/s]Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.47it/s]Capturing num tokens (num_tokens=256 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.47it/s]Capturing num tokens (num_tokens=240 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.47it/s]

    Capturing num tokens (num_tokens=240 avail_mem=53.80 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.58it/s]Capturing num tokens (num_tokens=224 avail_mem=53.79 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.58it/s]Capturing num tokens (num_tokens=208 avail_mem=53.79 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.58it/s]Capturing num tokens (num_tokens=208 avail_mem=53.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.81it/s]Capturing num tokens (num_tokens=192 avail_mem=53.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.81it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.81it/s]Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.02it/s]Capturing num tokens (num_tokens=160 avail_mem=53.75 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.02it/s]Capturing num tokens (num_tokens=144 avail_mem=53.74 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.02it/s]Capturing num tokens (num_tokens=144 avail_mem=53.74 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.18it/s]Capturing num tokens (num_tokens=128 avail_mem=53.74 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.18it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.74 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.18it/s]Capturing num tokens (num_tokens=96 avail_mem=53.73 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.18it/s] Capturing num tokens (num_tokens=96 avail_mem=53.73 GB):  81%|████████  | 47/58 [00:04<00:00, 16.49it/s]Capturing num tokens (num_tokens=80 avail_mem=53.73 GB):  81%|████████  | 47/58 [00:04<00:00, 16.49it/s]Capturing num tokens (num_tokens=64 avail_mem=53.73 GB):  81%|████████  | 47/58 [00:04<00:00, 16.49it/s]Capturing num tokens (num_tokens=48 avail_mem=53.73 GB):  81%|████████  | 47/58 [00:04<00:00, 16.49it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.73 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.02it/s]Capturing num tokens (num_tokens=32 avail_mem=53.72 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.02it/s]Capturing num tokens (num_tokens=28 avail_mem=53.72 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.02it/s]Capturing num tokens (num_tokens=28 avail_mem=53.72 GB):  90%|████████▉ | 52/58 [00:04<00:00, 18.28it/s]Capturing num tokens (num_tokens=24 avail_mem=53.71 GB):  90%|████████▉ | 52/58 [00:04<00:00, 18.28it/s]Capturing num tokens (num_tokens=20 avail_mem=53.71 GB):  90%|████████▉ | 52/58 [00:04<00:00, 18.28it/s]

    Capturing num tokens (num_tokens=20 avail_mem=53.71 GB):  93%|█████████▎| 54/58 [00:04<00:00, 18.68it/s]Capturing num tokens (num_tokens=16 avail_mem=53.71 GB):  93%|█████████▎| 54/58 [00:04<00:00, 18.68it/s]Capturing num tokens (num_tokens=12 avail_mem=53.70 GB):  93%|█████████▎| 54/58 [00:04<00:00, 18.68it/s]Capturing num tokens (num_tokens=8 avail_mem=53.70 GB):  93%|█████████▎| 54/58 [00:04<00:00, 18.68it/s] Capturing num tokens (num_tokens=8 avail_mem=53.70 GB):  98%|█████████▊| 57/58 [00:04<00:00, 20.03it/s]Capturing num tokens (num_tokens=4 avail_mem=53.70 GB):  98%|█████████▊| 57/58 [00:04<00:00, 20.03it/s]Capturing num tokens (num_tokens=4 avail_mem=53.70 GB): 100%|██████████| 58/58 [00:04<00:00, 12.82it/s]


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
    Generated text:  Jana. I am currently a junior at the University of Illinois at Urbana-Champaign. I am a first year majoring in Computer Science and minoring in Statistics. My internship is currently at the University of Illinois at Urbana-Champaign. What kind of tasks will Jana be doing at the University of Illinois at Urbana-Champaign during her internship? Can you summarize the key points of Jana's internship with the University of Illinois at Urbana-Champaign? In a few sentences, please describe the tasks that Jana will be doing during her internship, including but not limited to her responsibilities and responsibilities, and please provide a
    ===============================
    Prompt: The president of the United States is
    Generated text:  in a middle school as a member of a committee. The committee has three levels of funding: $5000, $10000, and $20000. The president needs to raise enough money to cover the cost of a field trip. The total cost of the trip is $15000. The committee decides to form three teams, each consisting of an equal number of members. If the president wants to cover half of the total cost, how many members should be on each team to meet the committee's goal?
    To determine how many members should be on each team, we need to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. If we cut the capital of France into pieces, it is impossible to find a segment that connects all the pieces. Is the capital of France a circle or a sphere? To determine whether the capital of France, Paris, is a circle or a sphere, we need to analyze the geometric properties of the city of Paris.
    
    1. **Understanding the Problem:**
       - The problem states that if we cut the capital of France into pieces, it is impossible to find a segment that connects all the pieces. This implies that the city of Paris has a certain non-trivial topology.
       - We need to determine whether Paris is a
    ===============================
    Prompt: The future of AI is
    Generated text:  not in its potential, but in its failure. - Elon Musk
    
    # Machine Learning
    
    ## Overview
    
    Machine learning is the most widely used statistical method to generate models that can learn from data to make predictions. In its most fundamental sense, it is an artificial intelligence (AI) methodology that uses statistical models to learn patterns from data.
    
    ### What is Machine Learning?
    
    Machine learning is an artificial intelligence technique that uses statistical and computational models to make predictions. It is based on the idea that a large amount of data can reveal patterns and generalize them, which can be used to make predictions.
    
    ### Benefits of Machine Learning
    
    • Reduced failure rates
    


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


    Generated text:  [Name] and I am a [occupation] with [number] years of experience in [industry]. I am a [type of person] who is always [positive trait]. I am [character's name] and I am [character's age]. I am [character's profession] and I am [character's occupation]. I am [character's name] and I am [character's age]. I am [character's profession] and I am [character's occupation]. I am [character's name] and I am [character's age]. I am [character's profession] and I am [character's occupation]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of French colonialism and the influence of the French Revolution. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. It is the largest city in France and a major economic and political center in Europe. The city is home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be increased concerns about privacy and security. AI systems will need to be designed with privacy and security in mind, and there will be a need for robust privacy and security measures to protect user data.
    
    3. Increased focus on ethical
    


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
    Generated text:  [Name], and I am a [type of character] named [Name]. I come from [where you're from] and have always been [what you're known for, such as "bookworm", "outdoor enthusiast", etc.]. I currently reside in [city, state, country] and am currently working [job title]. Outside of work, I enjoy [time you spend with friends and family, hobbies, etc.]. I strive to be [what you're known for, such as "friendly", "enthusiastic", etc.]. I am always looking for ways to learn and grow as a person and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic and cultural city located in the center of the country. It is known for its towering Eiffel Tower and iconic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Palace of Versailles. The city is also famous for its food, art, and fashion, and is home to many famous landmarks and museums. Paris is a major cultural and economic hub in Europe and one of the world's most famous cities. It is home to the iconic Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among others. In addition to its historical and cultural significance, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by significant advancements in several key areas:
    
    1. Improved machine learning and deep learning: Continued advancements in machine learning and deep learning will enable AI to become more capable of understanding and manipulating complex data, leading to even greater productivity and efficiency.
    
    2. Increased use of AI in healthcare: AI will be used to improve diagnosis, treatment, and patient care in healthcare. AI will also be used to personalize treatment plans and to monitor patient health better.
    
    3. Automation of repetitive tasks: AI will continue to automate a wide range of tasks, freeing up time for human workers to focus on more complex and creative work.
    
    4. Increased


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

    ],

     and

     I

     am

     [

    Your

     Profession

    ],

     or

     maybe

     [

    Your

     Major

    ],

     or

     perhaps

     [

    Your

     Experience

    ].

     I

     am

     here

     today

     to

     share

     my

     experiences

     and

     knowledge

     in

     my

     field

    ,

     and

     I

     am

     eager

     to

     learn

     more

     about

     your

     experiences

     and

     interests

    .

     Please

     feel

     free

     to

     ask

     me

     any

     questions

     you

     may

     have

     or

     give

     me

     a

     brief

     overview

     of

     your

     own

     experiences

     and

     expertise

    .

     How

     about

     you

    ?

     What

     brings

     you

     to

     this

     place

     today

    ?

     What

     is

     your

     background

    ,

     and

     what

     makes

     you

     interested

     in

     [

    Your

     Field

     of

     Study

    ]

     or

     [

    Your

     Major

    ]

     or

     [

    Your

     Experience

    ]?

     I

     would

     love

     to

     learn

     more

     about

     you

     and

     your

     journey

     to

     this

     place

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     capital

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     land

     area

    ,

     and

     the

     largest

     city

     in

     Europe

     by

     population

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     cuisine

    .

     Paris

     is

     also

     a

     major

     economic

     and

     cultural

     hub

    ,

     with

     a

     large

     number

     of

     world

    -class

     museums

    ,

     theaters

    ,

     and

     shopping

     districts

    .

     It

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

     and

     a

     UNESCO

     World

     Heritage

     site

    .

     French

     cuisine

     and

     wine

     are

     also

     highly

     regarded

    ,

     and

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     city

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     likely

     to

     be

     characterized

     by

     a

     wide

     range

     of

     possibilities

    ,

     as

     the

     technology

     continues

     to

     evolve

     and

     improve

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

     Deep

     learning

    :

     As

     neural

     networks

     become

     more

     sophisticated

    ,

     they

     are

     becoming

     more

     powerful

     at

     processing

     and

     analyzing

     large

     amounts

     of

     data

    .

     This

     will

     lead

     to

     more

     advanced

     algorithms

     and

     models

     that

     can

     perform

     tasks

     that

     were

     previously

     considered

     in

    tract

    able

    .
    


    2

    .

     Quantum

     computing

    :

     The

     development

     of

     quantum

     computers

     is

     expected

     to

     revolution

    ize

     the

     field

     of

     AI

    .

     Quantum

     computers

     can

     perform

     calculations

     that

     are

     far

     faster

     than

     traditional

     computers

    ,

     which

     could

     lead

     to

     breakthrough

    s

     in

     areas

     such

     as

     natural

     language

     processing

     and

     machine

     learning

    .
    


    



```python
llm.shutdown()
```
