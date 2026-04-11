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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.73it/s]


    2026-04-11 11:17:07,111 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 11:17:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.22it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.82it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.82it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.82it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.82it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.57it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.57it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.57it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.57it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.67it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.67it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:02, 14.04it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:02, 14.04it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:02, 14.04it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:02, 14.04it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 21.29it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 21.29it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 21.29it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 21.29it/s]

    Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 21.29it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.22it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.22it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.22it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.22it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.22it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 26.52it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 26.52it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 26.52it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 26.52it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 26.52it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 28.66it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 28.66it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 28.66it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 28.66it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 28.66it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 31.39it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 31.39it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 31.39it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 31.39it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 31.39it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 32.07it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 32.07it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 32.07it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 32.07it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 32.07it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 33.24it/s] 

    Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 37.24it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 37.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:07,  7.13it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:07,  7.13it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:06,  7.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:06,  7.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.04it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.64it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.24it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  21%|██        | 12/58 [00:01<00:05,  8.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  21%|██        | 12/58 [00:01<00:05,  8.98it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  21%|██        | 12/58 [00:01<00:05,  8.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.81it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.38 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  31%|███       | 18/58 [00:01<00:03, 10.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  31%|███       | 18/58 [00:01<00:03, 10.68it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  31%|███       | 18/58 [00:02<00:03, 10.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.07it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.07it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.56it/s]Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.56it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.56it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.82it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.82it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.82it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.03it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.03it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.03it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.18it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.18it/s]Capturing num tokens (num_tokens=480 avail_mem=58.35 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.18it/s]Capturing num tokens (num_tokens=480 avail_mem=58.35 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.37it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.37it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.37it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.41it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.41it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.41it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.37it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.37it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.37it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.38it/s]Capturing num tokens (num_tokens=256 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.38it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.38it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.47it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.47it/s]Capturing num tokens (num_tokens=208 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.47it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.55it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.55it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.65it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.65it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.65it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.65it/s]Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.65it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.65it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.62it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.62it/s]Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.62it/s]Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  86%|████████▌ | 50/58 [00:04<00:00, 11.46it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 11.46it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 11.46it/s]Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  90%|████████▉ | 52/58 [00:04<00:00, 11.08it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 11.08it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 11.08it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:04<00:00, 11.44it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:04<00:00, 11.44it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:04<00:00, 11.44it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.59it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.59it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 11.59it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:05<00:00, 11.97it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:05<00:00, 11.10it/s]


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
    Generated text:  Gail, and I'm a psychology student. I've been studying the impact of technology on mental health. Could you tell me more about the impact of technology on communication in the workplace? Yes, technology has had a significant impact on the way we communicate in the workplace. The use of email, instant messaging, and video conferencing has made it easier for employees to communicate with each other and collaborate on projects. These technologies have also made it easier for companies to communicate with their employees and clients remotely, and they have also enabled the sharing of information and updates between teams. However, these technologies have also created new challenges, such as the
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what gift to give to his wife. He has two choices: a $120 present for him or a $150 present for her. He decides to give the present to her for a gift that he does not know the cost of. He gives her the present for $150 because she is currently 12 years old. She knows that the present cost is a multiple of 5. How many years does she have to wait before the present will cost $150?
    
    The $150 present is for her to get a gift that she knows the cost is a multiple of 
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the **Billetterie** bridge, which is a **bridge** in a **bridge**, which is the **bridge** in a **truss** structure, which is the **truss** structure in a **fortified** country, which is a **fortified** country in a **river** structure, which is a **river** structure in a **fortified** country, which is a **fortified** country in a **truss** structure, which is a **truss** structure in a **fortified** country, which is a **fortified** country in a **bridge**, which
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, and as businesses and governments take steps to embrace it, they need to be aware of the risks that come with it. In this article, we’ll discuss the potential risks of AI, from privacy concerns to data bias, and provide some tips on how to mitigate these risks.
    Privacy concerns are a major concern for businesses and governments alike. AI can be used to analyze large amounts of data, which can help identify patterns and make predictions. However, this can also lead to the collection and use of sensitive information, such as personal data and financial information.
    One of the biggest risks of AI is the potential for bias in the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm [age] years old, and I have [number] years of experience in [industry]. I'm a [job title] with [company name] and I'm always looking for ways to [describe a challenge or opportunity]. I'm [job title] at [company name], and I'm always looking for ways to [describe a challenge or opportunity]. I'm [job title] at [company name], and I'm always looking for ways
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for fashion, art, and music. The city is known for its annual festivals and events, including the Eiffel Tower Parade and the World Cup of Flamenco. Paris is a popular tourist destination and a major economic hub in Europe. It is the seat of the French government and the headquarters of many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies such as blockchain, IoT, and quantum computing, which will enable new applications and innovations.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be increased concerns about privacy and security. There will be efforts to develop new technologies and protocols that will help to protect user data and prevent cyber attacks.
    
    3. Greater focus on ethical considerations: As AI becomes more integrated with other
    


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
    Generated text:  [Name] and I am a [Role] for [Company/Team/Team]! I bring a lot of [ability or personality trait] to my work, and am always looking for ways to [do something]. I enjoy [what makes me happy] and am always looking for ways to [what makes me happy] with [Company/Team/Team]. If you have any questions or if you want to discuss the opportunities for growth in this role, please don't hesitate to reach out. I'm here to learn and grow every day. Can you tell me more about yourself, what skills and qualities you bring to your role
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a city renowned for its stunning architecture, vibrant culture, and rich history. It is the largest city in France and the sixth-largest city in the world, with a population of over 2.5 million people. Paris is known for its beautiful museums, theaters, and cafes, as well as its iconic landmarks such as the Eiffel Tower and Louvre Museum. The city is home to several world-renowned universities, including the University of Paris-Sorbonne, and is a hub for art, science, and literature. Paris is also known for its role in the French Revolution and is considered a cultural center of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a mix of advances in depth and breadth, with the following potential future trends:
    
    1. AI will continue to advance in terms of accuracy, speed, and scale. Advances in machine learning algorithms, deep learning models, and the integration of AI with other technologies will likely lead to even greater capabilities.
    
    2. AI will become more ubiquitous, with more applications and devices being able to interact with it in real-time. This will likely lead to more widespread adoption of AI in various industries, from healthcare to transportation to manufacturing.
    
    3. AI will continue to be integrated into our daily lives, with more people using AI-powered tools and services


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

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

    'm

     a

     friendly

     and

     outgoing

     person

     who

     enjoys

     laughing

    ,

     playing

     games

     and

     connecting

     with

     people

     in

     a

     friendly

     way

    .

     I

     have

     a

     passion

     for

     learning

     new

     things

     and

     always

     strive

     to

     grow

     and

     improve

     myself

    .

     I

     believe

     that

     everyone

     has

     the

     potential

     to

     grow

     and

     succeed

    ,

     and

     I

    'm

     passionate

     about

     helping

     others

     achieve

     their

     goals

    .

     I

     enjoy

     working

     with

     people

     and

     learning

     from

     them

    ,

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

     and

     advance

    .

     If

     you

    're

     looking

     for

     someone

     who

     can

     inspire

    ,

     challenge

    ,

     and

     motivate

     you

     to

     reach

     your

     full

     potential

    ,

     I

    'm

     the

     one

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     known

     for

     its

     rich

     cultural

     heritage

     and

     a

     diverse

     population

     of

     around

     

    1

    5

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

     a

     major

     met

    ropolis

     and

     an

     international

     center

     of

     government

    ,

     business

    ,

     and

     culture

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     with

     many

     famous

     landmarks

     and

     museums

     to

     visit

    ,

     including

     the

     Lou

    vre

    ,

     the

     Sor

    bon

    ne

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     
    


    Paris

     is

     also

     known

     for

     its

     distinctive

     fashion

     scene

    ,

     with

     many

     high

    -end

     fashion

     designers

     and

     bout

    iques

     to

     explore

    .

     The

     city

     is

     home

     to

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     iterative

     and

     dynamic

    ,

     driven

     by

     several

     key

     trends

    :
    


    1

    .

     **

    Art

    ificial

     Intelligence

     (

    AI

    )

     will

     become

     more

     efficient

     and

     cost

    -effective

    **

     -

     As

     AI

     technology

     evolves

    ,

     the

     cost

     of

     developing

     and

     maintaining

     AI

     systems

     is

     expected

     to

     decrease

    .

     This

     will

     make

     AI

     systems

     more

     accessible

     and

     affordable

     for

     businesses

     and

     individuals

    .
    


    2

    .

     **

    AI

     will

     become

     more

     intelligent

     and

     adaptable

    **

     -

     AI

     is

     likely

     to

     become

     even

     more

     intelligent

     as

     it

     learns

     from

     experience

     and

     can

     adapt

     to

     new

     situations

    .

     AI

     models

     are

     also

     becoming

     more

     capable

     of

     handling

     complex

     problems

    .
    


    3

    .

     **

    AI

     will

     be

     integrated

     into

     all

     aspects

     of

     life

    **

     -

     As

     AI

     technology

     becomes

     more

     advanced

    



```python
llm.shutdown()
```
