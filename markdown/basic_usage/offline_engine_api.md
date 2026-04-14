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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.45it/s]


    2026-04-14 04:59:57,390 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 04:59:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:41,  1.32it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:41,  1.32it/s]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:27,  1.99it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:27,  1.99it/s]

    Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:27,  1.99it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.51it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.51it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.51it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:06,  6.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:06,  6.96it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:06,  6.96it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:05,  8.85it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:05,  8.85it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:05,  8.85it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04, 10.64it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04, 10.64it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:04, 10.64it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:03, 12.54it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:03, 12.54it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:03, 12.54it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:03, 12.54it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:02, 15.23it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:02, 15.23it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:02, 15.23it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:02, 15.23it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 17.34it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 17.34it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 17.34it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 17.34it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 19.19it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 19.19it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 19.19it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 19.19it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 20.18it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 20.18it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 20.18it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 20.18it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 20.18it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 24.36it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 24.36it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 24.36it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 24.36it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 24.36it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 22.73it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 22.73it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 22.73it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 22.73it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 22.73it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 25.56it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 25.56it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 25.56it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 25.56it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 25.56it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 28.44it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 28.44it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 28.44it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 28.44it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 28.44it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 31.40it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 31.40it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:07,  8.01it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:07,  8.01it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:06,  8.48it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:05,  9.54it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.54it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.04it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.95it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.36it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.05it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.05it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.05it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.44it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.44it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.44it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.59it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.59it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.59it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.59it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.86it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.86it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.86it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 19.86it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.90it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.90it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.90it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.90it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.07it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.07it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.07it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.07it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.72it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.72it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.72it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 20.72it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  74%|███████▍  | 43/58 [00:02<00:00, 20.25it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 20.25it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 20.25it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 20.25it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:03<00:00, 19.77it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 19.77it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 19.77it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:03<00:00, 18.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 18.02it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 18.02it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 18.02it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 16.73it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 16.73it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 16.73it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:03<00:00, 15.84it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:03<00:00, 15.84it/s]

    Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 15.84it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 15.14it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 15.14it/s] Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 15.14it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:03<00:00, 13.33it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:03<00:00, 14.51it/s]


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
    Generated text:  John and I'm a 53 year old Cuban American. I can't stand to be surrounded by children and this is my 42nd year of marriage. I'm single and I'm getting married soon. What advice would you give me and what will I do? You can tell me what I should do. I am alone and have no one else to confide in. 
    
    I'm finding myself feeling very sad and almost broken inside. I have been trying to find a way to make my husband and me get along but I have no luck. The closest thing I have is a boyfriend. I'm always trying to be
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she helps the government of the United States run the country. If the president is not very happy, the country will become unhappy. Even if the president is not very important, he or she can still have an important part in the country. A president is like a leader of a team. The president can make decisions and take decisions. The president also makes sure that everyone in the country is happy. The president is the leader of the country. He or she helps the country run things. The president is also the leader of the country. He or she helps the country run things. The president is the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris  
    B) Bordeaux  
    C) Lyon  
    D) Nice
    To determine the capital of France, we need to recall the official capital of France. The capital of France is Lyon.
    
    The options given are:
    A) Paris  
    B) Bordeaux  
    C) Lyon  
    D) Nice
    
    Based on the information provided, the correct answer is:
    
    \boxed{C}
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s transforming every aspect of our lives. However, how we use and interact with AI can be complex and challenging. As a result, there are ethical concerns surrounding the use of AI. What does AI mean to you? Let’s explore the complex world of AI and how it affects our lives. Are you ready to dive into the world of AI? If so, then I will be here to help you understand the technical and ethical implications of AI.
    In this article, I will discuss the technical and ethical implications of AI. I will also provide some practical tips on how to use AI in a responsible and ethical manner.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about your personality or background here]. And what can you tell me about your work at [company name]? I'm a [insert a few details about your job responsibilities here]. And what can you tell me about your hobbies or interests? I'm a [insert a few details about your interests here]. And what can you tell me about your favorite [insert a few details about your hobbies or interests here]? I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and sciences. It is also home to many famous landmarks, including the Arc de Triomphe and the Champs-Élysées. Paris is a vibrant and diverse city with a rich cultural and artistic heritage. The city is also known for its cuisine,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI systems that can better understand and respond to the needs of individuals.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more integrated with human intelligence, it is likely to be used in even more advanced ways, such as developing more accurate and personalized medical treatments.
    
    3. Increased use of AI in automation: AI is
    


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
    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn about your background. How can I assist you today? [Your response should be neutral and free of offensive language. It should be brief and positive. It should not reveal any personal information. It should not use any character or brand names. It should be concise yet engaging. ]... (Keep the introduction brief and positive. Avoid using any language that could be interpreted as offensive. End with a strong, positive statement that captures your character's personality or skills.)... (Make sure your answer is not too personal and that
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Choose your answer: Does the question "What is the capital city of France? " have the same meaning as one of these?
    
    Pick from:
     a). different meanings
     b). the same meaning
    
    a). different meanings
    
    The question "What is the capital city of France? " and the answer "Paris" have different meanings. The question is asking for the capital city of France, while the answer is providing a specific city that is the capital of France. While both questions are about France, they are asking for different information. Therefore, the correct answer is a). different meanings. The first question is more general, while
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but it is certain that it will continue to evolve and develop in ways that are both exciting and challenging. Here are some potential trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient outcomes. As the technology continues to improve and become more accessible, it is likely that we will see a greater emphasis on AI in healthcare.
    
    2. Integration of AI into everyday life: As AI becomes more advanced and integrated into our daily lives, we will see an increasing number of apps and technologies that use AI to improve our lives. This


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

    ...

     [

    Name

    ],

     and

     I

    'm

     [

    Age

    ],

     [

    Occup

    ation

    ].


    It

    's

     a

     pleasure

     to

     meet

     you

    ,

     [

    Name

    ].

     I

    'm

     an

     experienced

     [

    Occup

    ation

    ]

     with

     a

     wealth

     of

     experience

     and

     a

     keen

     sense

     of

     curiosity

     that

     drives

     me

     to

     explore

     new

     challenges

     and

     learn

     from

     the

     best

    .

     I

    'm

     excited

     to

     share

     my

     knowledge

     and

     passion

     for

     my

     field

     with

     you

    .

     What

     can

     I

     help

     you

     with

     today

    ?

     [

    Name

    ].

     I

     hope

     we

     can

     make

     an

     interesting

     and

     productive

     conversation

    !

     [

    Name

    ].

     How

     do

     you

     think

     we

     can

     make

     our

     connection

     stronger

    ?

     [

    Name

    ].

     And

     how

     do

     you

     see

     the

     future

     of

     this

     field

    ?

     [

    Name

    ].

     I

    'm

     looking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

    's

     a

     concise

     factual

     statement

     about

     France

    's

     capital

     city

    :
    


    The

     capital

     of

     France

     is

     Paris

    .

     
    


    This

     statement

     encaps

    ulates

     the

     basic

     facts

     about

     the

     French

     capital

    ,

     including

     its

     name

    ,

     political

     role

    ,

     and

     historical

     significance

    .

     It

     provides

     a

     quick

     overview

     of

     what

     readers

     can

     expect

     to

     find

     when

     researching

     Paris

    .

     
    


    If

     you

     need

     more

     detailed

     information

     about

     Paris

    ,

     such

     as

     its

     landmarks

    ,

     notable

     people

    ,

     or

     tourist

     attractions

    ,

     you

     can

     include

     those

     details

     in

     a

     more

     comprehensive

     response

     to

     your

     question

    .

     However

    ,

     the

     basic

     statement

     about

     Paris

     being

     its

     capital

     remains

     the

     most

     straightforward

     and

     informative

     version

     of

     the

     capital

     city

    's

     name

    .

     
    


    Is

     there

     anything

     else

     you

    'd

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     three

     main

     trends

    :
    


    1

    .

     Increased

     AI

     ethics

     and

     accountability

    :

     AI

     systems

     are

     becoming

     more

     integrated

     into

     various

     industries

    ,

     and

     there

     is

     a

     growing

     demand

     for

     transparency

    ,

     accountability

    ,

     and

     ethical

     considerations

    .

     Organizations

     will

     be

     required

     to

     address

     the

     potential

     misuse

     of

     AI

    ,

     such

     as

     the

     creation

     of

     biased

     or

     harmful

     algorithms

    .

     This

     will

     require

     increased

     investment

     in

     AI

     ethics

     research

     and

     development

    .
    


    2

    .

     Improved

     AI

     performance

     and

     accuracy

    :

     AI

     is

     becoming

     more

     capable

     of

     performing

     tasks

     that

     were

     previously

     considered

     in

    feas

    ible

    .

     This

     includes

     tasks

     that

     require

     complex

     decision

    -making

    ,

     such

     as

     language

     translation

     or

     medical

     diagnosis

    .

     There

     is

     also

     a

     growing

     need

     for

     AI

     systems

     to

     be

    



```python
llm.shutdown()
```
