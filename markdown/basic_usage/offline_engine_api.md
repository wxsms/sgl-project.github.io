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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.53it/s]


    2026-05-20 14:22:25,219 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 14:22:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.80it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.80it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.80it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.80it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.18it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.18it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.18it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.18it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 11.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 15.07it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 15.07it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 20.51it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 25.31it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 25.31it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 25.31it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 25.31it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 25.31it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 25.31it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 30.20it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 35.51it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 35.51it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 35.51it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 35.51it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 35.51it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 35.51it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 41.85it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 41.85it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 41.85it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 41.85it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 41.85it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 41.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:05, 10.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:05, 10.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:05, 10.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:05, 10.27it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.86it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  21%|██        | 12/58 [00:01<00:03, 13.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:03, 13.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  21%|██        | 12/58 [00:01<00:03, 13.13it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.30it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  31%|███       | 18/58 [00:01<00:02, 15.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  31%|███       | 18/58 [00:01<00:02, 15.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  31%|███       | 18/58 [00:01<00:02, 15.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.20it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.20it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.20it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:01<00:02, 17.37it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:01<00:02, 17.37it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:01<00:02, 17.37it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  40%|███▉      | 23/58 [00:01<00:02, 17.37it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 17.26it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 17.26it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 17.26it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.69it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.69it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.69it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.69it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.83it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.83it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.83it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.83it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.79it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.79it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.79it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.79it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.85it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.85it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.85it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  69%|██████▉   | 40/58 [00:02<00:01, 13.85it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 16.04it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 16.04it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 16.04it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  74%|███████▍  | 43/58 [00:03<00:00, 16.04it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.64it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.64it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.64it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.64it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.64it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 25.24it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:03<00:00, 25.24it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:03<00:00, 25.24it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:03<00:00, 25.24it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:03<00:00, 21.91it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:03<00:00, 21.91it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:03<00:00, 16.47it/s]


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
    Generated text:  David and I'm a medical student in the UK. I have a master's degree in medical education and teaching and have been working as a medical teacher at secondary schools in the UK for the last three years. I recently started my studies in the Netherlands and have been studying in that country for the past 2 years. I have completed the Dutch language course and have also completed the Dutch University of Applied Sciences Master in Education (MED) program. I also have a Master in Teaching and Learning, which I will graduate soon. My goal is to be a teacher in the Netherlands and I am looking for a teacher to mentor me. 
    
    I
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking nominations for the position of Secretary of State. The Secretary of State is the official responsible for representing the United States in international affairs and negotiating treaties with other countries. Currently, there are 35 candidates running for the position.
    A) If the president of the United States is to be elected president, he or she must be in favor of the nomination.
    B) If the president of the United States is to be elected president, he or she must be qualified to be nominated for the position.
    C) If the president of the United States is to be elected president, he or she must be qualified to be nominated for the position.
    
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the______.
    A. middle of the continent
    B. northern part of the continent
    C. southern part of the continent
    D. eastern part of the continent
    答案: C
    
    对小张、小王、小李和小张所作的投票，恰好满足二元组的构成条件是：
    A. {小张，小王}，{小李，小张}，{小张，小李}
    B. {小张，小李}，{小王，小张}，{小王，小李}
    C. {小王，小李}，
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about the capabilities of AI. It is about the future of how we live our lives. AI is going to be everywhere and it will change the way we do things. In order to understand the future of AI, we need to focus on the questions that AI will address.
    One of the most important questions that AI will address is the ethics of AI. The ethical implications of AI can be complex and nuanced, and they will shape how we use AI in our daily lives. For example, the development of AI-powered autonomous vehicles has raised ethical concerns around the potential for self-driving cars to cause accidents and harm to humans.
    Another important


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character, such as "a friendly, outgoing, and knowledgeable person" or "a dedicated, hardworking, and creative individual"]. I enjoy [insert a short description of your hobbies or interests, such as "reading, cooking, and playing sports"]. I'm always looking for new challenges and opportunities to grow and learn. What are some of your favorite things to do? I love [insert a short description of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and culture. It is located on the Seine River and is the largest city in France by population. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also known for its vibrant nightlife, fashion industry, and annual festivals such as the World of Dance and the Paris Fashion Week. Paris is a city of contrasts, with its historic architecture, modern art, and diverse cultural scene. It is a major hub for business, education, and entertainment, and is a UNESCO World
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI will continue to be used for tasks that require human-like intelligence, such as language translation and emotional intelligence. As AI becomes more integrated into our daily lives, we may see a shift towards more ethical and responsible use of the technology. However, there are also potential risks and challenges associated with AI, such as the potential for job displacement and the need for careful regulation and oversight.
    


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
    Generated text:  Jane, and I'm a writer. I'm a busy writer with a passion for storytelling. I love being able to turn my ideas into words that can inspire and captivate my readers. And I love the thrill of writing in the style of a professional. What's your writing style like? I'm a lot of things to people who don't know me. I'm always talking. And I'm super proud of my work. It's a bit like the work of a true professional, but also a bit of a joke. Do you write for a living? No, not really. I write for fun, just for fun.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a sprawling metropolis with a rich history and culture. It has been a city of great importance throughout its long and varied history, serving as the seat of government and the spiritual home of the nation. It is also home to many of the world's most famous landmarks, including the Eiffel Tower, the Notre-Dame Cathedral, and the Louvre Museum. Paris is a lively and vibrant city with a rich tapestry of culture, cuisine, and art. Its air is crisp and clean, and its streets are lined with narrow cobblestone alleys and towering glass buildings. In short, Paris is a fascinating and awe
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with several trends shaping the way that we live, work, and interact with technology. Here are some of the most likely future trends in AI:
    
    1. Increased use of AI in healthcare: As more people become aware of the benefits of AI in healthcare, we can expect to see more use cases in personalized medicine, diagnosis, and treatment planning. This could lead to better patient outcomes and a more efficient healthcare system.
    
    2. Enhanced cognitive abilities: AI is becoming more capable of performing tasks that were previously considered beyond its capabilities. This could lead to the development of new AI technologies, such as superhuman AI, that are beyond


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

    ].

     I

     am

     a

     [

    Your

     Title

    ]

     at

     [

    Your

     Company

    ]

     and

     I

     have

     been

     with

     [

    Your

     Company

    ]

     for

     [

    X

     years

    /

    years

    ].

     I

     have

     always

     been

     passionate

     about

     [

    Your

     Area

     of

     Interest

    ].

     My

     background

     includes

     [

    X

     amount

     of

     years

     of

     experience

     in

     this

     area

    ].

     I

     am

     currently

     [

    X

     years

     old

    ]

     years

     old

    .

     I

     enjoy

     [

    X

     hobby

     or

     interest

     that

     you

     have

    ].

     I

     have

     a

     strong

     sense

     of

     [

    X

    ]

     and

     strive

     to

     be

     a

     [

    X

    ]

     person

    .

     I

     am

     a

     [

    X

     type

     of

     person

    ]:

     [

    X

    ].

     I

     am

     [

    X

    ].

     Thank

     you

    .

     I

     appreciate

     you

     taking

     the

     time

     to

     learn

     more

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historic

     and

     culturally

     rich

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     

    1

    2

    th

     century

    .

     The

     city

     has

     a

     population

     of

     over

     

    7

     million

     people

     and

     is

     the

     largest

     city

     in

     Europe

    .

     Paris

     is

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

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     and

     its

     numerous

     museums

    ,

     theaters

    ,

     and

     cafes

    .

     Paris

     is

     also

     home

     to

     numerous

     world

    -ren

    owned

     artistic

     and

     cultural

     institutions

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     the

     Centre

     Pom

    pid

    ou

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     has

     a

     strong

     emphasis

     on

     tourism

    ,

     with

     many

     famous

     landmarks

     attracting

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     is

     expected

     to

     continue

     to

     evolve

     rapidly

    .

     Here

     are

     some

     possible

     trends

     that

     may

     emerge

     in

     the

     coming

     years

    :
    


    1

    .

     AI

     will

     become

     more

     personalized

    :

     With

     more

     data

     and

     machine

     learning

     algorithms

    ,

     AI

     systems

     will

     become

     increasingly

     personalized

    .

     As

     AI

     systems

     are

     trained

     on

     personal

     data

    ,

     they

     will

     learn

     to

     anticipate

     and

     respond

     to

     specific

     needs

     and

     preferences

     of

     individual

     users

    .
    


    2

    .

     AI

     will

     be

     more

     accurate

     and

     less

     prone

     to

     human

     errors

    :

     AI

     will

     continue

     to

     become

     more

     accurate

     and

     less

     prone

     to

     human

     errors

    .

     Machine

     learning

     algorithms

     will

     be

     trained

     on

     large

     datasets

    ,

     which

     will

     enable

     them

     to

     learn

     from

     a

     wide

     range

     of

     examples

     and

     improve

     their

     performance

     over

     time

    .
    


    



```python
llm.shutdown()
```
