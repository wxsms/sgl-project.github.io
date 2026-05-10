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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.91it/s]


    2026-05-10 04:34:08,414 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 04:34:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:44,  3.94s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.74it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.74it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.74it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  4.07it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  4.07it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:12,  4.07it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:12,  4.07it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.91it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.91it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.91it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 12.91it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 12.91it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.84it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.84it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.84it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.84it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.84it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.56it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.56it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.56it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.56it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.56it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.16it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.16it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.16it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.16it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.16it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.16it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 29.09it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 29.09it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 29.09it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 29.09it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 29.09it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 29.09it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 36.63it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 36.63it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 36.63it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 36.63it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 36.63it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 36.63it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 39.61it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 44.98it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 44.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=49.69 GB):   2%|▏         | 1/58 [00:00<00:12,  4.62it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.12 GB):   2%|▏         | 1/58 [00:00<00:12,  4.62it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.12 GB):   3%|▎         | 2/58 [00:00<00:11,  4.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.69 GB):   3%|▎         | 2/58 [00:00<00:11,  4.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.69 GB):   5%|▌         | 3/58 [00:00<00:10,  5.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.72 GB):   5%|▌         | 3/58 [00:00<00:10,  5.33it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=49.72 GB):   7%|▋         | 4/58 [00:00<00:09,  5.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.11 GB):   7%|▋         | 4/58 [00:00<00:09,  5.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.11 GB):   9%|▊         | 5/58 [00:00<00:09,  5.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.11 GB):   9%|▊         | 5/58 [00:00<00:09,  5.55it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=50.11 GB):  10%|█         | 6/58 [00:01<00:08,  5.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.75 GB):  10%|█         | 6/58 [00:01<00:08,  5.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.75 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.09 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.27it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=50.09 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.09 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.09 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.08 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.79it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=50.08 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.83 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.83 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.07 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.55it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=50.07 GB):  21%|██        | 12/58 [00:01<00:05,  7.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.07 GB):  21%|██        | 12/58 [00:01<00:05,  7.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.07 GB):  22%|██▏       | 13/58 [00:01<00:05,  7.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=49.85 GB):  22%|██▏       | 13/58 [00:01<00:05,  7.99it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=50.06 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.06 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.04 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.75it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=50.04 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.04 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.03 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.03 GB):  31%|███       | 18/58 [00:02<00:03, 10.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.87 GB):  31%|███       | 18/58 [00:02<00:03, 10.31it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=50.03 GB):  31%|███       | 18/58 [00:02<00:03, 10.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.03 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.92 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.19it/s]Capturing num tokens (num_tokens=960 avail_mem=50.00 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.19it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=50.00 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=896 avail_mem=50.00 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=832 avail_mem=49.99 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.06it/s]Capturing num tokens (num_tokens=832 avail_mem=49.99 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.64it/s]Capturing num tokens (num_tokens=768 avail_mem=49.99 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.64it/s]

    Capturing num tokens (num_tokens=704 avail_mem=49.94 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.64it/s]Capturing num tokens (num_tokens=704 avail_mem=49.94 GB):  45%|████▍     | 26/58 [00:03<00:02, 12.87it/s]Capturing num tokens (num_tokens=640 avail_mem=49.95 GB):  45%|████▍     | 26/58 [00:03<00:02, 12.87it/s]Capturing num tokens (num_tokens=576 avail_mem=49.92 GB):  45%|████▍     | 26/58 [00:03<00:02, 12.87it/s]

    Capturing num tokens (num_tokens=576 avail_mem=49.92 GB):  48%|████▊     | 28/58 [00:03<00:02, 12.94it/s]Capturing num tokens (num_tokens=512 avail_mem=49.95 GB):  48%|████▊     | 28/58 [00:03<00:02, 12.94it/s]Capturing num tokens (num_tokens=480 avail_mem=49.94 GB):  48%|████▊     | 28/58 [00:03<00:02, 12.94it/s]Capturing num tokens (num_tokens=480 avail_mem=49.94 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.05it/s]Capturing num tokens (num_tokens=448 avail_mem=49.95 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.05it/s]Capturing num tokens (num_tokens=416 avail_mem=49.96 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.05it/s]

    Capturing num tokens (num_tokens=416 avail_mem=49.96 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.30it/s]Capturing num tokens (num_tokens=384 avail_mem=49.95 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.30it/s]Capturing num tokens (num_tokens=352 avail_mem=49.95 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.30it/s]Capturing num tokens (num_tokens=352 avail_mem=49.95 GB):  59%|█████▊    | 34/58 [00:03<00:01, 15.24it/s]Capturing num tokens (num_tokens=320 avail_mem=49.94 GB):  59%|█████▊    | 34/58 [00:03<00:01, 15.24it/s]Capturing num tokens (num_tokens=288 avail_mem=49.93 GB):  59%|█████▊    | 34/58 [00:03<00:01, 15.24it/s]

    Capturing num tokens (num_tokens=288 avail_mem=49.93 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=256 avail_mem=49.92 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=240 avail_mem=49.92 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=224 avail_mem=49.91 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=224 avail_mem=49.91 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.88it/s]Capturing num tokens (num_tokens=208 avail_mem=49.90 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.88it/s]Capturing num tokens (num_tokens=192 avail_mem=49.90 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.88it/s]

    Capturing num tokens (num_tokens=176 avail_mem=49.89 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.88it/s]Capturing num tokens (num_tokens=176 avail_mem=49.89 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=160 avail_mem=49.90 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=144 avail_mem=49.89 GB):  72%|███████▏  | 42/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=128 avail_mem=49.89 GB):  72%|███████▏  | 42/58 [00:04<00:00, 18.87it/s]Capturing num tokens (num_tokens=128 avail_mem=49.89 GB):  78%|███████▊  | 45/58 [00:04<00:00, 19.68it/s]Capturing num tokens (num_tokens=112 avail_mem=49.86 GB):  78%|███████▊  | 45/58 [00:04<00:00, 19.68it/s]

    Capturing num tokens (num_tokens=96 avail_mem=49.87 GB):  78%|███████▊  | 45/58 [00:04<00:00, 19.68it/s] Capturing num tokens (num_tokens=80 avail_mem=49.87 GB):  78%|███████▊  | 45/58 [00:04<00:00, 19.68it/s]Capturing num tokens (num_tokens=80 avail_mem=49.87 GB):  83%|████████▎ | 48/58 [00:04<00:00, 20.40it/s]Capturing num tokens (num_tokens=64 avail_mem=49.86 GB):  83%|████████▎ | 48/58 [00:04<00:00, 20.40it/s]Capturing num tokens (num_tokens=48 avail_mem=49.85 GB):  83%|████████▎ | 48/58 [00:04<00:00, 20.40it/s]Capturing num tokens (num_tokens=32 avail_mem=49.85 GB):  83%|████████▎ | 48/58 [00:04<00:00, 20.40it/s]

    Capturing num tokens (num_tokens=32 avail_mem=49.85 GB):  88%|████████▊ | 51/58 [00:04<00:00, 20.54it/s]Capturing num tokens (num_tokens=28 avail_mem=49.84 GB):  88%|████████▊ | 51/58 [00:04<00:00, 20.54it/s]Capturing num tokens (num_tokens=24 avail_mem=49.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 20.54it/s]Capturing num tokens (num_tokens=20 avail_mem=49.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 20.54it/s]Capturing num tokens (num_tokens=20 avail_mem=49.82 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.98it/s]Capturing num tokens (num_tokens=16 avail_mem=49.82 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.98it/s]Capturing num tokens (num_tokens=12 avail_mem=49.81 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.98it/s]

    Capturing num tokens (num_tokens=8 avail_mem=49.81 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.98it/s] Capturing num tokens (num_tokens=8 avail_mem=49.81 GB):  98%|█████████▊| 57/58 [00:04<00:00, 21.28it/s]Capturing num tokens (num_tokens=4 avail_mem=49.80 GB):  98%|█████████▊| 57/58 [00:04<00:00, 21.28it/s]Capturing num tokens (num_tokens=4 avail_mem=49.80 GB): 100%|██████████| 58/58 [00:04<00:00, 12.44it/s]


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
    Generated text:  Alex and I’m a 24 year old with no major life issues. I have been in a relationship for 3 years now and am feeling like my partner is no longer the same. They were always very supportive and loyal to me, but now they seem to be a little more aloof and distant. I have been going to the doctor for the past 4 weeks and have had tests done, but nothing has come back positive for anything. How can I get my partner to be more attentive to me now?
    
    It sounds like you are experiencing a significant change in your relationship, and it's natural to feel overwhelmed by feelings of
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    The president of the United States is from the United States of America. The president of the United States is elected to a four-year term by the American people and is the head of state and head of government of the country. The country is a federal republic, with the president serving as both the head of state and the head of government. The country is situated in North America and borders Canada to the north and Mexico to the south. The country is known for its rich history, culture, and natural beauty, and is the world's third most populous country after China and India. The president is also a member of the
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. London
    D. New York
    Answer: A
    
    The National Cultural Heritage Protection and Management Agency was established in ___.
    A. 1998
    B. 1999
    C. 2000
    D. 2001
    Answer: C
    
    Which of the following is NOT a correct description of the purposes of the Fourth Plenary Session of the 18th CPC Central Committee? A. To uphold the spirit of the 19th National Congress of the Communist Party of China B. To comprehensively advance the rule
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about changing how we interact with machines. It's about how we interact with each other. This is the theme of the AI@UWS course, which is part of the Unit for Digital Futures, and is led by the University of Waikato Faculty of Computing. It aims to reimagine the future of AI for all New Zealanders. The course aims to provide an overview of the current state of AI and what the future might hold. The course will give the students the skills to work with people and the knowledge needed to understand the implications of these technologies.
    The course is taking place from 20 July to 


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Hobby] and I enjoy [What I Do for Fun]. I'm always ready to learn and adapt to new situations. I'm a [What I Do for Fun] and I love [What I Do for Fun]. I'm a [What I Do for Fun] and I love [What I Do for Fun
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also home to the French Parliament, the French National Museum of Modern Art, and the Eiffel Tower. The city is known for its beautiful architecture, vibrant nightlife, and delicious cuisine. Paris is a city of contrasts, with its rich history and modernity. It is a city that has been a hub of French culture and politics
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI that can learn from and adapt to human behavior and decision-making.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis, treatment, and patient care.
    
    3. Greater
    


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
    Generated text:  [name], and I'm a [general description of your character]. My interests are [list of interests]. I have [number of hobbies] and my favorite food is [name of dish]. I enjoy [list of activities/activities]. I have a [number of hobbies] and my favorite food is [name of dish]. I enjoy [list of activities/activities]. I am [age], and I love [interests that interest you]. I have always been curious about [interests], so I am always looking for new adventures and experiences. How can you tell if I am interested in a particular topic?
    
    I am a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is a major European city and the largest metropolitan area in terms of population. It is the financial capital of the world and the home of the European Parliament, the European Court of Justice, the European Central Bank, the International Bank for Reconstruction and Development, and the Eiffel Tower. Its population is estimated at over 2. 1 million people, and it is the world’s 15th most populous city. Paris is home to the French Parliament and is a cultural and political center of France. It is also the birthplace of many French leaders and presidents, including François Mitterand, Georges Pompidou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and evolving. Here are some of the potential trends that are expected to shape this field in the next decade:
    
    1. Increased Integration with Human Intelligence: As AI technology continues to advance, we can expect to see a more seamless integration between AI and human intelligence. This could mean a future where AI systems can learn and adapt to human behavior and preferences, further enhancing their ability to provide personalized and adaptive services.
    
    2. Increased Transparency and Explainability: AI systems are becoming more sophisticated, and they are becoming capable of understanding and explaining their decisions. This level of transparency and explainability could help reduce the potential for bias and improve trust in


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

    'm

     a

     software

     developer

    .

     I

    'm

     always

     excited

     to

     learn

     new

     things

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     tech

     trends

    .

     I

     enjoy

     helping

     people

     solve

     problems

     and

     creating

     solutions

     to

     complex

     problems

    .

     I

    'm

     a

     great

     communicator

     and

     enjoy

     working

     with

     others

     to

     achieve

     our

     goals

    .

     I

     believe

     that

     technology

     and

     innovation

     should

     be

     celebrated

     and

     supported

    .

     
    


    I

    'm

     a

     big

     fan

     of

     the

     [

    Company

    ]

     brand

     and

     I

     enjoy

     following

     them

     on

     social

     media

    .

     I

    'm

     always

     eager

     to

     see

     what

     they

    're

     working

     on

     and

     how

     they

    're

     making

     technology

     better

    .

     I

    'd

     love

     to

     work

     with

     any

     team

     that

     is

     passionate

     about

     technology

     and

     values

     innovation

    .

     If

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     historic

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     Located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    ,

     it

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     It

     is

     also

     known

     as

     "

    La

     Par

    ure

     du

     monde

    "

     (

    The

     World

    's

     Frame

    ),

     due

     to

     its

     cultural

     and

     architectural

     richness

    .

     Paris

     is

     a

     major

     tourist

     destination

     and

     a

     UNESCO

     World

     Heritage

     site

    ,

     and

     is

     often

     considered

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    .

     The

     French

     capital

     is

     a

     melting

     pot

     of

     cultures

     and

     influences

    ,

     with

     influences

     from

     various

     European

     and

     African

     traditions

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     many

     exciting

     developments

     that

     will

     continue

     to

     shape

     the

     way

     we

     interact

     with

     technology

     and

     communicate

     with

     each

     other

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     AI

     transparency

     and

     explain

    ability

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     we

     may

     expect

     to

     see

     more

     transparency

     and

     explain

    ability

     in

     how

     they

     work

    .

     This

     could

     involve

     making

     it

     easier

     for

     people

     to

     understand

     how

     an

     AI

     system

     is

     making

     a

     decision

    ,

     or

     how

     it

     arrived

     at

     a

     particular

     conclusion

    .
    


    2

    .

     AI

     ethics

     and

     bias

    :

     With

     the

     rise

     of

     AI

     in

     areas

     such

     as

     self

    -driving

     cars

    ,

     financial

     fraud

     detection

    ,

     and

     image

     recognition

    ,

     there

     is

     a

     growing

     concern

     about

     the

     potential

     for

     AI

    



```python
llm.shutdown()
```
