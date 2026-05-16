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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.97it/s]


    2026-05-16 00:21:21,321 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 00:21:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.69it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.69it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.11it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.11it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.11it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.11it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.11it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.91it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.91it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.91it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.91it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.91it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.81it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.81it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 18.47it/s]

    Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 25.01it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 27.59it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 27.59it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 27.59it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 27.59it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 27.59it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.57it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.57it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 27.57it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 27.57it/s]

    Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 27.57it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 28.45it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 28.45it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 28.45it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 28.45it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 28.45it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 30.27it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 30.27it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 30.27it/s]

    Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 30.27it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 30.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   2%|▏         | 1/58 [00:00<00:09,  5.80it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   2%|▏         | 1/58 [00:00<00:09,  5.80it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   3%|▎         | 2/58 [00:00<00:09,  5.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.91 GB):   3%|▎         | 2/58 [00:00<00:09,  5.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.91 GB):   5%|▌         | 3/58 [00:00<00:08,  6.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:08,  6.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:07,  6.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:07,  6.69it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.90 GB):  10%|█         | 6/58 [00:00<00:07,  7.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):  10%|█         | 6/58 [00:00<00:07,  7.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.12it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.14it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.32 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.32 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.83 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.60it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=52.83 GB):  26%|██▌       | 15/58 [00:01<00:05,  8.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.36 GB):  26%|██▌       | 15/58 [00:01<00:05,  8.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.36 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.82 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.07it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=52.82 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.82 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.82 GB):  31%|███       | 18/58 [00:02<00:05,  7.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.41 GB):  31%|███       | 18/58 [00:02<00:05,  7.91it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=52.41 GB):  33%|███▎      | 19/58 [00:02<00:05,  7.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.81 GB):  33%|███▎      | 19/58 [00:02<00:05,  7.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.81 GB):  34%|███▍      | 20/58 [00:02<00:04,  7.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.42 GB):  34%|███▍      | 20/58 [00:02<00:04,  7.87it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=52.42 GB):  36%|███▌      | 21/58 [00:02<00:04,  7.99it/s]Capturing num tokens (num_tokens=960 avail_mem=52.80 GB):  36%|███▌      | 21/58 [00:02<00:04,  7.99it/s] Capturing num tokens (num_tokens=960 avail_mem=52.80 GB):  38%|███▊      | 22/58 [00:02<00:04,  7.89it/s]Capturing num tokens (num_tokens=896 avail_mem=52.80 GB):  38%|███▊      | 22/58 [00:02<00:04,  7.89it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.79 GB):  38%|███▊      | 22/58 [00:02<00:04,  7.89it/s]Capturing num tokens (num_tokens=832 avail_mem=52.79 GB):  41%|████▏     | 24/58 [00:03<00:04,  8.36it/s]Capturing num tokens (num_tokens=768 avail_mem=52.79 GB):  41%|████▏     | 24/58 [00:03<00:04,  8.36it/s]

    Capturing num tokens (num_tokens=768 avail_mem=52.79 GB):  43%|████▎     | 25/58 [00:03<00:03,  8.47it/s]Capturing num tokens (num_tokens=704 avail_mem=52.51 GB):  43%|████▎     | 25/58 [00:03<00:03,  8.47it/s]Capturing num tokens (num_tokens=640 avail_mem=52.78 GB):  43%|████▎     | 25/58 [00:03<00:03,  8.47it/s]

    Capturing num tokens (num_tokens=640 avail_mem=52.78 GB):  47%|████▋     | 27/58 [00:03<00:03,  8.90it/s]Capturing num tokens (num_tokens=576 avail_mem=52.78 GB):  47%|████▋     | 27/58 [00:03<00:03,  8.90it/s]Capturing num tokens (num_tokens=576 avail_mem=52.78 GB):  48%|████▊     | 28/58 [00:03<00:03,  8.91it/s]Capturing num tokens (num_tokens=512 avail_mem=52.53 GB):  48%|████▊     | 28/58 [00:03<00:03,  8.91it/s]

    Capturing num tokens (num_tokens=480 avail_mem=52.77 GB):  48%|████▊     | 28/58 [00:03<00:03,  8.91it/s]Capturing num tokens (num_tokens=480 avail_mem=52.77 GB):  52%|█████▏    | 30/58 [00:03<00:03,  9.33it/s]Capturing num tokens (num_tokens=448 avail_mem=52.77 GB):  52%|█████▏    | 30/58 [00:03<00:03,  9.33it/s]

    Capturing num tokens (num_tokens=448 avail_mem=52.77 GB):  53%|█████▎    | 31/58 [00:03<00:02,  9.31it/s]Capturing num tokens (num_tokens=416 avail_mem=52.76 GB):  53%|█████▎    | 31/58 [00:03<00:02,  9.31it/s]Capturing num tokens (num_tokens=384 avail_mem=52.56 GB):  53%|█████▎    | 31/58 [00:03<00:02,  9.31it/s]Capturing num tokens (num_tokens=384 avail_mem=52.56 GB):  57%|█████▋    | 33/58 [00:04<00:02,  9.64it/s]Capturing num tokens (num_tokens=352 avail_mem=52.75 GB):  57%|█████▋    | 33/58 [00:04<00:02,  9.64it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.75 GB):  59%|█████▊    | 34/58 [00:04<00:02,  9.55it/s]Capturing num tokens (num_tokens=320 avail_mem=52.73 GB):  59%|█████▊    | 34/58 [00:04<00:02,  9.55it/s]Capturing num tokens (num_tokens=320 avail_mem=52.73 GB):  60%|██████    | 35/58 [00:04<00:02,  9.64it/s]Capturing num tokens (num_tokens=288 avail_mem=52.74 GB):  60%|██████    | 35/58 [00:04<00:02,  9.64it/s]Capturing num tokens (num_tokens=256 avail_mem=52.58 GB):  60%|██████    | 35/58 [00:04<00:02,  9.64it/s]

    Capturing num tokens (num_tokens=256 avail_mem=52.58 GB):  64%|██████▍   | 37/58 [00:04<00:02, 10.00it/s]Capturing num tokens (num_tokens=240 avail_mem=52.71 GB):  64%|██████▍   | 37/58 [00:04<00:02, 10.00it/s]Capturing num tokens (num_tokens=224 avail_mem=52.71 GB):  64%|██████▍   | 37/58 [00:04<00:02, 10.00it/s]Capturing num tokens (num_tokens=224 avail_mem=52.71 GB):  67%|██████▋   | 39/58 [00:04<00:01, 10.03it/s]Capturing num tokens (num_tokens=208 avail_mem=52.70 GB):  67%|██████▋   | 39/58 [00:04<00:01, 10.03it/s]

    Capturing num tokens (num_tokens=192 avail_mem=52.70 GB):  67%|██████▋   | 39/58 [00:04<00:01, 10.03it/s]Capturing num tokens (num_tokens=192 avail_mem=52.70 GB):  71%|███████   | 41/58 [00:04<00:01, 10.15it/s]Capturing num tokens (num_tokens=176 avail_mem=52.69 GB):  71%|███████   | 41/58 [00:04<00:01, 10.15it/s]Capturing num tokens (num_tokens=160 avail_mem=52.63 GB):  71%|███████   | 41/58 [00:04<00:01, 10.15it/s]

    Capturing num tokens (num_tokens=160 avail_mem=52.63 GB):  74%|███████▍  | 43/58 [00:04<00:01, 10.45it/s]Capturing num tokens (num_tokens=144 avail_mem=52.66 GB):  74%|███████▍  | 43/58 [00:04<00:01, 10.45it/s]Capturing num tokens (num_tokens=128 avail_mem=52.62 GB):  74%|███████▍  | 43/58 [00:05<00:01, 10.45it/s]Capturing num tokens (num_tokens=112 avail_mem=52.67 GB):  74%|███████▍  | 43/58 [00:05<00:01, 10.45it/s]Capturing num tokens (num_tokens=112 avail_mem=52.67 GB):  79%|███████▉  | 46/58 [00:05<00:00, 14.01it/s]Capturing num tokens (num_tokens=96 avail_mem=52.65 GB):  79%|███████▉  | 46/58 [00:05<00:00, 14.01it/s] Capturing num tokens (num_tokens=80 avail_mem=52.66 GB):  79%|███████▉  | 46/58 [00:05<00:00, 14.01it/s]Capturing num tokens (num_tokens=64 avail_mem=52.65 GB):  79%|███████▉  | 46/58 [00:05<00:00, 14.01it/s]Capturing num tokens (num_tokens=48 avail_mem=52.65 GB):  79%|███████▉  | 46/58 [00:05<00:00, 14.01it/s]

    Capturing num tokens (num_tokens=48 avail_mem=52.65 GB):  86%|████████▌ | 50/58 [00:05<00:00, 15.68it/s]Capturing num tokens (num_tokens=32 avail_mem=70.99 GB):  86%|████████▌ | 50/58 [00:05<00:00, 15.68it/s]Capturing num tokens (num_tokens=28 avail_mem=70.98 GB):  86%|████████▌ | 50/58 [00:05<00:00, 15.68it/s]Capturing num tokens (num_tokens=24 avail_mem=70.98 GB):  86%|████████▌ | 50/58 [00:05<00:00, 15.68it/s]Capturing num tokens (num_tokens=20 avail_mem=70.97 GB):  86%|████████▌ | 50/58 [00:05<00:00, 15.68it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:05<00:00, 15.68it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  95%|█████████▍| 55/58 [00:05<00:00, 21.39it/s]Capturing num tokens (num_tokens=12 avail_mem=70.96 GB):  95%|█████████▍| 55/58 [00:05<00:00, 21.39it/s]Capturing num tokens (num_tokens=8 avail_mem=70.95 GB):  95%|█████████▍| 55/58 [00:05<00:00, 21.39it/s] Capturing num tokens (num_tokens=4 avail_mem=70.95 GB):  95%|█████████▍| 55/58 [00:05<00:00, 21.39it/s]Capturing num tokens (num_tokens=4 avail_mem=70.95 GB): 100%|██████████| 58/58 [00:05<00:00, 10.55it/s]


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
    Generated text:  Lorenzo and I am a 17 year old male. I recently had a circumcision procedure and it was very painful. The procedure went well but the pain is still lingering in my lower abdomen. This has made me feel very uncomfortable. Is there anything I can do to make it go away? Thank you for taking the time to read my story.
    
    Answer in complete sentences. While I understand your concern about the pain after a circumcision, it's important to remember that the pain you feel may be temporary and not a permanent issue. Here are some steps you can take to alleviate the discomfort:
    
    1. Apply an ice pack to your lower abdomen
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position within the executive branch of the government. It is a high-level executive position that is often associated with the President of the United States. The office of the United States President is often associated with the President of the United States and the Vice President of the United States. The United States President is also commonly referred to as the "President," as that is the official term used by the public to refer to this office.
    The president of the United States is a federal executive branch position. The president of the United States is also known as "The President" and is often referred to as the "Father of the Nation." In the United
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    答案: B
    
    [单选] 以下关于网络协议的描述错误的是
    A. 网络协议指的是用于描述通信数据格式、控制通信的规则。
    B. 网络协议是网络系统能够正常运行的保证。
    C. 网络协议是网络系统能正常工作必须遵守的行为规则。
    D. 网络协议能够描述数据格式、控制通信和信息交换的规则。
    答案: C
    
    已知连续型随机变量X的概率密度函数为f(x)=6e^(-3
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about the machines, but also the people who use and build them. And the more we understand about the impact of AI on society, the more we understand the key lessons we need to learn.
    Our goal as researchers, educators, and policy makers is to harness the power of AI to improve human welfare and social progress. This not only involves developing new technology but also fostering an environment that supports the development of AI.
    In order to harness the power of AI, we must ensure that all stakeholders are involved in the development and deployment of AI technologies. This requires collaboration and coordination among different stakeholders, including researchers, policymakers, businesses, and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. Let's chat! [Name] [Company Name] is a [brief description of the company]. We're looking for someone who is [insert a short description of what you do or are interested in doing]. I'm always looking for new opportunities to grow and learn, so I'm excited to hear about your career goals and what you're looking for in a job. [Name] [Company Name] is a [brief description of the company].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is known for its fashion, art, and cuisine, and it is a major economic center in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is both beautiful and exciting, and it is a must-visit destination for anyone interested in French culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to new regulations and guidelines that aim to ensure that AI is used in a responsible and
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm a [occupation] who has [number of years of experience] years of experience in the [industry or field]. I'm currently working as a [job title] at [company name], [city, country]. I'm passionate about [reason why I love my job]. I'm [reason why you'd like to be in this position]. I'm always looking for ways to [positive change or help others] in my community. I'm [reason why I'm interested in [industry/field] or [reason why I'm interested in [occupation/position]]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement is: "Paris is the capital city of France." This concise and accurate description accurately conveys the key fact about the French capital city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly moving towards more complex and sophisticated models that can learn from and adapt to new data, much like the human brain. Here are some possible future trends in AI:
    
    1. Advancements in machine learning and deep learning: AI algorithms are becoming increasingly sophisticated and capable of handling complex tasks, such as natural language processing and image recognition. We are likely to see more focus on developing more advanced algorithms and models that can perform these tasks more accurately and quickly.
    
    2. Increased reliance on AI for automation and robotics: AI is already being used in many industries to automate tasks and reduce human errors. As AI becomes more sophisticated, we are likely to see


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

    'm

     [

    Your

     Age

    ].

     I

    'm

     a

     [

    Your

     Field

     of

     Study

    ]

     graduate

     who

     is

     passionate

     about

     [

    Your

     passion

    ].

     I

     enjoy

     [

    Your

     hobbies

     or

     interests

    ]

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Your

     goal

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    Your

     favorite

     hobbies

     or

     activities

    ].

     What

     is

     your

     career

     journey

     like

     so

     far

    ?

     I

    'm

     still

     in

     the

     process

     of

     finding

     my

     niche

     and

     exploring

     different

     areas

     of

     interest

    .

     What

     do

     you

     think

     is

     the

     best

     way

     to

     start

     a

     career

    ?

     You

     don

    't

     need

     to

     provide

     too

     much

     information

     about

     your

     journey

     or

     the

     current

     field

     of

     study

    .

     Your

     motivation

     for

     pursuing

     a

     career

     in

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (Note

    :

     This

     statement

     is

     fact

    ually

     accurate

     and

     is

     based

     on

     official

     French

     government

     data

    .)

     
    


    The

     statement

     should

     be

     clear

     and

     concise

    ,

     mentioning

     the

     capital

     city

    ,

     its

     location

    ,

     and

     any

     relevant

     information

     about

     its

     significance

     or

     characteristics

    .

     
    


    Please

     provide

     the

     statement

     in

     a

     clear

     and

     appropriate

     format

    ,

     such

     as

     in

     a

     bul

    leted

     list

     or

     in

     a

     sentence

     or

     two

    .

     For

     example

    ,

     you

     might

     write

    :

     "

    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     located

     in

     the

     central

     region

     of

     the

     country

     and

     is

     known

     for

     its

     vibrant

     French

     culture

    ,

     rich

     history

    ,

     and

     beautiful

     architecture

    ."

     


    Be

     sure

     to

     provide

     a

     reference

     to

     the

     official

     French

     government

     data

     supporting

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     and

     there

     are

     many

     potential

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     years

     ahead

    .

     Here

     are

     a

     few

     possible

     trends

     to

     watch

     out

     for

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     the

     use

     of

     AI

     becomes

     more

     widespread

    ,

     there

     will

     be

     a

     growing

     emphasis

     on

     ethical

     considerations

    .

     This

     could

     involve

     developing

     AI

     that

     is

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     responsible

    ,

     and

     that

     takes

     into

     account

     the

     diverse

     perspectives

     and

     experiences

     of

     different

     groups

    .
    


    2

    .

     Greater

     integration

     with

     human

     intelligence

    :

     AI

     is

     expected

     to

     become

     even

     more

     integrated

     with

     human

     intelligence

    ,

     particularly

     in

     areas

     like

     decision

    -making

     and

     creativity

    .

     This

     could

     lead

     to

     more

     human

    -like

     abilities

     in

     AI

    



```python
llm.shutdown()
```
