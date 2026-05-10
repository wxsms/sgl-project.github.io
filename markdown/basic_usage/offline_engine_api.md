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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]


    2026-05-10 06:43:10,132 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 06:43:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:10,  4.64it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  7.82it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  7.82it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  7.82it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:05,  7.82it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:05,  7.82it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.55it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.52it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.52it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.52it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.52it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.52it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 15.52it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:01, 20.96it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]

    Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 40.81it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 44.48it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 44.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.27it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.54it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.41 GB):   7%|▋         | 4/58 [00:00<00:07,  7.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.41 GB):   7%|▋         | 4/58 [00:00<00:07,  7.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.89it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.51it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:04,  9.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:04,  9.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:04,  9.98it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.86 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.87it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.85 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.83 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.80it/s]

    Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  34%|███▍      | 20/58 [00:02<00:02, 12.80it/s] Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=896 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=832 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.80it/s]Capturing num tokens (num_tokens=832 avail_mem=52.84 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.67it/s]Capturing num tokens (num_tokens=768 avail_mem=52.83 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.67it/s]

    Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.67it/s]Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.23it/s]Capturing num tokens (num_tokens=640 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.23it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.23it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.84it/s]Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.84it/s]

    Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.84it/s]Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.00it/s]Capturing num tokens (num_tokens=448 avail_mem=52.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.00it/s]Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.00it/s]Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.20it/s]Capturing num tokens (num_tokens=384 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.20it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.20it/s]Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.21it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.21it/s]Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.21it/s]Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:02<00:01, 16.22it/s]Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:02<00:01, 16.22it/s]

    Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.22it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.78it/s]Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.78it/s]Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.78it/s]

    Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 15.55it/s]Capturing num tokens (num_tokens=192 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 15.55it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 15.55it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.99it/s]Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.99it/s]Capturing num tokens (num_tokens=144 avail_mem=52.78 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.99it/s]

    Capturing num tokens (num_tokens=144 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.44it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.44it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.44it/s]

    Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  79%|███████▉  | 46/58 [00:03<00:01, 11.55it/s]Capturing num tokens (num_tokens=96 avail_mem=52.78 GB):  79%|███████▉  | 46/58 [00:03<00:01, 11.55it/s] Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  79%|███████▉  | 46/58 [00:03<00:01, 11.55it/s]

    Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  83%|████████▎ | 48/58 [00:04<00:01,  9.80it/s]Capturing num tokens (num_tokens=64 avail_mem=52.77 GB):  83%|████████▎ | 48/58 [00:04<00:01,  9.80it/s]Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  83%|████████▎ | 48/58 [00:04<00:01,  9.80it/s]

    Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:04<00:00,  8.79it/s]Capturing num tokens (num_tokens=32 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:04<00:00,  8.79it/s]Capturing num tokens (num_tokens=28 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:04<00:00,  8.79it/s]

    Capturing num tokens (num_tokens=28 avail_mem=52.76 GB):  90%|████████▉ | 52/58 [00:04<00:00,  8.47it/s]Capturing num tokens (num_tokens=24 avail_mem=52.75 GB):  90%|████████▉ | 52/58 [00:04<00:00,  8.47it/s]Capturing num tokens (num_tokens=24 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:04<00:00,  8.31it/s]Capturing num tokens (num_tokens=20 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:04<00:00,  8.31it/s]

    Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:04<00:00,  8.31it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  95%|█████████▍| 55/58 [00:04<00:00,  9.42it/s]Capturing num tokens (num_tokens=12 avail_mem=52.74 GB):  95%|█████████▍| 55/58 [00:04<00:00,  9.42it/s]Capturing num tokens (num_tokens=8 avail_mem=52.74 GB):  95%|█████████▍| 55/58 [00:04<00:00,  9.42it/s] Capturing num tokens (num_tokens=8 avail_mem=52.74 GB):  98%|█████████▊| 57/58 [00:04<00:00, 10.78it/s]Capturing num tokens (num_tokens=4 avail_mem=52.74 GB):  98%|█████████▊| 57/58 [00:04<00:00, 10.78it/s]

    Capturing num tokens (num_tokens=4 avail_mem=52.74 GB): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


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
    Generated text:  Zainab. I'm a student at the University of the West Indies (University of the West Indies), Carrefour International College (CUIC) in Barbados, where I study a Bachelor of Arts (Honours) in Politics. I'm a third-year student. I specialize in the study of economics and political science, as well as politics. I'm not a lawyer or a CPA but I do have a background in law, as I hold a Bachelor of Law (Honours) from the University of the West Indies (University of the West Indies), Carrefour International College (CUIC) in Barbados. Yes
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to send his family on a road trip or to the Olympics. He has two options, and he is considering two different strategies. The first strategy involves a 100% chance that he will be able to find the family members who want to go on the road trip. The second strategy involves a 20% chance that he will find the family members, but there is a 50% chance that he will not find anyone who wants to go. If the president uses the first strategy, what is the probability that he will find at least one person who wants to go on the road trip? Express
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of the region of which is this city?
    The answer is: Paris region. The capital of France is Paris, and Paris is located in the Paris region. The capital of the Paris region is the capital city, which is Paris. So, the capital of the Paris region is also Paris. 
    
    To provide more context, the Paris region is a geographical and cultural area in the north of France, covering parts of the departments of Aix-en-Provence, Bourgogne-Franche-Comté, Occitanie, and the Greater Paris area. It is known for its historical importance, rich culture, and major
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and promising, with potential to revolutionize the way we learn, work, communicate, and connect with others. Here are some of the most exciting developments in AI and their potential impact:
    
    1. Machine learning: Machine learning is a subset of AI that involves training computers to perform tasks using data. With the development of deep learning, AI has become more capable of handling complex and intricate tasks, and has the potential to transform industries such as healthcare, finance, and manufacturing.
    
    2. Natural language processing: Natural language processing is the ability of AI to understand and interpret human language. With the development of large language models, we can expect to


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As an AI language model, I don't have a physical presence, but I can assist you with any questions or tasks you may have. How can I help you today? Let's get started! [Name] [Company Name] [Job Title] [Company Name] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is home to many of the world’s most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its rich cultural heritage, including its art, music, and cuisine, and is a major center for business, politics, and entertainment. The city is home to many of the world’s most famous museums, including the Louvre and the Musée d'Orsay, and is a major center for fashion, design
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation and robotics: AI is already being used in manufacturing, healthcare, and transportation, and we can expect it to become even more prevalent in these areas as technology continues to advance.
    
    2. Improved natural language processing: AI is already capable of understanding and generating human language, but we can expect it to become even more sophisticated in the future, allowing machines to understand and respond to human speech in a more natural way.
    
    3. Enhanced machine learning: AI is already capable of learning from data, but we can expect it to become even more powerful in the future
    


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
    Generated text:  [Name]. I'm a friendly and engaging storyteller. I love to use my words to create a captivating narrative for the characters. My writing always aims to inspire others and make them feel something. I'm always looking to add something new to the literary world and I'm always eager to learn and grow as a writer. What would you like to call me? Name? Name? Name? Name? Name. Name? Name? Name? Name? Name? Name. Name? Name? Name? Name?
    Name?
    Hello, my name is [Name]. I'm a friendly and engaging storyteller. I love to use my words
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the seat of government, administration, and culture. Its population is over 2 million. Paris is renowned for its rich history, iconic architecture, and lively cultural scene. It is also a major financial center, being home to the Eiffel Tower, Louvre Museum, and the French Parliament. Paris is also a popular tourist destination, hosting numerous cultural events and events throughout the year. Its status as a global cultural hub has made Paris one of the most visited cities in the world, attracting millions of tourists every year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly dynamic and evolving, with numerous possibilities and trends shaping its direction. Here are some of the possible future trends in AI:
    
    1. Increased Machine Learning and Deep Learning: With the advancements in hardware and software, it is expected that machine learning and deep learning will continue to dominate AI. This means that AI systems will become increasingly sophisticated and capable of performing complex tasks, from natural language processing to computer vision.
    
    2. Increased Autonomy: With the continued advancement of AI, it is expected that it will become more autonomous, able to make decisions and take action without direct human intervention.
    
    3. AI in Healthcare: AI has already made significant


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

    career

     field

    ]

     with

     [

    number

    ]

     years

     of

     experience

    .

     I

    'm

     passionate

     about

     [

    career

     field

    ]

     and

     have

     a

     unique

     approach

     to

     problem

    -solving

    .

     I

     thrive

     in

     collaborative

     teams

     and

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     knowledge

    .

     Please

     let

     me

     know

     if

     you

    're

     interested

     in

     learning

     more

     about

     me

     or

     have

     any

     questions

     about

     my

     career

    .

     [

    Name

    ]

     (

    Optional

    )

     Hi

     there

    !

     I

    'm

     [

    Name

    ]

     and

     I

    'm

     a

     [

    career

     field

    ]

     with

     [

    number

    ]

     years

     of

     experience

    .

     I

    'm

     passionate

     about

     [

    career

     field

    ]

     and

     have

     a

     unique

     approach

     to

     problem

    -solving

    .

     I

     thrive

     in

     collaborative

     teams

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     renowned

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

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     exciting

     developments

    .

     Here are

     some

     of

     the

     possible

     trends

     that

     AI

     is

     likely

     to

     experience

    :
    


    1

    .

     Autonomous

     vehicles

    :

     The

     future

     of

     AI

     is

     likely

     to

     be

     full

     of

     autonomous

     vehicles

    .

     These

     vehicles

     are

     equipped

     with

     sensors

    ,

     cameras

    ,

     and

     algorithms

     that

     allow

     them

     to

     navigate

     roads

    ,

     recognize

     objects

     in

     the

     environment

    ,

     and

     make

     decisions

     on

     the

     road

    .

     As

     the

     technology

     advances

    ,

     it

    's

     expected

     that

     autonomous

     vehicles

     will

     become

     more

     common

     in

     the

     future

    .
    


    2

    .

     Cyber

    security

    :

     With

     the

     increasing

     amount

     of

     data

     being

     collected

     and

     stored

    ,

     the

     risk

     of

     cybersecurity

     attacks

     is

     becoming

     more

     and

     more

     serious

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     expected

     that

     cybersecurity

    



```python
llm.shutdown()
```
