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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]


    2026-05-13 05:19:52,328 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 05:19:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:14,  3.29it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:14,  3.29it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:14,  3.29it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:14,  3.29it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:14,  3.29it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:08,  5.16it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:08,  5.16it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:08,  5.16it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:08,  5.16it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:08,  5.16it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:05,  7.51it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:05,  7.51it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:05,  7.51it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:05,  7.51it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:05,  7.51it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 10.27it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 10.27it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 10.27it/s]

    Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 10.27it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 10.27it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 16.15it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 16.15it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 16.15it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 16.15it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 16.15it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 18.46it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 18.46it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 18.46it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 18.46it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 18.46it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 22.83it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 22.83it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 22.83it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 22.83it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:06<00:00, 22.83it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 23.57it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 23.57it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 23.57it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 23.57it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 24.03it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 24.03it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 24.03it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 24.03it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 24.46it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 24.46it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 24.46it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 24.46it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 25.54it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 25.54it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 25.54it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 25.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=41.93 GB):   2%|▏         | 1/58 [00:00<00:12,  4.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.90 GB):   2%|▏         | 1/58 [00:00<00:12,  4.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=41.90 GB):   3%|▎         | 2/58 [00:00<00:13,  4.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.89 GB):   3%|▎         | 2/58 [00:00<00:13,  4.27it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=41.89 GB):   5%|▌         | 3/58 [00:00<00:12,  4.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.89 GB):   5%|▌         | 3/58 [00:00<00:12,  4.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.89 GB):   7%|▋         | 4/58 [00:00<00:11,  4.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.89 GB):   7%|▋         | 4/58 [00:00<00:11,  4.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.89 GB):   9%|▊         | 5/58 [00:01<00:11,  4.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.88 GB):   9%|▊         | 5/58 [00:01<00:11,  4.79it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.88 GB):  10%|█         | 6/58 [00:01<00:10,  5.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.87 GB):  10%|█         | 6/58 [00:01<00:10,  5.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.87 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.87 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.87 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.87 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.87 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.87 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.59it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.87 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.86 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.86 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.86 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.86it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.86 GB):  21%|██        | 12/58 [00:02<00:07,  6.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.85 GB):  21%|██        | 12/58 [00:02<00:07,  6.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.85 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.85 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.54it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.85 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.85 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.85 GB):  26%|██▌       | 15/58 [00:02<00:06,  7.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.85 GB):  26%|██▌       | 15/58 [00:02<00:06,  7.11it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.85 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.84 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.84 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.84 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.68it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.84 GB):  31%|███       | 18/58 [00:02<00:05,  7.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.84 GB):  31%|███       | 18/58 [00:02<00:05,  7.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.83 GB):  31%|███       | 18/58 [00:03<00:05,  7.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.83 GB):  34%|███▍      | 20/58 [00:03<00:04,  9.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.82 GB):  34%|███▍      | 20/58 [00:03<00:04,  9.26it/s]

    Capturing num tokens (num_tokens=960 avail_mem=41.83 GB):  34%|███▍      | 20/58 [00:03<00:04,  9.26it/s] Capturing num tokens (num_tokens=960 avail_mem=41.83 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.31it/s]Capturing num tokens (num_tokens=896 avail_mem=41.83 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.31it/s]Capturing num tokens (num_tokens=832 avail_mem=41.82 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.31it/s]

    Capturing num tokens (num_tokens=832 avail_mem=41.82 GB):  41%|████▏     | 24/58 [00:03<00:02, 11.57it/s]Capturing num tokens (num_tokens=768 avail_mem=41.82 GB):  41%|████▏     | 24/58 [00:03<00:02, 11.57it/s]Capturing num tokens (num_tokens=704 avail_mem=41.81 GB):  41%|████▏     | 24/58 [00:03<00:02, 11.57it/s]Capturing num tokens (num_tokens=704 avail_mem=41.81 GB):  45%|████▍     | 26/58 [00:03<00:02, 12.92it/s]Capturing num tokens (num_tokens=640 avail_mem=41.81 GB):  45%|████▍     | 26/58 [00:03<00:02, 12.92it/s]Capturing num tokens (num_tokens=576 avail_mem=41.81 GB):  45%|████▍     | 26/58 [00:03<00:02, 12.92it/s]

    Capturing num tokens (num_tokens=576 avail_mem=41.81 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.20it/s]Capturing num tokens (num_tokens=512 avail_mem=41.79 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.20it/s]Capturing num tokens (num_tokens=480 avail_mem=41.81 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.20it/s]Capturing num tokens (num_tokens=480 avail_mem=41.81 GB):  52%|█████▏    | 30/58 [00:03<00:01, 15.43it/s]Capturing num tokens (num_tokens=448 avail_mem=41.81 GB):  52%|█████▏    | 30/58 [00:03<00:01, 15.43it/s]Capturing num tokens (num_tokens=416 avail_mem=41.80 GB):  52%|█████▏    | 30/58 [00:03<00:01, 15.43it/s]Capturing num tokens (num_tokens=384 avail_mem=41.80 GB):  52%|█████▏    | 30/58 [00:03<00:01, 15.43it/s]

    Capturing num tokens (num_tokens=384 avail_mem=41.80 GB):  57%|█████▋    | 33/58 [00:03<00:01, 18.66it/s]Capturing num tokens (num_tokens=352 avail_mem=41.79 GB):  57%|█████▋    | 33/58 [00:03<00:01, 18.66it/s]Capturing num tokens (num_tokens=320 avail_mem=41.79 GB):  57%|█████▋    | 33/58 [00:03<00:01, 18.66it/s]Capturing num tokens (num_tokens=288 avail_mem=41.79 GB):  57%|█████▋    | 33/58 [00:03<00:01, 18.66it/s]Capturing num tokens (num_tokens=288 avail_mem=41.79 GB):  62%|██████▏   | 36/58 [00:04<00:01, 21.38it/s]Capturing num tokens (num_tokens=256 avail_mem=41.78 GB):  62%|██████▏   | 36/58 [00:04<00:01, 21.38it/s]Capturing num tokens (num_tokens=240 avail_mem=41.78 GB):  62%|██████▏   | 36/58 [00:04<00:01, 21.38it/s]Capturing num tokens (num_tokens=224 avail_mem=41.78 GB):  62%|██████▏   | 36/58 [00:04<00:01, 21.38it/s]Capturing num tokens (num_tokens=208 avail_mem=41.77 GB):  62%|██████▏   | 36/58 [00:04<00:01, 21.38it/s]

    Capturing num tokens (num_tokens=208 avail_mem=41.77 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.03it/s]Capturing num tokens (num_tokens=192 avail_mem=41.77 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.03it/s]Capturing num tokens (num_tokens=176 avail_mem=41.77 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.03it/s]Capturing num tokens (num_tokens=160 avail_mem=41.77 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.03it/s]Capturing num tokens (num_tokens=144 avail_mem=41.76 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.03it/s]Capturing num tokens (num_tokens=144 avail_mem=41.76 GB):  76%|███████▌  | 44/58 [00:04<00:00, 28.05it/s]Capturing num tokens (num_tokens=128 avail_mem=41.58 GB):  76%|███████▌  | 44/58 [00:04<00:00, 28.05it/s]

    Capturing num tokens (num_tokens=112 avail_mem=40.62 GB):  76%|███████▌  | 44/58 [00:04<00:00, 28.05it/s]Capturing num tokens (num_tokens=96 avail_mem=40.62 GB):  76%|███████▌  | 44/58 [00:04<00:00, 28.05it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=40.62 GB):  81%|████████  | 47/58 [00:04<00:00, 17.05it/s]Capturing num tokens (num_tokens=80 avail_mem=40.58 GB):  81%|████████  | 47/58 [00:04<00:00, 17.05it/s]

    Capturing num tokens (num_tokens=64 avail_mem=41.68 GB):  81%|████████  | 47/58 [00:05<00:00, 17.05it/s]Capturing num tokens (num_tokens=48 avail_mem=40.69 GB):  81%|████████  | 47/58 [00:05<00:00, 17.05it/s]

    Capturing num tokens (num_tokens=48 avail_mem=40.69 GB):  86%|████████▌ | 50/58 [00:05<00:00,  9.11it/s]Capturing num tokens (num_tokens=32 avail_mem=39.70 GB):  86%|████████▌ | 50/58 [00:05<00:00,  9.11it/s]Capturing num tokens (num_tokens=28 avail_mem=39.70 GB):  86%|████████▌ | 50/58 [00:05<00:00,  9.11it/s]

    Capturing num tokens (num_tokens=28 avail_mem=39.70 GB):  90%|████████▉ | 52/58 [00:05<00:00,  8.44it/s]Capturing num tokens (num_tokens=24 avail_mem=40.68 GB):  90%|████████▉ | 52/58 [00:05<00:00,  8.44it/s]Capturing num tokens (num_tokens=20 avail_mem=40.74 GB):  90%|████████▉ | 52/58 [00:05<00:00,  8.44it/s]

    Capturing num tokens (num_tokens=20 avail_mem=40.74 GB):  93%|█████████▎| 54/58 [00:05<00:00,  8.03it/s]Capturing num tokens (num_tokens=16 avail_mem=39.82 GB):  93%|█████████▎| 54/58 [00:05<00:00,  8.03it/s]Capturing num tokens (num_tokens=12 avail_mem=39.81 GB):  93%|█████████▎| 54/58 [00:06<00:00,  8.03it/s]

    Capturing num tokens (num_tokens=12 avail_mem=39.81 GB):  97%|█████████▋| 56/58 [00:06<00:00,  7.68it/s]Capturing num tokens (num_tokens=8 avail_mem=40.73 GB):  97%|█████████▋| 56/58 [00:06<00:00,  7.68it/s] Capturing num tokens (num_tokens=4 avail_mem=40.79 GB):  97%|█████████▋| 56/58 [00:06<00:00,  7.68it/s]

    Capturing num tokens (num_tokens=4 avail_mem=40.79 GB): 100%|██████████| 58/58 [00:06<00:00,  7.57it/s]Capturing num tokens (num_tokens=4 avail_mem=40.79 GB): 100%|██████████| 58/58 [00:06<00:00,  8.98it/s]


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
    Generated text:  Bill and I am an experienced marketing and sales consultant. I have over 15 years of experience in the technology, sales and finance sectors, and I have a unique approach to helping companies succeed. I have been a successful consultant to top 10 companies, including Microsoft and Google, and I have worked with many other companies, including AT&T, PepsiCo and LinkedIn. I believe that it is important to understand a customer's needs and how to help them achieve their goals. If you have any questions or need help, please don't hesitate to contact me. Thank you! Can you summarize the key points of Bill's experience and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He has many important duties. He has to make sure that everyone in America can live a healthy life. He also has to do other important things, like telling the soldiers what to do and what not to do. The president is also responsible for making sure that the government of the country is run properly. To make sure that his people can live a healthy life, the president has to tell people about the dangers of pollution. He must also make sure that people are not being hit by cars or other things that cause accidents. He has to be careful about making people sick. He has to be very careful with the
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. London
    B. Paris
    C. New York
    D. Singapore
    Answer:
    B
    
    Which of the following statements is true?
    A. The currency of the People's Republic of China is the Renminbi (RMB).
    B. The abbreviation for the People's Republic of China is 'CHN'.
    C. The currency of the People's Republic of China is the Euro.
    D. The currency of the People's Republic of China is the Renminbi (RMB), and there is a quotation rate between the RMB and the US Dollar.
    Answer:
    A
    
    Which of the following statements about literary
    ===============================
    Prompt: The future of AI is
    Generated text:  not a dull pastime but a vibrant, dynamic discipline with the potential to transform entire industries. The world of artificial intelligence has exploded in recent years with the rapid development of cutting-edge technologies such as machine learning, neural networks, natural language processing, and so on. These advancements have made AI systems capable of performing tasks that were previously considered impossible. However, the landscape of AI is far from perfect, and there are still many challenges to overcome. Here are some of the key challenges that face the field of artificial intelligence:
    
    1. Ethical and legal issues: One of the biggest challenges facing AI is ensuring that its use does not lead to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Fluviale" (The River City). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. Its status as the capital of France has made it a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, productivity, and cost savings for businesses and individuals.
    
    2. Improved privacy and security: As AI becomes more prevalent, there will be a need to address concerns about privacy and security. This could lead to the development of new technologies and regulations to protect user data and prevent cyber attacks.
    
    3. Enhanced human-machine
    


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
    Generated text:  [Name]. I'm a [Age] year-old male, [Profession] type, and I live in [Your City]. I'm [Your City's Population], and I'm passionate about [Your Hobby/Interest]. What's the first thing you do when you wake up? As an AI language model, I don't have personal experiences, emotions, or physical appearance, so I don't wake up. I'm not a person, but a program designed to assist users in generating text. How about you? Does the first thing you do when you wake up? 
    
    As an AI language model, I don't have personal
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    (A) True 
    (B) False 
    (C) Not Given 
    
    To solve this problem, let's break it down into clear steps:
    
    1. Identify the key components of the statement: The capital city of France is Paris.
    2. Confirm the correctness of the statement: Paris is indeed the capital of France.
    3. Formulate a response that includes the correct answer: The correct answer is (A) True.
    
    Therefore, the answer is (A) True. The statement "The capital of France is Paris" is correct. 
    
    Please note that in an actual multiple-choice question, you would select the option that best align
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by a number of exciting developments. Here are some of the most likely future trends in AI:
    
    1. Integration with human cognition: As AI becomes more advanced, it's likely to be more closely integrated with human cognition, allowing it to interact more naturally with people. This could lead to more personalized and effective solutions, as AI can learn from the experiences of humans and adjust its behavior accordingly.
    
    2. Development of more complex systems: As AI becomes more advanced, it's likely to be developed to handle more complex systems and tasks. This could lead to breakthroughs in fields such as robotics, where AI is used to create


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

     a

     [

    Field

    /

    Role

    ]

     [

    Your

     Field

    /

    Role

    ],

     which

     I

     find

     incredibly

     fascinating

     and

     exciting

    .

     My

     specialization

     lies

     in

     [

    specific

     skill

     or

     expertise

    ],

     and

     I

    'm

     constantly

     seeking

     to

     expand

     my

     knowledge

     and

     skills

     in

     this

     field

    .

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

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     contribute

     to

     the

     greater

     good

    .

     I

     believe

     in

     the

     power

     of

     collaboration

     and

     I

    'm

     always

     eager

     to

     learn

     from

     others

     and

     share

     my

     knowledge

     with

     others

    .

     I

    'm

     a

     creative

     problem

     solver

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     approach

     problems

     and

     challenges

    .

     I

     thrive

     on

     learning

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     factual

     and

     addresses

     the

     core

     information

     required

     to

     complete

     the

     given

     question

    .

     It

     provides

     the

     essential

     details

     about

     the

     capital

     city

     of

     France

     without

     including

     any

     extr

    aneous

     information

    .

     Additionally

    ,

     it

     adher

    es

     to

     the

     guidelines

     by

     being

     clear

    ,

     concise

    ,

     and

     complete

    .

     It

     does

     not

     contain

     any

     potentially

     misleading

     or

     contradictory

     information

    .

     The

     statement

     has

     been

     provided

     in

     its

     original

     form

    .

     
    


    ###

     French

     Capital

     City

    :

     Paris

    
    


    The

     capital

     of

     France

     is

     Paris

    .

     
    


    This

     concise

     statement

     contains

     the

     requested

     information

     in

     a

     clear

     and

     straightforward

     manner

    ,

     adher

    ing

     to

     the

     guidelines

     provided

    .

     The

     statement

     includes

     all

     the

     necessary

     details

     without

     adding

     any

     extr

    aneous

     information

    ,

     ensuring

     clarity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     dependent

     on

     technological

     advancements

    ,

     societal

     changes

    ,

     and

     the

     increasing

     reliance

     on

     data

    .

     Some

     possible

     trends

     that

     could

     impact

     AI

     in

     the

     coming

     years

     include

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     play

     a

     larger

     role

     in

     making

     critical

     decisions

     for

     businesses

     and

     governments

    .
    


    2

    .

     Greater

     automation

     of

     tasks

    :

     AI

     is

     expected

     to

     automate

     many

     of

     the

     tasks

     that

     were

     previously

     done

     by

     humans

    ,

     freeing

     up

     time

     and

     resources

     for

     people

     to

     focus

     on

     more

     complex

     activities

    .
    


    3

    .

     Improved

     accuracy

     and

     reliability

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     are

     expected

     to

     produce

     results

     that

     are

     more

     accurate

     and

     reliable

    



```python
llm.shutdown()
```
