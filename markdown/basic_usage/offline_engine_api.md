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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 08:26:27] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.87it/s]


    2026-04-17 08:26:32,595 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 08:26:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]

    Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:03<00:03, 11.67it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]

    Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:03<00:01, 19.11it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]

    Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 37.40it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 45.11it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 52.72it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 52.72it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 52.72it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 52.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.69 GB):   3%|▎         | 2/58 [00:00<00:04, 11.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.68 GB):   3%|▎         | 2/58 [00:00<00:04, 11.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.68 GB):   3%|▎         | 2/58 [00:00<00:04, 11.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.68 GB):   7%|▋         | 4/58 [00:00<00:04, 13.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.68 GB):   7%|▋         | 4/58 [00:00<00:04, 13.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.67 GB):   7%|▋         | 4/58 [00:00<00:04, 13.11it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=118.67 GB):  10%|█         | 6/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.67 GB):  10%|█         | 6/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.67 GB):  10%|█         | 6/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.67 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.66 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.09it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=118.65 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.65 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.49 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.48 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.65it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=117.48 GB):  21%|██        | 12/58 [00:01<00:04, 11.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.48 GB):  21%|██        | 12/58 [00:01<00:04, 11.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.47 GB):  21%|██        | 12/58 [00:01<00:04, 11.09it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=117.47 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.46 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.46 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.45 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.37it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=117.44 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.44 GB):  31%|███       | 18/58 [00:01<00:03, 10.28it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.42 GB):  31%|███       | 18/58 [00:01<00:03, 10.28it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.43 GB):  31%|███       | 18/58 [00:01<00:03, 10.28it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=117.43 GB):  34%|███▍      | 20/58 [00:01<00:03, 10.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.41 GB):  34%|███▍      | 20/58 [00:01<00:03, 10.36it/s]Capturing num tokens (num_tokens=960 avail_mem=117.42 GB):  34%|███▍      | 20/58 [00:01<00:03, 10.36it/s] Capturing num tokens (num_tokens=960 avail_mem=117.42 GB):  38%|███▊      | 22/58 [00:01<00:03, 10.64it/s]Capturing num tokens (num_tokens=896 avail_mem=117.41 GB):  38%|███▊      | 22/58 [00:01<00:03, 10.64it/s]

    Capturing num tokens (num_tokens=832 avail_mem=117.40 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.64it/s]Capturing num tokens (num_tokens=832 avail_mem=117.40 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.74it/s]Capturing num tokens (num_tokens=768 avail_mem=117.40 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.74it/s]Capturing num tokens (num_tokens=704 avail_mem=117.39 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.74it/s]

    Capturing num tokens (num_tokens=704 avail_mem=117.39 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.07it/s]Capturing num tokens (num_tokens=640 avail_mem=117.38 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.07it/s]Capturing num tokens (num_tokens=576 avail_mem=117.40 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.07it/s]Capturing num tokens (num_tokens=576 avail_mem=117.40 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.17it/s]Capturing num tokens (num_tokens=512 avail_mem=117.38 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.17it/s]

    Capturing num tokens (num_tokens=480 avail_mem=117.39 GB):  48%|████▊     | 28/58 [00:02<00:02, 11.17it/s]Capturing num tokens (num_tokens=480 avail_mem=117.39 GB):  52%|█████▏    | 30/58 [00:02<00:02, 11.22it/s]Capturing num tokens (num_tokens=448 avail_mem=117.39 GB):  52%|█████▏    | 30/58 [00:02<00:02, 11.22it/s]Capturing num tokens (num_tokens=416 avail_mem=117.38 GB):  52%|█████▏    | 30/58 [00:02<00:02, 11.22it/s]

    Capturing num tokens (num_tokens=416 avail_mem=117.38 GB):  55%|█████▌    | 32/58 [00:02<00:02, 11.31it/s]Capturing num tokens (num_tokens=384 avail_mem=117.38 GB):  55%|█████▌    | 32/58 [00:02<00:02, 11.31it/s]Capturing num tokens (num_tokens=352 avail_mem=117.37 GB):  55%|█████▌    | 32/58 [00:02<00:02, 11.31it/s]Capturing num tokens (num_tokens=352 avail_mem=117.37 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.39it/s]Capturing num tokens (num_tokens=320 avail_mem=117.36 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.39it/s]

    Capturing num tokens (num_tokens=288 avail_mem=117.35 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.39it/s]Capturing num tokens (num_tokens=288 avail_mem=117.35 GB):  62%|██████▏   | 36/58 [00:03<00:01, 11.47it/s]Capturing num tokens (num_tokens=256 avail_mem=117.35 GB):  62%|██████▏   | 36/58 [00:03<00:01, 11.47it/s]Capturing num tokens (num_tokens=240 avail_mem=117.35 GB):  62%|██████▏   | 36/58 [00:03<00:01, 11.47it/s]

    Capturing num tokens (num_tokens=240 avail_mem=117.35 GB):  66%|██████▌   | 38/58 [00:03<00:01, 11.36it/s]Capturing num tokens (num_tokens=224 avail_mem=117.34 GB):  66%|██████▌   | 38/58 [00:03<00:01, 11.36it/s]Capturing num tokens (num_tokens=208 avail_mem=117.33 GB):  66%|██████▌   | 38/58 [00:03<00:01, 11.36it/s]Capturing num tokens (num_tokens=208 avail_mem=117.33 GB):  69%|██████▉   | 40/58 [00:03<00:01, 11.82it/s]Capturing num tokens (num_tokens=192 avail_mem=117.33 GB):  69%|██████▉   | 40/58 [00:03<00:01, 11.82it/s]

    Capturing num tokens (num_tokens=176 avail_mem=117.32 GB):  69%|██████▉   | 40/58 [00:03<00:01, 11.82it/s]Capturing num tokens (num_tokens=176 avail_mem=117.32 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.36it/s]Capturing num tokens (num_tokens=160 avail_mem=117.31 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.36it/s]Capturing num tokens (num_tokens=144 avail_mem=117.31 GB):  72%|███████▏  | 42/58 [00:03<00:01, 12.36it/s]Capturing num tokens (num_tokens=144 avail_mem=117.31 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.18it/s]Capturing num tokens (num_tokens=128 avail_mem=117.30 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.18it/s]

    Capturing num tokens (num_tokens=112 avail_mem=117.29 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.18it/s]Capturing num tokens (num_tokens=112 avail_mem=117.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 13.71it/s]Capturing num tokens (num_tokens=96 avail_mem=117.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 13.71it/s] Capturing num tokens (num_tokens=80 avail_mem=117.28 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.71it/s]Capturing num tokens (num_tokens=80 avail_mem=117.28 GB):  83%|████████▎ | 48/58 [00:04<00:00, 14.88it/s]Capturing num tokens (num_tokens=64 avail_mem=117.27 GB):  83%|████████▎ | 48/58 [00:04<00:00, 14.88it/s]

    Capturing num tokens (num_tokens=48 avail_mem=117.27 GB):  83%|████████▎ | 48/58 [00:04<00:00, 14.88it/s]Capturing num tokens (num_tokens=32 avail_mem=117.26 GB):  83%|████████▎ | 48/58 [00:04<00:00, 14.88it/s]Capturing num tokens (num_tokens=32 avail_mem=117.26 GB):  88%|████████▊ | 51/58 [00:04<00:00, 17.46it/s]Capturing num tokens (num_tokens=28 avail_mem=117.25 GB):  88%|████████▊ | 51/58 [00:04<00:00, 17.46it/s]Capturing num tokens (num_tokens=24 avail_mem=117.25 GB):  88%|████████▊ | 51/58 [00:04<00:00, 17.46it/s]Capturing num tokens (num_tokens=20 avail_mem=117.24 GB):  88%|████████▊ | 51/58 [00:04<00:00, 17.46it/s]Capturing num tokens (num_tokens=20 avail_mem=117.24 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.00it/s]Capturing num tokens (num_tokens=16 avail_mem=117.23 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.00it/s]

    Capturing num tokens (num_tokens=12 avail_mem=117.22 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.00it/s]Capturing num tokens (num_tokens=8 avail_mem=117.21 GB):  93%|█████████▎| 54/58 [00:04<00:00, 20.00it/s] Capturing num tokens (num_tokens=8 avail_mem=117.21 GB):  98%|█████████▊| 57/58 [00:04<00:00, 22.53it/s]Capturing num tokens (num_tokens=4 avail_mem=117.21 GB):  98%|█████████▊| 57/58 [00:04<00:00, 22.53it/s]Capturing num tokens (num_tokens=4 avail_mem=117.21 GB): 100%|██████████| 58/58 [00:04<00:00, 13.11it/s]


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
    Generated text:  Pepe, a day-to-day daydreamer. I have a lot of things I want to know, but I’m not sure where to start. What should I do?
    This post is designed to help you with your daily daydreaming.
    It’s common to have an idea, but to be quiet and not tell others what you have in your head, but here’s how you can help yourself.
    One of the best things I’ve found to do is practice your breathing. This is one of the most important things I’ve found to do for getting all those ideas out of my head.
    I’ve been taught by many people that
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president is like the king of the country. He is the head of government. He has a lot of important jobs. His job is to make important decisions. He is also the leader of the country. Most of the time the president meets with other government leaders. The president can make decisions for the country. But the other government leaders make decisions for the country.
    
    Write the first sentence of the text.
    The first sentence of the text is: "The president of the United States is a very important person." The president is like the king of the country. He is the head of government. He has a
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the coast of the ____.
    A. Mediterranean Sea
    B. Atlantic Ocean
    C. English Channel
    D. Black Sea
    Answer: A
    
    To promote the exchange and sharing of knowledge, the government has strengthened cooperation between universities and enterprises. This is a manifestation of ____
    A. International cooperation
    B. Regional cooperation
    C. Economic cooperation
    D. Technological cooperation
    Answer: D
    
    The total length of the Earth is approximately ____ kilometers.
    A. 40000
    B. 6371
    C. 6378
    D. 80000
    
    ===============================
    Prompt: The future of AI is
    Generated text:  already here. In 2020 alone, tech giants such as Google, Microsoft, and Amazon launched an estimated 500 million new users, and AI was making breakthroughs on a variety of problems. If we are not careful, we will enter a 'black hole' where AI is too powerful for human control, and the world might be very different. Which of the following arguments does the above passage emphasize?
    A. The rapidly developing technology industry is uncontrollable.
    B. The rapid development of AI technology poses significant challenges.
    C. The government should strengthen its control over AI technology.
    D. The rapid development of AI


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, making it a major economic and cultural hub in France. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the French Academy of Fine Arts. Paris is a city of contrasts, with its modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to the creation of new jobs, but it could also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a need to address ethical and privacy concerns. This could lead to new regulations and standards being developed to ensure that AI is used in a responsible and ethical
    


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
    Generated text:  [Your Name]. I'm a [Occupation] with a passion for [Job Title]. I'm [Your Age] years old and [Your Job Title] experience, having [mention any relevant achievements or skills]. I'm a [Your Profession] who is [Your Interests and hobbies], and I love [the thing that makes you unique]. I also have a [Your Unique Strength or Challenge], and I'm always striving to [mention any goals or dreams]. I love [the things that make me happy] and I'm always [your attitude]. I'm a [Your Personality] person who is [Your Values and Core
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and ornate architecture.
    
    France's capital city, Paris, is renowned for its iconic Eiffel Tower and ornate architecture, making it a cultural and historical gem among its metropolitan regions. 
    
    I. The Eiffel Tower: Standing at 324 meters tall and attracting millions of visitors each year, the Eiffel Tower is France's most recognizable landmark and one of the most photographed symbols of the nation. It was first built in 1889 and remains the tallest man-made structure in the world.
    
    II. Architectural Marvels: Paris is home to countless
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but here are some possible trends:
    
    1. Increased efficiency and automation: AI is becoming more efficient at performing tasks and making decisions, leading to increased productivity and automation in various industries. This trend could lead to the development of new types of AI systems that can replace humans in some roles.
    
    2. Improved accuracy and reliability: AI systems are becoming more accurate and reliable, which could lead to improvements in various fields, including healthcare, finance, and transportation.
    
    3. Greater integration of AI and human intelligence: AI is becoming more integrated with human intelligence, with AI systems learning from human interactions and adapting to their needs. This integration could lead


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

    job

     title

    ]

     in

     [

    city

     or

     country

    ],

     currently

     [

    job

     title

    ].

     I

    've

     been

     working

     here

     for

     [

    number

     of

     years

    ]

     years

    ,

     and

     I

    'm

     always

     looking

     for

     opportunities

     for

     growth

     and

     challenge

    .

     I

     enjoy

     [

    something

     about

     my

     job

    ],

     and

     I

    'm

     always

     learning

     and

     growing

    .

     I

    'm

     a

     [

    summary

     of

     my

     character

    's

     personality

    ].

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     excited

     to

     learn

     more

     about

     your

     job

    .

     How

     can

     I

     help

     you

     today

    ?


    [

    Name

    ]

     would

     love

     to

     hear

     from

     you

    !

     Here

    's

     what

     I

    'm

     looking

     for

    :

     what

     kind

     of

     information

     do

     you

     need

     from

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     home

     to

     many

     of

     the

     country

    's

     historical

     landmarks

     and

     cultural

     institutions

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

     stunning

     architecture

    ,

     and

     vibrant

     culture

    .

     It

     is

     also

     famous

     for

     its

     annual

     E

    iff

    el

     Tower

     celebrations

    ,

     which

     draw

     thousands

     of

     visitors

     to

     the

     city

     every

     year

    .

     Paris

     has

     been

     a

     cultural

     and

     economic

     hub

     for

     over

     

    4

    0

    0

     years

     and

     continues

     to

     be

     a

     major

     center

     of

     education

    ,

     politics

    ,

     and

     fashion

    .

     The

     city

     has

     a

     long

     and

     complex

     history

     and

     is

     home

     to

     many

     different

     ethnic

     and

     religious

     groups

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     with

     millions

     of

     visitors

     each

     year

     seeking

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     emerging

     technologies

     and

     applications

    ,

     including

     more

     advanced

     algorithms

    ,

     larger

     data

     sets

    ,

     and

     distributed

     computing

    ,

     as

     well

     as

     greater

     emphasis

     on

     ethical

     and

     social

     implications

    .

     Additionally

    ,

     there

     may

     be

     increased

     focus

     on

     developing

     AI

     that

     is

     more

     human

    -like

    ,

     with

     greater

     integration

     of

     natural

     language

     processing

    ,

     machine

     learning

    ,

     and

     deep

     learning

    ,

     as

     well

     as

     greater

     adoption

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     Overall

    ,

     the

     trend

     is

     likely

     to

     be

     characterized

     by

     continued

     innovation

    ,

     progress

    ,

     and

     integration

     of

     AI

     into

     various

     industries

     and

     domains

    .

    



```python
llm.shutdown()
```
