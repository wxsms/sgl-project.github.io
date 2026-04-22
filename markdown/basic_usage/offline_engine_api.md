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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 09:36:36] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


    2026-04-22 09:36:40,465 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 09:36:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:07,  1.21s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:26,  2.00it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:26,  2.00it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:26,  2.00it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:15,  3.40it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:15,  3.40it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:15,  3.40it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 11.01it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 11.01it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 11.01it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 11.01it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 11.01it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 15.69it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 15.69it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 15.69it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 15.69it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 15.69it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 24.18it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 24.81it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 24.81it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 24.81it/s]

    Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 24.81it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 24.81it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:00, 25.34it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:00, 25.34it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:00, 25.34it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:00, 25.34it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 25.35it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 25.35it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 25.35it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 25.35it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 27.22it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 27.22it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 27.22it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 27.22it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 27.63it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 27.63it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 27.63it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 27.63it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 27.63it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 28.21it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 28.21it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 28.21it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 28.21it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 28.21it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 29.31it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 29.31it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 29.31it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 29.31it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 29.20it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 29.20it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=40.84 GB):   2%|▏         | 1/58 [00:00<00:13,  4.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.82 GB):   2%|▏         | 1/58 [00:00<00:13,  4.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.82 GB):   3%|▎         | 2/58 [00:00<00:12,  4.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.74 GB):   3%|▎         | 2/58 [00:00<00:12,  4.66it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=41.74 GB):   5%|▌         | 3/58 [00:00<00:11,  4.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.88 GB):   5%|▌         | 3/58 [00:00<00:11,  4.94it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.88 GB):   7%|▋         | 4/58 [00:00<00:11,  4.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.88 GB):   7%|▋         | 4/58 [00:00<00:11,  4.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.88 GB):   9%|▊         | 5/58 [00:00<00:09,  5.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.73 GB):   9%|▊         | 5/58 [00:00<00:09,  5.48it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.73 GB):  10%|█         | 6/58 [00:01<00:09,  5.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.95 GB):  10%|█         | 6/58 [00:01<00:09,  5.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.95 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.82 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.82it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.82 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.74 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.74 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.01 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.85it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.01 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.73 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.07 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.08it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.07 GB):  21%|██        | 12/58 [00:01<00:05,  7.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.06 GB):  21%|██        | 12/58 [00:01<00:05,  7.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.06 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.73 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.13 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.17it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=41.13 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.12 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.71 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.71 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.18 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.51it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=41.71 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.71 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.21 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.19 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.11it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=41.19 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.36it/s]Capturing num tokens (num_tokens=960 avail_mem=41.70 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.36it/s] Capturing num tokens (num_tokens=896 avail_mem=41.23 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.36it/s]Capturing num tokens (num_tokens=896 avail_mem=41.23 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.50it/s]Capturing num tokens (num_tokens=832 avail_mem=41.69 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.50it/s]Capturing num tokens (num_tokens=768 avail_mem=41.25 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.50it/s]

    Capturing num tokens (num_tokens=768 avail_mem=41.25 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.51it/s]Capturing num tokens (num_tokens=704 avail_mem=41.69 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.51it/s]Capturing num tokens (num_tokens=640 avail_mem=41.28 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.51it/s]Capturing num tokens (num_tokens=640 avail_mem=41.28 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.41it/s]Capturing num tokens (num_tokens=576 avail_mem=40.49 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.41it/s]

    Capturing num tokens (num_tokens=512 avail_mem=40.12 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.41it/s]Capturing num tokens (num_tokens=512 avail_mem=40.12 GB):  50%|█████     | 29/58 [00:03<00:02, 10.56it/s]Capturing num tokens (num_tokens=480 avail_mem=40.50 GB):  50%|█████     | 29/58 [00:03<00:02, 10.56it/s]

    Capturing num tokens (num_tokens=448 avail_mem=41.32 GB):  50%|█████     | 29/58 [00:03<00:02, 10.56it/s]Capturing num tokens (num_tokens=448 avail_mem=41.32 GB):  53%|█████▎    | 31/58 [00:03<00:02, 10.54it/s]Capturing num tokens (num_tokens=416 avail_mem=41.65 GB):  53%|█████▎    | 31/58 [00:03<00:02, 10.54it/s]

    Capturing num tokens (num_tokens=384 avail_mem=40.36 GB):  53%|█████▎    | 31/58 [00:03<00:02, 10.54it/s]Capturing num tokens (num_tokens=384 avail_mem=40.36 GB):  57%|█████▋    | 33/58 [00:03<00:02, 10.00it/s]Capturing num tokens (num_tokens=352 avail_mem=40.66 GB):  57%|█████▋    | 33/58 [00:03<00:02, 10.00it/s]

    Capturing num tokens (num_tokens=320 avail_mem=40.39 GB):  57%|█████▋    | 33/58 [00:03<00:02, 10.00it/s]Capturing num tokens (num_tokens=320 avail_mem=40.39 GB):  60%|██████    | 35/58 [00:04<00:02,  9.64it/s]Capturing num tokens (num_tokens=288 avail_mem=41.63 GB):  60%|██████    | 35/58 [00:04<00:02,  9.64it/s]

    Capturing num tokens (num_tokens=288 avail_mem=41.63 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.33it/s]Capturing num tokens (num_tokens=256 avail_mem=40.71 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.33it/s]Capturing num tokens (num_tokens=240 avail_mem=40.50 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.33it/s]

    Capturing num tokens (num_tokens=240 avail_mem=40.50 GB):  66%|██████▌   | 38/58 [00:04<00:02,  9.49it/s]Capturing num tokens (num_tokens=224 avail_mem=41.63 GB):  66%|██████▌   | 38/58 [00:04<00:02,  9.49it/s]Capturing num tokens (num_tokens=224 avail_mem=41.63 GB):  67%|██████▋   | 39/58 [00:04<00:01,  9.55it/s]Capturing num tokens (num_tokens=208 avail_mem=41.62 GB):  67%|██████▋   | 39/58 [00:04<00:01,  9.55it/s]Capturing num tokens (num_tokens=192 avail_mem=40.62 GB):  67%|██████▋   | 39/58 [00:04<00:01,  9.55it/s]

    Capturing num tokens (num_tokens=192 avail_mem=40.62 GB):  71%|███████   | 41/58 [00:04<00:01,  9.86it/s]Capturing num tokens (num_tokens=176 avail_mem=40.76 GB):  71%|███████   | 41/58 [00:04<00:01,  9.86it/s]Capturing num tokens (num_tokens=160 avail_mem=41.62 GB):  71%|███████   | 41/58 [00:04<00:01,  9.86it/s]Capturing num tokens (num_tokens=160 avail_mem=41.62 GB):  74%|███████▍  | 43/58 [00:04<00:01, 10.26it/s]Capturing num tokens (num_tokens=144 avail_mem=40.70 GB):  74%|███████▍  | 43/58 [00:04<00:01, 10.26it/s]

    Capturing num tokens (num_tokens=128 avail_mem=40.82 GB):  74%|███████▍  | 43/58 [00:04<00:01, 10.26it/s]Capturing num tokens (num_tokens=128 avail_mem=40.82 GB):  78%|███████▊  | 45/58 [00:05<00:01, 10.14it/s]Capturing num tokens (num_tokens=112 avail_mem=40.94 GB):  78%|███████▊  | 45/58 [00:05<00:01, 10.14it/s]Capturing num tokens (num_tokens=96 avail_mem=41.59 GB):  78%|███████▊  | 45/58 [00:05<00:01, 10.14it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=41.59 GB):  81%|████████  | 47/58 [00:05<00:01, 10.50it/s]Capturing num tokens (num_tokens=80 avail_mem=40.86 GB):  81%|████████  | 47/58 [00:05<00:01, 10.50it/s]Capturing num tokens (num_tokens=64 avail_mem=40.77 GB):  81%|████████  | 47/58 [00:05<00:01, 10.50it/s]Capturing num tokens (num_tokens=64 avail_mem=40.77 GB):  84%|████████▍ | 49/58 [00:05<00:00, 10.99it/s]Capturing num tokens (num_tokens=48 avail_mem=41.57 GB):  84%|████████▍ | 49/58 [00:05<00:00, 10.99it/s]

    Capturing num tokens (num_tokens=32 avail_mem=40.91 GB):  84%|████████▍ | 49/58 [00:05<00:00, 10.99it/s]Capturing num tokens (num_tokens=32 avail_mem=40.91 GB):  88%|████████▊ | 51/58 [00:05<00:00, 10.77it/s]Capturing num tokens (num_tokens=28 avail_mem=41.64 GB):  88%|████████▊ | 51/58 [00:05<00:00, 10.77it/s]Capturing num tokens (num_tokens=24 avail_mem=41.56 GB):  88%|████████▊ | 51/58 [00:05<00:00, 10.77it/s]

    Capturing num tokens (num_tokens=24 avail_mem=41.56 GB):  91%|█████████▏| 53/58 [00:05<00:00, 11.40it/s]Capturing num tokens (num_tokens=20 avail_mem=40.91 GB):  91%|█████████▏| 53/58 [00:05<00:00, 11.40it/s]Capturing num tokens (num_tokens=16 avail_mem=41.56 GB):  91%|█████████▏| 53/58 [00:05<00:00, 11.40it/s]Capturing num tokens (num_tokens=16 avail_mem=41.56 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.14it/s]Capturing num tokens (num_tokens=12 avail_mem=41.02 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.14it/s]

    Capturing num tokens (num_tokens=8 avail_mem=41.01 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.14it/s] Capturing num tokens (num_tokens=8 avail_mem=41.01 GB):  98%|█████████▊| 57/58 [00:06<00:00, 12.20it/s]Capturing num tokens (num_tokens=4 avail_mem=41.54 GB):  98%|█████████▊| 57/58 [00:06<00:00, 12.20it/s]Capturing num tokens (num_tokens=4 avail_mem=41.54 GB): 100%|██████████| 58/58 [00:06<00:00,  9.52it/s]


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
    Generated text:  Mike and I was going to come here as a tutor. (5) I have gone to all the schools in South Africa but none have offered me a job yet. My name is Tamar and I’m going to come here as a tutor. (6) I was born in the same year as you, Tamar. I’m a student in the same school. (7) Our relationship was based on a parent-child relationship. We had always shared a common interest in the same hobby: football. Tamar asked me to be her football coach. I accepted. (8) We were married in the same church. We had
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to have his vice president serve as his personal secretary. The president knows that the vice president is a person who is between 40 and 60 years old. He also knows that the vice president is a man. Furthermore, the president knows that a secretary for a president of the United States serves in the executive branch of government and is not a person who is between 40 and 60 years old. Which of the following conclusions logically follows from the president's statement?  
    
    A. The president knows that the vice president is a man who is between 40 and 60 years old.
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) Lyon
    C) London
    D) Brussels
    
    To determine the capital of France, we can consider the following key facts:
    
    1. Paris is the largest city in France.
    2. Lyon, while also a major city in France, is not considered the capital.
    3. London is the capital of the United Kingdom, not of France.
    4. Brussels, the capital of Belgium, is not the capital of France.
    
    Based on these facts, the capital of France is Paris.
    
    Therefore, the correct answer is: \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but we don’t know yet what it will be like. How will AI impact society? AI is increasingly used in many areas such as healthcare, finance, transportation, education and marketing. Today, we can see that AI is evolving from a technology that is only available to the experts, into a tool that is widely used by people from all walks of life, including those who are not experts.
    The use of AI in healthcare has been increasing in recent years, with the development of more advanced algorithms that are able to analyze data and provide better healthcare outcomes. In the field of finance, AI has been used to automate complex transactions and


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to many famous museums and art galleries, including the Musée d'Orsay and the Musée Rodin. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered diagnostic tools, predictive analytics, and personalized medicine are all areas where AI is expected to have a significant impact.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce costs, and improve quality. As AI technology continues to improve
    


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
    Generated text:  [Your Name] and I am [Your Profession]. I have a passion for [Your Hobby/Interest]. I love [Your Hobby/Interest] and have been creating content related to it for [Your Hobby/Interest] for [Your Hobby/Interest] years. I am always up-to-date with the latest trends and techniques, and I enjoy sharing my knowledge and expertise with others. My goal is to inspire and motivate people to pursue their passions and achieve their goals. I am available for interviews, interviews, and events.
    [Your Name] is a [Your Profession] who is passionate about [Your Hobby/Interest]. She loves
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is an important cultural, economic, and political center, known for its historic architecture, art museums, museums, and historic landmarks. The city is also famous for its cuisine and is home to many popular tourist attractions. In addition to its importance in France, Paris is also a major hub for international trade and diplomacy, and is known for its influence on French culture and politics. It is a popular destination for visitors from around the world, and its history and culture are celebrated in popular culture. Paris is the capital city of France and is known for its rich history, art, architecture, and culture. It is also a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve rapidly, driven by new innovations and advancements in technology, and a growing appetite for better, more accurate, and more human-like AI systems. Some possible future trends in AI include:
    
    1. Increased autonomy and personalization: As AI systems become more sophisticated, they will be able to perform a wider range of tasks, from managing household chores to diagnosing medical conditions. This could lead to greater personalization of AI services, allowing users to receive tailored recommendations and assistance based on their individual needs and preferences.
    
    2. Improved security and privacy: As AI systems become more advanced, there will be a need for increased security and


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

    ].

     I

     am

     a

     [

    occupation

    ]

     who

     has

     been

     around

     for

     [

    number

    ]

     years

    .

     I

     have

     always

     been

     curious

     about

     the

     world

     and

     have

     always

     loved

     to

     learn

    .

     I

     have

     a

     passion

     for

     creating

     content

     for

     [

    platform

    ]

     and

     have

     been

     working

     on

     it

     for

     [

    number

    ]

     years

    .

     I

     am

     constantly

     trying

     to

     improve

     my

     skills

     and

     knowledge

     in

     this

     field

    .

     I

     am

     excited

     to

     share

     my

     experiences

     and

     learn

     from

     others

    .

     Thank

     you

     for

     asking

     to

     meet

     me

    .

     [

    Name

    ]

     [

    Contact

     Information

    ]

     [

    Link

     to

     my

     portfolio

    ]

     [

    Your

     Bio

    ]

     
    


    Remember

     to

     be

     respectful

     and

     consider

    ate

     when

     writing

     your

     self

    -int

    roduction

    .

     Remember

     to

     include

     relevant

     details

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     second

    -largest

     in

     the

     European

     Union

    .

     Its

     population

     is

     around

     

    2

    .

    1

     million

     and

     it

     is

     the

     third

    -largest

     in

     terms

     of

     area

    .

     Paris

     is

     known

     for

     its

     stunning

     architecture

    ,

     diverse

     neighborhoods

    ,

     and

     rich

     cultural

     heritage

    ,

     including

     the

     iconic

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     night

    -time

     economy

     and

     vibrant

     nightlife

    .

     Paris

     is

     the

     fifth

    -largest

     city

     in

     the

     world

     by

     population

    ,

     and

     it

     is

     the

     most

     populous

     French

     city

     outside

     of

     Paris

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     the

     arrival

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     marked

     by

     a

     rapid

     expansion

     of

     its

     applications

     and

     potential

     for

     transformative

     change

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     decision

    -making

    :

     AI

     is

     already

     being

     used

     in

     many

     decision

    -making

     processes

    ,

     but

     its

     integration

     with

     human

     decision

    -making

     is

     likely

     to

     be

     more

     prominent

     in

     the

     future

    .

     AI

     systems

     will

     be

     used

     to

     analyze

     large

     amounts

     of

     data

     and

     make

     decisions

     based

     on

     that

     information

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     and

     societal

     impact

    :

     AI

     is

     already

     causing

     ethical

     concerns

    ,

     such

     as

     concerns

     about

     bias

     and

     privacy

    .

     As

     AI

     continues

     to

     evolve

    ,

     there

     will

     be

     a

     greater

     focus

     on

     addressing

     these

     issues

     and

     ensuring

     that

     AI

     systems

     are

    



```python
llm.shutdown()
```
