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
    [2026-04-24 17:57:20] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]


    2026-04-24 17:57:24,634 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 17:57:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.18s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.18s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.09it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.09it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.09it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.28it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.28it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.28it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:05,  9.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:05,  9.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:05,  9.07it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04, 10.57it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04, 10.57it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04, 10.57it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04, 10.57it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:03, 13.27it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:03, 13.27it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:03, 13.27it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:03, 13.27it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 15.01it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 15.01it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 15.01it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 15.01it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 17.05it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 17.05it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 17.05it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 17.05it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.82it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.82it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.82it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:01, 18.82it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 19.89it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 19.89it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 19.89it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 19.89it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 21.10it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 21.10it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 21.10it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 21.10it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 23.09it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 23.09it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 23.09it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 23.09it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 23.02it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 23.02it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 23.02it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 23.02it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.59it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.59it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.59it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.59it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 25.95it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 26.58it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 26.58it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 26.58it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 26.58it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 26.43it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 26.43it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 26.43it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 26.43it/s]

    Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 26.43it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 28.13it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 28.13it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 28.13it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 28.13it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=40.90 GB):   2%|▏         | 1/58 [00:00<00:12,  4.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.87 GB):   2%|▏         | 1/58 [00:00<00:12,  4.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=40.87 GB):   3%|▎         | 2/58 [00:00<00:12,  4.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.87 GB):   3%|▎         | 2/58 [00:00<00:12,  4.33it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=40.87 GB):   5%|▌         | 3/58 [00:00<00:11,  4.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.86 GB):   5%|▌         | 3/58 [00:00<00:11,  4.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.86 GB):   7%|▋         | 4/58 [00:00<00:11,  4.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.87 GB):   7%|▋         | 4/58 [00:00<00:11,  4.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=40.87 GB):   9%|▊         | 5/58 [00:01<00:11,  4.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.86 GB):   9%|▊         | 5/58 [00:01<00:11,  4.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.86 GB):  10%|█         | 6/58 [00:01<00:10,  4.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.86 GB):  10%|█         | 6/58 [00:01<00:10,  4.91it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=40.86 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.86 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.86 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.86 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.15it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=40.86 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.85 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.85 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.85 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.52it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=40.85 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.85 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.85 GB):  21%|██        | 12/58 [00:02<00:07,  5.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.84 GB):  21%|██        | 12/58 [00:02<00:07,  5.90it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=40.84 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.84 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.84 GB):  24%|██▍       | 14/58 [00:02<00:07,  6.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.84 GB):  24%|██▍       | 14/58 [00:02<00:07,  6.11it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=40.84 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.83 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.83 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.83 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.47it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=40.83 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.83 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.83 GB):  31%|███       | 18/58 [00:03<00:06,  6.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.82 GB):  31%|███       | 18/58 [00:03<00:06,  6.67it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=40.82 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.82 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.82 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.80 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.18it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=40.80 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.36it/s]Capturing num tokens (num_tokens=960 avail_mem=40.81 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.36it/s] Capturing num tokens (num_tokens=960 avail_mem=40.81 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.50it/s]Capturing num tokens (num_tokens=896 avail_mem=40.81 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.50it/s]

    Capturing num tokens (num_tokens=896 avail_mem=40.81 GB):  40%|███▉      | 23/58 [00:03<00:04,  7.90it/s]Capturing num tokens (num_tokens=832 avail_mem=40.81 GB):  40%|███▉      | 23/58 [00:03<00:04,  7.90it/s]Capturing num tokens (num_tokens=832 avail_mem=40.81 GB):  41%|████▏     | 24/58 [00:03<00:04,  8.06it/s]Capturing num tokens (num_tokens=768 avail_mem=40.80 GB):  41%|████▏     | 24/58 [00:03<00:04,  8.06it/s]

    Capturing num tokens (num_tokens=768 avail_mem=40.80 GB):  43%|████▎     | 25/58 [00:04<00:03,  8.47it/s]Capturing num tokens (num_tokens=704 avail_mem=40.80 GB):  43%|████▎     | 25/58 [00:04<00:03,  8.47it/s]Capturing num tokens (num_tokens=704 avail_mem=40.80 GB):  45%|████▍     | 26/58 [00:04<00:03,  8.51it/s]Capturing num tokens (num_tokens=640 avail_mem=40.80 GB):  45%|████▍     | 26/58 [00:04<00:03,  8.51it/s]

    Capturing num tokens (num_tokens=640 avail_mem=40.80 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.84it/s]Capturing num tokens (num_tokens=576 avail_mem=40.80 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.84it/s]Capturing num tokens (num_tokens=576 avail_mem=40.80 GB):  48%|████▊     | 28/58 [00:04<00:03,  9.05it/s]Capturing num tokens (num_tokens=512 avail_mem=40.78 GB):  48%|████▊     | 28/58 [00:04<00:03,  9.05it/s]

    Capturing num tokens (num_tokens=512 avail_mem=40.78 GB):  50%|█████     | 29/58 [00:04<00:03,  9.12it/s]Capturing num tokens (num_tokens=480 avail_mem=40.79 GB):  50%|█████     | 29/58 [00:04<00:03,  9.12it/s]Capturing num tokens (num_tokens=480 avail_mem=40.79 GB):  52%|█████▏    | 30/58 [00:04<00:03,  9.27it/s]Capturing num tokens (num_tokens=448 avail_mem=40.79 GB):  52%|█████▏    | 30/58 [00:04<00:03,  9.27it/s]

    Capturing num tokens (num_tokens=448 avail_mem=40.79 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.35it/s]Capturing num tokens (num_tokens=416 avail_mem=40.79 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.35it/s]Capturing num tokens (num_tokens=416 avail_mem=40.79 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.40it/s]Capturing num tokens (num_tokens=384 avail_mem=40.79 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.40it/s]Capturing num tokens (num_tokens=352 avail_mem=40.78 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.40it/s]

    Capturing num tokens (num_tokens=352 avail_mem=40.78 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.36it/s]Capturing num tokens (num_tokens=320 avail_mem=40.78 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.36it/s]Capturing num tokens (num_tokens=288 avail_mem=40.77 GB):  59%|█████▊    | 34/58 [00:05<00:02, 10.36it/s]Capturing num tokens (num_tokens=288 avail_mem=40.77 GB):  62%|██████▏   | 36/58 [00:05<00:01, 11.50it/s]Capturing num tokens (num_tokens=256 avail_mem=40.77 GB):  62%|██████▏   | 36/58 [00:05<00:01, 11.50it/s]

    Capturing num tokens (num_tokens=240 avail_mem=39.58 GB):  62%|██████▏   | 36/58 [00:05<00:01, 11.50it/s]Capturing num tokens (num_tokens=240 avail_mem=39.58 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.66it/s]Capturing num tokens (num_tokens=224 avail_mem=39.58 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.66it/s]

    Capturing num tokens (num_tokens=208 avail_mem=39.57 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.66it/s]Capturing num tokens (num_tokens=208 avail_mem=39.57 GB):  69%|██████▉   | 40/58 [00:05<00:02,  8.56it/s]Capturing num tokens (num_tokens=192 avail_mem=40.73 GB):  69%|██████▉   | 40/58 [00:05<00:02,  8.56it/s]

    Capturing num tokens (num_tokens=192 avail_mem=40.73 GB):  71%|███████   | 41/58 [00:05<00:02,  8.48it/s]Capturing num tokens (num_tokens=176 avail_mem=40.73 GB):  71%|███████   | 41/58 [00:05<00:02,  8.48it/s]Capturing num tokens (num_tokens=176 avail_mem=40.73 GB):  72%|███████▏  | 42/58 [00:05<00:01,  8.43it/s]Capturing num tokens (num_tokens=160 avail_mem=39.74 GB):  72%|███████▏  | 42/58 [00:05<00:01,  8.43it/s]

    Capturing num tokens (num_tokens=160 avail_mem=39.74 GB):  74%|███████▍  | 43/58 [00:06<00:01,  7.90it/s]Capturing num tokens (num_tokens=144 avail_mem=39.73 GB):  74%|███████▍  | 43/58 [00:06<00:01,  7.90it/s]Capturing num tokens (num_tokens=144 avail_mem=39.73 GB):  76%|███████▌  | 44/58 [00:06<00:01,  7.53it/s]Capturing num tokens (num_tokens=128 avail_mem=39.73 GB):  76%|███████▌  | 44/58 [00:06<00:01,  7.53it/s]

    Capturing num tokens (num_tokens=128 avail_mem=39.73 GB):  78%|███████▊  | 45/58 [00:06<00:01,  7.32it/s]Capturing num tokens (num_tokens=112 avail_mem=40.72 GB):  78%|███████▊  | 45/58 [00:06<00:01,  7.32it/s]Capturing num tokens (num_tokens=112 avail_mem=40.72 GB):  79%|███████▉  | 46/58 [00:06<00:01,  7.27it/s]Capturing num tokens (num_tokens=96 avail_mem=39.79 GB):  79%|███████▉  | 46/58 [00:06<00:01,  7.27it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=39.79 GB):  81%|████████  | 47/58 [00:06<00:01,  7.10it/s]Capturing num tokens (num_tokens=80 avail_mem=39.79 GB):  81%|████████  | 47/58 [00:06<00:01,  7.10it/s]Capturing num tokens (num_tokens=80 avail_mem=39.79 GB):  83%|████████▎ | 48/58 [00:06<00:01,  6.94it/s]Capturing num tokens (num_tokens=64 avail_mem=39.78 GB):  83%|████████▎ | 48/58 [00:06<00:01,  6.94it/s]

    Capturing num tokens (num_tokens=64 avail_mem=39.78 GB):  84%|████████▍ | 49/58 [00:06<00:01,  7.06it/s]Capturing num tokens (num_tokens=48 avail_mem=40.71 GB):  84%|████████▍ | 49/58 [00:06<00:01,  7.06it/s]Capturing num tokens (num_tokens=48 avail_mem=40.71 GB):  86%|████████▌ | 50/58 [00:07<00:01,  7.12it/s]Capturing num tokens (num_tokens=32 avail_mem=39.85 GB):  86%|████████▌ | 50/58 [00:07<00:01,  7.12it/s]

    Capturing num tokens (num_tokens=32 avail_mem=39.85 GB):  88%|████████▊ | 51/58 [00:07<00:01,  6.93it/s]Capturing num tokens (num_tokens=28 avail_mem=39.84 GB):  88%|████████▊ | 51/58 [00:07<00:01,  6.93it/s]Capturing num tokens (num_tokens=28 avail_mem=39.84 GB):  90%|████████▉ | 52/58 [00:07<00:00,  7.03it/s]Capturing num tokens (num_tokens=24 avail_mem=40.70 GB):  90%|████████▉ | 52/58 [00:07<00:00,  7.03it/s]

    Capturing num tokens (num_tokens=24 avail_mem=40.70 GB):  91%|█████████▏| 53/58 [00:07<00:00,  7.22it/s]Capturing num tokens (num_tokens=20 avail_mem=39.91 GB):  91%|█████████▏| 53/58 [00:07<00:00,  7.22it/s]Capturing num tokens (num_tokens=20 avail_mem=39.91 GB):  93%|█████████▎| 54/58 [00:07<00:00,  7.03it/s]Capturing num tokens (num_tokens=16 avail_mem=39.91 GB):  93%|█████████▎| 54/58 [00:07<00:00,  7.03it/s]

    Capturing num tokens (num_tokens=16 avail_mem=39.91 GB):  95%|█████████▍| 55/58 [00:07<00:00,  6.95it/s]Capturing num tokens (num_tokens=12 avail_mem=39.90 GB):  95%|█████████▍| 55/58 [00:07<00:00,  6.95it/s]Capturing num tokens (num_tokens=12 avail_mem=39.90 GB):  97%|█████████▋| 56/58 [00:07<00:00,  7.32it/s]Capturing num tokens (num_tokens=8 avail_mem=40.69 GB):  97%|█████████▋| 56/58 [00:07<00:00,  7.32it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=40.69 GB):  98%|█████████▊| 57/58 [00:08<00:00,  7.39it/s]Capturing num tokens (num_tokens=4 avail_mem=39.96 GB):  98%|█████████▊| 57/58 [00:08<00:00,  7.39it/s]Capturing num tokens (num_tokens=4 avail_mem=39.96 GB): 100%|██████████| 58/58 [00:08<00:00,  7.13it/s]Capturing num tokens (num_tokens=4 avail_mem=39.96 GB): 100%|██████████| 58/58 [00:08<00:00,  7.09it/s]


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
    Generated text:  Anna and I'm 12 years old.
    As you can see, I'm a girl, but I can't have children.
    What do I need to do?
    I have not been given any medical advice. It's really hard for me to believe this. Does this sound like the child I am?
    I am a child who has no knowledge of things that would actually happen to a child like this. But I did not have any other children. Why does this happen to me?
    I'm not asking about sex, but about the baby. But as you can see, I don't have any children and it's really hard for
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He likes the idea of having  $n$  bases, but he is also afraid that this will lead to lots of unnecessary military activity. Each base uses $100$ tons of explosives. The cost to build the first base is $300$ million dollars. The cost to build the $n$ th base is $300(n-1)$ million dollars. The president also wants to ensure that the total cost of explosives for all bases does not exceed $1$ billion dollars. How many bases should the president have?
    
    To determine the maximum number of bases
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Paris
    C. Rennes
    D. Geneva
    Answer:
    
    A
    
    According to the structure of the capital city, which of the following cities is a province-level administrative body? 
    A. Shanghai
    B. Jilin
    C. Hohhot
    D. Xinjiang
    Answer:
    
    C
    
    The main carrier of the genetic material in the human body is ____.
    A. DNA
    B. RNA
    C. Cholesterol
    D. Protein
    Answer:
    
    A
    
    Under the same temperature and pressure, for a pure substance, the greater the volume of the gas, the ____ its density
    ===============================
    Prompt: The future of AI is
    Generated text:  digital
    
    In the age of digital technologies, the future of AI is digital. Big data, machine learning and artificial intelligence are the three key pillars to the future of AI. The need to have access to large amounts of data is the driving force behind the rise of AI, as well as the need for more data to be collected and analyzed. The only way to extract value from this data is to utilize AI. The benefits of AI are tangible and are evident in a number of industries. AI technology has been used in a variety of applications such as customer support, fraud detection, medical diagnosis and more. As the need for AI continues to


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [Describe your personality traits here]. I enjoy [What you like to do] and I'm always looking for new experiences and challenges. I'm always eager to learn and grow, and I'm always looking for ways to improve myself. I'm a [Type of Person] who is [Describe your personality traits here]. I'm always looking for new experiences and challenges. I'm always eager to learn and grow, and I'm always looking for ways to improve myself. I'm a [Type of Person]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Riviera, a popular tourist destination known for its beaches, wine, and luxury resorts. Overall, Paris is a vibrant and diverse city with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more sophisticated ways, such as in personalized medicine, drug discovery, and patient monitoring.
    
    3. Increased use of AI
    


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
    Generated text:  [Name], and I'm a professional in the technology industry. I have a strong background in software development and have experience working on large-scale projects with a team of experienced developers. I am also a skilled communicator, and I enjoy collaborating with cross-functional teams to solve complex problems and meet tight deadlines. My passion is to drive innovation and create products that make a difference in the world. I am always looking for new ways to improve my skills and stay up-to-date with the latest technologies. Thank you. I hope you enjoy our conversation. [Name] [Phone number] [Email address] [LinkedIn profile] [Twitter handle] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The French word "Paris" literally means "City of the Popes." The city, located on the River Seine, is the second most populous city in Europe and is also the second most populous urban area in the world. It's known for its medieval architecture, vibrant culture, and annual festival celebrations, including the World Cup. Paris is renowned for its fashion and gastronomy, as well as its status as a major transportation hub. It's home to some of Europe's most famous landmarks, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Champs-Élysées. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly exciting and promising, with new applications and improvements expected to continue to grow and evolve. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI technology becomes more advanced, there will be a growing demand for ethical considerations in its development and deployment. This will include issues such as bias, privacy, and transparency.
    
    2. More AI-driven innovations: The integration of AI into various industries and applications is expected to increase, leading to more sophisticated and personalized solutions.
    
    3. Greater use of AI for healthcare: AI can help improve the accuracy and speed of diagnoses, personalized treatments, and drug discovery.
    
    4. Enhanced


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

    'm

     a

     [

    occupation

    ]

     who

     has

     always

     been

     passionate

     about

     [

    job

     title

     or

     hobby

    /

    interest

    ].

     I

     believe

     in

     [

    core

     belief

     or

     philosophy

    ],

     and

     I

     strive

     to

     [

    something

     specific

    ,

     such

     as

     "

    be

     more

     responsible

     for

     my

     health

    ",

     "

    create

     more

     positive

     emotions

    ",

     "

    help

     others

    ",

     etc

    .

    ].

     I

     am

     always

     open

     to

     learning

     and

     always

     looking

     for

     opportunities

     to

     grow

     and

     improve

     myself

    .

     I

     am

     a

     [

    character

     trait

     or

     quality

    ],

     and

     I

     am

     always

     eager

     to

     learn

     more

     about

     myself

     and

     others

    .

     What

    's

     your

     name

    ,

     and

     what

    's

     your

     occupation

     or

     profession

    ?

     [

    Name

    ]:

     Hello

    !

     I

    'm

     [

    Name

    ],

     a

     [

    occupation

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     known

     for

     its

     iconic

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

     Other

     well

    -known

     attractions

     include

     the

     Palace

     of

     Vers

    ailles

    ,

     Ch

    amps

    -

    É

    lys

    ées

    ,

     the

     Op

    éra

    ,

     and

     the

     Latin

     Quarter

    .

     Paris

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     history

    ,

     culture

    ,

     and

     cuisine

    .

     It

     is

     also

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

     capital

     city

     of

     France

     is

     located

     in

     the

     region

     of

     Burg

    undy

    .

     Paris

     is

     often

     referred

     to

     as

     "

    La

     Ville

     Fl

    ott

    ante

    "

     which

     means

     "

    floating

     city

    "

     in

     French

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     range

     of

     potential

     trends

     that

     reflect

     the

     advances

     and

     developments

     in

     the

     field

     over

     time

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     use

     of

     machine

     learning

    :

     As

     AI

     technology

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     more

     machine

     learning

     algorithms

     being

     used

     to

     automate

     processes

    ,

     improve

     decision

    -making

    ,

     and

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    .
    


    2

    .

     Enhanced

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

     likely

     to

     become

     even

     more

     accurate

     and

     reliable

    .

     This

     could

     mean

     a

     shift

     from

     traditional

     algorithms

     to

     more

     complex

     models

     that

     can

     handle

     a

     wider

     range

     of

     inputs

     and

     produce

     more

     meaningful

     results

    .
    


    3

    .

     Greater

     integration

    



```python
llm.shutdown()
```
