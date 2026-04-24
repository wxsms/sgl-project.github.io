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
    [2026-04-24 15:31:16] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.98it/s]


    2026-04-24 15:31:20,937 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 15:31:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.37it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:15,  3.26it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:15,  3.26it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:15,  3.26it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:15,  3.26it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:02<00:15,  3.26it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.39it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]

    Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:03<00:03, 11.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 17.49it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:03<00:01, 25.38it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s] Compiling num tokens (num_tokens=80):  64%|██████▍   | 37/58 [00:03<00:00, 36.85it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 51.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.47 GB):   2%|▏         | 1/58 [00:00<00:06,  9.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.28 GB):   2%|▏         | 1/58 [00:00<00:06,  9.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.28 GB):   3%|▎         | 2/58 [00:00<00:06,  8.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.29 GB):   3%|▎         | 2/58 [00:00<00:06,  8.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.29 GB):   5%|▌         | 3/58 [00:00<00:06,  8.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.29 GB):   5%|▌         | 3/58 [00:00<00:06,  8.04it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.29 GB):   7%|▋         | 4/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.28 GB):   7%|▋         | 4/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.35 GB):   7%|▋         | 4/58 [00:00<00:06,  8.22it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.35 GB):  10%|█         | 6/58 [00:00<00:05,  8.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.35 GB):  10%|█         | 6/58 [00:00<00:05,  8.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.35 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.35 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.82it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=59.27 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.27 GB):  16%|█▌        | 9/58 [00:00<00:05,  9.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.41 GB):  16%|█▌        | 9/58 [00:00<00:05,  9.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.41 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.66it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.41 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.66it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.27 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.27 GB):  21%|██        | 12/58 [00:01<00:04, 11.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.48 GB):  21%|██        | 12/58 [00:01<00:04, 11.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.48 GB):  21%|██        | 12/58 [00:01<00:04, 11.01it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.48 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.26 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.26 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.26 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.53 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.06it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.53 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.53 GB):  31%|███       | 18/58 [00:01<00:03, 12.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.25 GB):  31%|███       | 18/58 [00:01<00:03, 12.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.58 GB):  31%|███       | 18/58 [00:01<00:03, 12.56it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.58 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.57 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.54it/s]Capturing num tokens (num_tokens=960 avail_mem=59.25 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.54it/s] Capturing num tokens (num_tokens=960 avail_mem=59.25 GB):  38%|███▊      | 22/58 [00:01<00:02, 13.43it/s]Capturing num tokens (num_tokens=896 avail_mem=58.65 GB):  38%|███▊      | 22/58 [00:01<00:02, 13.43it/s]Capturing num tokens (num_tokens=832 avail_mem=59.24 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.43it/s]

    Capturing num tokens (num_tokens=832 avail_mem=59.24 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.64it/s]Capturing num tokens (num_tokens=768 avail_mem=58.71 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.64it/s]Capturing num tokens (num_tokens=704 avail_mem=58.71 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.64it/s]Capturing num tokens (num_tokens=704 avail_mem=58.71 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.65it/s]Capturing num tokens (num_tokens=640 avail_mem=59.24 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.65it/s]Capturing num tokens (num_tokens=576 avail_mem=58.74 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.65it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.74 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.65it/s]Capturing num tokens (num_tokens=512 avail_mem=58.73 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.65it/s]Capturing num tokens (num_tokens=480 avail_mem=59.18 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.65it/s]Capturing num tokens (num_tokens=480 avail_mem=59.18 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.58it/s]Capturing num tokens (num_tokens=448 avail_mem=58.77 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.58it/s]

    Capturing num tokens (num_tokens=416 avail_mem=59.23 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.58it/s]Capturing num tokens (num_tokens=416 avail_mem=59.23 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.67it/s]Capturing num tokens (num_tokens=384 avail_mem=58.80 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.67it/s]Capturing num tokens (num_tokens=352 avail_mem=59.23 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.67it/s]Capturing num tokens (num_tokens=352 avail_mem=59.23 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=320 avail_mem=58.82 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.00it/s]

    Capturing num tokens (num_tokens=288 avail_mem=59.22 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=288 avail_mem=59.22 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.98it/s]Capturing num tokens (num_tokens=256 avail_mem=58.85 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.98it/s]Capturing num tokens (num_tokens=240 avail_mem=59.21 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.98it/s]

    Capturing num tokens (num_tokens=240 avail_mem=59.21 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.15it/s]Capturing num tokens (num_tokens=224 avail_mem=58.88 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.15it/s]Capturing num tokens (num_tokens=208 avail_mem=59.17 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.15it/s]Capturing num tokens (num_tokens=208 avail_mem=59.17 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.71it/s]Capturing num tokens (num_tokens=192 avail_mem=59.24 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.71it/s]Capturing num tokens (num_tokens=176 avail_mem=58.94 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.71it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.94 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.81it/s]Capturing num tokens (num_tokens=160 avail_mem=59.20 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.81it/s]Capturing num tokens (num_tokens=144 avail_mem=58.96 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.81it/s]Capturing num tokens (num_tokens=144 avail_mem=58.96 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.29it/s]Capturing num tokens (num_tokens=128 avail_mem=59.19 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.29it/s]Capturing num tokens (num_tokens=112 avail_mem=58.98 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.29it/s]

    Capturing num tokens (num_tokens=96 avail_mem=59.18 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.29it/s] Capturing num tokens (num_tokens=96 avail_mem=59.18 GB):  81%|████████  | 47/58 [00:03<00:00, 17.60it/s]Capturing num tokens (num_tokens=80 avail_mem=59.18 GB):  81%|████████  | 47/58 [00:03<00:00, 17.60it/s]Capturing num tokens (num_tokens=64 avail_mem=59.17 GB):  81%|████████  | 47/58 [00:03<00:00, 17.60it/s]Capturing num tokens (num_tokens=64 avail_mem=59.17 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.51it/s]Capturing num tokens (num_tokens=48 avail_mem=59.17 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.51it/s]

    Capturing num tokens (num_tokens=32 avail_mem=59.05 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.51it/s]Capturing num tokens (num_tokens=28 avail_mem=59.04 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.51it/s]Capturing num tokens (num_tokens=28 avail_mem=59.04 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=24 avail_mem=59.15 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=20 avail_mem=59.14 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.87it/s]Capturing num tokens (num_tokens=16 avail_mem=59.14 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.87it/s]

    Capturing num tokens (num_tokens=16 avail_mem=59.14 GB):  95%|█████████▍| 55/58 [00:03<00:00, 21.06it/s]Capturing num tokens (num_tokens=12 avail_mem=59.13 GB):  95%|█████████▍| 55/58 [00:03<00:00, 21.06it/s]Capturing num tokens (num_tokens=8 avail_mem=59.10 GB):  95%|█████████▍| 55/58 [00:03<00:00, 21.06it/s] Capturing num tokens (num_tokens=4 avail_mem=59.04 GB):  95%|█████████▍| 55/58 [00:03<00:00, 21.06it/s]Capturing num tokens (num_tokens=4 avail_mem=59.04 GB): 100%|██████████| 58/58 [00:04<00:00, 22.50it/s]Capturing num tokens (num_tokens=4 avail_mem=59.04 GB): 100%|██████████| 58/58 [00:04<00:00, 14.39it/s]


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
    Generated text:  Rafał and I am a full time student. I want to be a doctor. 
    
    I have been studying on the internet for several weeks about the anatomy and physiology of the human body and I have come across a lot of interesting things.
    
    The main problem I have is how to keep my weight down. I'm currently 178 cm tall, I weigh 145 pounds and have a BMI of 20.5. I have a diet of 2 meals a day. I'm still getting quite a few calories but I'm not getting them out of my body very quickly.
    
    I have been following a vegan
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of civil servant.
    A. Correct
    B. Incorrect
    Answer:
    A
    
    Before entering a confined space, the work supervisor should clearly explain the ____ to all personnel entering the confined space.
    A. emergency escape routes
    B. emergency response procedures
    C. safety precautions
    D. emergency measures
    Answer:
    C
    
    Male, 65 years old, with a history of type 2 diabetes for 15 years. The family history of coronary heart disease. A diagnosis of 'Type 2 Diabetes Mellitus, Hyperlipidemia, Coronary Heart Disease' is made, and the nurse should advise the patient to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the largest city in Europe. It is located in the north of the country, and sits on the River Seine in the Paris Basin. The city centre is a part of the Square of the City, which is a large, open area with a large number of buildings. The city centre is known as the “Cité de la Région” (City of the Region). On the south of the city centre, a large park surrounds the city.
    Paris is the capital of France. The area of the city centre is 7.346 square kilometers, including the squares of the city.
    The city centre is
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be a convergence of several technologies, including machine learning, blockchain, and blockchain-based smart contracts. These technologies could lead to the creation of a new type of data governance system that ensures transparency, accountability, and security in the data used in AI systems. While these technologies are still in the early stages of development, their potential is vast and we can expect to see them widely adopted in the coming years. As we continue to integrate these technologies into AI systems, it will be important to ensure that they are used responsibly and transparently. By doing so, we can ensure that AI systems are safe, reliable, and equitable. This will


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, and it is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a cultural and political center of France and a major tourist destination. The city is home to many museums, art galleries, and restaurants, and it is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more efficient and effective AI systems that can better understand and respond to human needs and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and
    


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
    Generated text:  [Name]. I'm a [Occupation] who has always loved [What you do best] since childhood. I've always been curious about the world and eager to learn new things. I'm passionate about [What you believe in]. I've always felt that [What you believe in] has the potential to make the world a better place. I'm always eager to share my ideas and experiences with others. I'm confident that I can make a difference, and I'm excited to contribute to the world in a meaningful way. Thank you for having me. What is the occupation of the character?
    The occupation of the character is a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, located on the Île de la Cité, on the River Seine, and is the country's cultural, commercial, and financial center. The city is home to many iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and the Arc de Triomphe, and has a rich history dating back to ancient times. Paris is also known for its fashion, music, and cuisine. The city is a hub for creativity and innovation, and has played a significant role in the history and development of France. Despite its size, Paris is a beautiful,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of technological advancements, shifts in societal values, and evolving ethical and regulatory frameworks. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: With the increasing availability of large amounts of medical data, AI is likely to play a more significant role in healthcare in the future. AI-powered diagnostic tools, personalized treatment plans, and predictive analytics are expected to improve patient outcomes and reduce costs.
    
    2. More autonomous vehicles: Self-driving cars are already on the roads, and it's possible that we will see more of them in the future. However, autonomous vehicles will continue to evolve and


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

    'm

     a

     [

    What

     profession

     or

     field

     do

     you

     work

     in

    ?

    ].

     I

    'm

     currently

     [

    What

     status

     or

     occupation

     you

     hold

    ?

    ].

     I

    'm

     [

    What

     is

     one

     thing

     you

     are

     passionate

     about

     or

     enjoy

     doing

    ?

    ].

     I

     like

     [

    What

     is

     your

     hobby

     or

     interest

     that

     you

     enjoy

     doing

    ?

    ].

     I

     enjoy

     spending

     my

     free

     time

     [

    What

     is

     something

     you

     like

     to

     do

    ,

     such

     as

     reading

    ,

     playing

     sports

    ,

     or

     exploring

     new

     places

    ?

    ].

     I

    'm

     [

    How

     about

     you

    ,

     [

    What

     is

     your

     height

    ,

     weight

    ,

     age

    ,

     etc

    .?

    ]]

    ?

     I

    'm

     a

     [

    What

     is

     your

     nationality

    ,

     language

    ,

     or

     culture

    ?

    ].

     I

    'm

     a

    
    
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

     its

     capital

    .

     It

     is

     located

     in

     the

     centre

     of

     the

     country

    ,

     in

     the

     Î

    le

     de

     la

     C

    ité

     on

     the

     Se

    ine

     River

    .

     The

     city

     is

     known

     for

     its

     historic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     rich

     cultural

     heritage

     and

     annual

     festivals

     such

     as

     the

     E

    ly

    see

     Festival

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     is

     also

     a

     major

     financial

     and

     business

     center

    ,

     hosting

     numerous

     high

    -profile

     events

     and

     attracting

     millions

     of

     visitors

     each

     year

    .

     Despite

     its

     importance

    ,

     Paris

     is

     also

     known

     for

     its

     challenging

     living

     conditions

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     in

     exciting

     and

     unpredictable

     ways

    ,

     and

     there

     are

     several

     potential

     trends

     that

     could

     play

     a

     significant

     role

     in

     shaping

     the

     direction

     of

     AI

     development

     in

     the

     years

     to

     come

    .
    


    One

     trend

     that

     is

     likely

     to

     continue

     is

     the

     increasing

     use

     of

     AI

     in

     various

     industries

    ,

     from

     healthcare

     to

     finance

     to

     transportation

    .

     AI

     is

     already

     being

     used

     to

     improve

     productivity

     and

     efficiency

     in

     a

     variety

     of

     sectors

    ,

     and

     it

     is

     expected

     to

     play

     an

     even

     larger

     role

     in

     these

     areas

     in

     the

     future

    .
    


    Another

     trend

     is

     the

     increasing

     integration

     of

     AI

     with

     other

     technologies

    ,

     such

     as

     blockchain

     and

     artificial

     neural

     networks

    .

     This

     integration

     could

     lead

     to

     new

     forms

     of

     AI

     that

     are

     even

     more

     advanced

     and

    



```python
llm.shutdown()
```
