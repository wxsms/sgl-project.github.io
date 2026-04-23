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
    [2026-04-23 04:48:13] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.11it/s]


    2026-04-23 04:48:18,658 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 04:48:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.13it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.13it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.13it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]

    Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:06,  7.13it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:06,  7.13it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:06,  7.13it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:06,  7.13it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:06,  7.13it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:03, 11.32it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 18.50it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]

    Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 27.54it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:03<00:00, 37.70it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:03<00:00, 48.43it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 57.46it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 57.46it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 57.46it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 57.46it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 57.46it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 57.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.33 GB):   2%|▏         | 1/58 [00:00<00:06,  9.01it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.31 GB):   2%|▏         | 1/58 [00:00<00:06,  9.01it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=72.31 GB):   3%|▎         | 2/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.15 GB):   3%|▎         | 2/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.15 GB):   5%|▌         | 3/58 [00:00<00:05,  9.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.93 GB):   5%|▌         | 3/58 [00:00<00:05,  9.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.93 GB):   7%|▋         | 4/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.97 GB):   7%|▋         | 4/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.31 GB):   7%|▋         | 4/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.31 GB):  10%|█         | 6/58 [00:00<00:04, 10.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.31 GB):  10%|█         | 6/58 [00:00<00:04, 10.44it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.31 GB):  10%|█         | 6/58 [00:00<00:04, 10.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.31 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.31 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.30 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.44it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.30 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.13 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.13 GB):  21%|██        | 12/58 [00:01<00:03, 13.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.12 GB):  21%|██        | 12/58 [00:01<00:03, 13.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.20 GB):  21%|██        | 12/58 [00:01<00:03, 13.02it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.28 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.28 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.25 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.25 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.27 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.26 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.11it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.25 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.25 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.41it/s]Capturing num tokens (num_tokens=960 avail_mem=72.22 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.41it/s] Capturing num tokens (num_tokens=896 avail_mem=72.24 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.41it/s]Capturing num tokens (num_tokens=832 avail_mem=72.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.41it/s]Capturing num tokens (num_tokens=832 avail_mem=72.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.87it/s]Capturing num tokens (num_tokens=768 avail_mem=72.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.87it/s]Capturing num tokens (num_tokens=704 avail_mem=72.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.87it/s]Capturing num tokens (num_tokens=640 avail_mem=72.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.87it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.87it/s]Capturing num tokens (num_tokens=576 avail_mem=72.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.46it/s]Capturing num tokens (num_tokens=512 avail_mem=72.19 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.46it/s]Capturing num tokens (num_tokens=480 avail_mem=72.20 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.46it/s]Capturing num tokens (num_tokens=448 avail_mem=72.22 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.46it/s]Capturing num tokens (num_tokens=416 avail_mem=72.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.46it/s]Capturing num tokens (num_tokens=384 avail_mem=72.19 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.46it/s]Capturing num tokens (num_tokens=384 avail_mem=72.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=352 avail_mem=72.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=320 avail_mem=72.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.46it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=256 avail_mem=72.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=256 avail_mem=72.18 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.61it/s]Capturing num tokens (num_tokens=240 avail_mem=72.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.61it/s]Capturing num tokens (num_tokens=224 avail_mem=72.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.61it/s]Capturing num tokens (num_tokens=208 avail_mem=72.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.61it/s]Capturing num tokens (num_tokens=192 avail_mem=72.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.61it/s]Capturing num tokens (num_tokens=176 avail_mem=72.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.61it/s]Capturing num tokens (num_tokens=176 avail_mem=72.15 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=160 avail_mem=72.14 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=144 avail_mem=72.14 GB):  72%|███████▏  | 42/58 [00:02<00:00, 36.00it/s]

    Capturing num tokens (num_tokens=128 avail_mem=72.13 GB):  72%|███████▏  | 42/58 [00:02<00:00, 36.00it/s]Capturing num tokens (num_tokens=112 avail_mem=72.13 GB):  72%|███████▏  | 42/58 [00:02<00:00, 36.00it/s]Capturing num tokens (num_tokens=96 avail_mem=72.12 GB):  72%|███████▏  | 42/58 [00:02<00:00, 36.00it/s] Capturing num tokens (num_tokens=96 avail_mem=72.12 GB):  81%|████████  | 47/58 [00:02<00:00, 38.04it/s]Capturing num tokens (num_tokens=80 avail_mem=72.12 GB):  81%|████████  | 47/58 [00:02<00:00, 38.04it/s]Capturing num tokens (num_tokens=64 avail_mem=72.11 GB):  81%|████████  | 47/58 [00:02<00:00, 38.04it/s]Capturing num tokens (num_tokens=48 avail_mem=72.11 GB):  81%|████████  | 47/58 [00:02<00:00, 38.04it/s]Capturing num tokens (num_tokens=32 avail_mem=72.11 GB):  81%|████████  | 47/58 [00:02<00:00, 38.04it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  81%|████████  | 47/58 [00:02<00:00, 38.04it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 39.99it/s]Capturing num tokens (num_tokens=24 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 39.99it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 39.99it/s]Capturing num tokens (num_tokens=16 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 39.99it/s]Capturing num tokens (num_tokens=12 avail_mem=72.09 GB):  90%|████████▉ | 52/58 [00:02<00:00, 39.99it/s]Capturing num tokens (num_tokens=8 avail_mem=72.09 GB):  90%|████████▉ | 52/58 [00:02<00:00, 39.99it/s] Capturing num tokens (num_tokens=8 avail_mem=72.09 GB):  98%|█████████▊| 57/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB):  98%|█████████▊| 57/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB): 100%|██████████| 58/58 [00:02<00:00, 24.67it/s]


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
    Generated text: . I am from the African continent. I live in the west of the continent. I speak English and also have some knowledge of other languages. I like to travel, read, watch movies, and listen to music. I also enjoy playing tennis and hiking. I have been travelling around the world for several years and have seen so many amazing things. I have a great sense of adventure and love to explore new places. I am always happy and have a good attitude. I am a very friendly person, and I love helping others. I believe that everyone should be treated fairly and equally. I am very open-minded and consider myself to be
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposed by the joint session of Congress and approved by the vice president and signed into executive order by the president. The vice president is the head of the executive branch of the United States government and must have had at least 4 years of experience in the executive branch of the United States government before serving as president. The vice president must also be a member of the Democratic Party and have held the position of secretary of state for at least 4 years before serving as president. Does this mean that the vice president must have a background of experience in the executive branch? No, the Vice President does not need to have a background in the executive branch
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Rome
    C. London
    D. Moscow
    Answer: A
    
    The characteristic of a buffer solution is ____.
    A. pH value remains unchanged
    B. pH value is maintained at a certain value
    C. pH value fluctuates
    D. pH value changes
    Answer: B
    
    A certain student is using a microscope to observe a slide specimen. Which of the following should he do correctly?
    A. Rotate the stage lens to change the magnification
    B. Turn the coarse focus knob to adjust the focal length
    C. Turn the fine focus knob to adjust the focal length
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly evolving, and it is crucial to understand the latest developments and technologies to stay ahead of the curve. One of the most significant advancements in AI is the development of self-driving cars. These cars have become increasingly popular in recent years, and they have the potential to revolutionize transportation by reducing traffic congestion, improving safety, and reducing carbon emissions. However, there are also concerns about the safety of self-driving cars and the potential for them to be used in dangerous situations.
    To ensure the safety of self-driving cars, it is essential to have a comprehensive safety system in place. This includes features such as automatic braking, lane departure warning, and


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I have [Number of Years] years of experience in [Field of Work]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new challenges and opportunities to grow and learn. What do you enjoy doing? I enjoy [What I Enjoy Doing], and I'm always looking for new challenges
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its cuisine, including its famous croissants and its traditional French wine. The city is home to many international organizations and is a major center for business and finance. Paris is a vibrant and dynamic city with a rich history and a diverse population. Its status as the capital of France is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be an increased need for privacy and security measures to protect personal data and prevent misuse of AI systems. This could lead to more stringent regulations and standards for AI development and deployment.
    
    3. Increased focus on ethical AI: As AI systems
    


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
    Generated text:  [Name] and I'm a [Job Title] at [Company Name]. I've been working at this company for [Number of Years] years, and I enjoy [Reason for Job]. What's been your most memorable experience at this company? I'd love to hear about your background and what led you to become a [Job Title] at [Company Name]. Also, please include any relevant achievements or notable projects you've done for the company. Lastly, if you have any advice for someone starting their career in this field, please share your thoughts. Let's make history together! [Name] [Company Name] [Number of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the most populous city in France, with an estimated population of over 2 million. The city is home to the official residence of the French President, the Eiffel Tower, and is known for its historic landmarks and French cuisine. Paris is also a major international hub for business, politics, and culture, with its influence extending beyond the French borders. The city is also known for its fashion industry, with the iconic couture house Chanel being based in Paris. Paris is a global cultural and political center, and its reputation for fine cuisine, art, and architecture continues to be a major draw for travelers. While the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable, and it is difficult to predict the exact direction in which it will evolve. However, there are several possible trends that are likely to shape the AI landscape in the coming years:
    
    1. Enhanced AI: AI will continue to become more and more capable of performing tasks that were previously impossible, such as playing chess, playing the stock market, or creating personalized advertisements. This will lead to the development of new types of AI, such as cognitive agents that can understand and learn from human language, or even AI that can operate independently and make decisions based on new information.
    
    2. Increased Ethical and Legal Concerns: As


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

     [

    Level

     of

     Experience

    ]

     who

     is

     always

     ready

     to

     lend

     a

     helping

     hand

    .

     I

    've

     been

     working

     in

     the

     [

    Industry

    ]

     sector

     for

     [

    Number

     of

     Years

    ]

     years

    ,

     and

     I

    'm

     known

     for

     my

     ability

     to

     handle

     any

     challenge

     that

     comes

     my

     way

    .

     Whether

     it

    's

     helping

     someone

     out

     of

     a

     difficult

     situation

     or

     solving

     a

     complex

     problem

    ,

     I

    'm

     always

     there

     to

     lend

     a

     hand

    .

     I

    'm

     confident

     and

     knowledgeable

    ,

     and

     I

    'm

     here

     to

     help

     any

     time

     you

     need

     me

    .

     How

     about

     you

    ?

     What

     brings

     you

     here

     today

    ?

     (

    The

     character

     may

     ask

     or

     respond

     with

     "

    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     de

     Paris

    ".

     It

     is

     a

     historic

     city

     with

     a

     rich

     cultural

     heritage

    ,

     known

     for

     its

     Notre

    -D

    ame

     Cathedral

    ,

     Latin

     Quarter

    ,

     and

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     famous

     for

     its

     distinctive

     cuisine

     and

     fashion

    .

     It

     is

     a

     cosm

    opolitan

     and

     diverse

     city

    ,

     with

     French

    ,

     Spanish

    ,

     Italian

    ,

     and

     other

     cultural

     influences

    .

     Paris

     is the

     country

    's

     largest

     city

     and

     the

     most

     populous

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

     million

     people

     as

     of

     

    2

    0

    2

    1

    .

     The

     city

     has

     a

     rich

     history

    ,

     including

     the

     medieval

     town

     of

     Paris

     and

     the

     French

     Revolution

    .

     Paris

     is

     also

     a

     major

     transportation

     hub

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     increasingly

     focused

     on

     enhancing

     its

     capabilities

     to

     interact

     more

     naturally

     with

     humans

    ,

     making

     it

     more

     personal

     and

     intelligent

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

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

     ethical

     and

     transparent

    :

     With

     more

     and

     more

     ethical

     concerns

     surrounding

     AI

     systems

    ,

     there

     will

     be

     a

     push

     towards

     making

     them

     more

     transparent

     and

     accountable

    .

     This

     means

     that

     we

     will

     see

     a

     greater

     emphasis

     on

     AI

     ethics

    ,

     as

     well

     as

     greater

     transparency

     and

     accountability

     in

     how

     it

     is

     used

     and

     operated

    .
    


    2

    .

     AI

     will

     become

     more

     personalized

    :

     With

     the

     rise

     of

     big

     data

     and

     machine

     learning

    ,

     we

     will

     see

     more

     personalized

     AI

     systems

     that

     are

     able

     to

     understand

     and

    



```python
llm.shutdown()
```
