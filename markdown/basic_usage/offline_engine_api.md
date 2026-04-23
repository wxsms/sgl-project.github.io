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
    [2026-04-23 04:09:25] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]


    2026-04-23 04:09:30,729 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 04:09:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:09,  1.24s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:09,  1.24s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:03<01:09,  1.24s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:27,  1.98it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:27,  1.98it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:27,  1.98it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:15,  3.42it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:15,  3.42it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:15,  3.42it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:15,  3.42it/s]

    Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:15,  3.42it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:06,  7.05it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:06,  7.05it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:06,  7.05it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:06,  7.05it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:06,  7.05it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:06,  7.05it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]

    Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:03<00:03, 12.25it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:01, 19.18it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 46.36it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 51.22it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 51.22it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 51.22it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 51.22it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 51.22it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 51.22it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 51.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.17 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.16 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.16 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.16 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.16 GB):   9%|▊         | 5/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.16 GB):   9%|▊         | 5/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.16 GB):   9%|▊         | 5/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):   9%|▊         | 5/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.38it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.56it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.29it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=960 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.12it/s] Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=832 avail_mem=116.10 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.12it/s]Capturing num tokens (num_tokens=832 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.94it/s]Capturing num tokens (num_tokens=768 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.94it/s]Capturing num tokens (num_tokens=704 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.94it/s]

    Capturing num tokens (num_tokens=640 avail_mem=116.09 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.94it/s]Capturing num tokens (num_tokens=640 avail_mem=116.09 GB):  47%|████▋     | 27/58 [00:01<00:01, 15.51it/s]Capturing num tokens (num_tokens=576 avail_mem=116.09 GB):  47%|████▋     | 27/58 [00:01<00:01, 15.51it/s]Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  47%|████▋     | 27/58 [00:01<00:01, 15.51it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:01<00:01, 15.51it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:01<00:01, 15.51it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  47%|████▋     | 27/58 [00:01<00:01, 15.51it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.80it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.80it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.80it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.80it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.80it/s]Capturing num tokens (num_tokens=256 avail_mem=115.91 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.80it/s]Capturing num tokens (num_tokens=256 avail_mem=115.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.59it/s]Capturing num tokens (num_tokens=240 avail_mem=115.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.59it/s]Capturing num tokens (num_tokens=224 avail_mem=115.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.59it/s]Capturing num tokens (num_tokens=208 avail_mem=115.90 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.59it/s]Capturing num tokens (num_tokens=192 avail_mem=115.90 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.59it/s]Capturing num tokens (num_tokens=192 avail_mem=115.90 GB):  71%|███████   | 41/58 [00:01<00:00, 27.36it/s]Capturing num tokens (num_tokens=176 avail_mem=115.71 GB):  71%|███████   | 41/58 [00:01<00:00, 27.36it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.14 GB):  71%|███████   | 41/58 [00:01<00:00, 27.36it/s]Capturing num tokens (num_tokens=144 avail_mem=117.95 GB):  71%|███████   | 41/58 [00:01<00:00, 27.36it/s]Capturing num tokens (num_tokens=128 avail_mem=117.95 GB):  71%|███████   | 41/58 [00:02<00:00, 27.36it/s]Capturing num tokens (num_tokens=128 avail_mem=117.95 GB):  78%|███████▊  | 45/58 [00:02<00:00, 21.80it/s]Capturing num tokens (num_tokens=112 avail_mem=117.95 GB):  78%|███████▊  | 45/58 [00:02<00:00, 21.80it/s]Capturing num tokens (num_tokens=96 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 21.80it/s] Capturing num tokens (num_tokens=80 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 21.80it/s]

    Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:02<00:00, 21.80it/s]Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.21it/s]Capturing num tokens (num_tokens=48 avail_mem=117.94 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.21it/s]Capturing num tokens (num_tokens=32 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.21it/s]Capturing num tokens (num_tokens=28 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.21it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.21it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.12it/s]Capturing num tokens (num_tokens=20 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.12it/s]Capturing num tokens (num_tokens=16 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.12it/s]

    Capturing num tokens (num_tokens=12 avail_mem=117.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.12it/s]Capturing num tokens (num_tokens=8 avail_mem=117.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.12it/s] Capturing num tokens (num_tokens=8 avail_mem=117.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 28.19it/s]Capturing num tokens (num_tokens=4 avail_mem=117.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 28.19it/s]Capturing num tokens (num_tokens=4 avail_mem=117.91 GB): 100%|██████████| 58/58 [00:02<00:00, 23.74it/s]


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
    Generated text:  Julia and I'm a 4th year research assistant at The University of Manchester. I'm interested in prehistoric human settlements, archaeology, paleoecology, and human evolution. I'm particularly drawn to the study of adaptation to extreme environments, and particularly those in the tropical rainforest. Prior to coming to the University of Manchester, I worked at the Natural History Museum in London, where I helped prepare the museum's collection for display in the new planetarium. I also worked in the environmental science department at University College London, where I was involved in research on the impact of climate change on tropical forests. I am currently working
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a 10-year term. After the election of a new president, the president will serve 5 years as president before serving 4 years as vice president. If the president decides to serve 6 years as president, how many years will the president serve as vice president?
    
    Let's break down the problem step by step.
    
    1. The president's term as president is 10 years.
    2. The president will serve 5 years as president before serving 4 years as vice president.
    3. If the president decides to serve 6 years as president, then he will need to serve an additional year as vice president.
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. 
    A. Paris
    B. London
    C. New York
    D. Tokyo
    Answer:
    
    A
    
    Which of the following equations is correct?
    A. 3a+4a=7a^2
    B. 3a+2b=5ab
    C. 2(a+1)=2a+2
    D. 2(a-b)=2a-2b
    Answer:
    
    C
    
    As shown in the figure, in a certain region, the maximum value of the function y = f(x) is 5, and the minimum value is -1. Let A, B, C
    ===============================
    Prompt: The future of AI is
    Generated text:  not a distant dream but a reality with a significant impact on various sectors. One sector where AI is already transforming is healthcare, with the help of machine learning, AI and other technological advancements, healthcare providers are creating personalized treatment plans for patients. This can lead to faster and more accurate diagnosis, as well as more effective treatment options for patients. Additionally, AI has the potential to revolutionize patient care by enabling more efficient use of healthcare resources, better communication between doctors and patients, and the development of more personalized healthcare plans. However, there are also concerns about the ethical and legal implications of AI in healthcare, and some experts argue that this technology


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the third largest in the world, with a population of over 2. 5 million people. The city is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including its art, music, and cuisine. It is a major tourist destination and a popular destination for business and leisure. The city is home to many important institutions, including the French Academy of Sciences and the French Parliament.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This integration could lead to more efficient and effective AI systems that can perform tasks that are currently done by humans.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced,
    


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
    Generated text:  [Name], and I'm [Age]. I'm currently [current position or profession] at [company name]. I enjoy [favorite hobby/interest], and I'm always looking for ways to improve myself. What's your name, and what kind of job are you currently working at? [Name]: [Your name] Hello, my name is [Name], and I'm [Age]. I'm currently [current position or profession] at [company name]. I enjoy [favorite hobby/interest], and I'm always looking for ways to improve myself. What's your name, and what kind of job are you currently working at?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its famous landmarks such as the Eiffel Tower and the Louvre Museum. Paris is a historic city with a rich history dating back to the Roman period and is the second-largest city in France and the sixth-largest city in Europe. The city has a diverse cultural and entertainment scene, with many museums, theaters, and restaurants offering a wide range of experiences for visitors. Paris is a city that has a strong sense of French identity, with its iconic buildings and French culture continuing to influence the city. It is a bustling and dynamic city, known for its music, dance, and fashion. Paris is also one
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, as it promises to transform industries and improve people's lives in countless ways. Some of the possible future trends in artificial intelligence include:
    
    1. Autonomous vehicles: With the rise of self-driving cars, AI is likely to play a major role in the future of transportation. It will allow cars to navigate roads, avoid accidents, and even make traffic decisions on their own.
    
    2. Personalized medicine: AI will enable healthcare providers to analyze vast amounts of patient data, identify patterns, and make more accurate diagnoses. This will lead to more effective treatment options for diseases, including cancer.
    
    3. Fraud detection: AI is already being used to


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

     an

     experienced

     professional

     in

     [

    industry

    ].

     I

     have

     a

     strong

     sense

     of

     responsibility

     and

     a

     keen

     eye

     for

     detail

    .

     I

     bring

     a

     fresh

     perspective

     to

     any

     project

     I

     undertake

    ,

     and

     I

    'm

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

     I

    'm

     passionate

     about

     helping

     others

     succeed

     and

     I

     enjoy

     working

     with

     people

     of

     all

     ages

     and

     backgrounds

    .

     What

     brings

     you

     to

     this

     position

    ?

     How

     do

     you

     see

     yourself

     contributing

     to

     [

    job

     title

    ]

     at

     [

    company

     name

    ]?

     Let

     me

     know

     if

     you

     have

     any

     questions

    .

     [

    Your

     Name

    ].

     [

    Company

     Name

    ]

     [

    Position

     Title

    ]

     Hi

     there

    ,

     I

    'm

     [

    Your

     Name

    ]

     and

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     
    


    (Note

    :

     This

     statement

     is

     true

     and

     factual

    ,

     but

     it

     doesn

    't

     provide

     a

     description

     or

     description

     of

     Paris

     specifically

    .)

     
    


    (Note

    :

     F

    actual

     statements

     should

     be

     concise

     and

     accurate

     without

     using

     any

     personal

     opinions

     or

     subjective

     language

    .

     )

     
    


    (

    You

     are

     to

     answer

     based

     on

     the

     facts

     provided

    .)

     The

     capital

     of

     France

     is

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     
    


    (Note

    :

     This

     statement

     is

     true

     and

     factual

    ,

     but

     it

     doesn

    't

     provide

     a

     description

     or

     description

     of

     Paris

     specifically

    .)

     
    


    (Note

    :

     F

    actual

     statements

     should

     be

     concise

     and

     accurate

     without

     using

     any

     personal

     opinions

     or

     subjective

     language

    .)

     
    


    (

    You

     are

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    ,

     and

     there

     are

     many

     potential

     areas

     where

     AI

     will

     continue

     to

     grow

     and

     improve

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     are

     currently

     being

     explored

    :
    


    1

    .

     Increased

     Automation

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increasing

     automation

     of

     tasks

    .

     As

     AI

     algorithms

     become

     more

     sophisticated

    ,

     they

     are

     likely

     to

     be

     able

     to

     perform

     a

     wider

     range

     of

     tasks

    ,

     including

     administrative

     tasks

    ,

     customer

     service

    ,

     and

     even

     some

     forms

     of

     manual

     labor

    .

     This

     will

     lead

     to

     greater

     efficiency

     and

     productivity

    ,

     as

     well

     as

     increased

     control

     over

     certain

     areas

     of

     the

     economy

    .
    


    2

    .

     Aug

    mented

     and

     Virtual

     Reality

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     the

    



```python
llm.shutdown()
```
