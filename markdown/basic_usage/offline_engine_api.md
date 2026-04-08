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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    2026-04-08 07:55:05.897 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 07:55:05] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 07:55:05.897 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 07:55:05] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 07:55:05.897 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 07:55:05] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 07:55:05.897 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 07:55:05] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 07:55:05.897 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 07:55:05] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.92it/s]


    2026-04-08 07:55:08,979 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 07:55:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:50,  3.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:50,  3.00s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:50,  3.00s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:45,  1.20it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:45,  1.20it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:03<00:45,  1.20it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:03<00:45,  1.20it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:17,  2.90it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:17,  2.90it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:17,  2.90it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:17,  2.90it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:17,  2.90it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]

    Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.78it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:03, 11.15it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]

    Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:03<00:02, 16.98it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 33.76it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 33.76it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 33.76it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 33.76it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 33.76it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 33.76it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 33.76it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 38.02it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 45.11it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 45.11it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 45.11it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 45.11it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 45.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.28 GB):   3%|▎         | 2/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.27 GB):   3%|▎         | 2/58 [00:00<00:04, 11.59it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.27 GB):   3%|▎         | 2/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.27 GB):   7%|▋         | 4/58 [00:00<00:04, 12.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.26 GB):   7%|▋         | 4/58 [00:00<00:04, 12.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.26 GB):   7%|▋         | 4/58 [00:00<00:04, 12.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.26 GB):  10%|█         | 6/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.26 GB):  10%|█         | 6/58 [00:00<00:03, 14.81it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.25 GB):  10%|█         | 6/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.24 GB):  10%|█         | 6/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.24 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.66it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.66it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.23 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.22 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.19 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.36it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.15 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.15 GB):  31%|███       | 18/58 [00:00<00:01, 23.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 23.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.15 GB):  31%|███       | 18/58 [00:00<00:01, 23.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.13 GB):  31%|███       | 18/58 [00:00<00:01, 23.64it/s]Capturing num tokens (num_tokens=960 avail_mem=72.14 GB):  31%|███       | 18/58 [00:01<00:01, 23.64it/s] Capturing num tokens (num_tokens=960 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.49it/s]Capturing num tokens (num_tokens=896 avail_mem=72.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.49it/s]Capturing num tokens (num_tokens=832 avail_mem=72.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.49it/s]Capturing num tokens (num_tokens=768 avail_mem=72.12 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.49it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.49it/s]Capturing num tokens (num_tokens=704 avail_mem=72.13 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.22it/s]Capturing num tokens (num_tokens=640 avail_mem=72.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.22it/s]Capturing num tokens (num_tokens=576 avail_mem=72.11 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.22it/s]Capturing num tokens (num_tokens=512 avail_mem=72.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.22it/s]Capturing num tokens (num_tokens=480 avail_mem=71.84 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.22it/s]Capturing num tokens (num_tokens=480 avail_mem=71.84 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.74it/s]Capturing num tokens (num_tokens=448 avail_mem=71.82 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.74it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.81 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.74it/s]Capturing num tokens (num_tokens=384 avail_mem=71.83 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.74it/s]Capturing num tokens (num_tokens=384 avail_mem=71.83 GB):  57%|█████▋    | 33/58 [00:01<00:01, 24.60it/s]Capturing num tokens (num_tokens=352 avail_mem=71.82 GB):  57%|█████▋    | 33/58 [00:01<00:01, 24.60it/s]Capturing num tokens (num_tokens=320 avail_mem=71.79 GB):  57%|█████▋    | 33/58 [00:01<00:01, 24.60it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.78 GB):  57%|█████▋    | 33/58 [00:01<00:01, 24.60it/s]Capturing num tokens (num_tokens=288 avail_mem=71.78 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=256 avail_mem=71.78 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=240 avail_mem=71.77 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=224 avail_mem=71.79 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=224 avail_mem=71.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.54it/s]Capturing num tokens (num_tokens=208 avail_mem=71.78 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.54it/s]Capturing num tokens (num_tokens=192 avail_mem=71.78 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.54it/s]Capturing num tokens (num_tokens=176 avail_mem=71.77 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.54it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.76 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.54it/s]Capturing num tokens (num_tokens=160 avail_mem=71.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 27.33it/s]Capturing num tokens (num_tokens=144 avail_mem=71.76 GB):  74%|███████▍  | 43/58 [00:01<00:00, 27.33it/s]Capturing num tokens (num_tokens=128 avail_mem=71.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 27.33it/s]Capturing num tokens (num_tokens=112 avail_mem=71.74 GB):  74%|███████▍  | 43/58 [00:01<00:00, 27.33it/s]Capturing num tokens (num_tokens=96 avail_mem=71.73 GB):  74%|███████▍  | 43/58 [00:01<00:00, 27.33it/s] Capturing num tokens (num_tokens=96 avail_mem=71.73 GB):  81%|████████  | 47/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=80 avail_mem=71.73 GB):  81%|████████  | 47/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=64 avail_mem=71.72 GB):  81%|████████  | 47/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  81%|████████  | 47/58 [00:02<00:00, 29.65it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.71 GB):  81%|████████  | 47/58 [00:02<00:00, 29.65it/s]Capturing num tokens (num_tokens=32 avail_mem=71.71 GB):  88%|████████▊ | 51/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=28 avail_mem=71.70 GB):  88%|████████▊ | 51/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=24 avail_mem=71.69 GB):  88%|████████▊ | 51/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=20 avail_mem=71.68 GB):  88%|████████▊ | 51/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=16 avail_mem=71.68 GB):  88%|████████▊ | 51/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=16 avail_mem=71.68 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.31it/s]Capturing num tokens (num_tokens=12 avail_mem=71.67 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.31it/s]Capturing num tokens (num_tokens=8 avail_mem=71.67 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.31it/s] Capturing num tokens (num_tokens=4 avail_mem=71.66 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.31it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.66 GB): 100%|██████████| 58/58 [00:02<00:00, 25.89it/s]


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
    Generated text:  Tanya, 25 years old, an English girl, from the south. I'm staying at a hotel, and there is a lounge. I don't know how to use the internet on my phone, and I want to know how to do that. I can find a free library in the hotel. However, I want to know the best way to use the internet on my phone.
    
    A. My phone doesn't have a Wi-Fi connection. I can also find Wi-Fi on the lounge.
    
    B. I can search the Internet on the website of the hotel.
    
    C. I can use an app on my phone to search
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected office. Candidates for the office must be nominated by the Democratic Party. The term of office for the presidency is four years. In 1913, the president was a man named Roosevelt, and he was the first one to be a Democrat. In 1933, the president was a woman named Hoover, who was the first one to be a Republican. The president has to be at least 40 years old. The current president of the United States is Donald Trump, and he was elected in 2016. Which of the following best describes the relationship between the president and the Democratic Party
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Lille
    B. Paris
    C. Strasbourg
    D. Nice
    The capital of France is Paris. 
    
    Therefore, the correct answer is:
    
    B. Paris
    
    To further elaborate:
    Paris is the capital city of France and is located in the north-central region of the country, next to the English Channel. It is the second-largest city in France and the largest in the North of France. Paris is known for its iconic Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and many other attractions, making it a major cultural and economic hub in France. The name "Paris" comes from the
    ===============================
    Prompt: The future of AI is
    Generated text:  now. In a world where an abundance of data, technology and creativity are available to us, we have the potential to build a future where humans are outmatched by machines. While the concept of AI is currently quite futuristic, the technology is already here and will continue to advance as the technology advances. The implications of this technology are wide-ranging, and include employment, privacy, and a new way of life. AI is also a topic of public debate, with many people questioning the ethics of machines and the impact of AI on society.
    In this article, we will explore the future of AI, its implications, and the ethical considerations that need


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its cuisine, fashion, and art scene. The city is also home to many world-renowned museums, theaters, and other cultural institutions. Paris is a major transportation hub and a popular tourist destination, with many visitors coming to explore its beautiful architecture, vibrant culture, and rich history. The city is also home to the French Parliament, the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and needs.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and personal information that is collected and used by AI systems. This could lead to more
    


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
    Generated text:  [Your Name]. I am a [Your Profession] with a [X number] degree. Currently, I am [X years old]. My [X favourite] hobby is [X activity]. I enjoy [X activity] and my [X interest] in my field of study is [X interest]. Overall, I am a [X type of person]. I am dedicated to [X goal]. I am excited to see [X person] in the future. Thank you for asking.
    Hello, my name is [Your Name]. I am a [Your Profession] with a [X number] degree. Currently, I am [X
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic Eiffel Tower, opulent Le Seine River, and its famous museums and art museums. Additionally, the city is known for its rich cultural heritage and traditions, and is home to many famous landmarks and attractions. According to the French Riviera, Paris is one of the most beautiful cities in Europe, with its warm Mediterranean climate and picturesque beaches. Paris has also been recognized for its role in the French Revolution and its contributions to modern French culture, and is known as the "City of Love" for its romantic and picturesque attractions. Overall, Paris is a city that is both historic and modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  quite promising and has the potential to greatly impact many areas of society. Some possible trends that could emerge are:
    
    1. Increased automation: As AI becomes more sophisticated, it is likely to be integrated into more and more industries. This could lead to significant automation of tasks and processes, potentially reducing the need for human workers. This could have both positive and negative impacts on society, depending on how it is implemented.
    
    2. Enhanced privacy: AI systems may become more sophisticated and capable of processing and analyzing data. This could lead to increased privacy concerns, as the data used to train these systems could be sensitive. It is important to develop privacy protections


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

    name

    ].

     I

     am

     [

    age

    ]

     years

     old

    ,

     and

     I

     am

     a

     [

    gender

    ]

     [

    gender

     trait

    ].

     I

     am

     [

    occupation

    ],

     and

     I

     am

     passionate

     about

     [

    occupation

    ],

     and

     I

     have

     a

     [

    interest

    ],

     [

    major

     interest

    ],

     or

     [

    char

    ity

    ].

     I

     enjoy

     [

    skill

    /

    interest

    /

    char

    ity

    ]

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     grow

     as

     a

     person

    .

     I

     am

     [

    interest

     level

    ],

     and

     I

     believe

     in

     [

    phil

    osoph

    ical

     principle

    ,

     value

    ,

     or

     concept

    ].

     I

     am

     [

    smart

     or

     quick

    -w

    itted

    ],

     and

     I

     am

     always

     up

     for

     a

     good

     challenge

    ,

     and

     I

     am

     always

     willing

     to

     learn

     new

     things

    .

     I

     am

     [

    professional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    )

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

    .


    B

    )

     It

     has

     the

     most

     statues

     in

     the

     world

    .


    C

    )

     It

     is

     the

     oldest

     city

     in

     the

     world

    .


    D

    )

     It

     has

     the

     best

     weather

     in

     Europe

    .


    E

    )

     It

     is

     the

     largest

     city

     in

     terms

     of

     population

    .

     C

    )

     It

     is

     the

     oldest

     city

     in

     the

     world

    .

     
    


    A

     city

     with

     the

     longest

     existence

     is

     Paris

    .

     It

     was

     founded

     on

     September

     

    2

    0

    ,

     

    7

    8

    9

     AD

     by

     Charles

     Mart

    el

    ,

     the

     leader

     of

     the

     Fr

    anks

    .

     Paris

     is

     one

     of

     the

     oldest

     cities

     in

     the

     world

    ,

     and

     was

     the

     capital

     of

     France

     from

     

    7

    8

    9

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     for

     a

     wide

     range

     of

     exciting

     developments

     and

     applications.

     Here

     are

     some

     possible

     trends

     and

     potential

     impacts

    :
    


    1

    .

     Improved

     accuracy

     of

     AI

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

     development

     of

     more

     accurate

     algorithms

     and

     models

    .

     As

     technology

     advances

    ,

     we

     may

     see

     AI

     systems

     become

     more

     adept

     at

     detecting

     and

     identifying

     patterns

     in

     data

    ,

     improving

     their

     ability

     to

     understand

     and

     predict

     human

     behavior

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     various

     healthcare

     applications

    ,

     such

     as

     personalized

     medicine

    ,

     diagnosis

    ,

     and

     drug

     discovery

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     even

     more

     integration

     of

     AI

     into

     healthcare

     practices

    .
    


    3

    .

     AI

     in

     transportation

    



```python
llm.shutdown()
```
