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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.03it/s]


    2026-04-12 22:12:32,493 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 22:12:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.07it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.07it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.49it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.49it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.49it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  7.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  7.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  7.99it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  7.99it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 11.05it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 11.05it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 11.05it/s]

    Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 11.05it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:02, 14.26it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:02, 14.26it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:02, 14.26it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:02, 14.26it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 17.01it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 19.66it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 19.66it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 19.66it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 19.66it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 19.66it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 24.20it/s]

    Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 24.84it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 24.84it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 24.84it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:00, 28.39it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 29.33it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 37.05it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 37.05it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 37.05it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 37.05it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 37.05it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 37.63it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 42.67it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 42.67it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 42.67it/s]

    Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 42.67it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 42.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.01 GB):   2%|▏         | 1/58 [00:00<00:09,  6.29it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.02 GB):   2%|▏         | 1/58 [00:00<00:09,  6.29it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.02 GB):   3%|▎         | 2/58 [00:00<00:08,  6.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.01 GB):   3%|▎         | 2/58 [00:00<00:08,  6.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.01 GB):   5%|▌         | 3/58 [00:00<00:08,  6.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.01 GB):   5%|▌         | 3/58 [00:00<00:08,  6.84it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=50.01 GB):   7%|▋         | 4/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.03 GB):   7%|▋         | 4/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.02 GB):   7%|▋         | 4/58 [00:00<00:07,  7.41it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=50.02 GB):  10%|█         | 6/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.02 GB):  10%|█         | 6/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.02 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.01 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.43it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=50.01 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.00 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.00 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.00 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.03it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=50.00 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.99 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.99 GB):  19%|█▉        | 11/58 [00:01<00:05,  7.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.96 GB):  19%|█▉        | 11/58 [00:01<00:05,  7.92it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=49.97 GB):  19%|█▉        | 11/58 [00:01<00:05,  7.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=49.97 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=49.97 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.96 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.49it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=49.96 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.97 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.96 GB):  26%|██▌       | 15/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.96 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.95 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.95 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.06it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=49.95 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.94 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.92 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.92 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.04it/s]Capturing num tokens (num_tokens=960 avail_mem=49.93 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.04it/s] Capturing num tokens (num_tokens=896 avail_mem=49.92 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.04it/s]

    Capturing num tokens (num_tokens=896 avail_mem=49.92 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.18it/s]Capturing num tokens (num_tokens=832 avail_mem=49.90 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.18it/s]Capturing num tokens (num_tokens=768 avail_mem=49.91 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.18it/s]Capturing num tokens (num_tokens=768 avail_mem=49.91 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.99it/s]Capturing num tokens (num_tokens=704 avail_mem=49.90 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.99it/s]Capturing num tokens (num_tokens=640 avail_mem=49.90 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.99it/s]

    Capturing num tokens (num_tokens=640 avail_mem=49.90 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.54it/s]Capturing num tokens (num_tokens=576 avail_mem=49.89 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.54it/s]Capturing num tokens (num_tokens=512 avail_mem=49.88 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.54it/s]Capturing num tokens (num_tokens=512 avail_mem=49.88 GB):  50%|█████     | 29/58 [00:02<00:01, 17.13it/s]Capturing num tokens (num_tokens=480 avail_mem=49.89 GB):  50%|█████     | 29/58 [00:02<00:01, 17.13it/s]Capturing num tokens (num_tokens=448 avail_mem=49.88 GB):  50%|█████     | 29/58 [00:02<00:01, 17.13it/s]

    Capturing num tokens (num_tokens=448 avail_mem=49.88 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.73it/s]Capturing num tokens (num_tokens=416 avail_mem=49.88 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.73it/s]Capturing num tokens (num_tokens=384 avail_mem=49.85 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.73it/s]Capturing num tokens (num_tokens=352 avail_mem=49.87 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.73it/s]Capturing num tokens (num_tokens=352 avail_mem=49.87 GB):  59%|█████▊    | 34/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=320 avail_mem=49.86 GB):  59%|█████▊    | 34/58 [00:02<00:01, 18.46it/s]

    Capturing num tokens (num_tokens=288 avail_mem=49.85 GB):  59%|█████▊    | 34/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=288 avail_mem=49.85 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=256 avail_mem=49.85 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=240 avail_mem=49.84 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=240 avail_mem=49.84 GB):  66%|██████▌   | 38/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=224 avail_mem=49.83 GB):  66%|██████▌   | 38/58 [00:02<00:01, 18.94it/s]

    Capturing num tokens (num_tokens=208 avail_mem=49.83 GB):  66%|██████▌   | 38/58 [00:03<00:01, 18.94it/s]Capturing num tokens (num_tokens=208 avail_mem=49.83 GB):  69%|██████▉   | 40/58 [00:03<00:00, 19.00it/s]Capturing num tokens (num_tokens=192 avail_mem=49.82 GB):  69%|██████▉   | 40/58 [00:03<00:00, 19.00it/s]Capturing num tokens (num_tokens=176 avail_mem=49.82 GB):  69%|██████▉   | 40/58 [00:03<00:00, 19.00it/s]Capturing num tokens (num_tokens=160 avail_mem=49.81 GB):  69%|██████▉   | 40/58 [00:03<00:00, 19.00it/s]Capturing num tokens (num_tokens=160 avail_mem=49.81 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.53it/s]Capturing num tokens (num_tokens=144 avail_mem=49.81 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.53it/s]Capturing num tokens (num_tokens=128 avail_mem=49.81 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.53it/s]

    Capturing num tokens (num_tokens=112 avail_mem=49.80 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.53it/s]Capturing num tokens (num_tokens=96 avail_mem=49.80 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.53it/s] Capturing num tokens (num_tokens=96 avail_mem=49.80 GB):  81%|████████  | 47/58 [00:03<00:00, 24.54it/s]Capturing num tokens (num_tokens=80 avail_mem=49.80 GB):  81%|████████  | 47/58 [00:03<00:00, 24.54it/s]Capturing num tokens (num_tokens=64 avail_mem=49.79 GB):  81%|████████  | 47/58 [00:03<00:00, 24.54it/s]Capturing num tokens (num_tokens=48 avail_mem=49.79 GB):  81%|████████  | 47/58 [00:03<00:00, 24.54it/s]Capturing num tokens (num_tokens=32 avail_mem=49.79 GB):  81%|████████  | 47/58 [00:03<00:00, 24.54it/s]Capturing num tokens (num_tokens=32 avail_mem=49.79 GB):  88%|████████▊ | 51/58 [00:03<00:00, 26.38it/s]Capturing num tokens (num_tokens=28 avail_mem=49.78 GB):  88%|████████▊ | 51/58 [00:03<00:00, 26.38it/s]

    Capturing num tokens (num_tokens=24 avail_mem=49.78 GB):  88%|████████▊ | 51/58 [00:03<00:00, 26.38it/s]Capturing num tokens (num_tokens=20 avail_mem=49.77 GB):  88%|████████▊ | 51/58 [00:03<00:00, 26.38it/s]Capturing num tokens (num_tokens=16 avail_mem=49.77 GB):  88%|████████▊ | 51/58 [00:03<00:00, 26.38it/s]Capturing num tokens (num_tokens=16 avail_mem=49.77 GB):  95%|█████████▍| 55/58 [00:03<00:00, 29.02it/s]Capturing num tokens (num_tokens=12 avail_mem=49.77 GB):  95%|█████████▍| 55/58 [00:03<00:00, 29.02it/s]Capturing num tokens (num_tokens=8 avail_mem=49.77 GB):  95%|█████████▍| 55/58 [00:03<00:00, 29.02it/s] Capturing num tokens (num_tokens=4 avail_mem=49.76 GB):  95%|█████████▍| 55/58 [00:03<00:00, 29.02it/s]Capturing num tokens (num_tokens=4 avail_mem=49.76 GB): 100%|██████████| 58/58 [00:03<00:00, 16.00it/s]


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
    Generated text:  Nora and I'm a multilingual English speaker. I am best known as a singer and songwriter, but also a musician, artist, and my background is in both worlds of music and literature.
    I originally signed with the London label, but have since worked with both Ascent Records and I Am Music. I am based in London and I’m currently based in Paris and London. I’m particularly known for my cover songs and duets, and I’m frequently touring with other musicians and musicians I’m close to.
    I started making music in the early 2000s, playing in a band with Jason Blackford and Ida
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 6 inches tall. If it is a certain day, the president is taller than the current president. The current president is 3 feet 4 inches tall. How many feet taller is the president now compared to the current president? First, we need to convert the heights of both presidents into the same unit. Since the president of the United States is 5 feet 6 inches tall and the current president is 3 feet 4 inches tall, we can convert these heights to the same unit, which is inches. 
    
    5 feet 6 inches is equal to \(5 \times 12 + 6 =
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Rome
    Answer: A
    
    The position where the slope of a straight line with a positive slope is greater than 1 is ____.
    A. Below the x-axis
    B. Above the x-axis
    C. On the x-axis
    D. Any of the above
    Answer: B
    
    One of the most significant impacts of the 'billion-dollar crash' is that it ___.
    A. Reduced the number of people employed
    B. Increased the income of the largest corporations
    C. Reduced the number of technological innovations
    D. Increased the number of successful companies
    Answer
    ===============================
    Prompt: The future of AI is
    Generated text:  currently in the early stages, with the industries that are expected to benefit from AI in the near future will include healthcare, the automotive industry, the banking industry, and the finance industry.
    AI is considered as a powerful tool that can solve complex problems. It can work with humans to solve a wide range of problems, but it also has the potential to make the world a better place.
    AI is one of the most significant changes in the technology sector. This technology has the potential to bring the greatest impact to the world in the future. In the early days of AI, it was more about the idea of creating a machine that can do the


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] and [Country]. I have a [Number] degree in [Field of Study], and I've been working in [Industry] for [Number] years. I'm a [Number] year old, [Gender] and [Country]. I have a [Number] degree in [Field of Study], and I've been working in [Industry] for [Number] years. I'm a [Number] year old, [Gender] and [Country]. I have a [Number] degree in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Academy of Sciences, which is one of the oldest and most prestigious scientific academies in the world. Paris is a vibrant and dynamic city that is a must-visit for anyone interested in French culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and evaluation of AI systems, as well as greater
    


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
    Generated text:  [Name], and I'm a [career or profession]! I have a passion for [what you could do to impress me with your skills?]. What makes you unique?
    As an AI language model, my unique selling point is my ability to understand and respond to human language in natural language processing. I can provide context, analyze nuances, and respond to queries in a way that is both informative and engaging. I can also generate and respond to creative prompts, text, and images in real-time, making me an invaluable resource for anyone looking to communicate and express themselves in a meaningful way. As an AI, I am here to assist
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city in the country. It is located on the Mediterranean coast and is known for its rich history and modern architecture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of the French Revolution and has been a center of art, literature, and music for centuries. As of 2021, Paris has an estimated population of over 2.2 million people. It is a popular tourist destination and is home to many museums, theaters, and restaurants. Paris is also known for its cuisine and has a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly diverse and rapidly evolving, with a wide range of potential applications and technologies that will continue to change the way we live, work, and interact with the world. Some of the most promising trends in AI include:
    
    1. Increased integration with human decision-making: As AI becomes more sophisticated, it will become more integrated with human decision-making processes, allowing it to make more accurate and informed decisions. This will enable AI to help humans make better choices in areas such as healthcare, finance, and transportation.
    
    2. More efficient and cost-effective AI: AI will continue to become more efficient and cost-effective, allowing companies to operate with


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

    Character

     Name

    ],

     and

     I

     am

     a

     digital

     AI

     assistant

     designed

     to

     assist

     with

     various

     tasks

     and

     provide

     information

    .

     How

     can

     I

     help

     you

     today

    ?

     I

    'm

     here

     to

     help

     answer

     any

     questions

     you

     might

     have

     or

     provide

     you

     with

     information

     that

    's

     not

     readily

     available

     online

    .

     What

     can

     I

     do

     for

     you

    ?

     Let

    's

     get

     started

    !

     [

    Character

     Name

    ]

     can

     assist

     you

     with

     any

     question

     you

     might

     have

     or

     provide

     you

     with

     information

     that

     you

    're

     not

     familiar

     with

    .

     What

     do

     you

     want

     to

     know

     today

    ?

     I

    'm

     here

     to

     help

     and

     provide

     you

     with

     the

     information

     you

     need

    .

     [

    Character

     Name

    ]

     is

     here

     to

     assist

     you

     and

     help

     you

     achieve

     your

     goals

    .

     How

     can

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Gar

    de

    "

     or

     simply

     "

    Paris

    ."

     It

     is

     a

     historic

     city

     that

     is

     home

     to

     many

     of

     the

     country

    's

     most

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     the

     seat

     of

     the

     French

     government

    ,

     the

     national

     capital

    ,

     and

     a

     major

     transportation

     hub

    .

     Paris

     has

     a

     rich

     cultural

     heritage

     and

     is

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     nightlife

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     attracts

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     a

     diverse

     population

     and

     is

     a

     major

     economic

     center

     in

     Europe

    .

     Paris

     is

     a

     fascinating

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     one

     of

     continuous

     improvement

     and

     divers

    ification

    ,

     as

     companies

     and

     researchers

     work

     to

     develop

     new

     algorithms

    ,

     technologies

    ,

     and

     applications

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


     

     

    1

    .

     Increased

     integration

     with

     human

     capabilities

    :

     As

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     we

     may

     see

     more

     widespread

     adoption

     of

     AI

    -powered

     technologies

     and

     services

     that

     are

     complementary

     to

     human

     capabilities

    ,

     such

     as

     personal

     assistants

    ,

     virtual

     assistants

    ,

     and

     self

    -driving

     cars

    .


     

     

    2

    .

     Enhanced

     creativity

     and

     innovation

    :

     AI

     is

     often

     used

     to

     help

     humans

     solve

     complex

     problems

     and

     develop

     new

     ideas

    ,

     but

     there

     is

     also

     the

     potential

     for

     AI

     to

     be

     used

     to

     inspire

     new

     ideas

     and

     lead

     to

     breakthrough

    



```python
llm.shutdown()
```
