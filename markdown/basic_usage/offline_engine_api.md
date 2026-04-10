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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.54it/s]


    2026-04-10 17:07:30,526 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 17:07:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:04,  1.14s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:04,  1.14s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:04,  1.14s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.16it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.16it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.16it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:11,  4.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:11,  4.43it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.43it/s]

    Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.43it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:06,  7.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:06,  7.14it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:06,  7.14it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:06,  7.14it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:06,  7.14it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 11.12it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 11.12it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 11.12it/s]

    Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 11.12it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 11.12it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 15.36it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 15.36it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 15.36it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 15.36it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 15.36it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 19.34it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 19.34it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 19.34it/s]

    Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 19.34it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 19.34it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 23.14it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 23.14it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 23.14it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 23.14it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 23.14it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 25.56it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 25.56it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.47it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.69it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.69it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.69it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.69it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.69it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 30.69it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 33.33it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 33.33it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 33.33it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 33.33it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 33.33it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 33.60it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 33.60it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 33.60it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 33.60it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 33.60it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 35.15it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 35.15it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 35.15it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 35.15it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 35.15it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 35.15it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 38.97it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 38.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.57 GB):   2%|▏         | 1/58 [00:00<00:07,  8.08it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.54 GB):   2%|▏         | 1/58 [00:00<00:07,  8.08it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=54.54 GB):   3%|▎         | 2/58 [00:00<00:06,  8.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.54 GB):   3%|▎         | 2/58 [00:00<00:06,  8.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.54 GB):   5%|▌         | 3/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.53 GB):   5%|▌         | 3/58 [00:00<00:06,  8.35it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.53 GB):   7%|▋         | 4/58 [00:00<00:06,  8.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.54 GB):   7%|▋         | 4/58 [00:00<00:06,  8.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.54 GB):   9%|▊         | 5/58 [00:00<00:05,  8.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.53 GB):   9%|▊         | 5/58 [00:00<00:05,  8.89it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=54.53 GB):   9%|▊         | 5/58 [00:00<00:05,  8.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.53 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.53 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.50it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=54.53 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.53 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.58it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.53 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.52 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.48it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=54.52 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.52 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.52 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.52 GB):  21%|██        | 12/58 [00:01<00:05,  7.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.51 GB):  21%|██        | 12/58 [00:01<00:05,  7.81it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=54.51 GB):  21%|██        | 12/58 [00:01<00:05,  7.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.51 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.51 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.50 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.22it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=54.50 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.50 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.50 GB):  28%|██▊       | 16/58 [00:01<00:04, 10.27it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=54.50 GB):  31%|███       | 18/58 [00:02<00:04,  9.21it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.49 GB):  31%|███       | 18/58 [00:02<00:04,  9.21it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:02<00:05,  7.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:02<00:05,  7.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.47 GB):  33%|███▎      | 19/58 [00:02<00:05,  7.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.47 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=960 avail_mem=54.48 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.63it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=54.48 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=896 avail_mem=54.48 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.00it/s]Capturing num tokens (num_tokens=832 avail_mem=54.48 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.00it/s]Capturing num tokens (num_tokens=768 avail_mem=54.47 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.00it/s]

    Capturing num tokens (num_tokens=768 avail_mem=54.47 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.16it/s]Capturing num tokens (num_tokens=704 avail_mem=54.47 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.16it/s]Capturing num tokens (num_tokens=640 avail_mem=54.47 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.16it/s]Capturing num tokens (num_tokens=640 avail_mem=54.47 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.10it/s]Capturing num tokens (num_tokens=576 avail_mem=54.47 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.10it/s]

    Capturing num tokens (num_tokens=512 avail_mem=54.46 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.10it/s]Capturing num tokens (num_tokens=512 avail_mem=54.46 GB):  50%|█████     | 29/58 [00:03<00:02, 12.67it/s]Capturing num tokens (num_tokens=480 avail_mem=54.47 GB):  50%|█████     | 29/58 [00:03<00:02, 12.67it/s]Capturing num tokens (num_tokens=448 avail_mem=54.47 GB):  50%|█████     | 29/58 [00:03<00:02, 12.67it/s]Capturing num tokens (num_tokens=448 avail_mem=54.47 GB):  53%|█████▎    | 31/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=416 avail_mem=54.47 GB):  53%|█████▎    | 31/58 [00:03<00:01, 14.12it/s]

    Capturing num tokens (num_tokens=384 avail_mem=54.47 GB):  53%|█████▎    | 31/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=352 avail_mem=54.46 GB):  53%|█████▎    | 31/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=352 avail_mem=54.46 GB):  59%|█████▊    | 34/58 [00:03<00:01, 15.85it/s]Capturing num tokens (num_tokens=320 avail_mem=54.46 GB):  59%|█████▊    | 34/58 [00:03<00:01, 15.85it/s]Capturing num tokens (num_tokens=288 avail_mem=54.45 GB):  59%|█████▊    | 34/58 [00:03<00:01, 15.85it/s]

    Capturing num tokens (num_tokens=288 avail_mem=54.45 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.38it/s]Capturing num tokens (num_tokens=256 avail_mem=54.45 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.38it/s]Capturing num tokens (num_tokens=240 avail_mem=54.45 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.38it/s]Capturing num tokens (num_tokens=240 avail_mem=54.45 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.59it/s]Capturing num tokens (num_tokens=224 avail_mem=54.44 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.59it/s]

    Capturing num tokens (num_tokens=208 avail_mem=54.44 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.59it/s]Capturing num tokens (num_tokens=208 avail_mem=54.44 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.62it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.62it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.62it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.49it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.49it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.49it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.08it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.08it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.08it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.37it/s]Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.37it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.37it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.80it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.80it/s]Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.80it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  86%|████████▌ | 50/58 [00:04<00:00, 13.17it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 13.17it/s]Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:04<00:00, 13.17it/s]Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.37it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.37it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.37it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:04<00:00, 13.52it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  93%|█████████▎| 54/58 [00:04<00:00, 13.52it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:04<00:00, 13.52it/s]

    Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 13.80it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 13.80it/s] Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:05<00:00, 13.80it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:05<00:00, 13.97it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:05<00:00, 11.24it/s]


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
    Generated text:  Lisa, and I'm an avid traveler, both physically and mentally. My main goal is to explore a wide variety of locations and experiences to learn and grow as an individual. I'm also interested in learning about how different cultures and environments shape our experiences and perspectives.
    Can you help me with anything related to travel or culture?
    Sure, I'd be happy to help! What would you like to know about traveling or culture? Please let me know how I can assist you further. #travel #culture #traveltips #culturetips #travelindustries #traveltipsfortravelers #travelindustriesfortravelers #travelind
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to pick a band to represent the country. He has only a certain number of bands available, which he will select. He has 300 millionaires, 550 million people with a job, and 150 million people with a college degree. The president thinks that if he selects a band based on income, he will have the highest percentage of the population represented by the band he picks. What is the minimum number of bands the president needs to choose? To determine the minimum number of bands the president needs to choose to represent the highest percentage of the population, we first need to calculate the total number of
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. Lyon C. Marseille D. Brussels
    Answer:
    A
    
    When you look at a scalded area, how should you handle it?
    A. Seek help immediately
    B. Cover it with a cloth or cloth pad
    C. Cool it with cold water
    D. Put it on a towel
    Answer:
    B
    
    What is the main function of an aircraft's wing?
    A. Increase lift
    B. Reduce lift
    C. Enhance stability
    D. Enhance maneuverability
    Answer:
    A
    
    Which of the following is considered a logical fallacy?
    A. All men are mortal.
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s already impacting many areas of our lives. But do you know what your AI is doing? AI isn’t always programming itself to be helpful or to improve human systems, but it’s constantly improving itself. I’m going to explain how AI has become more powerful and powerful faster than you’ve ever seen it be in the past.
    See, the future is here. The future is here. The future is here.
    A few years ago, researchers figured out how to make a computer that could learn from the environment, and I’ll show you how AI can make even more intelligent machines. But this time, the researchers took


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is the largest city in France by population. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many famous museums, theaters, and restaurants. The city is known for its fashion industry, with many famous fashion houses and boutiques located in the city. Paris is a vibrant and diverse city with a rich history and culture that continues to inspire and captivate people around
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to become even more advanced in the future. AI-powered diagnostic tools, personalized medicine, and virtual assistants for patients are all areas where AI is expected to have a significant impact.
    
    2. Increased use of AI in finance: AI is already being used in finance to help with fraud detection, risk management
    


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
    Generated text:  [Name], and I'm a skilled [job title]. I have a passion for [occupation]. I'm confident in [strengths or personality traits], and I have a reputation for [hobbies or interests]. I love [something enjoyable about my job]. I'm always eager to learn and improve [ability], and I'm always ready to help anyone who needs a hand. I'm always looking for new challenges and opportunities to grow and improve. Thank you for taking the time to meet me. How can I help you? Is there a particular project or task that I should start with? Is there a particular topic or subject that you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The other option is incorrect. The correct answer is Paris. Paris is the capital and largest city of France, located in the northern region of the country. Its economic, political, and cultural influence is widespread, and it is a major tourist destination. France's cultural, literary, and historical heritage is also notable in Paris. Paris is also known for its contemporary art scene and fashion industry. The city is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among other landmarks and attractions. Paris's role as a major European city and its rich cultural landscape make it a significant contributor
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in areas such as machine learning, natural language processing, computer vision, robotics, and autonomous systems. In addition, there is a growing focus on AI ethics and privacy, with concerns about the potential impact of AI on society and the environment. This includes discussions about the ethical use of AI, such as how it can be used to address issues like climate change and poverty. AI will also continue to be a driving force in the development of new technologies and industries, such as quantum computing, biotechnology, and artificial general intelligence. Overall, the future of AI is likely to be one of continued growth and change, as


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

     Emily

    ,

     and

     I

    'm

     a

     

    3

    5

    -year

    -old

     software

     engineer

     who

     has

     been

     working

     in

     the

     tech

     industry

     for

     over

     a

     decade

    .

     I

     have

     a

     passion

     for

     learning

     new

     technologies

     and

     continuously

     improving

     my

     skills

    .

     I

    'm

     always

     looking

     for

     opportunities

     to

     grow

     and

     develop

     my

     expertise

     in

     my

     field

    .

     
    


    My

     goal

     is

     to

     help

     others

     find

     success

     in

     the

     tech

     industry

    ,

     and

     I

    'm

     always

     eager

     to

     share

     my

     knowledge

     and

     insights

     with

     anyone

     who

     is

     interested

     in

     learning

    .

     I

     have

     a

     friendly

     and

     approach

    able

     demeanor

    ,

     and

     I

     enjoy

     working

     with

     people

     from

     all

     walks

     of

     life

    ,

     from

     beginners

     to

     experts

    .

     I

     am

     always

     eager

     to

     assist

     with

     any

     questions

     or

     concerns

     that

     arise

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     with

     a

     rich

     history

     and

     culture

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     and

     is

     home

     to

     many

     world

    -ren

    owned

     museums

     and

     art

     galleries

    .

     The

     city

     is

     also

     famous

     for

     its

     vibrant

     arts

     scene

     and

     music

     festivals

    ,

     including

     the

     Op

    éra

     and

     the

     Mou

    lin

     Rouge

    .

     Paris

     is

     a

     vibrant

    ,

     cosm

    opolitan

     city

     with

     a

     rich

     history

     and

     culture

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     It

    's

     a

     great

     city

     to

     explore

    ,

     visit

    ,

     and

     experience

     the

     best

     of

     Europe

    .

     Paris

    's

     unique

     architecture

    ,

     vibrant

     culture

    ,

     and

     historic

     landmarks

     have

     made

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     wide

     range

     of

     technological

     and

     societal

     developments

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     shape

     its

     future

    :
    


    1

    .

     Increased

     AI

     efficiency

    :

     As

     AI

     technologies

     become

     more

     advanced

    ,

     they

     are

     expected

     to

     become

     more

     efficient

     and

     effective

     in

     their

     tasks

    .

     This

     could

     lead

     to

     more

     accurate

     predictions

     and

     better

     decision

    -making

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
    


    2

    .

     AI

     will

     become

     more

     pervasive

    :

     AI

     is

     likely

     to

     become

     more

     pervasive

     in

     our

     daily

     lives

    ,

     with

     more

     and

     more

     devices

     and

     systems

     incorporating

     AI

     features

    .

     This

     could

     lead

     to

     a

     greater

     reliance

     on

     AI

     for

     everyday

     tasks

    ,

     such

     as

     in

     home

     automation

    ,

     transportation

    ,

     and

    



```python
llm.shutdown()
```
