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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


    2026-04-11 22:04:17,266 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 22:04:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:02,  1.11s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:02,  1.11s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:02,  1.11s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.76it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.76it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.76it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.76it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:02, 14.10it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:02, 14.10it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:02, 14.10it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:02, 14.10it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:02, 14.10it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.43it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.43it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.43it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.43it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.43it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 18.43it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 23.83it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 23.83it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 23.83it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 23.83it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 23.83it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 27.27it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 27.27it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 27.27it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 27.27it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 27.27it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 27.27it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 32.60it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 32.60it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 32.60it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 32.60it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 32.60it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 32.60it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 35.64it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 40.33it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 45.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.61 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.41 GB):   7%|▋         | 4/58 [00:00<00:05,  9.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.41 GB):   7%|▋         | 4/58 [00:00<00:05,  9.98it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=56.57 GB):   7%|▋         | 4/58 [00:00<00:05,  9.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.57 GB):  10%|█         | 6/58 [00:00<00:04, 10.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.57 GB):  10%|█         | 6/58 [00:00<00:04, 10.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.58 GB):  10%|█         | 6/58 [00:00<00:04, 10.60it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.58 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.58 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.57 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.57 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.56 GB):  17%|█▋        | 10/58 [00:00<00:04, 10.69it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.63 GB):  17%|█▋        | 10/58 [00:01<00:04, 10.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.63 GB):  21%|██        | 12/58 [00:01<00:04, 10.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.63 GB):  21%|██        | 12/58 [00:01<00:04, 10.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.56 GB):  21%|██        | 12/58 [00:01<00:04, 10.79it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.56 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.93 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.69 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.69 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.69 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.44it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=56.54 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.54 GB):  31%|███       | 18/58 [00:01<00:03, 12.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.75 GB):  31%|███       | 18/58 [00:01<00:03, 12.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.75 GB):  31%|███       | 18/58 [00:01<00:03, 12.14it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.75 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.52 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.39it/s]Capturing num tokens (num_tokens=960 avail_mem=56.53 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.39it/s] Capturing num tokens (num_tokens=960 avail_mem=56.53 GB):  38%|███▊      | 22/58 [00:01<00:02, 13.23it/s]Capturing num tokens (num_tokens=896 avail_mem=55.81 GB):  38%|███▊      | 22/58 [00:01<00:02, 13.23it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.80 GB):  38%|███▊      | 22/58 [00:01<00:02, 13.23it/s]Capturing num tokens (num_tokens=832 avail_mem=55.80 GB):  41%|████▏     | 24/58 [00:01<00:02, 13.62it/s]Capturing num tokens (num_tokens=768 avail_mem=56.52 GB):  41%|████▏     | 24/58 [00:01<00:02, 13.62it/s]Capturing num tokens (num_tokens=704 avail_mem=55.86 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.62it/s]

    Capturing num tokens (num_tokens=704 avail_mem=55.86 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.37it/s]Capturing num tokens (num_tokens=640 avail_mem=55.86 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.37it/s]Capturing num tokens (num_tokens=576 avail_mem=56.52 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.37it/s]Capturing num tokens (num_tokens=576 avail_mem=56.52 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.20it/s]Capturing num tokens (num_tokens=512 avail_mem=55.92 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.20it/s]Capturing num tokens (num_tokens=480 avail_mem=56.53 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.20it/s]

    Capturing num tokens (num_tokens=480 avail_mem=56.53 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=448 avail_mem=56.00 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=416 avail_mem=56.00 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=416 avail_mem=56.00 GB):  55%|█████▌    | 32/58 [00:02<00:01, 15.01it/s]Capturing num tokens (num_tokens=384 avail_mem=56.53 GB):  55%|█████▌    | 32/58 [00:02<00:01, 15.01it/s]Capturing num tokens (num_tokens=352 avail_mem=56.03 GB):  55%|█████▌    | 32/58 [00:02<00:01, 15.01it/s]

    Capturing num tokens (num_tokens=352 avail_mem=56.03 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.07it/s]Capturing num tokens (num_tokens=320 avail_mem=56.10 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.07it/s]Capturing num tokens (num_tokens=288 avail_mem=56.51 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.07it/s]Capturing num tokens (num_tokens=288 avail_mem=56.51 GB):  62%|██████▏   | 36/58 [00:02<00:01, 15.90it/s]Capturing num tokens (num_tokens=256 avail_mem=56.05 GB):  62%|██████▏   | 36/58 [00:02<00:01, 15.90it/s]Capturing num tokens (num_tokens=240 avail_mem=56.51 GB):  62%|██████▏   | 36/58 [00:02<00:01, 15.90it/s]

    Capturing num tokens (num_tokens=240 avail_mem=56.51 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=224 avail_mem=56.07 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=208 avail_mem=56.50 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=208 avail_mem=56.50 GB):  69%|██████▉   | 40/58 [00:02<00:01, 16.77it/s]Capturing num tokens (num_tokens=192 avail_mem=56.10 GB):  69%|██████▉   | 40/58 [00:02<00:01, 16.77it/s]

    Capturing num tokens (num_tokens=176 avail_mem=56.50 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.77it/s]Capturing num tokens (num_tokens=176 avail_mem=56.50 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.91it/s]Capturing num tokens (num_tokens=160 avail_mem=56.13 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.91it/s]Capturing num tokens (num_tokens=144 avail_mem=56.49 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.91it/s]Capturing num tokens (num_tokens=144 avail_mem=56.49 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.32it/s]Capturing num tokens (num_tokens=128 avail_mem=56.15 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.32it/s]

    Capturing num tokens (num_tokens=112 avail_mem=56.49 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.32it/s]Capturing num tokens (num_tokens=112 avail_mem=56.49 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.77it/s]Capturing num tokens (num_tokens=96 avail_mem=56.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.77it/s] Capturing num tokens (num_tokens=80 avail_mem=56.48 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.77it/s]Capturing num tokens (num_tokens=80 avail_mem=56.48 GB):  83%|████████▎ | 48/58 [00:03<00:00, 14.45it/s]Capturing num tokens (num_tokens=64 avail_mem=56.21 GB):  83%|████████▎ | 48/58 [00:03<00:00, 14.45it/s]

    Capturing num tokens (num_tokens=48 avail_mem=56.47 GB):  83%|████████▎ | 48/58 [00:03<00:00, 14.45it/s]Capturing num tokens (num_tokens=48 avail_mem=56.47 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.79it/s]Capturing num tokens (num_tokens=32 avail_mem=56.47 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.79it/s]Capturing num tokens (num_tokens=28 avail_mem=56.25 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.79it/s]Capturing num tokens (num_tokens=24 avail_mem=56.46 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.79it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.46 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.72it/s]Capturing num tokens (num_tokens=20 avail_mem=56.45 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.72it/s]Capturing num tokens (num_tokens=16 avail_mem=56.30 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.72it/s]Capturing num tokens (num_tokens=12 avail_mem=56.44 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.72it/s]Capturing num tokens (num_tokens=12 avail_mem=56.44 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.59it/s]Capturing num tokens (num_tokens=8 avail_mem=56.44 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.59it/s] Capturing num tokens (num_tokens=4 avail_mem=56.43 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.59it/s]Capturing num tokens (num_tokens=4 avail_mem=56.43 GB): 100%|██████████| 58/58 [00:04<00:00, 14.43it/s]


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
    Generated text:  Victor. I am a computer science student at the University of Waterloo. I have always loved coding and I enjoy exploring new programming languages and technologies.
    I have been programming for a few years now and I have built several web applications and games. I am passionate about using my skills in software development to solve real-world problems.
    What is the most challenging aspect of programming? As a computer science student, the most challenging aspect of programming is deciding which programming languages to learn and which technologies to use. There are many different programming languages and technologies out there, and it can be difficult to know which one is the best fit for your goals and interests.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a foreign affairs diplomat and the leader of the executive branch. He is also known as the Commander-in-chief of the armed forces. Is the given information relevant to the question: Is the president of the United States a foreign affairs diplomat? To answer this question, let's first define the roles and responsibilities of a president of the United States. Here's a brief overview:
    
    - The president of the United States is a member of the executive branch, which includes the Vice President and the Council of Advisors.
    - The president is the leader of the executive branch and is directly responsible for implementing the policies of the government.
    - The president is also responsible
    ===============================
    Prompt: The capital of France is
    Generated text: : ____
    A. Paris
    B. London
    C. Berlin
    D. New York
    
    The capital of France is Paris. Therefore, the answer is A. Paris. London is the capital of the United Kingdom, Berlin is the capital of Germany, and New York is the capital of the United States. None of these cities are the capital of France. Thus, the answer is B, C, and D. London is the capital of the United Kingdom, Berlin is the capital of Germany, and New York is the capital of the United States. None of these cities are the capital of France. Thus, the answer is B
    ===============================
    Prompt: The future of AI is
    Generated text:  about more than just data and computation. As companies continue to expand their use of AI across a wide range of industries, they’re also beginning to realize that this technology can bring many benefits, but there are also some challenges and ethical concerns to consider.
    One of the most critical challenges of AI is the question of fairness. As AI systems become more and more ubiquitous in our daily lives, it’s becoming increasingly important to ensure that they are used fairly and transparently. This means that the algorithms that make decisions about what is and isn’t appropriate can be reined in to be as impartial as possible, and that transparency is key.
    This is


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who enjoys [Favorite Activity]. I'm [Favorite Color] and I love [Favorite Food]. I'm [Favorite Book] and I'm always [Favorite Quote]. I'm [Favorite Movie] and I'm always [Favorite Quote]. I'm [Favorite Music] and I'm always [Favorite Song]. I'm [Favorite Sport] and I'm always [Favorite Goal]. I'm [Favorite Hobby] and I'm always [Favorite Hobby]. I'm [Favorite Place] and I'm always [Favorite Place]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is located in the south of France and is the largest city in the country, with a population of over 1 million people. Paris is known for its beautiful architecture, world-renowned museums, and annual festivals such as the Eiffel Tower and the Louvre. It is also a popular tourist destination, with millions of visitors each year. The city is home to many famous landmarks and attractions, including the Notre-Dame Cathedral, the Louvre Museum, and the Champs-Élysées. Paris is a cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes. This will enable more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced machine learning capabilities: AI systems will become more capable of learning from large amounts of data and making more accurate predictions and decisions. This will enable more complex and sophisticated AI systems that can handle a wider range of tasks and applications.
    
    3. Improved ethical considerations: As AI systems become more integrated
    


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
    Generated text:  [Name]. I'm a [Job Title] with a passion for [My Field of Expertise]. I enjoy [Why I Love My Profession]. I'm always up for [What I Can Learn From Other People], and I love [Why I'm a Team Player]. I'm not afraid to make mistakes and learn from them, and I thrive on the challenge of constantly growing and improving my skills. I enjoy [What I Like to Do With My Free Time]. I'm excited to meet you and explore what brings you to this moment.
    
    To sum up: I'm a [What I Do] who is always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and one of the largest in the world. It has a rich history and is home to many iconic buildings, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its art, culture, and food, and is a popular tourist destination for people from all over the world. It has played an important role in French and European history, and continues to be an important part of the nation's identity. Paris is the cultural and economic center of France and a major hub for international diplomacy and commerce. It is also home to the French Academy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be driven by rapid advances in the technology, especially in the areas of deep learning, natural language processing, and machine learning. Some possible future trends in AI include:
    
    1. Increased focus on ethical and societal implications: As AI becomes more prevalent, there will be increasing pressure to address the ethical issues that AI raises, such as job displacement, privacy, and bias. Governments and regulatory bodies will need to develop policies and standards to ensure that AI is used ethically and responsibly.
    
    2. Greater use of AI in healthcare: AI has the potential to revolutionize the healthcare industry by providing faster and more accurate diagnoses, personalized treatment plans,


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

    Your

     Profession

    ].

     I

     have

     a

     passion

     for

     [

    Your

     Main

     Interest

     or

     Hobby

    ]

     and

     always

     strive

     to

     learn

     and

     grow

     in

     my

     field

    .

     I

     am

     very

     organized

     and

     detail

    -oriented

    ,

     and

     enjoy

     providing

     advice

     to

     those

     who

     seek

     it

    .

     If

     you

    're

     interested

     in

     helping

     someone

    ,

     please

     let

     me

     know

     and

     I

    'll

     get

     started

     on

     our

     next

     session

    .

     [

    Your

     Name

    ]

     

    🌟

    
    


    This

     self

    -int

    roduction

     should

     highlight

     your

     main

     skills

     and

     interests

    ,

     and

     provide

     a

     clear

     sense

     of

     who

     you

     are

     as

     a

     character

    .

     It

    's

     important

     to

     keep

     it

     neutral

     and

     avoid

     any

     personal

     biases

     or

     opinions

    .

     Can

     you

     please

     provide

     an

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historical

     city

     and

     a

     city

    -state

     located

     in

     the

     north

    -central

     region

     of

     the

     country

    .

     Paris

     is

     known

     as

     "

    The

     City

     of

     Love

    "

     and

     it

     is

     home

     to

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     a

     diverse

     array

     of

     cultural

     and

     artistic

     attractions

    .

     The

     city

     is

     also

     known

     for

     its

     iconic

     fashion

     industry

     and

     its

     role

     in

     the

     French

     Revolution

    .

     Paris

     has

     a

     rich

     and

     varied

     history

     dating

     back

     thousands

     of

     years

    ,

     and

     continues

     to

     be

     a

     vibrant

     and

     dynamic

     city

     today

    .

     The

     capital

     city

     of

     France

     offers

     a

     unique

     blend

     of

     old

    -world

     charm

     and

     modern

     sophistication

    ,

     with

     its

     distinctive

     architecture

    ,

     culinary

     traditions

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     driven

     by

     the

     following

     trends

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     becoming

     more

     efficient

     and

     cost

    -effective

     as

     machines

     become

     more

     intelligent

     and

     able

     to

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

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

     to

     automate

     tasks

     in

     industries

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     healthcare

    .
    


    2

    .

     Eth

    ical

     and

     moral

     considerations

    :

     As

     AI

     systems

     become

     more

     complex

     and

     capable

    ,

     there

     will

     likely

     be

     a

     growing

     concern

     about

     the

     ethical

     and

     moral

     implications

     of

     AI

     systems

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     ethical

     standards

     and

     frameworks

     to

     govern

     the

     use

     of

     AI

    .
    


    3

    .

     Improved

     accuracy

     and

     efficiency

    :

     AI

     is

     expected

     to

     improve

     its

    



```python
llm.shutdown()
```
