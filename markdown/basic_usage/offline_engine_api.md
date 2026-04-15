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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.15it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.14it/s]


    2026-04-15 21:22:21,718 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 21:22:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.64it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.64it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.64it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.64it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.31it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.31it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.31it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.41it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.41it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.74it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.74it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.74it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.74it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 13.74it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 18.35it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 26.40it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:00, 33.40it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]

    Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 47.76it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 53.14it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 53.14it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 53.14it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 53.14it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 53.14it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 53.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.69 GB):   2%|▏         | 1/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.66 GB):   2%|▏         | 1/58 [00:00<00:07,  7.49it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.66 GB):   3%|▎         | 2/58 [00:00<00:07,  7.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.65 GB):   3%|▎         | 2/58 [00:00<00:07,  7.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.65 GB):   5%|▌         | 3/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.65 GB):   5%|▌         | 3/58 [00:00<00:07,  7.64it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=50.65 GB):   7%|▋         | 4/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.65 GB):   7%|▋         | 4/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.65 GB):   7%|▋         | 4/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.65 GB):  10%|█         | 6/58 [00:00<00:05,  9.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.65 GB):  10%|█         | 6/58 [00:00<00:05,  9.35it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=50.65 GB):  10%|█         | 6/58 [00:00<00:05,  9.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.65 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.65 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.64 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.30it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=50.64 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.64 GB):  21%|██        | 12/58 [00:01<00:03, 13.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.63 GB):  21%|██        | 12/58 [00:01<00:03, 13.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.63 GB):  21%|██        | 12/58 [00:01<00:03, 13.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.63 GB):  24%|██▍       | 14/58 [00:01<00:02, 14.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.63 GB):  24%|██▍       | 14/58 [00:01<00:02, 14.76it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=50.09 GB):  24%|██▍       | 14/58 [00:01<00:02, 14.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.09 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.08 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.67it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=50.08 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.08 GB):  31%|███       | 18/58 [00:01<00:04,  9.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.08 GB):  31%|███       | 18/58 [00:01<00:04,  9.06it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=50.07 GB):  31%|███       | 18/58 [00:01<00:04,  9.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.07 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.07 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.35it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=50.07 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.13it/s]Capturing num tokens (num_tokens=960 avail_mem=50.07 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.13it/s] Capturing num tokens (num_tokens=960 avail_mem=50.07 GB):  38%|███▊      | 22/58 [00:02<00:04,  7.95it/s]Capturing num tokens (num_tokens=896 avail_mem=50.07 GB):  38%|███▊      | 22/58 [00:02<00:04,  7.95it/s]

    Capturing num tokens (num_tokens=832 avail_mem=50.06 GB):  38%|███▊      | 22/58 [00:02<00:04,  7.95it/s]Capturing num tokens (num_tokens=832 avail_mem=50.06 GB):  41%|████▏     | 24/58 [00:02<00:03,  9.44it/s]Capturing num tokens (num_tokens=768 avail_mem=50.06 GB):  41%|████▏     | 24/58 [00:02<00:03,  9.44it/s]Capturing num tokens (num_tokens=704 avail_mem=50.06 GB):  41%|████▏     | 24/58 [00:02<00:03,  9.44it/s]Capturing num tokens (num_tokens=704 avail_mem=50.06 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.19it/s]Capturing num tokens (num_tokens=640 avail_mem=50.05 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.19it/s]

    Capturing num tokens (num_tokens=576 avail_mem=50.05 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.19it/s]Capturing num tokens (num_tokens=512 avail_mem=50.06 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.19it/s]Capturing num tokens (num_tokens=480 avail_mem=50.06 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.19it/s]Capturing num tokens (num_tokens=480 avail_mem=50.06 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=448 avail_mem=50.06 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=416 avail_mem=50.06 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=384 avail_mem=50.05 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.86it/s]Capturing num tokens (num_tokens=352 avail_mem=50.05 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.86it/s]

    Capturing num tokens (num_tokens=352 avail_mem=50.05 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.71it/s]Capturing num tokens (num_tokens=320 avail_mem=71.14 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.71it/s]Capturing num tokens (num_tokens=288 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.71it/s]Capturing num tokens (num_tokens=256 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.71it/s]Capturing num tokens (num_tokens=240 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.71it/s]Capturing num tokens (num_tokens=224 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.71it/s]Capturing num tokens (num_tokens=224 avail_mem=71.13 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.06it/s]Capturing num tokens (num_tokens=208 avail_mem=71.12 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.06it/s]Capturing num tokens (num_tokens=192 avail_mem=71.12 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.06it/s]Capturing num tokens (num_tokens=176 avail_mem=71.12 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.06it/s]Capturing num tokens (num_tokens=160 avail_mem=71.11 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.06it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.11 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.06it/s]Capturing num tokens (num_tokens=144 avail_mem=71.11 GB):  76%|███████▌  | 44/58 [00:03<00:00, 23.21it/s]Capturing num tokens (num_tokens=128 avail_mem=71.11 GB):  76%|███████▌  | 44/58 [00:03<00:00, 23.21it/s]Capturing num tokens (num_tokens=112 avail_mem=71.10 GB):  76%|███████▌  | 44/58 [00:03<00:00, 23.21it/s]Capturing num tokens (num_tokens=96 avail_mem=71.10 GB):  76%|███████▌  | 44/58 [00:03<00:00, 23.21it/s] Capturing num tokens (num_tokens=80 avail_mem=71.10 GB):  76%|███████▌  | 44/58 [00:03<00:00, 23.21it/s]Capturing num tokens (num_tokens=64 avail_mem=71.09 GB):  76%|███████▌  | 44/58 [00:03<00:00, 23.21it/s]Capturing num tokens (num_tokens=64 avail_mem=71.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.21it/s]Capturing num tokens (num_tokens=48 avail_mem=71.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.21it/s]Capturing num tokens (num_tokens=32 avail_mem=71.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.21it/s]Capturing num tokens (num_tokens=28 avail_mem=71.08 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.21it/s]Capturing num tokens (num_tokens=24 avail_mem=71.08 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.21it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.08 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.21it/s]Capturing num tokens (num_tokens=20 avail_mem=71.08 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.82it/s]Capturing num tokens (num_tokens=16 avail_mem=71.08 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.82it/s]Capturing num tokens (num_tokens=12 avail_mem=71.07 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.82it/s]Capturing num tokens (num_tokens=8 avail_mem=71.07 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.82it/s] Capturing num tokens (num_tokens=4 avail_mem=71.06 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.82it/s]Capturing num tokens (num_tokens=4 avail_mem=71.06 GB): 100%|██████████| 58/58 [00:03<00:00, 15.82it/s]


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
    Generated text:  Daria and I'm 22 years old, and I'm a female. I would like to know more about the history of science in my country, and how I can contribute to scientific advancement.
    
    Certainly! Science has a rich history in many countries and across many fields, including medicine, physics, chemistry, biology, and more. Here are some key points to help you understand the history of science in your country and how you can contribute to scientific advancement:
    
    ### History of Science in Your Country
    
    1. **Ancient Civilizations**:
       - **Egypt**: The ancient Egyptians, who were highly skilled in the sciences, developed
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is like a leader. He makes decisions about the country. This year, he made a decision to change the way that the president is elected. He will ask for the help of more people in the future. So, the people in the United States have a lot of questions. They want to know when will he ask for the help of more people. It's a very big decision for him. The president made a decision to change the way he is elected. He will ask for more people's help. That's why people in the United States have a lot of questions. They want to know when will
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of France. A capital is a city in which a legislature meets to make rules for the governance of the state. The city is not itself a country.
    Does this next sentence follow, given the above sentence. The capital of France is in the capital of France.
    Choices: (a). yes. (b). it is not possible to tell. (c). no.
    
    (a).
    ===============================
    Prompt: The future of AI is
    Generated text:  becoming increasingly promising, with many researchers, developers, and companies looking to harness the power of artificial intelligence to make the world a better place. But what exactly does AI really do, and how can it be used to address some of the biggest challenges facing humanity?
    To explore this question, let's take a look at the different areas of AI that are currently being explored and how they are being used to solve real-world problems. We'll also take a look at some of the challenges that still need to be addressed, and how AI is being used to address them.
    The Future of AI: A Look at the Current State
    The field of


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many famous museums, theaters, and restaurants. The city is known for its fashion industry, with Paris Fashion Week being one of the world's largest and most prestigious fashion events. Overall, Paris is a city of contrasts and beauty, with its unique blend of history, culture, and modernity.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, and transportation. Automation will likely lead to the development of new types of AI, such as machine learning and deep learning, which will enable machines to perform tasks that were previously done by humans.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security measures
    


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
    Generated text:  [Name], and I'm a [role] at [company]. I'm excited to join you all here today to share my insights and expertise in [topic]. I look forward to learning from you all and contributing to your discussions. What can you tell me about yourself? As an AI language model, my training includes a vast amount of knowledge and capabilities. I can analyze and interpret large amounts of text, generate human-like responses, and provide feedback on writing and communication. I am also able to understand and respond to a wide range of topics and subject areas, from science and technology to literature and history. In summary, I am a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    [Mark down]
    **Statement:** The capital of France is Paris. [End of statement]
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by two main trends: the expansion of AI into new areas, particularly beyond the realm of traditional computing tasks, and the development of more sophisticated algorithms that can learn from large amounts of data and adapt to new situations. 
    
    One of the most significant trends is the expansion of AI into new areas. This trend has already begun with the development of new technologies such as quantum computing, which has the potential to revolutionize the field of artificial intelligence. Other potential areas of expansion include natural language processing, computer vision, and robotics.
    
    Another trend is the development of more sophisticated algorithms that can learn from large amounts of data and adapt to


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

     am

     a

     [

    background

    ]

     individual

     who

     is

     known

     for

     [

    what

     you

     do

    ]

     in

     [

    your

     current

     location

     or

     field

    ].

     I

     am

     excited

     to

     introduce

     myself

     to

     you

     today

     and

     see

     how

     we

     can

     work

     together

     to

     make

     a

     positive

     impact

     in

     the

     world

    .

     What

     is

     your

     name

    ,

     and

     what

     is

     your

     role

     in

     the

     organization

     you

     work

     for

    ?


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

     am

     a

     [

    background

    ]

     individual

     who

     is

     known

     for

     [

    what

     you

     do

    ]

     in

     [

    your

     current

     location

     or

     field

    ].

     I

     am

     excited

     to

     introduce

     myself

     to

     you

     today

     and

     see

     how

     we

     can

     work

     together

     to

     make

     a

     positive

     impact

     in

     the

     world

    .

    
    
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

     France

     and

     a

     major

     tourist

     destination

    .

     Paris

     is

     home

     to

     iconic

     landmarks

     such

     as

     the

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

    ,

     and

     has

     a

     rich

     history

     dating

     back

     to

     Roman

     times

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     which

     is

     a

     major

     part

     of

     the

     French

     culture

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     sleek

     architecture

     and

     charming

     neighborhoods

    ,

     as

     well

     as

     a

     diverse

     population

     of

     over

     

    2

     million

     people

    .

     Its

     cultural

    ,

     culinary

    ,

     and

     political

     influence

     extends

     far

     beyond

     its

     borders

    ,

     and

     is

     a

     major

     hub

     for

     international

     trade

     and

     diplomacy

    .

     The

     city

    's

     status

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     automation

     and

     precision

    :

     With

     the

     development

     of

     machine

     learning

     and

     deep

     learning

     algorithms

    ,

     we

     can

     expect

     to

     see

     a

     shift

     in

     AI

     towards

     more

     precise

    ,

     efficient

    ,

     and

     human

    -like

     applications

    .

     This

     could

     lead

     to

     the

     development

     of

     robots

     that

     can

     perform

     complex

     tasks

     with

     greater

     accuracy

     and

     speed

     than

     humans

    ,

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    .
    


    2

    .

     AI

     ethics

     and

     governance

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     are

     likely

     to

     see

     more

     efforts

     to

     develop

     ethical

     and

     responsible

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

     regulations

     and

     guidelines

     for

     the

     development

     and

     use

     of

     AI

    ,

     with

     a

    



```python
llm.shutdown()
```
