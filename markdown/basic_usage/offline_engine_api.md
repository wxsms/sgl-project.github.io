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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.36it/s]


    2026-04-10 20:28:51,727 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 20:28:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.17s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.17s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.17s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.08it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.08it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.08it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.55it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.55it/s]

    Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.55it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.55it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.20it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.20it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.20it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.20it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.23it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.23it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 16.74it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 16.74it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 16.74it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 16.74it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 16.74it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 20.21it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 20.21it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 20.21it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 20.21it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 20.21it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 23.45it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 23.45it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 23.45it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 23.45it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 23.45it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 26.03it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 26.03it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 26.03it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 26.03it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 26.03it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.76it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.76it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 28.76it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 28.76it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 28.76it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 30.06it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 30.06it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 30.06it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 30.06it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 30.06it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 31.73it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 31.73it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 31.73it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 31.73it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 31.73it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 31.54it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 31.54it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 31.54it/s]

    Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 31.54it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 31.54it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 32.00it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 32.00it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 32.00it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 32.00it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 32.00it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 33.20it/s] 

    Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.44 GB):   2%|▏         | 1/58 [00:00<00:09,  5.96it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.41 GB):   2%|▏         | 1/58 [00:00<00:09,  5.96it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.41 GB):   3%|▎         | 2/58 [00:00<00:08,  6.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.40 GB):   3%|▎         | 2/58 [00:00<00:08,  6.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.40 GB):   5%|▌         | 3/58 [00:00<00:07,  7.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.40 GB):   5%|▌         | 3/58 [00:00<00:07,  7.15it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.40 GB):   7%|▋         | 4/58 [00:00<00:08,  6.58it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.87 GB):   7%|▋         | 4/58 [00:00<00:08,  6.58it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=52.87 GB):   9%|▊         | 5/58 [00:00<00:09,  5.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.86 GB):   9%|▊         | 5/58 [00:00<00:09,  5.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.86 GB):  10%|█         | 6/58 [00:00<00:09,  5.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.87 GB):  10%|█         | 6/58 [00:00<00:09,  5.74it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=52.87 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.86 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.92it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.86 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.86 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.86 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.86 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.32it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=52.86 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.85 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.85 GB):  19%|█▉        | 11/58 [00:01<00:08,  5.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.85 GB):  19%|█▉        | 11/58 [00:01<00:08,  5.86it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.85 GB):  21%|██        | 12/58 [00:02<00:07,  6.22it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.85 GB):  21%|██        | 12/58 [00:02<00:07,  6.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.85 GB):  21%|██        | 12/58 [00:02<00:07,  6.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.85 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.84 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.84 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.21it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=52.83 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.83 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.18 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.18 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.08it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=71.18 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.17 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.15 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.84it/s]Capturing num tokens (num_tokens=960 avail_mem=71.16 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.84it/s] Capturing num tokens (num_tokens=960 avail_mem=71.16 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.34it/s]Capturing num tokens (num_tokens=896 avail_mem=71.16 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.34it/s]Capturing num tokens (num_tokens=832 avail_mem=71.16 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.34it/s]Capturing num tokens (num_tokens=768 avail_mem=71.15 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.34it/s]Capturing num tokens (num_tokens=704 avail_mem=71.15 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.34it/s]

    Capturing num tokens (num_tokens=704 avail_mem=71.15 GB):  45%|████▍     | 26/58 [00:02<00:01, 19.93it/s]Capturing num tokens (num_tokens=640 avail_mem=71.15 GB):  45%|████▍     | 26/58 [00:02<00:01, 19.93it/s]Capturing num tokens (num_tokens=576 avail_mem=71.15 GB):  45%|████▍     | 26/58 [00:02<00:01, 19.93it/s]Capturing num tokens (num_tokens=512 avail_mem=71.14 GB):  45%|████▍     | 26/58 [00:02<00:01, 19.93it/s]Capturing num tokens (num_tokens=512 avail_mem=71.14 GB):  50%|█████     | 29/58 [00:02<00:01, 22.03it/s]Capturing num tokens (num_tokens=480 avail_mem=71.15 GB):  50%|█████     | 29/58 [00:02<00:01, 22.03it/s]Capturing num tokens (num_tokens=448 avail_mem=71.15 GB):  50%|█████     | 29/58 [00:02<00:01, 22.03it/s]Capturing num tokens (num_tokens=416 avail_mem=71.15 GB):  50%|█████     | 29/58 [00:02<00:01, 22.03it/s]Capturing num tokens (num_tokens=384 avail_mem=71.15 GB):  50%|█████     | 29/58 [00:02<00:01, 22.03it/s]Capturing num tokens (num_tokens=352 avail_mem=71.14 GB):  50%|█████     | 29/58 [00:02<00:01, 22.03it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.14 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.13it/s]Capturing num tokens (num_tokens=320 avail_mem=71.14 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.13it/s]Capturing num tokens (num_tokens=288 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.13it/s]Capturing num tokens (num_tokens=256 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.13it/s]Capturing num tokens (num_tokens=240 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.13it/s]Capturing num tokens (num_tokens=240 avail_mem=71.13 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.10it/s]Capturing num tokens (num_tokens=224 avail_mem=71.12 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.10it/s]Capturing num tokens (num_tokens=208 avail_mem=71.12 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.10it/s]Capturing num tokens (num_tokens=192 avail_mem=71.12 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.10it/s]Capturing num tokens (num_tokens=176 avail_mem=71.12 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.10it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.12 GB):  72%|███████▏  | 42/58 [00:03<00:00, 26.92it/s]Capturing num tokens (num_tokens=160 avail_mem=70.54 GB):  72%|███████▏  | 42/58 [00:03<00:00, 26.92it/s]Capturing num tokens (num_tokens=144 avail_mem=71.08 GB):  72%|███████▏  | 42/58 [00:03<00:00, 26.92it/s]Capturing num tokens (num_tokens=128 avail_mem=70.61 GB):  72%|███████▏  | 42/58 [00:03<00:00, 26.92it/s]Capturing num tokens (num_tokens=128 avail_mem=70.61 GB):  78%|███████▊  | 45/58 [00:03<00:00, 22.77it/s]Capturing num tokens (num_tokens=112 avail_mem=71.07 GB):  78%|███████▊  | 45/58 [00:03<00:00, 22.77it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.64 GB):  78%|███████▊  | 45/58 [00:03<00:00, 22.77it/s] Capturing num tokens (num_tokens=80 avail_mem=71.06 GB):  78%|███████▊  | 45/58 [00:03<00:00, 22.77it/s]Capturing num tokens (num_tokens=80 avail_mem=71.06 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.05it/s]Capturing num tokens (num_tokens=64 avail_mem=71.06 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.05it/s]Capturing num tokens (num_tokens=48 avail_mem=71.03 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.05it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.06 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.05it/s]Capturing num tokens (num_tokens=32 avail_mem=71.06 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.62it/s]Capturing num tokens (num_tokens=28 avail_mem=70.71 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.62it/s]Capturing num tokens (num_tokens=24 avail_mem=71.04 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.62it/s]Capturing num tokens (num_tokens=20 avail_mem=71.04 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.62it/s]Capturing num tokens (num_tokens=20 avail_mem=71.04 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.05it/s]Capturing num tokens (num_tokens=16 avail_mem=70.76 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.05it/s]

    Capturing num tokens (num_tokens=12 avail_mem=71.04 GB):  93%|█████████▎| 54/58 [00:04<00:00, 21.05it/s]Capturing num tokens (num_tokens=8 avail_mem=71.03 GB):  93%|█████████▎| 54/58 [00:04<00:00, 21.05it/s] Capturing num tokens (num_tokens=8 avail_mem=71.03 GB):  98%|█████████▊| 57/58 [00:04<00:00, 21.67it/s]Capturing num tokens (num_tokens=4 avail_mem=70.80 GB):  98%|█████████▊| 57/58 [00:04<00:00, 21.67it/s]Capturing num tokens (num_tokens=4 avail_mem=70.80 GB): 100%|██████████| 58/58 [00:04<00:00, 13.95it/s]


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
    Generated text:  Alice. I love to play basketball. I play it with my friends on Saturdays. We practice and then play every day. My favorite basketball player is Yao Ming. He is the best basketball player in the world. He can shoot the ball from 30 meters away. He plays for the Houston Rockets. He is a star in basketball. He plays basketball on Sundays. I go to his home. He lives in New York. I want to be his basketball player one day. He is a very kind person. He always helps people. I hope to be a basketball player one day too. I have two brothers. One is a
    ===============================
    Prompt: The president of the United States is
    Generated text:  planning a trip to the United States. To get there, he must travel by airplane. The cost of each round trip is $350. The president plans to return home by car, which costs him $30 per day for 7 days. Calculate the total cost of the trip to the United States for the president. To calculate the total cost of the trip to the United States for the president, we need to consider both the cost of the round trip airplane ticket and the cost of the car trip.
    
    1. **Calculate the cost of the round trip airplane ticket:**
       - The cost of one round trip is $
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Marseille
    C. Lille
    D. Bordeaux
    Answer:
    
    A
    
    Which of the following sentences does NOT contain a word with a different meaning in the past tense and present tense?
    A. He is diligent.
    B. He is diligent now.
    C. He had been diligent before.
    D. He is now diligent.
    Answer:
    
    B
    
    Among the following options, which one has the same meaning for the word 'her' as in 'the other her'? A. My friend once borrowed a book from her. B. The dog thinks it's my dog. C. She took the man
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and complex. As technology continues to evolve, so too does the role that AI plays in society. Some experts predict that AI will continue to revolutionize industries such as healthcare, finance, and transportation, while others see it as a threat to jobs and privacy. What are some potential outcomes of the future of AI?  Answer the above questions based on the below text.  Text: The future of AI is uncertain and complex. As technology continues to evolve, so too does the role that AI plays in society. Some experts predict that AI will continue to revolutionize industries such as healthcare, finance, and transportation, while others see it


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Website URL] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Website URL] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [LinkedIn Profile] [Twitter
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major transportation hub, with the Paris Metro and the Eiffel Tower serving as major modes of transportation. Paris is a popular tourist destination, with millions of visitors each year. The city is also home to many important institutions, including the Louvre Museum,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to improve, we can expect to see even more innovative applications emerge. Additionally, there is a growing concern about the ethical implications of AI, including issues such as bias, transparency, and accountability. As these issues are addressed, we can expect to see further developments in AI that are more aligned with ethical principles. Overall, the future of AI is likely to be one of continued innovation and progress
    


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
    Generated text:  [Name] and I'm a [职业] who has a passion for [职业的活动或成就] and I'm always looking for ways to grow my skills and knowledge. Whether it's through reading, writing, or any other hobby, I'm always trying to expand my horizons and make new friends. I enjoy sharing my knowledge and inspiring others to do the same, so I'm always eager to learn and help others. I'm looking forward to learning more about your day and our conversation about our shared interests. What's your name, and what do you do for a living? Hey, I'm [Name] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The factual statement is: 
    
    Paris is the largest city in France. 
    Is the given statement true or false? The statement is false. Paris is not the largest city in France. It is the capital and largest city of France by population. The largest city in France by population is Paris itself. However, Paris is the capital city and its status as a major cultural, economic, and political center is unparalleled. The statement mistakenly claims that it is the largest city, which is not true. 
    The correct statement would be: "Paris is the capital of France, and it is the largest city by population among the 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve several trends that will shape the technology's development, applications, and ethical considerations. Here are some of the most likely future trends:
    
    1. Advancements in machine learning and deep learning: The use of machine learning and deep learning algorithms to improve AI models and capabilities is expected to continue. This will lead to the development of even more complex and sophisticated AI systems that can learn from vast amounts of data and adapt to new situations.
    
    2. Increased focus on ethical AI: As AI systems become more advanced, there is a growing emphasis on addressing ethical concerns such as bias, privacy, and accountability. AI developers will need to prioritize these


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

    First

     Name

    ],

     and

     I

    'm

     [

    Last

     Name

    ]

     from

     [

    Your

     Country

    /

    Location

    ].

     I

    'm

     currently

     [

    insert

     the

     year

     in

     which

     you

     were

     born

    ]

     and

     have

     always

     been

     passionate

     about

     [

    a

     hobby

    ,

     interest

    ,

     or

     activity

    ].

     I

    've

     traveled

     the

     world

     and

     have

     visited

     [

    insert

     the

     countries

     you

    've

     visited

    ],

     and

     I

    'm

     a

     [

    insert

     a

     personal

     trait

     or

     characteristic

     that

     you

    're

     proud

     of

    ]

     who

     has

     led

     me

     through

     the

     trials

     and

     trib

    ulations

     of

     life

    .

     I

    'm

     always

     looking

     for

     new

     experiences

    ,

     learning

    ,

     and

     growing

    .

     I

    'm

     the

     type

     of

     person

     who

     believes

     in

     the

     power

     of

     [

    insert

     something

     positive

     or

     inspirational

    ]

     to

     change

     the

     world

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     world

    -ren

    owned

     museums

     and

     landmarks

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     festivals

     and

     cultural

     events

    .

     Paris

     is

     a

     cultural

     hub

     of

     the

     world

     and

     a

     major

     transportation

     and

     business

     center

    .

     It

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     many

     others

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     city

     is

     known

     for

     its

     romantic

     and

     historical

     atmosphere

    .

     The

     city

     is

     also

     known

     for

     its

     vibrant

     nightlife

     and

     shopping

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     beautiful

     architecture

    ,

     quirky

     streets

    ,

     and

     diverse

     cultural

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     uncertain

    ,

     but

     there

     are

     several

     potential

     trends

     that

     could

     shape

     its

     development

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

     prevalent

     in

     almost

     every

     industry

    ,

     from

     healthcare

     to

     finance

    ,

     manufacturing

    ,

     and

     education

    .

     With

     the

     integration

     of

     automation

    ,

     AI

     could

     handle

     routine

     tasks

    ,

     freeing

     up

     human

     workers

     to

     focus

     on

     more

     creative

     and

     complex

     work

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     allow

     for

     more

     personalized

     interactions

     with

     users

    .

     As

     AI

     algorithms

     learn

     and

     adapt

     to

     users

    '

     behavior

    ,

     they

     can

     provide

     increasingly

     accurate

     and

     relevant

     recommendations

    ,

     such

     as

     personalized

     shopping

    ,

     entertainment

    ,

     and

     education

    .
    


    3

    .

     Aug

    mented

     intelligence

    :

     AI

     will

     continue

     to

     evolve

     and

     integrate

     with

     humans

    ,

    



```python
llm.shutdown()
```
