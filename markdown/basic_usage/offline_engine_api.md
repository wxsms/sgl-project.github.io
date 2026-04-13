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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.74it/s]


    2026-04-13 08:13:02,140 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 08:13:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:25,  2.55s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:25,  2.55s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:14,  3.57it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:14,  3.57it/s]

    Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.57it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]

    Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:05,  8.28it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 17.57it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 17.57it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 17.57it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 17.57it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 17.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 20.62it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 23.27it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 23.27it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 23.27it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 23.27it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 23.27it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 24.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 24.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 24.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 24.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 24.43it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 26.83it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 26.83it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 26.83it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 26.83it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 26.83it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 27.30it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 27.30it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 27.30it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 27.30it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 27.30it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 29.46it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 29.46it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 29.46it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 29.46it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 29.46it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 30.12it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 30.12it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 30.12it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 30.12it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 30.12it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 31.27it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 31.27it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 31.27it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 31.27it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 31.27it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 31.27it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 33.94it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 33.94it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 33.94it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 33.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=52.87 GB):   2%|▏         | 1/58 [00:00<00:12,  4.67it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.83 GB):   2%|▏         | 1/58 [00:00<00:12,  4.67it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.83 GB):   3%|▎         | 2/58 [00:00<00:12,  4.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.47 GB):   3%|▎         | 2/58 [00:00<00:12,  4.39it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=52.47 GB):   5%|▌         | 3/58 [00:00<00:12,  4.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.43 GB):   5%|▌         | 3/58 [00:00<00:12,  4.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.43 GB):   7%|▋         | 4/58 [00:00<00:10,  4.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.83 GB):   7%|▋         | 4/58 [00:00<00:10,  4.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=52.83 GB):   9%|▊         | 5/58 [00:01<00:10,  4.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.83 GB):   9%|▊         | 5/58 [00:01<00:10,  4.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.83 GB):  10%|█         | 6/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.83 GB):  10%|█         | 6/58 [00:01<00:10,  5.06it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=52.83 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.60 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.60 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=70.90 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.90 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.17 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.19 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.19 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.12 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.71it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.17 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.17 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.94 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.15 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.15 GB):  26%|██▌       | 15/58 [00:02<00:03, 11.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.14 GB):  26%|██▌       | 15/58 [00:02<00:03, 11.91it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.14 GB):  26%|██▌       | 15/58 [00:02<00:03, 11.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.99 GB):  26%|██▌       | 15/58 [00:02<00:03, 11.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.99 GB):  31%|███       | 18/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.13 GB):  31%|███       | 18/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.12 GB):  31%|███       | 18/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.10 GB):  31%|███       | 18/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.10 GB):  36%|███▌      | 21/58 [00:02<00:01, 18.96it/s]Capturing num tokens (num_tokens=960 avail_mem=70.98 GB):  36%|███▌      | 21/58 [00:02<00:01, 18.96it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.10 GB):  36%|███▌      | 21/58 [00:02<00:01, 18.96it/s]Capturing num tokens (num_tokens=832 avail_mem=71.09 GB):  36%|███▌      | 21/58 [00:02<00:01, 18.96it/s]Capturing num tokens (num_tokens=768 avail_mem=71.09 GB):  36%|███▌      | 21/58 [00:02<00:01, 18.96it/s]Capturing num tokens (num_tokens=768 avail_mem=71.09 GB):  43%|████▎     | 25/58 [00:02<00:01, 23.03it/s]Capturing num tokens (num_tokens=704 avail_mem=71.08 GB):  43%|████▎     | 25/58 [00:02<00:01, 23.03it/s]Capturing num tokens (num_tokens=640 avail_mem=71.05 GB):  43%|████▎     | 25/58 [00:02<00:01, 23.03it/s]Capturing num tokens (num_tokens=576 avail_mem=71.05 GB):  43%|████▎     | 25/58 [00:02<00:01, 23.03it/s]Capturing num tokens (num_tokens=512 avail_mem=71.04 GB):  43%|████▎     | 25/58 [00:02<00:01, 23.03it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.04 GB):  50%|█████     | 29/58 [00:02<00:01, 26.17it/s]Capturing num tokens (num_tokens=480 avail_mem=71.07 GB):  50%|█████     | 29/58 [00:02<00:01, 26.17it/s]Capturing num tokens (num_tokens=448 avail_mem=71.06 GB):  50%|█████     | 29/58 [00:02<00:01, 26.17it/s]Capturing num tokens (num_tokens=416 avail_mem=71.06 GB):  50%|█████     | 29/58 [00:02<00:01, 26.17it/s]Capturing num tokens (num_tokens=384 avail_mem=71.05 GB):  50%|█████     | 29/58 [00:02<00:01, 26.17it/s]Capturing num tokens (num_tokens=384 avail_mem=71.05 GB):  57%|█████▋    | 33/58 [00:02<00:00, 29.02it/s]Capturing num tokens (num_tokens=352 avail_mem=71.04 GB):  57%|█████▋    | 33/58 [00:02<00:00, 29.02it/s]Capturing num tokens (num_tokens=320 avail_mem=71.03 GB):  57%|█████▋    | 33/58 [00:02<00:00, 29.02it/s]Capturing num tokens (num_tokens=288 avail_mem=71.02 GB):  57%|█████▋    | 33/58 [00:02<00:00, 29.02it/s]Capturing num tokens (num_tokens=256 avail_mem=71.02 GB):  57%|█████▋    | 33/58 [00:02<00:00, 29.02it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.02 GB):  64%|██████▍   | 37/58 [00:02<00:00, 31.64it/s]Capturing num tokens (num_tokens=240 avail_mem=71.01 GB):  64%|██████▍   | 37/58 [00:02<00:00, 31.64it/s]Capturing num tokens (num_tokens=224 avail_mem=71.01 GB):  64%|██████▍   | 37/58 [00:02<00:00, 31.64it/s]Capturing num tokens (num_tokens=208 avail_mem=70.98 GB):  64%|██████▍   | 37/58 [00:02<00:00, 31.64it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:02<00:00, 31.64it/s]Capturing num tokens (num_tokens=176 avail_mem=70.98 GB):  64%|██████▍   | 37/58 [00:02<00:00, 31.64it/s]Capturing num tokens (num_tokens=176 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.82it/s]Capturing num tokens (num_tokens=160 avail_mem=70.97 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.82it/s]Capturing num tokens (num_tokens=144 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.82it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.82it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.82it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.97 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.82it/s] Capturing num tokens (num_tokens=96 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:02<00:00, 37.25it/s]Capturing num tokens (num_tokens=80 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:02<00:00, 37.25it/s]Capturing num tokens (num_tokens=64 avail_mem=70.95 GB):  81%|████████  | 47/58 [00:02<00:00, 37.25it/s]Capturing num tokens (num_tokens=48 avail_mem=70.95 GB):  81%|████████  | 47/58 [00:02<00:00, 37.25it/s]Capturing num tokens (num_tokens=32 avail_mem=70.93 GB):  81%|████████  | 47/58 [00:03<00:00, 37.25it/s]Capturing num tokens (num_tokens=32 avail_mem=70.93 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.44it/s]Capturing num tokens (num_tokens=28 avail_mem=70.91 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.44it/s]Capturing num tokens (num_tokens=24 avail_mem=70.93 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.44it/s]Capturing num tokens (num_tokens=20 avail_mem=70.92 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.44it/s]Capturing num tokens (num_tokens=16 avail_mem=70.92 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.44it/s]

    Capturing num tokens (num_tokens=12 avail_mem=70.91 GB):  88%|████████▊ | 51/58 [00:03<00:00, 37.44it/s]Capturing num tokens (num_tokens=12 avail_mem=70.91 GB):  97%|█████████▋| 56/58 [00:03<00:00, 39.29it/s]Capturing num tokens (num_tokens=8 avail_mem=70.91 GB):  97%|█████████▋| 56/58 [00:03<00:00, 39.29it/s] Capturing num tokens (num_tokens=4 avail_mem=70.88 GB):  97%|█████████▋| 56/58 [00:03<00:00, 39.29it/s]Capturing num tokens (num_tokens=4 avail_mem=70.88 GB): 100%|██████████| 58/58 [00:03<00:00, 18.13it/s]


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
    Generated text:  John. I am 14 years old. I have been learning English for two years. My English teacher is Mrs. White. She is very kind. She is strict and strict. She often makes us listen to English songs, but she thinks listening to English songs is good for us. She also often helps us learn the new words. She also often gives us some exercises. Her favorite subject is math. She also teaches us to do our homework and read the newspaper. She is very nice. She is my English teacher. She is very patient. She is not strict when we are in trouble. I have learned English well.
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected for a four-year term. What is the probability that the president will serve three years, given that they have served two years?
    
    To determine the probability that the president will serve three years given that they have served two years, we need to consider the nature of the presidential term and the possible outcomes.
    
    1. **Identify the total number of possible outcomes:**
       - The president can serve for 1, 2, 3, or 4 years.
       - There are 5 possible outcomes: \( S_1, S_2, S_3, S_4 \).
    
    2. **Identify the
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the _______ of the island of the same name.
    A. southwest
    B. northeast
    C. southeast
    D. northwest
    Answer:
    
    C
    
    If a country has a net population growth, then the country is ____. 
    A. Low population growth rate
    B. High population growth rate
    C. The same as the current population
    D. Unable to determine
    Answer:
    
    A
    
    Which of the following statements about the environmental carrying capacity is correct?
    A. The environmental carrying capacity refers to the maximum number of units of a particular natural resource that can be produced in a unit area of a specific type of land.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  here. In 2019, data generation tools like the one created by DeepMind AI will be used to power a $4 billion AI drug discovery project, advance 5G networks, improve the health of millions of lives, and even improve the accuracy of forensic systems. But, just like in the early days of the internet, there are some issues that have yet to be addressed. One problem is data privacy. Another is the integration of AI into the current health care system. Here's a look at the first of these challenges, a potential solution. Dr. Arifshah Ali, executive director of the Health Data


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


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic hub, with a diverse population and a thriving arts scene. It is a popular tourist destination and a major center for politics, business, and science. The city is home to many famous museums, including the Louvre, the Musée d'Or
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more and more AI systems are being developed, there is a growing awareness of the potential ethical implications of AI. This includes issues such as bias, transparency, accountability, and the impact of AI on society as a whole. As a result, there is likely to be increased focus on ethical AI, with more developers and policymakers working to ensure that AI systems are developed and used in a way that
    


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
    Generated text:  ____. I am a software engineer at ____. I specialize in ____. I have over __ years of experience in ____. What are your skills and what languages do you speak? I’m always looking for new challenges to solve. I’m excited to learn and grow, and I’m always ready to contribute to the company’s growth. What’s your dream job?
    
    Hello, my name is ____. I am a software engineer at ____. I specialize in ____. I have over __ years of experience in ____. What are your skills and what languages do you speak? I’m always looking for new challenges to solve. I’m excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and is the capital of metropolitan France and the largest city in metropolitan France. Paris is known for its stunning architecture, rich history, and lively atmosphere. It is a UNESCO World Heritage site and one of the most visited cities in the world. The city is also home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a cultural and artistic center that attracts millions of visitors each year, making it one of the most important cities in the world. Paris is also known for its cuisine, fashion, and music scene.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by an increase in the sophistication and integration of AI into various aspects of society. This could include improvements in the accuracy and efficiency of AI systems, as well as the integration of AI into new domains of human activity, such as healthcare, transportation, and education. Additionally, there is likely to be an increase in the use of AI in areas that are considered “hard” or “hard to quantify” in terms of human decision-making, such as natural language processing and robotics.
    
    Another possible trend in AI is the increasing use of AI in a variety of industries, including the pharmaceutical industry and manufacturing. This could lead to the


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

     __

    ________

    .

     I

    'm

     an

     AI

     language

     model

     created

     by

     Alibaba

     Cloud

    ,

     and

     I

    'm

     here

     to

     assist

     you

     with

     any

     questions

     or

     tasks

     you

     may

     have

    .

     How

     can

     I

     help

     you

     today

    ?

     You

     can

     either

     ask

     me

     a

     question

     or

     give

     me

     a

     task

     to

     complete

    .

     Let

     me

     know

     what

     you

     need

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

    're

     looking

     for

    .

     So

    ,

     what

    's

     on

     your

     mind

    ?

     Let

    's

     get

     started

    !

     What

    's

     your

     name

    ?

     How

     can

     I

     help

     you

     today

    ?

     How

     can

     I

     assist

     you

    ?

     What

    's

     your

     name

    ?

     What

     task

     would

     you

     like

     to

     complete

    ?

     How

     can

     I

     assist

     you

     today

    ?

     What

    's

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    ,

     with

     a

     population

     of

     over

     

    1

    1

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

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     world

    -class

     museums

    ,

     as

     well

     as

     its

     iconic

     landmarks

     such

     as

     Notre

     Dame

     de

     Paris

    ,

     Lou

    vre

     Museum

    ,

     and

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     a

     major

     hub

     for

     the

     French

     economy

    ,

     with

     a

     thriving

     culture

    ,

     tourism

    ,

     and

     international

     business

     community

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     glamorous

     night

     life

     and

     cultural

     influence

    .

     Overall

    ,

     Paris

     is

     a

     quint

    ess

    entially

     French

     city

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     growing

     emphasis

     on

     developing

     more

     sophisticated

     and

     powerful

     AI

     systems

    .

     This

     may

     include

     the

     development

     of

     more

     powerful

     hardware

     and

     software

    ,

     as

     well

     as

     the

     development

     of

     more

     advanced

     models

     of

     AI

     that

     are

     better

     able

     to

     understand

     and

     reason

     about

     complex

     information

    .
    


    One

     possible

     future

     trend

     is

     the

     development

     of

     AI

     that

     is

     capable

     of

     performing

     tasks

     that

     were

     previously

     thought

     to

     be

     too

     complex

     or

     difficult

     for

     AI

     systems

     to

     handle

    .

     For

     example

    ,

     in

     the

     future

    ,

     we

     may

     see

     the

     development

     of

     AI

     that

     is

     capable

     of

     understanding

     human

     emotions

     and

     language

    ,

     or

     that

     is

     capable

     of

     performing

     tasks

     that

     were

     previously

     thought

     to

     require

     complex

     decision

    -making

     and

     problem

    -solving

    .
    


    Another

     possible

     future

    



```python
llm.shutdown()
```
