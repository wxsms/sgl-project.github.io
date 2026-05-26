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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.85it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.85it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.85it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:28,  1.85it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.94it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.94it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.94it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.94it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.94it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.72it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.72it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.72it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 12.72it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 12.72it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 16.66it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 16.66it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 16.66it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 16.66it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 16.66it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 16.66it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 21.98it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 30.71it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 30.71it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 30.71it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 30.71it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 30.71it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:00, 30.71it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 37.66it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 37.66it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 37.66it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 37.66it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 37.66it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 37.66it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 40.30it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 43.66it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 43.66it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 43.66it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 43.66it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 43.66it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 43.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:07,  7.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:07,  7.52it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:06,  7.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:06,  7.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.90 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.56it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.62it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:01<00:04,  9.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:04, 10.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:04, 10.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:04, 10.25it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.86 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.08it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.83 GB):  31%|███       | 18/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.83 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.19it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=52.84 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.19it/s]Capturing num tokens (num_tokens=896 avail_mem=52.84 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.17it/s]Capturing num tokens (num_tokens=832 avail_mem=52.84 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.17it/s]

    Capturing num tokens (num_tokens=768 avail_mem=52.83 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.17it/s]Capturing num tokens (num_tokens=768 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.85it/s]Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.85it/s]

    Capturing num tokens (num_tokens=640 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.85it/s]Capturing num tokens (num_tokens=640 avail_mem=52.83 GB):  47%|████▋     | 27/58 [00:02<00:03, 10.14it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  47%|████▋     | 27/58 [00:02<00:03, 10.14it/s]

    Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  47%|████▋     | 27/58 [00:02<00:03, 10.14it/s]Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  50%|█████     | 29/58 [00:02<00:03,  9.02it/s]Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  50%|█████     | 29/58 [00:02<00:03,  9.02it/s]

    Capturing num tokens (num_tokens=448 avail_mem=52.82 GB):  50%|█████     | 29/58 [00:02<00:03,  9.02it/s]Capturing num tokens (num_tokens=448 avail_mem=52.82 GB):  53%|█████▎    | 31/58 [00:03<00:03,  8.59it/s]Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  53%|█████▎    | 31/58 [00:03<00:03,  8.59it/s]

    Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:03<00:03,  8.48it/s]Capturing num tokens (num_tokens=384 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:03<00:03,  8.48it/s]Capturing num tokens (num_tokens=384 avail_mem=52.82 GB):  57%|█████▋    | 33/58 [00:03<00:02,  8.44it/s]Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  57%|█████▋    | 33/58 [00:03<00:02,  8.44it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.62it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.62it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:03<00:02,  8.90it/s]Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:03<00:02,  8.90it/s]

    Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:03<00:02,  9.06it/s]Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:03<00:02,  9.06it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  62%|██████▏   | 36/58 [00:03<00:02,  9.06it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:02,  9.59it/s]Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:02,  9.59it/s]

    Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  66%|██████▌   | 38/58 [00:03<00:02,  9.59it/s]Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.95it/s]Capturing num tokens (num_tokens=192 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.95it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.95it/s]

    Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.82it/s]Capturing num tokens (num_tokens=160 avail_mem=52.78 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.82it/s]Capturing num tokens (num_tokens=144 avail_mem=52.78 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.82it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.82it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.57it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.57it/s]Capturing num tokens (num_tokens=96 avail_mem=52.77 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.57it/s] Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.57it/s]Capturing num tokens (num_tokens=64 avail_mem=52.76 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.57it/s]Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.57it/s]

    Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.09it/s]Capturing num tokens (num_tokens=32 avail_mem=71.11 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.09it/s]Capturing num tokens (num_tokens=28 avail_mem=71.11 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.09it/s]Capturing num tokens (num_tokens=24 avail_mem=71.10 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.09it/s]Capturing num tokens (num_tokens=24 avail_mem=71.10 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.22it/s]Capturing num tokens (num_tokens=20 avail_mem=71.10 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.22it/s]Capturing num tokens (num_tokens=16 avail_mem=71.10 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.22it/s]Capturing num tokens (num_tokens=12 avail_mem=71.09 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.22it/s]

    Capturing num tokens (num_tokens=8 avail_mem=71.09 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.22it/s] Capturing num tokens (num_tokens=8 avail_mem=71.09 GB):  98%|█████████▊| 57/58 [00:04<00:00, 24.32it/s]Capturing num tokens (num_tokens=4 avail_mem=70.54 GB):  98%|█████████▊| 57/58 [00:04<00:00, 24.32it/s]Capturing num tokens (num_tokens=4 avail_mem=70.54 GB): 100%|██████████| 58/58 [00:04<00:00, 12.25it/s]


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
    Generated text:  Evan and I am a first year student at the University of Colorado Boulder. My main areas of research are in the field of technology for autonomous vehicles, specifically with a focus on the design and development of autonomous driving systems. I am currently a PhD student at the University of Colorado Boulder under the supervision of Professor Derek Johnson. My research interests include the design of advanced sensors for autonomous vehicles, sensor fusion, and visual perception. I am particularly interested in leveraging machine learning and deep neural networks to improve the accuracy and reliability of sensor data.
    My research is currently supported by the NSF and the Department of Defense (DoD) and I am interested in
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. Every day, he has to answer hundreds of phone calls. He spends his whole day to answer the calls. Most of the calls are from people who come to him for a job. The president calls about 3, 500 people every day. Most of the people he talks to are very kind. They help him with important things. On weekends, the president can talk to people for 2 hours each. He says, "I don't have to wait for people to call me. I can talk to them at any time. I can talk to them for 2 hours each day. That way,
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Lyon C. Marseille D. Nice
    Answer:
    
    A
    
    In the following sentences, which one has the same type of sentence expression as the example sentence? Example sentence: This is not a thing to be ashamed of.
    A. A scholar who has traveled far and wide has broadened his horizons.
    B. I must not be held responsible for him.
    C. What matters is not what is seen, but what is unseen.
    D. Don't be surprised.
    Answer:
    
    C
    
    The scientist, _____, is named after his family name, is well-known for his outstanding contributions to physics.
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching, but, with a lot of confusion around the term and the whole area of AI itself, there is a lot of room for confusion and uncertainty. In this post, we’re going to delve into some of the key terms and concepts that are commonly used when discussing AI and see if you can spot the differences between them. Also, we’ll cover some of the key applications of AI, and the key hurdles that we’ll need to navigate as we move forward.
    When we think of AI, we’re often thinking of a sort of technology that can perform tasks that would usually be performed by a human, but it can also be


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I love [insert a short description of your favorite activity or hobby]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. What's your favorite color? I love [insert a short description of your favorite color]. What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is known for its vibrant nightlife, fashion, and food scene, and is a popular tourist destination for its beautiful architecture and historical landmarks. The city is also home to many famous French artists and writers, including the painter Monet and the novelist Dumas. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its status as the capital of France has made it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: AI is likely to become more prevalent in many industries, including manufacturing, transportation, and healthcare. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI becomes more prevalent, there will likely be increased scrutiny of its use and potential misuse. There will be a need for ethical guidelines and regulations to ensure that AI is used in
    


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
    Generated text:  [Name] and I'm a [occupation] who has been learning about different cultures around the world. I enjoy [mention something interesting or interesting to know] about other cultures, and I'm always eager to learn from others. What's your name? I'm [name] and I'm a [occupation]. I've always been interested in how people from different cultures live their lives, so I'm constantly learning about different cultures, languages, and traditions. I enjoy learning about other cultures because I want to understand how they see the world and how they live their lives. I'm always curious to know more about different cultures and how they
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Is the above statement true? Yes, it is true. Paris is the capital city of France and is known for its rich history, stunning architecture, and vibrant culture. It is often referred to as "the city of lights" and is a UNESCO World Heritage site. The city is also famous for its famous landmarks, including the Eiffel Tower and Notre-Dame Cathedral. Paris is a popular tourist destination and is home to many iconic museums and attractions. In addition, it is also known as the "City of Light" due to its status as the center of the European Union. Is there anything else you would like
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and will be influenced by a number of factors, including technological advancements, societal changes, and changing priorities. Some possible trends that are currently being explored or projected include:
    
    1. Increased efficiency: AI is becoming more efficient and effective in performing tasks, such as task automation, drug discovery, and financial forecasting. These trends are expected to continue as AI becomes more powerful and accessible to a wider range of users.
    
    2. AI in healthcare: AI is already being used in a variety of healthcare applications, including image analysis, drug discovery, and patient diagnosis. As AI technology continues to improve, it is likely that we will see even more


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

    ].

     I

     am

     a

     [

    background

    ]

     person

     with

     a

     love

     for

     [

    something

     specific

    ].

     I

     am

     [

    age

    ].

     I

     am

     [

    job

     title

    ].

     I

     love

     [

    what

     I

     do

     or

     am

     involved

     with

    ].

     I

     am

     [

    language

    ].

     I

     hope

     you

     find

     [

    job

     title

    ]

     rewarding

     and

     [

    what

     you

     love

    ].

     As

     a

     [

    job

     title

    ],

     I

     am

     [

    insert

     any

     notable

     achievements

     or

     experiences

    ].

     Thank

     you

    .

     Let

     me

     know

     if

     you

     have

     any

     questions

     about

     my

     self

    -int

    roduction

    .

     Hey

     there

    !

     Nice

     to

     meet

     you

    .

     My

     name

     is

     [

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    background

    ]

     person

     with

     a

     passion

     for

     [

    something

     specific

    ].

     I

     am

     [

    age

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     

    3

    9

    


    B

    .

     

    4

    0

    


    C

    .

     

    4

    1

    


    D

    .

     

    4

    2

    


    E

    .

     

    4

    3

    
    


    To

     determine

     the

     correct

     answer

    ,

     I

     will

     follow

     these

     steps

    :
    


    1

    .

     Identify

     the

     capital

     city

     of

     France

    .


    2

    .

     Confirm

     that

     the

     capital

     city

     is

     Paris

    .


    3

    .

     Present

     the

     answer

    .
    


    Step

     

    1

    :

     The

     capital

     city

     of

     France

     is

     Paris

    .
    


    Step

     

    2

    :

     Confirm

     that

     the

     capital

     city

     is

     Paris

    .


    The

     capital

     city

     of

     France

    ,

     Paris

    ,

     is

     the

     most

     populous

     city

     in

     France

    ,

     with

     over

     

    1

    0

     million

     inhabitants

    .
    


    Step

     

    3

    :

     Present

     the

     answer

    .


    The

     correct

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     marked

     by

     rapid

     advancements

     and

     significant

     changes

    .

     Here

     are

     some

     potential

     trends

     to

     look

     out

     for

    :
    


    1

    .

     Increased

     integration

     with

     human

     AI

    :

     As

     AI

     becomes

     more

     autonomous

    ,

     we

    're

     likely

     to

     see

     more

     integration

     between

     it

     and

     human

     AI

    .

     For

     example

    ,

     we

     could

     see

     more

     sophisticated

     human

    -like

     AI

     interacting

     with

     machines

    ,

     or

     AI

     systems

     learning

     from

     and

     interacting

     with

     humans

     in

     a

     more

     meaningful

     way

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     and

     responsible

     AI

    :

     With

     the

     potential

     for

     AI

     to

     cause

     harm

     or

     be

     used

     for

     nef

    arious

     purposes

    ,

     we

    're

     likely

     to

     see

     an

     increased

     focus

     on

     ethical

     and

     responsible

     AI

     development

    .

     This

     could

     include

     developing

     AI

     that

     is

     transparent

    ,

    



```python
llm.shutdown()
```
