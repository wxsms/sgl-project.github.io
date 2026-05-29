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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:39,  3.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:39,  3.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:39,  3.84s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:39,  3.84s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:40,  1.33it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:05,  7.53it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 19.27it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 23.11it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 27.79it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.56it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.56it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.56it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 31.56it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 31.56it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 31.56it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 34.10it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.68 GB):   2%|▏         | 1/58 [00:00<00:07,  7.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.50 GB):   2%|▏         | 1/58 [00:00<00:07,  7.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.50 GB):   3%|▎         | 2/58 [00:00<00:07,  7.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.71 GB):   3%|▎         | 2/58 [00:00<00:07,  7.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.71 GB):   5%|▌         | 3/58 [00:00<00:07,  7.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.71 GB):   5%|▌         | 3/58 [00:00<00:07,  7.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.71 GB):   7%|▋         | 4/58 [00:00<00:06,  8.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.49 GB):   7%|▋         | 4/58 [00:00<00:06,  8.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.49 GB):   9%|▊         | 5/58 [00:00<00:05,  8.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.76 GB):   9%|▊         | 5/58 [00:00<00:05,  8.87it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.76 GB):  10%|█         | 6/58 [00:00<00:05,  9.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.75 GB):  10%|█         | 6/58 [00:00<00:05,  9.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.44 GB):  10%|█         | 6/58 [00:00<00:05,  9.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.44 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.82 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.25it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.47 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.47 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:01<00:04, 11.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:03, 11.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.46 GB):  21%|██        | 12/58 [00:01<00:03, 11.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.94 GB):  21%|██        | 12/58 [00:01<00:03, 11.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.94 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.46 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.20it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=52.96 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.96 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.95 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.44 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.73it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=53.44 GB):  31%|███       | 18/58 [00:01<00:02, 13.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.98 GB):  31%|███       | 18/58 [00:01<00:02, 13.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.43 GB):  31%|███       | 18/58 [00:01<00:02, 13.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.43 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.98 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.66it/s]Capturing num tokens (num_tokens=960 avail_mem=53.43 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.66it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=53.43 GB):  38%|███▊      | 22/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=896 avail_mem=53.02 GB):  38%|███▊      | 22/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=832 avail_mem=53.42 GB):  38%|███▊      | 22/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=832 avail_mem=53.42 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.29it/s]Capturing num tokens (num_tokens=768 avail_mem=53.05 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.29it/s]Capturing num tokens (num_tokens=704 avail_mem=53.41 GB):  41%|████▏     | 24/58 [00:02<00:02, 16.29it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.41 GB):  45%|████▍     | 26/58 [00:02<00:01, 16.92it/s]Capturing num tokens (num_tokens=640 avail_mem=53.07 GB):  45%|████▍     | 26/58 [00:02<00:01, 16.92it/s]Capturing num tokens (num_tokens=576 avail_mem=53.41 GB):  45%|████▍     | 26/58 [00:02<00:01, 16.92it/s]Capturing num tokens (num_tokens=576 avail_mem=53.41 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.07it/s]Capturing num tokens (num_tokens=512 avail_mem=53.09 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.07it/s]Capturing num tokens (num_tokens=480 avail_mem=53.40 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.07it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.40 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.13it/s]Capturing num tokens (num_tokens=448 avail_mem=53.14 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.13it/s]Capturing num tokens (num_tokens=416 avail_mem=53.37 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.13it/s]Capturing num tokens (num_tokens=384 avail_mem=53.39 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.13it/s]Capturing num tokens (num_tokens=384 avail_mem=53.39 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.67it/s]Capturing num tokens (num_tokens=352 avail_mem=53.18 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.67it/s]Capturing num tokens (num_tokens=320 avail_mem=53.20 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.67it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.36 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.67it/s]Capturing num tokens (num_tokens=288 avail_mem=53.36 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.28it/s]Capturing num tokens (num_tokens=256 avail_mem=53.36 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.28it/s]Capturing num tokens (num_tokens=240 avail_mem=53.35 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.28it/s]Capturing num tokens (num_tokens=224 avail_mem=53.23 GB):  62%|██████▏   | 36/58 [00:02<00:01, 20.28it/s]Capturing num tokens (num_tokens=224 avail_mem=53.23 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.01it/s]Capturing num tokens (num_tokens=208 avail_mem=53.23 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.01it/s]Capturing num tokens (num_tokens=192 avail_mem=53.33 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.01it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.33 GB):  67%|██████▋   | 39/58 [00:02<00:00, 22.01it/s]Capturing num tokens (num_tokens=176 avail_mem=53.33 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=160 avail_mem=53.32 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=144 avail_mem=53.32 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=128 avail_mem=53.31 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=128 avail_mem=53.31 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.54it/s]Capturing num tokens (num_tokens=112 avail_mem=53.30 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.54it/s]Capturing num tokens (num_tokens=96 avail_mem=53.24 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.54it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=53.24 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.54it/s]Capturing num tokens (num_tokens=64 avail_mem=53.28 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.54it/s]Capturing num tokens (num_tokens=64 avail_mem=53.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.68it/s]Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.68it/s]Capturing num tokens (num_tokens=32 avail_mem=53.27 GB):  84%|████████▍ | 49/58 [00:03<00:00, 26.68it/s]Capturing num tokens (num_tokens=28 avail_mem=53.26 GB):  84%|████████▍ | 49/58 [00:03<00:00, 26.68it/s]Capturing num tokens (num_tokens=24 avail_mem=53.26 GB):  84%|████████▍ | 49/58 [00:03<00:00, 26.68it/s]Capturing num tokens (num_tokens=24 avail_mem=53.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 28.26it/s]Capturing num tokens (num_tokens=20 avail_mem=53.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 28.26it/s]

    Capturing num tokens (num_tokens=16 avail_mem=53.21 GB):  91%|█████████▏| 53/58 [00:03<00:00, 28.26it/s]Capturing num tokens (num_tokens=12 avail_mem=53.24 GB):  91%|█████████▏| 53/58 [00:03<00:00, 28.26it/s]Capturing num tokens (num_tokens=8 avail_mem=53.24 GB):  91%|█████████▏| 53/58 [00:03<00:00, 28.26it/s] Capturing num tokens (num_tokens=8 avail_mem=53.24 GB):  98%|█████████▊| 57/58 [00:03<00:00, 30.05it/s]Capturing num tokens (num_tokens=4 avail_mem=53.23 GB):  98%|█████████▊| 57/58 [00:03<00:00, 30.05it/s]Capturing num tokens (num_tokens=4 avail_mem=53.23 GB): 100%|██████████| 58/58 [00:03<00:00, 17.75it/s]


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
    Generated text:  Matt and I am a chef and amateur researcher. I started out making homemade pizza and trying to teach people about cooking and the ingredients they can use to make delicious meals. Now I have learned that I am not a chef. My interests are in research and studying cooking. I currently work at a restaurant in Austin, Texas.
    
    As a chef and amateur researcher, what do you enjoy the most about your job?
    
    As a chef and amateur researcher, the most enjoyable aspects of my job are the combination of passion for cooking and the joy of learning. Chef Matt enjoys the challenge of inventing new recipes, experimenting with new ingredients, and trying to
    ===============================
    Prompt: The president of the United States is
    Generated text:  expected to have the most power in the world. They are the boss of all the country, and they are expected to work in their own way and make the decisions in the country. They are the only ones that can make the decisions in the country. However, they have to be responsible for the country. They are the ones that are responsible for the safety, the health, the environment and the people’s lives in the country. They have to be in control. If they have a problem, they are the ones that can solve it.
    
    The president is the one who appoints the highest-ranking executive to represent the country. There are
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Paris is in the north of France and is the capital of France.
    Paris is the largest city in the world.
    The population of Paris is 2.3 million and it is the largest city in Europe.
    Paris was founded in 789 by Charlemagne and it was renamed Paris in 1793.
    It was named after the name of a king: Charles VI.
    It was named after a king: Charles VI.
    It was named after a king: Charles VI.
    The inhabitants of Paris speak French.
    The inhabitants of Paris are an ethnic group: the French.
    The inhabitants of Paris speak French.
    The
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is also a place where we are now, and that is where we have to pause to ask how the technology can be made more human-like.
    Take a look at what will happen in the near future.
    New research is being done on the development of artificial intelligence (AI) and how it can assist humans in areas like life sciences and healthcare. Researchers are currently creating AI that can accurately predict the spread of diseases, detect mutations in the human genome, and predict the effects of drugs. The goal is to use this technology to develop more effective treatments for illnesses.
    AI has also been used in areas like robotics and drones.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is also a major transportation hub, with many international airports and train stations. The city is known for its fashion industry, with many famous designers and boutiques. Paris is a popular tourist destination, with millions of visitors each year. It is also home to many museums, including the Louvre, the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in manufacturing, transportation, and other industries, where it can perform tasks that are currently done by humans. This could lead to job displacement, but also create new opportunities for workers.
    
    2. AI ethics and privacy: As AI becomes more advanced, there will be a need to address ethical and privacy concerns. This could lead to new regulations and standards for AI development and use.
    
    3. AI for healthcare: AI is already
    


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
    Generated text:  [insert name], and I'm a [insert profession or role] with a passion for [insert hobby or activity]. I enjoy [insert reason for hobby or activity], and I'm always looking for ways to [insert action or improve my skills]. Whether it's working on a project, learning something new, or just enjoying the moment, I'm always striving to grow and improve. I'm excited to meet you and learn more about you, and I look forward to building a connection with you.
    
    I hope this intro is brief and neutral, and that you're able to get to know me better through our conversation. Let's make some
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and the largest metropolitan area in the European Union. It is also the most populous city in the world, with a population of approximately 20 million people. The city is known for its rich cultural heritage, including its museums, art galleries, and theaters. Paris is home to numerous famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city also plays a significant role in the French economy and is a major center of politics, education, and business. Paris is a vibrant and dynamic city that attracts millions of visitors annually. It is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve continued growth and diversification, with a focus on enhancing AI capabilities in areas such as speech recognition, natural language processing, and machine learning. AI is also likely to become more autonomous, with machines being able to make decisions and take action without human intervention. This could lead to a more ethical and responsible use of AI, with machines being designed with a focus on minimizing unintended consequences and ensuring that their actions are aligned with societal goals. AI will also become more integrated with other technologies, such as 5G and blockchain, creating new possibilities for advanced technologies such as self-driving cars and decentralized financial systems. Finally, AI will likely


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

    ]

     and

     I

    'm

     [

    Age

    ]

     years

     old

    .

     I

     have

     [

    X

    ]

     years

     of

     experience

     in

     the

     [

    industry

    ]

     sector

    .

     Currently

    ,

     I

     work

     as

     a

     [

    Position

    ]

     in

     [

    Company

    ].

     I

     am

     [

    Qual

    ification

    ]

     and

     I

     have

     [

    Number

     of

     Projects

    ]

     years

     of

     experience

     in

     this

     field

    .

     I

     am

     [

    Communication

     Style

    ]

     and

     I

     am

     always

     looking

     to

     learn

     and

     grow

    .

     I

    'm

     a

     [

    Professional

     Ten

    acity

    ]

     and

     I

     am

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

     [

    Person

    ality

     Trait

    ]

     and

     I

     am

     always

     willing

     to

     help

     others

     and

     take

     on

     new

     challenges

    .

     I

     am

     [

    Aff

    iliation

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    F

    acts

     about

     France

    's

     capital

     city

    :
    


    1

    .

     Paris

     is

     the

     largest

     city

     in

     France

     by

     area

     and

     population

    ,

     with

     a

     population

     of

     about

     

    2

    .

    2

     million

     people

    .


    2

    .

     The

     city

     is

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     on

     the

     Î

    le

     de

     Paris

    .


    3

    .

     Paris

     is

     the

     capital

     of

     the

     French

     Republic

     and

     the

     largest

     city

     in

     Europe

     by

     area

    .


    4

    .

     The

     city

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

     other

     notable

     landmarks

    .


    5

    .

     Paris

     is

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     the

     presence

     of

     the

     Lou

    vre

     Museum

    ,

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     some

     potential

     trends

     that

     may

     occur

     are

    :
    


    1

    .

     Adv

    ancements

     in

     AI

     ethics

     and

     privacy

    :

     As

     AI

     becomes

     more

     integrated

     into

     daily

     life

    ,

     there

     will

     be

     increasing

     demands

     for

     ethical

     considerations

     and

     privacy

     protections

    .

     We

     may

     see

     more

     stringent

     regulations

     and

     stricter

     standards

     to

     ensure

     that

     AI

     systems

     are

     designed

     and

     deployed

     in

     ways

     that

     respect

     human

     values

     and

     protect

     individual

     rights

    .
    


    2

    .

     Increased

     integration

     with

     human

     workers

    :

     AI

     has

     the

     potential

     to

     automate

     many

     tasks

     that

     humans

     currently

     perform

    ,

     making

     it

     necessary

     to

     find

     ways

     to

     integrate

     AI

     into

     existing

     work

    forces

    .

     This

     could

     involve

     creating

     more

     jobs

     in

     new

     areas

    ,

     such

     as

     AI

     research

     and

     development

    ,

     as

     well

     as

     re

    



```python
llm.shutdown()
```
