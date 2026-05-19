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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.56it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.56it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.70it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.70it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.70it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.70it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  8.23it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 12.25it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 12.25it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 12.25it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 12.25it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 12.25it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 16.38it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 16.38it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 16.38it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 16.38it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 16.38it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 16.38it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:01, 21.55it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:01, 21.55it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 21.55it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:01, 21.55it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:01, 21.55it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:01, 21.55it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:00, 26.37it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:00, 26.37it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:00, 26.37it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:00, 26.37it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:00, 26.37it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:00, 26.37it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 35.76it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 35.76it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 35.76it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 35.76it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 35.76it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 35.76it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 38.59it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 41.08it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 41.08it/s]

    Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 41.08it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 41.08it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 41.08it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 41.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.27 GB):   2%|▏         | 1/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.24 GB):   2%|▏         | 1/58 [00:00<00:07,  7.31it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=54.24 GB):   3%|▎         | 2/58 [00:00<00:07,  7.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.24 GB):   3%|▎         | 2/58 [00:00<00:07,  7.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.24 GB):   5%|▌         | 3/58 [00:00<00:07,  7.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.24 GB):   5%|▌         | 3/58 [00:00<00:07,  7.46it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.24 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.24 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.23 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.23 GB):  10%|█         | 6/58 [00:00<00:05,  9.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.22 GB):  10%|█         | 6/58 [00:00<00:05,  9.04it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=54.22 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.22 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.22 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.22 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.23it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.21 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.21 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.21 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.21 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.57it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=54.21 GB):  21%|██        | 12/58 [00:01<00:04,  9.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.20 GB):  21%|██        | 12/58 [00:01<00:04,  9.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.20 GB):  21%|██        | 12/58 [00:01<00:04,  9.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.20 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.20 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.34it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=54.19 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.19 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.19 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.19 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.69it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=54.19 GB):  31%|███       | 18/58 [00:01<00:03, 11.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.18 GB):  31%|███       | 18/58 [00:01<00:03, 11.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.18 GB):  31%|███       | 18/58 [00:01<00:03, 11.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.18 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.16 GB):  34%|███▍      | 20/58 [00:01<00:03, 12.18it/s]Capturing num tokens (num_tokens=960 avail_mem=54.18 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.18it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=54.18 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.66it/s]Capturing num tokens (num_tokens=896 avail_mem=54.17 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.66it/s]Capturing num tokens (num_tokens=832 avail_mem=54.17 GB):  38%|███▊      | 22/58 [00:02<00:02, 13.66it/s]Capturing num tokens (num_tokens=832 avail_mem=54.17 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.42it/s]Capturing num tokens (num_tokens=768 avail_mem=54.17 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.42it/s]Capturing num tokens (num_tokens=704 avail_mem=54.16 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.42it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.16 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=640 avail_mem=54.16 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=576 avail_mem=54.16 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=576 avail_mem=54.16 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.87it/s]Capturing num tokens (num_tokens=512 avail_mem=54.15 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.87it/s]Capturing num tokens (num_tokens=480 avail_mem=54.16 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.87it/s]

    Capturing num tokens (num_tokens=480 avail_mem=54.16 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.69it/s]Capturing num tokens (num_tokens=448 avail_mem=54.16 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.69it/s]Capturing num tokens (num_tokens=416 avail_mem=54.16 GB):  52%|█████▏    | 30/58 [00:02<00:01, 16.69it/s]Capturing num tokens (num_tokens=416 avail_mem=54.16 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.37it/s]Capturing num tokens (num_tokens=384 avail_mem=53.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.37it/s]Capturing num tokens (num_tokens=352 avail_mem=53.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.37it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 17.64it/s]Capturing num tokens (num_tokens=320 avail_mem=53.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 17.64it/s]Capturing num tokens (num_tokens=288 avail_mem=53.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 17.64it/s]Capturing num tokens (num_tokens=288 avail_mem=53.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.06it/s]Capturing num tokens (num_tokens=256 avail_mem=53.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.06it/s]Capturing num tokens (num_tokens=240 avail_mem=53.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.06it/s]

    Capturing num tokens (num_tokens=224 avail_mem=53.31 GB):  62%|██████▏   | 36/58 [00:02<00:01, 18.06it/s]Capturing num tokens (num_tokens=224 avail_mem=53.31 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=208 avail_mem=53.31 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=192 avail_mem=53.31 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=176 avail_mem=53.30 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.99it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.24it/s]Capturing num tokens (num_tokens=160 avail_mem=53.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.24it/s]Capturing num tokens (num_tokens=144 avail_mem=53.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.24it/s]Capturing num tokens (num_tokens=144 avail_mem=53.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 19.06it/s]Capturing num tokens (num_tokens=128 avail_mem=53.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 19.06it/s]Capturing num tokens (num_tokens=112 avail_mem=53.29 GB):  76%|███████▌  | 44/58 [00:03<00:00, 19.06it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.98it/s]Capturing num tokens (num_tokens=96 avail_mem=53.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.98it/s] Capturing num tokens (num_tokens=80 avail_mem=53.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 18.98it/s]Capturing num tokens (num_tokens=80 avail_mem=53.29 GB):  83%|████████▎ | 48/58 [00:03<00:00, 18.98it/s]Capturing num tokens (num_tokens=64 avail_mem=53.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 18.98it/s]Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 18.98it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 18.83it/s]Capturing num tokens (num_tokens=32 avail_mem=53.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 18.83it/s]Capturing num tokens (num_tokens=28 avail_mem=53.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 18.83it/s]Capturing num tokens (num_tokens=28 avail_mem=53.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 19.02it/s]Capturing num tokens (num_tokens=24 avail_mem=53.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 19.02it/s]Capturing num tokens (num_tokens=20 avail_mem=53.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 19.02it/s]

    Capturing num tokens (num_tokens=20 avail_mem=53.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 19.09it/s]Capturing num tokens (num_tokens=16 avail_mem=53.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 19.09it/s]Capturing num tokens (num_tokens=12 avail_mem=53.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 19.09it/s]Capturing num tokens (num_tokens=12 avail_mem=53.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.31it/s]Capturing num tokens (num_tokens=8 avail_mem=53.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.31it/s] Capturing num tokens (num_tokens=4 avail_mem=53.25 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.31it/s]

    Capturing num tokens (num_tokens=4 avail_mem=53.25 GB): 100%|██████████| 58/58 [00:03<00:00, 19.33it/s]Capturing num tokens (num_tokens=4 avail_mem=53.25 GB): 100%|██████████| 58/58 [00:03<00:00, 14.53it/s]


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
    Generated text:  Nisek, and I'm a 10 year old boy. I was born on September 24, 2018 in a village in Peru. I have a brother named Kiry. We live in a small village. We have a small house and live on a diet of potatoes, corn, beans, squash, corn, and tomatoes. We eat lots of fresh produce every day, including lettuce, cabbage, and tomatoes. We don't like to eat meat or dairy products. How do we say "we live in a small village" in English?
    
    The question about the meaning and pronunciation of the sentence "
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the legislative branch of the government. Who is the leader of the executive branch?
    A) The President
    B) The Vice President
    C) The Governor
    D) The Prime Minister
    
    To determine which leader of the executive branch the president is, let's first understand the roles of the different branches of government in the United States. The legislative branch is responsible for making laws, and the executive branch is responsible for enforcing those laws. The president, acting as the head of the executive branch, serves as the chief executive and is responsible for implementing the laws passed by Congress.
    
    Given this information, we can evaluate the options:
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, located in the north of the country and situated along the River Seine. It is the largest city in France and the second largest in Europe, after Rome. Paris is a cosmopolitan city with a very diverse population. In 2012, the city was ranked as the 13th most expensive city in the world by The Economist, but the population in 2013 was 2,401,000.
    The city has a long and rich history. The first inhabitants of the area were the Celts. In 480 BC, the region was conquered
    ===============================
    Prompt: The future of AI is
    Generated text:  looking different than what we currently imagine, and with that comes a need to embrace change and reengineer how we design and develop. We must create a more inclusive AI system that includes diverse perspectives and addresses biases, including gender, race, and ethnicity, as well as other factors that can impact the design of AI. We can only achieve this by emphasizing the importance of diversity in AI development and perpetuating and supporting diversity in the research community. Building and implementing diverse data sets, developing policies that promote diversity and equity in the workplace, and ensuring that diverse voices are included in AI design and development can all lead to a more inclusive AI system


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name]. I'm currently [Current Location]. I'm [Current Activity]. I'm [Current Goal]. I'm [Current Personality]. I'm [Current Motivation]. I'm [Current Strength]. I'm [Current Weakness]. I'm [Current Interests]. I'm [Current Skills]. I'm [Current Education]. I'm [Current Hobby]. I'm [Current Religion]. I'm [Current Family]. I'm [Current Pets]. I'm [Current Friends]. I'm [Current Hobbies
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter. Paris is a cultural and historical center with a rich history dating back to ancient times. It is also a major financial center and a major tourist destination. The city is known for its cuisine, fashion, and art, and is a popular destination for tourists and locals alike. Paris is a vibrant and dynamic city with a rich history and culture. Its iconic landmarks and attractions make it a must-visit destination for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including personalized medicine
    


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
    Generated text:  [Your Name]. I'm a [Your Profession or Interest] with a passion for [Your Hobby, Interest or Hobby]. As someone who has always been fascinated by the world of technology and innovation, I'm constantly on the lookout for new and exciting ideas to bring to light. Whether it's developing new software or creating innovative products, I'm always up for the challenge of pushing boundaries and pushing the boundaries of what's possible. I'm looking forward to meeting you! [Your Name] [Your Contact Information] [Your LinkedIn Profile] [Your Twitter Profile] [Your Instagram Profile] [Your Facebook Profile] [Your GitHub Profile]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that serves as the political and cultural heart of the nation and is known for its magnificent palaces, stunning architecture, and a vibrant, diverse community of people and places. Its iconic landmarks, such as Notre Dame Cathedral and the Eiffel Tower, are also a major tourist attraction. French cuisine, particularly its famous croissants and fritters, is also famous worldwide. Paris is known for its art, literature, and music, and is a popular destination for tourists and locals alike. According to the latest figures, Paris has a population of over two million people. So, here’s a concise factual statement about
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be full of incredible possibilities and opportunities. Here are some possible future trends that AI is expected to undergo:
    
    1. AI will become more accessible and affordable: As AI technology continues to advance, more people will be able to use it for a variety of tasks and applications. This will make AI more accessible to a wider range of people, and it will become more affordable as technology improves and more people can afford it.
    
    2. AI will be integrated into more industries: As AI technology becomes more advanced, it is expected that it will be integrated into more industries. For example, AI could be used to automate production processes, improve customer


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

    insert

     your

     occupation

    ]

    !

     
    


    I

     have

     been

     working

     in

     the

     industry

     for

     [

    insert

     number

    ]

     years

     and

     have

     always

     been

     passionate

     about

     [

    insert

     your

     field

     of

     interest

    ].

     I

     am

     always

     looking

     for

     new

     challenges

     and

     learning

     new

     skills

     to

     keep

     myself

     up

     to

     date

     with

     the

     latest

     trends

     in

     [

    insert

     your

     field

     of

     interest

    ].

     
    


    I

     have

     a

     strong

     work

     ethic

     and

     always

     strive

     to

     meet

     or

     exceed

     my

     clients

    '

     expectations

    .

     I

    'm

     always

     available

     to

     answer

     any

     questions

     or

     provide

     assistance

     in

     case

     something

     goes

     wrong

     with

     a

     project

    .

     
    


    I

     am

     a

     team

     player

    ,

     and

     I

     enjoy

     collaborating

     with

     others

     to

     achieve

     a

     common

     goal

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

    ,

     a

     medieval

     city

    ,

     and

     the

     largest

     metropolitan

     area

     in

     the

     European

     Union

    .
    


    The

     Paris

     Metro

     is

     the

     world

    's

     longest

     underground

     metro

     system

    .

     It

     connects

     

    1

    3

    6

     stations

     throughout

     the

     city

     and

     serves

     

    1

    4

     million

     passengers

     annually

    .

     It

     is

     considered

     the

     world

    's

     most

     advanced

     metro

     system

    .

     Its

     

    1

    9

    8

    ,

     

    0

    0

    0

     passengers

     per

     year

     are

     the

     most

     of

     any

     metro

     system

     in

     the

     world

    .

     The

     Paris

     Mét

    ro

     has

     served

     as

     the

     largest

     metro

     system

     in

     the

     world

     for

     many

     years

    ,

     and

     it

     is

     still

     operating

     at

     a

     high

     level

    .
    


    The

     city

     is

     home

     to

     numerous

     architectural

     landmarks

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     depends

     on

     a

     variety

     of

     factors

    ,

     including

     ongoing

     technological

     advances

    ,

     regulatory

     changes

    ,

     and

     changes

     in

     societal

     attitudes

     towards

     AI

    .

     However

    ,

     some

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

     automation

     and

     robotics

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     may

     see

     an

     increase

     in

     the

     use

     of

     robots

     and

     automation

     in

     various

     industries

    ,

     from

     manufacturing

     to

     transportation

    .
    


    2

    .

     Improved

     understanding

     of

     AI

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     may

     see

     more

     accurate

     and

     precise

     algorithms

     that

     can

     better

     understand

     and

     interpret

     human

     behavior

    .
    


    3

    .

     AI

    -based

     healthcare

    :

     AI

     can

     help

     with

     the

     diagnosis

     and

     treatment

     of

     various

     diseases

    ,

     from

     diagn

    osing

     medical

     conditions

     to

     developing

     personalized

    



```python
llm.shutdown()
```
