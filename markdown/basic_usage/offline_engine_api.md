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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.48it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.48it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.48it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:13,  3.68it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:13,  3.68it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:13,  3.68it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:13,  3.68it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 12.47it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 12.47it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 12.47it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 12.47it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 12.47it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.16it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.16it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.16it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.16it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.16it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.18it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.18it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.18it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.18it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.18it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.18it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:00, 28.61it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:00, 28.61it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 33.12it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 36.57it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 36.57it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 36.57it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 36.57it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 36.57it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 36.57it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 39.66it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.22 GB):   3%|▎         | 2/58 [00:00<00:07,  7.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.60 GB):   3%|▎         | 2/58 [00:00<00:07,  7.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.60 GB):   5%|▌         | 3/58 [00:00<00:08,  6.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.13 GB):   5%|▌         | 3/58 [00:00<00:08,  6.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.13 GB):   7%|▋         | 4/58 [00:00<00:08,  6.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.69 GB):   7%|▋         | 4/58 [00:00<00:08,  6.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.69 GB):   9%|▊         | 5/58 [00:00<00:08,  6.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.69 GB):   9%|▊         | 5/58 [00:00<00:08,  6.30it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.69 GB):  10%|█         | 6/58 [00:00<00:08,  6.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.29 GB):  10%|█         | 6/58 [00:00<00:08,  6.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.29 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.68 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.39it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.68 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.67 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.67 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.32 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.13it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.32 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.73 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.73 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.73 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.15it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.73 GB):  21%|██        | 12/58 [00:01<00:06,  7.22it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.65 GB):  21%|██        | 12/58 [00:01<00:06,  7.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.79 GB):  21%|██        | 12/58 [00:01<00:06,  7.22it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.79 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.78 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.78 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.64 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.62 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.10it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=52.62 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.83 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.83 GB):  31%|███       | 18/58 [00:02<00:04,  8.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.82 GB):  31%|███       | 18/58 [00:02<00:04,  8.58it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.61 GB):  31%|███       | 18/58 [00:02<00:04,  8.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.61 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.85 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.61it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=52.85 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.53it/s]Capturing num tokens (num_tokens=960 avail_mem=53.60 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.53it/s] Capturing num tokens (num_tokens=896 avail_mem=53.45 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.53it/s]Capturing num tokens (num_tokens=896 avail_mem=53.45 GB):  40%|███▉      | 23/58 [00:02<00:03,  9.78it/s]Capturing num tokens (num_tokens=832 avail_mem=52.91 GB):  40%|███▉      | 23/58 [00:02<00:03,  9.78it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.91 GB):  41%|████▏     | 24/58 [00:03<00:03,  9.72it/s]Capturing num tokens (num_tokens=768 avail_mem=52.94 GB):  41%|████▏     | 24/58 [00:03<00:03,  9.72it/s]Capturing num tokens (num_tokens=704 avail_mem=53.46 GB):  41%|████▏     | 24/58 [00:03<00:03,  9.72it/s]Capturing num tokens (num_tokens=704 avail_mem=53.46 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.13it/s]Capturing num tokens (num_tokens=640 avail_mem=52.88 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.13it/s]

    Capturing num tokens (num_tokens=576 avail_mem=52.96 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.13it/s]Capturing num tokens (num_tokens=576 avail_mem=52.96 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.04it/s]Capturing num tokens (num_tokens=512 avail_mem=53.01 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.04it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.53 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.04it/s]Capturing num tokens (num_tokens=480 avail_mem=53.53 GB):  52%|█████▏    | 30/58 [00:03<00:03,  9.05it/s]Capturing num tokens (num_tokens=448 avail_mem=53.02 GB):  52%|█████▏    | 30/58 [00:03<00:03,  9.05it/s]Capturing num tokens (num_tokens=416 avail_mem=53.02 GB):  52%|█████▏    | 30/58 [00:03<00:03,  9.05it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.02 GB):  55%|█████▌    | 32/58 [00:03<00:02, 10.31it/s]Capturing num tokens (num_tokens=384 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:03<00:02, 10.31it/s]Capturing num tokens (num_tokens=352 avail_mem=53.05 GB):  55%|█████▌    | 32/58 [00:03<00:02, 10.31it/s]Capturing num tokens (num_tokens=352 avail_mem=53.05 GB):  59%|█████▊    | 34/58 [00:03<00:02, 10.89it/s]Capturing num tokens (num_tokens=320 avail_mem=53.52 GB):  59%|█████▊    | 34/58 [00:03<00:02, 10.89it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.08 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.89it/s]Capturing num tokens (num_tokens=288 avail_mem=53.08 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.70it/s]Capturing num tokens (num_tokens=256 avail_mem=53.51 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.70it/s]Capturing num tokens (num_tokens=240 avail_mem=53.10 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.70it/s]

    Capturing num tokens (num_tokens=240 avail_mem=53.10 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.21it/s]Capturing num tokens (num_tokens=224 avail_mem=53.49 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.21it/s]Capturing num tokens (num_tokens=208 avail_mem=53.12 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.21it/s]Capturing num tokens (num_tokens=208 avail_mem=53.12 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.71it/s]Capturing num tokens (num_tokens=192 avail_mem=53.47 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.71it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.47 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.71it/s]Capturing num tokens (num_tokens=176 avail_mem=53.47 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.36it/s]Capturing num tokens (num_tokens=160 avail_mem=53.46 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.36it/s]Capturing num tokens (num_tokens=144 avail_mem=53.45 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.36it/s]Capturing num tokens (num_tokens=144 avail_mem=53.45 GB):  76%|███████▌  | 44/58 [00:04<00:01, 13.53it/s]Capturing num tokens (num_tokens=128 avail_mem=53.19 GB):  76%|███████▌  | 44/58 [00:04<00:01, 13.53it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.44 GB):  76%|███████▌  | 44/58 [00:04<00:01, 13.53it/s]Capturing num tokens (num_tokens=112 avail_mem=53.44 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.99it/s]Capturing num tokens (num_tokens=96 avail_mem=53.43 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.99it/s] Capturing num tokens (num_tokens=80 avail_mem=53.22 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.99it/s]Capturing num tokens (num_tokens=80 avail_mem=53.22 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.36it/s]Capturing num tokens (num_tokens=64 avail_mem=53.42 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.36it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.40 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.36it/s]Capturing num tokens (num_tokens=48 avail_mem=53.40 GB):  86%|████████▌ | 50/58 [00:05<00:00, 16.00it/s]Capturing num tokens (num_tokens=32 avail_mem=53.39 GB):  86%|████████▌ | 50/58 [00:05<00:00, 16.00it/s]Capturing num tokens (num_tokens=28 avail_mem=53.25 GB):  86%|████████▌ | 50/58 [00:05<00:00, 16.00it/s]Capturing num tokens (num_tokens=24 avail_mem=53.25 GB):  86%|████████▌ | 50/58 [00:05<00:00, 16.00it/s]Capturing num tokens (num_tokens=24 avail_mem=53.25 GB):  91%|█████████▏| 53/58 [00:05<00:00, 18.20it/s]Capturing num tokens (num_tokens=20 avail_mem=53.34 GB):  91%|█████████▏| 53/58 [00:05<00:00, 18.20it/s]

    Capturing num tokens (num_tokens=16 avail_mem=53.35 GB):  91%|█████████▏| 53/58 [00:05<00:00, 18.20it/s]Capturing num tokens (num_tokens=12 avail_mem=53.34 GB):  91%|█████████▏| 53/58 [00:05<00:00, 18.20it/s]Capturing num tokens (num_tokens=12 avail_mem=53.34 GB):  97%|█████████▋| 56/58 [00:05<00:00, 19.84it/s]Capturing num tokens (num_tokens=8 avail_mem=53.33 GB):  97%|█████████▋| 56/58 [00:05<00:00, 19.84it/s] Capturing num tokens (num_tokens=4 avail_mem=53.33 GB):  97%|█████████▋| 56/58 [00:05<00:00, 19.84it/s]Capturing num tokens (num_tokens=4 avail_mem=53.33 GB): 100%|██████████| 58/58 [00:05<00:00, 10.87it/s]


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
    Generated text:  jyoti. I'm the only person here today. I'm a special kind of person. I'm a kid. I'm a student. I'm a girl. I'm 15 years old. I like to go to the beach. I like to play with my friends. I like to ride my skateboard. I like to read books. I'm very active. I like to eat tacos. I love all kinds of things. It's important to me that I be liked and respected. I'm a good person. I like to play sports. I have a great social life and I like to try new things.
    ===============================
    Prompt: The president of the United States is
    Generated text:  32 years older than the president of Brazil. The president of Brazil is 30 years younger than the president of India. If the president of India is 100 years old now, how old will the president of India be in 10 years?
    To determine the current age of the president of India, we start with the information given:
    
    1. The president of India is currently 100 years old.
    2. The president of Brazil is 30 years younger than the president of India. Therefore, the president of Brazil is \(100 - 30 = 70\) years old
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the most populous city in Europe and the second largest city in the European Union after Rome. The inhabitants are French and they speak French, the official language of France. Paris was founded in the 12th century and is located on the western bank of the Seine River. The city is famous for its architecture, museums, and opera. The city is considered to be a cultural and historical center of the world. The city is located in the western part of France. It is surrounded by the Alps and the Mediterranean Sea. The city of Paris has a population of 2.1 million in 201
    ===============================
    Prompt: The future of AI is
    Generated text:  already here. With the advent of new technologies and continuous technological innovations, the AI revolution is just getting started. The current generation of AI is more advanced, faster, and more accurate than ever before. This means that AI is becoming an increasingly important and valuable tool for businesses and organizations to enhance their operations and make informed decisions. It can be used for everything from marketing and sales, to customer service and product development. The future of AI is exciting and there are many possibilities for what it could do.
    The first step in building a successful AI project is to understand the problem you are trying to solve. This involves identifying the specific requirements and constraints


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Character] who has always been [Positive Traits]. I am [Type of Character] and I am [Type of Character]. I am [Type of Character] and I am [Type of Character]. I am [Type of Character] and I am [Type of Character]. I am [Type of Character] and I am [Type of Character]. I am [Type of Character] and I am [Type of Character]. I am [Type of Character] and I am [Type of Character]. I am [Type of Character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to many famous museums and attractions, including the Louvre, the Musée d'Orsay, and the Musée d'Orsay. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased Integration of AI into Everyday Life: AI is already being integrated into many aspects of our lives, from self-driving cars to virtual assistants like Siri and Alexa. As the technology continues to advance, we can expect to see even more integration of AI into our daily lives, such as in healthcare, finance, and transportation.
    
    2. AI will become more Personalized: As AI technology continues to improve, we can expect to see
    


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
    Generated text:  [Your Name]. I am a [Your Profession/Position] at [Your Company/Institution]. I am passionate about [Your Passion/Interest]. I have [Number of years] years of experience in [Your Field], and I have always been driven by [Your Motivation]. I am known for my [Strengths/Attitudes/Problems]. I am always ready to learn and grow. What kind of projects or experiences do you have that can help me understand my character better? Let's begin! I'll do my best to provide you with a well-rounded view of who I am and what I bring to the table
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its medieval architecture, iconic landmarks such as the Eiffel Tower, and lively cultural scene. Paris is also a popular tourist destination with its iconic landmarks, museums, and restaurants. The city is known for its French cuisine and its role in the world of fashion, with many renowned fashion houses and boutiques. Paris has a rich history and is a major cultural hub in Europe. Its unique blend of history, art, and culture has made it a global tourist destination and a cultural treasure. Can you identify any specific attractions or cultural landmarks in Paris that are particularly famous and iconic? Based on the given context, here are
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and in many ways, it seems like there is no limit to what can be achieved through the technology. Here are some possible trends that could shape the future of AI:
    
    1. Improved Explainability: AI systems are becoming more and more complex, and we are seeing more and more that AI can be difficult to understand and explain. With the development of techniques such as deep learning and neural networks, we are seeing progress in making AI more transparent and understandable.
    
    2. Increased Personalization: AI is becoming more and more capable of adapting to the needs of individuals, as well as adapting to the needs of the whole population. This could lead


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

    'm

     a

     [

    Age

    ]

     year

     old

    ,

     [

    Gender

    ]

     person

     who

     maj

    ored

     in

     [

    Major

    ].

     I

     work

     at

     [

    Company

     Name

    ]

     and

     have

     been

     a

     [

    Position

    ]

     for

     [

    Number

     of

     Years

    ].

     I

     enjoy

     [

    Job

     Description

    ].

     I

     also

     have

     a

     passion

     for

     [

    Current

     Hobby

     or

     Skill

    ].

     What

    's

     your

     main

     goal

    ,

     and

     what

     do

     you

     hope

     to

     achieve

     in

     the

     near

     future

    ?
    


    [

    Name

    ]:

     I

     am

     a

     self

    -prof

    essed

     problem

     solver

    ,

     passionate

     about

     creativity

    ,

     and

     driven

     by

     curiosity

    .

     I

     thrive

     on

     learning

     new

     skills

     and

     challenging

     myself

     in

     various

     projects

    .

     My

     goal

     is

     to

     become

     an

     expert

     in

     my

     field

     and

     inspire

     others

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     about

     France

    's

     capital

     city

     is

     accurate

    .

     Paris

     is

     the

     capital

     of

     France

    ,

     serving

     as

     its

     largest

     city

     and

     a

     significant

     cultural

     and

     economic

     center

    .

     It

     is

     the

     birth

    place

     of

     many

     famous

     French

     figures

    ,

     including

     Napoleon

     Bon

    ap

    arte

     and

     Vol

    taire

    ,

     and

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

     rich

     history

    .

     Paris

     is

     also

     a

     major

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    ,

     making

     it

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     galleries

    ,

     and

     cultural

     institutions

    ,

     as

     well

     as

     a

     diverse

     array

     of

     restaurants

    ,

     shops

    ,

     and

     entertainment

     venues

    .

     Paris

     is

     a

     vibrant

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     exciting

    ,

     with

     endless

     possibilities

     and

     challenges

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     Integration

     of

     AI

     into

     Everyday

     Life

    :

     AI

     is

     already

     being

     integrated

     into

     a

     wide

     range

     of

     products

     and

     services

    ,

     from

     smart

     home

     devices

     to

     self

    -driving

     cars

    .

     We

     can

     expect

     this

     trend

     to

     continue

     in

     the

     future

    ,

     with

     more

     and

     more

     AI

     technologies

     being

     integrated

     into

     everyday

     life

    .
    


    2

    .

     Better

     and

     More

     Personal

    ized

     AI

    :

     AI

     is

     becoming

     more

     powerful

     and

     able

     to

     learn

     from

     data

    ,

     which

     means

     that

     it

     can

     provide

     more

     personalized

     and

     relevant

     advice

     and

     recommendations

    .

     This

     will

     lead

     to

     better

     customer

     experiences

     and

     a

     more

     efficient

     use

     of

     resources

    .
    


    3

    .

     Greater

    



```python
llm.shutdown()
```
