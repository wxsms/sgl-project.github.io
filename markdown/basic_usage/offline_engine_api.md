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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:50,  1.97s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:50,  1.97s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:16,  3.07it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:16,  3.07it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:16,  3.07it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:10,  4.46it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:10,  4.46it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:10,  4.46it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:07,  6.09it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:07,  6.09it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:07,  6.09it/s]

    Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:07,  6.09it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.88it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.88it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.88it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.88it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.83it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.83it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.83it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.83it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.63it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.63it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.63it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.63it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:01, 19.30it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:01, 19.30it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:01, 19.30it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 19.30it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:01, 20.09it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:01, 20.09it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:01, 20.09it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 20.09it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:06<00:01, 22.20it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:06<00:01, 22.20it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:06<00:01, 22.20it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:06<00:01, 22.20it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:06<00:01, 22.20it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:00, 24.76it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:00, 24.76it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:00, 24.76it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:00, 24.76it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:06<00:00, 25.59it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:06<00:00, 25.59it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:00, 25.59it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:00, 25.59it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:06<00:00, 25.59it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 27.45it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 28.05it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 28.05it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 28.05it/s]

    Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 28.05it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 27.48it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 27.48it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 27.48it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 27.48it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 27.48it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 30.43it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 30.43it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 30.43it/s]

    Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:07<00:00, 30.43it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:07<00:00, 30.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00, 31.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  8.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.71 GB):   2%|▏         | 1/58 [00:00<00:11,  4.90it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.95 GB):   2%|▏         | 1/58 [00:00<00:11,  4.90it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.95 GB):   3%|▎         | 2/58 [00:00<00:11,  4.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.70 GB):   3%|▎         | 2/58 [00:00<00:11,  4.97it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.70 GB):   5%|▌         | 3/58 [00:00<00:09,  5.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.94 GB):   5%|▌         | 3/58 [00:00<00:09,  5.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.94 GB):   7%|▋         | 4/58 [00:00<00:09,  5.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.94 GB):   7%|▋         | 4/58 [00:00<00:09,  5.59it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=38.94 GB):   9%|▊         | 5/58 [00:00<00:09,  5.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.74 GB):   9%|▊         | 5/58 [00:00<00:09,  5.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.74 GB):  10%|█         | 6/58 [00:01<00:08,  6.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.92 GB):  10%|█         | 6/58 [00:01<00:08,  6.33it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.92 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.92 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.92 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.92 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.68it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.92 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.79 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.79 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.89 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.36it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.89 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.89 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.89 GB):  21%|██        | 12/58 [00:01<00:05,  7.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.77 GB):  21%|██        | 12/58 [00:01<00:05,  7.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=38.88 GB):  21%|██        | 12/58 [00:01<00:05,  7.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.88 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.87 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.35it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.87 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.86 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.86 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.86 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.85 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.12it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=38.85 GB):  31%|███       | 18/58 [00:02<00:04,  8.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.85 GB):  31%|███       | 18/58 [00:02<00:04,  8.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.85 GB):  33%|███▎      | 19/58 [00:02<00:04,  8.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.84 GB):  33%|███▎      | 19/58 [00:02<00:04,  8.67it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=38.84 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.82 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=960 avail_mem=38.82 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.84it/s] Capturing num tokens (num_tokens=960 avail_mem=38.82 GB):  38%|███▊      | 22/58 [00:02<00:03,  9.71it/s]Capturing num tokens (num_tokens=896 avail_mem=38.82 GB):  38%|███▊      | 22/58 [00:02<00:03,  9.71it/s]

    Capturing num tokens (num_tokens=832 avail_mem=38.82 GB):  38%|███▊      | 22/58 [00:02<00:03,  9.71it/s]Capturing num tokens (num_tokens=832 avail_mem=38.82 GB):  41%|████▏     | 24/58 [00:03<00:03, 10.25it/s]Capturing num tokens (num_tokens=768 avail_mem=38.81 GB):  41%|████▏     | 24/58 [00:03<00:03, 10.25it/s]

    Capturing num tokens (num_tokens=704 avail_mem=38.81 GB):  41%|████▏     | 24/58 [00:03<00:03, 10.25it/s]Capturing num tokens (num_tokens=704 avail_mem=38.81 GB):  45%|████▍     | 26/58 [00:03<00:03,  9.42it/s]Capturing num tokens (num_tokens=640 avail_mem=38.80 GB):  45%|████▍     | 26/58 [00:03<00:03,  9.42it/s]

    Capturing num tokens (num_tokens=640 avail_mem=38.80 GB):  47%|████▋     | 27/58 [00:03<00:03,  9.38it/s]Capturing num tokens (num_tokens=576 avail_mem=38.80 GB):  47%|████▋     | 27/58 [00:03<00:03,  9.38it/s]Capturing num tokens (num_tokens=576 avail_mem=38.80 GB):  48%|████▊     | 28/58 [00:03<00:03,  9.14it/s]Capturing num tokens (num_tokens=512 avail_mem=38.78 GB):  48%|████▊     | 28/58 [00:03<00:03,  9.14it/s]Capturing num tokens (num_tokens=480 avail_mem=38.75 GB):  48%|████▊     | 28/58 [00:03<00:03,  9.14it/s]

    Capturing num tokens (num_tokens=480 avail_mem=38.75 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.27it/s]Capturing num tokens (num_tokens=448 avail_mem=38.78 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.27it/s]Capturing num tokens (num_tokens=416 avail_mem=38.78 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.27it/s]Capturing num tokens (num_tokens=416 avail_mem=38.78 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.87it/s]Capturing num tokens (num_tokens=384 avail_mem=38.77 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.87it/s]

    Capturing num tokens (num_tokens=352 avail_mem=38.76 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.87it/s]Capturing num tokens (num_tokens=352 avail_mem=38.76 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.82it/s]Capturing num tokens (num_tokens=320 avail_mem=38.74 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.82it/s]Capturing num tokens (num_tokens=288 avail_mem=38.75 GB):  59%|█████▊    | 34/58 [00:04<00:02, 11.82it/s]

    Capturing num tokens (num_tokens=288 avail_mem=38.75 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.68it/s]Capturing num tokens (num_tokens=256 avail_mem=38.74 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.68it/s]Capturing num tokens (num_tokens=240 avail_mem=38.73 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.68it/s]Capturing num tokens (num_tokens=240 avail_mem=38.73 GB):  66%|██████▌   | 38/58 [00:04<00:01, 11.53it/s]Capturing num tokens (num_tokens=224 avail_mem=38.73 GB):  66%|██████▌   | 38/58 [00:04<00:01, 11.53it/s]

    Capturing num tokens (num_tokens=208 avail_mem=38.72 GB):  66%|██████▌   | 38/58 [00:04<00:01, 11.53it/s]Capturing num tokens (num_tokens=208 avail_mem=38.72 GB):  69%|██████▉   | 40/58 [00:04<00:01, 11.74it/s]Capturing num tokens (num_tokens=192 avail_mem=38.72 GB):  69%|██████▉   | 40/58 [00:04<00:01, 11.74it/s]Capturing num tokens (num_tokens=176 avail_mem=38.72 GB):  69%|██████▉   | 40/58 [00:04<00:01, 11.74it/s]

    Capturing num tokens (num_tokens=176 avail_mem=38.72 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.48it/s]Capturing num tokens (num_tokens=160 avail_mem=38.71 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.48it/s]Capturing num tokens (num_tokens=144 avail_mem=38.69 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.48it/s]Capturing num tokens (num_tokens=144 avail_mem=38.69 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.68it/s]Capturing num tokens (num_tokens=128 avail_mem=38.70 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.68it/s]

    Capturing num tokens (num_tokens=112 avail_mem=38.70 GB):  76%|███████▌  | 44/58 [00:04<00:01, 11.68it/s]Capturing num tokens (num_tokens=112 avail_mem=38.70 GB):  79%|███████▉  | 46/58 [00:04<00:01, 11.60it/s]Capturing num tokens (num_tokens=96 avail_mem=38.69 GB):  79%|███████▉  | 46/58 [00:04<00:01, 11.60it/s] Capturing num tokens (num_tokens=80 avail_mem=38.66 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.60it/s]

    Capturing num tokens (num_tokens=80 avail_mem=38.66 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.89it/s]Capturing num tokens (num_tokens=64 avail_mem=38.68 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.89it/s]Capturing num tokens (num_tokens=48 avail_mem=38.67 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.89it/s]Capturing num tokens (num_tokens=48 avail_mem=38.67 GB):  86%|████████▌ | 50/58 [00:05<00:00, 12.04it/s]Capturing num tokens (num_tokens=32 avail_mem=38.66 GB):  86%|████████▌ | 50/58 [00:05<00:00, 12.04it/s]

    Capturing num tokens (num_tokens=28 avail_mem=38.65 GB):  86%|████████▌ | 50/58 [00:05<00:00, 12.04it/s]Capturing num tokens (num_tokens=28 avail_mem=38.65 GB):  90%|████████▉ | 52/58 [00:05<00:00, 12.39it/s]Capturing num tokens (num_tokens=24 avail_mem=38.65 GB):  90%|████████▉ | 52/58 [00:05<00:00, 12.39it/s]Capturing num tokens (num_tokens=20 avail_mem=38.64 GB):  90%|████████▉ | 52/58 [00:05<00:00, 12.39it/s]

    Capturing num tokens (num_tokens=20 avail_mem=38.64 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.38it/s]Capturing num tokens (num_tokens=16 avail_mem=38.64 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.38it/s]Capturing num tokens (num_tokens=12 avail_mem=38.63 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.38it/s]Capturing num tokens (num_tokens=12 avail_mem=38.63 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.49it/s]Capturing num tokens (num_tokens=8 avail_mem=38.63 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.49it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=38.62 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.49it/s]Capturing num tokens (num_tokens=4 avail_mem=38.62 GB): 100%|██████████| 58/58 [00:05<00:00, 12.37it/s]Capturing num tokens (num_tokens=4 avail_mem=38.62 GB): 100%|██████████| 58/58 [00:05<00:00,  9.78it/s]


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
    Generated text:  James and I live in the United States of America. My favorite subject is English, and I am currently studying French at the moment. I also speak a little bit of Spanish and read books about Spanish. I like movies, but I don't like watching TV. I am a bit of a bookworm, and I really enjoy reading books. I have a few friends who like the same subjects as me, like French and Spanish. We all usually study together and do homework together.
    Based on that paragraph can we conclude that this scene is real?
    Pick your answer from: *yes *no *maybe
    no
    This scene is not
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. He needs 520 million votes to become the winner. He has already collected 520 million votes from chocolate voters. He needs to collect 804 million votes to reach the number needed for a second term. If he has already collected 360 million votes from other voters, how many more votes does he need to collect in order to become the winner? The president needs to collect a total of 520 million + 804 million = 1324 million votes
    He has already collected 520 million + 360 million =
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a very old and important city. In the 14th century, the first Parisians were miners. In the 15th century, they were farmers. In the 16th century, they were farmers again. In the 17th century, they were workers of a royal castle (castle means castle) on the banks of the Seine River. In the 18th century, they were workers of the royal court (court means court, so now it is a court in a court) on the banks of the Seine River. In the 19th century, they
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising. While we all know about the rise of new technologies like AI and robotics, there is another trend in the field of technology that is also very significant. This is the trend of machine learning and artificial intelligence (AI). Machine learning is a field that deals with the use of algorithms and data to perform tasks that would normally require human intelligence. It is based on the idea that a computer can learn from data, and make decisions based on the patterns it has learned from the data.
    
    The main idea behind machine learning is to train a model to learn from the data, and use it to make predictions or decisions. The data used to


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Positive Trait]. I'm passionate about [What I Love to Do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the largest metropolitan area in Europe. It is located on the Seine River and is the seat of the French government, the French parliament, and the headquarters of the French military. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. It is also a major economic center and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to evolve, we can expect to see even more innovative applications and improvements in AI. Additionally, there is a growing focus on ethical considerations and the potential impact of AI on society, which will likely lead to further developments in areas such as privacy and security. Overall, the future of AI is likely to be one of continued innovation, progress, and ethical considerations.
    


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
    Generated text:  [First Name] and I'm a [Last Name] who has been a [Job Title] for [Number of Years] years. I'm very passionate about [Favorite Activity/Interest/Condition]. I'm always looking for new opportunities to [What I Hope To Achieve/Change]. If you're looking for someone to support you in your journey, I'm here. I'd love to chat! [First Name] [Last Name] [Date] [Phone Number/Email] [Location] [Note: Feel free to add any relevant information to make the introduction more dynamic and engaging.]
    Hello, my name is [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and is located on the Seine River in the north-central region of the country. It was founded in the 6th century and has been the seat of government of France since 1792, when it was renamed the capital of France. Paris is home to many iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its diverse cuisine, fashion industry, and a rich cultural history that includes many famous artists and writers. Paris is a bustling metropolis with a large population and a vibrant cultural scene. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and constantly evolving. Here are some potential trends that could shape the field in the coming years:
    
    1. Increased focus on ethical considerations: With the increasing concerns around AI's potential to violate human rights and privacy, it's likely that ethical considerations will become a key focus for AI researchers. This could lead to greater regulation of AI systems and the development of new ethical standards.
    
    2. Improved accuracy and speed: As AI systems become more sophisticated, their ability to process data and solve problems will likely become even better. This could result in faster and more accurate AI systems that are better able to handle a wide range of tasks.
    
    3. Greater


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

     

    3

    0

    -year

    -old

     software

     engineer

     with

     a

     passion

     for

     [

    insert

     relevant

     tech

     field

    ].

     I

    'm

     an

     experienced

     developer

     who

     loves

     solving

     complex

     problems

     with

     code

     and

     constantly

     iterating

     on

     my

     skills

    .

     I

    'm

     a

     team

     player

    ,

     often

     leading

     projects

     and

     mentoring

     junior

     developers

    .

     I

    'm

     open

     to

     new

     challenges

     and

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

    .

     If

     you

     need

     anything

    ,

     feel

     free

     to

     reach

     out

    .

     What

     do

     you

     do

     for

     a

     living

    ?

     My

     career

     path

     has

     taken

     me

     from

     a

     software

     developer

     to

     a

     software

     engineer

     and

     now

     I

    'm

     in

     the

     position

     of

     helping

     other

     developers

     become

     great

    .

     What

     do

     you

     think

     is

     the

     most

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historic

     city

     with

     a

     rich

     cultural

     heritage

     and

     is

     known

     for

     its

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

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     food

     culture

    ,

     and

     annual

     festivals

     such

     as

     the

     World

     of

     D

    ancer

     and

     the

     G

    last

    on

    bury

     Music

     Festival

    .

     As

     the

     capital

    ,

     Paris

     plays

     a

     crucial

     role

     in

     France

    's

     politics

     and

     economy

    ,

     and

     is

     a

     major

     center

     for

     international

     trade

     and

     diplomacy

    .

     Despite

     facing

     challenges

     such

     as

     climate

     change

     and

     terrorism

    ,

     Paris

     continues

     to

     be

     a

     vibrant

     and

     dynamic

     city

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     increasingly

     bright

     as

     it

     continues

     to

     evolve

     and

     become

     more

     sophisticated

    .

     Here

     are

     some

     potential

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     with

     human

    -like

     intelligence

    :

     AI

     systems

     will

     continue

     to

     become

     more

     sophisticated

     and

     human

    -like

    ,

     with

     more

     sophisticated

     decision

    -making

     and

     self

    -im

    pro

    ving

     capabilities

    .

     This

     means

     that

     AI

     systems

     will

     be

     able

     to

     adapt

     to

     new

     situations

     and

     make

     decisions

     based

     on

     context

     and

     human

     values

    .
    


    2

    .

     Greater

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     ability

     to

     collect

     and

     analyze

     large

     amounts

     of

     medical

     data

    ,

     AI

     has

     the

     potential

     to

     revolution

    ize

     the

     way

     we

     treat

     and

     diagnose

     diseases

    .

     AI

     systems

     could

     be

     used

     to

     analyze

     medical

     images

    ,

     predict

     disease

    



```python
llm.shutdown()
```
