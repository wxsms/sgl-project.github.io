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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:44,  1.87s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:44,  1.87s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.07s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.07s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:58,  1.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:27,  1.94it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:27,  1.94it/s]

    Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:27,  1.94it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.14it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.14it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:16,  3.14it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:10,  4.60it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:10,  4.60it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:10,  4.60it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:10,  4.60it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:06,  7.10it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:06,  7.10it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:06,  7.10it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:06,  7.10it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 12.25it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 12.25it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 12.25it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.46it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 19.94it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 19.94it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:01, 19.94it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:01, 22.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:00, 23.71it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:00, 23.71it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:00, 23.71it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:00, 23.71it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:06<00:00, 24.81it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:06<00:00, 24.81it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:00, 24.81it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:00, 24.81it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 25.75it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 25.75it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 25.75it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 25.75it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 26.01it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 26.01it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 26.01it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 26.01it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 26.91it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 26.91it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 26.91it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 26.91it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 26.71it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 26.71it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 26.71it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 26.71it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 26.71it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 27.94it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 27.94it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 27.94it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 27.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.34 GB):   2%|▏         | 1/58 [00:00<00:13,  4.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.31 GB):   2%|▏         | 1/58 [00:00<00:13,  4.22it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.31 GB):   3%|▎         | 2/58 [00:00<00:12,  4.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.96 GB):   3%|▎         | 2/58 [00:00<00:12,  4.45it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.96 GB):   5%|▌         | 3/58 [00:00<00:12,  4.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.36 GB):   5%|▌         | 3/58 [00:00<00:12,  4.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.36 GB):   7%|▋         | 4/58 [00:00<00:12,  4.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.96 GB):   7%|▋         | 4/58 [00:00<00:12,  4.47it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=38.96 GB):   9%|▊         | 5/58 [00:01<00:11,  4.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.41 GB):   9%|▊         | 5/58 [00:01<00:11,  4.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.41 GB):  10%|█         | 6/58 [00:01<00:10,  4.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.40 GB):  10%|█         | 6/58 [00:01<00:10,  4.76it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.40 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.94 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.94 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.45 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.33it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.45 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.94 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.94 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.88it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.49 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.88it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.49 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.49 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.49 GB):  21%|██        | 12/58 [00:02<00:07,  6.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.93 GB):  21%|██        | 12/58 [00:02<00:07,  6.14it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.93 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.51 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.51 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.92 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.31it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.92 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.91 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.91 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.53 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.73it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.53 GB):  29%|██▉       | 17/58 [00:02<00:05,  6.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.90 GB):  29%|██▉       | 17/58 [00:02<00:05,  6.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.90 GB):  31%|███       | 18/58 [00:03<00:05,  7.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.54 GB):  31%|███       | 18/58 [00:03<00:05,  7.18it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=38.54 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.90 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.90 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.55 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.00it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.55 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.98it/s]Capturing num tokens (num_tokens=960 avail_mem=38.89 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.98it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=38.89 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.98it/s]Capturing num tokens (num_tokens=896 avail_mem=38.89 GB):  40%|███▉      | 23/58 [00:03<00:05,  6.32it/s]Capturing num tokens (num_tokens=832 avail_mem=38.58 GB):  40%|███▉      | 23/58 [00:03<00:05,  6.32it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.88 GB):  40%|███▉      | 23/58 [00:04<00:05,  6.32it/s]Capturing num tokens (num_tokens=768 avail_mem=38.88 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.09it/s]Capturing num tokens (num_tokens=704 avail_mem=38.60 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.09it/s]

    Capturing num tokens (num_tokens=704 avail_mem=38.60 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.50it/s]Capturing num tokens (num_tokens=640 avail_mem=38.87 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.50it/s]Capturing num tokens (num_tokens=640 avail_mem=38.87 GB):  47%|████▋     | 27/58 [00:04<00:04,  7.60it/s]Capturing num tokens (num_tokens=576 avail_mem=38.62 GB):  47%|████▋     | 27/58 [00:04<00:04,  7.60it/s]

    Capturing num tokens (num_tokens=576 avail_mem=38.62 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.01it/s]Capturing num tokens (num_tokens=512 avail_mem=38.85 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.01it/s]Capturing num tokens (num_tokens=512 avail_mem=38.85 GB):  50%|█████     | 29/58 [00:04<00:03,  7.97it/s]Capturing num tokens (num_tokens=480 avail_mem=38.65 GB):  50%|█████     | 29/58 [00:04<00:03,  7.97it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.86 GB):  50%|█████     | 29/58 [00:04<00:03,  7.97it/s]Capturing num tokens (num_tokens=448 avail_mem=38.86 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.48it/s]Capturing num tokens (num_tokens=416 avail_mem=38.86 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.48it/s]

    Capturing num tokens (num_tokens=384 avail_mem=38.85 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.48it/s]Capturing num tokens (num_tokens=384 avail_mem=38.85 GB):  57%|█████▋    | 33/58 [00:05<00:02,  8.87it/s]Capturing num tokens (num_tokens=352 avail_mem=38.85 GB):  57%|█████▋    | 33/58 [00:05<00:02,  8.87it/s]

    Capturing num tokens (num_tokens=352 avail_mem=38.85 GB):  59%|█████▊    | 34/58 [00:05<00:02,  8.93it/s]Capturing num tokens (num_tokens=320 avail_mem=38.70 GB):  59%|█████▊    | 34/58 [00:05<00:02,  8.93it/s]Capturing num tokens (num_tokens=288 avail_mem=38.84 GB):  59%|█████▊    | 34/58 [00:05<00:02,  8.93it/s]Capturing num tokens (num_tokens=288 avail_mem=38.84 GB):  62%|██████▏   | 36/58 [00:05<00:02,  9.30it/s]Capturing num tokens (num_tokens=256 avail_mem=38.83 GB):  62%|██████▏   | 36/58 [00:05<00:02,  9.30it/s]

    Capturing num tokens (num_tokens=240 avail_mem=38.71 GB):  62%|██████▏   | 36/58 [00:05<00:02,  9.30it/s]Capturing num tokens (num_tokens=240 avail_mem=38.71 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.66it/s]Capturing num tokens (num_tokens=224 avail_mem=38.81 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.66it/s]

    Capturing num tokens (num_tokens=224 avail_mem=38.81 GB):  67%|██████▋   | 39/58 [00:05<00:01,  9.59it/s]Capturing num tokens (num_tokens=208 avail_mem=38.81 GB):  67%|██████▋   | 39/58 [00:05<00:01,  9.59it/s]Capturing num tokens (num_tokens=192 avail_mem=38.72 GB):  67%|██████▋   | 39/58 [00:05<00:01,  9.59it/s]Capturing num tokens (num_tokens=192 avail_mem=38.72 GB):  71%|███████   | 41/58 [00:05<00:01,  9.82it/s]Capturing num tokens (num_tokens=176 avail_mem=38.80 GB):  71%|███████   | 41/58 [00:05<00:01,  9.82it/s]

    Capturing num tokens (num_tokens=176 avail_mem=38.80 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.85it/s]Capturing num tokens (num_tokens=160 avail_mem=38.79 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.85it/s]Capturing num tokens (num_tokens=144 avail_mem=38.73 GB):  72%|███████▏  | 42/58 [00:06<00:01,  9.85it/s]Capturing num tokens (num_tokens=144 avail_mem=38.73 GB):  76%|███████▌  | 44/58 [00:06<00:01, 10.24it/s]Capturing num tokens (num_tokens=128 avail_mem=38.78 GB):  76%|███████▌  | 44/58 [00:06<00:01, 10.24it/s]

    Capturing num tokens (num_tokens=112 avail_mem=38.77 GB):  76%|███████▌  | 44/58 [00:06<00:01, 10.24it/s]Capturing num tokens (num_tokens=112 avail_mem=38.77 GB):  79%|███████▉  | 46/58 [00:06<00:01, 10.56it/s]Capturing num tokens (num_tokens=96 avail_mem=38.77 GB):  79%|███████▉  | 46/58 [00:06<00:01, 10.56it/s] Capturing num tokens (num_tokens=80 avail_mem=38.72 GB):  79%|███████▉  | 46/58 [00:06<00:01, 10.56it/s]

    Capturing num tokens (num_tokens=80 avail_mem=38.72 GB):  83%|████████▎ | 48/58 [00:06<00:00, 10.64it/s]Capturing num tokens (num_tokens=64 avail_mem=38.75 GB):  83%|████████▎ | 48/58 [00:06<00:00, 10.64it/s]Capturing num tokens (num_tokens=48 avail_mem=38.74 GB):  83%|████████▎ | 48/58 [00:06<00:00, 10.64it/s]

    Capturing num tokens (num_tokens=48 avail_mem=38.74 GB):  86%|████████▌ | 50/58 [00:06<00:00, 10.37it/s]Capturing num tokens (num_tokens=32 avail_mem=38.75 GB):  86%|████████▌ | 50/58 [00:06<00:00, 10.37it/s]Capturing num tokens (num_tokens=28 avail_mem=38.74 GB):  86%|████████▌ | 50/58 [00:06<00:00, 10.37it/s]Capturing num tokens (num_tokens=28 avail_mem=38.74 GB):  90%|████████▉ | 52/58 [00:06<00:00, 10.52it/s]Capturing num tokens (num_tokens=24 avail_mem=38.73 GB):  90%|████████▉ | 52/58 [00:06<00:00, 10.52it/s]

    Capturing num tokens (num_tokens=20 avail_mem=38.72 GB):  90%|████████▉ | 52/58 [00:06<00:00, 10.52it/s]Capturing num tokens (num_tokens=20 avail_mem=38.72 GB):  93%|█████████▎| 54/58 [00:07<00:00, 10.52it/s]Capturing num tokens (num_tokens=16 avail_mem=38.72 GB):  93%|█████████▎| 54/58 [00:07<00:00, 10.52it/s]Capturing num tokens (num_tokens=12 avail_mem=38.71 GB):  93%|█████████▎| 54/58 [00:07<00:00, 10.52it/s]

    Capturing num tokens (num_tokens=12 avail_mem=38.71 GB):  97%|█████████▋| 56/58 [00:07<00:00, 10.71it/s]Capturing num tokens (num_tokens=8 avail_mem=38.71 GB):  97%|█████████▋| 56/58 [00:07<00:00, 10.71it/s] Capturing num tokens (num_tokens=4 avail_mem=38.70 GB):  97%|█████████▋| 56/58 [00:07<00:00, 10.71it/s]Capturing num tokens (num_tokens=4 avail_mem=38.70 GB): 100%|██████████| 58/58 [00:07<00:00, 10.67it/s]Capturing num tokens (num_tokens=4 avail_mem=38.70 GB): 100%|██████████| 58/58 [00:07<00:00,  7.81it/s]


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
    Generated text:  Sasha and I'm a self-made success. I have a special skill that I use to help people. It's my ability to tap into the power of positive thinking. With this power, I can help people overcome challenges and achieve their goals. One of the things that has always been my passion is helping people. I know that there is something you could not overcome. If you ask me, I know there is something you can't avoid. My goal is to help you overcome challenges and take control of your life. Is it true that you have helped many people overcome challenges? Yes, I have helped many people overcome challenges. One of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the Cabinet of the Executive Branch of the U. S. government. He has the power to make the laws. The president is also known as the chief executive officer (C. E. ). He is the highest executive officer in the United States government. He can appoint cabinet members (such as the Secretary of the Treasury, the Secretary of State, and the Attorney General) and they can act for the president and make the laws for the country. The President can also act as a commander-in-chief, the power to take command of the armed forces. The president acts as a check and balance on the Congress (the legislative
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) London
    B) Paris
    C) Milan
    D) Berlin
    
    The capital of France is Paris. 
    Therefore, the answer is B. Paris. 
    
    It is important to note that the capital of France is actually located in the region of the same name in the centre of France. While Paris is the largest city in the region, it is not the capital of France. The capital of France is actually in the city of Paris, which is the capital of the French department of the same name. 
    
    The other options are not capitals of France:
    - London is the capital of England, not France.
    - Milan is
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. The field is growing rapidly and making significant progress. However, it is still a field that is not regulated and is often criticized by the public. In this article, we will discuss what is AI, who are its developers, the most common problems in the field, and some potential solutions to these problems. You may also find useful links to learn more about the topic.
    What is AI?
    AI is a subfield of computer science that deals with building machines that can perform tasks that usually require human intelligence. These machines are trained on large datasets to learn how to perform tasks like recognizing objects, playing games, and even driving cars.


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who has always been [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, and it is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is home to many famous French artists, writers, and musicians, and it is a major tourist destination. Despite its size, Paris is a city of contrasts, with its modern architecture and vibrant nightlife complement
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems become involved in decision-making processes, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and fairness.
    
    2. Integration of AI with other technologies: AI will continue to be integrated with other technologies such as blockchain, quantum computing, and biotechnology. This will create new opportunities for AI to be used in new ways and to solve complex problems.
    
    3.
    


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
    Generated text:  [Name]. I'm a [Occupation] who has been [Objective] for [Number of Years]. I enjoy [Enjoyment]. In my free time, I love [Activities]. I'm passionate about [Advantage]. What's your background and how have you come to be in your current position? [Optional: Feel free to provide any additional context or background information about your character that might help potential readers understand who you are and what you do.] [Your Name] [Your Occupation] [Your Objective] [Your Enjoyment] [Your Activities] [Your Background] [Your Advantages] [Optional: Additional Context]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and largest city of France, located on the northern bank of the Seine river and on the right bank of the Orne river in the Var department of the north-central region of France. It is situated on the Île de la Cité, just north of the Seine, and has a population of about 2.2 million, of which about half lives in the city. Paris is a major cultural, economic, and tourist centre. It is the oldest capital city in the world, the second oldest capital city in Europe, and the third oldest capital city in the world after the city of Rome
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly complex and interconnected, and to evolve rapidly. Here are some possible future trends in AI:
    
    1. Autonomous systems: AI will become more advanced and intelligent, allowing machines to operate independently without human oversight. This will have significant implications for industries such as transportation, manufacturing, and healthcare.
    
    2. Explainable AI: AI will become more transparent and explainable, allowing for better decision-making and accountability. This will require the development of advanced models that can be easily understood and explained to humans.
    
    3. Personalization: AI will be able to learn from user data to provide more personalized experiences. This will be achieved through the use


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

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm

     passionate

     about

     [

    Specific

     Achievement

     or

     Interest

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Challenge

     or

     Insp

    ire

     others

    ].

     What

    's

     your

     experience

     like

     at

     this

     company

    ?

     I

     thrive

     in

     [

    Role

    /

    Job

    ],

     and

     I

     love

     [

    Favorite

     Part

    icular

    ity

    /

    Activity

    /

    Event

    ].

     What

     makes

     you

     unique

    ?

     I

    'm

     [

    Unique

     Qual

    ification

     or

     Accom

    pl

    ishment

    ].

     How

     do

     you

     contribute

     to

     [

    Company

    's

     Mission

     or

     Values

    ]?

     I

    'm

     always

     striving

     to

     [

    Goal

     or

     Vision

    ]

     at

     [

    Company

     Name

    ],

     and

     I

    'm

     always

     open

     to

     [

    Challenge

     or

     Opportunity

    ].

     What

     do

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

    .

     It

     is

     also

     known

     as

     "

    La

     Ro

    che

     de

     Paris

    "

     and

     is

     the

     city

     where

     the

     E

    iff

    el

     Tower

     was

     originally

     built

    .

     The

     city

     has

     a

     rich

     history

     and

     culture

    ,

     with

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     the

     seat

     of

     the

     French

     government

     and

     the

     home

     of

     the

     French

     royal

     family

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     some

     trends

     that

     are

     likely

     to

     continue

     include

    :
    


    1

    .

     Increased

     AI

     diversity

     and

     inclus

    ivity

    :

     AI

     development

     will

     continue

     to

     grow

    ,

     with

     more

     developers

     and

     organizations

     embracing

     diversity

     and

     inclusion

     in

     the

     field

    .

     This

     will

     help

     to

     ensure

     that

     AI

     is

     used

     for

     positive

     and

     ethical

     reasons

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     AI

    :

     As

     more

     people

     become

     aware

     of

     the

     potential

     risks

     of

     AI

    ,

     there

     will

     be

     increased

     scrutiny

     of

     how

     AI

     is

     developed

     and

     used

    .

     This

     will

     lead

     to

     greater

     focus

     on

     ethical

     principles

     and

     standards

     in

     AI

     development

    .
    


    3

    .

     AI

     will

     become

     more

     integrated

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     with

     other

     technologies

    ,

     such

     as

     quantum

    



```python
llm.shutdown()
```
