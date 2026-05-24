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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.86it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:16,  3.01it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  7.79it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  7.79it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  7.79it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:05,  7.79it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:05,  7.79it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 15.79it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 15.79it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 15.79it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:02, 15.79it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:02, 15.79it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:02, 15.79it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:01, 21.18it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:01, 21.18it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:01, 21.18it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:01, 21.18it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:01, 21.18it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:01, 21.18it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 26.06it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 26.06it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 26.06it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 26.06it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 26.06it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 26.06it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 31.07it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 31.07it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 31.07it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 36.52it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 36.52it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 36.52it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 36.52it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 36.52it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 36.52it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 39.56it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 43.96it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 43.96it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 43.96it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 43.96it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 43.96it/s] 

    Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 43.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.58 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.55 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=54.55 GB):   3%|▎         | 2/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.55 GB):   3%|▎         | 2/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.55 GB):   5%|▌         | 3/58 [00:00<00:07,  7.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.54 GB):   5%|▌         | 3/58 [00:00<00:07,  7.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.54 GB):   7%|▋         | 4/58 [00:00<00:06,  7.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.54 GB):   7%|▋         | 4/58 [00:00<00:06,  7.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.54 GB):   9%|▊         | 5/58 [00:00<00:06,  7.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.54 GB):   9%|▊         | 5/58 [00:00<00:06,  7.98it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=54.54 GB):  10%|█         | 6/58 [00:00<00:06,  8.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.53 GB):  10%|█         | 6/58 [00:00<00:06,  8.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.53 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.52 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.62it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.52 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.52 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.52 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.51 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.43it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=54.51 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.51 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.51 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.51 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.51 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.79it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=54.50 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.50 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.50 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.50 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.65it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=54.50 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.49 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.49 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.47 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.55it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=54.47 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.80it/s]Capturing num tokens (num_tokens=960 avail_mem=54.48 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.80it/s] Capturing num tokens (num_tokens=896 avail_mem=54.48 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.80it/s]Capturing num tokens (num_tokens=896 avail_mem=54.48 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.93it/s]Capturing num tokens (num_tokens=832 avail_mem=54.48 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.93it/s]Capturing num tokens (num_tokens=768 avail_mem=54.47 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.93it/s]Capturing num tokens (num_tokens=704 avail_mem=54.47 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.93it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.47 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.78it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.78it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.78it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.78it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  50%|█████     | 29/58 [00:02<00:01, 16.71it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 16.71it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 16.71it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 16.71it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.96it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.96it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.96it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.96it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  60%|██████    | 35/58 [00:02<00:01, 19.01it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 19.01it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 19.01it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 19.01it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.84it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.84it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.84it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.84it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:02<00:00, 20.20it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:02<00:00, 20.20it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:02<00:00, 20.20it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:03<00:00, 20.20it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 20.97it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 20.97it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 20.97it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  76%|███████▌  | 44/58 [00:03<00:00, 20.97it/s] Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 21.32it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 21.32it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 21.32it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 21.32it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.56it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.56it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.56it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.56it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.01it/s]

    Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 22.11it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 22.11it/s] Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  97%|█████████▋| 56/58 [00:03<00:00, 22.11it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 15.68it/s]


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
    Generated text:  Dominique and I am a graphic designer, editor, writer, photographer, and travel blogger. I have always been fascinated with how the world looks and feels, and I use my creative abilities to help others discover, share and enjoy it. I hold a Bachelor's degree in Visual Communication from the University of Massachusetts. In 2013, I published a book of my work, Traveling the World Through My Eyes, which is now available on Amazon and over 2000 other retailers.
    I have worked for a variety of organizations in marketing, graphic design, and photography. I have created and worked with a variety of
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil, and the president of Brazil is 2/3 the age of the president of Russia. If the president of Russia is 50 years old, what is the average age of the three presidents?
    
    To determine the average age of the three presidents, we need to follow these steps:
    
    1. Identify the age of the president of Brazil.
    2. Determine the age of the president of Russia.
    3. Calculate the average age of the three presidents.
    
    First, we know that the president of Russia is 50 years old. According to the problem, the president of Brazil is \
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of the country of France. There are no other capitals. The capital of France is Paris.
    
    This is a true statement. The capital of France is Paris.
    
    Does this sentence follow logically from the given sentence? Yes, it follows logically from the given sentence.
    
    To verify this, I will compare the given statement and the conclusion:
    
    Given statement: "The capital of France is the capital of the country of France. There are no other capitals. The capital of France is Paris."
    
    Conclusion: "The capital of France is Paris."
    
    The conclusion matches exactly with the information provided in the given statement. Therefore, this sentence follows logically from
    ===============================
    Prompt: The future of AI is
    Generated text:  predicting cyber risks from real-world phishing attempts that show up in these network security tools, according to a new report from CrowdStrike, one of the world’s largest cybersecurity companies. Crowdstrike’s study shows that phishing attacks that targeted a large number of computers will continue to increase, leading to more sophisticated phishing tactics and more cyber crimes. CrowdStrike’s research, which is based on surveys of 50,000 employees worldwide, was published in the May 2023 edition of the Journal of Homeland Security and Security Review. According to the report, cyber criminals are already taking advantage of the increasing popularity of phishing attacks.


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name], and I'm excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic center, with a diverse population and a thriving arts scene. It is also home to many famous landmarks and attractions, including the Louvre, the Eiffel Tower, and the Notre-Dame Cathedral. Paris is a popular tourist destination, with millions of visitors annually. It is also a major center for politics, with the French Parliament located in the city. Overall, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI technologies. This could include things like voice assistants, self-driving cars, and chatbots that can understand and respond to human language.
    
    2. Greater focus on ethical and social implications: As AI becomes more integrated into our daily lives, there will be a greater focus on its ethical and social implications.
    


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
    Generated text:  ____. I have been working in the __ industry for __ years, gaining a wealth of knowledge in this field. I am a ____. My ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I am ____. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its beautiful architecture, rich culture, and iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum.
    
    The French capital city is widely regarded as one of the most beautiful cities in the world and has a rich cultural history dating back centuries. It is home to many iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion, art, and food scenes. Paris has a vibrant nightlife and is known for its fashion shows, art exhibitions, and other cultural events. The city is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  one of rapid evolution and growth, with many exciting developments shaping the next decade. Here are some possible trends in AI:
    
    1. Increased Use of AI in Healthcare: AI is already playing a significant role in healthcare, with predictive analytics and machine learning being used to help doctors make more accurate diagnoses and develop personalized treatment plans. With more data being collected and analyzed, AI is expected to become even more accurate and efficient in healthcare.
    
    2. AI in Finance: AI is already making a significant impact in finance, with algorithms used to identify investment opportunities, analyze market trends, and manage risk. With more data being collected and analyzed, AI is expected


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

     am

     a

     [

    position

    ]

     at

     [

    company

    ].

     My

     career

     in

     tech

     has

     been

     my

     path

     to

     [

    goal

    (s

    )],

     and

     I

     am

     excited

     to

     join

     [

    company

    ]

     as

     a

     [

    position

    ]

     in

     [

    role

    ].

     I

     am

     [

    age

    ]

     years

     old

     and

     I

     am

     an

     [

    occupation

    ]

     with

     [

    number

     of

     years

     in

     tech

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

     am

     passionate

     about

     [

    field

     of

     interest

    ],

     and

     I

     am

     committed

     to

     making

     a

     positive

     impact

     on

     [

    industry

    ]

     through

     my

     work

    .

     I

     am

     a

     [

    interest

     or

     hobby

    ]

     in

     [

    interest

     or

     hobby

    ],

     and

     I

     am

     determined

     to

     make

     my

     work

     meaningful

     and

     enjoyable

    .

     Thank

     you

    
    
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

     the

     most

     populous

     urban

     area

     in

     Europe

    .

     Its

     population

     is

     approximately

     

    1

    1

     million

     people

    .

     Paris

     is

     known

     for

     its

     historic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     distinctive

     French

     cuisine

     and

     vibrant

     street

     life

    .

     The

     city

     is

     a

     cultural

     and

     economic

     center

    ,

     with

     many

     famous

     museums

    ,

     art

     galleries

    ,

     theaters

    ,

     and

     restaurants

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     annually

    .

     The

     city

     is

     also

     an

     important

     transportation

     hub

    ,

     connecting

     major

     cities

     and

     countries

     across

     Europe

     and

     the

     world

    .

     Its

     history

    ,

     art

    ,

     and

     culture

     make

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     extremely

     diverse

     and

     complex

    ,

     with

     potential

     applications

     in

     a

     wide

     range

     of

     industries

     and

     fields

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     More

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     expected

     to

     become

     more

     common

     in

     the

     future

    ,

     but

     there

     will

     still

     be

     challenges

     and

     obstacles

     to

     overcome

    .

     The

     development

     of

     better

     algorithms

     and

     machine

     learning

     will

     be

     key

     to

     achieving

     a

     safer

     and

     more

     efficient

     future

     for

     these

     vehicles

    .
    


    2

    .

     Improved

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     improve

     diagnostics

    ,

     treatment

     planning

    ,

     and

     patient

     care

    .

     As

     AI

     technology

     continues

     to

     develop

    ,

     we

     can

     expect

     to

     see

     even

     more

     applications

     in

     healthcare

    ,

     including

     personalized

     medicine

    ,

     disease

     prediction

    ,

     and

     drug

    



```python
llm.shutdown()
```
