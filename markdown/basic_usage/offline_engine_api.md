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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:57,  1.04s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:57,  1.04s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.04s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]

    Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:04,  9.67it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]

    Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 15.86it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 21.16it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 21.16it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 21.16it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 21.16it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 21.16it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 21.16it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:04<00:00, 25.54it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 30.88it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 37.74it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:04<00:00, 47.80it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:04<00:00, 47.80it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:04<00:00, 47.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.93 GB):   2%|▏         | 1/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.90 GB):   2%|▏         | 1/58 [00:00<00:07,  7.53it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.90 GB):   3%|▎         | 2/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.90 GB):   3%|▎         | 2/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.90 GB):   5%|▌         | 3/58 [00:00<00:07,  7.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.90 GB):   5%|▌         | 3/58 [00:00<00:07,  7.55it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.90 GB):   7%|▋         | 4/58 [00:00<00:06,  7.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.90 GB):   7%|▋         | 4/58 [00:00<00:06,  7.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.90 GB):   9%|▊         | 5/58 [00:00<00:06,  7.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.89 GB):   9%|▊         | 5/58 [00:00<00:06,  7.96it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.89 GB):  10%|█         | 6/58 [00:00<00:06,  8.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.88 GB):  10%|█         | 6/58 [00:00<00:06,  8.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.88 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.88 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=53.88 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.88 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.87 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.87 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.87 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.86 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.86 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.86 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.86 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.17it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=53.85 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.85 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.85 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.85it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.85 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.85it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.85 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.84 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.84 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.84 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.84 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.82 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.63it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=53.82 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.90it/s]Capturing num tokens (num_tokens=960 avail_mem=53.83 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.90it/s] Capturing num tokens (num_tokens=896 avail_mem=53.83 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.90it/s]Capturing num tokens (num_tokens=896 avail_mem=53.83 GB):  40%|███▉      | 23/58 [00:01<00:02, 16.14it/s]Capturing num tokens (num_tokens=832 avail_mem=53.83 GB):  40%|███▉      | 23/58 [00:01<00:02, 16.14it/s]Capturing num tokens (num_tokens=768 avail_mem=53.82 GB):  40%|███▉      | 23/58 [00:02<00:02, 16.14it/s]Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  40%|███▉      | 23/58 [00:02<00:02, 16.14it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 17.93it/s]Capturing num tokens (num_tokens=640 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 17.93it/s]Capturing num tokens (num_tokens=576 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 17.93it/s]Capturing num tokens (num_tokens=512 avail_mem=53.80 GB):  45%|████▍     | 26/58 [00:02<00:01, 17.93it/s]Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 17.93it/s]Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 22.97it/s]Capturing num tokens (num_tokens=448 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 22.97it/s]Capturing num tokens (num_tokens=416 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 22.97it/s]

    Capturing num tokens (num_tokens=384 avail_mem=53.81 GB):  52%|█████▏    | 30/58 [00:02<00:01, 22.97it/s]Capturing num tokens (num_tokens=384 avail_mem=53.81 GB):  57%|█████▋    | 33/58 [00:02<00:01, 22.50it/s]Capturing num tokens (num_tokens=352 avail_mem=53.81 GB):  57%|█████▋    | 33/58 [00:02<00:01, 22.50it/s]Capturing num tokens (num_tokens=320 avail_mem=53.80 GB):  57%|█████▋    | 33/58 [00:02<00:01, 22.50it/s]Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  57%|█████▋    | 33/58 [00:02<00:01, 22.50it/s]Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.97it/s]Capturing num tokens (num_tokens=256 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.97it/s]

    Capturing num tokens (num_tokens=240 avail_mem=53.79 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.97it/s]Capturing num tokens (num_tokens=224 avail_mem=53.79 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.97it/s]Capturing num tokens (num_tokens=224 avail_mem=53.79 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.89it/s]Capturing num tokens (num_tokens=208 avail_mem=53.79 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.89it/s]Capturing num tokens (num_tokens=192 avail_mem=53.79 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.89it/s]Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.89it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.07it/s]Capturing num tokens (num_tokens=160 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.07it/s]Capturing num tokens (num_tokens=144 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.07it/s]Capturing num tokens (num_tokens=128 avail_mem=53.77 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.07it/s]Capturing num tokens (num_tokens=128 avail_mem=53.77 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.31it/s]Capturing num tokens (num_tokens=112 avail_mem=53.77 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.31it/s]Capturing num tokens (num_tokens=96 avail_mem=53.77 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.31it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=53.76 GB):  78%|███████▊  | 45/58 [00:03<00:00, 22.31it/s]Capturing num tokens (num_tokens=80 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.57it/s]Capturing num tokens (num_tokens=64 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.57it/s]Capturing num tokens (num_tokens=48 avail_mem=53.75 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.57it/s]Capturing num tokens (num_tokens=32 avail_mem=53.75 GB):  83%|████████▎ | 48/58 [00:03<00:00, 22.57it/s]Capturing num tokens (num_tokens=32 avail_mem=53.75 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.58it/s]Capturing num tokens (num_tokens=28 avail_mem=53.75 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.58it/s]

    Capturing num tokens (num_tokens=24 avail_mem=53.74 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.58it/s]Capturing num tokens (num_tokens=20 avail_mem=53.74 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.58it/s]Capturing num tokens (num_tokens=20 avail_mem=53.74 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.88it/s]Capturing num tokens (num_tokens=16 avail_mem=53.74 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.88it/s]Capturing num tokens (num_tokens=12 avail_mem=53.70 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.88it/s]

    Capturing num tokens (num_tokens=8 avail_mem=53.70 GB):  93%|█████████▎| 54/58 [00:03<00:00, 22.88it/s] Capturing num tokens (num_tokens=8 avail_mem=53.70 GB):  98%|█████████▊| 57/58 [00:03<00:00, 20.31it/s]Capturing num tokens (num_tokens=4 avail_mem=53.70 GB):  98%|█████████▊| 57/58 [00:03<00:00, 20.31it/s]Capturing num tokens (num_tokens=4 avail_mem=53.70 GB): 100%|██████████| 58/58 [00:03<00:00, 16.39it/s]


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
    Generated text:  Rob and this is the latest installment in the smart speaker world. This week we are talking about sustainability and its impact on the environment. Rob is a great speaker with a great presence. So I thought I would start by asking Rob what sustainability means to him. Rob is a realist and he seems to be very much in touch with what it means to him to be sustainable. Sustainability to Rob means to look at the future, to think about the impact of our actions on the environment and to make decisions that are in line with that thinking. Rob spoke about some of the things he has done to try and make a difference with his own
    ===============================
    Prompt: The president of the United States is
    Generated text:  51 years old now. If the president's current age is represented by \( P \) and the president's age in the year 2033 is represented by \( P_{2033} \), how many years from now will the president be three times as old as he will be in the year 2033? 
    
    (A) 5
    (B) 8
    (C) 10
    (D) 12
    (E) 15
    To determine how many years from now the president will be three times as old as he will be in the year 2033
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. Brussels C. Lyon D. Nice
    Answer: A
    
    In which area of the Forbidden City is the Qingming Festival pavilion located? A. Hall of Supreme Harmony B. Hall of Preserving Virtue C. Hall of Supreme Harmony (Duplicate option, ignore) D. Hall of Preserving Virtue (Duplicate option, ignore)
    Answer: C
    
    In the case of a fire, which of the following practices is incorrect?
    A. Immediately go to the refuge floor to escape;
    B. Cover your mouth and nose with a wet towel;
    C. Evacuate quickly using the nearest emergency evacuation
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain as new, complex and unstable technologies are constantly emerging. Some experts suggest that the automotive industry could be the first to adopt these new technologies because they can help create a greener future. The automobile industry is not the only one that is looking to adopt new technologies in order to achieve sustainability. Some companies are already thinking about adopting new technologies in their supply chain and supply chain management.
    Some of these companies are utilizing AI and robotics in order to streamline their operations. By using AI, they can analyze data and make better decisions. They can also automate some of the tasks that would otherwise be done by people. For example, they can optimize


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. The city is a popular tourist destination, with millions of visitors each year. Paris is a cultural and artistic center, with many museums, galleries, and theaters. It is also a major financial center, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. More advanced hardware: As AI technology continues to advance, we may see the development of even more powerful hardware that can process and analyze large amounts of data more efficiently.
    
    3. Greater integration with other technologies: AI is likely to become more integrated with other technologies, such as machine
    


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
    Generated text:  ___________ and I am ___________.
    
    Hello, my name is ___________ and I am ___________.
    
    Choose from the following options:
    - Just my name.
    - Just my role, such as a teacher or a police officer.
    - My age, height, and weight.
    - Something personal, such as a pet or a hobby. 
    - Something else. (Please provide an example) Hello, my name is [Your Name] and I am [Your Profession/Role]. What brings you to the table today?
    
    Hello, my name is [Your Name] and I am [Your Profession/Role]. What brings
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To elaborate on this statement, Paris is the largest city in France, located in the department of Paris, in the center of the country, and serves as the official capital of France. It was founded in the 6th century by the Romans, and has been the capital since the 14th century. Paris is known for its rich history, vibrant culture, and beautiful architecture, including the Eiffel Tower and Notre-Dame Cathedral. It is also a major center for fashion, finance, and international trade, and is home to many of the world's most famous landmarks and attractions, including the Louvre Museum
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but some possible trends that are likely to shape the development of the technology include:
    
    1. Improved accuracy and precision: One of the biggest challenges facing AI is its ability to achieve the level of accuracy and precision required in certain applications. Future trends could see AI algorithms become more sophisticated, capable of performing tasks with greater accuracy and precision.
    
    2. Increased transparency and explainability: As AI systems become more complex, they may need to be made more transparent and explainable. This could involve developing algorithms that can be easily understood and debugged, as well as providing feedback to users on how their decisions were made.
    
    3. Increased ethical considerations


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

     Jane

    ,

     and

     I

    'm

     a

     talented

     writer

     with

     a

     knack

     for

     storytelling

    .

     I

    'm

     passionate

     about

     using

     my

     creative

     skills

     to

     bring

     characters

     to

     life

     on

     the

     page

    .

     Whether

     it

    's

     in

     fiction

     or

     non

    -fiction

    ,

     I

    'm

     always

     up

     for

     a

     good

     adventure

    .

     Thank

     you

     for

     considering

     me

     for

     an

     interview

    .

     That

     sounds

     like

     a

     great

     fit

    .

     What

    's

     one

     of

     your

     favorite

     writing

     projects

     to

     work

     on

    ?

     As

     a

     freelance

     writer

    ,

     I

     love

     collaborating

     with

     clients

     to

     bring

     their

     ideas

     to

     life

     on

     the

     page

    .

     Sometimes

     it

    's

     a

     little

     challenging

     to

     create

     a

     project

     that

    's

     both

     compelling

     and

     engaging

    ,

     so

     I

     try

     to

     take

     a

     break

     from

     writing

     to

     think

     about

     the

     client

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     Correct

    


    B

    .

     Incorrect

    


    A

    .

     Correct

    
    


    Paris

     is

     the

     capital

     of

     France

     and

     the

     largest

     city

     in

     the

     country

    .

     It

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

     vibrant

     culture

    .

     The

     city

     is

     also

     a

     major

     economic

     center

    ,

     with

     a

     strong

     influence

     on

     French

     politics

     and

     international

     affairs

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

     each

     year

     to

     explore

     its

     beautiful

     sights

    ,

     vibrant

     nightlife

    ,

     and

     cultural

     attractions

    .

     The

     city

     is

     also

     home

     to

     important

     educational

     institutions

     such

     as

     the

     Paris

     Institute

     of

     Fine

     Arts

     and

     the

     University

     of

     Paris

    .

     Paris

     is

     a

     city

     of

     contrasts

     and

     challenges

    ,

     making

     it

     a

     fascinating

     and

     fascinating

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     transformative

     in

     a

     number

     of

     ways

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     More

     widespread

     use

     of

     AI

     in

     healthcare

     and

     medicine

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

     diagnosis

    ,

     treatment

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

     improve

    ,

     we

     can

     expect

     to

     see

     it

     becoming

     more

     widely

     used

     in

     medicine

    ,

     such

     as

     in

     developing

     new

     drugs

     and

     treatments

     for

     diseases

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     transportation

    :

     As

     AI

     becomes

     more

     widespread

    ,

     we

     can

     expect

     to

     see

     it

     used

     in

     transportation

     to

     improve

     safety

    ,

     efficiency

    ,

     and

     reduce

     the

     number

     of

     accidents

    .

     Autonomous

     vehicles

     and

     AI

    -powered

     self

    -driving

     cars

     are

     already

     being

     developed

    ,

    



```python
llm.shutdown()
```
