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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.53it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.53it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.53it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:20,  2.53it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.42it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.42it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  7.56it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  7.56it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  7.56it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:05,  7.56it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:05,  7.56it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.09it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.09it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.09it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.09it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 13.09it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 13.09it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 13.09it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 13.09it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 15.55it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 15.55it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 15.55it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 15.55it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 19.63it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 22.61it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 22.61it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:01, 22.61it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:06<00:01, 22.61it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 23.54it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 23.54it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 23.54it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 23.54it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 23.54it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 26.79it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 26.79it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 26.79it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 26.79it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 26.79it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 28.64it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 28.64it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 28.64it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 28.64it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 28.64it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 29.82it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 29.82it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 29.82it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 29.82it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 29.82it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 30.52it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 30.52it/s]

    Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 30.52it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 30.52it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 30.52it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 32.09it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 32.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.00 GB):   2%|▏         | 1/58 [00:00<00:09,  6.31it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.97 GB):   2%|▏         | 1/58 [00:00<00:09,  6.31it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.97 GB):   3%|▎         | 2/58 [00:00<00:08,  6.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.96 GB):   3%|▎         | 2/58 [00:00<00:08,  6.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.96 GB):   5%|▌         | 3/58 [00:00<00:08,  6.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.96 GB):   5%|▌         | 3/58 [00:00<00:08,  6.55it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.96 GB):   7%|▋         | 4/58 [00:00<00:08,  6.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.95 GB):   7%|▋         | 4/58 [00:00<00:08,  6.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.95 GB):   9%|▊         | 5/58 [00:00<00:07,  6.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.95 GB):   9%|▊         | 5/58 [00:00<00:07,  6.82it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.95 GB):  10%|█         | 6/58 [00:00<00:07,  7.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.93 GB):  10%|█         | 6/58 [00:00<00:07,  7.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.93 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.92 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.49it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.92 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.92 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.92 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.91 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.24it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.91 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.90 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.90 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.90 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.90 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.86it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.90 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.90 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.87 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.87 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.88 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.88 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.98it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.88 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.87 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.87 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.87 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.86 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.84 GB):  33%|███▎      | 19/58 [00:02<00:02, 14.80it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=53.84 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=960 avail_mem=53.85 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.01it/s] Capturing num tokens (num_tokens=896 avail_mem=53.84 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=832 avail_mem=53.84 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=832 avail_mem=53.84 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.98it/s]Capturing num tokens (num_tokens=768 avail_mem=53.83 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.98it/s]Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.98it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.42it/s]Capturing num tokens (num_tokens=640 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.42it/s]Capturing num tokens (num_tokens=576 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.42it/s]Capturing num tokens (num_tokens=512 avail_mem=53.80 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.42it/s]Capturing num tokens (num_tokens=512 avail_mem=53.80 GB):  50%|█████     | 29/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  50%|█████     | 29/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=448 avail_mem=53.81 GB):  50%|█████     | 29/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=416 avail_mem=53.81 GB):  50%|█████     | 29/58 [00:02<00:01, 20.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.81 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.76it/s]Capturing num tokens (num_tokens=384 avail_mem=53.81 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.76it/s]Capturing num tokens (num_tokens=352 avail_mem=53.80 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.76it/s]Capturing num tokens (num_tokens=320 avail_mem=53.80 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.76it/s]Capturing num tokens (num_tokens=320 avail_mem=53.80 GB):  60%|██████    | 35/58 [00:02<00:00, 24.07it/s]Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  60%|██████    | 35/58 [00:02<00:00, 24.07it/s]Capturing num tokens (num_tokens=256 avail_mem=53.79 GB):  60%|██████    | 35/58 [00:02<00:00, 24.07it/s]Capturing num tokens (num_tokens=240 avail_mem=53.79 GB):  60%|██████    | 35/58 [00:02<00:00, 24.07it/s]

    Capturing num tokens (num_tokens=240 avail_mem=53.79 GB):  66%|██████▌   | 38/58 [00:02<00:00, 25.32it/s]Capturing num tokens (num_tokens=224 avail_mem=53.79 GB):  66%|██████▌   | 38/58 [00:02<00:00, 25.32it/s]Capturing num tokens (num_tokens=208 avail_mem=53.78 GB):  66%|██████▌   | 38/58 [00:02<00:00, 25.32it/s]Capturing num tokens (num_tokens=192 avail_mem=53.78 GB):  66%|██████▌   | 38/58 [00:02<00:00, 25.32it/s]Capturing num tokens (num_tokens=192 avail_mem=53.78 GB):  71%|███████   | 41/58 [00:02<00:00, 25.80it/s]Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  71%|███████   | 41/58 [00:02<00:00, 25.80it/s]Capturing num tokens (num_tokens=160 avail_mem=53.78 GB):  71%|███████   | 41/58 [00:02<00:00, 25.80it/s]Capturing num tokens (num_tokens=144 avail_mem=53.77 GB):  71%|███████   | 41/58 [00:02<00:00, 25.80it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.77 GB):  76%|███████▌  | 44/58 [00:02<00:00, 26.66it/s]Capturing num tokens (num_tokens=128 avail_mem=53.77 GB):  76%|███████▌  | 44/58 [00:02<00:00, 26.66it/s]Capturing num tokens (num_tokens=112 avail_mem=53.77 GB):  76%|███████▌  | 44/58 [00:02<00:00, 26.66it/s]Capturing num tokens (num_tokens=96 avail_mem=53.76 GB):  76%|███████▌  | 44/58 [00:02<00:00, 26.66it/s] Capturing num tokens (num_tokens=80 avail_mem=53.76 GB):  76%|███████▌  | 44/58 [00:03<00:00, 26.66it/s]Capturing num tokens (num_tokens=80 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.61it/s]Capturing num tokens (num_tokens=64 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.61it/s]Capturing num tokens (num_tokens=48 avail_mem=53.75 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.61it/s]Capturing num tokens (num_tokens=32 avail_mem=53.75 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.61it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.74 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.61it/s]Capturing num tokens (num_tokens=28 avail_mem=53.74 GB):  90%|████████▉ | 52/58 [00:03<00:00, 29.83it/s]Capturing num tokens (num_tokens=24 avail_mem=53.74 GB):  90%|████████▉ | 52/58 [00:03<00:00, 29.83it/s]Capturing num tokens (num_tokens=20 avail_mem=53.74 GB):  90%|████████▉ | 52/58 [00:03<00:00, 29.83it/s]Capturing num tokens (num_tokens=16 avail_mem=53.74 GB):  90%|████████▉ | 52/58 [00:03<00:00, 29.83it/s]Capturing num tokens (num_tokens=12 avail_mem=53.73 GB):  90%|████████▉ | 52/58 [00:03<00:00, 29.83it/s]Capturing num tokens (num_tokens=12 avail_mem=53.73 GB):  97%|█████████▋| 56/58 [00:03<00:00, 31.23it/s]Capturing num tokens (num_tokens=8 avail_mem=53.73 GB):  97%|█████████▋| 56/58 [00:03<00:00, 31.23it/s] Capturing num tokens (num_tokens=4 avail_mem=53.73 GB):  97%|█████████▋| 56/58 [00:03<00:00, 31.23it/s]

    Capturing num tokens (num_tokens=4 avail_mem=53.73 GB): 100%|██████████| 58/58 [00:03<00:00, 17.25it/s]


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
    Generated text:  Jacob, I am 15 years old, and I was born in 1999. What is your favorite color? My favorite color is blue. It brings me joy and peacefulness. What type of hobby do you have? I love to read books, watch movies, and play video games. What is your dream job? I would like to be a teacher. What is your favorite food? I love cookies and chocolate. What is your favorite TV show or movie? I love "Friends" and "The Office." What is your favorite place to stay in your room? I like my bed. What is your favorite
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have on different parts of the country. The United States has 50 states, and each state can be represented by a different state. For the upcoming election, the president wants to know the total number of ways he can choose which state will have the most military bases and which state will have the fewest. Each state can have anywhere from 0 to 10 military bases, and the president wants to know the probability of this happening.
    
    To help with this, the president has calculated that there are 250 possible combinations for the number of military bases, ranging from 0 to
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    B. Lyon
    C. Nice
    D. Rome
    
    To determine the capital of France, let's examine each option step by step:
    
    1. **Paris**: This is the capital of France. It is known for its historical significance and is a major city in both the European and American contexts.
    2. **Lyon**: This is a city in France, but it is not the capital. Lyon is a city in the Languedoc region, and it is known for its beautiful architecture and history.
    3. **Nice**: This is a city in France, but it is not the capital. Nice
    ===============================
    Prompt: The future of AI is
    Generated text:  imminent, and many of the recent developments in the field are well underway. There is a need to understand and predict the potential consequences of these developments, and this requires a multidisciplinary approach, including economists, engineers, and policymakers. Currently, there is a growing body of research on the future of AI, including its impact on employment, privacy, and security. This paper analyzes these issues and proposes a range of policy responses to help navigate the complex challenges of the future of AI. I will discuss the challenges of implementing AI, the potential benefits, and the ethical considerations that should guide AI development. I will also explore the intersection of AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have [number] years of experience in [industry]. I'm a [gender] person, and I have [number] friends. I'm [number] years old, and I have [number] children. I'm [number] years old, and I have [number] pets. I'm [number] years old, and I have [number] hobbies. I'm [number] years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination and a major economic center. Paris is home to many famous French artists, writers, and musicians, and is known for its rich cultural heritage and historical significance. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the European Parliament. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage, and is a major center of politics, business, and entertainment in Europe.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased automation: AI is already becoming more and more integrated into our daily lives, from self-driving cars to chatbots that can understand and respond to customer inquiries. As the technology continues to advance, we can expect to see even more automation in the workplace, as well as in other areas such as healthcare, finance, and manufacturing.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm [Type] [Example Job] and I'm originally from [Location] and I'm a [Favorite Activity] enthusiast. How can I be helpful today? Let me know if you need help with anything. [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your Activity]? [Name] [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That statement accurately captures the historical and cultural importance of Paris as the capital city of France. The city is widely recognized as the most famous city in the world, with its rich history, diverse culture, and iconic landmarks, such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is also known for its fashion industry, gourmet food scene, and its residents' love of Parisian cuisine and fashion. Its size and importance also make it one of the most populous cities in the world, with a population of approximately 2.2 million people. Paris is a bustling metropolis with a diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a complex and rapidly evolving field with many possible trends. Here are some of the most promising areas where AI is expected to continue to advance:
    
    1. Advancements in natural language processing (NLP): As AI systems become more sophisticated, they will be able to understand and generate human language better than ever before. This will allow for more complex interactions between humans and machines, such as natural language translation, language modeling, and speech synthesis.
    
    2. AI-driven healthcare: AI is already being used to analyze medical images and provide diagnosis and treatment recommendations. As AI technology continues to improve, it may be able to provide even more accurate diagnoses and treatments


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

     an

     [

    occupation

    ].

     I

    'm

     currently

     working

     as

     a

     [

    career

     or

     profession

    ]

     and

     I

     have

     been

     passionate

     about

     [

    personal

     interest

     or

     hobby

    ]

     since

     childhood

    .

     I

     enjoy

     [

    reason

     for

     interest

    ]

     and

     I

     strive

     to

     be

     [

    character

     trait

     or

     behavior

    ].

     I

    'm

     always

     looking

     to

     learn

     new

     things

     and

     I

    'm

     always

     eager

     to

     contribute

     to

     [

    specific

     field

     or

     cause

    ].

     I

     believe

     in

     [

    value

     or

     belief

    ]

     and

     I

     strive

     to

     be

     a

     [

    other

     trait

     or

     trait

    ].

     I

    'm

     always

     up

     for

     a

     challenge

     and

     I

    'm

     excited

     to

     contribute

     to

     the

     community

    .

     What

    's

     your

     favorite

     color

     and

     why

    ?


    In

     your

     free

     time

    ,

     you

     enjoy

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     also

     known

     as

     the

     City

     of

     Light

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     known

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

     vibrant

     art

     scene

    ,

     unique

     architecture

    ,

     and

     excellent

     food

     and

     wine

    .

     Its

     famous

     landmarks

     include

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     it

    's

     also

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

     also

     known

     for

     its

     French

     cuisine

    ,

     which

     is

     famous

     for

     its

     cro

    iss

    ants

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     exponential

     growth

    ,

     significant

     changes

     in

     the

     way

     AI

     is

     used

    ,

     and

     an

     increasing

     emphasis

     on

     ethical

     and

     social

     implications

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

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     increasingly

     being

     integrated

     with

     other

     technologies

    ,

     such

     as

     IoT

    ,

     blockchain

    ,

     and

     advanced

     analytics

    ,

     to

     create

     new

     and

     more

     sophisticated

     applications

    .
    


    2

    .

     Personal

    ized

     AI

    :

     AI

     systems

     will

     become

     more

     personalized

    ,

     with

     each

     user

     receiving

     a

     unique

     and

     targeted

     experience

     based

     on

     their

     needs

     and

     preferences

    .
    


    3

    .

     Artificial

     intelligence

     that

     is

     more

     ethical

     and

     reliable

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     make

     decisions

     and

     take

     actions

     that

     are

     beneficial

     to

    



```python
llm.shutdown()
```
