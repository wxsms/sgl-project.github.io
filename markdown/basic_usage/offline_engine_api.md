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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:55,  4.14s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:55,  4.14s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:39,  1.77s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:39,  1.77s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:04<01:39,  1.77s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:38,  1.42it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:38,  1.42it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:38,  1.42it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:21,  2.45it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:21,  2.45it/s]

    Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:21,  2.45it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:13,  3.78it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:13,  3.78it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:13,  3.78it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:08,  5.35it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:03, 11.15it/s]

    Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.97it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.97it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.97it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.97it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.97it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.26it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 19.26it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 22.33it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 22.33it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 22.33it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 22.33it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 22.33it/s]

    Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 25.25it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 25.25it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 25.25it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 25.25it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 25.25it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 27.52it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 27.52it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 27.52it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 27.52it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 27.52it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 32.70it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 32.70it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 32.70it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 32.70it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 32.70it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 33.46it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 33.46it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 34.50it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 34.50it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 37.08it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 37.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.84 GB):   2%|▏         | 1/58 [00:00<00:08,  6.49it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.78 GB):   2%|▏         | 1/58 [00:00<00:08,  6.49it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.78 GB):   3%|▎         | 2/58 [00:00<00:08,  6.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.74 GB):   3%|▎         | 2/58 [00:00<00:08,  6.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.74 GB):   5%|▌         | 3/58 [00:00<00:08,  6.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.78 GB):   5%|▌         | 3/58 [00:00<00:08,  6.63it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.78 GB):   7%|▋         | 4/58 [00:00<00:07,  6.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.79 GB):   7%|▋         | 4/58 [00:00<00:07,  6.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.79 GB):   9%|▊         | 5/58 [00:00<00:07,  7.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.78 GB):   9%|▊         | 5/58 [00:00<00:07,  7.13it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.78 GB):  10%|█         | 6/58 [00:00<00:06,  7.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.77 GB):  10%|█         | 6/58 [00:00<00:06,  7.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.77 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.76 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.69it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.76 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.76 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.76 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.75 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.37it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=52.75 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.74 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.74 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.72 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.90it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=52.71 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.71 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.72 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.71 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.41it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=52.71 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.72 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.71 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.71 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.70 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.96it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=52.70 GB):  31%|███       | 18/58 [00:02<00:04,  9.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.70 GB):  31%|███       | 18/58 [00:02<00:04,  9.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.69 GB):  31%|███       | 18/58 [00:02<00:04,  9.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.69 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.67 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.56it/s]

    Capturing num tokens (num_tokens=960 avail_mem=52.68 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.56it/s] Capturing num tokens (num_tokens=960 avail_mem=52.68 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.15it/s]Capturing num tokens (num_tokens=896 avail_mem=52.66 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.15it/s]Capturing num tokens (num_tokens=832 avail_mem=52.67 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.15it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.67 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=768 avail_mem=52.66 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=704 avail_mem=52.65 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=704 avail_mem=52.65 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.39it/s]Capturing num tokens (num_tokens=640 avail_mem=52.65 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.39it/s]Capturing num tokens (num_tokens=576 avail_mem=52.65 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.39it/s]Capturing num tokens (num_tokens=512 avail_mem=52.63 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.39it/s]

    Capturing num tokens (num_tokens=480 avail_mem=52.62 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.39it/s]Capturing num tokens (num_tokens=480 avail_mem=52.62 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.19it/s]Capturing num tokens (num_tokens=448 avail_mem=52.61 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.19it/s]Capturing num tokens (num_tokens=416 avail_mem=52.63 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.19it/s]Capturing num tokens (num_tokens=416 avail_mem=52.63 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.19it/s]Capturing num tokens (num_tokens=384 avail_mem=52.63 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.19it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.62 GB):  55%|█████▌    | 32/58 [00:03<00:01, 17.19it/s]Capturing num tokens (num_tokens=352 avail_mem=52.62 GB):  59%|█████▊    | 34/58 [00:03<00:01, 16.21it/s]Capturing num tokens (num_tokens=320 avail_mem=52.61 GB):  59%|█████▊    | 34/58 [00:03<00:01, 16.21it/s]Capturing num tokens (num_tokens=288 avail_mem=52.60 GB):  59%|█████▊    | 34/58 [00:03<00:01, 16.21it/s]Capturing num tokens (num_tokens=288 avail_mem=52.60 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.02it/s]Capturing num tokens (num_tokens=256 avail_mem=52.60 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.02it/s]

    Capturing num tokens (num_tokens=240 avail_mem=52.59 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.02it/s]Capturing num tokens (num_tokens=240 avail_mem=52.59 GB):  66%|██████▌   | 38/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=224 avail_mem=52.58 GB):  66%|██████▌   | 38/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=208 avail_mem=52.57 GB):  66%|██████▌   | 38/58 [00:03<00:01, 16.28it/s]Capturing num tokens (num_tokens=208 avail_mem=52.57 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.38it/s]Capturing num tokens (num_tokens=192 avail_mem=52.57 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.38it/s]

    Capturing num tokens (num_tokens=176 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.38it/s]Capturing num tokens (num_tokens=176 avail_mem=52.56 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.17it/s]Capturing num tokens (num_tokens=160 avail_mem=52.56 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.17it/s]Capturing num tokens (num_tokens=144 avail_mem=52.56 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.17it/s]Capturing num tokens (num_tokens=128 avail_mem=52.55 GB):  72%|███████▏  | 42/58 [00:03<00:00, 17.17it/s]Capturing num tokens (num_tokens=128 avail_mem=52.55 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.51it/s]Capturing num tokens (num_tokens=112 avail_mem=52.55 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.51it/s]

    Capturing num tokens (num_tokens=96 avail_mem=52.55 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.51it/s] Capturing num tokens (num_tokens=80 avail_mem=52.54 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.51it/s]Capturing num tokens (num_tokens=80 avail_mem=52.54 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.09it/s]Capturing num tokens (num_tokens=64 avail_mem=52.54 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.09it/s]Capturing num tokens (num_tokens=48 avail_mem=52.54 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.09it/s]Capturing num tokens (num_tokens=32 avail_mem=52.53 GB):  83%|████████▎ | 48/58 [00:03<00:00, 21.09it/s]Capturing num tokens (num_tokens=32 avail_mem=52.53 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.32it/s]Capturing num tokens (num_tokens=28 avail_mem=52.53 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.32it/s]

    Capturing num tokens (num_tokens=24 avail_mem=52.53 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.32it/s]Capturing num tokens (num_tokens=20 avail_mem=52.52 GB):  88%|████████▊ | 51/58 [00:04<00:00, 22.32it/s]Capturing num tokens (num_tokens=20 avail_mem=52.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 23.26it/s]Capturing num tokens (num_tokens=16 avail_mem=52.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 23.26it/s]Capturing num tokens (num_tokens=12 avail_mem=52.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 23.26it/s]Capturing num tokens (num_tokens=8 avail_mem=52.51 GB):  93%|█████████▎| 54/58 [00:04<00:00, 23.26it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=52.51 GB):  98%|█████████▊| 57/58 [00:04<00:00, 23.57it/s]Capturing num tokens (num_tokens=4 avail_mem=52.51 GB):  98%|█████████▊| 57/58 [00:04<00:00, 23.57it/s]Capturing num tokens (num_tokens=4 avail_mem=52.51 GB): 100%|██████████| 58/58 [00:04<00:00, 13.75it/s]


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
    Generated text:  Garth and I am 15 years old. I have a vivid imagination and am always in the mood to write fantasy stories. I recently started working with my teacher to write a fantasy story. I have found a book on the internet and I am not sure how to write this story. The story is going to be about a character who has to fight a demon. The demon is very powerful and has a lot of magical properties. Can you give me some advice on how to start the story? And maybe some tips on how to write about a demon with magical properties? I am very new to writing and I would greatly appreciate any
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to assess the efficacy of a new program that promises to improve the education system for middle school students. The president received 580 responses from students and educators regarding their opinions on the program. He wants to know the percentage of students who rated the program highly. How can he calculate this percentage? To calculate the percentage of students who rated the program highly, the president can use the following formula:
    
    \[
    \text{Percentage of highly rated students} = \left( \frac{\text{Number of highly rated students}}{\text{Total number of students}} \right) \times 100
    \]
    
    Given
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The main language spoken in Paris is French. It is the capital city of the country of France. Paris is the political, cultural, and economic centre of France. Paris was founded on January 1, 789 by Charles Martel. In 1801, it became the capital of the French Empire. In 1969, it became the capital of the French Republic. Paris is a city of many different neighborhoods and districts, such as the 11th, 13th, 15th, 17th, and 19th arrondissements.
    ===============================
    Prompt: The future of AI is
    Generated text:  transformative, but when it comes to autonomous vehicles, it is less a matter of when than where.
    The race is on to develop safe autonomous vehicles. This is necessary because autonomous vehicles could be responsible for a large number of deaths and injuries, and there is no clear solution to this problem. In 2018, the European Commission launched the Horizon 2020 program to develop autonomous vehicles. This program is one of the largest funding programs for autonomous vehicle research in the world, and it has produced many successful autonomous vehicles.
    In fact, the European Commission is working towards an autonomous vehicle fleet of 5 million vehicles. This


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


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the seat of the French government and the country's cultural and political capital. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is famous for its fashion, art, and cuisine, and is home to many world-renowned landmarks and attractions. It is a popular tourist destination and a major economic center in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its annual festivals and events,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more integrated into our daily lives, from manufacturing to healthcare to transportation. This will lead to increased automation and artificial intelligence, which will automate many tasks that are currently done by humans, such as manufacturing, transportation, and healthcare.
    
    2. AI-powered healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to become even more advanced in the future
    


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
    Generated text:  [Name], and I'm an experienced freelance writer with a deep passion for writing. I have a knack for crafting compelling stories and helping my clients tell their stories with confidence. My writing skills are second nature, and I bring a fresh perspective to every piece of content I produce. Whether it's a novel, a screenplay, or a blog post, I'm always eager to learn from and collaborate with new writers. Join me in exploring the world of writing with me! 🕺✨
    
    That's a great self-introduction! Can you tell me more about your writing process? What are some of the key elements you focus on during
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    I apologize, but I cannot fulfill this request. As an AI language model, I do not have access to current information about specific cities. If you have any other questions or need information about a different topic, feel free to ask!
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising and is likely to continue to grow and develop in many different ways. Here are some possible future trends in artificial intelligence:
    
    1. Increased Personalization: With the help of machine learning and natural language processing, AI will become more and more personal in its interactions with users. This will lead to more tailored experiences that are specifically designed to meet individual needs and preferences.
    
    2. Robotic and autonomous vehicles: We are seeing a growing trend towards more autonomous vehicles, which will be equipped with advanced AI capabilities such as self-driving technology, lane-keeping, and emergency braking.
    
    3. AI in Healthcare: AI is already making significant progress


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

    insert

     character

     name

    ],

     and

     I

    'm

     a

     [

    insert

     profession

     or

     hobby

    ]

     who enjoys

     [

    insert

     why

    ]

     in

     life

    .

     I

    'm

     currently

     working

     on

     [

    insert

     a

     current

     project

     or

     task

    ]

     and

     my

     goal

     is

     [

    insert

     what

     I

     want

     to

     achieve

     with

     this

     project

    ].

     I

    'm

     a

     [

    insert

     your

     occupation

    ],

     and

     I

    'm

     excited

     to

     find

     out

     what

     I

     can

     bring

     to

     the

     table

    .

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     contribute

     to

     the

     world

    .

     Thank

     you

     for

     having

     me

    !

     
    


    Please

     note

     that

     you

     can

     replace

     the

     placeholder

     names

    ,

     titles

    ,

     and

     tasks

     with

     your

     own

    ,

     and

     feel

     free

     to

     adjust

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     provides

     a

     clear

    ,

     concise

    ,

     and

     accurate

     overview

     of

     the

     capital

     city

    's

     location

     and

     name

    .

     It

    's

     a

     simple

     yet

     informative

     answer

     that

     adher

    es

     to

     the

     guidelines

     provided

    ,

     making

     it

     easier

     for

     the

     reader

     to

     understand

     the

     answer

    .

     Additionally

    ,

     it

     avoids

     any

     potential

     confusion

     or

     ambiguity

     by

     using

     the

     word

     "

    capital

    "

     instead

     of

     "

    capital

     city

    "

     which

     might

     be

     used

     interchange

    ably

     with

     "

    capital

    "

     in

     some

     contexts

    .

     
    


    Here

    's

     an

     alternative

     version

     using

     "

    city

    "

     instead

     of

     "

    capital

    ":
    


    The

     capital

     of

     France

     is

     Paris

    .

     
    


    This

     version

     uses

     the

     word

     "

    city

    "

     instead

     of

     "

    capital

    "

     and

     avoids

     any

     potential

     confusion

     or

     ambiguity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     vast

    .

     Here

     are

     some

     possible

     trends

     that

     are

     expected

     to

     shape

     the

     development

     of

     AI

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

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

     analyze

     medical

     images

    ,

     predict

     disease

     outbreaks

    ,

     and

     personalize

     treatments

    .

     As

     AI

     continues

     to

     improve

    ,

     it

     is

     likely

     that

     we

     will

     see

     increased

     use

     of

     AI

     in

     healthcare

    ,

     such

     as

     in

     the

     development

     of

     predictive

     models

     to

     aid

     in

     disease

     diagnosis

     and

     treatment

    .
    


    2

    .

     AI

     in

     manufacturing

    :

     AI

     is

     already

     being

     used

     in

     manufacturing

     to

     optimize

     production

     processes

    ,

     improve

     efficiency

    ,

     and

     reduce

     waste

    .

     As

     AI

     continues

     to

     improve

    ,

     it

     is

     likely

     that

     we

     will

     see

     more

     widespread

     adoption

     of

     AI

     in

    



```python
llm.shutdown()
```
