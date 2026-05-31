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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:12,  3.83it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  6.14it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  8.84it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  8.84it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  8.84it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:04,  8.84it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:04,  8.84it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 12.90it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 12.90it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 12.90it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 12.90it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 12.90it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 17.30it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 23.04it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 23.04it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 23.04it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 23.04it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 23.04it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 23.04it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 42.68it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 42.68it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 42.68it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 42.68it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 42.68it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:06<00:00, 42.68it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:06<00:00, 42.68it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:06<00:00, 42.68it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 48.01it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 48.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.85 GB):   2%|▏         | 1/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.82 GB):   2%|▏         | 1/58 [00:00<00:07,  7.36it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.82 GB):   3%|▎         | 2/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.81 GB):   3%|▎         | 2/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.81 GB):   5%|▌         | 3/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.81 GB):   5%|▌         | 3/58 [00:00<00:06,  7.97it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.81 GB):   7%|▋         | 4/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.81 GB):   7%|▋         | 4/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.81 GB):   9%|▊         | 5/58 [00:00<00:06,  8.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.81 GB):   9%|▊         | 5/58 [00:00<00:06,  8.25it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.81 GB):  10%|█         | 6/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.80 GB):  10%|█         | 6/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.80 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.79 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.79 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.79 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.79 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.79 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.99it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.79 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.78 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.78 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.78 GB):  21%|██        | 12/58 [00:01<00:04,  9.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.78 GB):  21%|██        | 12/58 [00:01<00:04,  9.72it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.78 GB):  21%|██        | 12/58 [00:01<00:04,  9.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.78 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.77 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.77 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.17it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.77 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  31%|███       | 18/58 [00:01<00:03, 12.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  31%|███       | 18/58 [00:01<00:03, 12.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  31%|███       | 18/58 [00:01<00:03, 12.27it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 13.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  34%|███▍      | 20/58 [00:01<00:02, 13.66it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 13.66it/s] Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.93it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.93it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.93it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 16.19it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 16.19it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:02<00:02, 16.19it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:02<00:02, 16.19it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:02<00:02, 16.19it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.31it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.31it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.31it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.31it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 20.24it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 20.24it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 20.24it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 20.24it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.71it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.71it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.71it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 20.71it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.13it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.13it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.13it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 21.13it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.50it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.50it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.50it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  69%|██████▉   | 40/58 [00:02<00:00, 21.50it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 21.75it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 21.75it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.75it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  74%|███████▍  | 43/58 [00:03<00:00, 21.75it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 21.86it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 21.86it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 21.86it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  79%|███████▉  | 46/58 [00:03<00:00, 21.86it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 21.39it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  84%|████████▍ | 49/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.72it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.72it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.72it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:03<00:00, 21.72it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.16it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.16it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.16it/s] Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  95%|█████████▍| 55/58 [00:03<00:00, 22.16it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 22.65it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 15.95it/s]


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
    Generated text:  Leo, a college student at Stonehill College. I am currently working on my Ph.D. in computational neuroscience, and have been working on a project called "Systems Biophysical Modeling of Neural Circuit Dynamics" (SBMNC). The project seeks to understand the complex interactions between neurons and neurotransmitters to better comprehend the mechanisms behind neural signals and their neural signatures. I am extremely interested in studying the impact of drugs on neural activity and synaptic plasticity. Can you provide me with a brief overview of the main concepts and techniques that are being used in the project? Additionally, could you provide me with a comprehensive overview of the various studies that have
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to determine the median salary for all employees in the country. The salaries of the employees are as follows: 
    
    - Salaries: $70,000, $80,000, $90,000, $100,000, $110,000, $120,000, $130,000, $140,000, $150,000, $160,000, $170,000, $180,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. How to prove this?
    
    To prove that the capital of France is Paris, we need to demonstrate that Paris is the only capital city in France. Let's analyze the given information step by step.
    
    1. **Definition of Capital City**: A capital city is a city whose government is responsible for the day-to-day administration of the state. It is the seat of the government and the main administrative center of a nation.
    
    2. **History of Capital Cities**: The term "capital city" has a long and complex history in European history. It is not always the same city, but it is generally the main city or the city most
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be reshaped by the rapid advancement of quantum computing. This type of computing relies on quantum mechanics, which involves particles with both spin up and down. In other words, particles that are entangled and behave as one entity in a single quantum state. Quantum computers are capable of performing certain complex calculations much faster than traditional computers, which has significant implications for the development of AI.
    
    The development of quantum computers has been a long-standing goal of researchers in the field of quantum information science. In the last few years, significant progress has been made, and companies like IBM and Google have developed advanced quantum computing hardware. This has opened up new


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [field]. I'm a [type of person] who is [positive or negative] about [job or hobby]. I'm [age] years old and [gender] [race]. I'm [interests or hobbies] and [personal traits]. I'm [positive or negative] about [job or hobby]. I'm [age] years old and [gender] [race]. I'm [interests or hobbies] and [personal traits]. I'm [positive or negative] about [job or hobby]. I'm [age] years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also known for its rich history, art, and culture, and is a major tourist destination. Paris is a cultural and intellectual center of Europe and a major hub for international business and diplomacy. The city is home to many famous museums, theaters, and art galleries, and is a major center for the arts and entertainment industry. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in healthcare.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection, and investment decision-making. As AI technology continues to improve, we can
    


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
    Generated text:  [Name], and I'm a [insert profession or title] with a background in [insert relevant skills or experience]. I'm an [insert age] year old, [insert occupation] with [insert experience level] years of experience. I'm always looking to learn and grow, and am always eager to provide new insights and new perspectives. I've been [insert relevant experience level] with [insert relevant skills] and have always been passionate about [insert relevant interest or hobby]. I'm always looking for new challenges and opportunities to grow and learn, and I'm excited to continue my journey of knowledge and discovery. Thank you for asking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country, with a population of over 2. 5 million. The city is home to numerous iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is known for its rich history, cultural attractions, and vibrant nightlife, making it a popular destination for tourists and locals alike. It is considered one of the world's most beautiful cities and is recognized as a UNESCO World Heritage site. The city has a diverse population, consisting of people from all over the world, and has played an important role in the history and development of France and Europe
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and diverse, with potential applications and impacts in various sectors. Here are some possible trends that are likely to occur:
    
    1. Integration with human work: AI will continue to become more integrated with human work, improving efficiency and effectiveness. AI will be used to automate routine tasks, provide personalized assistance and recommendations, and support decision-making.
    
    2. Increased transparency and accountability: As AI becomes more sophisticated, there will be an increased need for transparency and accountability. The more complex and unpredictable the AI algorithms, the more important it will be to ensure that they are being used in a safe and ethical manner.
    
    3. Emphasis on fairness and bias


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

     fictional

     name

    ].

     I

     am

     a

     [

    insert

     fictional

     profession

     or

     title

    ],

     and

     I

     have

     always

     been

     fascinated

     by

     the

     world

     of

     technology

     and

     innovation

    .

     I

     have

     a

     keen

     interest

     in

     how

     technology

     can

     be

     used

     to

     solve

     real

    -world

     problems

     and

     make

     the

     world

     a

     better

     place

    .

     I

     am

     always

     learning

     new

     technologies

     and

     exploring

     new

     ways

     of

     using

     them

     to

     create

     positive

     change

    .

     I

     enjoy

     working

     with

     a

     team

     of

     people

     and

     being

     able

     to

     collaborate

     on

     projects

     that

     will

     benefit

     others

    .

     I

     am

     a

     professional

     at

     heart

     and

     I

     thrive

     on

     the

     challenge

     of

     pushing

     the

     boundaries

     of

     what

     is

     possible

    .

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

     stay

     current

     with

     the

     latest

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

    .

     It

     is

     located

     in

     the

     center

     of

     the

     country

     in

     the

     Lo

    ire

     Valley

     and

     covers

     an

     area

     of

     

    9

    4

    8

    .

    2

    4

     square

     kilometres

    .

     It

     is

     the

     capital

     of

     France

     and

     the

     second

     largest

     city

     in

     the

     country

    ,

     after

     Paris

    ,

     and

     is

     also

     the

     largest

     city

     in

     metropolitan

     France

    .

     Paris

     is

     the

     most

     visited

     city

     in

     France

     and

     is

     home

     to

     many

     attractions

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

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

     the

     Palace

     of

     Vers

    ailles

    .

     It

     also

     has

     a

     rich

     cultural

     heritage

     and

     is

     known

     for

     its

     fashion

     and

     food

     scene

    .

     Despite

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     developments

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     increasingly

     precise

     and

     accurate

    .

     This

     will

     make

     it

     easier

     to

     automate

     complex

     tasks

     and

     improve

     the

     efficiency

     of

     our

     daily

     lives

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     allow

     us

     to

     tailor

     our

     experiences

     to

     our

     specific

     needs

     and

     preferences

    .

     We

     will

     be

     able

     to

     access

     personalized

     recommendations

     and

     products

     that

     are

     tailored

     to

     our

     individual

     tastes

    .
    


    3

    .

     Autonomous

     and

     self

    -driving

     vehicles

    :

     AI

     is

     already

     playing

     a

     significant

     role

     in

     autonomous

     and

     self

    -driving

     vehicles

    .

     We

     expect

     this

     trend

     to

     continue

     as

     technology

     advances

     and

     becomes

     more

     widespread

    



```python
llm.shutdown()
```
