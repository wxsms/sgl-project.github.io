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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.11s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.11s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]

    Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.41it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.41it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.41it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.41it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.41it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.64it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.64it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.64it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.64it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.64it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.41it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.41it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.41it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 12.41it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 12.41it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 16.17it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 16.17it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 16.17it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 16.17it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.17it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 23.62it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 23.62it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 23.62it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 23.62it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 23.62it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 23.62it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 27.98it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 27.98it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 27.98it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 27.98it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 27.98it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 27.98it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 32.01it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 32.01it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 32.01it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 32.01it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 32.01it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 32.01it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 34.34it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 34.34it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 34.34it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 34.34it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 34.34it/s]

    Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 34.34it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 36.80it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 40.89it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 40.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.30 GB):   2%|▏         | 1/58 [00:00<00:06,  8.95it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.27 GB):   2%|▏         | 1/58 [00:00<00:06,  8.95it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=40.27 GB):   3%|▎         | 2/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.26 GB):   3%|▎         | 2/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.26 GB):   3%|▎         | 2/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.26 GB):   7%|▋         | 4/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.26 GB):   7%|▋         | 4/58 [00:00<00:04, 10.92it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=40.25 GB):   7%|▋         | 4/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.25 GB):  10%|█         | 6/58 [00:00<00:04, 12.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.24 GB):  10%|█         | 6/58 [00:00<00:04, 12.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.24 GB):  10%|█         | 6/58 [00:00<00:04, 12.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.24 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.24 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.29it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=40.20 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.20 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.20 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.20 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.18 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.13it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=40.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=960 avail_mem=40.17 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s] Capturing num tokens (num_tokens=896 avail_mem=40.16 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=832 avail_mem=40.16 GB):  36%|███▌      | 21/58 [00:01<00:01, 32.27it/s]Capturing num tokens (num_tokens=768 avail_mem=40.16 GB):  36%|███▌      | 21/58 [00:01<00:01, 32.27it/s]Capturing num tokens (num_tokens=704 avail_mem=40.15 GB):  36%|███▌      | 21/58 [00:01<00:01, 32.27it/s]

    Capturing num tokens (num_tokens=704 avail_mem=40.15 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=640 avail_mem=39.02 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=576 avail_mem=39.02 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.51it/s]

    Capturing num tokens (num_tokens=512 avail_mem=39.00 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=480 avail_mem=40.12 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=480 avail_mem=40.12 GB):  52%|█████▏    | 30/58 [00:01<00:01, 21.17it/s]Capturing num tokens (num_tokens=448 avail_mem=40.12 GB):  52%|█████▏    | 30/58 [00:01<00:01, 21.17it/s]Capturing num tokens (num_tokens=416 avail_mem=39.13 GB):  52%|█████▏    | 30/58 [00:01<00:01, 21.17it/s]

    Capturing num tokens (num_tokens=384 avail_mem=39.13 GB):  52%|█████▏    | 30/58 [00:01<00:01, 21.17it/s]Capturing num tokens (num_tokens=384 avail_mem=39.13 GB):  57%|█████▋    | 33/58 [00:01<00:01, 17.96it/s]Capturing num tokens (num_tokens=352 avail_mem=39.12 GB):  57%|█████▋    | 33/58 [00:01<00:01, 17.96it/s]Capturing num tokens (num_tokens=320 avail_mem=40.10 GB):  57%|█████▋    | 33/58 [00:01<00:01, 17.96it/s]

    Capturing num tokens (num_tokens=288 avail_mem=39.18 GB):  57%|█████▋    | 33/58 [00:01<00:01, 17.96it/s]Capturing num tokens (num_tokens=288 avail_mem=39.18 GB):  62%|██████▏   | 36/58 [00:01<00:01, 16.71it/s]Capturing num tokens (num_tokens=256 avail_mem=39.17 GB):  62%|██████▏   | 36/58 [00:01<00:01, 16.71it/s]Capturing num tokens (num_tokens=240 avail_mem=39.17 GB):  62%|██████▏   | 36/58 [00:02<00:01, 16.71it/s]

    Capturing num tokens (num_tokens=240 avail_mem=39.17 GB):  66%|██████▌   | 38/58 [00:02<00:01, 15.76it/s]Capturing num tokens (num_tokens=224 avail_mem=40.09 GB):  66%|██████▌   | 38/58 [00:02<00:01, 15.76it/s]Capturing num tokens (num_tokens=208 avail_mem=39.23 GB):  66%|██████▌   | 38/58 [00:02<00:01, 15.76it/s]Capturing num tokens (num_tokens=208 avail_mem=39.23 GB):  69%|██████▉   | 40/58 [00:02<00:01, 15.23it/s]Capturing num tokens (num_tokens=192 avail_mem=39.23 GB):  69%|██████▉   | 40/58 [00:02<00:01, 15.23it/s]

    Capturing num tokens (num_tokens=176 avail_mem=40.08 GB):  69%|██████▉   | 40/58 [00:02<00:01, 15.23it/s]Capturing num tokens (num_tokens=176 avail_mem=40.08 GB):  72%|███████▏  | 42/58 [00:02<00:01, 15.01it/s]Capturing num tokens (num_tokens=160 avail_mem=40.08 GB):  72%|███████▏  | 42/58 [00:02<00:01, 15.01it/s]Capturing num tokens (num_tokens=144 avail_mem=39.29 GB):  72%|███████▏  | 42/58 [00:02<00:01, 15.01it/s]

    Capturing num tokens (num_tokens=144 avail_mem=39.29 GB):  76%|███████▌  | 44/58 [00:02<00:00, 14.27it/s]Capturing num tokens (num_tokens=128 avail_mem=39.29 GB):  76%|███████▌  | 44/58 [00:02<00:00, 14.27it/s]Capturing num tokens (num_tokens=112 avail_mem=40.07 GB):  76%|███████▌  | 44/58 [00:02<00:00, 14.27it/s]Capturing num tokens (num_tokens=112 avail_mem=40.07 GB):  79%|███████▉  | 46/58 [00:02<00:00, 14.98it/s]Capturing num tokens (num_tokens=96 avail_mem=39.34 GB):  79%|███████▉  | 46/58 [00:02<00:00, 14.98it/s] Capturing num tokens (num_tokens=80 avail_mem=39.34 GB):  79%|███████▉  | 46/58 [00:02<00:00, 14.98it/s]

    Capturing num tokens (num_tokens=80 avail_mem=39.34 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.52it/s]Capturing num tokens (num_tokens=64 avail_mem=40.06 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.52it/s]Capturing num tokens (num_tokens=48 avail_mem=40.05 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.52it/s]Capturing num tokens (num_tokens=48 avail_mem=40.05 GB):  86%|████████▌ | 50/58 [00:02<00:00, 15.35it/s]Capturing num tokens (num_tokens=32 avail_mem=39.39 GB):  86%|████████▌ | 50/58 [00:02<00:00, 15.35it/s]Capturing num tokens (num_tokens=28 avail_mem=39.39 GB):  86%|████████▌ | 50/58 [00:02<00:00, 15.35it/s]

    Capturing num tokens (num_tokens=28 avail_mem=39.39 GB):  90%|████████▉ | 52/58 [00:03<00:00, 15.44it/s]Capturing num tokens (num_tokens=24 avail_mem=40.04 GB):  90%|████████▉ | 52/58 [00:03<00:00, 15.44it/s]Capturing num tokens (num_tokens=20 avail_mem=39.45 GB):  90%|████████▉ | 52/58 [00:03<00:00, 15.44it/s]Capturing num tokens (num_tokens=20 avail_mem=39.45 GB):  93%|█████████▎| 54/58 [00:03<00:00, 15.19it/s]Capturing num tokens (num_tokens=16 avail_mem=40.04 GB):  93%|█████████▎| 54/58 [00:03<00:00, 15.19it/s]Capturing num tokens (num_tokens=12 avail_mem=40.03 GB):  93%|█████████▎| 54/58 [00:03<00:00, 15.19it/s]

    Capturing num tokens (num_tokens=12 avail_mem=40.03 GB):  97%|█████████▋| 56/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=8 avail_mem=39.51 GB):  97%|█████████▋| 56/58 [00:03<00:00, 16.09it/s] Capturing num tokens (num_tokens=4 avail_mem=40.03 GB):  97%|█████████▋| 56/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=4 avail_mem=40.03 GB): 100%|██████████| 58/58 [00:03<00:00, 16.97it/s]Capturing num tokens (num_tokens=4 avail_mem=40.03 GB): 100%|██████████| 58/58 [00:03<00:00, 17.20it/s]


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
    Generated text:  Terri, a professor in the Department of Computer Science at the University of California, Riverside. In this role, I am a Computer Science and Engineering faculty member, teaching and researching a wide range of computer science topics in the areas of algorithms, databases, multimedia computing, and computer security.
    
    I am a member of the Computer Science Department’s Cognitive Systems Lab, which is a research group that focuses on the cognitive and neural basis of human and machine learning.
    
    My areas of interest include:
    
    • Machine learning and neuroscience
    
    • Human-computer interaction
    
    • Cognitive neurosciences
    
    • Neural networks and deep learning
    
    • Brain-com
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small town in the Midwest. When the president arrives in the town, he notices that the population is 30,000. During his visit, the president has a total of 400 people. How many more people are there in the town compared to the president in the town?
    To determine how many more people are in the town compared to the president, we need to follow these steps:
    
    1. Identify the population of the town.
    2. Identify the population of the president.
    3. Subtract the president's population from the town's population.
    
    The population of the town is 30,00
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    
    B. Rome
    
    C. London
    
    D. Berlin
    
    The correct answer is A. Paris. Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The other options listed are not capitals of France: Rome (Italy) is the capital of Italy, and London (UK) is the capital of the United Kingdom. Berlin, known for its West Germanic architecture and cultural diversity, is not a capital city.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of programmers. Here are 7 reasons why they must understand it.
    
    1. The future of AI is in the hands of programmers.
      2. The future of AI will be as complex as life, and will require a vast array of skills and techniques.
      3. In the future, AI will become more sophisticated, increasingly capable of predicting behavior.
      4. The future of AI will be dependent on the ability to model, understand, and manipulate the complex world of the digital world.
      5. AI will become the key to revolutionizing our lives and future work.
      6. If


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new opportunities to grow and learn, and I'm eager to contribute to [insert a short description of your company or field of interest here]. Thank you for having me. [Name] [Company Name] [Job Title] [Company Website] [LinkedIn Profile] [Twitter Profile]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its cuisine, fashion, and art scene, making it a popular tourist destination. The city is also home to many cultural institutions, including the Louvre Museum, the Musée d'Orsay, and the Musée Rodin. Overall, Paris is a vibrant and exciting city that is a must-visit for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration of AI into various industries: AI will continue to be integrated into various industries, including healthcare, finance, transportation, and manufacturing. This integration will lead to more efficient and effective use of resources, as well as better decision-making and problem-solving.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into
    


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
    Generated text:  [Name] and I'm a [age] year old [occupation] [job title]. I've always been fascinated by [your favorite hobby, hobby that you've been passionate about for a long time] and I've always wanted to be [the type of person you aspire to be, like a [noun]], [if possible, include a profession or career you aspire to be] or something that relates to your interest. How can you tell me what your hobby is? If you can't tell me, then ask me directly. However, for my age, I'd prefer not to mention it.
    
    But don't worry, you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its historical architecture, vibrant culture, and exquisite cuisine. The city is home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral, among other landmarks. Paris is also famous for its fashion, art, and food scene, making it an important cultural and economic hub in France. The city is a UNESCO World Heritage site and is often referred to as the "City of Light" and "The City of Light." Paris is considered the capital of the world and is a major transportation hub for Europe. Its status as a global cultural and political center is evident in its diverse range
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and future-oriented. Here are some possible trends that we can expect:
    
    1. Increased integration with human intelligence: AI will become more integrated with our human intelligence. This will lead to the development of AI-powered robots that can operate effectively with humans, improving our quality of life.
    
    2. Enhanced decision-making: AI will become more adept at making decisions based on data, leading to more informed and ethical decision-making.
    
    3. Improved communication: AI will be able to understand and respond to human emotions, leading to more empathetic and effective communication.
    
    4. Enhanced creativity: AI will be able to generate new ideas and patterns of creativity, leading


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

    ].

     I

     am

     a

     talented

     writer

     and

     creator

    ,

     best

     known

     for

     my

     novels

     [

    Book

     Title

     

    1

    ]

     and

     [

    Book

     Title

     

    2

    ].

     I

    'm

     a

     very

     passionate

     writer

    ,

     always

     eager

     to

     learn

     new

     things

     and

     improve

     my

     craft

    .

     I

     also

     love

     to

     share

     my

     stories

     with

     people

    ,

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     connect

     with

     others

    .

     I

    'm

     excited

     to

     meet

     you

     and

     let

    's

     see

     where

     our

     writing

     can

     take

     us

    !

     Let

    's

     do

     this

    !

     [

    Your

     Name

    ]

     [

    Your

     Signature

    ]

     [

    Your

     Contact

     Information

    ]


    Write

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    .

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     de

     Paris

    "

     and

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     architecture

     and

     historical

     significance

    .

     
    


    The

     city

     is

     renowned

     for

     its

     iconic

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

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

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     also

     famous

     for

     its

     culinary

     scene

    ,

     with

     its

     famous

     French

     cuisine

     and

     a

     wide

     array

     of

     restaurants

     catering

     to

     different

     tastes

     and

     dietary

     preferences

    .

     The

     city

     is

     home

     to

     several

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     d

    '

    Art

     Moder

    ne

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     emerge

    :
    


    1

    .

     Increased

     AI

     integration

     with

     other

     technologies

    :

     AI

     will

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     blockchain

    ,

     IoT

    ,

     and

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     to

     create

     more

     complex

     and

     sophisticated

     AI

     systems

    .
    


    2

    .

     AI

     will

     become

     more

     ethical

     and

     transparent

    :

     As

     more

     AI

     systems

     become

     autonomous

     and

     operate

     in

     real

    -time

    ,

     there

     will

     be

     an

     increasing

     demand

     for

     ethical

     and

     transparent

     AI

     systems

     that

     can

     operate

     without

     bias

     or

     favor

    it

    ism

    .
    


    3

    .

     AI

     will

     become

     more

     personalized

    :

     AI

     will

     be

     able

     to

     learn

     from

     large

     amounts

     of

     data

     to

     provide

     more

     accurate

     and

     personalized

     recommendations

     and

     solutions

    .
    


    4

    



```python
llm.shutdown()
```
