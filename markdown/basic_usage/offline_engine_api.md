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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  7.77it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  7.77it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  7.77it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  7.77it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  7.77it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 11.68it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 11.68it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 11.68it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 11.68it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 11.68it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 15.73it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 15.73it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.73it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 21.20it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 21.20it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 21.20it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 21.20it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 21.20it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 21.20it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:00, 26.63it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 32.76it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 38.57it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 43.08it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 49.44it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 49.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.81 GB):   2%|▏         | 1/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=7680 avail_mem=48.77 GB):   2%|▏         | 1/58 [00:00<00:07,  7.35it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=48.77 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.77 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.77 GB):   5%|▌         | 3/58 [00:00<00:07,  7.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=48.77 GB):   5%|▌         | 3/58 [00:00<00:07,  7.52it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=48.77 GB):   7%|▋         | 4/58 [00:00<00:07,  7.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=48.77 GB):   7%|▋         | 4/58 [00:00<00:07,  7.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=48.77 GB):   9%|▊         | 5/58 [00:00<00:06,  7.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=48.76 GB):   9%|▊         | 5/58 [00:00<00:06,  7.92it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=48.76 GB):  10%|█         | 6/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=48.75 GB):  10%|█         | 6/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=48.75 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=48.75 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.62it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=48.75 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=48.75 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=48.75 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=48.74 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.75it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=48.74 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.74 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=48.73 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=48.73 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=48.73 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.41it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=48.73 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=48.73 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=48.73 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=48.72 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=48.72 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=48.72 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.56it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=48.72 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=48.69 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=48.67 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=48.67 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=960 avail_mem=48.69 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.57it/s] Capturing num tokens (num_tokens=896 avail_mem=48.69 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.57it/s]

    Capturing num tokens (num_tokens=832 avail_mem=48.68 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=832 avail_mem=48.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.23it/s]Capturing num tokens (num_tokens=768 avail_mem=48.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.23it/s]Capturing num tokens (num_tokens=704 avail_mem=48.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.23it/s]Capturing num tokens (num_tokens=640 avail_mem=48.67 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.23it/s]Capturing num tokens (num_tokens=640 avail_mem=48.67 GB):  47%|████▋     | 27/58 [00:02<00:01, 19.91it/s]Capturing num tokens (num_tokens=576 avail_mem=48.67 GB):  47%|████▋     | 27/58 [00:02<00:01, 19.91it/s]

    Capturing num tokens (num_tokens=512 avail_mem=48.66 GB):  47%|████▋     | 27/58 [00:02<00:01, 19.91it/s]Capturing num tokens (num_tokens=480 avail_mem=48.67 GB):  47%|████▋     | 27/58 [00:02<00:01, 19.91it/s]Capturing num tokens (num_tokens=480 avail_mem=48.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.26it/s]Capturing num tokens (num_tokens=448 avail_mem=48.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.26it/s]Capturing num tokens (num_tokens=416 avail_mem=48.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.26it/s]Capturing num tokens (num_tokens=384 avail_mem=48.67 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.26it/s]

    Capturing num tokens (num_tokens=384 avail_mem=48.67 GB):  57%|█████▋    | 33/58 [00:02<00:01, 20.83it/s]Capturing num tokens (num_tokens=352 avail_mem=48.66 GB):  57%|█████▋    | 33/58 [00:02<00:01, 20.83it/s]Capturing num tokens (num_tokens=320 avail_mem=48.66 GB):  57%|█████▋    | 33/58 [00:02<00:01, 20.83it/s]Capturing num tokens (num_tokens=288 avail_mem=48.65 GB):  57%|█████▋    | 33/58 [00:02<00:01, 20.83it/s]Capturing num tokens (num_tokens=288 avail_mem=48.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.19it/s]Capturing num tokens (num_tokens=256 avail_mem=48.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.19it/s]Capturing num tokens (num_tokens=240 avail_mem=48.65 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.19it/s]

    Capturing num tokens (num_tokens=224 avail_mem=48.64 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.19it/s]Capturing num tokens (num_tokens=224 avail_mem=48.64 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.93it/s]Capturing num tokens (num_tokens=208 avail_mem=48.64 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.93it/s]Capturing num tokens (num_tokens=192 avail_mem=48.64 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.93it/s]Capturing num tokens (num_tokens=176 avail_mem=48.64 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.93it/s]Capturing num tokens (num_tokens=176 avail_mem=48.64 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.29it/s]Capturing num tokens (num_tokens=160 avail_mem=48.63 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.29it/s]

    Capturing num tokens (num_tokens=144 avail_mem=48.63 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.29it/s]Capturing num tokens (num_tokens=128 avail_mem=48.63 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.29it/s]Capturing num tokens (num_tokens=128 avail_mem=48.63 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.95it/s]Capturing num tokens (num_tokens=112 avail_mem=48.63 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.95it/s]Capturing num tokens (num_tokens=96 avail_mem=48.62 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.95it/s] Capturing num tokens (num_tokens=80 avail_mem=48.62 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.95it/s]

    Capturing num tokens (num_tokens=80 avail_mem=48.62 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.20it/s]Capturing num tokens (num_tokens=64 avail_mem=48.61 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.20it/s]Capturing num tokens (num_tokens=48 avail_mem=48.61 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.20it/s]Capturing num tokens (num_tokens=32 avail_mem=48.61 GB):  83%|████████▎ | 48/58 [00:03<00:00, 23.20it/s]Capturing num tokens (num_tokens=32 avail_mem=48.61 GB):  88%|████████▊ | 51/58 [00:03<00:00, 23.09it/s]Capturing num tokens (num_tokens=28 avail_mem=48.60 GB):  88%|████████▊ | 51/58 [00:03<00:00, 23.09it/s]Capturing num tokens (num_tokens=24 avail_mem=48.60 GB):  88%|████████▊ | 51/58 [00:03<00:00, 23.09it/s]

    Capturing num tokens (num_tokens=20 avail_mem=48.60 GB):  88%|████████▊ | 51/58 [00:03<00:00, 23.09it/s]Capturing num tokens (num_tokens=20 avail_mem=48.60 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.17it/s]Capturing num tokens (num_tokens=16 avail_mem=48.60 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.17it/s]Capturing num tokens (num_tokens=12 avail_mem=48.59 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.17it/s]Capturing num tokens (num_tokens=8 avail_mem=48.59 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.17it/s] Capturing num tokens (num_tokens=8 avail_mem=48.59 GB):  98%|█████████▊| 57/58 [00:03<00:00, 23.26it/s]Capturing num tokens (num_tokens=4 avail_mem=48.58 GB):  98%|█████████▊| 57/58 [00:03<00:00, 23.26it/s]

    Capturing num tokens (num_tokens=4 avail_mem=48.58 GB): 100%|██████████| 58/58 [00:03<00:00, 17.17it/s]


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
    Generated text:  David Guan and I'm a recent graduate of the MA program in Translative Studies at the University of California, Los Angeles. My field of expertise is narrative theory, and in the recent past, I have been exploring the relationship between autobiography and autobiography by watching a series of film clips that I will call "Autobiographical Film Clips."
    These are slow, often unsettling, self-reflexive film clips that I created using techniques such as slow motion, audio isolation, and montage.
    The clips that I made are about characters who can relate to something that is written in the news (like a murder in the city of New York).
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to reduce global warming by implementing a policy that requires all cars to have an emissions cap. The policy is expected to result in a 20% reduction in the amount of greenhouse gases in the atmosphere. If the emissions cap is set at $200 per car, and the current average emissions per car is $100, how much will the reduction in emissions due to the cap be?
    To determine the reduction in emissions due to the cap, we need to calculate the total reduction in emissions from the cap. The reduction in emissions is the product of the emissions cap per car and the reduction in emissions per car.
    
    First
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. As one of the most famous cities of the world, Paris has a long and rich history, and is an important cultural, economic and political centre. It has a large population of around 2. 2 million. The capital is also home to some of the world's best museums and galleries. Visitors from all over the world come to Paris to see its beautiful buildings and see the famous works of art, including the Louvre and the Eiffel Tower. Paris is also home to many important and important universities. In fact, more than 120 universities are located within the city. The city is also famous for
    ===============================
    Prompt: The future of AI is
    Generated text:  complex and challenging, but it’s possible. Here’s what you need to know.
    The “big data” that has dominated the focus of AI research and development is getting more complex. While many companies are hoping to leverage the insights from big data, they are also recognizing that the data itself is difficult to collect and analyze.
    Recent developments in deep learning and other AI models have shown that even in the presence of complex data, the task of classification can be solved. This means that more complex models can be created to do more useful things with less data, making the field more accessible. In 2017, Google published the world


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


    Generated text:  [Name] and I'm a [occupation] with [number of years] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [character's age] years old and I'm [character's gender]. I'm [character's profession] and I'm [character's role in the company]. I'm [character's personality type] and I'm [character's personality trait]. I'm [character's favorite hobby] and I'm [character's favorite food]. I'm [character's favorite place to eat]. I'm [character's favorite book or movie]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political capital. Paris is a bustling metropolis with a rich history and diverse culture, making it a popular tourist destination. It is also known for its fashion industry and its role in the French economy. The city is home to many famous landmarks and attractions, including the Champs-Élysées, the Louvre, and the Eiffel Tower. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we are likely to see more automation and artificial intelligence in our daily lives. This could include the automation of tasks such as customer service, manufacturing, and transportation, as well as the development of AI-powered robots and drones.
    
    2. Improved privacy and security: As AI becomes more advanced, we are likely to see more privacy and security concerns. This could include the development
    


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
    Generated text:  [Your Name], and I specialize in helping people in need. I specialize in supporting homeless individuals and families in my community. I'm currently volunteering at [Name of organization or program] and I'm always on the lookout for new opportunities to help people in need. I'm looking for someone who is also passionate about environmental sustainability and who can work towards creating a more sustainable future for our planet. How can I reach out to you if you're interested in joining our team?
    Hello, my name is [Your Name] and I specialize in helping people in need. I specialize in supporting homeless individuals and families in my community. I'm currently
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France by population, with an estimated 2.1 million inhabitants. It is the country's political, economic, cultural, and historical center, and home to the nation's rich heritage and modern culture. The city is also a major transportation hub and seat of government, as well as a world-renowned destination for food, fashion, and art.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and we can expect many exciting trends to emerge in the years to come. Here are some potential future trends in AI:
    
    1. Increased integration of AI with other technologies: AI is increasingly being integrated with other technologies such as blockchain, IoT, and machine learning. These technologies will enable us to build more sophisticated and secure AI systems that can handle complex tasks.
    
    2. Enhanced privacy and security: With the increasing use of AI, there will be an increased need for privacy and security. We will see more stringent regulations and measures to protect user data and prevent misuse of AI systems.
    
    3. Improved natural language processing: Natural language processing


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

    Character

     Name

    ]

     and

     I

     am

     an

     [

    age

    ]

     year

     old

     [

    Occup

    ation

    ],

     a

     [

    noun

    ]

     in

     the

     [

    industry

    ]

     industry

    .

     I

     have

     always

     been

     [

    some

     characteristic

    ],

     [

    reason

     why

    ].

     If

     you

    'd

     like

     to

     meet

     me

    ,

     feel

     free

     to

     introduce

     yourself

    .

     (

    give

     example

     of

     how

     to

     start

    )

     (

    go

     on

     to

     describe

     your

     personality

    ,

     work

     ethic

    ,

     hobbies

    ,

     etc

    .)

     (

    stop

    )

     That

    's

     great

    !

     Tell

     me

     more

    .

     (

    give

     example

     of

     how

     to

     end

    )

     Good

    .

     Now

     that

     I

     have

     a

     brief

     introduction

    ,

     let

    's

     dive

     into

     the

     details

     of

     your

     character

    .

     Can

     you

     tell

     me

     about

     your

     daily

     routine

    ?

     How

     do

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     Lou

    v

    ain

    ,

     which

     is

     located

     on

     the

     Se

    ine

     River

     in

     the

     northeastern

     region

     of

     the

     country

    .

     
    


    The

     city

     is

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    ,

     hosting

     the

     country

    's

     most

     important

     museums

    ,

     government

     offices

    ,

     and

     government

     buildings

    ,

     as

     well

     as

     being

     a

     major

     hub

     for

     many

     of

     France

    's

     major

     industries

    .

     It

    's

     also

     home

     to

     numerous

     art

     galleries

    ,

     theaters

    ,

     opera

     houses

    ,

     and

     theaters

    .

     
    


    The

     French

     capital

     has

     a

     diverse

     population

     that

     speaks

     many

     languages

    ,

     and

     is

     known

     for

     its

     romantic

     and

     historical

     atmosphere

    .

     It

    's

     a

     city

     of

     contrasts

    ,

     where

     traditional

     French

     culture

     meets

     the

     modern

     world

    ,

     and

     where

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     rapidly

     changing

    ,

     with

     many

     potential

     trends

     emerging

    .

     Here

     are

     some

     of

     the

     most

     promising

     and

     promising

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     AI

    :

     As

     AI

     becomes

     more

     integrated

     with

     human

     AI

    ,

     we

    'll

     see

     more

     complex

     and

     nuanced

     interactions

     between

     humans

     and

     machines

    .

     This

     could

     lead

     to

     more

     empath

    etic

     and

     context

    -aware

     AI

     that

     understands

     and

     responds

     to

     human

     emotions

     and

     experiences

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     have

     the

     potential

     to

     revolution

    ize

     transportation

     by

     reducing

     accidents

     and

     speeding

     up

     delivery

     services

    .

     However

    ,

     there

     are

     still

     technical

     challenges

     to

     overcome

    ,

     such

     as

     safety

     regulations

     and

     the

     need

     for

     higher

     levels

     of

     autonomy

    .
    


    3

    .

     Increased

     ethical

     considerations

    :

     As

    



```python
llm.shutdown()
```
