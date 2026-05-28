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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:00,  1.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.76it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:17,  2.86it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.43it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.43it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.43it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.43it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.43it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.29it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.29it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.29it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.29it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 15.17it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:01, 20.77it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:01, 20.77it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.77it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.77it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.77it/s]

    Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 20.77it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 25.55it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]

    Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 31.90it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 37.37it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 41.30it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 45.44it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 45.44it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 45.44it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 45.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.41it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.14it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:08,  6.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:08,  6.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   9%|▊         | 5/58 [00:00<00:07,  6.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):   9%|▊         | 5/58 [00:00<00:07,  6.65it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):  10%|█         | 6/58 [00:00<00:07,  7.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  10%|█         | 6/58 [00:00<00:07,  7.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.29it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.20it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.39 GB):  21%|██        | 12/58 [00:01<00:04,  9.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.38 GB):  21%|██        | 12/58 [00:01<00:04,  9.30it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.37 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.36 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.87it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=53.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.34 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.34 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.35it/s]Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.35it/s] Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.35it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.35 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.35it/s]Capturing num tokens (num_tokens=832 avail_mem=53.35 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.62it/s]Capturing num tokens (num_tokens=768 avail_mem=53.35 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.62it/s]Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.62it/s]

    Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.20it/s]Capturing num tokens (num_tokens=640 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.20it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.20it/s]Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.20it/s]Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  50%|█████     | 29/58 [00:02<00:01, 16.85it/s]Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  50%|█████     | 29/58 [00:02<00:01, 16.85it/s]Capturing num tokens (num_tokens=448 avail_mem=52.83 GB):  50%|█████     | 29/58 [00:02<00:01, 16.85it/s]

    Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  50%|█████     | 29/58 [00:02<00:01, 16.85it/s]Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.21it/s]Capturing num tokens (num_tokens=384 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.21it/s]Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.21it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.21it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:02<00:01, 19.17it/s]Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:02<00:01, 19.17it/s]

    Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:02<00:01, 19.17it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  60%|██████    | 35/58 [00:02<00:01, 19.17it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:02<00:00, 20.40it/s]Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:02<00:00, 20.40it/s]Capturing num tokens (num_tokens=208 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:02<00:00, 20.40it/s]Capturing num tokens (num_tokens=192 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:02<00:00, 20.40it/s]

    Capturing num tokens (num_tokens=192 avail_mem=52.80 GB):  71%|███████   | 41/58 [00:03<00:00, 21.01it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:00, 21.01it/s]Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:00, 21.01it/s]Capturing num tokens (num_tokens=144 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:00, 21.01it/s]Capturing num tokens (num_tokens=144 avail_mem=52.79 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.77it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.77it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.77it/s]

    Capturing num tokens (num_tokens=96 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.77it/s] Capturing num tokens (num_tokens=96 avail_mem=52.78 GB):  81%|████████  | 47/58 [00:03<00:00, 22.24it/s]Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  81%|████████  | 47/58 [00:03<00:00, 22.24it/s]Capturing num tokens (num_tokens=64 avail_mem=52.77 GB):  81%|████████  | 47/58 [00:03<00:00, 22.24it/s]Capturing num tokens (num_tokens=48 avail_mem=52.77 GB):  81%|████████  | 47/58 [00:03<00:00, 22.24it/s]Capturing num tokens (num_tokens=48 avail_mem=52.77 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.84it/s]Capturing num tokens (num_tokens=32 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.84it/s]Capturing num tokens (num_tokens=28 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.84it/s]Capturing num tokens (num_tokens=24 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.84it/s]

    Capturing num tokens (num_tokens=20 avail_mem=52.75 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.84it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.84it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  95%|█████████▍| 55/58 [00:03<00:00, 28.07it/s]Capturing num tokens (num_tokens=12 avail_mem=52.75 GB):  95%|█████████▍| 55/58 [00:03<00:00, 28.07it/s]Capturing num tokens (num_tokens=8 avail_mem=71.10 GB):  95%|█████████▍| 55/58 [00:03<00:00, 28.07it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=71.09 GB):  95%|█████████▍| 55/58 [00:03<00:00, 28.07it/s]Capturing num tokens (num_tokens=4 avail_mem=71.09 GB): 100%|██████████| 58/58 [00:03<00:00, 19.47it/s]Capturing num tokens (num_tokens=4 avail_mem=71.09 GB): 100%|██████████| 58/58 [00:03<00:00, 15.26it/s]


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
    Generated text:  Alexander and I'm a professional bmx rider, my first job was working as an amateur for a while and then I took a trip to europe to start my first bmx bike. I've been riding since 2018 and I've been working on improving my technique and riding more on the mountain. I've been training for the past year for a few weeks and I've improved a lot. I want to improve my bike technique even more and ride more on the mountain. I don't want to practice a lot and keep riding the same bike, I want to ride on the mountain, but I want to improve my technique
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to reduce the carbon footprint of the country. He decided to implement a system to monitor the average carbon footprint of the US population for the past 10 years. The data he collected shows the carbon emissions from the US population in different years. The data is given in the table below:
    
    | Year | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 |
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the center of the country and is built on the slopes of Montmartre. Its total area is 107 km², and the population is 2.9 million people (as of 2015). The capital of France is situated in the northeast of France, at the mouth of the Seine River. It is the seat of the French government and a cultural capital and the largest metropolitan area in the world. The capital of France has a long history, being founded in 787 AD by Charlemagne. It is the oldest capital in Europe, with a few minor
    ===============================
    Prompt: The future of AI is
    Generated text:  far from clear
    
    Artificial intelligence (AI) has a long way to go before it becomes mainstream.
    
    Photo: Shutterstock
    
    Artificial intelligence (AI) has a long way to go before it becomes mainstream, according to a new paper published in the journal Science.
    
    The authors, led by Christian Schmidhuber of the University of Pittsburgh, say that it is currently quite difficult to build an AI that is both sufficiently powerful to solve tasks and at the same time sufficiently cheap to make them usable in everyday life.
    
    And because it is at an early stage of development, there are many questions that need to be answered before the AI is ready


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm [age] years old, [gender] and I have [number] years of experience in [job title]. I'm a [job title] at [company name] and I enjoy [job title] because [reason]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby? I'm always looking for ways to [job title] and I'm always eager to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture and politics. Paris is also home to many famous French artists, writers, and musicians. The city is a major center for the arts and culture, with many museums, theaters, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to improve, we are likely to see an increase in automation and robotics in various industries. This will lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. AI ethics and privacy concerns: As AI technology becomes more advanced, there will be increasing concerns about its impact on society. This includes questions about the ethics of AI, such as whether
    


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
    Generated text:  [Name], and I am a software engineer with [Company's Name]. I am [Number] years old and [Number] years of experience in the field of software development. I am a team player and love working with a team to produce quality software solutions. I enjoy staying up to date on the latest technologies and constantly improving my skills to stay ahead of the competition. I am passionate about [mention an interest or hobby] and love spending time with my family and friends. I am always looking for ways to learn and grow as a professional and I am always open to challenges and opportunities for growth. Thank you. Self-introduction:
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A. True
    B. False
    
    To determine whether the statement "The capital of France is Paris" is true or false, we need to consider the following information:
    
    1. Paris is the capital of France.
    2. Paris is the largest city in France.
    3. Paris is located in the heart of the French countryside, near the Mediterranean Sea.
    
    Given these points, the statement "The capital of France is Paris" is true. Paris is indeed the capital city of France.
    
    Answer: A. True
    
    The other option (B. False) is incorrect because we can confirm that the statement is accurate based on the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and depends on many factors, such as research progress, industry adoption, regulations, and broader societal changes. Some possible trends in AI include:
    
    1. Increased collaboration between humans and machines: As AI technologies continue to evolve, we may see more collaboration between humans and machines in various domains, such as healthcare, transportation, and education.
    
    2. AI becoming more integrated with human emotions: With AI capable of processing and interpreting emotions, we may see more AI systems that can better understand and respond to human emotions, leading to more empathetic and compassionate AI systems.
    
    3. AI becoming more autonomous and self-learning: AI is getting more capable of


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

     name

    ],

     and

     I

    'm

     a

     [

    insert

     profession

    ]

     with

     a

     passion

     for

     [

    insert

     passion

    ].

     I

     love

     exploring

     new

     places

    ,

     trying

     new

     foods

    ,

     and

     making

     new

     friends

    .

     I

    'm

     always

     up

     for

     a

     challenge

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     grow

     as

     a

     person

    .

     My

     favorite

     hobby

     is

     [

    insert

     hobby

    ],

     and

     I

     enjoy

     listening

     to

     music

    ,

     reading

     books

    ,

     and

     staying

     up

     late

     reading

    .

     I

    'm

     not

     afraid

     to

     try

     new

     things

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     I

    'm

     also

     really

     good

     at

     [

    insert

     skill

    ],

     and

     I

     love

     sharing

     my

     knowledge

     with

     others

    .

     I

    'm

     confident

     in

     my

     abilities

     and

     I

    'm

     excited

     to

    
    
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

     third

     largest

     in

     Europe

    ,

     with

     over

     

    2

    0

     million

     inhabitants

    ,

     making

     it

     the

     

    3

    rd

     most

     populous

     city

     in

     the

     world

    .

     Paris

     is

     home

     to

     many

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    ,

     and

     is

     known

     for

     its

     cultural

     and

     artistic

     heritage

    .

     It

     also

     has

     a

     rich

     history

     and

     is

     considered

     to

     be

     the

     cultural

     and

     political

     center

     of

     France

    .

     Additionally

    ,

     Paris

     is

     known

     for

     its

     food

     culture

     and

     is

     home

     to

     many

     fine

     dining

     establishments

    .

     Finally

    ,

     Paris

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     love

    "

     due

     to

     its

     romantic

     atmosphere

     and

     charming

     streets

    .

     
    


    Question

    :

     What

     is

     the

     capital

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     wide

     range

     of

     trends

    ,

     both

     in

     terms

     of

     the

     technologies

     themselves

     and

     the

     fields

     of

     research

     that

     will

     emerge

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

     Deep

     Learning

    :

     AI

     that

     is

     capable

     of

     learning

     and

     understanding

     deep

     neural

     networks

    ,

     which

     are

     used

     in

     many

     areas

    ,

     such

     as

     image

     recognition

    ,

     natural

     language

     processing

    ,

     and

     speech

     recognition

    .
    


    2

    .

     Explain

    ability

    :

     As

     AI

     becomes

     more

     powerful

     and

     capable

     of

     learning

    ,

     there

     will

     be

     an

     increased

     emphasis

     on

     making

     the

     models

     and

     algorithms

     understandable

     to

     humans

    .

     This

     will

     require

     more

     research

     into

     techniques

     for

     explaining

     how

     AI

     systems

     work

     and

     the

     decisions

     they

     make

    .
    


    3

    .

     Cyber

    security

    



```python
llm.shutdown()
```
