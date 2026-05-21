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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.31it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.31it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.31it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.31it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 11.56it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 11.56it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 11.56it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 11.56it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.56it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.47it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.67it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.67it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.67it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.67it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.67it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 20.67it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 25.69it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 25.69it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 25.69it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 25.69it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 25.69it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 25.69it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 36.66it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 41.34it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 45.75it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 45.75it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 45.75it/s] 

    Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 45.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.85 GB):   2%|▏         | 1/58 [00:00<00:08,  6.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.82 GB):   2%|▏         | 1/58 [00:00<00:08,  6.36it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.82 GB):   3%|▎         | 2/58 [00:00<00:08,  6.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.81 GB):   3%|▎         | 2/58 [00:00<00:08,  6.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.81 GB):   5%|▌         | 3/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.81 GB):   5%|▌         | 3/58 [00:00<00:07,  7.33it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.81 GB):   7%|▋         | 4/58 [00:00<00:07,  7.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.81 GB):   7%|▋         | 4/58 [00:00<00:07,  7.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.81 GB):   9%|▊         | 5/58 [00:00<00:06,  7.84it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.81 GB):   9%|▊         | 5/58 [00:00<00:06,  7.84it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.81 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.78 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.78 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.77 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.70it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.77 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.70it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.77 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.02it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.83it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.83it/s] Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:02<00:02, 15.83it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.36it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.36it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.36it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.36it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  50%|█████     | 29/58 [00:02<00:01, 19.53it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 19.53it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 19.53it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 19.53it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.68it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.68it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.68it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.68it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 20.67it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.58it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.68it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.68it/s]

    Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.68it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.68it/s] Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 22.87it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 22.87it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 22.87it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 22.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.01it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.01it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.01it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 23.01it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 23.69it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 23.69it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 23.69it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 23.69it/s]

    Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 23.77it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 23.77it/s] Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  97%|█████████▋| 56/58 [00:03<00:00, 23.77it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 16.05it/s]


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
    Generated text:  Zhang Xiaogang, I am 16 years old and am currently in Class 7, Grade 1 at a middle school in Xiamen City, Fujian Province. My parents and I are currently in charge of managing the school's dining hall, which employs 100 students. As a student, I am passionate about studying, but my passion for studying has been somewhat intermittent. On one occasion, I had a problem with my math homework and I was unable to work on it for several hours. One day, I received an urgent call from my teacher asking me to help someone with math homework. I was forced
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected for a 4-year term. The president is eligible for re-election, but not more than 2 consecutive terms. After how many terms can the president be re-elected if the president is eligible for re-election every 4 years?
    
    To determine how many terms the president can be re-elected, we need to understand the conditions given: the president is eligible for re-election every 4 years, and the president can be re-elected not more than 2 consecutive terms. Let's break this down step by step.
    
    1. Identify the number of terms in one cycle of 4 years:
       - The president is eligible for re-election
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a very large city in western France. It has a population of over 2 million. The city is on the Seine River. The river runs through the city center. It has a beautiful view of the city. 
    
    While Paris is a very big city, there are many smaller cities in the nearby area. These smaller cities are called suburbs. There are many suburbs in Paris. 
    
    Sometimes, people in Paris have to travel a long way to go to work. This is because Paris has a lot of buildings and lots of roads. Paris is very large, so it has many places where people can go for shopping
    ===============================
    Prompt: The future of AI is
    Generated text:  promising, but it also presents a number of risks. Here’s a look at some of the areas where the tech is growing and expanding, and the areas where it’s not.
    30% of the world’s population have access to the internet, and it’s been growing at an astounding rate. This means that the connected world is being flooded with data, and that data is growing faster than the network of computers that process it.
    It’s also true that some of the most advanced AI is being used to generate more accurate language translations, create more realistic video games, and create more accurate ways to diagnose diseases.
    In addition, there are


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity], and I'm always looking for ways to expand my skills and knowledge in this field. What's your favorite book or movie? I love [book or movie], and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant culture, including its annual Eiffel Tower Festival and its annual fashion week. The city is a major transportation hub and is home to many of France's major cities and regions. Paris is a popular tourist destination and is known for its rich history, art, and cuisine. It is also home to many of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies such as sensors, actuators, and power systems, leading to more efficient and effective use of resources.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical
    


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
    Generated text:  [Name] and I am a [Age] year old [Occupation] with a passion for [Interest or Hobby]. I am always looking for opportunities to grow and learn, and I am always up for trying new things. I believe that education is key to my personal and professional growth, and I am always willing to share my knowledge and experiences with others. I am excited to meet you and contribute to your journey of growth and learning. What's your name? What's your occupation? What's your passion? What's your interest or hobby? What's your education? How do you like to learn? What's your favorite food
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the 19th-largest city in the world by population. Paris has a population of around 2. 7 million people and is the largest city in metropolitan France. Its economic and cultural center, it is also known as the "city of love" and hosts the world’s largest art museum, the Louvre. It has been named "the world's favorite city" by various publications and is a popular tourist destination. It has a rich history and is known for its art, architecture, and music. Paris is an important center for politics, science, and culture, and is a major hub for international trade and finance
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends, including:
    
    1. Advancements in machine learning and deep learning: With the increasing availability of large amounts of data and powerful computers, AI algorithms will continue to get smarter and more accurate. This will lead to applications such as image and speech recognition, natural language processing, autonomous vehicles, and more.
    
    2. Increased reliance on AI for decision-making: As AI becomes more integrated into everyday life, we may see increased reliance on it for decision-making in everything from healthcare to finance to business operations.
    
    3. Integration with other technologies: AI will continue to be integrated into other technologies, such as virtual reality


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

    ]

     and

     I

    'm

     a

    /an

     [

    insert

     occupation

     or

     profession

    ]

     from

     [

    insert

     country

     or

     location

    ].

     I

     have

     a

    /an

     [

    insert

     personality

     trait

     or

     hobby

    ]

     that

     makes

     me

     unique

     and

     interesting

    .

     I

     enjoy

     [

    insert

     why

     you

     enjoy

     what

     you

     do

    ]

     and

     I

     strive

     to

     do

     my

     best

     every

     day

    ,

     no

     matter

     what

    .

     [

    insert

     how

     you

     started

     to

     write

     this

     introduction

    ,

     which

     could

     be

     something

     like

     "

    I

     grew

     up

     in

     a

     small

     town

     with

     my

     grandmother

    ,

     who

     was

     a

     folk

     artist

    .

     I

     learned

     to

     draw

     and

     paint

    ,

     and

     my

     grandmother

    's

     passion

     for

     her

     art

     has

     always

     inspired

     me

     to

     pursue

     a

     creative

     career

    .

     I

    've

     always

     been

     a

    
    
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

    ,"

     an

     iconic

     city

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     and

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

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

    ,

     and

     is

     the

     birth

    place

     of

     many

     famous

     figures

     in

     literature

     and

     art

    .

     Paris

     has

     a

     vibrant

     and

     diverse

     culture

    ,

     and

     is

     known

     for

     its

     jazz

    ,

     film

    ,

     and

     music

     scenes

    .

     The

     city

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     which

     are

     iconic

     landmarks

     in

     Paris

    .

     Paris

     is

     a

     bustling

     and

     dynamic

     city

     with

     a

     rich

     cultural

     heritage

     and

     a

     rich

     history

    .

     The

     French

     language

     is

     also

     spoken

     throughout

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     integration

    :

     As

     more

     devices

     and

     platforms

     become

     connected

     to

     the

     internet

    ,

     we

     are

     likely

     to

     see

     a

     continued

     integration

     of

     AI

     into

     all

     aspects

     of

     our

     lives

    .

     This

     could

     include

     more

     advanced

     chat

    bots

    ,

     voice

     assistants

    ,

     and

     wearable

     devices

    ,

     which

     will

     rely

     on

     AI

     algorithms

     to

     provide

     more

     personalized

     and

     efficient

     services

    .
    


    2

    .

     Artificial

     intelligence

     becoming

     more

     autonomous

    :

     In

     the

     near

     future

    ,

     we

     may

     see

     more

     AI

     systems

     that

     can

     make

     decisions

     and

     take

     action

     without

     human

     oversight

    ,

     leading

     to

     more

     automated

     and

     efficient

     systems

    .

     This

     could

     be

     seen

     in

     areas

     such

     as

     transportation

    ,

     healthcare

    ,

     and

     manufacturing

    .
    


    3

    .

     AI

     becoming

    



```python
llm.shutdown()
```
