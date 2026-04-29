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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]


    2026-04-29 04:14:50,871 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 04:14:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.51it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.51it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.51it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:20,  2.51it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.39it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.67it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.67it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.67it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.67it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.67it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.16it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.16it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.16it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.16it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.16it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 18.26it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 18.26it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 18.26it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 18.26it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 18.26it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 21.96it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 21.96it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 21.96it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 21.96it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 21.96it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 28.47it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 28.47it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 28.47it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 28.47it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 28.47it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 32.90it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 32.90it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 32.90it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 32.90it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 32.90it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 32.90it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 35.85it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 35.85it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 35.85it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 35.85it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 35.85it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:06<00:00, 35.85it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 39.40it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 39.40it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 39.40it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 39.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.39 GB):   2%|▏         | 1/58 [00:00<00:09,  5.85it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.36 GB):   2%|▏         | 1/58 [00:00<00:09,  5.85it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=57.36 GB):   2%|▏         | 1/58 [00:00<00:09,  5.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.36 GB):   5%|▌         | 3/58 [00:00<00:05, 10.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.35 GB):   5%|▌         | 3/58 [00:00<00:05, 10.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.35 GB):   5%|▌         | 3/58 [00:00<00:05, 10.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.35 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.34 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=57.33 GB):   9%|▊         | 5/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.33 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.33 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.33 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.33 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.91it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.31 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.31 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.31 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.30 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.30 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.30 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.30 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=960 avail_mem=57.29 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s] Capturing num tokens (num_tokens=896 avail_mem=57.29 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]

    Capturing num tokens (num_tokens=832 avail_mem=57.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=768 avail_mem=57.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=704 avail_mem=57.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=640 avail_mem=57.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=640 avail_mem=57.27 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=576 avail_mem=57.27 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=512 avail_mem=56.29 GB):  47%|████▋     | 27/58 [00:01<00:00, 40.59it/s]

    Capturing num tokens (num_tokens=480 avail_mem=56.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=448 avail_mem=56.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=416 avail_mem=57.46 GB):  47%|████▋     | 27/58 [00:01<00:00, 40.59it/s]

    Capturing num tokens (num_tokens=416 avail_mem=57.46 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.94it/s]Capturing num tokens (num_tokens=384 avail_mem=57.24 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.94it/s]Capturing num tokens (num_tokens=352 avail_mem=57.23 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.94it/s]Capturing num tokens (num_tokens=320 avail_mem=56.41 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.94it/s]

    Capturing num tokens (num_tokens=288 avail_mem=56.40 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.94it/s]Capturing num tokens (num_tokens=288 avail_mem=56.40 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.04it/s]Capturing num tokens (num_tokens=256 avail_mem=56.40 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.04it/s]Capturing num tokens (num_tokens=240 avail_mem=57.22 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.04it/s]Capturing num tokens (num_tokens=224 avail_mem=57.21 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.04it/s]

    Capturing num tokens (num_tokens=224 avail_mem=57.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.43it/s]Capturing num tokens (num_tokens=208 avail_mem=56.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.43it/s]Capturing num tokens (num_tokens=192 avail_mem=56.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.43it/s]Capturing num tokens (num_tokens=176 avail_mem=56.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.43it/s]

    Capturing num tokens (num_tokens=176 avail_mem=56.44 GB):  72%|███████▏  | 42/58 [00:02<00:00, 17.26it/s]Capturing num tokens (num_tokens=160 avail_mem=57.21 GB):  72%|███████▏  | 42/58 [00:02<00:00, 17.26it/s]Capturing num tokens (num_tokens=144 avail_mem=57.20 GB):  72%|███████▏  | 42/58 [00:02<00:00, 17.26it/s]Capturing num tokens (num_tokens=128 avail_mem=56.49 GB):  72%|███████▏  | 42/58 [00:02<00:00, 17.26it/s]

    Capturing num tokens (num_tokens=128 avail_mem=56.49 GB):  78%|███████▊  | 45/58 [00:02<00:00, 15.92it/s]Capturing num tokens (num_tokens=112 avail_mem=56.49 GB):  78%|███████▊  | 45/58 [00:02<00:00, 15.92it/s]Capturing num tokens (num_tokens=96 avail_mem=57.20 GB):  78%|███████▊  | 45/58 [00:02<00:00, 15.92it/s] Capturing num tokens (num_tokens=96 avail_mem=57.20 GB):  81%|████████  | 47/58 [00:02<00:00, 15.56it/s]Capturing num tokens (num_tokens=80 avail_mem=57.19 GB):  81%|████████  | 47/58 [00:02<00:00, 15.56it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.53 GB):  81%|████████  | 47/58 [00:02<00:00, 15.56it/s]Capturing num tokens (num_tokens=64 avail_mem=56.53 GB):  84%|████████▍ | 49/58 [00:02<00:00, 15.02it/s]Capturing num tokens (num_tokens=48 avail_mem=56.52 GB):  84%|████████▍ | 49/58 [00:02<00:00, 15.02it/s]Capturing num tokens (num_tokens=32 avail_mem=57.18 GB):  84%|████████▍ | 49/58 [00:02<00:00, 15.02it/s]

    Capturing num tokens (num_tokens=32 avail_mem=57.18 GB):  88%|████████▊ | 51/58 [00:02<00:00, 14.72it/s]Capturing num tokens (num_tokens=28 avail_mem=57.17 GB):  88%|████████▊ | 51/58 [00:02<00:00, 14.72it/s]Capturing num tokens (num_tokens=24 avail_mem=56.57 GB):  88%|████████▊ | 51/58 [00:02<00:00, 14.72it/s]Capturing num tokens (num_tokens=24 avail_mem=56.57 GB):  91%|█████████▏| 53/58 [00:02<00:00, 14.31it/s]Capturing num tokens (num_tokens=20 avail_mem=56.56 GB):  91%|█████████▏| 53/58 [00:02<00:00, 14.31it/s]

    Capturing num tokens (num_tokens=16 avail_mem=57.17 GB):  91%|█████████▏| 53/58 [00:02<00:00, 14.31it/s]Capturing num tokens (num_tokens=16 avail_mem=57.17 GB):  95%|█████████▍| 55/58 [00:02<00:00, 15.03it/s]Capturing num tokens (num_tokens=12 avail_mem=56.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 15.03it/s]Capturing num tokens (num_tokens=8 avail_mem=56.61 GB):  95%|█████████▍| 55/58 [00:03<00:00, 15.03it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=56.61 GB):  98%|█████████▊| 57/58 [00:03<00:00, 14.62it/s]Capturing num tokens (num_tokens=4 avail_mem=57.16 GB):  98%|█████████▊| 57/58 [00:03<00:00, 14.62it/s]Capturing num tokens (num_tokens=4 avail_mem=57.16 GB): 100%|██████████| 58/58 [00:03<00:00, 18.28it/s]


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
    Generated text:  Emily and I'm a digital marketing expert. Can you tell me about the features and benefits of my personal blog website? Please, feel free to provide specific examples to highlight the key points.
    
    Sure, I'd be happy to help! Can you please tell me what kind of information you want to include in your personal blog website? Whether it's about your current projects, your passion, your travel experiences, your latest news or any other topic related to your interests? Additionally, do you have any specific requirements or preferences regarding the website layout, color scheme, and design elements? 
    
    Once I have a better understanding of your needs and preferences,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to continue a policy that has been implemented for many years. The policy is to increase the price of gasoline, which has been increasing at an exponential rate for a period of time. The president wants to know how long the policy will last before it starts to have negative effects on the economy. The price of gasoline in the United States has been increasing at an annual rate of 1.2% for the last 10 years. Calculate the growth rate of the price of gasoline over time and determine how long it will take for the policy to become detrimental to the economy. To determine how long the policy to increase the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Marseille
    D. Nice
    
    The capital of France is:
    
    A. Paris
    
    Paris is the capital city of France and is also known as "La Prefecture" in French. It is a historic city on the River Seine with a population of around 10.5 million people. The city is famous for its museums, music, art, fashion, and food. Paris is also known for its architecture, including its iconic Eiffel Tower and the Louvre Museum. The city is a major tourist destination and a popular place for French culture, music, and entertainment.
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable, and any AI project is in the early stages of development. That means that there are a lot of ways that it can go wrong and a lot of steps that need to be taken to make sure that the project will succeed. It’s important for the AI team to have a well-thought-out plan in place, including a team meeting to discuss progress, budget, timelines, and risk management. But, without a clear plan, there is also a risk of a major mistake being made. One of the things to consider is the impact of data on the AI project.
    
    Data is the lifeblood of an AI project, and it


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What you do best]. I'm always looking for ways to [What you do to improve]. I'm a [What you do for fun]. I'm [What you do for fun]. I'm [What you do for fun]. I'm [What you do for fun]. I'm [What you do for fun]. I'm [What you do for fun]. I'm [What you do for fun]. I'm [What you do for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    The statement is factual because it provides a clear and unambiguous definition of the capital city of France, which is Paris. It does not contain any assumptions, make any claims, or introduce any new information that is not explicitly stated. The statement is a straightforward and accurate description of the capital city of France. 
    
    To summarize, the statement "The capital of France is Paris" is a factual statement that accurately describes the location of the capital city of France. It is a concise and clear statement that provides a clear and unambiguous definition of the capital city of France. 
    
    The statement is factual because it is a direct and un
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing ethical AI systems that are designed to minimize harm and maximize benefits.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As more companies and governments invest in AI, it is likely that we will see more integration of AI with other technologies, such as blockchain and quantum computing
    


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
    Generated text:  [Name], I'm a [Occupation] with a passion for [Your hobby or interest]. 🌟 I'm a [job title] at [Company], where I work hard to [What you do at work]. I'm always looking for new experiences and ideas, and I'm excited to learn more about [Why you're passionate about this field]. I'm always ready to contribute to the company's success and have a great time working with [Other Team Members or Colleagues]. I'm always looking for ways to improve my skills and stay up-to-date with the latest technology and trends. I'm a [how you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south of the country and is the largest city in Europe.
    
    Can you provide some more information about Paris and its famous landmarks? Sure! Paris, the capital of France, is a beautiful city known for its iconic landmarks such as Notre Dame Cathedral, the Eiffel Tower, Louvre Museum, the Musée d'Orsay, the Palais des Papes, the Arc de Triomphe, the Palace of Versailles, the Louvre Museums, and the Notre Dame Cathedral. Paris is also renowned for its rich history, art, and architecture, and has been a popular tourist destination for centuries.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly dynamic and evolving, with numerous possibilities and potential applications. Here are some of the most promising trends in AI that could shape the industry in the next decade:
    
    1. **Integration with Human Intelligence**: AI systems are becoming more integrated with human intelligence, combining the strengths of both to create more sophisticated and adaptive systems. This could lead to more holistic and empathetic AI, capable of understanding and responding to complex human emotions and behaviors.
    
    2. **Enhanced Natural Language Understanding**: AI is evolving to be more adept at understanding and generating natural language. This could lead to more sophisticated conversational interfaces, more accurate information retrieval, and better predictive analytics


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

    'm

     a

     [

    short

     title

    ]

     from

     [

    Location

    ].

     I

    'm

     a

     [

    character

     type

    ]

     [

    description

    ].

     I

    'm

     passionate

     about

     [

    career

     goal

     or

     hobby

    ]

     and

     I

    'm

     always

     looking

     to

     learn

     and

     grow

    .

     I

    'm

     a

     [

    character

     trait

     or

     quality

    ]

     [

    description

    ].

     I

    'm

     eager

     to

     meet

     you

     and

     share

     my

     experiences

    ,

     knowledge

    ,

     and

     ideas

    .

     I

    'm

     ready

     to

     inspire

     and

     challenge

     you

     to

     achieve

     your

     own

     goals

    .

     Let

    's

     connect

    !

     How

     about

     [

    Name

    ]?

     Let

    's

     make

     some

     connections

    !

     [

    Name

    ]

    !


    I

    .

     INT

    RODUCTION

    
    


    II

    .

     DESCRIPTION

    
    


    III

    .

     PURPOSE

    
    


    IV

    .

     G

    reetings

    
    


    V

    .

     THE

     NEED

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

     


    B

    .

     False

    
    


    To

     determine

     the

     correct

     answer

    ,

     I

     will

     follow

     these

     steps

    :
    


    1

    .

     Recall

     the

     name

     of

     the

     capital

     city

     of

     France

    .


    2

    .

     Compare

     the

     statement

     to

     the

     information

     I

     recall

    .


    3

    .

     Provide

     the

     answer

     based

     on

     my

     recall

     and

     the

     given

     options

    .
    


    Step

     

    1

    :

     Rec

    alling

     the

     name

     of

     the

     capital

     city

     of

     France

    .


    The

     capital

     city

     of

     France

     is

     Paris

    .
    


    Step

     

    2

    :

     Compar

    ing

     the

     statement

     to

     the

     information

     I

     recall

    .


    The

     statement

     "

    The

     capital

     of

     France

     is

     Paris

    "

     is

     accurate

     based

     on

     the

     information

     I

     recalled

    .
    


    Step

     

    3

    :

     Providing

     the

     answer

    .


    The

     correct

     answer

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     trends

     that

     are

     currently

     in

     development

     or

     in

     the

     early

     stages

     of

     development

    ,

     but

     could

     potentially

     evolve

     to

     revolution

    ize

     industries

     and

     transform

     the

     way

     we

     live

     and

     work

    .
    


    One

     trend

     that

     is

     likely

     to

     grow

     in

     importance

     is

     the

     ability

     of

     AI

     to

     learn

     and

     adapt

     to

     new

     situations

     and

     data

    .

     This

     means

     that

     AI

     systems

     will

     become

     more

     capable

     of

     recognizing

     patterns

     and

     making

     decisions

     that

     are

     better

     suited

     to

     the

     unique

     needs

     of

     different

     users

     and

     situations

    .
    


    Another

     trend

     is

     the

     increasing

     integration

     of

     AI

     into

     existing

     technologies

    .

     This

     could

     include

     things

     like

     autonomous

     vehicles

    ,

     voice

     assistants

    ,

     and

     smart

     cities

    ,

     all

     of

     which

     will

     benefit

     from

     the

     power

     and

     efficiency

     of

    



```python
llm.shutdown()
```
