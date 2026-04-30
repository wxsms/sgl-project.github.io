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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.59it/s]


    2026-04-30 07:20:30,162 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 07:20:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:21,  2.38it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:21,  2.38it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:21,  2.38it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:21,  2.38it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:11,  4.16it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:11,  4.16it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:07,  6.38it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:07,  6.38it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:07,  6.38it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:07,  6.38it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:07,  6.38it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.90it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.90it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.90it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04,  9.90it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:04,  9.90it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 13.70it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 18.98it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:06<00:01, 18.98it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:06<00:01, 18.98it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:06<00:01, 24.36it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:00, 30.71it/s]

    Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 36.78it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 40.96it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 44.49it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 44.49it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 44.49it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 44.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.07 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.04 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.04 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.03 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.03 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.03 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.03 GB):   7%|▋         | 4/58 [00:00<00:06,  7.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.03 GB):   7%|▋         | 4/58 [00:00<00:06,  7.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.03 GB):   9%|▊         | 5/58 [00:00<00:06,  8.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.03 GB):   9%|▊         | 5/58 [00:00<00:06,  8.26it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.02 GB):   9%|▊         | 5/58 [00:00<00:06,  8.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.02 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.01 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.01 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.01 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.01 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.00 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.00 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.00 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.00 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.14it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.00 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.00 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.99 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.99 GB):  26%|██▌       | 15/58 [00:01<00:02, 14.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.99 GB):  26%|██▌       | 15/58 [00:01<00:02, 14.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.98 GB):  26%|██▌       | 15/58 [00:01<00:02, 14.70it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.98 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.96 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.12it/s]Capturing num tokens (num_tokens=960 avail_mem=41.97 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.12it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=41.97 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.12it/s]Capturing num tokens (num_tokens=896 avail_mem=41.97 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=832 avail_mem=41.97 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=768 avail_mem=41.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=704 avail_mem=41.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=704 avail_mem=41.96 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.62it/s]Capturing num tokens (num_tokens=640 avail_mem=41.96 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.62it/s]Capturing num tokens (num_tokens=576 avail_mem=41.96 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.62it/s]

    Capturing num tokens (num_tokens=512 avail_mem=41.94 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.62it/s]Capturing num tokens (num_tokens=512 avail_mem=41.94 GB):  50%|█████     | 29/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=480 avail_mem=41.96 GB):  50%|█████     | 29/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=448 avail_mem=41.95 GB):  50%|█████     | 29/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=416 avail_mem=41.95 GB):  50%|█████     | 29/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=416 avail_mem=41.95 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.82it/s]Capturing num tokens (num_tokens=384 avail_mem=41.95 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.82it/s]

    Capturing num tokens (num_tokens=352 avail_mem=41.94 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.82it/s]Capturing num tokens (num_tokens=320 avail_mem=41.94 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.82it/s]Capturing num tokens (num_tokens=320 avail_mem=41.94 GB):  60%|██████    | 35/58 [00:02<00:01, 22.53it/s]Capturing num tokens (num_tokens=288 avail_mem=41.93 GB):  60%|██████    | 35/58 [00:02<00:01, 22.53it/s]Capturing num tokens (num_tokens=256 avail_mem=41.93 GB):  60%|██████    | 35/58 [00:02<00:01, 22.53it/s]Capturing num tokens (num_tokens=240 avail_mem=41.93 GB):  60%|██████    | 35/58 [00:02<00:01, 22.53it/s]

    Capturing num tokens (num_tokens=240 avail_mem=41.93 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.14it/s]Capturing num tokens (num_tokens=224 avail_mem=41.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.14it/s]Capturing num tokens (num_tokens=208 avail_mem=41.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.14it/s]Capturing num tokens (num_tokens=192 avail_mem=41.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.14it/s]Capturing num tokens (num_tokens=192 avail_mem=41.92 GB):  71%|███████   | 41/58 [00:02<00:00, 20.82it/s]Capturing num tokens (num_tokens=176 avail_mem=41.92 GB):  71%|███████   | 41/58 [00:02<00:00, 20.82it/s]Capturing num tokens (num_tokens=160 avail_mem=41.91 GB):  71%|███████   | 41/58 [00:02<00:00, 20.82it/s]

    Capturing num tokens (num_tokens=144 avail_mem=41.91 GB):  71%|███████   | 41/58 [00:02<00:00, 20.82it/s]Capturing num tokens (num_tokens=144 avail_mem=41.91 GB):  76%|███████▌  | 44/58 [00:02<00:00, 20.81it/s]Capturing num tokens (num_tokens=128 avail_mem=41.91 GB):  76%|███████▌  | 44/58 [00:02<00:00, 20.81it/s]Capturing num tokens (num_tokens=112 avail_mem=41.91 GB):  76%|███████▌  | 44/58 [00:02<00:00, 20.81it/s]Capturing num tokens (num_tokens=96 avail_mem=41.90 GB):  76%|███████▌  | 44/58 [00:02<00:00, 20.81it/s] Capturing num tokens (num_tokens=96 avail_mem=41.90 GB):  81%|████████  | 47/58 [00:02<00:00, 20.84it/s]Capturing num tokens (num_tokens=80 avail_mem=41.90 GB):  81%|████████  | 47/58 [00:02<00:00, 20.84it/s]

    Capturing num tokens (num_tokens=64 avail_mem=41.90 GB):  81%|████████  | 47/58 [00:02<00:00, 20.84it/s]Capturing num tokens (num_tokens=48 avail_mem=41.89 GB):  81%|████████  | 47/58 [00:02<00:00, 20.84it/s]Capturing num tokens (num_tokens=48 avail_mem=41.89 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.08it/s]Capturing num tokens (num_tokens=32 avail_mem=41.89 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.08it/s]

    Capturing num tokens (num_tokens=28 avail_mem=41.88 GB):  86%|████████▌ | 50/58 [00:03<00:00, 19.08it/s]Capturing num tokens (num_tokens=28 avail_mem=41.88 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.51it/s]Capturing num tokens (num_tokens=24 avail_mem=41.88 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.51it/s]Capturing num tokens (num_tokens=20 avail_mem=41.88 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.51it/s]Capturing num tokens (num_tokens=16 avail_mem=41.88 GB):  90%|████████▉ | 52/58 [00:03<00:00, 18.51it/s]Capturing num tokens (num_tokens=16 avail_mem=41.88 GB):  95%|█████████▍| 55/58 [00:03<00:00, 19.44it/s]Capturing num tokens (num_tokens=12 avail_mem=41.87 GB):  95%|█████████▍| 55/58 [00:03<00:00, 19.44it/s]

    Capturing num tokens (num_tokens=8 avail_mem=41.87 GB):  95%|█████████▍| 55/58 [00:03<00:00, 19.44it/s] Capturing num tokens (num_tokens=4 avail_mem=41.87 GB):  95%|█████████▍| 55/58 [00:03<00:00, 19.44it/s]Capturing num tokens (num_tokens=4 avail_mem=41.87 GB): 100%|██████████| 58/58 [00:03<00:00, 19.92it/s]Capturing num tokens (num_tokens=4 avail_mem=41.87 GB): 100%|██████████| 58/58 [00:03<00:00, 17.24it/s]


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
    Generated text:  Toshiro and I am a programmer. I am interested in general information about the advancement of technology, and I am looking for information on the development of computers. I would like to know if anyone has written a book about this topic.
    
    I would be grateful if you could share your knowledge about computer development, programming languages, computer architecture, hardware architecture, and so on. I would also like to know what the most popular programming languages are today. I would also like to know what the most interesting topics to learn about in computer programming are. Please provide your response in a clear and concise manner.
    
    Additionally, I would like to know about
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is in charge of a country. He or she is like the boss of the country. The president is different from the Prime Minister of a country. The Prime Minister is like the head of state. The Prime Minister is like the boss of the country. But the president can't be a person. He or she can be a person or a group of people. The president is like a grown-up who gets to say all the important things. He or she is like a captain of a ship. He or she has to always be careful of the country. In the US, there are 4
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is well known for its rich history and beautiful architecture. It's also famous for its famous landmarks like Notre Dame Cathedral, the Eiffel Tower, and the Louvre. But there's more to Paris than its famous landmarks. It's also home to many different kinds of people. For example, there are many immigrants and their descendants who live in the city. There are also many people of color who come from Africa and Asia. There are also many people who are gay and lesbian. Paris is a wonderful city for everyone to visit. There's something for everyone. So, it's hard to imagine what life would be
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of humans, not machines
    
    Artificial intelligence (AI) is a rapidly evolving field, with opportunities and challenges for the future. The key is to use the technology responsibly and to ensure that its development is aligned with ethical and societal goals.
    
    Read on to learn why AI is here to stay and what we can expect from the future of AI.
    
    Why is AI being so important?
    
    AI is an important field for a few reasons:
    
      1. It can enable us to make discoveries that we could not possibly have made without it. For example, AI has the potential to solve complex problems that would be too difficult for humans


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also home to the French Riviera, a popular tourist destination for its beautiful beaches and Mediterranean climate. The city is known for its cuisine, including French cuisine, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its historical architecture and modern art, and is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater reliance on machine learning: Machine learning is expected to become more prevalent in AI, allowing machines to learn from data and improve their performance over time. This could lead to more efficient and effective AI systems that can adapt
    


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
    Generated text:  [Name] and I am a [role] at [company]. I love [a specific hobby or activity that you enjoy], and I am always looking for [a specific reason for learning new things or improving my skills]. I believe in [a principle or belief that resonates with you as a reader], and I am always eager to learn more about [a topic that interests you]. I am passionate about [a personal hobby or passion that you enjoy], and I am always eager to share my knowledge with others. How can I help you today? [Name]: "Hello, my name is [Name] and I am a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the centre of the country and is known for its beautiful parks, historical sites, and cultural attractions. It has been a major European hub for centuries and is the birthplace of many famous French artists and intellectuals. The city also hosts the headquarters of some of France's most prestigious institutions and is a major tourist destination. Paris is home to many international landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Montmartre. The city is known for its diverse cuisine, with many famous French dishes and a rich history of wine and cheese. Paris is a city of contrasts and is a popular destination for international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and changing rapidly. Here are some possible trends that could impact the field in the next decade:
    
    1. Machine learning and deep learning: Machine learning and deep learning are two subfields of AI that are currently making significant progress. These technologies have the potential to revolutionize a wide range of industries, from healthcare to finance to entertainment.
    
    2. Cybersecurity: With the increasing amount of data being generated and stored online, cybersecurity has become a top priority for AI researchers. New tools and techniques are being developed to protect against cyber attacks and to detect and respond to them more quickly and efficiently.
    
    3. Explainability: AI systems are becoming more


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

    ],

     and

     I

    'm

     a

     [

    career

     goal

    ]

     [

    type

     of

     profession

    ].

     I

    'm

     currently

     pursuing

     [

    job

     title

    ]

     at

     [

    company

     name

    ],

     where

     I

    've

     gained

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     this

     field

    .

     I

    'm

     passionate

     about

     [

    why

     you

    're

     passionate

     about

     your

     job

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    what

     you

     hope

     to

     achieve

    ]

     in

     your

     career

    .

     I

    'm

     always

     looking

     for

     opportunities

     to

     [

    what

     you

     hope

     to

     learn

    ]

     in

     your

     job

    .

     I

    'm

     also

     [

    what

     you

     hope

     to

     do

     as

     a

     leader

    ].

     I

     believe

     in

     [

    pr

    inciple

     of

     leadership

    ],

     and

     I

    'm

     always

     willing

     to

     learn

     and

     grow

    ,

     no

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .


    Paris

     is

     the

     capital

     of

     France

    ,

     a

     country

     located

     in

     the

     central

     part

     of

     the

     European

     continent

    ,

     at

     the

     cross

    roads

     of

     three

     major

     civilizations

    :

     the

     Roman

    ,

     the

     Islamic

    ,

     and

     the

     Christian

    .

     The

     city

     is

     renowned

     for

     its

     rich

     history

    ,

     artistic

     culture

    ,

     and

     scenic

     beauty

    .

     Paris

     has

     a

     unique

     blend

     of

     French

    ,

     British

    ,

     and

     Mediterranean

     influences

    ,

     making

     it

     a

     melting

     pot

     of

     various

     ethnic

     groups

     and

     cultures

    .

     The

     city

     also

     has

     a

     long

     and

     fascinating

     history

    ,

     having

     been

     ruled

     by

     various

     em

    pires

    ,

     including

     the

     Roman

    ,

     Arab

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     next

     few

     years

    :
    


    1

    .

     Autonomous

     and

     semi

    -aut

    onomous

     vehicles

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     may

     see

     more

     advanced

     autonomous

     and

     semi

    -aut

    onomous

     vehicles

     on

     the

     road

    .

     These

     vehicles

     will

     be

     able

     to

     learn

     from

     their

     environment

     and

     make

     decisions

     on

     their

     own

    ,

     which

     could

     significantly

     reduce

     traffic

     accidents

     and

     reduce

     reliance

     on

     human

     drivers

    .
    


    2

    .

     Personal

    ized

     medicine

    :

     AI

     could

     help

     doctors

     to

     make

     more

     accurate

     diagnoses

     and

     develop

     more

     effective

     treatments

     for

     diseases

    .

     Personal

    ized

     medicine

     could

     also

     help

     to

     treat

     diseases

     more

     efficiently

    ,

     since

     it

     could

     identify

     which

     treatments

     are

     most

     likely

     to

     work

    



```python
llm.shutdown()
```
