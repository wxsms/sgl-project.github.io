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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.31it/s]


    2026-05-01 17:30:54,802 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 17:30:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:59,  5.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:59,  5.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:59,  5.25s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:22,  2.30it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:22,  2.30it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:22,  2.30it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:22,  2.30it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:07,  6.20it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:07,  6.20it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:07,  6.20it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.69it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.69it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.69it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:04,  9.69it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:06<00:04,  9.69it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:02, 13.19it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:02, 13.19it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:02, 13.19it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:06<00:02, 13.19it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:06<00:02, 13.19it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:01, 17.28it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:01, 17.28it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:06<00:01, 17.28it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:06<00:01, 17.28it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:06<00:01, 17.28it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:01, 21.21it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:01, 21.21it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:01, 21.21it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 21.21it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:01, 21.21it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 24.74it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 24.74it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 24.74it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:01, 24.74it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:06<00:01, 24.74it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 28.10it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 28.10it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 28.10it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 28.10it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 28.10it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 28.10it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 33.09it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 33.09it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 33.09it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 33.09it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:06<00:00, 33.09it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:06<00:00, 33.09it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 36.08it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 36.08it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 36.08it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 36.08it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 36.08it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 36.08it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 38.81it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 44.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   2%|▏         | 1/58 [00:00<00:09,  5.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   2%|▏         | 1/58 [00:00<00:09,  5.88it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   2%|▏         | 1/58 [00:00<00:09,  5.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:06,  8.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:06,  8.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:06,  8.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:05, 10.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:05, 10.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):   9%|▊         | 5/58 [00:00<00:05, 10.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.14it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:03, 12.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:03, 12.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  21%|██        | 12/58 [00:01<00:03, 12.44it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.86 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.60it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.85 GB):  31%|███       | 18/58 [00:01<00:03, 11.82it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=52.85 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.83 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.87it/s]Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.87it/s] Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:01<00:02, 12.58it/s]Capturing num tokens (num_tokens=896 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:01<00:02, 12.58it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:01<00:02, 12.58it/s]Capturing num tokens (num_tokens=832 avail_mem=52.84 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.94it/s]Capturing num tokens (num_tokens=768 avail_mem=52.83 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.94it/s]Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.94it/s]Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.93it/s]Capturing num tokens (num_tokens=640 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.93it/s]

    Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.93it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.54it/s]Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.54it/s]Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  48%|████▊     | 28/58 [00:02<00:02, 14.54it/s]

    Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  52%|█████▏    | 30/58 [00:02<00:02, 11.99it/s]Capturing num tokens (num_tokens=448 avail_mem=52.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 11.99it/s]Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 11.99it/s]

    Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.01it/s]Capturing num tokens (num_tokens=384 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.01it/s]Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.01it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.88it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.88it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:03<00:02,  8.70it/s]Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  60%|██████    | 35/58 [00:03<00:02,  8.70it/s]

    Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:03<00:02,  8.55it/s]Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:03<00:02,  8.55it/s]Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  64%|██████▍   | 37/58 [00:03<00:02,  8.37it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  64%|██████▍   | 37/58 [00:03<00:02,  8.37it/s]

    Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:02,  8.19it/s]Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:02,  8.19it/s]Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  67%|██████▋   | 39/58 [00:03<00:02,  7.99it/s]Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  67%|██████▋   | 39/58 [00:03<00:02,  7.99it/s]

    Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:02,  8.17it/s]Capturing num tokens (num_tokens=192 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:02,  8.17it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:02,  8.17it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:04<00:01,  8.91it/s]Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:04<00:01,  8.91it/s]

    Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  74%|███████▍  | 43/58 [00:04<00:01,  9.05it/s]Capturing num tokens (num_tokens=144 avail_mem=52.78 GB):  74%|███████▍  | 43/58 [00:04<00:01,  9.05it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  74%|███████▍  | 43/58 [00:04<00:01,  9.05it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  78%|███████▊  | 45/58 [00:04<00:01,  9.48it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  78%|███████▊  | 45/58 [00:04<00:01,  9.48it/s]

    Capturing num tokens (num_tokens=96 avail_mem=52.77 GB):  78%|███████▊  | 45/58 [00:04<00:01,  9.48it/s] Capturing num tokens (num_tokens=96 avail_mem=52.77 GB):  81%|████████  | 47/58 [00:04<00:01,  9.85it/s]Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  81%|████████  | 47/58 [00:04<00:01,  9.85it/s]Capturing num tokens (num_tokens=64 avail_mem=52.76 GB):  81%|████████  | 47/58 [00:04<00:01,  9.85it/s]

    Capturing num tokens (num_tokens=64 avail_mem=52.76 GB):  84%|████████▍ | 49/58 [00:04<00:00, 10.21it/s]Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  84%|████████▍ | 49/58 [00:04<00:00, 10.21it/s]Capturing num tokens (num_tokens=32 avail_mem=52.76 GB):  84%|████████▍ | 49/58 [00:04<00:00, 10.21it/s]Capturing num tokens (num_tokens=32 avail_mem=52.76 GB):  88%|████████▊ | 51/58 [00:04<00:00, 10.70it/s]Capturing num tokens (num_tokens=28 avail_mem=52.75 GB):  88%|████████▊ | 51/58 [00:04<00:00, 10.70it/s]

    Capturing num tokens (num_tokens=24 avail_mem=52.75 GB):  88%|████████▊ | 51/58 [00:04<00:00, 10.70it/s]Capturing num tokens (num_tokens=24 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:04<00:00, 12.17it/s]Capturing num tokens (num_tokens=20 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:04<00:00, 12.17it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.17it/s]Capturing num tokens (num_tokens=12 avail_mem=52.74 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.17it/s]Capturing num tokens (num_tokens=12 avail_mem=52.74 GB):  97%|█████████▋| 56/58 [00:05<00:00, 15.54it/s]Capturing num tokens (num_tokens=8 avail_mem=52.74 GB):  97%|█████████▋| 56/58 [00:05<00:00, 15.54it/s] Capturing num tokens (num_tokens=4 avail_mem=52.73 GB):  97%|█████████▋| 56/58 [00:05<00:00, 15.54it/s]

    Capturing num tokens (num_tokens=4 avail_mem=52.73 GB): 100%|██████████| 58/58 [00:05<00:00, 11.24it/s]


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
    Generated text:  Alex. My name means something to me. It means I was born in a place named Alex and in a year called 1995, I was born. A year before I was born, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year before that, the year
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small village in rural New Jersey. The village is located 10 miles away from the president's headquarters. The president has a friend who lives in the village and can travel at a speed of 10 miles per hour. If the president takes a 30-minute break during his visit, how much longer will it take him to travel the distance to the village and back to his headquarters?
    
    To determine how much longer it will take the president to travel from the village to his headquarters and back to his headquarters, we need to calculate the travel time for each leg of the journey and then sum them up.
    
    First, we
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Berlin
    C. Moscow
    D. St. Petersburg
    
    The capital of France is Paris. Paris is the capital city of France and serves as the governmental, cultural, and economic center of the country. The other cities listed are located in different countries, such as:
    
    B. Berlin is the capital of Germany.
    C. Moscow is the capital of Russia.
    D. St. Petersburg is the capital of Russia.
    
    Therefore, the correct answer is:
    
    A. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  now
    
    A glimpse of the future of AI, October 2019
    
    As we look back on 2019 and the most important developments of the year, AI and AI policy are two of the most talked about topics. We will review what the future holds for AI and what it means for AI policy. How is AI being used? What are the big issues? How are governments responding? We will also look at how AI will continue to develop, with a focus on the potential pitfalls. We'll see what the future looks like for AI and how the future is being shaped.
    
    It is a critical year for AI


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [Brief Biography] who has been working in [Industry] for [Number of Years] years. I'm always looking for new opportunities to grow and learn, and I'm always eager to contribute to the success of [Company Name]. What are your hobbies or interests? I enjoy [Hobby/Interest], and I'm always looking for new experiences to try. What's your favorite book or movie? I'm a huge [Favorite Book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Library. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. It is the second-largest city in France and the most populous city in the country. Paris is also home to many famous landmarks and attractions, including the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and drones to personalized medicine and virtual assistants. Additionally, AI is likely to continue to be used for ethical and social purposes, such as improving access to healthcare and reducing poverty. However, there are also potential risks and challenges associated with AI, such as job displacement and privacy concerns. As technology continues to evolve, it is likely that we will see continued innovation and progress in the field of AI.
    


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
    Generated text:  Emily and I am an AI assistant created by Anthropic. I'm here to help you with your questions, answer your doubts and provide you with the answers to your queries. If you have any questions, feel free to ask and I'll do my best to provide the best possible answer. Let's get started! What's your name? And what do you do? I'm here to help you with any questions you have. What's your name? What do you do? I'm here to help you with any questions you have. Let's get started! What's your name? What do you do? I'm here to help
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    How can I increase my chances of being accepted into Paris? There is no set method to increase your chances of being accepted into Paris, but there are a few steps you can take to increase your chances of making it. Here are some suggestions:
    
      1. Get in touch with French locals: French locals are often helpful and can provide you with insider tips and insider knowledge about the city.
      2. Get a Paris internship: Interning in Paris can provide you with a valuable networking opportunity and a sense of purpose.
      3. Build a Parisian accent: Learning French, especially in the Parisian dialect,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and is expected to continue to change rapidly as new technologies are developed and implemented. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As more and more AI systems are being developed, there is a growing recognition that these systems should be designed with the ethical considerations in mind. This includes considerations of privacy, bias, and fairness, and may lead to new legal and regulatory frameworks for AI development.
    
    2. Continued development and integration of AI into other sectors: AI is already being used in a wide range of sectors, from healthcare to transportation to manufacturing. As the technology advances and becomes more integrated into everyday life


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

    ],

     and

     I

     am

     a

     [

    Job

     Title

    ]

     who

     has

     been

     involved

     in

     [

    Area

     of

     Expert

    ise

     or

     Expert

    ise

     in

     which

     you

     have

     gained

     extensive

     knowledge

     and

     experience

    ].

     I

     believe

     that

     my

     experience

     and

     knowledge

     in

     [

    Area

     of

     Expert

    ise

    ]

     have

     given

     me

     a

     unique

     perspective

     on

     [

    Subject

    ],

     and

     I

     am

     here

     to

     help

     you

     achieve

     [

    Your

     Goal

    ].

     I

     am

     looking

     for

     a

     [

    Job

     Title

    ]

     position

     that

     will

     allow

     me

     to

     contribute

     to

     [

    Reason

     for

     App

    ropriate

     Job

     Title

    ].

     Please

     feel

     free

     to

     ask

     me

     any

     questions

     you

     may

     have

     about

     my

     background

    ,

     experience

    ,

     or

     skills

    .

     [

    Your

     Name

    ]

     [

    Your

     Job

     Title

    ]

     [

    Your

     Area

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    Paris

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     the

     capital

     of

     the

     country

    ,

     located

     on

     the

     River

     Se

    ine

     and

     on

     the

     left

     bank

     of

     the

     Se

    ine

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

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

     the

     Lou

    vre

     Museum

    .

     France

    's

     capital

     city

     is

     a

     vibrant

     and

     culturally

     rich

     place

    ,

     attracting

     millions

     of

     visitors

     every

     year

    .

     Paris

     is

     also

     home

     to

     some

     of

     France

    's

     most

     famous

     landmarks

    ,

     including

     the

     Lou

    vre

     Museum

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

     St

    .

    e

    -P

    ierre

     Church

    .

     Despite

     its

     size

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     rapid

     and

     significant

     evolution

    .

     Some

     possible

     trends

     that

     could

     be

     expected

     in

     the

     future

     of

     AI

     include

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     As

     AI

     becomes

     more

     advanced

     and

     capable

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     with

     human

     intelligence

     to

     provide

     more

     accurate

     and

     personalized

     responses

     to

     users

    .
    


    2

    .

     Emer

    gence

     of

     new

     AI

     technologies

    :

     The

     development

     of

     new

     AI

     technologies

     like

     super

    human

     computers

    ,

     artificial

     consciousness

    ,

     and

     consciousness

     augmentation

     systems

     may

     lead

     to

     the

     creation

     of

     more

     advanced

     and

     intelligent

     AI

     systems

     that

     could

     potentially

     surpass

     human

     intelligence

     in

     some

     areas

    .
    


    3

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

     AI

     becomes

     more

     advanced

     and

     capable

    ,

    



```python
llm.shutdown()
```
