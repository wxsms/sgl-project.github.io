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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.41it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.41it/s]


    2026-05-04 16:14:24,612 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 16:14:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:50,  1.97s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:50,  1.97s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:19,  2.62it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:05<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:12,  4.01it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:12,  4.01it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:12,  4.01it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:12,  4.01it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:07,  6.56it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:07,  6.56it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:07,  6.56it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:07,  6.56it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:04,  9.21it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:04,  9.21it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:04,  9.21it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:04,  9.21it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.79it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.79it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.79it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.79it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.70it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.82it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.82it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.82it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:06<00:02, 15.82it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:01, 18.04it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:01, 18.04it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:01, 18.04it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 18.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:01, 19.76it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:01, 19.76it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:01, 19.76it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 19.76it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:06<00:01, 21.52it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:06<00:01, 21.52it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:06<00:01, 21.52it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:06<00:01, 21.52it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:01, 22.86it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:01, 22.86it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:01, 22.86it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:06<00:01, 22.86it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:00, 23.16it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:00, 23.16it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:00, 23.16it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:00, 23.16it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 24.66it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 24.66it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 24.66it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 24.66it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 25.68it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 25.25it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 25.25it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 25.25it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 25.25it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 25.20it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 25.20it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 25.89it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 25.89it/s]

    Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 25.89it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 25.89it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:07<00:00, 26.41it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:07<00:00, 26.41it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:07<00:00, 26.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.41 GB):   2%|▏         | 1/58 [00:00<00:11,  5.09it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.99 GB):   2%|▏         | 1/58 [00:00<00:11,  5.09it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.99 GB):   3%|▎         | 2/58 [00:00<00:11,  4.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.44 GB):   3%|▎         | 2/58 [00:00<00:11,  4.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.44 GB):   5%|▌         | 3/58 [00:00<00:11,  4.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.98 GB):   5%|▌         | 3/58 [00:00<00:11,  4.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.98 GB):   7%|▋         | 4/58 [00:00<00:10,  5.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.49 GB):   7%|▋         | 4/58 [00:00<00:10,  5.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=38.49 GB):   9%|▊         | 5/58 [00:00<00:10,  5.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.48 GB):   9%|▊         | 5/58 [00:00<00:10,  5.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.48 GB):  10%|█         | 6/58 [00:01<00:09,  5.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.96 GB):  10%|█         | 6/58 [00:01<00:09,  5.45it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.96 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.52 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.52 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.96 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.76it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.96 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.54 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.54 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.54 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.54 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.95 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.95 GB):  21%|██        | 12/58 [00:02<00:07,  6.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.56 GB):  21%|██        | 12/58 [00:02<00:07,  6.38it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.56 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.94 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.94 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.57 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.87it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.57 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.57 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.57 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.92 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.18it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.92 GB):  29%|██▉       | 17/58 [00:02<00:05,  6.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.59 GB):  29%|██▉       | 17/58 [00:02<00:05,  6.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.59 GB):  31%|███       | 18/58 [00:02<00:05,  7.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.92 GB):  31%|███       | 18/58 [00:02<00:05,  7.20it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=38.92 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.61 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.61 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.90 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.50it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.90 GB):  36%|███▌      | 21/58 [00:03<00:04,  7.75it/s]Capturing num tokens (num_tokens=960 avail_mem=38.63 GB):  36%|███▌      | 21/58 [00:03<00:04,  7.75it/s] Capturing num tokens (num_tokens=960 avail_mem=38.63 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.76it/s]Capturing num tokens (num_tokens=896 avail_mem=38.90 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.76it/s]

    Capturing num tokens (num_tokens=896 avail_mem=38.90 GB):  40%|███▉      | 23/58 [00:03<00:04,  8.05it/s]Capturing num tokens (num_tokens=832 avail_mem=38.65 GB):  40%|███▉      | 23/58 [00:03<00:04,  8.05it/s]Capturing num tokens (num_tokens=832 avail_mem=38.65 GB):  41%|████▏     | 24/58 [00:03<00:04,  7.91it/s]Capturing num tokens (num_tokens=768 avail_mem=38.90 GB):  41%|████▏     | 24/58 [00:03<00:04,  7.91it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.90 GB):  43%|████▎     | 25/58 [00:03<00:03,  8.29it/s]Capturing num tokens (num_tokens=704 avail_mem=38.67 GB):  43%|████▎     | 25/58 [00:03<00:03,  8.29it/s]Capturing num tokens (num_tokens=704 avail_mem=38.67 GB):  45%|████▍     | 26/58 [00:03<00:03,  8.39it/s]Capturing num tokens (num_tokens=640 avail_mem=38.89 GB):  45%|████▍     | 26/58 [00:03<00:03,  8.39it/s]Capturing num tokens (num_tokens=576 avail_mem=38.69 GB):  45%|████▍     | 26/58 [00:03<00:03,  8.39it/s]

    Capturing num tokens (num_tokens=576 avail_mem=38.69 GB):  48%|████▊     | 28/58 [00:04<00:03,  9.13it/s]Capturing num tokens (num_tokens=512 avail_mem=38.87 GB):  48%|████▊     | 28/58 [00:04<00:03,  9.13it/s]Capturing num tokens (num_tokens=512 avail_mem=38.87 GB):  50%|█████     | 29/58 [00:04<00:03,  8.84it/s]Capturing num tokens (num_tokens=480 avail_mem=38.72 GB):  50%|█████     | 29/58 [00:04<00:03,  8.84it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.88 GB):  50%|█████     | 29/58 [00:04<00:03,  8.84it/s]Capturing num tokens (num_tokens=448 avail_mem=38.88 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.53it/s]Capturing num tokens (num_tokens=416 avail_mem=38.88 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.53it/s]

    Capturing num tokens (num_tokens=416 avail_mem=38.88 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.54it/s]Capturing num tokens (num_tokens=384 avail_mem=38.86 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.54it/s]Capturing num tokens (num_tokens=352 avail_mem=38.74 GB):  55%|█████▌    | 32/58 [00:04<00:02,  9.54it/s]Capturing num tokens (num_tokens=352 avail_mem=38.74 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.88it/s]Capturing num tokens (num_tokens=320 avail_mem=38.84 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.88it/s]

    Capturing num tokens (num_tokens=288 avail_mem=38.84 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.88it/s]Capturing num tokens (num_tokens=288 avail_mem=38.84 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.54it/s]Capturing num tokens (num_tokens=256 avail_mem=38.75 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.54it/s]Capturing num tokens (num_tokens=240 avail_mem=38.83 GB):  62%|██████▏   | 36/58 [00:04<00:01, 11.54it/s]Capturing num tokens (num_tokens=240 avail_mem=38.83 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.87it/s]Capturing num tokens (num_tokens=224 avail_mem=38.80 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.87it/s]

    Capturing num tokens (num_tokens=208 avail_mem=38.77 GB):  66%|██████▌   | 38/58 [00:05<00:01, 12.87it/s]Capturing num tokens (num_tokens=208 avail_mem=38.77 GB):  69%|██████▉   | 40/58 [00:05<00:01, 11.23it/s]Capturing num tokens (num_tokens=192 avail_mem=38.79 GB):  69%|██████▉   | 40/58 [00:05<00:01, 11.23it/s]

    Capturing num tokens (num_tokens=176 avail_mem=38.78 GB):  69%|██████▉   | 40/58 [00:05<00:01, 11.23it/s]Capturing num tokens (num_tokens=176 avail_mem=38.78 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.96it/s]Capturing num tokens (num_tokens=160 avail_mem=38.78 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.96it/s]Capturing num tokens (num_tokens=144 avail_mem=38.77 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.96it/s]

    Capturing num tokens (num_tokens=144 avail_mem=38.77 GB):  76%|███████▌  | 44/58 [00:05<00:01, 10.87it/s]Capturing num tokens (num_tokens=128 avail_mem=38.76 GB):  76%|███████▌  | 44/58 [00:05<00:01, 10.87it/s]Capturing num tokens (num_tokens=112 avail_mem=38.76 GB):  76%|███████▌  | 44/58 [00:05<00:01, 10.87it/s]Capturing num tokens (num_tokens=112 avail_mem=38.76 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.14it/s]Capturing num tokens (num_tokens=96 avail_mem=38.76 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.14it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=38.75 GB):  79%|███████▉  | 46/58 [00:05<00:01, 11.14it/s]Capturing num tokens (num_tokens=80 avail_mem=38.75 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.72it/s]Capturing num tokens (num_tokens=64 avail_mem=38.74 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.72it/s]Capturing num tokens (num_tokens=48 avail_mem=38.74 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.72it/s]

    Capturing num tokens (num_tokens=48 avail_mem=38.74 GB):  86%|████████▌ | 50/58 [00:06<00:00, 11.76it/s]Capturing num tokens (num_tokens=32 avail_mem=38.73 GB):  86%|████████▌ | 50/58 [00:06<00:00, 11.76it/s]Capturing num tokens (num_tokens=28 avail_mem=38.73 GB):  86%|████████▌ | 50/58 [00:06<00:00, 11.76it/s]Capturing num tokens (num_tokens=28 avail_mem=38.73 GB):  90%|████████▉ | 52/58 [00:06<00:00, 12.30it/s]Capturing num tokens (num_tokens=24 avail_mem=38.72 GB):  90%|████████▉ | 52/58 [00:06<00:00, 12.30it/s]

    Capturing num tokens (num_tokens=20 avail_mem=38.71 GB):  90%|████████▉ | 52/58 [00:06<00:00, 12.30it/s]Capturing num tokens (num_tokens=20 avail_mem=38.71 GB):  93%|█████████▎| 54/58 [00:06<00:00, 12.37it/s]Capturing num tokens (num_tokens=16 avail_mem=38.71 GB):  93%|█████████▎| 54/58 [00:06<00:00, 12.37it/s]Capturing num tokens (num_tokens=12 avail_mem=38.67 GB):  93%|█████████▎| 54/58 [00:06<00:00, 12.37it/s]

    Capturing num tokens (num_tokens=12 avail_mem=38.67 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.56it/s]Capturing num tokens (num_tokens=8 avail_mem=38.70 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.56it/s] Capturing num tokens (num_tokens=4 avail_mem=38.69 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.56it/s]Capturing num tokens (num_tokens=4 avail_mem=38.69 GB): 100%|██████████| 58/58 [00:06<00:00, 12.70it/s]Capturing num tokens (num_tokens=4 avail_mem=38.69 GB): 100%|██████████| 58/58 [00:06<00:00,  8.74it/s]


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
    Generated text:  Yuki, I'm an AI language model created by Alibaba Cloud. I'm a language model trained on over 100 million Chinese characters, and I'm fully bilingual in Chinese and English. Can you tell me a joke? Of course! Here's a joke for you: Why did the tomato turn red? Because it saw the salad dressing! 
    
    That's all from me, so don't ask me to make any jokes. How about we play a game of tag instead? 
    
    (Answers will be announced after the game.) How about a game of tag? Let's start! Which team will tag the other? Which tag
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. Which of the following is NOT a reason why the president's term is shorter than that of a general in the U.S. military? ① He is not a commander-in-chief ② He does not serve as a member of the U.S. military's highest decision-making body ③ He does not have to serve in the military ④ He is not responsible for military operations
    A. ①②③④
    B. ①②③
    C. ②③④
    D. ①③④
    Answer
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Paris is the capital of France, and the second largest city in the European Union. It is located on the right bank of the Seine river, in the southeastern region of the country. It is known as the "City of Love" due to its romantic architecture, many museums, and landmarks such as Notre Dame Cathedral, the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city also has a rich history and culture, and is home to many famous landmarks and attractions. Paris is a major urban area that includes many other smaller cities and towns, as well as the city of Toulouse
    ===============================
    Prompt: The future of AI is
    Generated text:  being shaped by multiple factors, including economic, political, technological, and social factors. Economic factors include access to resources, the demand for technology, and the need for innovation. Political factors include the laws and regulations surrounding the use of AI, and the political climate of the region. Technological factors include the pace of development, the availability of technology, and the costs of development. Social factors include the level of education, the availability of data, and the level of trust in technology. It is important to consider all these factors when evaluating the future of AI. 
    
    The impact of AI on society is a complex issue. While AI has the


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


    Generated text:  Paris. It is the largest city in the country and is home to the Eiffel Tower, the Louvre Museum, and many other famous landmarks. Paris is known for its rich history, art, and culture, and is a popular tourist destination. It is also home to many important institutions such as the French Academy of Sciences and the French Parliament. The city is known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to attract visitors from around the world. The city is also home to many important political and economic institutions,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field by improving patient outcomes and reducing costs.
    
    3. Greater use of AI in manufacturing: AI is already being used in manufacturing to optimize production processes and improve quality control. As AI technology continues to improve, it is likely to be used even more extensively in manufacturing
    


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
    Generated text:  [Name] and I am a dedicated and dedicated professional with over [number] years of experience in the field of [Title/industry]. I am passionate about [specific skill or area of expertise] and am always eager to learn and grow. My approach to work is pragmatic, efficient, and result-oriented. I am always ready to collaborate with others and excel in my field, and I am committed to delivering exceptional service and results. I am highly organized, detail-oriented, and have a strong work ethic. I am passionate about [specific career goal or area of interest] and I am committed to achieving it. I am always looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France. It serves as the administrative, political, and cultural center of the country. Paris is renowned for its rich history, diverse culture, and world-renowned art and music scenes. The city is also known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. Paris is one of the largest and most populous cities in the world, with a population of over 18 million people. The city is a major transportation hub and a significant economic center in Europe. Paris is home to many of France's famous landmarks, including the Palace of Vers
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and unpredictable, with numerous possibilities and potential applications. Here are some possible future trends in artificial intelligence:
    
    1. Increased AI autonomy: As AI systems become more capable, we can expect to see an increase in AI autonomy. For example, we may see more autonomous vehicles, robots that can make decisions based on ethical and moral standards, and AI that can communicate with humans and even develop empathy.
    
    2. Improved AI ethics: With more and more AI systems becoming autonomous, it's essential that we establish ethical standards for AI. This means that we need to ensure that AI systems are developed and used in a way that is beneficial for society as


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

     a

     friendly

    ,

     kind

    -hearted

     individual

     who

     loves

     to

     help

     others

     and

     share

     their

     stories

     with

     the

     world

    .

     I

     have

     an

     impressive

     background

     in

     creative

     writing

    ,

     with

     a

     passion

     for

     exploring

     different

     genres

     and

     writing

     in

     genres

     that

     would

     attract

     different

     audiences

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     as

     a

     writer

     and

     to

     help

     people

    .

     I

     believe

     that

     all

     people

     are

     worthy

     of

     happiness

     and

     success

    ,

     and

     I

    'm

     committed

     to

     using

     my

     knowledge

     and

     skills

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     confident

     that

     my

     experiences

     and

     expertise

     will

     help

     me

     to

     make

     a

     difference

     in

     the

     lives

     of

     others

     and

     I

    'm

     excited

     to

     share

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     which

     has

     a

     history

     of

     

    6

    5

    0

     years

     and

     is

     home

     to

     numerous

     iconic

     landmarks

    ,

     including

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

    .

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     that

     is

     known

     for

     its

     artistic

    ,

     cultural

    ,

     and

     culinary

     scenes

    ,

     as

     well

     as

     its

     important

     role

     in

     French

     politics

     and

     its

     status

     as

     a

     major

     global

     financial

     center

    .

     The

     city

     is

     also

     home

     to

     a

     large

     and

     diverse

     population

    ,

     making

     it

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .

     
    


    Paris

     is

     a

     city

     that

     is

     not

     only

     beautiful

     and

     charming

    ,

     but

     also

     full

     of

     energy

     and

     excitement

    ,

     making

     it

     an

     exciting

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     complex

     and

     evolving

    ,

     with

     several

     trends

     shaping

     its

     development

     and

     applications

    .

     Here

     are

     some

     potential

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     Automation

     and

     Personal

    ization

    :

     As

     AI

     continues

     to

     become

     more

     advanced

    ,

     we

     can

     expect

     to

     see

     even

     more

     automation

     in

     many

     areas

    ,

     from

     manufacturing

     and

     transportation

     to

     customer

     service

     and

     healthcare

    .

     AI

    -powered

     personal

     assistants

     and

     virtual

     assistants

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     making

     them

     easier

     to

     use

     and

     more

     personalized

    .
    


    2

    .

     Artificial

     Intelligence

     and

     Eth

    ical

     Consider

    ations

    :

     AI

     will

     continue

     to

     evolve

     in

     ways

     that

     raise

     ethical

     questions

    .

     There

     will

     be

     increasing

     pressure

     to

     develop

     AI

     that

     is

     transparent

    ,

     accountable

    ,

     and

     responsible

     for

     its

     actions

    .

    



```python
llm.shutdown()
```
