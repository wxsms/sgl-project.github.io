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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.90it/s]


    2026-05-14 09:42:07,009 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 09:42:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:47,  1.92s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:47,  1.92s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:04<01:47,  1.92s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:41,  1.30it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:41,  1.30it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:41,  1.30it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:14,  3.51it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:14,  3.51it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:14,  3.51it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:14,  3.51it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:08,  5.77it/s]

    Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:08,  5.77it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.99it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.99it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 17.17it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 17.17it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 17.17it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 17.17it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 17.17it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 19.95it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 19.95it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 19.95it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 19.95it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 19.95it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 22.62it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 22.62it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 22.62it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 22.62it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 22.62it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 22.62it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 27.23it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 27.23it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 27.23it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 27.23it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:00, 27.23it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:00, 27.23it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 32.14it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 32.14it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 32.14it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 32.14it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 32.14it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 32.14it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 36.09it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 36.09it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 36.09it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 36.09it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 36.09it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 36.09it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 34.56it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 34.56it/s]

    Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 34.56it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 34.56it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 34.56it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 32.48it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 32.48it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 32.48it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 32.48it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=39.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=39.94 GB):   2%|▏         | 1/58 [00:00<00:13,  4.18it/s]Capturing num tokens (num_tokens=7680 avail_mem=39.91 GB):   2%|▏         | 1/58 [00:00<00:13,  4.18it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=39.91 GB):   3%|▎         | 2/58 [00:00<00:13,  4.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.89 GB):   3%|▎         | 2/58 [00:00<00:13,  4.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.89 GB):   5%|▌         | 3/58 [00:00<00:12,  4.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=39.97 GB):   5%|▌         | 3/58 [00:00<00:12,  4.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=39.97 GB):   7%|▋         | 4/58 [00:00<00:11,  4.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=39.97 GB):   7%|▋         | 4/58 [00:00<00:11,  4.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=39.97 GB):   9%|▊         | 5/58 [00:01<00:11,  4.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=39.96 GB):   9%|▊         | 5/58 [00:01<00:11,  4.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=39.96 GB):  10%|█         | 6/58 [00:01<00:10,  4.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.87 GB):  10%|█         | 6/58 [00:01<00:10,  4.98it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=40.87 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.01 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.01 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.01 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.19it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=40.01 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.86 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.86 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.07 GB):  17%|█▋        | 10/58 [00:01<00:08,  5.61it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=40.07 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.07 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.07 GB):  21%|██        | 12/58 [00:02<00:07,  5.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.85 GB):  21%|██        | 12/58 [00:02<00:07,  5.89it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=40.85 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.13 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.13 GB):  24%|██▍       | 14/58 [00:02<00:07,  6.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.12 GB):  24%|██▍       | 14/58 [00:02<00:07,  6.03it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=40.12 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.84 GB):  26%|██▌       | 15/58 [00:02<00:06,  6.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.84 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.18 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.48it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=40.18 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.17 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.17 GB):  31%|███       | 18/58 [00:03<00:05,  6.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.83 GB):  31%|███       | 18/58 [00:03<00:05,  6.85it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=40.83 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.24 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.24 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.81 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.88it/s]Capturing num tokens (num_tokens=960 avail_mem=40.30 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.88it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=40.30 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.92it/s]Capturing num tokens (num_tokens=896 avail_mem=40.30 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.92it/s]Capturing num tokens (num_tokens=832 avail_mem=40.82 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.92it/s]Capturing num tokens (num_tokens=832 avail_mem=40.82 GB):  41%|████▏     | 24/58 [00:03<00:03, 11.01it/s]Capturing num tokens (num_tokens=768 avail_mem=40.32 GB):  41%|████▏     | 24/58 [00:03<00:03, 11.01it/s]Capturing num tokens (num_tokens=704 avail_mem=40.37 GB):  41%|████▏     | 24/58 [00:03<00:03, 11.01it/s]

    Capturing num tokens (num_tokens=704 avail_mem=40.37 GB):  45%|████▍     | 26/58 [00:03<00:02, 13.00it/s]Capturing num tokens (num_tokens=640 avail_mem=40.81 GB):  45%|████▍     | 26/58 [00:03<00:02, 13.00it/s]Capturing num tokens (num_tokens=576 avail_mem=40.34 GB):  45%|████▍     | 26/58 [00:03<00:02, 13.00it/s]Capturing num tokens (num_tokens=576 avail_mem=40.34 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.75it/s]Capturing num tokens (num_tokens=512 avail_mem=40.79 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.75it/s]

    Capturing num tokens (num_tokens=480 avail_mem=40.37 GB):  48%|████▊     | 28/58 [00:04<00:02, 13.75it/s]Capturing num tokens (num_tokens=480 avail_mem=40.37 GB):  52%|█████▏    | 30/58 [00:04<00:01, 14.37it/s]Capturing num tokens (num_tokens=448 avail_mem=40.80 GB):  52%|█████▏    | 30/58 [00:04<00:01, 14.37it/s]Capturing num tokens (num_tokens=416 avail_mem=40.73 GB):  52%|█████▏    | 30/58 [00:04<00:01, 14.37it/s]

    Capturing num tokens (num_tokens=416 avail_mem=40.73 GB):  55%|█████▌    | 32/58 [00:04<00:02, 12.72it/s]Capturing num tokens (num_tokens=384 avail_mem=40.40 GB):  55%|█████▌    | 32/58 [00:04<00:02, 12.72it/s]Capturing num tokens (num_tokens=352 avail_mem=40.79 GB):  55%|█████▌    | 32/58 [00:04<00:02, 12.72it/s]

    Capturing num tokens (num_tokens=352 avail_mem=40.79 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.57it/s]Capturing num tokens (num_tokens=320 avail_mem=40.78 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.57it/s]Capturing num tokens (num_tokens=288 avail_mem=40.76 GB):  59%|█████▊    | 34/58 [00:04<00:02, 10.57it/s]

    Capturing num tokens (num_tokens=288 avail_mem=40.76 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.75it/s]Capturing num tokens (num_tokens=256 avail_mem=40.78 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.75it/s]Capturing num tokens (num_tokens=240 avail_mem=40.47 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.75it/s]

    Capturing num tokens (num_tokens=240 avail_mem=40.47 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.40it/s]Capturing num tokens (num_tokens=224 avail_mem=40.77 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.40it/s]Capturing num tokens (num_tokens=208 avail_mem=40.50 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.40it/s]

    Capturing num tokens (num_tokens=208 avail_mem=40.50 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.27it/s]Capturing num tokens (num_tokens=192 avail_mem=40.76 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.27it/s]Capturing num tokens (num_tokens=192 avail_mem=40.76 GB):  71%|███████   | 41/58 [00:05<00:01,  8.99it/s]Capturing num tokens (num_tokens=176 avail_mem=40.76 GB):  71%|███████   | 41/58 [00:05<00:01,  8.99it/s]

    Capturing num tokens (num_tokens=160 avail_mem=40.75 GB):  71%|███████   | 41/58 [00:05<00:01,  8.99it/s]Capturing num tokens (num_tokens=160 avail_mem=40.75 GB):  74%|███████▍  | 43/58 [00:05<00:01,  9.18it/s]Capturing num tokens (num_tokens=144 avail_mem=40.75 GB):  74%|███████▍  | 43/58 [00:05<00:01,  9.18it/s]

    Capturing num tokens (num_tokens=144 avail_mem=40.75 GB):  76%|███████▌  | 44/58 [00:05<00:01,  8.93it/s]Capturing num tokens (num_tokens=128 avail_mem=40.73 GB):  76%|███████▌  | 44/58 [00:05<00:01,  8.93it/s]Capturing num tokens (num_tokens=128 avail_mem=40.73 GB):  78%|███████▊  | 45/58 [00:05<00:01,  9.14it/s]Capturing num tokens (num_tokens=112 avail_mem=40.60 GB):  78%|███████▊  | 45/58 [00:05<00:01,  9.14it/s]Capturing num tokens (num_tokens=96 avail_mem=40.72 GB):  78%|███████▊  | 45/58 [00:05<00:01,  9.14it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=40.72 GB):  81%|████████  | 47/58 [00:06<00:01,  9.49it/s]Capturing num tokens (num_tokens=80 avail_mem=40.72 GB):  81%|████████  | 47/58 [00:06<00:01,  9.49it/s]Capturing num tokens (num_tokens=80 avail_mem=40.72 GB):  83%|████████▎ | 48/58 [00:06<00:01,  9.46it/s]Capturing num tokens (num_tokens=64 avail_mem=40.71 GB):  83%|████████▎ | 48/58 [00:06<00:01,  9.46it/s]Capturing num tokens (num_tokens=48 avail_mem=40.59 GB):  83%|████████▎ | 48/58 [00:06<00:01,  9.46it/s]

    Capturing num tokens (num_tokens=48 avail_mem=40.59 GB):  86%|████████▌ | 50/58 [00:06<00:00,  9.82it/s]Capturing num tokens (num_tokens=32 avail_mem=40.70 GB):  86%|████████▌ | 50/58 [00:06<00:00,  9.82it/s]Capturing num tokens (num_tokens=28 avail_mem=40.69 GB):  86%|████████▌ | 50/58 [00:06<00:00,  9.82it/s]Capturing num tokens (num_tokens=28 avail_mem=40.69 GB):  90%|████████▉ | 52/58 [00:06<00:00,  9.96it/s]Capturing num tokens (num_tokens=24 avail_mem=40.68 GB):  90%|████████▉ | 52/58 [00:06<00:00,  9.96it/s]

    Capturing num tokens (num_tokens=20 avail_mem=40.61 GB):  90%|████████▉ | 52/58 [00:06<00:00,  9.96it/s]Capturing num tokens (num_tokens=20 avail_mem=40.61 GB):  93%|█████████▎| 54/58 [00:06<00:00, 10.27it/s]Capturing num tokens (num_tokens=16 avail_mem=40.67 GB):  93%|█████████▎| 54/58 [00:06<00:00, 10.27it/s]Capturing num tokens (num_tokens=12 avail_mem=40.67 GB):  93%|█████████▎| 54/58 [00:06<00:00, 10.27it/s]

    Capturing num tokens (num_tokens=12 avail_mem=40.67 GB):  97%|█████████▋| 56/58 [00:06<00:00, 10.28it/s]Capturing num tokens (num_tokens=8 avail_mem=40.66 GB):  97%|█████████▋| 56/58 [00:06<00:00, 10.28it/s] Capturing num tokens (num_tokens=4 avail_mem=40.65 GB):  97%|█████████▋| 56/58 [00:06<00:00, 10.28it/s]Capturing num tokens (num_tokens=4 avail_mem=40.65 GB): 100%|██████████| 58/58 [00:07<00:00, 10.34it/s]Capturing num tokens (num_tokens=4 avail_mem=40.65 GB): 100%|██████████| 58/58 [00:07<00:00,  8.20it/s]


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
    Generated text:  Jacob. I'm a computer science major at the University of Colorado, Boulder and I'm currently working on a project at the University's robotics lab. I'm also a part of the team that built the Turtles robot, which is a neural robot that mimics the way the brain works.
    This past year, I've been working on my own project at the University's robotics lab, where I'm trying to use machine learning to generate new neural patterns and then learn from them to create new neural patterns. I've been working on a project that is in the early stages of development, but I've already been able to generate some results
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting several cities in the Midwest. The first stop is Chicago, which is 150 miles away from Washington, D. C. If the president travels at a constant speed of 60 miles per hour and finishes the trip in 5 hours, how many hours would it take for the president to travel to the next city, which is 200 miles away from D. C. on his way to Chicago?
    To determine how long it would take for the president to travel to the next city, we need to first calculate the total distance of the trip from Washington, D. C. to Chicago.
    
    The president travels
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the region that is closest to which of the following?
    A. The Alps
    B. The Mediterranean
    C. The Atlantic
    D. The Arctic
    
    To determine which region is closest to the capital of France, we need to consider the general geographic and political location of France.
    
    1. **Alps (A)**: The Alps are a mountain range that runs through France, and they are located in the northeastern part of the country. However, the capital city of France is not located in the Alps.
    2. **Mediterranean (B)**: The Mediterranean is a sea that runs through the majority of France,
    ===============================
    Prompt: The future of AI is
    Generated text:  not in tomorrow’s new tech, but in how we use it.
    — Steve Jobs
    My favorite episode of the podcast, Exploring AI, is a video of the Parrot company with a robotic parrot named Ollie. Ollie has been making noise for over 140 days, and now it seems she’s even moving to the UK to take us on a tour of her home.
    According to T.J. Ruff, founder and CEO of Parrot, Ollie has changed the world, not only for the people who use it, but also for those who get to meet her.
    What does it


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many important museums, including the Musée d'Orsay and the Musée d'Orsay. Paris is also known for its rich history and cultural heritage, and is home to many museums, including the Louvre, the Musée d'Orsay, and the Musée de l'Orangerie. The city is also known for its food and drink scene, with many restaurants and cafes serving
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, productivity, and cost savings for businesses.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to increased regulations and standards to protect user data and prevent misuse of AI systems.
    
    3. Greater integration of
    


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
    Generated text:  [Name] and I am a [role or profession] with a passion for [job/field]. I am [age] years old and [job title] in my [job title] for [number of years]. I love [job title] because [why you love the job]. I have been [number of years] in my chosen field and I am always eager to [what you can do for the job]. I am always [how you keep yourself motivated]. I am a [interest/skill/characteristic] and I have a [realistic or ambitious goal] that I am pursuing. Thank you for asking me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Summarize in a few sentences the history of the city, including its origins, political and economic importance, and notable landmarks and attractions. Include a brief statement about the city's cultural identity and its role in French society.
    
    Certainly! The history of Paris is rich and multifaceted. It is the capital of France and one of the oldest continuously inhabited cities in the world. Paris was founded by the Romans in 753 BC as a Roman colony, which later became a French outpost under the influence of the Hellenic, Roman, and Arab occupants. In 800 AD, the city was besie
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by rapid advancements in both hardware and software, as well as new forms of AI that are both transformative and disruptive. Here are some potential trends that could be expected in the field of artificial intelligence in the coming years:
    
    1. Increased Use of AI in Healthcare: AI is increasingly being used in healthcare to improve patient outcomes and streamline diagnostic processes. For example, AI-powered diagnostic tools can analyze medical images and identify patterns that may be missed by human clinicians. AI-powered predictive analytics can also be used to inform treatment plans and disease management.
    
    2. AI in Finance: AI is also playing an increasingly important role in the finance industry


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

     [

    Age

    ].

     I

     am

     a

     [

    gender

    ]

     with

     [

    height

    ]

     and

     [

    weight

    ].

     I

     have

     [

    number

     of

     pets

    ]

     pets

     that

     I

     love

     to

     play

     with

    .

     I

     am

     [

    gender

    ]

     and

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    field

    ].

     I

     am

     [

    job

     title

    ].

     I

     am

     [

    gender

    ]

     and

     I

     love

     [

    fun

     fact

    ].

     
    


    Now

    ,

     please

     proceed

     with

     the

     question

    :

     What

     is

     your

     favorite

     hobby

    ?

     
    


    This

     is

     a

     great

     start

    ,

     but

     could

     you

     elaborate

     on

     your

     favorite

     hobby

     in

     more

     detail

    ?

     What

     kind

     of

     activities

     or

     games

     do

     you

     enjoy

     most

    ,

     and

     why

     do

     you

     find

     them

     enjoyable

    ?

     Additionally

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     "

    La

     Ville

     de

     Paris

    "

     in

     French

     and

     "

    Paris

    "

     in

     English

    .
    


    The

     capital

     of

     France

     is

     Paris

    ,

     known

     as

     "

    La

     Ville

     de

     Paris

    "

     in

     French

     and

     "

    Paris

    "

     in

     English

    .

     The

     city

     is

     a

     UNESCO

     World

     Heritage

     Site

     and

     is

     famous

     for

     its

     art

    ,

     cuisine

    ,

     and

     architecture

    .

     It

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     National

     Museum

     of

     Modern

     Art

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     also

     has

     a

     vibrant

     music

     scene

    ,

     particularly

     in

     the

     annual

     Les

     Invalid

    es

     Jazz

     Festival

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     financial

     center

     in

     Europe

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     challenges

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     emerge

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     continues

     to

     evolve

     and

     become

     more

     advanced

    ,

     it

     is

     likely

     that

     we

     will

     see

     increased

     automation

     in

     many

     different

     industries

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     resources

     and

     less

     need

     for

     human

     intervention

    .
    


    2

    .

     Development

     of

     AI

     ethics

    :

     With

     the

     growing

     number

     of

     AI

     systems

     in

     use

    ,

     it

     is

     becoming

     increasingly

     important

     to

     establish

     ethical

     guidelines

     for

     how

     AI

     systems

     should

     be

     used

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     AI

     ethics

     frameworks

     and

     regulations

    .
    


    3

    .

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     is

     a

     potential

    



```python
llm.shutdown()
```
