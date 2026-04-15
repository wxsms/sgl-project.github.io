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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.77it/s]


    2026-04-15 07:18:51,531 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 07:18:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:03,  1.14s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.15it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:14,  3.68it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:14,  3.68it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.68it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.68it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.44it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.44it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 12.66it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 17.18it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 17.18it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 17.18it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 17.18it/s] 

    Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 17.18it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 21.26it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 21.26it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 21.26it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 21.26it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 21.26it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 24.62it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 24.62it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 24.62it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 24.62it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 24.62it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:04,  6.61it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:04,  6.61it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:04,  6.61it/s]

    Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:04,  6.61it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:02,  8.23it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:02,  8.23it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:02,  8.23it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:02,  8.23it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:02,  9.90it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:02,  9.90it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:02,  9.90it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:02,  9.90it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:01, 11.93it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:01, 11.93it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:01, 11.93it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:01, 11.93it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:01, 13.99it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:01, 13.99it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:01, 13.99it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:01, 13.99it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 15.81it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 15.81it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 15.81it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 15.81it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 17.84it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 17.84it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 17.84it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 17.84it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 19.62it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 19.62it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 19.62it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 19.62it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 21.53it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 21.53it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 21.53it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 21.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.98 GB):   2%|▏         | 1/58 [00:00<00:13,  4.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.95 GB):   2%|▏         | 1/58 [00:00<00:13,  4.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.95 GB):   3%|▎         | 2/58 [00:00<00:13,  4.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.95 GB):   3%|▎         | 2/58 [00:00<00:13,  4.28it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.95 GB):   5%|▌         | 3/58 [00:00<00:12,  4.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.94 GB):   5%|▌         | 3/58 [00:00<00:12,  4.38it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.94 GB):   7%|▋         | 4/58 [00:00<00:12,  4.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.95 GB):   7%|▋         | 4/58 [00:00<00:12,  4.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=38.95 GB):   9%|▊         | 5/58 [00:01<00:12,  4.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.94 GB):   9%|▊         | 5/58 [00:01<00:12,  4.41it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.94 GB):  10%|█         | 6/58 [00:01<00:11,  4.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.94 GB):  10%|█         | 6/58 [00:01<00:11,  4.55it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.94 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.92 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.92 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.92 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.91it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.92 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.91 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.91 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.91 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.28it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.91 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.91 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.91 GB):  21%|██        | 12/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.90 GB):  21%|██        | 12/58 [00:02<00:08,  5.60it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.90 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.90 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.90 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.90 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.80it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.90 GB):  26%|██▌       | 15/58 [00:02<00:07,  5.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.90 GB):  26%|██▌       | 15/58 [00:02<00:07,  5.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.90 GB):  28%|██▊       | 16/58 [00:03<00:06,  6.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.89 GB):  28%|██▊       | 16/58 [00:03<00:06,  6.02it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.89 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.89 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.89 GB):  31%|███       | 18/58 [00:03<00:06,  6.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.89 GB):  31%|███       | 18/58 [00:03<00:06,  6.23it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=38.89 GB):  33%|███▎      | 19/58 [00:03<00:06,  6.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.88 GB):  33%|███▎      | 19/58 [00:03<00:06,  6.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.88 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.86 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.36it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.86 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.51it/s]Capturing num tokens (num_tokens=960 avail_mem=38.87 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.51it/s] Capturing num tokens (num_tokens=960 avail_mem=38.87 GB):  38%|███▊      | 22/58 [00:03<00:05,  6.62it/s]Capturing num tokens (num_tokens=896 avail_mem=38.87 GB):  38%|███▊      | 22/58 [00:03<00:05,  6.62it/s]

    Capturing num tokens (num_tokens=896 avail_mem=38.87 GB):  40%|███▉      | 23/58 [00:04<00:05,  6.75it/s]Capturing num tokens (num_tokens=832 avail_mem=38.87 GB):  40%|███▉      | 23/58 [00:04<00:05,  6.75it/s]Capturing num tokens (num_tokens=832 avail_mem=38.87 GB):  41%|████▏     | 24/58 [00:04<00:05,  6.79it/s]Capturing num tokens (num_tokens=768 avail_mem=38.86 GB):  41%|████▏     | 24/58 [00:04<00:05,  6.79it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.14it/s]Capturing num tokens (num_tokens=704 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.14it/s]Capturing num tokens (num_tokens=704 avail_mem=38.86 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.07it/s]Capturing num tokens (num_tokens=640 avail_mem=38.86 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.07it/s]

    Capturing num tokens (num_tokens=640 avail_mem=38.86 GB):  47%|████▋     | 27/58 [00:04<00:04,  7.27it/s]Capturing num tokens (num_tokens=576 avail_mem=38.86 GB):  47%|████▋     | 27/58 [00:04<00:04,  7.27it/s]Capturing num tokens (num_tokens=576 avail_mem=38.86 GB):  48%|████▊     | 28/58 [00:04<00:04,  7.41it/s]Capturing num tokens (num_tokens=512 avail_mem=38.85 GB):  48%|████▊     | 28/58 [00:04<00:04,  7.41it/s]

    Capturing num tokens (num_tokens=512 avail_mem=38.85 GB):  50%|█████     | 29/58 [00:04<00:03,  7.41it/s]Capturing num tokens (num_tokens=480 avail_mem=38.86 GB):  50%|█████     | 29/58 [00:04<00:03,  7.41it/s]Capturing num tokens (num_tokens=480 avail_mem=38.86 GB):  52%|█████▏    | 30/58 [00:05<00:03,  7.79it/s]Capturing num tokens (num_tokens=448 avail_mem=38.86 GB):  52%|█████▏    | 30/58 [00:05<00:03,  7.79it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.86 GB):  53%|█████▎    | 31/58 [00:05<00:03,  7.93it/s]Capturing num tokens (num_tokens=416 avail_mem=38.86 GB):  53%|█████▎    | 31/58 [00:05<00:03,  7.93it/s]Capturing num tokens (num_tokens=416 avail_mem=38.86 GB):  55%|█████▌    | 32/58 [00:05<00:03,  6.84it/s]Capturing num tokens (num_tokens=384 avail_mem=38.86 GB):  55%|█████▌    | 32/58 [00:05<00:03,  6.84it/s]

    Capturing num tokens (num_tokens=352 avail_mem=38.85 GB):  55%|█████▌    | 32/58 [00:05<00:03,  6.84it/s]Capturing num tokens (num_tokens=352 avail_mem=38.85 GB):  59%|█████▊    | 34/58 [00:05<00:02,  9.32it/s]Capturing num tokens (num_tokens=320 avail_mem=38.85 GB):  59%|█████▊    | 34/58 [00:05<00:02,  9.32it/s]Capturing num tokens (num_tokens=288 avail_mem=38.84 GB):  59%|█████▊    | 34/58 [00:05<00:02,  9.32it/s]Capturing num tokens (num_tokens=256 avail_mem=38.84 GB):  59%|█████▊    | 34/58 [00:05<00:02,  9.32it/s]Capturing num tokens (num_tokens=240 avail_mem=38.84 GB):  59%|█████▊    | 34/58 [00:05<00:02,  9.32it/s]Capturing num tokens (num_tokens=240 avail_mem=38.84 GB):  66%|██████▌   | 38/58 [00:05<00:01, 15.66it/s]Capturing num tokens (num_tokens=224 avail_mem=38.83 GB):  66%|██████▌   | 38/58 [00:05<00:01, 15.66it/s]Capturing num tokens (num_tokens=208 avail_mem=38.83 GB):  66%|██████▌   | 38/58 [00:05<00:01, 15.66it/s]

    Capturing num tokens (num_tokens=192 avail_mem=38.83 GB):  66%|██████▌   | 38/58 [00:05<00:01, 15.66it/s]Capturing num tokens (num_tokens=176 avail_mem=38.82 GB):  66%|██████▌   | 38/58 [00:05<00:01, 15.66it/s]Capturing num tokens (num_tokens=176 avail_mem=38.82 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.67it/s]Capturing num tokens (num_tokens=160 avail_mem=38.82 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.67it/s]Capturing num tokens (num_tokens=144 avail_mem=38.81 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.67it/s]Capturing num tokens (num_tokens=128 avail_mem=38.81 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.67it/s]Capturing num tokens (num_tokens=112 avail_mem=38.81 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.67it/s]Capturing num tokens (num_tokens=112 avail_mem=38.81 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s]Capturing num tokens (num_tokens=96 avail_mem=38.81 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s] Capturing num tokens (num_tokens=80 avail_mem=38.80 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s]

    Capturing num tokens (num_tokens=64 avail_mem=38.80 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s]Capturing num tokens (num_tokens=48 avail_mem=38.80 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s]Capturing num tokens (num_tokens=32 avail_mem=38.79 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s]Capturing num tokens (num_tokens=28 avail_mem=38.79 GB):  79%|███████▉  | 46/58 [00:05<00:00, 24.27it/s]Capturing num tokens (num_tokens=28 avail_mem=38.79 GB):  90%|████████▉ | 52/58 [00:05<00:00, 31.79it/s]Capturing num tokens (num_tokens=24 avail_mem=38.78 GB):  90%|████████▉ | 52/58 [00:05<00:00, 31.79it/s]Capturing num tokens (num_tokens=20 avail_mem=38.78 GB):  90%|████████▉ | 52/58 [00:05<00:00, 31.79it/s]Capturing num tokens (num_tokens=16 avail_mem=38.78 GB):  90%|████████▉ | 52/58 [00:06<00:00, 31.79it/s]Capturing num tokens (num_tokens=12 avail_mem=38.77 GB):  90%|████████▉ | 52/58 [00:06<00:00, 31.79it/s]Capturing num tokens (num_tokens=8 avail_mem=38.77 GB):  90%|████████▉ | 52/58 [00:06<00:00, 31.79it/s] Capturing num tokens (num_tokens=4 avail_mem=38.77 GB):  90%|████████▉ | 52/58 [00:06<00:00, 31.79it/s]Capturing num tokens (num_tokens=4 avail_mem=38.77 GB): 100%|██████████| 58/58 [00:06<00:00, 37.46it/s]Capturing num tokens (num_tokens=4 avail_mem=38.77 GB): 100%|██████████| 58/58 [00:06<00:00,  9.54it/s]


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
    Generated text:  Natalia and I am a bit of a public speaker and creative writing professional. I have a degree in journalism, and I am currently in my sixth year of working with the Hearst Corporation. I have covered and written extensively on topics ranging from politics and history to entertainment and science.
    
    As a young woman, I had a lot of the same concerns and struggles that many people face today. I felt like I didn't fit in with my peers and I didn't know what I wanted to do with my life. I used to be very shy and I wasn't very outgoing. Now I feel like I have a strong support system, both
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office, and the person in charge of the country's affairs and the highest official of the United States. Is this statement true or false? ____
    A. True
    B. False
    Answer: B
    
    When the generator stops running, it should be powered off immediately to prevent fire accidents.
    A. Correct
    B. Incorrect
    Answer: A
    
    When using a megohmmeter to measure insulation resistance, the equipment under test should be disconnected from all power sources, and the circuit breaker of the equipment under test should be opened, and the equipment under test should be discharged.
    A. Correct
    B. Incorrect
    Answer
    ===============================
    Prompt: The capital of France is
    Generated text:  ( )
    
    A: Paris  
    B: Lyon  
    C: Marseille  
    D: Nice
    To determine the capital of France, let's go through the options step by step:
    
    1. **Paris**: This is the capital of France, located in the south of the country.
    2. **Lyon**: This is a city in northeastern France, not the capital.
    3. **Marseille**: This is a city in the north of France, not the capital.
    4. **Nice**: This is a city in the Loire Valley, not the capital.
    
    Based on this analysis, the capital of France is Paris.
    
    Therefore, the correct
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising. It is a technology that has the potential to revolutionize various fields, including healthcare, finance, transportation, and education. However, with the rapid advancement of technology, it is essential to ensure that the AI systems being developed are safe and ethical. In this blog post, we will explore the future of AI and how it can be implemented in different industries.
    Firstly, in healthcare, the use of AI in medical imaging has the potential to revolutionize the field. AI can analyze medical images such as X-rays, MRIs, and CT scans to detect diseases and identify patterns that may be difficult to identify by human eyes.


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I'm [Favorite Color] and I love [Favorite Activity]. I'm [Favorite Book] and I enjoy [Favorite Food]. I'm [Favorite Movie] and I love [Favorite Music]. I'm [Favorite Sport] and I play [Favorite Game]. I'm [Favorite Place] and I love [Favorite Hobby]. I'm [Favorite Animal] and I have [Number of Pets] pets. I'm [Favorite Book] and I enjoy [Favorite Food]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and diverse culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 10 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, music, and fashion, and is home to many world-renowned museums, theaters, and restaurants. Paris is a popular tourist destination and a major economic center in France. The city is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust AI systems that are designed to be transparent, accountable, and
    


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
    Generated text:  [Your Name], and I am a [position] at [Company]. I am passionate about [occupation], and I enjoy [job responsibilities]. I am confident in my abilities, and I am always looking for opportunities to improve myself and grow as a professional. How would you describe your character? [Short, neutral self-introduction] Hi there, my name is [Your Name], and I am a [position] at [Company]. I am passionate about [occupation], and I enjoy [job responsibilities]. I am confident in my abilities, and I am always looking for opportunities to improve myself and grow as a professional. How would you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is also the largest city in the country, with a population of over 2.3 million. Paris is known for its rich history and cultural heritage, including the Eiffel Tower, Notre-Dame Cathedral, and Montmartre. The city is also renowned for its fashion industry, food culture, and art scene. Additionally, Paris is a popular tourist destination, with millions of visitors each year. The city is known for its iconic landmarks, including the Louvre Museum and Notre Dame Cathedral, and its food and beverage scene, which includes the famous "plum" pastry. Overall, Paris is a city that offers
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends, including:
    
    1. Increased focus on ethical AI: As more people become aware of the potential negative impacts of AI, there will be a greater focus on developing AI that is more ethical and aligned with human values.
    
    2. AI will become more capable of self-learning and problem-solving: As AI becomes more capable of self-learning and problem-solving, it will be able to adapt to new situations and provide better results than ever before.
    
    3. AI will become more integrated with natural language processing: As AI becomes more integrated with natural language processing, it will be able to better understand and respond to human language


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

    Job

     Title

    ]

     with

     over

     [

    number

     of

     years

     of

     experience

    ].

     I

    'm

     excited

     to

     be

     here

     and

     to

     continue

     to

     make

     a

     positive

     impact

     on

     [

    field

     or

     industry

    ].

     If

     you

     have

     any

     questions

     or

     would

     like

     to

     learn

     more

     about

     me

    ,

     feel

     free

     to

     ask

     me

     anything

    .

     I

    'm

     always

     here

     to

     help

    .

     
    


    [

    Name

    ],

     the

     new

     employee

    ,

     is

     looking

     forward

     to

     learning

     and

     growing

     in

     their

     role

    ,

     and

     I

    'd

     be

     thrilled

     to

     share

     my

     background

     and

     experience

     with

     them

    .

     I

    'm

     committed

     to

     making

     a

     difference

     in

     the

     world

    ,

     and

     I

    'm

     confident

     that

     our

     combined

     efforts

     can

     create

     meaningful

     change

    .

     What

     do

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     and

     culturally

     rich

     city

     known

     for

     its

     iconic

     architecture

    ,

     charming

     streets

    ,

     and

     world

    -class

     museums

     and

     theaters

    .

     Known

     for

     its

     Paris

    ian

     cuisine

     and

     wine

    ,

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     that

     attracts

     millions

     of

     tourists

     every

     year

    .

     The

     city

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     among

     other

     historical

     and

     cultural

     landmarks

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     birth

    place

     of

     numerous

     influential

     figures

     and

     artists

     throughout

     history

    .

     Its

     unique

     blend

     of

     historical

     and

     modern

     elements

     has

     made

     it

     a

     major

     cultural

     hub

     and

     global

     capital

     of

     France

    .

     The

     city

    's

     reputation

     for

     being

     an

     exciting

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     and

     exciting

     possibilities

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     AI

     will

     continue

     to

     evolve

     and

     adapt

     to

     new

     technologies

     and

     applications

    .

     This

     means

     that

     AI

     will

     become

     even

     more

     integrated

     into

     our

     daily

     lives

     and

     work

     environments

    ,

     from

     smart

     homes

     to

     autonomous

     vehicles

    .
    


    2

    .

     AI

     will

     become

     more

     diverse

     and

     inclusive

    .

     The

     AI

     industry

     is

     increasingly

     diverse

    ,

     with

     more

     people

     from

     different

     backgrounds

     and

     cultures

     participating

     in

     the

     development

     and

     deployment

     of

     AI

     technologies

    .

     This

     will

     make

     AI

     systems

     more

     representative

     and

     inclusive

     of

     a

     wider

     range

     of

     people

    .
    


    3

    .

     AI

     will

     continue

     to

     be

     used

     for

     more

     tasks

     and

     tasks

    .

     AI

     will

     continue

     to

     be

     used

     for

     a

    



```python
llm.shutdown()
```
