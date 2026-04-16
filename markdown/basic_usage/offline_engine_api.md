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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.58it/s]


    2026-04-16 01:40:05,355 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 01:40:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:50,  3.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:50,  3.00s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:03<01:12,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:28,  1.92it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:12,  3.99it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:12,  3.99it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:12,  3.99it/s]

    Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:12,  3.99it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.48it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04, 10.24it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04, 10.24it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04, 10.24it/s]

    Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04, 10.24it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04, 10.24it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 14.33it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 14.33it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 18.39it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 18.39it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 18.39it/s]

    Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 18.39it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 18.39it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 22.43it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 22.43it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 22.43it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 22.43it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:01, 22.43it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 25.17it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 25.17it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 25.17it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 25.17it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 25.17it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:00, 28.11it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:00, 28.11it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:00, 28.11it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:00, 28.11it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:00, 28.11it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 29.11it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 29.11it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 29.11it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 29.11it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 29.11it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 30.95it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 30.95it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 30.95it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 30.95it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 30.95it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 31.45it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 31.45it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 31.45it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 31.45it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 31.45it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.28it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.28it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.28it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.28it/s]

    Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.28it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.28it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 35.29it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 35.29it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 35.29it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 35.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:07,  7.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:07,  7.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:06,  8.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:06,  8.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.21it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:08,  6.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:08,  6.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.89it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.73it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.82it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.37it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.38 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.05it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.47it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.47it/s] Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.47it/s]Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.27it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.27it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.27it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.99it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.99it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.99it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.89it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.89it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.89it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  50%|█████     | 29/58 [00:02<00:02, 12.64it/s]Capturing num tokens (num_tokens=480 avail_mem=58.35 GB):  50%|█████     | 29/58 [00:02<00:02, 12.64it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:02, 12.64it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:02, 12.38it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:02, 12.38it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:03<00:02, 12.38it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.36it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.36it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.36it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.36it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.92it/s]Capturing num tokens (num_tokens=256 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.92it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.92it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.92it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.00it/s]Capturing num tokens (num_tokens=208 avail_mem=58.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.00it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 17.00it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  71%|███████   | 41/58 [00:03<00:01, 15.50it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:01, 15.50it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:01, 15.50it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  74%|███████▍  | 43/58 [00:03<00:01, 12.08it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:01, 12.08it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:01, 12.08it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 11.25it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 11.25it/s]Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 11.25it/s] Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:04<00:00, 11.51it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:04<00:00, 11.51it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:04<00:00, 11.51it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:04<00:00, 11.76it/s]Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:04<00:00, 11.76it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:04<00:00, 11.76it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.82it/s]Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.82it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:04<00:00, 11.82it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 13.26it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 13.26it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 13.26it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.17it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.17it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.17it/s] Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 14.17it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:04<00:00, 16.59it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:04<00:00, 11.93it/s]


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
    Generated text:  Mary. I'm 12 years old. I'm in Class 3, Grade 6. I have a great friend. His name is Jack. We go to school together every day. He is very busy. He has no time for me. One day, he told me, "I have a lot of homework to do this week. I want you to help me with it. I'm not good at maths. And I can't finish the homework by the end of the week. " Mary said, "Yes. I will help you. Please tell me how I can help you. " Jack thought about it for
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by ______ people.
    A. all the people of the state
    B. all the people in the country
    C. all the people of the United States
    D. all the people in the state
    Answer: B
    
    Which of the following statements about public relations is incorrect?
    A. Public relations is a type of information dissemination
    B. Public relations is a type of public welfare activity
    C. Public relations is a type of marketing activity
    D. Public relations is a type of promotion activity
    Answer: C
    
    Which of the following statements about public relations is incorrect?
    A. Public relations is a type of information dissemination
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is located in the ____ of France.
    A. center
    B. south
    C. north
    D. east
    
    To determine the correct answer, let's analyze the information step by step:
    
    1. The capital of France is Paris.
    2. The capital of France is located in the center of France.
    
    Based on this information, we can conclude that the capital of France is located in the center of France.
    
    Therefore, the correct answer is: A. center
    
    So, the final answer is: A. center.
    ===============================
    Prompt: The future of AI is
    Generated text:  intrinsically linked to the understanding of neuroscience. The brain is the source of our thoughts and emotions, and it is an incredible resource for developing computer algorithms that can create a realistic simulation of the brain. Researchers are working on developing more advanced algorithms and methodologies that can accurately predict brain activity and behavior in real-time, which will allow us to better understand how our brains work and how we can use this knowledge to develop new and more effective treatments for neurological disorders.
    Neuroscience is a fascinating and complex field, and it requires a deep understanding of how the brain works to develop effective algorithms. AI researchers must be able to translate their knowledge of


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], [Experience], [Education], [Skills], and [Personal Traits]. I'm always looking for new challenges and opportunities to grow and learn. Thank you for taking the time to meet me. [Name] [Company Name] [Company Address] [Company Phone Number] [Company Email] [Company Website] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major transportation hub, with many international airports and train stations. The city is home to many famous French artists, writers, and musicians, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city of people, with a diverse population of over
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI continues to improve, we can expect to see even more integration into our daily routines.
    
    2. AI becoming more autonomous: As AI becomes more advanced, we can expect to see more autonomous vehicles on the roads, with AI taking over many of the tasks that humans currently perform. This could lead to a more efficient and safer transportation system
    


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
    Generated text:  [Name] and I am an experienced [field] professional with [number of years] years of experience in [related field]. Currently, I am [name of current company], working as [position] in [location]. I am passionate about [career goal or interest]. I am [born in] and [place of birth], and I am [age]. I believe in [value system or personal beliefs]. I value [what] most, and I'm always looking to [growth mindset or motivation]. I am [any other personal traits or qualities].
    [Name] is a [type of professional]. I bring with me a blend
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To double-check this information, I'll consider the following points:
    1. How many people live in Paris?
    2. Are there any famous landmarks in Paris?
    3. How long has Paris been a capital city?
    4. What is the official language of Paris?
    
    Taking these points into account, here's a concise statement about France's capital city:
    
    Paris is the capital city of France and is home to approximately 2.1 million people.
    
    Additional information could include:
    - Paris is the oldest continuously inhabited capital city in Europe.
    - It is known for its iconic Notre Dame Cathedral, Louvre Museum, and Seine River
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking promising, with a number of possible trends that could shape the technology and impact on society. Here are some of the most likely future trends in artificial intelligence:
    
    1. Increased specialization and focus: AI is becoming more specialized, with more focus on specific applications such as medical diagnosis, financial prediction, and cybersecurity. This will lead to more tailored and efficient AI systems that can handle a wider range of tasks.
    
    2. Autonomous vehicles and drones: AI is already being used in autonomous vehicles, which can help reduce traffic congestion and air pollution. As AI technology continues to improve, we may see more widespread adoption of autonomous vehicles in the future.
    
    3


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

     __

    ________

     and

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     excited

     to

     be

     here

     and

     to

     be

     part

     of

     this

     team

    .

     Let

    's

     discuss

     __

    ________

     and

     see

     where

     it

     takes

     us

    .

     I

    'm

     up

     for

     the

     challenge

     and

     I

    'm

     __

    ________

    .

     Let

    's

     get

     started

     and

     make

     the

     most

     of

     our

     time

     together

    !

     What

     do

     you

     think

    ?

     


    Option

     A

    :

     


    Option

     B

    :


    Choose

     the

     option

     that

     best

     fits

     the

     character

    's

     personality

     and

     style

    .

     


    As

     a

     fictional

     character

    ,

     it

     is

     likely

     that

     the

     chosen

     option

     will

     reflect

     the

     characteristics

     of

     the

     person

     being

     introduced

    .

     For

     example

    ,

     if

     the

     character

     is

     outgoing

    ,

     a

     positive

    ,

     and

     energetic

    ,

     the

     chosen

     option

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     and

     it

     is

     the

     largest

     city

     in

     the

     country

    .

     It

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     and

     is

     home

     to

     many

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

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     home

     to

     many

     beloved

     artists

    ,

     writers

    ,

     and

     musicians

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     history

    ,

     including

     the

     fall

     of

     the

     Bast

    ille

     and

     the

     opening

     of

     the

     World

    ’s

     Fair

     in

     

    1

    8

    8

    9

    .

     Paris

     is

     a

     vibrant

     and

     dynamic

     city

     with

     a

     diverse

     population

     and

     a

     thriving

     economy

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     encompasses

     a

     wide

     range

     of

     trends

     and

     areas

     of

     interest

    .

     Some

     of

     the

     most

     common

     trends

     in

     AI

     include

    :
    


    1

    .

     More

     specialized

     AI

    :

     AI

     systems

     are

     becoming

     more

     specialized

     and

     focused

     on

     specific

     tasks

    .

     This

     trend

     will

     allow

     for

     greater

     efficiency

     and

     accuracy

     in

     applications

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     financial

     services

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     agriculture

    :

     AI

     is

     being

     used

     to

     optimize

     crop

     production

    ,

     reduce

     costs

    ,

     and

     improve

     crop

     yields

    .

     This

     trend

     will

     likely

     continue

     to

     grow

     as

     AI

     systems

     are

     integrated

     into

     existing

     agricultural

     practices

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     healthcare

    ,

     with

     applications

     ranging

     from

     personalized

     medicine

     to

    



```python
llm.shutdown()
```
