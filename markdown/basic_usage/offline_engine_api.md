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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.15it/s]


    2026-05-13 18:14:10,247 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 18:14:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:45,  1.89s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:45,  1.89s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:04<01:45,  1.89s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:41,  1.31it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:41,  1.31it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:41,  1.31it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:14,  3.53it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:09,  5.03it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:09,  5.03it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:09,  5.03it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:09,  5.03it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  7.66it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  7.66it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  7.66it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:05,  7.66it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:03, 10.67it/s]

    Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:02, 13.78it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:02, 13.78it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:02, 13.78it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:02, 13.78it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 16.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 16.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 16.75it/s]

    Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 16.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 16.75it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.25it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.25it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.25it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.25it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.25it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 26.69it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 26.69it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 26.69it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 26.69it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 26.69it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:00, 28.80it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:00, 28.80it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:00, 28.80it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:00, 28.80it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:00, 28.80it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:06<00:00, 28.80it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 32.38it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 32.38it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 32.38it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 32.38it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 32.38it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 33.68it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 33.68it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 33.68it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 33.68it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 33.68it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 34.43it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 34.43it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 34.43it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 34.43it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 34.43it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 34.43it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:06<00:00, 36.96it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:06<00:00, 36.96it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:06<00:00, 36.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.98 GB):   2%|▏         | 1/58 [00:00<00:08,  6.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.95 GB):   2%|▏         | 1/58 [00:00<00:08,  6.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.95 GB):   3%|▎         | 2/58 [00:00<00:08,  6.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.94 GB):   3%|▎         | 2/58 [00:00<00:08,  6.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.94 GB):   5%|▌         | 3/58 [00:00<00:07,  6.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.93 GB):   5%|▌         | 3/58 [00:00<00:07,  6.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.93 GB):   7%|▋         | 4/58 [00:00<00:07,  7.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.94 GB):   7%|▋         | 4/58 [00:00<00:07,  7.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.94 GB):   9%|▊         | 5/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.93 GB):   9%|▊         | 5/58 [00:00<00:07,  7.37it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.93 GB):  10%|█         | 6/58 [00:00<00:06,  7.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.92 GB):  10%|█         | 6/58 [00:00<00:06,  7.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.92 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.91 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.89it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.91 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.90 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.90 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.90 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.50it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.90 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.89 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.89 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.89 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.06it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.89 GB):  21%|██        | 12/58 [00:01<00:05,  9.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.88 GB):  21%|██        | 12/58 [00:01<00:05,  9.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.87 GB):  21%|██        | 12/58 [00:01<00:05,  9.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.87 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.87 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.23it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=53.86 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.85 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.69it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=53.85 GB):  31%|███       | 18/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.84 GB):  31%|███       | 18/58 [00:01<00:03, 10.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.84 GB):  31%|███       | 18/58 [00:02<00:03, 10.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.84 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.82 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.91it/s]Capturing num tokens (num_tokens=960 avail_mem=53.83 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.91it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=53.83 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.91it/s]Capturing num tokens (num_tokens=896 avail_mem=53.83 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=832 avail_mem=53.83 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=768 avail_mem=53.82 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.15it/s]Capturing num tokens (num_tokens=704 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.53it/s]Capturing num tokens (num_tokens=640 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.53it/s]

    Capturing num tokens (num_tokens=576 avail_mem=53.82 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.53it/s]Capturing num tokens (num_tokens=576 avail_mem=53.82 GB):  48%|████▊     | 28/58 [00:02<00:01, 18.57it/s]Capturing num tokens (num_tokens=512 avail_mem=53.80 GB):  48%|████▊     | 28/58 [00:02<00:01, 18.57it/s]Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  48%|████▊     | 28/58 [00:02<00:01, 18.57it/s]Capturing num tokens (num_tokens=480 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.86it/s]Capturing num tokens (num_tokens=448 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.86it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.82 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.86it/s]Capturing num tokens (num_tokens=384 avail_mem=53.81 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.86it/s]Capturing num tokens (num_tokens=384 avail_mem=53.81 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.55it/s]Capturing num tokens (num_tokens=352 avail_mem=53.81 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.55it/s]Capturing num tokens (num_tokens=320 avail_mem=53.80 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.55it/s]Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  57%|█████▋    | 33/58 [00:02<00:01, 19.55it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.95it/s]Capturing num tokens (num_tokens=256 avail_mem=53.80 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.95it/s]Capturing num tokens (num_tokens=240 avail_mem=53.79 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.95it/s]Capturing num tokens (num_tokens=224 avail_mem=53.79 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.95it/s]Capturing num tokens (num_tokens=224 avail_mem=53.79 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.06it/s]Capturing num tokens (num_tokens=208 avail_mem=53.79 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.06it/s]Capturing num tokens (num_tokens=192 avail_mem=53.79 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.06it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.06it/s]Capturing num tokens (num_tokens=176 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=160 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=144 avail_mem=53.78 GB):  72%|███████▏  | 42/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=128 avail_mem=53.77 GB):  72%|███████▏  | 42/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=128 avail_mem=53.77 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=112 avail_mem=53.77 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.49it/s]

    Capturing num tokens (num_tokens=96 avail_mem=53.77 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.49it/s] Capturing num tokens (num_tokens=80 avail_mem=53.76 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=80 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=64 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=48 avail_mem=53.76 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.49it/s]Capturing num tokens (num_tokens=32 avail_mem=53.75 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.49it/s]

    Capturing num tokens (num_tokens=32 avail_mem=53.75 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=28 avail_mem=53.75 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=24 avail_mem=53.75 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=20 avail_mem=53.74 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=20 avail_mem=53.74 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=16 avail_mem=53.74 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=12 avail_mem=53.74 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.57it/s]

    Capturing num tokens (num_tokens=8 avail_mem=53.73 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.57it/s] Capturing num tokens (num_tokens=8 avail_mem=53.73 GB):  98%|█████████▊| 57/58 [00:03<00:00, 20.88it/s]Capturing num tokens (num_tokens=4 avail_mem=53.73 GB):  98%|█████████▊| 57/58 [00:03<00:00, 20.88it/s]Capturing num tokens (num_tokens=4 avail_mem=53.73 GB): 100%|██████████| 58/58 [00:03<00:00, 14.83it/s]


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
    Generated text:  Julie and I am a primary care physician who has over 10 years of experience in helping people find the care they need, especially to help them achieve their goals. I'm also a passionate local business owner and owner of a local non-profit organization. I have a passion for helping people live their lives to the fullest. I enjoy the experience of seeing people live through life changing events, as well as the natural beauty of the places where they live and work. I believe that it's important for people to have the ability to grow, to succeed, and to thrive, and to have access to the resources and support that they need to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a five (5) -member political party. This party is known as the Democratic Party. It is the largest party in the United States, and it has a high percentage of the total population. The party has its headquarters in Washington, D.C. It has a strong influence on the political activities of its members. It is a major force in national politics and has a strong influence on the political activities of its members.
    
    The Democratic Party was formed in 1824, and the party name is derived from the Latin word "Diedis," meaning "liberating." The party's ideology is based on the idea of
    ===============================
    Prompt: The capital of France is
    Generated text:  a city that is still a very small country, and has a population of only about 800,000. It is the capital of the department of the Île-de-France. The capital of France is Paris.
    The city of Paris is often compared to Paris in the United States, where it is known as the "City of Light". Paris was the capital of France from 1804 to 1870, but was dissolved and renamed to the "City of Light" in 1949.
    The Paris Commune, which lasted from 1871 to 18
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but how can you determine what is happening and what is the future of AI? Here’s how to do it.
    
    The future of AI is bright, but how can you determine what is happening and what is the future of AI?
    
    We’re looking at the future of AI with AI Week 2022. We’re looking at the future of AI with 12 days of exciting events from all over the world. Here’s how to determine what is happening and what is the future of AI.
    
    How to determine the future of AI with AI Week 2022.
    
    The future of AI is bright, but how


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I am a [job title] with [number of years] years of experience in [field of expertise]. I am passionate about [reason for interest in the field of expertise]. I enjoy [reason for interest in the field of expertise] and I am always looking for ways to [reason for interest in the field of expertise]. I am a [reason for interest in the field of expertise] and I am always looking for ways to [reason for interest in the field of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a vibrant and diverse city with a rich cultural heritage and is a major tourist
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud
    


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
    Generated text:  [Name] and I am a [Type] person. I am a [Grade] year [Grade] student majoring in [Major]. I am a [Occupation]. I am [Personality]. I like to [Favorite Hobby, Sport, Food, etc.] and I am [Loyalty, Kindness, Hobbies, etc.].
    
    I am an [Attitude]. I believe [Motivation]. I am [Environment]. I am [Inspiration]. I am [Self-Confidence]. I am [Health]. I am [Growth]. I am [Growth]. I am [Growth]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the country's second-largest city by population. It is the third-largest city in the European Union by population and is home to the headquarters of many major French companies and institutions. The city is known for its rich history, world-class museums and monuments, and vibrant cultural scene. It is also the seat of the French government, the French parliament, and the French presidency of the European Union. Its economy is based on finance and services, and the city is home to many high-tech and innovative companies. Paris is a major global financial hub and a symbol of French culture and identity. Its beautiful
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve, with many new trends and developments shaping how AI is used and applied in different industries and contexts. Here are some potential future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare, with applications such as personalized medicine, image recognition, and drug discovery. In the future, we can expect even more AI-driven healthcare innovations, such as more accurate and personalized disease diagnosis, more effective and efficient surgical procedures, and better treatment outcomes for patients.
    
    2. Greater integration of AI with other technologies: AI is already being integrated into various industries, such as transportation, retail, and


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

     character

    's

     name

    ]

     and

     I

    'm

     a

     [

    insert

     character

    's

     occupation

    ]

     who

     is

     passionate

     about

     [

    insert

     one

     or

     two

     interesting

     facts

     about

     your

     profession

     or

     hobby

    ].

     I

    'm

     always

     looking

     to

     learn

     more

     about

     the

     world

     and

     trying

     to

     get

     new

     skills

     to

     make

     me

     a

     better

     person

    .

     I

    'm

     always

     willing

     to

     share

     my

     knowledge

     and

     experience

     with

     anyone

     who

     wants

     to

     learn

    .

     I

    'm

     a

     [

    insert

     one

     or

     two

     positive

     words

     to

     describe

     your

     personality

     and

     interests

    ].

     I

     enjoy

     [

    insert

     one

     or

     two

     things

     that

     make

     me

     happy

    ].

     How

     about

     you

    ?

     How

     about

     you

    ?

     


    Remember

    ,

     this

     is

     just

     a

     short

     self

    -int

    roduction

    ,

     and

     you

     don

    't

     need

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     vie

     dans

     l

    '

    air

    ".

     It

     is

     a

     historical

     and

     cultural

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

     and

     the

     time

     of

     Louis

     XIV

    .

     Paris

     is

     famous

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

     the

     Lou

    vre

     Museum

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     a

     major

     center

     for

     art

    ,

     music

    ,

     and

     literature

    ,

     and

     is

     home

     to

     many

     important

     museums

     and

     art

     galleries

    .

     Paris

     is

     an

     important

     international

     city

    ,

     hosting

     numerous

     world

    -ren

    owned

     events

     and

     cultural

     festivals

    ,

     including

     the

     E

    iff

    el

     Tower

     Par

    c

     de

     La

     Vil

    lette

    ,

     and

     the

     famous

     Cam

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     new

     trends

     and

     possibilities

     emerging

     every

     year

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     be

     used

     in

     increasing

     automation

     of

     tasks

    ,

     from

     manufacturing

     and

     retail

     to

     healthcare

     and

     financial

     services

    .

     This

     will

     lead

     to

     increased

     efficiency

     and

     productivity

    .
    


    2

    .

     Emotional

     intelligence

    :

     AI

     will

     continue

     to

     be

     used

     for

     emotional

     intelligence

    ,

     such

     as

     language

     translation

    ,

     customer

     service

    ,

     and

     emotional

     support

    .

     This

     will

     help

     AI

     systems

     understand

     and

     respond

     to

     human

     emotions

    ,

     making

     interactions

     more

     meaningful

     and

     human

    -like

    .
    


    3

    .

     AI

     ethics

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     complex

    ,

     which

     means

     that

     ethical

     considerations

     will

     become

     more

    



```python
llm.shutdown()
```
