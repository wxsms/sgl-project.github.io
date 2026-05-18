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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.65it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.64it/s]


    2026-05-18 12:03:00,661 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 12:03:00] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:10,  1.29s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:10,  1.29s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:10,  1.29s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:21,  2.40it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:21,  2.40it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:21,  2.40it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:13,  3.53it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:08,  5.69it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:08,  5.69it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:08,  5.69it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:06,  7.22it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:06,  7.22it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:06,  7.22it/s]

    Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:06,  7.22it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:03, 12.14it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:03, 12.14it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:03, 12.14it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:03, 12.14it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:03, 12.14it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 16.64it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 16.64it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 16.64it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 16.64it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:06<00:02, 16.64it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:08<00:07,  3.83it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:08<00:07,  3.83it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:08<00:07,  3.83it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:08<00:07,  3.83it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:08<00:07,  3.83it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:08<00:04,  5.54it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:08<00:02,  8.91it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:08<00:02,  8.91it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:08<00:02,  8.91it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:08<00:02,  8.91it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:08<00:02,  8.91it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:08<00:02,  8.91it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:08<00:01, 12.27it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 16.91it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 16.91it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 16.91it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 16.91it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 16.91it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 16.91it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:08<00:00, 20.92it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:08<00:00, 20.92it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:08<00:00, 20.92it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:08<00:00, 20.92it/s] 

    Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:09<00:00, 20.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.26 GB):   2%|▏         | 1/58 [00:00<00:12,  4.63it/s]Capturing num tokens (num_tokens=7680 avail_mem=37.83 GB):   2%|▏         | 1/58 [00:00<00:12,  4.63it/s]Capturing num tokens (num_tokens=7680 avail_mem=37.83 GB):   3%|▎         | 2/58 [00:00<00:10,  5.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=37.86 GB):   3%|▎         | 2/58 [00:00<00:10,  5.33it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=37.86 GB):   5%|▌         | 3/58 [00:00<00:10,  5.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.23 GB):   5%|▌         | 3/58 [00:00<00:10,  5.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.23 GB):   7%|▋         | 4/58 [00:00<00:09,  5.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=37.89 GB):   7%|▋         | 4/58 [00:00<00:09,  5.50it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=37.89 GB):   9%|▊         | 5/58 [00:00<00:08,  5.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.22 GB):   9%|▊         | 5/58 [00:00<00:08,  5.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.22 GB):  10%|█         | 6/58 [00:01<00:08,  5.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.21 GB):  10%|█         | 6/58 [00:01<00:08,  5.95it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.21 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=37.94 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=37.94 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.20 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.91it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.20 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.19 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.19 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=37.98 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.55it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.18 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.18 GB):  21%|██        | 12/58 [00:01<00:05,  8.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.16 GB):  21%|██        | 12/58 [00:01<00:05,  8.33it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.16 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.16 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.03 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.03 GB):  26%|██▌       | 15/58 [00:02<00:04, 10.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.15 GB):  26%|██▌       | 15/58 [00:02<00:04, 10.12it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.14 GB):  26%|██▌       | 15/58 [00:02<00:04, 10.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.14 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.14 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.13 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.31it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=38.13 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.03 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.10 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.10 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.99it/s]Capturing num tokens (num_tokens=960 avail_mem=38.12 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.99it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=38.11 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.99it/s]Capturing num tokens (num_tokens=896 avail_mem=38.11 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.70it/s]Capturing num tokens (num_tokens=832 avail_mem=38.10 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.70it/s]Capturing num tokens (num_tokens=768 avail_mem=38.04 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.70it/s]Capturing num tokens (num_tokens=768 avail_mem=38.04 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.17it/s]Capturing num tokens (num_tokens=704 avail_mem=38.09 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.17it/s]

    Capturing num tokens (num_tokens=640 avail_mem=38.09 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.17it/s]Capturing num tokens (num_tokens=640 avail_mem=38.09 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.28it/s]Capturing num tokens (num_tokens=576 avail_mem=38.08 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.28it/s]Capturing num tokens (num_tokens=512 avail_mem=38.06 GB):  47%|████▋     | 27/58 [00:02<00:02, 14.28it/s]Capturing num tokens (num_tokens=512 avail_mem=38.06 GB):  50%|█████     | 29/58 [00:02<00:01, 15.12it/s]Capturing num tokens (num_tokens=480 avail_mem=38.07 GB):  50%|█████     | 29/58 [00:02<00:01, 15.12it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.03 GB):  50%|█████     | 29/58 [00:03<00:01, 15.12it/s]Capturing num tokens (num_tokens=416 avail_mem=38.07 GB):  50%|█████     | 29/58 [00:03<00:01, 15.12it/s]Capturing num tokens (num_tokens=416 avail_mem=38.07 GB):  55%|█████▌    | 32/58 [00:03<00:01, 16.62it/s]Capturing num tokens (num_tokens=384 avail_mem=38.07 GB):  55%|█████▌    | 32/58 [00:03<00:01, 16.62it/s]Capturing num tokens (num_tokens=352 avail_mem=38.06 GB):  55%|█████▌    | 32/58 [00:03<00:01, 16.62it/s]

    Capturing num tokens (num_tokens=352 avail_mem=38.06 GB):  59%|█████▊    | 34/58 [00:03<00:01, 17.05it/s]Capturing num tokens (num_tokens=320 avail_mem=38.05 GB):  59%|█████▊    | 34/58 [00:03<00:01, 17.05it/s]Capturing num tokens (num_tokens=288 avail_mem=38.04 GB):  59%|█████▊    | 34/58 [00:03<00:01, 17.05it/s]Capturing num tokens (num_tokens=288 avail_mem=38.04 GB):  62%|██████▏   | 36/58 [00:03<00:01, 17.36it/s]Capturing num tokens (num_tokens=256 avail_mem=38.04 GB):  62%|██████▏   | 36/58 [00:03<00:01, 17.36it/s]Capturing num tokens (num_tokens=240 avail_mem=38.03 GB):  62%|██████▏   | 36/58 [00:03<00:01, 17.36it/s]

    Capturing num tokens (num_tokens=240 avail_mem=38.03 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.90it/s]Capturing num tokens (num_tokens=224 avail_mem=38.02 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.90it/s]Capturing num tokens (num_tokens=208 avail_mem=38.01 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.90it/s]Capturing num tokens (num_tokens=192 avail_mem=38.01 GB):  66%|██████▌   | 38/58 [00:03<00:01, 17.90it/s]Capturing num tokens (num_tokens=192 avail_mem=38.01 GB):  71%|███████   | 41/58 [00:03<00:00, 18.94it/s]Capturing num tokens (num_tokens=176 avail_mem=38.01 GB):  71%|███████   | 41/58 [00:03<00:00, 18.94it/s]Capturing num tokens (num_tokens=160 avail_mem=38.01 GB):  71%|███████   | 41/58 [00:03<00:00, 18.94it/s]

    Capturing num tokens (num_tokens=160 avail_mem=38.01 GB):  74%|███████▍  | 43/58 [00:03<00:00, 18.88it/s]Capturing num tokens (num_tokens=144 avail_mem=38.00 GB):  74%|███████▍  | 43/58 [00:03<00:00, 18.88it/s]Capturing num tokens (num_tokens=128 avail_mem=37.99 GB):  74%|███████▍  | 43/58 [00:03<00:00, 18.88it/s]Capturing num tokens (num_tokens=128 avail_mem=37.99 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.05it/s]Capturing num tokens (num_tokens=112 avail_mem=37.99 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.05it/s]Capturing num tokens (num_tokens=96 avail_mem=37.98 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.05it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=37.98 GB):  81%|████████  | 47/58 [00:03<00:00, 19.29it/s]Capturing num tokens (num_tokens=80 avail_mem=37.97 GB):  81%|████████  | 47/58 [00:03<00:00, 19.29it/s]Capturing num tokens (num_tokens=64 avail_mem=37.97 GB):  81%|████████  | 47/58 [00:03<00:00, 19.29it/s]Capturing num tokens (num_tokens=48 avail_mem=37.96 GB):  81%|████████  | 47/58 [00:04<00:00, 19.29it/s]Capturing num tokens (num_tokens=48 avail_mem=37.96 GB):  86%|████████▌ | 50/58 [00:04<00:00, 19.72it/s]Capturing num tokens (num_tokens=32 avail_mem=37.95 GB):  86%|████████▌ | 50/58 [00:04<00:00, 19.72it/s]Capturing num tokens (num_tokens=28 avail_mem=37.95 GB):  86%|████████▌ | 50/58 [00:04<00:00, 19.72it/s]

    Capturing num tokens (num_tokens=28 avail_mem=37.95 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.69it/s]Capturing num tokens (num_tokens=24 avail_mem=37.94 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.69it/s]Capturing num tokens (num_tokens=20 avail_mem=37.93 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.69it/s]Capturing num tokens (num_tokens=16 avail_mem=37.93 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.69it/s]Capturing num tokens (num_tokens=16 avail_mem=37.93 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.92it/s]Capturing num tokens (num_tokens=12 avail_mem=37.92 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.92it/s]Capturing num tokens (num_tokens=8 avail_mem=37.91 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.92it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=37.91 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.92it/s]Capturing num tokens (num_tokens=4 avail_mem=37.91 GB): 100%|██████████| 58/58 [00:04<00:00, 13.13it/s]


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
    Generated text:  Lenz.
    I am a psychotherapist from Germany, who specializes in psychological counseling and psychotherapy. I specialize in trauma, depression, anxiety, relationship issues, and relationship counseling.
    I believe that everything that is not understood and/or not being explained can be a source of psychological distress, whether this is related to a past trauma, a traumatic event, or something that has happened since.
    I help my clients to understand their feelings, the events that have happened and how to deal with them in a healthy and appropriate manner.
    I work with clients of all ages, including children, adolescents, adults and seniors. I can be in a private
    ===============================
    Prompt: The president of the United States is
    Generated text:  6 feet tall. His daughter is 6 feet 10 inches tall. His son is 6 feet 12 inches tall. If the president stands on the president's chair for 1 minute, how much taller will he be compared to his daughter? First, we need to convert all measurements to inches because the daughter's height is given in inches. The president is 6 feet tall, which is 6 * 12 = 72 inches tall. His daughter is 6 feet 10 inches tall, which is 6 * 12 + 10 = 78 inches tall.
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A: Paris B: Strasbourg C: Nantes D: Nice
    
    To determine the capital of France, let's list the capital cities in alphabetical order:
    
    1. Paris
    2. Strasbourg
    3. Nantes
    4. Nice
    
    From this list, we can see that the capital cities of France are listed in order from the smallest to the largest. Therefore, the capital of France is Paris.
    
    The correct answer is: A: Paris
    
    So, the capital of France is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  more complex than ever. As we move into the future, it will be important to understand the broader implications of the technology. This will require a deep understanding of the research that is being conducted in the field, as well as an appreciation of the potential challenges and opportunities that AI presents.
    AI is a rapidly evolving field, and it is unlikely that we will see a complete overhaul of the way we perceive and interact with the digital world in the near future. However, the technology is already making significant strides in areas such as image and speech recognition, natural language processing, and predictive analytics.
    One of the key areas that will continue to drive progress


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] who has always been [Occupation] and [Skill]. I'm a [Favorite Hobby] enthusiast and [Favorite Book] lover. I'm also a [Favorite Movie] fan and [Favorite Music] lover. I'm always looking for new experiences and adventures, and I'm always eager to learn new things. I'm a [Favorite Sport] enthusiast and [Favorite Food] lover. I'm always looking for new ways to improve myself and expand my horizons. I'm a [Favorite Music] lover and [Favorite Book] lover. I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and restaurants. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a popular tourist destination, attracting millions of visitors each year. It is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, with dishes like croissants, boudin, and escargot being popular among locals and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more prevalent in various industries, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability.
    
    3. Development of more advanced AI systems: AI systems are likely to become more advanced and capable of performing tasks that were previously considered impossible, such as playing chess or playing musical instruments.
    
    4. Increased use of AI in healthcare: AI
    


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
    Generated text:  [Name]. I'm a [Age] year old [Occupation], and I have always been passionate about learning and seeking knowledge. I've always been curious about new things and always sought to expand my horizons. I've always had a strong interest in technology and am always looking for ways to stay up-to-date with the latest innovations. I'm always eager to share my knowledge with others, and I love to inspire others with my passion for learning. How would you describe your personality? Personality Traits: 1. Curious and persistent. 2. Enthusiastic and passionate about learning. 3. Learning-oriented and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with many potential applications and developments that are likely to shape the way we live, work, and interact with technology in the years ahead. Here are some possible trends that may impact AI in the coming years:
    
    1. Improved understanding of human emotions: AI will continue to learn and improve at understanding human emotions, behaviors, and attitudes. This will enable AI systems to provide more personalized and empathetic responses, which can lead to more meaningful interactions with users.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare, such as in personalized medicine, disease prediction, and treatment planning. As AI technology improves


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

    ].

     I

    'm

     a

     [

    job

     title

    ]

     who

     has

     been

     with

     the

     company

     for

     [

    number

    ]

     years

    .

     I

     have

     a

     passion

     for

     [

    the

     character

    's

     career

     goal

     or

     interest

    ]

     and

     always

     strive

     to

     help

     the

     company

     achieve

     its

     goals

    .

     I

    'm

     a

     great

     communicator

     and

     always

     make

     things

     easy

     for

     people

     to

     understand

     and

     follow

     through

     with

     their

     tasks

    .

     I

    'm

     also

     very

     reliable

     and

     have

     a

     strong

     work

     ethic

    .

     I

    'm

     always

     ready

     to

     help

     and

     support

     my

     team

     members

    .

     I

     enjoy

     working

     with

     people

     and

     always

     strive

     to

     be

     a

     great

     teammate

    .

     My

     style

     of

     communication

     is

     easy

     to

     understand

     and

     follow

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

     and

     improve

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Which

     of

     the

     following

     options

     is

     closest

     to

     this

     statement

    ?

     (

    A

    )

     Rome

     (

    B

    )

     Tokyo

     (

    C

    )

     Mumbai

     (

    D

    )

     Paris

    


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

     correct

    .

     Therefore

    ,

     the

     closest

     option

     to

     this

     statement

     is

     (

    D

    )

     Paris

    .

     
    


    -

     (

    A

    )

     Rome

     is

     not

     the

     capital

     of

     France

    .


    -

     (

    B

    )

     Tokyo

     is

     not

     the

     capital

     of

     France

    .


    -

     (

    C

    )

     Mumbai

     is

     not

     the

     capital

     of

     France

    .

     
    


    Thus

    ,

     the

     answer

     is

     (

    D

    )

     Paris

    .

     The

     capital

     of

     France

     is

     indeed

     Paris

    .

     
    


    Therefore

    ,

     the

     answer

     is

     (

    D

    ).

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     and

     expand

    ,

     with

     new

     technologies

     and

     breakthrough

    s

     emerging

     on

     a

     regular

     basis

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     AI

     will

     become

     more

     integrated

     into

     everyday

     life

    :

     One

     of

     the

     biggest

     trends

     in

     AI

     is

     the

     increasing

     integration

     of

     AI

     into

     everyday

     life

    .

     From

     smart

     home

     devices

     to

     self

    -driving

     cars

    ,

     AI

     is

     being

     used

     to

     simplify

     our

     lives

     and

     improve

     our

     productivity

     and

     efficiency

    .
    


    2

    .

     AI

     will

     become

     more

     autonomous

    :

     With

     the

     development

     of

     AI

    ,

     we

     will

     see

     a

     significant

     increase

     in

     the

     number

     of

     autonomous

     vehicles

    .

     These

     vehicles

     will

     be

     able

     to

     drive

     safely

     on

     the

     roads

     and

     highways

    ,

     with

     no

     human

    



```python
llm.shutdown()
```
