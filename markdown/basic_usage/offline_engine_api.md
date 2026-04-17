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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 07:11:21] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]


    2026-04-17 07:11:25,953 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 07:11:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:27,  2.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:27,  2.59s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:03,  1.13s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:03,  1.13s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:03,  1.13s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.19it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.73it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.73it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.73it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:13,  3.73it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04, 10.60it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04, 10.60it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04, 10.60it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04, 10.60it/s]

    Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04, 10.60it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:02, 15.11it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:02, 15.11it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:02, 15.11it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:02, 15.11it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:02, 15.11it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:01, 18.84it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:01, 18.84it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:01, 18.84it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:01, 18.84it/s]

    Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:01, 18.84it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 22.97it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 22.97it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 22.97it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 22.97it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 22.97it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 26.04it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 26.04it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 26.04it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 26.04it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 26.04it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.50it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.50it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 32.77it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 32.77it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 32.77it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 32.77it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 34.07it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 34.07it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 34.85it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 34.85it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 34.85it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 34.85it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 34.85it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 34.85it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 37.71it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 37.71it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 37.71it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 37.71it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 37.71it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.88 GB):   2%|▏         | 1/58 [00:00<00:07,  7.61it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.85 GB):   2%|▏         | 1/58 [00:00<00:07,  7.61it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.85 GB):   3%|▎         | 2/58 [00:00<00:07,  7.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.85 GB):   3%|▎         | 2/58 [00:00<00:07,  7.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.85 GB):   5%|▌         | 3/58 [00:00<00:07,  7.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.85 GB):   5%|▌         | 3/58 [00:00<00:07,  7.32it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.85 GB):   7%|▋         | 4/58 [00:00<00:07,  7.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.85 GB):   7%|▋         | 4/58 [00:00<00:07,  7.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.85 GB):   9%|▊         | 5/58 [00:00<00:06,  7.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.84 GB):   9%|▊         | 5/58 [00:00<00:06,  7.86it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.84 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.85 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.85 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.84 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.99it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.84 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.84 GB):  14%|█▍        | 8/58 [00:01<00:06,  7.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.84 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.84 GB):  16%|█▌        | 9/58 [00:01<00:06,  8.09it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.84 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.83 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.83 GB):  19%|█▉        | 11/58 [00:01<00:05,  7.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.83 GB):  19%|█▉        | 11/58 [00:01<00:05,  7.97it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.83 GB):  19%|█▉        | 11/58 [00:01<00:05,  7.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.83 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.83 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.82 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.60it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=53.82 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.82 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.82 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.82 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.81 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.52it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=53.81 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.81 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.80 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.78 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.95it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=53.78 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.70it/s]Capturing num tokens (num_tokens=960 avail_mem=53.80 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.70it/s] Capturing num tokens (num_tokens=896 avail_mem=53.80 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.70it/s]Capturing num tokens (num_tokens=896 avail_mem=53.80 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.19it/s]Capturing num tokens (num_tokens=832 avail_mem=53.79 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.19it/s]

    Capturing num tokens (num_tokens=768 avail_mem=53.79 GB):  40%|███▉      | 23/58 [00:02<00:02, 13.19it/s]Capturing num tokens (num_tokens=768 avail_mem=53.79 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.44it/s]Capturing num tokens (num_tokens=704 avail_mem=53.79 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.44it/s]Capturing num tokens (num_tokens=640 avail_mem=53.76 GB):  43%|████▎     | 25/58 [00:02<00:02, 13.44it/s]

    Capturing num tokens (num_tokens=640 avail_mem=53.76 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.62it/s]Capturing num tokens (num_tokens=576 avail_mem=53.76 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.62it/s]Capturing num tokens (num_tokens=512 avail_mem=53.75 GB):  47%|████▋     | 27/58 [00:02<00:02, 12.62it/s]Capturing num tokens (num_tokens=512 avail_mem=53.75 GB):  50%|█████     | 29/58 [00:02<00:02, 12.55it/s]Capturing num tokens (num_tokens=480 avail_mem=53.77 GB):  50%|█████     | 29/58 [00:02<00:02, 12.55it/s]Capturing num tokens (num_tokens=448 avail_mem=53.76 GB):  50%|█████     | 29/58 [00:02<00:02, 12.55it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.76 GB):  50%|█████     | 29/58 [00:02<00:02, 12.55it/s]Capturing num tokens (num_tokens=384 avail_mem=53.76 GB):  50%|█████     | 29/58 [00:02<00:02, 12.55it/s]Capturing num tokens (num_tokens=384 avail_mem=53.76 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.38it/s]Capturing num tokens (num_tokens=352 avail_mem=53.75 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.38it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:03<00:01, 17.38it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  60%|██████    | 35/58 [00:03<00:01, 13.14it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  60%|██████    | 35/58 [00:03<00:01, 13.14it/s]Capturing num tokens (num_tokens=256 avail_mem=58.33 GB):  60%|██████    | 35/58 [00:03<00:01, 13.14it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.33 GB):  64%|██████▍   | 37/58 [00:03<00:01, 12.67it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:03<00:01, 12.67it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:03<00:01, 12.67it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 12.06it/s]Capturing num tokens (num_tokens=208 avail_mem=58.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 12.06it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 12.06it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  71%|███████   | 41/58 [00:03<00:01, 11.57it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:01, 11.57it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:01, 11.57it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  74%|███████▍  | 43/58 [00:03<00:01, 11.90it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:01, 11.90it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:01, 11.90it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 12.06it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 12.06it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 12.06it/s] Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:04<00:00, 12.14it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:04<00:00, 12.14it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:04<00:00, 12.14it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:04<00:00, 12.90it/s]Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:04<00:00, 12.90it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:04<00:00, 12.90it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:04<00:00, 10.96it/s]Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:04<00:00, 10.96it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:04<00:00, 10.96it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 11.87it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 11.87it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 11.87it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:04<00:00, 12.23it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 12.23it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 12.23it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:05<00:00, 12.33it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:05<00:00, 12.33it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:05<00:00, 11.36it/s]


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
    Generated text:  Dalish and I am from the United States. Before coming to the U. S., I had lived in the Philippines for six years as a guest worker. During that time, I had lived with my family in the Philippines, which was a poor community. The residents of the community lived in shanty villages and were very poor, earning almost nothing. I would go out to shop every day to buy food for my family and help the families in my neighborhood in order to make ends meet. But at the same time, I was worried that my daughter, who is 11 years old, would be left behind to go to
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many wars to start in the future. He has decided to test the waters and start 10 different wars. Each war has a different potential cost and benefits, and the president wants to make an informed decision. He has divided the costs of starting each war into 5 categories: low, medium, high, and high. He has also divided the potential benefits of starting each war into 5 categories: low, medium, high, and very high. If the president starts all 10 wars, how many different combinations of costs and benefits will there be? Also, if the president is considering the impact of
    ===============================
    Prompt: The capital of France is
    Generated text:  in which city? The capital of France is Paris. It is located in the northwest of the country, and it is known as the "City of Light" due to its famous canals and grand cathedrals. Paris is the capital of the Var department and the Île-de-France region. France is a country with a diverse and culturally rich history and traditions, and Paris is a symbol of this. The city is home to many world-renowned landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is also an important center for science, culture, and art, and
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the next generation of developers. Here’s why it’s so important that we teach the next generation of AI to code today.
    AI is here to stay. But the future of AI is not just about developing new products that can automate tasks. It’s also about creating a new way of thinking about how we use technology. It’s about teaching the next generation of developers to code today.
    The way that we code is one of the most important aspects of our digital lives. As we code, we’re learning about programming languages, data structures, algorithms, and more. This is a valuable skill that will stay with us for


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. I'm a [job title] at [company name], and I'm always looking for ways to [job title] and I'm always eager to learn new things. I'm a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry, art scene, and its role in hosting the Olympics and World Cup football matches. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has played a significant role in shaping French culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust and transparent AI systems that are designed to minimize harm and maximize
    


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
    Generated text:  [insert name], and I'm a [insert profession] who's passionate about [insert something related to the profession]. I've always been fascinated by [insert a specific topic or idea related to the profession], and I want to make a name for myself as someone who can bring [insert a specific skill or talent] to the table. What's your name, and what's your profession? How can I help you with anything you need? Come talk to me. [Insert name] 
    [Insert profession] 
    Hey, I'm [insert name], a [insert profession] who's passionate about [insert something related to the profession
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A concise statement about Paris's importance and cultural significance would be: Paris is the cultural, political, and economic center of France, known for its rich history, art, and food. 
    
    A more detailed statement might include: Paris is the birthplace of the French Revolution, the site of Napoleon's conquest of Egypt, and the center of the French Riviera. It is also one of the most visited cities in the world, attracting millions of tourists each year due to its beautiful architecture, historic landmarks, and delicious cuisine. 
    
    Overall, Paris plays a crucial role in France's political, cultural, and economic landscape, making
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to expand and evolve, with a number of possible trends and developments.
    
    1. Integration with other technologies: One of the most likely trends in AI in the future will be the integration of AI with other technologies, such as augmented reality, virtual reality, and IoT. This will enable AI-powered devices to interact with the real world, enabling more seamless and intuitive user experiences.
    
    2. Increased use of AI for healthcare: AI has the potential to revolutionize healthcare by improving diagnosis, treatment, and patient care. For example, AI-powered diagnostic systems can help doctors identify diseases more accurately and quickly, while AI-powered drug discovery can accelerate


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

    'm

     a

     [

    Role

    ]

    !

     

    🎨

    
    


    Let

    's

     talk

     about

     [

    Something

     you

    're

     passionate

     about

     or

     interested

     in

    ],

     it

    's

     such

     a

     [

    Your

     favorite

     word

     or

     phrase

    ].

     I

    'm

     always

     [

    the

     past

     tense

     of

     something

    ],

     and

     I

     enjoy

     [

    any

     creative

     or

     creative

     word

     or

     phrase

    ]

    !

     

    🎨

    ✨

    
    


    That

    's

     it

    !

     I

    'm

     ready

     to

     meet

     you

    .

     What

    's

     your

     name

    ,

     [

    Your

     Name

    ]?

     

    📚

    ✨

    
    


    [

    Your

     Name

    ]

     

    ✨

    ✨

    ✨

    
    


    ---
    


    Remember

     to

     tailor

     your

     introduction

     to

     fit

     the

     character

    's

     interests

     and

     personality

    .

     Additionally

    ,

     make

     sure

     to

     consider

     the

     role

     they

     play

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     as

     the

     "

    City

     of

     Light

    "

     and

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

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    .

     Paris

     is

     home

     to

     a

     diverse

     population

     of

     

    2

    .

    1

     million

     people

     and

     is

     a

     major

     hub

     for

     business

    ,

     culture

    ,

     and

     politics

     in

     France

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     "

    F

    ête

     de

     la

     Tou

    ss

    aint

    "

     (

    Saint

     Anthony

    's

     Day

    )

     which

     celebrates

     the

     end

     of

     the

     Hait

    ian

     Revolution

     and

     is

     a

     significant

     cultural

     event

     in

     the

     country

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

     dynamic

     city

     that

     is

     an

     essential

     part

     of

     French

     culture

     and

     identity

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     growth

     and

     adoption

     of

     advanced

     algorithms

     and

     models

    ,

     which

     will

     allow

     for

     more

     sophisticated

     and

     complex

     interactions

     with

     humans

    .

     AI

     is

     also

     likely

     to

     continue

     to

     evolve

     and

     adapt

    ,

     allowing

     for

     the

     development

     of

     new

     applications

     and

     applications

     that

     are

     based

     on

     human

     intelligence

     rather

     than

     solely

     relying

     on

     machine

     learning

     algorithms

    .

     Additionally

    ,

     there

     may

     be

     a

     greater

     emphasis

     on

     ethical

     considerations

     and

     the

     development

     of

     frameworks

     and

     standards

     to

     ensure

     that

     AI

     systems

     are

     developed

     and

     used

     in

     a

     responsible

     and

     ethical

     manner

    .

     Finally

    ,

     there

     may

     be

     a

     growing

     interest

     in

     the

     ethical

     and

     social

     impacts

     of

     AI

    ,

     with

     more

     people

     and

     organizations

     recognizing

     the

     need

     to

     address

     issues

     such

     as

     bias

    ,

     transparency

    ,

    



```python
llm.shutdown()
```
