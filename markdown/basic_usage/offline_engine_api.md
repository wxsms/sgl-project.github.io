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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]


    2026-04-13 01:13:46,797 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 01:13:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:04,  1.15s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:04,  1.15s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:04,  1.15s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.14it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:14,  3.66it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:14,  3.66it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.66it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.45it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.45it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.45it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.45it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.68it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.68it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.68it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 16.68it/s]

    Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:01, 19.28it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 26.52it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 26.52it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 26.52it/s]

    Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 26.52it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 26.52it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 30.66it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 30.66it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 30.66it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 30.66it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 30.66it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.92it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 30.92it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 30.92it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 30.92it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 32.26it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 32.26it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 32.26it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 31.95it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 31.95it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 31.95it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 31.95it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 31.95it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 32.71it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 32.71it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 32.71it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 32.71it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 32.71it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 34.57it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 34.57it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 34.57it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 34.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.04 GB):   3%|▎         | 2/58 [00:00<00:06,  8.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.03 GB):   3%|▎         | 2/58 [00:00<00:06,  8.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.03 GB):   5%|▌         | 3/58 [00:00<00:07,  7.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.03 GB):   5%|▌         | 3/58 [00:00<00:07,  7.59it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.03 GB):   7%|▋         | 4/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.03 GB):   7%|▋         | 4/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.03 GB):   9%|▊         | 5/58 [00:00<00:06,  7.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.03 GB):   9%|▊         | 5/58 [00:00<00:06,  7.65it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=59.03 GB):  10%|█         | 6/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.03 GB):  10%|█         | 6/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.03 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.02 GB):  12%|█▏        | 7/58 [00:00<00:06,  7.91it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.02 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.02 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.02 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.02 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.01 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.78it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.01 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.01 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.01 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.01 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.01 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.98it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=59.00 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.00 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.00 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.00 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.99 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.99 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.99 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.97 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s]Capturing num tokens (num_tokens=960 avail_mem=58.98 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s] Capturing num tokens (num_tokens=896 avail_mem=58.98 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.27it/s]Capturing num tokens (num_tokens=896 avail_mem=58.98 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=832 avail_mem=58.97 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=768 avail_mem=58.97 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.97 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=640 avail_mem=58.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=576 avail_mem=58.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=512 avail_mem=58.95 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.71it/s]Capturing num tokens (num_tokens=512 avail_mem=58.95 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=480 avail_mem=58.97 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=448 avail_mem=58.97 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=416 avail_mem=58.96 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=384 avail_mem=58.96 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=352 avail_mem=58.96 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=320 avail_mem=58.95 GB):  50%|█████     | 29/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=320 avail_mem=58.95 GB):  60%|██████    | 35/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=288 avail_mem=58.95 GB):  60%|██████    | 35/58 [00:01<00:00, 38.29it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.95 GB):  60%|██████    | 35/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=240 avail_mem=58.94 GB):  60%|██████    | 35/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  60%|██████    | 35/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=208 avail_mem=58.94 GB):  60%|██████    | 35/58 [00:02<00:00, 38.29it/s]Capturing num tokens (num_tokens=208 avail_mem=58.94 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=192 avail_mem=58.94 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=176 avail_mem=58.93 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=160 avail_mem=58.93 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=128 avail_mem=58.92 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  69%|██████▉   | 40/58 [00:02<00:00, 41.20it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  79%|███████▉  | 46/58 [00:02<00:00, 44.14it/s]Capturing num tokens (num_tokens=96 avail_mem=58.92 GB):  79%|███████▉  | 46/58 [00:02<00:00, 44.14it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 44.14it/s]Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 44.14it/s]Capturing num tokens (num_tokens=48 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 44.14it/s]Capturing num tokens (num_tokens=32 avail_mem=58.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 44.14it/s]Capturing num tokens (num_tokens=32 avail_mem=58.91 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s]Capturing num tokens (num_tokens=28 avail_mem=58.90 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s]Capturing num tokens (num_tokens=24 avail_mem=58.90 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s]Capturing num tokens (num_tokens=12 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s]Capturing num tokens (num_tokens=8 avail_mem=58.88 GB):  88%|████████▊ | 51/58 [00:02<00:00, 45.63it/s] Capturing num tokens (num_tokens=8 avail_mem=58.88 GB):  98%|█████████▊| 57/58 [00:02<00:00, 47.41it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB):  98%|█████████▊| 57/58 [00:02<00:00, 47.41it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:02<00:00, 24.29it/s]


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
    Generated text:  Sebastian. I am a student, and I am currently enrolled in the computer science program at the University of California, Berkeley. I have been studying computer science for the past four years and have been working on projects in various areas of computer science, including but not limited to, artificial intelligence, machine learning, and computer vision.
    
    I have a keen interest in developing and implementing software solutions to solve real-world problems, and I am always looking for ways to improve my skills and knowledge. I am also a keen reader and enjoy reading different types of books, such as books on computer science, science, and philosophy.
    
    What do you like to do
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He likes the idea of 10 bases, but he doesn't want more than 30. He also likes the idea of 2 Marines in each base, but he wants at least 10. If he has 150 Marines, how many bases can he have if he wants to keep the minimum number of Marines in each base?
    To determine how many bases the president of the United States can have while keeping the minimum number of Marines in each base, we need to follow these steps:
    
    1. Determine the minimum number of Marines per base.
    2. Calculate how
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Nice
    C. Marseille
    D. Lyon
    Answer:
    A
    
    A and B have an agreement to loan money to C. The agreement stipulates that if the loan is repaid before the maturity date, the repayment amount must be no less than 80% of the principal loan amount. However, A has already repaid 40% of the principal. According to the agreement, what amount must B repay to A if he wants to fully repay the loan?
    A. 200
    B. 160
    C. 120
    D. 
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and promises significant advancements in areas like healthcare, transportation, and personalization. However, it's important to understand how to effectively and ethically utilize this technology. In this essay, we will explore the ethical issues surrounding AI and the ways in which these issues can be addressed.
    
    One of the primary ethical concerns surrounding AI is the potential for bias in its algorithms. Bias can occur when certain types of data are unfairly favored or discriminated against, leading to inaccurate or unfair outcomes. For example, a facial recognition software algorithm used in law enforcement can be biased against certain groups of people, leading to discrimination and potential harm.
    
    Another ethical issue is


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who has been [Number of Years] years in this field. I'm passionate about [What I Love About My Profession], and I'm always looking for ways to [What I Want to Improve]. I'm [What I Do Best]. I'm [What I'm Looking Forward To Doing Next]. I'm excited to meet you and learn more about you. How about you? [Name] [Age] [Occupation] [Skill/Ability] [What I Love About My Profession] [What I Want to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of French culture and language on other European countries. It is also home to many notable French artists, writers, and musicians. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to attract visitors from around the world. The city is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI
    


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
    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and discuss what I can contribute to [specific industry or team]. I'm a [type of person], [charisma, witty, humorous, etc.]. I'm [personality trait], [example, example, etc.]. I'm [role in the company], [example of role, example, etc.]. I enjoy [specific hobby or activity], [example, example, etc.]. I am [attitude, personality, etc.]. Thank you for having me. Let me know if there's anything I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the second largest city in the world by population. The city is known for its rich history, beautiful architecture, and vibrant cultural scene. It is also a major financial and financial hub, hosting many global financial institutions. Paris has a diverse population, with a high concentration of wealthy and middle-class residents. The city is famous for its museums, art galleries, and other cultural attractions, including the Louvre Museum. Paris is also home to many world-renowned universities and research institutions. Overall, Paris is a vibrant, diverse, and cosmopolitan city that has played an important role in French history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are several key trends shaping its direction. Here are some of the most likely developments:
    
    1. Increased integration with other technologies: AI is becoming more integrated with other technologies, such as machine learning and big data. This integration will allow AI to learn from new data more effectively and adapt to new situations.
    
    2. Greater use of AI in healthcare: AI is being used in a variety of healthcare applications, including medical imaging, disease diagnosis, and personalized medicine. As AI technology improves, it is expected to become even more widely used in healthcare.
    
    3. Increased use of AI in consumer goods: AI is being used to


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

     an

     experienced

     [

    insert

     your

     area

     of

     expertise

     or

     field

     of

     study

    ].

     I

     have

     been

     working

     in

     the

     [

    insert

     your

     job

     title

    ]

     for

     [

    insert

     your

     years

     of

     experience

    ]

     years

    ,

     and

     I

     am

     passionate

     about

     [

    insert

     your

     area

     of

     interest

     or

     hobby

    ].

     I

     believe

     that

     my

     experiences

     have

     equipped

     me

     with

     the

     skills

     to

     make

     a

     significant

     impact

     in

     the

     world

     and

     I

     am

     excited

     to

     contribute

     my

     knowledge

     and

     skills

     to

     the

     world

    .

     If

     you

     need

     help

    ,

     I

     am

     always

     here

     to

     help

    .

     
    


    Thank

     you

     for

     having

     me

    .

     
    


    *

    Please

     note

     that

     this

     is

     a

     fictional

     character

     introduction

     and

     does

     not

     reflect

     any

     real

     person

     or

     organization

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     accurate

     and

     factual

    .

     Paris

    ,

     officially

     known

     as

     the

     "

    City

     of

     Love

    ,"

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    .

     It

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

     (

    F

    ign

    ol

    )

     on

     the

     Se

    ine

     River

     and

     is

     the

     seat

     of

     the

     French

     government

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     Roman

     times

    ,

     and

     it

     has

     evolved

     into

     a

     modern

     met

    ropolis

     with

     a

     diverse

     cultural

     and

     artistic

     scene

    .

     Paris

     is

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

    ,

     as

     well

     as

     for

     its

     gastr

    onomy

    ,

     nightlife

    ,

     and

     cultural

     events

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     advancements

     that

     are

     shaping

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

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

     Increased

     automation

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

     automation

     in

     various

     industries

    ,

     leading

     to

     more

     efficient

     and

     cost

    -effective

     solutions

    .
    


    2

    .

     Natural

     language

     processing

    :

     With

     the

     increasing

     use

     of

     AI

     in

     natural

     language

     processing

    ,

     we

     may

     see

     more

     personalized

     and

     intelligent

     customer

     service

     experiences

    .
    


    3

    .

     Increased

     collaboration

     and

     cooperation

    :

     As

     AI

     becomes

     more

     prevalent

    ,

     we

     may

     see

     a

     greater

     need

     for

     collaboration

     and

     cooperation

     between

     humans

     and

     machines

    ,

     leading

     to

     new

     industries

     and

     technologies

    .
    


    4

    .

     Enhanced

     data

    



```python
llm.shutdown()
```
