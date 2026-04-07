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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.49it/s]


    2026-04-07 12:39:02,971 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 12:39:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.16s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.60it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.35it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.35it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.35it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.55it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.55it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.55it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.55it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 11.76it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 11.76it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 11.76it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 11.76it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 11.76it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 16.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 16.20it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 16.20it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 16.20it/s]

    Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 16.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 19.97it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 23.32it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 23.32it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 23.32it/s]

    Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 23.32it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 23.32it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 25.62it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 25.62it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 25.62it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 25.62it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 25.62it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 28.56it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 28.56it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 28.56it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 28.56it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:00, 28.56it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 29.58it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 29.58it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 29.58it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 29.58it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 29.58it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 31.75it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 31.75it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 31.75it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 31.75it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 31.75it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 32.13it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 32.13it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 32.13it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 32.13it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 32.13it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.61it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 35.51it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:10,  5.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   2%|▏         | 1/58 [00:00<00:10,  5.22it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:14,  3.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   3%|▎         | 2/58 [00:00<00:14,  3.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:12,  4.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   5%|▌         | 3/58 [00:00<00:12,  4.44it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:10,  5.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:10,  5.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:10,  5.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:07,  7.33it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:01<00:06,  8.21it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.24it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.11it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.38 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.33it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.71it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.65it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.65it/s] Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.65it/s]

    Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.34it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.34it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.34it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:03, 10.92it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:03, 10.92it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:03, 10.92it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.37it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.37it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:03<00:02, 11.37it/s]

    Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  50%|█████     | 29/58 [00:03<00:02, 11.68it/s]Capturing num tokens (num_tokens=480 avail_mem=58.35 GB):  50%|█████     | 29/58 [00:03<00:02, 11.68it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:03<00:02, 11.68it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.64it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.64it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.64it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  57%|█████▋    | 33/58 [00:03<00:02,  8.73it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:03<00:02,  8.73it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.50it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.50it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.50it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:03<00:02,  9.42it/s]Capturing num tokens (num_tokens=256 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:03<00:02,  9.42it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:04<00:02,  9.42it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:04<00:01, 10.28it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:04<00:01, 10.28it/s]Capturing num tokens (num_tokens=208 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:04<00:01, 10.28it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.94it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.94it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:04<00:01,  9.94it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.76it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.76it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.76it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:04<00:01, 10.76it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 12.90it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 12.90it/s]Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:04<00:01, 12.90it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:04<00:00, 12.68it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:04<00:00, 12.68it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:04<00:00, 12.68it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:04<00:00, 12.47it/s]Capturing num tokens (num_tokens=48 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:04<00:00, 12.47it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:05<00:00, 12.47it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:05<00:00, 12.08it/s]Capturing num tokens (num_tokens=28 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:05<00:00, 12.08it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:05<00:00, 12.08it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.01it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.01it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:05<00:00, 12.01it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.11it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.11it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:05<00:00, 12.11it/s] Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:05<00:00, 12.19it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:05<00:00, 12.19it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:05<00:00, 10.12it/s]


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
    Generated text:  John and I am studying genetics. I am working on a project that involves finding the genetic code for a new plant species. I have identified a few genes in the plant that are involved in the production of various metabolites. Now, I need to determine the specific amino acid sequence that corresponds to each of these metabolites. Can you help me with this?
    
    Certainly! To determine the amino acid sequence that corresponds to each of these metabolites, we need to know the sequence of amino acids that are produced by the genes involved. 
    
    Assuming that you have already identified the genes and their respective mRNA sequences, the next step would be to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with many duties. He is the leader of the country. Before the president takes office, he must pass a test. The test he must take is called the "Nobel Test". The test is very difficult. The test is given at the end of the month. There is no prize if he passes the test. The president is given a card. He must not give any other person the card. The president must not look at the card. He must not use any other paper or any writing on the card. If he does any of these things, he will be told to send the president away. This is very
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. London
    B. Paris
    C. Moscow
    D. Tokyo
    Answer:
    B
    
    The most common cause of sudden death in elderly people is ____
    A. Atrial fibrillation
    B. Ventricular fibrillation
    C. Heart failure
    D. Atrial fibrillation
    E. Hypertension
    Answer:
    C
    
    The capital of France is ____.
    A. London
    B. Paris
    C. Moscow
    D. Tokyo
    Answer:
    B
    
    Which of the following is NOT a form of official document?
    A. Opinion
    B. Resolution
    C. Notice
    D. Report
    Answer:
    
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the path will be paved with conflict, said Alvin Teng, founder of the Chinese Institute of Technology, in a keynote address at the 2022 World Economic Forum in Davos, Switzerland.
    
    Teng said that the future of AI could be “hazy” but that the path will be “pathetic” for those who fail to prepare. “AI is changing the world in ways that might be unprecedented, but the key question is how we build and manage such changes,” Teng said.
    
    The conflict between AI and humans will be the most pressing problem that governments will need to address in the future,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I'm always looking for new challenges and opportunities to grow and learn. What are some of your favorite things to do? I love to travel, read, and explore new places. What's your favorite hobby? I love to play sports, especially soccer. What's your favorite book or movie? I love to read and watch movies. I'm always looking for new experiences and adventures. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center, with a diverse population of over 2 million people. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Overall, Paris is a city of contrasts and attractions that make it a must-visit destination for anyone interested in France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, reduced costs, and improved quality of life for people.
    
    2. Enhanced personalization: AI will enable more personalized experiences for users, with algorithms that can learn from user behavior and preferences to provide tailored recommendations and services.
    
    3. Improved privacy and security: As
    


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
    Generated text:  [Name] and I'm a [Occupation] who has always been [What motivates you to be in this field?]. I'm passionate about [Why do you think your field is important?]. I've been studying [What subject matter are you currently learning?]. I'm looking forward to [What fun activity or event is in progress?]. And I'm always seeking new opportunities and challenges. What's your name, and what do you do for a living?
    **[Your Name]**  
    **Occupation:** [Your Occupation]  
    **Why You're Famous:** [You're famous because of [Your Motivation]],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and was founded in 789 by Charlemagne, who named it after his wife, Empress Charlotte of Artois. The city is located on the Seine River and has a rich history dating back to prehistoric times. It is known for its architecture, art, music, and cuisine, and has been home to several famous figures such as Charles IX and Napoleon Bonaparte. Paris is also a popular tourist destination, known for its iconic landmarks like Notre-Dame Cathedral and the Eiffel Tower, as well as for its rich cultural heritage and food scene. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several trends and advancements:
    
    1. Increased integration with other technologies: AI will continue to integrate with other technologies, such as blockchain and quantum computing, to create new applications and solve complex problems.
    
    2. Enhanced ethical and responsible AI: As concerns about AI ethics and responsibility grow, there will be an increase in efforts to develop AI that is transparent, accountable, and accountable, with a focus on maximizing societal benefits.
    
    3. Improved AI development and deployment: AI will become more scalable, efficient, and adaptable, leading to more widespread deployment and adoption in various industries.
    
    4. AI-driven innovations: AI will continue to drive


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

     first

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

    ]

     who

     is

     currently

     [

    insert

     occupation

    ].

     I

    ’m

     passionate

     about

     [

    insert

     something

     related

     to

     your

     profession

     or

     interests

    ].

     I

     love

     to

     [

    insert

     something

     related

     to

     your

     hobbies

     or

     interests

    ].

     And

     I

    ’m

     really

     [

    insert

     something

     like

     "

    ener

    getic

    ",

     "

    exc

    ited

    ",

     or

     "

    conf

    ident

    "

     based

     on

     what

     you

     would

     call

     that

    ].

     Thank

     you

     for

     asking

    ,

     and

     I

     hope

     to

     meet

     you

     soon

    !

     :)

     [

    Insert

     end

     of

     introduction

    ]

     I

    'm

     [

    insert

     first

     name

    ]

     and

     I

    'm

     a

     [

    insert

     occupation

    ]

     with

     an

     extensive

     [

    insert

     occupation

    ].

     I

    'm

     passionate

     about

     [

    insert

     something

     related

     to

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Î

    le

     de

     la

     C

    ité

     island

     in

     the

     Se

    ine

     River

    ,

     and

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    .

     It

     is

     also

     known

     as

     the

     City

     of

     Light

     and

     the

     City

     of

     Love

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     several

     trends

     are

     likely

     to

     shape

     the

     industry

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     significant

     trends

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     rise

     of

     AI

    -powered

     diagnostic

     tools

     and

     virtual

     assistants

    ,

     we

     are

     seeing

     more

     AI

     being

     used

     in

     healthcare

     to

     diagnose

     and

     treat

     patients

     more

     accurately

     and

     efficiently

    .

     This

     trend

     will

     continue

     as

     AI

     is

     increasingly

     used

     in

     medical

     research

     and

     drug

     discovery

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     the

     use

     of

     AI

     in

     various

     industries

     continues

     to

     grow

    ,

     there

     will

     be

     increased

     focus

     on

     ethical

     considerations

    .

     This

     will

     include

     areas

     like

     bias

     in

     AI

    ,

     transparency

    ,

     accountability

    ,

     and

     safety

    .

     AI

    



```python
llm.shutdown()
```
