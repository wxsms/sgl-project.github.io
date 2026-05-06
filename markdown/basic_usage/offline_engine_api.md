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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.42it/s]


    2026-05-06 15:07:21,839 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 15:07:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.44it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.44it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.44it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:20,  2.44it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:11,  4.28it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:11,  4.28it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:11,  4.28it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:11,  4.28it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.52it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.52it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.52it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.52it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.20it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.20it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.20it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 14.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 14.92it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 14.92it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 14.92it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 14.92it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 14.92it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.23it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.23it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.23it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.23it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.23it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:01, 20.49it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:01, 20.49it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:06<00:01, 20.49it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:06<00:01, 20.49it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:00, 25.64it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 30.22it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 30.22it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 30.22it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 30.22it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 30.22it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 30.22it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 33.63it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 33.63it/s] 

    Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 33.63it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 33.63it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 33.63it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 33.63it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 36.70it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 41.80it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 41.80it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.43 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.43 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.85it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.42 GB):  10%|█         | 6/58 [00:00<00:06,  8.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.44it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.59it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.07it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.18it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.97it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.97it/s] Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 13.97it/s]

    Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.05it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.05it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.05it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 16.12it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.36it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.36it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.36it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.36it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.32it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.32it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 18.32it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.68it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.68it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.68it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 18.68it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.16it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.16it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.16it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.20it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.20it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:02<00:01, 19.20it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 19.19it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 19.19it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 19.19it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 19.19it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  74%|███████▍  | 43/58 [00:02<00:00, 19.66it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 19.66it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 19.66it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.72it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.72it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 19.72it/s] Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 19.73it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 19.73it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 19.73it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 19.73it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 19.91it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 19.91it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 19.91it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 19.91it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 20.12it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 20.12it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 20.12it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 20.12it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 20.10it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 20.10it/s] Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 20.10it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:03<00:00, 15.57it/s]


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
    Generated text:  Alexander R. S. Plebačka. I am a professor at the Faculty of Electrical Engineering, University of Ljubljana, Slovenia. My research interests are in algebraic topology, especially on mapping class groups and their actions on moduli spaces. In particular, I study the mapping class group as a generalization of the classical group. I am interested in the topology of singularities (especially toric fibrations and four-folds), noncommutative geometry, and the geometry of moduli spaces. I am also interested in the algebraic geometry of moduli spaces of curves, their Grothendieck rings
    ===============================
    Prompt: The president of the United States is
    Generated text:  a kind of ____.
    A. state organ
    B. executive organ
    C. political party
    D. state power
    Answer: A
    
    Which of the following statements about the public security organ's "People's Republic of China Public Security Police Rank Regulations" is correct?
    A. It is a general law.
    B. It is a normative legal document.
    C. It is a procedural law.
    D. It is a substantive legal document.
    Answer: B
    
    According to the provisions of the "Notice of the Supreme People's Court on Several Issues Concerning the Trial of Dispute Cases in Civil and Commercial Cases," when the court
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A: Paris B: London C: New York D: Moscow
    
    To determine the capital of France, we need to recall the official capital of France. The capital of France is Paris.
    
    Let's break it down step by step:
    
    1. Identify the capital of France: Paris.
    2. Check the given options: London, New York, and Moscow.
    3. Determine the correct capital: Paris.
    
    Therefore, the correct answer is: \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  promising and it is essential for businesses to stay ahead of the curve. In this article, we will explore how to identify the future of AI, how it has already changed, and how it will continue to change in the future.
    1. Predicting the Future of AI
    The future of AI is unpredictable and unpredictable. It is difficult to predict the exact changes that AI will bring to the world, but there are several factors that make it possible.
    Firstly, AI is rapidly evolving, and new technologies are constantly emerging. This means that AI is constantly changing and developing. The best way to predict the future of AI is to study the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character here]. I enjoy [insert a short description of your character's interests or hobbies here]. I'm always looking for new experiences and learning opportunities, so I'm always eager to learn more about the world around me. What's your favorite hobby or activity? I'm always looking for new challenges and experiences, so I'm always eager to learn more about the world around me. What's your favorite book or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is accurate and brief, providing the essential information about the capital city of France. It is a widely recognized and well-known city in the world, known for its rich history, beautiful architecture, and vibrant culture. Paris is the largest city in France and is home to many of the country's most famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its fashion industry, art scene, and food culture, making it a popular tourist destination for millions of visitors each year. 
    
    Therefore, the statement accurately and concisely describes the capital city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for workers and businesses.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more sophisticated applications in healthcare, including personalized medicine
    


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
    Generated text:  [Name] and I'm a [job title] at [company name], with a passion for [job title]. I've always been interested in [occupation] and [career objective], and I'm constantly learning and growing as a professional. I'm a [character trait] and [character trait] to my heart, and I'm always looking for ways to contribute to [company name] and help make a positive impact. How can I make a positive impact in my career? As an [occupation] with a passion for [career objective], I aim to help [company name] achieve its goals by [job title] and [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    You are an AI assistant that helps you understand the reasons behind certain conclusions. Try to provide explanations in simple words with no new information. To see how your explanation has been used, switch to advanced mode.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable, but here are some potential trends that could occur:
    
    1. Increased autonomy and self-learning capabilities: As AI becomes more capable, it's possible that it could develop autonomous behavior, allowing it to make decisions and take actions without human intervention. This could lead to a more seamless and efficient interaction between humans and machines.
    
    2. Greater emphasis on ethics and fairness: As AI is used in a wider range of applications, there's a greater emphasis on ethical considerations and fairness. This could lead to the development of more sophisticated algorithms that take into account variables such as bias, discrimination, and privacy.
    
    3. Cognitive computing: AI


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

    name

    ],

     and

     I

     am

     a

     [

    major

    ity

     of

    ]

     young

    ,

     charismatic

    ,

     and

     adventurous

     [

    person

    ality

    ]

     who

     has

     [

    number

     of

     years

     of

     experience

    ]

     years

     of

     experience

     in

     the

     [

    industry

    ]

     field

    .

     I

     am

     passionate

     about

     [

    what

     you

     do

     best

    ],

     and

     my

     [

    strength

     or

     skill

    ]

     is

     [

    describe

     your

     strength

     or

     skill

    ].

     I

     am

     [

    average

     or

     energetic

    ]

     in

     the

     sun

    ,

     [

    able

     to

    ]

     [

    describe

     a

     skill

     or

     accomplishment

    ].

     I

     am

     very

     [

    ener

    getic

    ,

     optimistic

    ,

     proactive

    ,

     or

     friendly

    ],

     and

     always

     [

    describe

     a

     behavior

     or

     personality

     trait

    ].

     I

     am

     [

    gender

    less

    ]

     and

     I

     am

     [

    career

     or

     hobby

    ].

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Lo

    ire

     Valley

     region

    ,

     and

     is

     the

     largest

     city

     in

     both

     the

     country

     and

     Europe

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     architecture

     and

     fashion

     scene

    .

     It

     is

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     historical

     sites

    ,

     including

     the

     Lou

    vre

     Museum

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     distinctive

     red

     and

     white

     lines

     and

     its

     bustling

     nightlife

    .

     Paris

     is

     the

     

    9

    th

     most

     populous

     city

     in

     the

     world

     and

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    ."

     It

     is

     also

     the

     world

    's

     eighth

    -largest

     city

     by

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     promising

    ,

     with

     many

     potential

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     notable

     trends

     include

    :
    


    1

    .

     Increased

     integration

     with

     human

     decision

    -making

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     process

     and

     interpret

     more

     complex

     information

    ,

     leading

     to

     a

     greater

     reliance

     on

     human

     oversight

     and

     decision

    -making

    .

     This

     could

     result

     in

     a

     more

     human

    -centered

     AI

     that

     is

     better

     able

     to

     make

     informed

     decisions

     based

     on

     a

     range

     of

     factors

    .
    


    2

    .

     Personal

    ization

     and

     adaptive

     learning

    :

     AI

     is

     already

     making

     significant

     progress

     in

     personal

    izing

     and

     adapting

     its

     responses

     to

     users

    ,

     but

     as

     it

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     even

     greater

    



```python
llm.shutdown()
```
