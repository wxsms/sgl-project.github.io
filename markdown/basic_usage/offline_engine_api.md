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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.43it/s]


    2026-05-02 09:24:29,091 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 09:24:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:19,  2.59it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.84it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.84it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.84it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.84it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.84it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.62it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.62it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.62it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.62it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.27it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 13.27it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 17.53it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.66it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.66it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.66it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.66it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.66it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.66it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 27.28it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 27.28it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 27.28it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 27.28it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 27.28it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:00, 27.28it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 31.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 31.43it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 31.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 31.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 31.43it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 31.43it/s]

    Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 35.54it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 35.54it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 35.54it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 35.54it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 35.54it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 35.54it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]

    Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:06<00:00, 38.07it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 41.97it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 41.97it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 41.97it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 41.97it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 41.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.39 GB):   2%|▏         | 1/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.36 GB):   2%|▏         | 1/58 [00:00<00:06,  8.28it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.36 GB):   3%|▎         | 2/58 [00:00<00:06,  8.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.35 GB):   3%|▎         | 2/58 [00:00<00:06,  8.49it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.35 GB):   3%|▎         | 2/58 [00:00<00:06,  8.49it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.35 GB):   7%|▋         | 4/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.35 GB):   7%|▋         | 4/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.34 GB):   7%|▋         | 4/58 [00:00<00:04, 12.00it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.34 GB):  10%|█         | 6/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.33 GB):  10%|█         | 6/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.33 GB):  10%|█         | 6/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.33 GB):  10%|█         | 6/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.33 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.33 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.51it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.69it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=56.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.33 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.93 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.27 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.27 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.26 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.64it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=56.42 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.42 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.07it/s]Capturing num tokens (num_tokens=960 avail_mem=56.44 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.07it/s] Capturing num tokens (num_tokens=896 avail_mem=56.44 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.07it/s]

    Capturing num tokens (num_tokens=896 avail_mem=56.44 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=832 avail_mem=57.25 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=768 avail_mem=57.25 GB):  40%|███▉      | 23/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=768 avail_mem=57.25 GB):  43%|████▎     | 25/58 [00:01<00:02, 15.99it/s]Capturing num tokens (num_tokens=704 avail_mem=56.48 GB):  43%|████▎     | 25/58 [00:01<00:02, 15.99it/s]Capturing num tokens (num_tokens=640 avail_mem=56.48 GB):  43%|████▎     | 25/58 [00:01<00:02, 15.99it/s]

    Capturing num tokens (num_tokens=640 avail_mem=56.48 GB):  47%|████▋     | 27/58 [00:01<00:02, 14.71it/s]Capturing num tokens (num_tokens=576 avail_mem=56.48 GB):  47%|████▋     | 27/58 [00:01<00:02, 14.71it/s]Capturing num tokens (num_tokens=512 avail_mem=57.23 GB):  47%|████▋     | 27/58 [00:01<00:02, 14.71it/s]Capturing num tokens (num_tokens=512 avail_mem=57.23 GB):  50%|█████     | 29/58 [00:01<00:02, 14.48it/s]Capturing num tokens (num_tokens=480 avail_mem=57.24 GB):  50%|█████     | 29/58 [00:01<00:02, 14.48it/s]

    Capturing num tokens (num_tokens=448 avail_mem=56.53 GB):  50%|█████     | 29/58 [00:01<00:02, 14.48it/s]Capturing num tokens (num_tokens=448 avail_mem=56.53 GB):  53%|█████▎    | 31/58 [00:01<00:01, 14.08it/s]Capturing num tokens (num_tokens=416 avail_mem=56.53 GB):  53%|█████▎    | 31/58 [00:01<00:01, 14.08it/s]Capturing num tokens (num_tokens=384 avail_mem=56.53 GB):  53%|█████▎    | 31/58 [00:02<00:01, 14.08it/s]

    Capturing num tokens (num_tokens=384 avail_mem=56.53 GB):  57%|█████▋    | 33/58 [00:02<00:01, 14.07it/s]Capturing num tokens (num_tokens=352 avail_mem=57.23 GB):  57%|█████▋    | 33/58 [00:02<00:01, 14.07it/s]Capturing num tokens (num_tokens=320 avail_mem=56.57 GB):  57%|█████▋    | 33/58 [00:02<00:01, 14.07it/s]Capturing num tokens (num_tokens=320 avail_mem=56.57 GB):  60%|██████    | 35/58 [00:02<00:01, 14.02it/s]Capturing num tokens (num_tokens=288 avail_mem=56.57 GB):  60%|██████    | 35/58 [00:02<00:01, 14.02it/s]

    Capturing num tokens (num_tokens=256 avail_mem=56.56 GB):  60%|██████    | 35/58 [00:02<00:01, 14.02it/s]Capturing num tokens (num_tokens=256 avail_mem=56.56 GB):  64%|██████▍   | 37/58 [00:02<00:01, 14.10it/s]Capturing num tokens (num_tokens=240 avail_mem=57.22 GB):  64%|██████▍   | 37/58 [00:02<00:01, 14.10it/s]Capturing num tokens (num_tokens=224 avail_mem=56.61 GB):  64%|██████▍   | 37/58 [00:02<00:01, 14.10it/s]

    Capturing num tokens (num_tokens=224 avail_mem=56.61 GB):  67%|██████▋   | 39/58 [00:02<00:01, 13.84it/s]Capturing num tokens (num_tokens=208 avail_mem=56.61 GB):  67%|██████▋   | 39/58 [00:02<00:01, 13.84it/s]Capturing num tokens (num_tokens=192 avail_mem=57.21 GB):  67%|██████▋   | 39/58 [00:02<00:01, 13.84it/s]Capturing num tokens (num_tokens=192 avail_mem=57.21 GB):  71%|███████   | 41/58 [00:02<00:01, 14.54it/s]Capturing num tokens (num_tokens=176 avail_mem=56.66 GB):  71%|███████   | 41/58 [00:02<00:01, 14.54it/s]

    Capturing num tokens (num_tokens=160 avail_mem=56.66 GB):  71%|███████   | 41/58 [00:02<00:01, 14.54it/s]Capturing num tokens (num_tokens=160 avail_mem=56.66 GB):  74%|███████▍  | 43/58 [00:02<00:01, 14.04it/s]Capturing num tokens (num_tokens=144 avail_mem=57.20 GB):  74%|███████▍  | 43/58 [00:02<00:01, 14.04it/s]Capturing num tokens (num_tokens=128 avail_mem=57.20 GB):  74%|███████▍  | 43/58 [00:02<00:01, 14.04it/s]Capturing num tokens (num_tokens=128 avail_mem=57.20 GB):  78%|███████▊  | 45/58 [00:02<00:00, 14.70it/s]Capturing num tokens (num_tokens=112 avail_mem=56.70 GB):  78%|███████▊  | 45/58 [00:02<00:00, 14.70it/s]

    Capturing num tokens (num_tokens=96 avail_mem=56.83 GB):  78%|███████▊  | 45/58 [00:03<00:00, 14.70it/s] Capturing num tokens (num_tokens=96 avail_mem=56.83 GB):  81%|████████  | 47/58 [00:03<00:00, 15.23it/s]Capturing num tokens (num_tokens=80 avail_mem=57.19 GB):  81%|████████  | 47/58 [00:03<00:00, 15.23it/s]Capturing num tokens (num_tokens=64 avail_mem=56.75 GB):  81%|████████  | 47/58 [00:03<00:00, 15.23it/s]Capturing num tokens (num_tokens=64 avail_mem=56.75 GB):  84%|████████▍ | 49/58 [00:03<00:00, 14.85it/s]Capturing num tokens (num_tokens=48 avail_mem=57.18 GB):  84%|████████▍ | 49/58 [00:03<00:00, 14.85it/s]

    Capturing num tokens (num_tokens=32 avail_mem=57.17 GB):  84%|████████▍ | 49/58 [00:03<00:00, 14.85it/s]Capturing num tokens (num_tokens=32 avail_mem=57.17 GB):  88%|████████▊ | 51/58 [00:03<00:00, 15.48it/s]Capturing num tokens (num_tokens=28 avail_mem=56.76 GB):  88%|████████▊ | 51/58 [00:03<00:00, 15.48it/s]Capturing num tokens (num_tokens=24 avail_mem=57.17 GB):  88%|████████▊ | 51/58 [00:03<00:00, 15.48it/s]Capturing num tokens (num_tokens=24 avail_mem=57.17 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.28it/s]Capturing num tokens (num_tokens=20 avail_mem=56.78 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.28it/s]

    Capturing num tokens (num_tokens=16 avail_mem=56.78 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.28it/s]Capturing num tokens (num_tokens=16 avail_mem=56.78 GB):  95%|█████████▍| 55/58 [00:03<00:00, 16.03it/s]Capturing num tokens (num_tokens=12 avail_mem=57.16 GB):  95%|█████████▍| 55/58 [00:03<00:00, 16.03it/s]Capturing num tokens (num_tokens=8 avail_mem=56.79 GB):  95%|█████████▍| 55/58 [00:03<00:00, 16.03it/s] Capturing num tokens (num_tokens=8 avail_mem=56.79 GB):  98%|█████████▊| 57/58 [00:03<00:00, 16.00it/s]Capturing num tokens (num_tokens=4 avail_mem=57.15 GB):  98%|█████████▊| 57/58 [00:03<00:00, 16.00it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.15 GB): 100%|██████████| 58/58 [00:03<00:00, 15.45it/s]


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
    Generated text:  Ana.
    
    I'm studying for my physics major at the University of Michigan. I have a degree in math and a degree in physics. I have been an active member of the Physics Club at the University of Michigan for four years.
    
    I play soccer and have been playing since I was a child. I have been in both the men's and women's soccer teams.
    
    I am currently a member of the University of Michigan's Mathematics Club. I was a member of the University of Michigan's Physics Club for several years. I was the president of the University of Michigan's Physics Club for three years. I served as a member of the University of
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 32 years older than the president of Peru. The president of Peru is 20 years younger than the president of China. If the president of China will be 70 years old in 10 years, how old is the president of the United States?
    
    To determine the age of the president of the United States, we need to follow the information given step by step.
    
    1. **Identify the current age of the president of Peru:**
       - The president of Peru is currently 20 years younger than the president of China.
       - If the president of China will be 70 years old in
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) Rome
    
    C) Moscow
    
    D) Istanbul
    
    To determine the capital of France, we can use our knowledge of world geography and pop culture. France is a country, and the capital city of France is the city where the French language was first spoken and where the French Revolution began. Paris is the capital of France and is the largest and most populous city in the country.
    
    Here are the steps to identify the capital of France:
    
    1. Identify the capital of France. The capital city of France is Paris.
    2. Verify the correctness of the answer by checking the options provided. Options A, B,
    ===============================
    Prompt: The future of AI is
    Generated text:  here and you need to know how to grow it
    
    AI is here to stay. It has the potential to change the way we live, work and play. With the rapid growth of the field, businesses and governments are looking for ways to understand and harness the power of AI. In this article, we discuss the basics of AI, how it’s used, and the future of the field. We also discuss what it means to grow AI and how to do so.
    
    Understanding AI
    
    AI is a broad field that encompasses a variety of technologies, including machine learning, natural language processing, computer vision, and robotics. The goal of AI is


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a popular tourist destination and a major cultural hub in Europe. It is home to many world-renowned museums, theaters, and art galleries. The city is also known for its annual festivals and events, such as the Eiffel Tower Parade and the Louvre Festival. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in shaping the future of work, with automation and artificial intelligence becoming more prevalent in industries such as manufacturing, finance, and healthcare. Finally, AI is likely to continue to be a topic of debate and discussion, with concerns about privacy, bias, and the potential for AI to replace human workers. Overall, the future of AI
    


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
    Generated text:  [Name], and I'm a [age] year-old artist. I'm originally from [country], but I grew up in [city], where my mother and father taught me the basics of [artist's medium]. My goal is to create beautiful, artful work that touches people's hearts and souls. I'm passionate about using my art to promote positivity and kindness. I'm also interested in sharing my own life experiences and how they have influenced my work. What inspires me to create what I do? What's something I love doing that I've found particularly rewarding? What's something that makes you feel blessed to be alive? Write
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    She has a population of over 2 million and is known for its rich history, beautiful architecture, and famous landmarks such as the Eiffel Tower and the Louvre Museum. The city also has a vibrant culture, with its many festivals and events throughout the year, and is one of the most important cultural capitals in the world. Paris is located on the Île de la Cité and is situated on the right bank of the Seine river. It is the largest city in France by area and the sixth largest city in the world by population. The city has a long history dating back to the Roman Empire and has
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  extremely promising and involves a wide range of possibilities. Some possible trends include:
    
    1. Increased AI ethics and transparency: As AI becomes more integrated into our daily lives, there will be an increased emphasis on AI ethics and transparency. This means that we will need to ensure that AI systems are designed and used in ways that are fair and unbiased, and that they do not perpetuate inequalities in society.
    
    2. Increased focus on AI safety: As we become more dependent on AI for various tasks, there will be an increased focus on ensuring that AI systems are safe and secure. This means that we will need to develop new security measures and technologies to


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

     [

    insert

     first

     name

    ]

     at

     [

    insert

     employer

    ].

     I

    'm

     passionate

     about

     [

    insert

     a

     specific

     skill

     or

     interest

     that

     sets

     me

     apart

     from

     others

    ].

     I

     am

     confident

     in

     my

     abilities

     to

     [

    insert

     a

     specific

     achievement

     or

     accomplishment

     that

     demonstrates

     this

     skill

    ].


    My

     love

     for

     learning

     and

     self

    -im

    pro

    vement

     has

     led

     me

     to

     [

    insert

     a

     specific

     hobby

     or

     activity

     that

     I

     enjoy

     or

     have

     developed

     into

     a

     passion

    ].

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     opportunities

     to

     grow

     and

     develop

     as

     a

     person

    .


    I

    'm

     excited

     to

     be

     here

     at

     [

    insert

     employer

    's

     name

    ]

     and

     have

     the

     opportunity

     to

     share

     my

     knowledge

     and

     experience

     with

     you

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

    -central

     part

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     sixth

    -largest

     city

     in

     the

     world

     by

     population

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    ,

     as

     well

     as

     its

     vibrant

     culture

    ,

     including

     the

     French

     Quarter

     and

     Mont

    mart

    re

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

     industry

    ,

     which

     has

     a

     long

     history

     dating

     back

     to

     the

     

    1

    9

    th

     century

    .

     Paris

     is

     the

     political

    ,

     economic

    ,

     and

     cultural

     center

     of

     France

     and

     is

     home

     to

     many

     of

     the

     country

    's

     most

     famous

     landmarks

     and

     attractions

    .

     With

     a

     population

     of

     over

     

    2

     million

     people

    ,

     Paris

     is

     a

     bustling

     met

    ropolis

     that

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     like

     it

     is

     very

     bright

     and

     bright

    !

     Here

     are

     some

     of

     the

     potential

     trends

     in

     AI

     that

     could

     come

     to

     fruition

    :
    


    1

    .

     Increased

     AI

     integration

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     AI

     integrated

     into

     everyday

     life

    .

     This

     could

     include

     facial

     recognition

    ,

     self

    -driving

     cars

    ,

     and

     personalized

     recommendations

     from

     AI

    -powered

     search

     engines

    .
    


    2

    .

     AI

     ethics

     and

     privacy

    :

     As

     AI

     becomes

     more

     integrated

     into

     society

    ,

     we

     will

     need

     to

     address

     the

     ethical

     and

     privacy

     concerns

     that

     arise

    .

     This

     could

     involve

     creating

     new

     regulations

     and

     standards

     for

     AI

     development

     and

     use

    ,

     as

     well

     as

     addressing

     concerns

     around

     data

     privacy

     and

     security

    .
    


    3

    .

     AI

    -driven

     healthcare

    :

     AI

     is

    



```python
llm.shutdown()
```
