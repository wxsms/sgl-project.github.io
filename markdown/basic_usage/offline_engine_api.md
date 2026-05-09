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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.03it/s]


    2026-05-09 09:31:03,796 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 09:31:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.31it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.31it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.31it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.34it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.34it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.34it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.34it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  7.76it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  7.76it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  7.76it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  7.76it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  7.76it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 11.66it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 11.66it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 11.66it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 11.66it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 11.66it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.67it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.67it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.67it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.67it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.67it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 15.67it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 27.88it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 31.30it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 31.30it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 31.30it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 31.30it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 31.30it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 31.30it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 35.37it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 35.37it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 37.59it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 37.59it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 37.59it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 37.59it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 37.59it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 37.59it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 39.98it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 39.98it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 39.98it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 39.98it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 39.98it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 39.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.48 GB):   2%|▏         | 1/58 [00:00<00:07,  7.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.44 GB):   2%|▏         | 1/58 [00:00<00:07,  7.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.44 GB):   3%|▎         | 2/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.44 GB):   3%|▎         | 2/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.44 GB):   5%|▌         | 3/58 [00:00<00:07,  7.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.44 GB):   5%|▌         | 3/58 [00:00<00:07,  7.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.44 GB):   7%|▋         | 4/58 [00:00<00:06,  7.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.44 GB):   7%|▋         | 4/58 [00:00<00:06,  7.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.44 GB):   9%|▊         | 5/58 [00:00<00:06,  8.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.43 GB):   9%|▊         | 5/58 [00:00<00:06,  8.07it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.43 GB):  10%|█         | 6/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.42 GB):  10%|█         | 6/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.42 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.42 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.68it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.42 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.42 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.41 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.41 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.34it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.41 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.41 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.40 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.40 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.40 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.22it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.40 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.40 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.40 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.39 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.64it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.39 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.39 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.38 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.23it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.49it/s]Capturing num tokens (num_tokens=960 avail_mem=58.38 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.49it/s] Capturing num tokens (num_tokens=896 avail_mem=58.38 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.49it/s]

    Capturing num tokens (num_tokens=896 avail_mem=58.38 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.29it/s]Capturing num tokens (num_tokens=832 avail_mem=58.37 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.29it/s]Capturing num tokens (num_tokens=768 avail_mem=58.37 GB):  40%|███▉      | 23/58 [00:02<00:02, 12.29it/s]Capturing num tokens (num_tokens=768 avail_mem=58.37 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.81it/s]Capturing num tokens (num_tokens=704 avail_mem=58.37 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.81it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.36 GB):  43%|████▎     | 25/58 [00:02<00:02, 12.81it/s]Capturing num tokens (num_tokens=640 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:02<00:02, 13.46it/s]Capturing num tokens (num_tokens=576 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:02<00:02, 13.46it/s]Capturing num tokens (num_tokens=512 avail_mem=58.35 GB):  47%|████▋     | 27/58 [00:02<00:02, 13.46it/s]Capturing num tokens (num_tokens=480 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:02<00:02, 13.46it/s]Capturing num tokens (num_tokens=448 avail_mem=58.36 GB):  47%|████▋     | 27/58 [00:02<00:02, 13.46it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.36 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.84it/s]Capturing num tokens (num_tokens=416 avail_mem=58.36 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.84it/s]Capturing num tokens (num_tokens=384 avail_mem=58.36 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.84it/s]Capturing num tokens (num_tokens=384 avail_mem=58.36 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.07it/s]Capturing num tokens (num_tokens=352 avail_mem=58.35 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.07it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.34 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.07it/s]Capturing num tokens (num_tokens=320 avail_mem=58.34 GB):  60%|██████    | 35/58 [00:02<00:01, 16.98it/s]Capturing num tokens (num_tokens=288 avail_mem=58.34 GB):  60%|██████    | 35/58 [00:02<00:01, 16.98it/s]Capturing num tokens (num_tokens=256 avail_mem=58.34 GB):  60%|██████    | 35/58 [00:02<00:01, 16.98it/s]Capturing num tokens (num_tokens=240 avail_mem=58.34 GB):  60%|██████    | 35/58 [00:02<00:01, 16.98it/s]Capturing num tokens (num_tokens=240 avail_mem=58.34 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.02it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.02it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.02it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.02it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 19.23it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 19.23it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:03<00:00, 19.23it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 18.04it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 18.04it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:03<00:00, 18.04it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.80it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.80it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.80it/s] Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 17.33it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 17.33it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 17.33it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.25it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.25it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:03<00:00, 17.25it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:03<00:00, 16.99it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 16.99it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 16.99it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.86it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.86it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 16.86it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.05it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.05it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.05it/s] Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:04<00:00, 17.41it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:04<00:00, 17.41it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:04<00:00, 13.85it/s]


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
    Generated text:  Shala. I'm from Idaho, United States. I'm currently living in New York City, where I study English, and I'm currently a student at The New School. I'm a published author, but I'm not a writer. I'm a former performance artist and a singer. I have a passion for creating, writing, performing, and creating a sustainable lifestyle through personal and environmental activism. I'm committed to making a positive impact in my community, and I'm always looking for ways to improve my life and make a difference in the world. I like to think of myself as someone who is self-motivated and has
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. The president of the United States is the head of state of the United States. The president of the United States is in charge of the government and the executive branch of the government. They are the most powerful person in the United States.
    Does this next sentence follow, given the above text?
    The president of the United States has no powers.
    
    Pick from:
    (i). yes
    (ii). no
    (i). yes
    
    The sentence "The president of the United States has no powers" does not follow from the given text. The text states that the president of the United States is in charge of the government and the executive branch,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in the world and the third largest in the European Union. It's not only famous for its grandiose architecture and its famous landmarks, but also for its unique cuisine and wines. This place has something for everyone - from the casual and friendly local to the expensive and refined. In this series we take a look at the city from different perspectives: urban design, the city's past, the city's present, and the city's future. Each of these perspectives tells a story about Paris in different ways, and the stories are presented through the lens of the city's local residents. With a little bit
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of humans, and so is the future of data privacy. Privacy and data breaches are becoming more and more important and this is why the European Union’s recently adopted Regulation on the Protection of Personal Data (POPD) is so important. It is the first time that the European Parliament has published a formal statement on data protection and privacy.
    The Regulation establishes a framework for the protection of data, specifies the rights of individuals with respect to their personal data, and establishes a general principle of data protection. It also establishes the conditions for the processing of personal data and defines the rights and responsibilities of data controllers and processors.
    At its core


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a cultural and economic center of France and a major tourist destination. It is home to many famous museums, including the Louvre and the Musée d'Orsay. The city is also known for its cuisine, with many famous French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more accurate and efficient decision-making, improved user experience, and the ability to automate complex tasks. Additionally, AI will likely become more integrated with other technologies, such as the Internet of Things (IoT), to create more connected and interconnected systems. AI will also continue to be used in areas such as healthcare, finance, and transportation, where it can help to improve efficiency, reduce costs, and provide better outcomes for users. Finally, AI will likely continue to be used in ways that
    


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
    Generated text:  [Name], and I am a [job title] at [company]. I specialize in [your profession or hobby]. [Your character's background and personality traits are included here, but they should be kept to a minimum to ensure the self-introduction is clear and concise. For example: I am a [age], have [number of years] years of experience, and have a passion for [anything related to your profession or hobby].]
    Hello, my name is [Name], and I am a [job title] at [company]. I specialize in [your profession or hobby]. I am a [age], have [number of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Light" and is a cosmopolitan and vibrant city with a rich cultural heritage. Paris is located on the Seine River and is the cultural and political center of the country. The city is home to numerous museums, theaters, and other cultural institutions, as well as a large and diverse population of citizens. It is also known for its stunning architecture and picturesque views of the city and its surroundings. Paris is a city of contrasts, with both history and modernity. It is a popular tourist destination and is home to many world-renowned landmarks, including Notre-Dame Cathedral, the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve and advance at a rapid pace. Here are some potential trends that experts predict for the future of AI:
    
    1. Increased Integration with Humans: One of the biggest trends in AI is the growing integration of AI into human systems. AI is becoming more integrated with human systems such as healthcare, finance, and transportation. AI is being used to improve decision-making, automate tasks, and enhance human capabilities.
    
    2. Improved Explainability: AI models are becoming more sophisticated, and they are becoming better at explaining their predictions to humans. As AI models become more sophisticated, they will become more transparent and understandable.
    
    3. AI is


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

    __.

     I

     am

     a

     software

     engineer

     at

     __

    __,

     and

     I

     have

     been

     working

     here

     for

     __

     years

    .

     I

     am

     currently

     working

     on

     a

     project

     that

     involves

     __

    __.

     I

     am

     a

     __

    __,

     __

    __.

     My

     expertise

     lies

     in

     __

    __.

     I

     am

     passionate

     about

     __

    __,

     __

    __,

     and

     __

    __.

     I

     am

     always

     looking

     for

     ways

     to

     __

    __.

     I

     enjoy

     __

    __.

     I

     am

     always

     eager

     to

     learn

     new

     things

    ,

     and

     I

     am

     a

     __

    __.

     I

     have

     a

     __

    __,

     and

     I

     enjoy

     __

    __.

     I

     have

     a

     lot

     of

     __

    __.

     If

     you

     have

     any

     questions

     about

     me

    ,

     please

     don

    't

     hesitate

     to

     ask

    .

     Have

     a

     good

     day

    !

     [

    insert

     your

     name

    ]

     [

    insert

     company

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

     is

     a

     factual

     statement

    .

     How

     can

     I

     assist

     you

     further

    ?

     Please

     provide

     more

     details

     or

     ask

     a

     specific

     question

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


    1

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     AI

     is

     already

     being

     integrated

     into

     various

     aspects

     of

     our

     lives

    ,

     such

     as

     smart

     homes

    ,

     self

    -driving

     cars

    ,

     and

     virtual

     assistants

    .

     In

     the

     future

    ,

     we

     may

     see

     even

     more

     widespread

     integration

    ,

     from

     smart

     cities

     to

     self

    -s

    ufficient

     homes

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     assist

     doctors

     in

     diagnoses

    ,

     to

     predict

     disease

     progression

    ,

     and

     to

     optimize

     treatment

     plans

    .

     In

     the

     future

    ,

     we

     may

     see

     AI

     being

     used

     to

     improve

     patient

     care

     in

     other

     areas

     as

     well

    .
    


    3

    .

     Development

     of

     AI

     for

     autonomous

     systems

    :

     AI

     is

    



```python
llm.shutdown()
```
