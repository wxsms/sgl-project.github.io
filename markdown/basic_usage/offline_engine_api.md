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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.53it/s]


    2026-05-11 02:00:01,021 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 02:00:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.29it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.30it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.30it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.30it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.30it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  7.70it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.48it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 13.97it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.38it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.23it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.23it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.23it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.23it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.23it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.23it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 28.17it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 28.17it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 28.17it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 28.17it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 28.17it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 28.17it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 31.91it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 35.05it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 35.05it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 35.05it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 35.05it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 35.05it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 35.05it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 37.97it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 42.92it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 42.92it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 42.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:09,  6.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:09,  6.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:08,  6.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:08,  6.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  6.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  6.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:07,  7.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:07,  7.42it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  7.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  7.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:01<00:06,  8.24it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.74it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.29it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.69it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.51it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.51it/s] Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.16it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.16it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.16it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.16it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.55it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.55it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.55it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.55it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 19.09it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:02<00:01, 19.09it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 19.09it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.09it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.09it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 19.09it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.03it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.03it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 19.03it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.05it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.05it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 19.05it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.23it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.23it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.23it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  62%|██████▏   | 36/58 [00:02<00:01, 19.23it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.68it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.68it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.68it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  67%|██████▋   | 39/58 [00:02<00:00, 19.68it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 19.99it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 19.99it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 19.99it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.99it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.11it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.11it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.11it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.11it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.14it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.14it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.14it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.14it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.35it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.35it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.35it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.35it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.34it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.34it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.34it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 20.34it/s] Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:03<00:00, 20.60it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:03<00:00, 20.60it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 15.61it/s]


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
    Generated text:  Sam and I am a little bit of a fan of cooking. My grandmother had a huge collection of pots and pans that she passed down to me. One day, I went to visit her and she was very surprised to see that I was a chef. She asked me what I did for a living, and I explained that I was a cook. She was quite impressed and said that I was a real chef. That made me very happy and I became very proud of myself. I also want to make sure that people know that I am a fan of cooking. So I decided to cook my grandmother a present. I put all the ingredients
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the head of the government and the most influential person in the country. As a president, you, as a child, have no say in the way the government is run. But as you grow older, you will have a say in how the government is run. As a president, you have the power to change the laws. You also have the power to veto a bill. Veto is the power to stop a bill or a law. In the United States, a president has to get 51 votes to win. Because of this, the president has to have a high level of
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. New York
    D. Hong Kong
    Answer:
    
    A
    
    The two main sources of water in the ocean are the ____________ and _____________.
    A. surface water, deep water
    B. surface water, seawater
    C. deep water, seawater
    D. deep water, surface water
    Answer:
    
    C
    
    The characteristic of a joint sprain is ____
    A. Swelling and pain in the joint area
    B. A fullness and tenderness in the joint area
    C. Swelling and heat in the joint area
    D. A full
    ===============================
    Prompt: The future of AI is
    Generated text:  only as good as the data it’s trained on. With this in mind, a recent study published in Nature Reviews Neuroscience found that the amount of data used in training AI systems is significantly correlated with their performance. This finding suggests that more data is required to create effective AI systems. By focusing on improving data quality, we can improve the performance of AI systems and pave the way for more ethical AI.
    In order to understand the correlation between data quality and AI performance, we compared the performance of AI systems trained on three different datasets: a standard dataset, a dataset with 10% less data, and a dataset with 100


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] and make a difference in the world. I'm always eager to learn and grow, and I'm always looking for new challenges and opportunities to grow and succeed. I'm a [job title] at [company name], and I'm always looking for ways to [job title] and make a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its food, fashion, and music scene, and is a popular tourist destination for visitors from around the world. Paris is a vibrant and dynamic city that continues to be a major center of culture and activity.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increasing focus on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential impact of their technology on society and take steps to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with human intelligence: AI is likely
    


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
    Generated text:  [Your Name], and I'm a [Your Profession] with [Your Years of Experience]. I've been working for [Your Company/Group] for [Your Years of Experience] and have been a strong leader in my department. I'm always eager to learn new things and stay up to date with my field. I have a genuine interest in helping people, and I'm always willing to provide assistance to those in need. I am always seeking ways to improve myself, and I am always looking for new opportunities to grow and succeed in my career. I'm a friendly and approachable person, and I look forward to meeting new people
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the "City of Love" and is known for its rich history, art, and cuisine. It is a popular tourist destination and a major economic center for the country. The city is home to many of France's cultural institutions, including the Louvre and Notre-Dame Cathedral. Paris is known for its various fashion and art流, and its landmarks, including the Eiffel Tower and the Arc de Triomphe, are iconic examples of French architectural art. It is also home to numerous museums and historical sites, such as the Musée d'Orsay and the Musée Rodin. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of technological progress, regulatory changes, and shifts in societal attitudes towards artificial intelligence. Here are some potential trends that could occur in the field of AI in the next few years:
    
    1. Advancements in machine learning and deep learning: As AI systems become more complex, we may see continued advancements in machine learning and deep learning algorithms, which could lead to even greater levels of intelligence and flexibility.
    
    2. Increased reliance on AI in healthcare: AI is already being used in a variety of healthcare applications, from diagnostics and treatment planning to patient care coordination and predictive analytics. As AI becomes more integrated into healthcare systems,


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

    ].

     I

    'm

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Location

    ],

     currently

     living

     in

     [

    Your

     City

    ,

     Country

    ].

     I

     enjoy

     [

    Your

     Passion

    /

    Interest

    /

    Professional

     Hobby

    /

    Op

    inion

    ].

     I

     have

     a

     love

     for

     [

    Your

     H

    obbies

    /

    Inter

    ests

    /

    Op

    inions

    /

    Tr

    av

    els

    /

    Activities

    ],

     and

     I

     try

     to

     make

     the

     most

     out

     of

     my

     time

    .

     I

     strive

     to

     be

     a

     good

    ,

     ethical

    ,

     and

     responsible

     citizen

    .

     I

     take

     pride

     in

     helping

     others

     and

     making

     a

     positive

     impact

     on

     the

     world

    .

     I

     am

     always

     looking

     for

     ways

     to

     expand

     my

     knowledge

     and

     skills

    ,

     and

     I

     am

     always

     eager

     to

     learn

     new

     things

    .

     I

     am

     an

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    A

     concise

     factual

     statement

     about

     France

    ’s

     capital

     city

     is

     that

     it

     is

     Paris

    .

     
    


    This

     statement

     is

     fact

    ually

     accurate

     as

     it

     provides

     the

     name

     of

     the

     capital

     city

     of

     France

    ,

     which

     is

     Paris

    .

     
    


    However

    ,

     it

     is

     worth

     noting

     that

     Paris

     has

     a

     more

     complex

     political

     landscape

     than

     just

     being

     its

     capital

    .

     It

     is

     also

     a

     major

     economic

     and

     cultural

     center

    ,

     home

     to

     numerous

     museums

    ,

     galleries

    ,

     and

     landmarks

    ,

     and

     plays

     a

     significant

     role

     in

     France

    's

     politics

     and

     society

    .

     
    


    Overall

    ,

     while

     Paris

     is

     the

     name

     of

     the

     capital

     city

    ,

     it

     is

     an

     important

     part

     of

     the

     country

    's

     identity

     and

     a

     significant

     contributor

     to

     its

     cultural

    ,

     economic

    ,

     and

     political

    
    
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

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     will

     continue

     to

     be

     more

     integrated

     with

     human

     intelligence

    ,

     allowing

     for

     more

     sophisticated

     and

     nuanced

     interactions

     between

     AI

     systems

     and

     humans

    .
    


    2

    .

     Improved

     transparency

     and

     explain

    ability

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     they

     will

     become

     more

     transparent

     and

     explain

    able

    ,

     making

     it

     easier

     for

     humans

     to

     understand

     how

     they

     work

     and

     make

     decisions

    .
    


    3

    .

     Enhanced

     data

     collection

     and

     analysis

    :

     AI

     will

     continue

     to

     use

     more

     advanced

     data

     collection

     and

     analysis

     techniques

    ,

     making

     it

     possible

     to

     gather

     more

     information

     and

     make

     more

     accurate

     predictions

    .
    


    4

    .

     Development

     of

     ethical

     AI

    :

     There

     will

     be

     a

     growing

     concern

    



```python
llm.shutdown()
```
