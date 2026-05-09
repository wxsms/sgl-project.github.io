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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.48it/s]


    2026-05-09 22:59:59,299 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 22:59:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.20s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.20s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.67it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  7.04it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  7.04it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.04it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.04it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.04it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.84it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.84it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.84it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.84it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.84it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.82it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.82it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.82it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.82it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.82it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.19it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.19it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.19it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.19it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.19it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.19it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 27.84it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 27.84it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 27.84it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 27.84it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 27.84it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 27.84it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 31.28it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 31.28it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 31.28it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 31.28it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 31.28it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 31.28it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 34.45it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 34.45it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 36.31it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 36.31it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 36.31it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 36.31it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 36.31it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:06<00:00, 36.31it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 38.49it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 38.49it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 38.49it/s] 

    Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 38.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:06,  7.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:06,  7.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.88it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.52it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.27it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.13it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:04,  9.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.37 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.33it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.72it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=53.36 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.36 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.34 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.56it/s]Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.56it/s] Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.31it/s]Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.31it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.35 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.31it/s]Capturing num tokens (num_tokens=832 avail_mem=53.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.70it/s]Capturing num tokens (num_tokens=768 avail_mem=53.35 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.70it/s]Capturing num tokens (num_tokens=704 avail_mem=53.34 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.70it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.40it/s]Capturing num tokens (num_tokens=640 avail_mem=53.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.40it/s]Capturing num tokens (num_tokens=576 avail_mem=53.34 GB):  45%|████▍     | 26/58 [00:02<00:02, 13.40it/s]Capturing num tokens (num_tokens=576 avail_mem=53.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.79it/s]Capturing num tokens (num_tokens=512 avail_mem=53.32 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.79it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.34 GB):  48%|████▊     | 28/58 [00:02<00:02, 13.79it/s]Capturing num tokens (num_tokens=480 avail_mem=53.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.53it/s]Capturing num tokens (num_tokens=448 avail_mem=53.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.53it/s]Capturing num tokens (num_tokens=416 avail_mem=53.34 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.53it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.34it/s]Capturing num tokens (num_tokens=384 avail_mem=53.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.34it/s]Capturing num tokens (num_tokens=352 avail_mem=53.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.34it/s]Capturing num tokens (num_tokens=352 avail_mem=53.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 13.33it/s]Capturing num tokens (num_tokens=320 avail_mem=53.32 GB):  59%|█████▊    | 34/58 [00:03<00:01, 13.33it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.32 GB):  59%|█████▊    | 34/58 [00:03<00:01, 13.33it/s]Capturing num tokens (num_tokens=288 avail_mem=53.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.10it/s]Capturing num tokens (num_tokens=256 avail_mem=53.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.10it/s]Capturing num tokens (num_tokens=240 avail_mem=53.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.10it/s]Capturing num tokens (num_tokens=240 avail_mem=53.32 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.72it/s]Capturing num tokens (num_tokens=224 avail_mem=53.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.72it/s]

    Capturing num tokens (num_tokens=208 avail_mem=53.31 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.72it/s]Capturing num tokens (num_tokens=208 avail_mem=53.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 15.18it/s]Capturing num tokens (num_tokens=192 avail_mem=53.31 GB):  69%|██████▉   | 40/58 [00:03<00:01, 15.18it/s]Capturing num tokens (num_tokens=176 avail_mem=53.30 GB):  69%|██████▉   | 40/58 [00:03<00:01, 15.18it/s]Capturing num tokens (num_tokens=176 avail_mem=53.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.50it/s]Capturing num tokens (num_tokens=160 avail_mem=53.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.50it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.30 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.50it/s]Capturing num tokens (num_tokens=144 avail_mem=53.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=128 avail_mem=53.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=112 avail_mem=53.29 GB):  76%|███████▌  | 44/58 [00:03<00:00, 16.09it/s]Capturing num tokens (num_tokens=112 avail_mem=53.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.49it/s]Capturing num tokens (num_tokens=96 avail_mem=53.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.49it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=53.29 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.49it/s]Capturing num tokens (num_tokens=80 avail_mem=53.29 GB):  83%|████████▎ | 48/58 [00:03<00:00, 16.91it/s]Capturing num tokens (num_tokens=64 avail_mem=53.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 16.91it/s]Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 16.91it/s]Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.16it/s]Capturing num tokens (num_tokens=32 avail_mem=53.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.16it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.27 GB):  86%|████████▌ | 50/58 [00:04<00:00, 17.16it/s]Capturing num tokens (num_tokens=28 avail_mem=53.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 17.58it/s]Capturing num tokens (num_tokens=24 avail_mem=53.27 GB):  90%|████████▉ | 52/58 [00:04<00:00, 17.58it/s]Capturing num tokens (num_tokens=20 avail_mem=53.26 GB):  90%|████████▉ | 52/58 [00:04<00:00, 17.58it/s]Capturing num tokens (num_tokens=20 avail_mem=53.26 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.17it/s]Capturing num tokens (num_tokens=16 avail_mem=53.26 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.17it/s]

    Capturing num tokens (num_tokens=12 avail_mem=53.26 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.17it/s]Capturing num tokens (num_tokens=12 avail_mem=53.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 17.27it/s]Capturing num tokens (num_tokens=8 avail_mem=53.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 17.27it/s] Capturing num tokens (num_tokens=4 avail_mem=53.25 GB):  97%|█████████▋| 56/58 [00:04<00:00, 17.27it/s]Capturing num tokens (num_tokens=4 avail_mem=53.25 GB): 100%|██████████| 58/58 [00:04<00:00, 17.38it/s]Capturing num tokens (num_tokens=4 avail_mem=53.25 GB): 100%|██████████| 58/58 [00:04<00:00, 13.10it/s]


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
    Generated text:  Damian and I am a software developer. 
    
    Please, I need help with a task I have been working on. I have a table of a child's medical history, including multiple attributes. I would like to create a table of child's diagnosis. 
    
    I am not sure what to do with each attribute of the child's medical history. Each attribute should be a unique column of the new table of the child's diagnosis.
    
    For example, I want to create a table with each attribute of the child's medical history as a separate column. 
    
    I want to create a table of diagnoses for each child.
    
    I should use the child's ID
    ===============================
    Prompt: The president of the United States is
    Generated text:  a(n) ______. A. musician B. scientist C. politician D. farmer D. farmer
    
    According to the given options, the president of the United States is a politician. Therefore, the correct answer is C. politician. 
    
    To elaborate, the presidency is a position held by the president of the United States, who is the head of the executive branch of the federal government and is responsible for leading the country and carrying out the policies of the federal government. As a politician, the president represents the interests of the country and plays a crucial role in shaping the policies and laws of the federal government. The other options (musician
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Bordeaux
    C. Lyon
    D. Montpellier
    Answer: A
    
    According to the international standard, for construction companies to obtain qualifications in their respective fields, they must hold a construction enterprise qualification certificate. A. Correct B. Incorrect
    Answer: B
    
    The characteristics of a construction project include ____
    A. Multi-objectivity
    B. Multi-levelity
    C. Complexity
    D. Territoriality
    E. Industryality
    Answer: A,B,C
    
    [Multiple Choice Question] The topic sentence that should be filled in the blank is: I ______ the effect of my new product
    ===============================
    Prompt: The future of AI is
    Generated text:  about data, not emotions. – The debate on artificial intelligence (AI) is filled with claims about its efficiency, its impact on society, and its role in the world of work. Among the claims about the potential of AI, one is particularly resonant with many people, one that has not yet been fully analyzed and which has been gaining traction in the media and in its own sphere – the claim that, in the future, AI will be as emotionally intelligent as humans. In this article, we will discuss the role of data in the future of AI and how it may be used to improve the performance of AI. The focus is on


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is also known for its cuisine, with its famous French cuisine, including croissants, boudin
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This will likely lead
    


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
    Generated text:  [insert fictional character's name]. I am a [insert fictional character's age] year-old, [insert fictional character's occupation or profession] who is passionate about [insert fictional character's hobbies or interests]. I am a [insert fictional character's personality trait or quality] and have a deep understanding of [insert fictional character's field of interest or expertise]. I love [insert fictional character's hobby or activity] and strive to always keep [insert fictional character's goal or aspiration] in mind. I am [insert fictional character's age] years old and my hobbies and interests are a cornerstone of my personality. I am [insert fictional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France, located on the Seine River and the Île de la Cité. It is known for its iconic landmarks, such as the Eiffel Tower and Louvre Museum, as well as its rich history and cultural scene. The city is home to over 1.3 million people and is considered one of the world's most important cities. Paris has a unique blend of old-world charm and modernity, making it a popular tourist destination.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and depends on many factors, but some possible trends that are currently being explored or discussed include:
    
    1. Increased integration with other technologies: AI is becoming more integrated with other technologies such as the internet, blockchain, and the cloud, leading to new applications and opportunities.
    
    2. Enhanced privacy and security concerns: As AI becomes more prevalent, there will be increased concerns about privacy and security, especially with the increasing use of AI-powered systems that access personal data.
    
    3. Ethical considerations: AI will continue to evolve and develop, and ethical considerations will become increasingly important as AI systems become more complex and autonomous. There will likely be increased


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

    ],

     and

     I

     am

     [

    Age

    ].

     I

     am

     a

     [

    Career

    /

    Job

    ]

     who

     have

     been

     dedicated

     to

     [

    Occup

    ation

    /

    Job

    ]

     for

     [

    Number

     of

     Years

    ].

     Despite

     my

     professional

     life

    ,

     I

     remain

     true

     to

     my

     true

     self

    ,

     my

     hobbies

     and

     interests

     are

     [

    Any

     Inter

    ests

     or

     Inter

    ests

    ],

     and

     I

     enjoy

     [

    Anything

     I

     Like

     to

     Do

    ].

     I

     believe

     in

     the

     power

     of

     my

     personality

     to

     bring

     out

     the

     best

     in

     others

     and

     have

     a

     deep

     respect

     for

     creativity

     and

     innovation

    .

     I

     am

     passionate

     about

     [

    A

     Skill

     or

     Something

     You

     Do

     Well

    ],

     and

     I

     am

     always

     looking

     for

     ways

     to

     grow

     and

     improve

    .

     I

     am

     someone

     who

     is

     always

     learning

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     is

    :
    


    Paris

     is

     the

     capital

     of

     France

    .

     
    


    This

     is

     a

     correct

     statement

    ,

     as

     it

     accurately

     describes

     the

     capital

     city

     of

     France

    .

     The

     capital

     of

     France

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

     a

     historic

     and

     culturally

     rich

     city

    .

     It

     is

     home

     to

     many

     famous

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

     the

     Lou

    vre

     Museum

    .

     
    


    Paris

     is

     also

     known

     for

     its

     cuisine

    ,

     with

     dishes

     such

     as

     cro

    iss

    ants

    ,

     p

    ât

    é

    ,

     and

     aux

     erre

    uses

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

     is

     famous

     for

     the

     work

     of

     designers

     such

     as

     Coco

     Chanel

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

     and

     will

     likely

     continue

     to

     evolve

     significantly

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

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

     become

     more

     advanced

    ,

     allowing

     for

     greater

     automation

     of

     mundane

     tasks

    .

     This

     will

     result

     in

     fewer

     jobs

     being

     created

    ,

     but

     it

     will

     also

     mean

     that

     people

     will

     have

     more

     time

     to

     focus

     on

     more

     important

     tasks

    .
    


    2

    .

     Improved

     ethical

     AI

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     we

     will

     need

     to

     ensure

     that

     they

     are

     used

     eth

    ically

     and

     responsibly

    .

     This

     will

     require

     ongoing

     research

     and

     development

    ,

     as

     well

     as

     a

     focus

     on

     creating

     AI

     that

     is

     transparent

    ,

     accountable

    ,

     and

     aligned

     with

     ethical

     values

    .
    


    3

    .

     Improved

     accessibility

    :

     AI

    



```python
llm.shutdown()
```
