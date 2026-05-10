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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.05it/s]


    2026-05-10 11:09:11,023 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 11:09:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:45,  3.95s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.59it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.59it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.59it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.59it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  8.09it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  8.09it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  8.09it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  8.09it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  8.09it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 12.05it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 12.05it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 12.05it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 12.05it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 16.01it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 16.01it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 16.01it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 16.01it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 16.01it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 16.01it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:00, 26.03it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:00, 26.03it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:00, 26.03it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:00, 26.03it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:00, 26.03it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:00, 26.03it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 34.26it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 34.26it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 43.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   2%|▏         | 1/58 [00:00<00:07,  7.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.54it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.98it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.58it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.58it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.37 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.37 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.37 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.16it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.37 GB):  21%|██        | 12/58 [00:01<00:06,  7.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.36 GB):  21%|██        | 12/58 [00:01<00:06,  7.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.36 GB):  22%|██▏       | 13/58 [00:01<00:05,  7.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.28 GB):  22%|██▏       | 13/58 [00:01<00:05,  7.58it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.28 GB):  24%|██▍       | 14/58 [00:01<00:05,  7.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.71 GB):  24%|██▍       | 14/58 [00:01<00:05,  7.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.71 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.70 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.96it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=57.70 GB):  28%|██▊       | 16/58 [00:01<00:05,  8.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.70 GB):  28%|██▊       | 16/58 [00:01<00:05,  8.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.70 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.70 GB):  31%|███       | 18/58 [00:02<00:04,  9.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.69 GB):  31%|███       | 18/58 [00:02<00:04,  9.27it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=57.69 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.69 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.69 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.67 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.13it/s]

    Capturing num tokens (num_tokens=960 avail_mem=57.69 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.13it/s] Capturing num tokens (num_tokens=960 avail_mem=57.69 GB):  38%|███▊      | 22/58 [00:02<00:03,  9.75it/s]Capturing num tokens (num_tokens=896 avail_mem=57.68 GB):  38%|███▊      | 22/58 [00:02<00:03,  9.75it/s]Capturing num tokens (num_tokens=832 avail_mem=57.68 GB):  38%|███▊      | 22/58 [00:02<00:03,  9.75it/s]

    Capturing num tokens (num_tokens=832 avail_mem=57.68 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.50it/s]Capturing num tokens (num_tokens=768 avail_mem=57.68 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.50it/s]Capturing num tokens (num_tokens=704 avail_mem=57.67 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.50it/s]Capturing num tokens (num_tokens=704 avail_mem=57.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.20it/s]Capturing num tokens (num_tokens=640 avail_mem=57.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.20it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.67 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.20it/s]Capturing num tokens (num_tokens=576 avail_mem=57.67 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.68it/s]Capturing num tokens (num_tokens=512 avail_mem=57.65 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.68it/s]Capturing num tokens (num_tokens=480 avail_mem=57.67 GB):  48%|████▊     | 28/58 [00:03<00:02, 11.68it/s]

    Capturing num tokens (num_tokens=480 avail_mem=57.67 GB):  52%|█████▏    | 30/58 [00:03<00:02, 12.08it/s]Capturing num tokens (num_tokens=448 avail_mem=57.67 GB):  52%|█████▏    | 30/58 [00:03<00:02, 12.08it/s]Capturing num tokens (num_tokens=416 avail_mem=57.67 GB):  52%|█████▏    | 30/58 [00:03<00:02, 12.08it/s]Capturing num tokens (num_tokens=416 avail_mem=57.67 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.35it/s]Capturing num tokens (num_tokens=384 avail_mem=57.66 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.35it/s]

    Capturing num tokens (num_tokens=352 avail_mem=57.66 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.35it/s]Capturing num tokens (num_tokens=352 avail_mem=57.66 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.65it/s]Capturing num tokens (num_tokens=320 avail_mem=57.65 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.65it/s]Capturing num tokens (num_tokens=288 avail_mem=57.65 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.65it/s]

    Capturing num tokens (num_tokens=288 avail_mem=57.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.94it/s]Capturing num tokens (num_tokens=256 avail_mem=57.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.94it/s]Capturing num tokens (num_tokens=240 avail_mem=57.65 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.94it/s]Capturing num tokens (num_tokens=240 avail_mem=57.65 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.27it/s]Capturing num tokens (num_tokens=224 avail_mem=57.64 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.27it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.64 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.27it/s]Capturing num tokens (num_tokens=208 avail_mem=57.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.19it/s]Capturing num tokens (num_tokens=192 avail_mem=57.64 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.19it/s]Capturing num tokens (num_tokens=176 avail_mem=57.63 GB):  69%|██████▉   | 40/58 [00:04<00:01, 13.19it/s]

    Capturing num tokens (num_tokens=176 avail_mem=57.63 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.22it/s]Capturing num tokens (num_tokens=160 avail_mem=57.18 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.22it/s]Capturing num tokens (num_tokens=144 avail_mem=54.12 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.22it/s]Capturing num tokens (num_tokens=144 avail_mem=54.12 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.89it/s]Capturing num tokens (num_tokens=128 avail_mem=54.12 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.89it/s]

    Capturing num tokens (num_tokens=112 avail_mem=54.12 GB):  76%|███████▌  | 44/58 [00:04<00:01, 12.89it/s]Capturing num tokens (num_tokens=112 avail_mem=54.12 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.79it/s]Capturing num tokens (num_tokens=96 avail_mem=54.11 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.79it/s] Capturing num tokens (num_tokens=80 avail_mem=54.11 GB):  79%|███████▉  | 46/58 [00:04<00:00, 12.79it/s]

    Capturing num tokens (num_tokens=80 avail_mem=54.11 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.74it/s]Capturing num tokens (num_tokens=64 avail_mem=54.10 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.74it/s]Capturing num tokens (num_tokens=48 avail_mem=54.10 GB):  83%|████████▎ | 48/58 [00:04<00:00, 12.74it/s]Capturing num tokens (num_tokens=48 avail_mem=54.10 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.55it/s]Capturing num tokens (num_tokens=32 avail_mem=54.10 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.55it/s]

    Capturing num tokens (num_tokens=28 avail_mem=54.09 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.55it/s]Capturing num tokens (num_tokens=28 avail_mem=54.09 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.50it/s]Capturing num tokens (num_tokens=24 avail_mem=54.09 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.50it/s]Capturing num tokens (num_tokens=20 avail_mem=54.09 GB):  90%|████████▉ | 52/58 [00:04<00:00, 12.50it/s]

    Capturing num tokens (num_tokens=20 avail_mem=54.09 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.64it/s]Capturing num tokens (num_tokens=16 avail_mem=54.09 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.64it/s]Capturing num tokens (num_tokens=12 avail_mem=54.08 GB):  93%|█████████▎| 54/58 [00:05<00:00, 12.64it/s]Capturing num tokens (num_tokens=12 avail_mem=54.08 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.77it/s]Capturing num tokens (num_tokens=8 avail_mem=54.08 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.77it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=54.07 GB):  97%|█████████▋| 56/58 [00:05<00:00, 12.77it/s]Capturing num tokens (num_tokens=4 avail_mem=54.07 GB): 100%|██████████| 58/58 [00:05<00:00, 12.74it/s]Capturing num tokens (num_tokens=4 avail_mem=54.07 GB): 100%|██████████| 58/58 [00:05<00:00, 10.82it/s]


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
    Generated text:  Khiimani and I’m a PhD candidate in the CS Department at the University of Manitoba, Regina. My research focuses on the mathematics of computation and learning, with a particular interest in the problem of learning rules and a focus on learning rules in higher-order functions.\nSince I’m a PhD candidate, I’ve been working on various open problems that require substantial mathematical skill and have attracted a lot of attention from researchers in my area. I’m also interested in a broad range of applications of mathematics. I like to tackle open problems that are beyond the scope of my own expertise and I enjoy learning new ideas, problem solving, and collaborating
    ===============================
    Prompt: The president of the United States is
    Generated text:  expected to be a presidential candidate in an upcoming election. The probability of winning an election is 50%. What is the probability that the president will not win the election? Express your answer as a decimal.
    To determine the probability that the president of the United States will not win the election, we start by understanding the problem and the given information.
    
    1. The probability of the president winning an election is 50%, which can be expressed as a decimal:
       \[
       P(\text{win}) = 0.50
       \]
    
    2. The probability of the president not winning the election is the complement of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. France
    
    Which of the following statements about the capital city of France is correct? A. The capital of France is Paris B. The capital of France is Nice C. The capital of France is London D. The capital of France is Berlin
    Answer:
    A
    
    The main cause of postpartum hemorrhage is ____.
    A. Puerperal infection
    B. Uterine atony
    C. Coagulation dysfunction
    D. Uterine scar
    E. Soft birth canal injury
    Answer:
    B
    
    When carrying out construction work in the area below a culvert, a ____ should be set up at
    ===============================
    Prompt: The future of AI is
    Generated text:  a wake-up call for the company
    
    AI is the next big thing. It's going to change the world - or at least make it harder for it to do so. The potential consequences of the next generation of technology could seriously change the way we all interact with our computers.
    
    And one of the early beneficiaries of this new technology is DeepMind, the artificial intelligence company behind the current generation of supercomputers. The company's team of researchers have spent the past several years working on the use of artificial intelligence in the workplace. To try to get a better sense of how AI might change the workplace, they surveyed 1,00


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


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major hub for business, finance, and tourism in France. Paris is a city that is constantly evolving and changing, with new developments and cultural events taking place all the time. It is a city that is a true reflection of France's rich history and culture. Paris is a city that is a must-visit for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI technology continues to advance, we can expect to see even more integration into our daily lives.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, from personalized treatment plans to disease diagnosis and prevention. As AI technology continues to improve, we can expect to see even more use in healthcare.
    
    
    


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
    Generated text:  [Name], I'm a [Brief Introduction to the character] who has always been [What is your unique trait or ability]. I'm [How many years old are you]? And what's your favorite [one thing about yourself]? I've always been [What is your secret?]. Hi there, I'm [Name] and I'm an AI language model that's been in development for over 10 years. My unique trait is that I'm always learning and improving, which is why I'm here to assist you with your language needs. My favorite thing about myself is that I can understand and respond to multiple languages, making
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Here's a concise factual statement about Paris, France:
    
    Paris, the capital of France, is known for its iconic Eiffel Tower, world-famous landmarks such as the Louvre and Notre-Dame Cathedral, and vibrant cultural scene. It also hosts major events like the Eiffel Tower Festival and the annual Spring Festival. Paris is a city of rich history, art, and cuisine. Additionally, it has a bustling street life and is known for its festivals, fashion industry, and gastronomy. Known for its beautiful architecture and charming streets, Paris is a beloved city in France and beyond. It is often referred to as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid development and adoption of new technologies and applications. Here are some potential trends that could shape AI in the coming years:
    
    1. Increased Transparency and Explainability: As AI systems become more complex and interconnected, there will be a need for greater transparency and explainability. This will require more detailed descriptions of how AI systems work, how they make decisions, and why they make certain choices. This will lead to more robust, reliable, and trustworthy AI systems that are easier to understand and control.
    
    2. Enhanced Collaboration and Communication: As AI systems become more integrated with other technologies, there will be a need for greater collaboration


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

     am

     a

     [

    insert

     character

    's

     profession

    ]

     in

     the

     [

    insert

     location

    ]

     who

     has

     been

     around

     for

     [

    insert

     character

    's

     lifespan

    ]

     years

    .

     I

     started

     out

     as

     a

     simple

     [

    insert

     a

     profession

    ],

     but

     over

     the

     years

     I

     have

     grown

     into

     a

     [

    insert

     character

    's

     character

     trait

     or

     characteristic

    ],

     and

     I

     am

     truly

     dedicated

     to

     [

    insert

     one

     of

     the

     following

    :

     helping

     others

    ,

     being

     a

     good

     listener

    ,

     being

     a

     helpful

     assistant

    ,

     or

     being

     a

     mentor

    ].

     I

     am

     always

     willing

     to

     learn

     and

     improve

     my

     skills

     and

     knowledge

    ,

     and

     I

     am

     always

     striving

     to

     do

     my

     best

     to

     help

     those

     around

     me

    .

     Whether

     it

    's

     through

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     most

     populous

     city

     in

     the

     country

     and

     home

     to

     the

     French

     government

     and

     many

     of

     its

     institutions

    .

     Paris

     is

     known

     for

     its

     rich

     culture

    ,

     iconic

     landmarks

    ,

     and

     diverse

     cuisine

    .

     It

     is

     home

     to

     some

     of

     the

     world

    's

     most

     prestigious

     universities

    ,

     and

     the

     French

     language

     is

     widely

     spoken

    .

     France

    ’s

     capital

     is

     situated

     in

     the

     heart

     of

     the

     Lo

    ire

     Valley

    ,

     near

     the

     river

     Se

    ine

    ,

     and

     is

     the

     world

    's

     fourth

    -largest

     city

     by

     population

    .

     Paris

     is

     also

     one

     of

     the

     world

    's

     most

     popular

     tourist

     destinations

    ,

     with

     millions

     of

     visitors

     each

     year

    .

     The

     city

     has

     a

     long

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     is

     home

     to

     many

     ancient

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     four

     main

     trends

    :
    


    1

    .

     Increased

     development

     of

     AI

     technologies

     and

     applications

    :

     AI

     is

     rapidly

     evolving

    ,

     and

     companies

     are

     investing

     more

     in

     developing

     AI

     technologies

     and

     applications

    .

     This

     will

     likely

     lead

     to

     the

     creation

     of

     more

     advanced

     AI

     systems

     and

     applications

    ,

     as

     well

     as

     greater

     integration

     of

     AI

     into

     various

     industries

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     increasingly

     being

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     drones

    ,

     and

     autonomous

     vehicles

    ,

     to

     improve

     their

     efficiency

     and

     safety

    .

     This

     integration

     will

     likely

     continue

    ,

     with

     more

     companies

     developing

     AI

    -powered

     applications

     that

     combine

     AI

     with

     other

     technologies

    .
    


    3

    .

     Rise

     of

     AI

    -based

     decision

    -making

     systems

    :

     AI

    



```python
llm.shutdown()
```
