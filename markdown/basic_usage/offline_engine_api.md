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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.33it/s]


    2026-05-09 10:55:12,278 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 10:55:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]

    Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  7.60it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  7.60it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  7.60it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  7.60it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  7.60it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.38it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.29it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.24it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.24it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.24it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.24it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.24it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 23.82it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 23.82it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 23.82it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 23.82it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 23.82it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 23.82it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 28.12it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 28.12it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 28.12it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 28.12it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 28.12it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 28.12it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 31.82it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 31.82it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 31.82it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 31.82it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 31.82it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 31.82it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 34.76it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 34.76it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 34.76it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 34.76it/s]

    Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 34.76it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 34.76it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.79it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.79it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.92 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.92 GB):   2%|▏         | 1/58 [00:00<00:08,  6.91it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.88 GB):   2%|▏         | 1/58 [00:00<00:08,  6.91it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.88 GB):   3%|▎         | 2/58 [00:00<00:08,  6.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.88 GB):   3%|▎         | 2/58 [00:00<00:08,  6.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.88 GB):   5%|▌         | 3/58 [00:00<00:07,  7.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.87 GB):   5%|▌         | 3/58 [00:00<00:07,  7.12it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.87 GB):   7%|▋         | 4/58 [00:00<00:07,  7.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.87 GB):   7%|▋         | 4/58 [00:00<00:07,  7.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.87 GB):   9%|▊         | 5/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.86 GB):   9%|▊         | 5/58 [00:00<00:06,  7.76it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.86 GB):  10%|█         | 6/58 [00:00<00:06,  8.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.85 GB):  10%|█         | 6/58 [00:00<00:06,  8.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.85 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.85 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.85 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.85 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.84 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.84 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.84 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.84 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.84 GB):  21%|██        | 12/58 [00:01<00:04, 10.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.83 GB):  21%|██        | 12/58 [00:01<00:04, 10.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.83 GB):  21%|██        | 12/58 [00:01<00:04, 10.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.83 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.83 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.82 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.82 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.82 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.82 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.83it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=55.80 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.79 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.79 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.77 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=960 avail_mem=55.79 GB):  34%|███▍      | 20/58 [00:02<00:02, 14.14it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=55.79 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.36it/s]Capturing num tokens (num_tokens=896 avail_mem=55.79 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.36it/s]Capturing num tokens (num_tokens=832 avail_mem=55.78 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.36it/s]Capturing num tokens (num_tokens=832 avail_mem=55.78 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.87it/s]Capturing num tokens (num_tokens=768 avail_mem=55.78 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.87it/s]Capturing num tokens (num_tokens=704 avail_mem=55.78 GB):  41%|████▏     | 24/58 [00:02<00:02, 14.87it/s]

    Capturing num tokens (num_tokens=704 avail_mem=55.78 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.60it/s]Capturing num tokens (num_tokens=640 avail_mem=55.77 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.60it/s]Capturing num tokens (num_tokens=576 avail_mem=55.77 GB):  45%|████▍     | 26/58 [00:02<00:02, 15.60it/s]Capturing num tokens (num_tokens=576 avail_mem=55.77 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=512 avail_mem=55.76 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.00it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.77 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.00it/s]Capturing num tokens (num_tokens=480 avail_mem=55.77 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.13it/s]Capturing num tokens (num_tokens=448 avail_mem=55.77 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.13it/s]Capturing num tokens (num_tokens=416 avail_mem=55.77 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.13it/s]Capturing num tokens (num_tokens=416 avail_mem=55.77 GB):  55%|█████▌    | 32/58 [00:02<00:01, 15.44it/s]Capturing num tokens (num_tokens=384 avail_mem=55.77 GB):  55%|█████▌    | 32/58 [00:02<00:01, 15.44it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.76 GB):  55%|█████▌    | 32/58 [00:02<00:01, 15.44it/s]Capturing num tokens (num_tokens=352 avail_mem=55.76 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.56it/s]Capturing num tokens (num_tokens=320 avail_mem=55.75 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.56it/s]Capturing num tokens (num_tokens=288 avail_mem=55.75 GB):  59%|█████▊    | 34/58 [00:02<00:01, 15.56it/s]Capturing num tokens (num_tokens=288 avail_mem=55.75 GB):  62%|██████▏   | 36/58 [00:02<00:01, 15.68it/s]Capturing num tokens (num_tokens=256 avail_mem=55.75 GB):  62%|██████▏   | 36/58 [00:02<00:01, 15.68it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.74 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.68it/s]Capturing num tokens (num_tokens=240 avail_mem=55.74 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.65it/s]Capturing num tokens (num_tokens=224 avail_mem=55.74 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.65it/s]Capturing num tokens (num_tokens=208 avail_mem=55.70 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.65it/s]

    Capturing num tokens (num_tokens=208 avail_mem=55.70 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.22it/s]Capturing num tokens (num_tokens=192 avail_mem=55.70 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.22it/s]Capturing num tokens (num_tokens=176 avail_mem=55.70 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.22it/s]Capturing num tokens (num_tokens=176 avail_mem=55.70 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.49it/s]Capturing num tokens (num_tokens=160 avail_mem=55.70 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.49it/s]

    Capturing num tokens (num_tokens=144 avail_mem=55.69 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.49it/s]Capturing num tokens (num_tokens=144 avail_mem=55.69 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.54it/s]Capturing num tokens (num_tokens=128 avail_mem=55.69 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.54it/s]Capturing num tokens (num_tokens=112 avail_mem=55.69 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.54it/s]Capturing num tokens (num_tokens=112 avail_mem=55.69 GB):  79%|███████▉  | 46/58 [00:03<00:00, 15.05it/s]Capturing num tokens (num_tokens=96 avail_mem=55.68 GB):  79%|███████▉  | 46/58 [00:03<00:00, 15.05it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=55.68 GB):  79%|███████▉  | 46/58 [00:03<00:00, 15.05it/s]Capturing num tokens (num_tokens=80 avail_mem=55.68 GB):  83%|████████▎ | 48/58 [00:03<00:00, 15.64it/s]Capturing num tokens (num_tokens=64 avail_mem=55.68 GB):  83%|████████▎ | 48/58 [00:03<00:00, 15.64it/s]Capturing num tokens (num_tokens=48 avail_mem=55.67 GB):  83%|████████▎ | 48/58 [00:03<00:00, 15.64it/s]

    Capturing num tokens (num_tokens=48 avail_mem=55.67 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.77it/s]Capturing num tokens (num_tokens=32 avail_mem=55.67 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.77it/s]Capturing num tokens (num_tokens=28 avail_mem=55.66 GB):  86%|████████▌ | 50/58 [00:03<00:00, 14.77it/s]Capturing num tokens (num_tokens=28 avail_mem=55.66 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.43it/s]Capturing num tokens (num_tokens=24 avail_mem=55.66 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.43it/s]Capturing num tokens (num_tokens=20 avail_mem=55.66 GB):  90%|████████▉ | 52/58 [00:04<00:00, 15.43it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:04<00:00, 16.19it/s]Capturing num tokens (num_tokens=16 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:04<00:00, 16.19it/s]Capturing num tokens (num_tokens=12 avail_mem=55.65 GB):  93%|█████████▎| 54/58 [00:04<00:00, 16.19it/s]Capturing num tokens (num_tokens=12 avail_mem=55.65 GB):  97%|█████████▋| 56/58 [00:04<00:00, 16.56it/s]Capturing num tokens (num_tokens=8 avail_mem=55.65 GB):  97%|█████████▋| 56/58 [00:04<00:00, 16.56it/s] Capturing num tokens (num_tokens=4 avail_mem=55.65 GB):  97%|█████████▋| 56/58 [00:04<00:00, 16.56it/s]

    Capturing num tokens (num_tokens=4 avail_mem=55.65 GB): 100%|██████████| 58/58 [00:04<00:00, 16.85it/s]Capturing num tokens (num_tokens=4 avail_mem=55.65 GB): 100%|██████████| 58/58 [00:04<00:00, 13.22it/s]


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
    Generated text:  Samantha and I am from the US. I’m trying to get my 17 year old daughter to say ‘no’ to the idea of getting a cat. I’ve been trying to persuade her that cats are very important for our family. I’ve also been trying to get her to understand the importance of cleanliness and to say no to things like having a cat.
    It’s hard for me. I just don’t know what to say. I’ve tried to be very firm and authoritative with her, but it doesn’t seem to be working. I’ve tried explaining that it’s important to our family for you to be responsible for it
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. In the United States, the president is popularly elected by the people and serves a term of four years. Since 1865, the term of the president has been extended by one year for each President of the Senate.
    When the term of the president of the United States expires, the next person in line to assume the presidency is the Vice President of the United States. The Vice President is the political figure who assumes office when the President is not available to serve.
    If the Vice President is unable to serve due to an illness, resignation, or other reason, the Vice President will have to wait until the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The coordinates of the capital of China are 30°N, 120°E. What is the approximate distance between the capital of France and China?
    
    A: 600 kilometers  
    B: 900 kilometers  
    C: 1200 kilometers  
    D: 1800 kilometers
    To determine the approximate distance between the capital of France, Paris, and the capital of China, Beijing, we need to calculate the great-circle distance between these two points. The coordinates of Paris are given as (30°N, 120°E), and the coordinates of
    ===============================
    Prompt: The future of AI is
    Generated text:  very much dependent on the availability of sufficient computing power. In the early days of the Internet, the landscape of the technology was largely dominated by commodity processors. Those were found in a variety of different hardware platforms, ranging from CPUs (central processing units) to GPUs (graphics processing units). Over the last decade, we have seen the transition from commodity processors to custom chips that have multiple cores and specialized processors such as FPGAs (field-programmable gate arrays) and ASICs (application-specific integrated circuits). In this article, we examine the evolution of computer architecture and architecture for AI in detail. We will also discuss the trends and potential


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known fact that Paris is the capital of France, and this statement accurately reflects that fact. The statement is also grammatically correct and follows standard English syntax. Overall, it is a well-written and informative statement that provides a basic understanding of the capital city's location and importance in French society. 
    
    However, if you would like to provide a more detailed or specific statement, please let me know and I will do my best to assist you further. 
    
    In
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and personalized medicine to virtual assistants and chatbots. Additionally, AI is likely to continue to be used for tasks such as fraud detection, cybersecurity, and environmental monitoring, as well as for tasks such as language translation and image recognition. However, there are also potential risks and challenges associated with AI, including issues such as bias, privacy, and security. As AI continues to evolve, it is likely to play an increasingly important role
    


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
    Generated text:  [Name] and I am a [age] year old [occupation]. I have a passion for [interest/interest], and I'm always looking for ways to [describe an activity or hobby]. I'm confident and outgoing, and I love spending time with friends and family. I have a keen sense of humor, and I enjoy trying new things and experimenting with different ideas. I have always been interested in learning new things, and I'm always eager to expand my horizons. I hope to always find joy in helping others and make a positive impact in my community. I'm always eager to learn and grow, and I'm excited
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is a historic city known for its iconic Notre-Dame Cathedral and various museums and cultural landmarks.
    
    That's correct! Paris is known as the "City of Light" and is the capital city of France, located on the bank of the Seine River in the north-central region of the country. The city is home to some of the world's most famous landmarks, including the iconic Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also a major center of French culture, known for its rich history, art, and food culture. In addition to its historical significance, Paris is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable. However, there are several possible trends that could shape the direction of AI development in the coming years:
    
    1. Advancements in machine learning: With the advancement of machine learning, AI systems will become increasingly capable of learning from data and making predictions on new scenarios. This could lead to the development of more advanced algorithms and techniques for tasks such as image and speech recognition, natural language processing, and automated decision-making.
    
    2. Emergence of biotechnology: AI could play a critical role in the development of biotechnology. For example, AI could be used to develop new therapies for diseases such as cancer, HIV, and


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

    ]

     and

     I

    'm

     a

     [

    age

    ]

     year

     old

     girl

    .

     I

    'm

     a

     [

    occupation

    ],

     and

     I

     enjoy

     [

    what

     I

     do

     best

    ].

     I

    'm

     a

     [

    career

     goal

    ],

     and

     I

     believe

     [

    why

     I

     want

     to

     be

     a

     [

    career

     goal

    )].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .
    


    Thank

     you

     for

     asking

    .

     What

     about

     you

    ,

     [

    Name

    ]?

     What

    's

     your

     name

    ?

     How

     old

     are

     you

    ?

     What

     do

     you

     do

     best

    ?

     What

    's

     your

     career

     goal

    ?

     I

    'd

     love

     to

     meet

     you

    .

     What

     are

     you

     looking

     for

     in

     a

     person

    ?

     Let

    's

     see

    ...

     [

    insert

     what

     you

     want

     me

     to

     say

     next

    ].

     Alright

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     by

     population

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

     historical

     association

     with

     Paris

    ian

     love

     stories

     and

     romantic

     poems

    .

     The

     city

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     many

     famous

     landmarks

     such

     as

     the

     Notre

    -D

    ame

     Cathedral

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     also

     renowned

     for

     its

     rich

     cultural

     and

     artistic

     heritage

    ,

     including

     the

     Op

    éra

     Garn

    ier

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Museum

     of

     Modern

     Art

     (

    Mo

    MA

    ).

     It

     is

     a

     major

     international

     center

     for

     business

     and

     finance

    ,

     hosting

     the

     World

     Economic

     Forum

     and

     the

     G

    2

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     an

     explosion

     of

     new

     technology

    ,

     a

     massive

     increase

     in

     data

    ,

     and

     more

     widespread

     use

     of

     AI

     in

     all

     sectors

     of

     society

    .

     In

     

    2

    0

    2

    1

    ,

     

    2

    0

    2

    2

    ,

     and

     

    2

    0

    2

    3

    ,

     we

     can

     expect

     to

     see

     the

     following

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

     and

     robotics

    :

     With

     the

     rise

     of

     the

     Internet

     of

     Things

     and

     machine

     learning

    ,

     robots

     are

     becoming

     more

     sophisticated

     and

     able

     to

     perform

     tasks

     more

     efficiently

     than

     humans

    .

     AI

     systems

     are

     expected

     to

     become

     more

     capable

     of

     replic

    ating

     human

     behavior

     and

     decision

    -making

    ,

     leading

     to

     the

     development

     of

     autonomous

     vehicles

    ,

     manufacturing

     automation

    ,

     and

     more

    .
    


    2

    



```python
llm.shutdown()
```
