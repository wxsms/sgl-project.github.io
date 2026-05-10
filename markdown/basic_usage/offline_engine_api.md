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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]


    2026-05-10 09:45:56,672 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 09:45:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.34it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.34it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.34it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.34it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.34it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 13.76it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 13.76it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.76it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.76it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 18.00it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 18.00it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 18.00it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 18.00it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 18.00it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 18.00it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 23.53it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 27.78it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 27.78it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 27.78it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 27.78it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 27.78it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 27.78it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 32.55it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 36.37it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 38.53it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 42.52it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 42.52it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 42.52it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 42.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.50 GB):   2%|▏         | 1/58 [00:00<00:06,  8.69it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.47 GB):   2%|▏         | 1/58 [00:00<00:06,  8.69it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.47 GB):   3%|▎         | 2/58 [00:00<00:06,  8.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.46 GB):   3%|▎         | 2/58 [00:00<00:06,  8.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.46 GB):   5%|▌         | 3/58 [00:00<00:06,  7.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.46 GB):   5%|▌         | 3/58 [00:00<00:06,  7.99it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.46 GB):   7%|▋         | 4/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.46 GB):   7%|▋         | 4/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.46 GB):   7%|▋         | 4/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.46 GB):  10%|█         | 6/58 [00:00<00:04, 11.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.45 GB):  10%|█         | 6/58 [00:00<00:04, 11.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.44 GB):  10%|█         | 6/58 [00:00<00:04, 11.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.44 GB):  10%|█         | 6/58 [00:00<00:04, 11.74it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=56.44 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.44 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.43 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.43 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.43 GB):  21%|██        | 12/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.43 GB):  21%|██        | 12/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.43 GB):  21%|██        | 12/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.42 GB):  21%|██        | 12/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.42 GB):  21%|██        | 12/58 [00:00<00:02, 19.24it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=56.42 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.42 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.41 GB):  28%|██▊       | 16/58 [00:01<00:01, 23.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.41 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.41 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.46it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=56.39 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.46it/s]Capturing num tokens (num_tokens=960 avail_mem=56.40 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.46it/s] Capturing num tokens (num_tokens=960 avail_mem=56.40 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.29it/s]Capturing num tokens (num_tokens=896 avail_mem=56.40 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.29it/s]

    Capturing num tokens (num_tokens=832 avail_mem=56.40 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.29it/s]Capturing num tokens (num_tokens=832 avail_mem=56.40 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.72it/s]Capturing num tokens (num_tokens=768 avail_mem=56.39 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.72it/s]Capturing num tokens (num_tokens=704 avail_mem=56.39 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.72it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.39 GB):  45%|████▍     | 26/58 [00:01<00:02, 14.17it/s]Capturing num tokens (num_tokens=640 avail_mem=58.99 GB):  45%|████▍     | 26/58 [00:01<00:02, 14.17it/s]Capturing num tokens (num_tokens=576 avail_mem=58.99 GB):  45%|████▍     | 26/58 [00:01<00:02, 14.17it/s]Capturing num tokens (num_tokens=576 avail_mem=58.99 GB):  48%|████▊     | 28/58 [00:01<00:02, 14.69it/s]Capturing num tokens (num_tokens=512 avail_mem=58.98 GB):  48%|████▊     | 28/58 [00:01<00:02, 14.69it/s]Capturing num tokens (num_tokens=480 avail_mem=58.99 GB):  48%|████▊     | 28/58 [00:01<00:02, 14.69it/s]

    Capturing num tokens (num_tokens=480 avail_mem=58.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 14.44it/s]Capturing num tokens (num_tokens=448 avail_mem=58.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 14.44it/s]Capturing num tokens (num_tokens=416 avail_mem=58.99 GB):  52%|█████▏    | 30/58 [00:02<00:01, 14.44it/s]Capturing num tokens (num_tokens=416 avail_mem=58.99 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.43it/s]Capturing num tokens (num_tokens=384 avail_mem=58.99 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.43it/s]Capturing num tokens (num_tokens=352 avail_mem=58.98 GB):  55%|█████▌    | 32/58 [00:02<00:01, 14.43it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.98 GB):  59%|█████▊    | 34/58 [00:02<00:01, 14.77it/s]Capturing num tokens (num_tokens=320 avail_mem=58.97 GB):  59%|█████▊    | 34/58 [00:02<00:01, 14.77it/s]Capturing num tokens (num_tokens=288 avail_mem=58.97 GB):  59%|█████▊    | 34/58 [00:02<00:01, 14.77it/s]Capturing num tokens (num_tokens=288 avail_mem=58.97 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.99it/s]Capturing num tokens (num_tokens=256 avail_mem=58.97 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.99it/s]Capturing num tokens (num_tokens=240 avail_mem=58.97 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.99it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.96 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.99it/s]Capturing num tokens (num_tokens=224 avail_mem=58.96 GB):  67%|██████▋   | 39/58 [00:02<00:01, 17.40it/s]Capturing num tokens (num_tokens=208 avail_mem=58.96 GB):  67%|██████▋   | 39/58 [00:02<00:01, 17.40it/s]Capturing num tokens (num_tokens=192 avail_mem=58.96 GB):  67%|██████▋   | 39/58 [00:02<00:01, 17.40it/s]Capturing num tokens (num_tokens=176 avail_mem=58.95 GB):  67%|██████▋   | 39/58 [00:02<00:01, 17.40it/s]Capturing num tokens (num_tokens=160 avail_mem=58.95 GB):  67%|██████▋   | 39/58 [00:02<00:01, 17.40it/s]Capturing num tokens (num_tokens=144 avail_mem=58.95 GB):  67%|██████▋   | 39/58 [00:02<00:01, 17.40it/s]Capturing num tokens (num_tokens=144 avail_mem=58.95 GB):  76%|███████▌  | 44/58 [00:02<00:00, 24.26it/s]Capturing num tokens (num_tokens=128 avail_mem=58.95 GB):  76%|███████▌  | 44/58 [00:02<00:00, 24.26it/s]Capturing num tokens (num_tokens=112 avail_mem=58.94 GB):  76%|███████▌  | 44/58 [00:02<00:00, 24.26it/s]Capturing num tokens (num_tokens=96 avail_mem=58.94 GB):  76%|███████▌  | 44/58 [00:02<00:00, 24.26it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=58.94 GB):  81%|████████  | 47/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:02<00:00, 27.99it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 33.00it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 33.00it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:03<00:00, 19.30it/s]


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
    Generated text:  Tom and I'm a teacher. I'm writing this post to ask for help with something I'm doing for my class. I'm doing a unit on the Great Depression, and I'm trying to create a story for the character's life. The character is 26 years old, and their age is on the scale of 1-40. 
    
    **I want the character to be an actor in the play, so I'm looking to come up with some details about the character. I'm thinking about the character being 26 years old in the story, and I want them to be the one who is able to
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He has two options, Base A and Base B. Base A has 200 military bases and Base B has 150 military bases. However, he wants to have at least 10 more military bases than Base B. How many more military bases should the president have than Base B?
    
    To determine how many more military bases the president should have than Base B, we start by identifying the number of military bases in each option. Base A has 200 military bases, and Base B has 150 military bases. According to the problem, the president
    ===============================
    Prompt: The capital of France is
    Generated text:  located at the center of which country?
    A. Belgium
    B. Switzerland
    C. Switzerland
    D. Luxembourg
    E. France
    Answer:
    E
    
    Which of the following statements about the structure and function of the respiratory system is correct?
    A. The respiratory system includes the pharynx, larynx, trachea, bronchi, and lungs.
    B. The lungs are the primary organs of the respiratory system.
    C. The pharynx is responsible for warming the air.
    D. The trachea is located at the top of the trachea.
    E. The bronchi are located in the center of
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people. But how can we make sure that the benefits of AI are distributed equitably? The last three decades of research and development in artificial intelligence have made significant progress, but the benefits are not evenly distributed across the world. Some people are benefiting more than others. AI and machine learning have the potential to transform many aspects of life, including healthcare, transportation, finance, and education. But how can we ensure that these benefits are shared fairly among all members of society?
    
    One approach is to address the root causes of AI inequality, such as job displacement and economic inequality. To address the root causes of AI inequality,


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am passionate about [job title] and have been working in the field for [number of years] years. I am always looking for new challenges and opportunities to grow and learn. I am a team player and enjoy collaborating with others to achieve our goals. I am always looking for ways to improve my skills and knowledge. I am excited to continue learning and growing in my career. Thank you. [Name] [Company Name] [Date] [Name] [Company Name] [Date] [Name] [Company Name] [Date] [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and is home to many cultural and artistic institutions. It is also known for its cuisine, with many famous French dishes being popular worldwide. Overall, Paris is a vibrant and exciting city with a rich history and a unique
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI that can better understand and respond to the needs of individuals.
    
    2. Greater reliance on data: AI will become more data-driven, with more data being used to train and improve AI systems. This could lead to more efficient and effective use of resources, as well as more accurate and reliable predictions and recommendations.
    
    3. Increased ethical considerations: As AI becomes more integrated with human intelligence
    


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
    Generated text:  [Name], and I'm a [job title or background] at [company name]. I'm excited to meet you and learn more about you. How are you, [friend, audience member, or anyone else]? I would love to have a chance to connect with you and learn more about you. What's the most interesting or surprising thing you've learned recently? Let's get to know you better! Let's chat, shall we? Alright, now what? Let's get started! What's your name? How do you like to get started? I'm looking forward to the conversation! When was the last time you took a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest city in France and serves as the seat of government, administration, and culture. It is known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Arc de Triomphe. Paris is also famous for its cuisine, art, and fashion. It is a significant cultural and economic center in Europe and plays a crucial role in French society and politics. The city is known for its warm climate, elegant atmosphere, and historical significance. The French capital is home to a large population of people and is a major tourist destination. Paris has a rich history dating back to the Roman era
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly one of rapid advancement and transformation, characterized by both opportunities and challenges. Here are some potential future trends in the field of artificial intelligence:
    
    1. Increased use of AI in healthcare: As AI technology advances, it will become more integrated into healthcare systems to improve patient outcomes and reduce costs. This will include the use of AI algorithms to detect diseases, predict patient outcomes, and optimize treatments.
    
    2. AI in agriculture: AI will play a key role in optimizing crop yields, improving crop quality, and reducing waste. This will include the use of AI-powered decision-making tools to optimize farming practices and reduce environmental impact.
    
    3. AI in finance


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

     name

    ]

     and

     I

    ’m

     a

     [

    insert

     profession

    ]

     who

     was

     born

     [

    insert

     birth

     year

     and

     month

    ]

     in

     [

    insert

     hometown

    ]

     and

     I

    ’ve

     been

     in

     this

     city

     since

     [

    insert

     current

     year

    ]

     at

     [

    insert

     job

     title

    ]

     in

     [

    insert

     city

    ,

     state

    ,

     zip

     code

    ].

     I

     love

     [

    insert

     hobbies

     or

     interests

    ]

     and

     I

    ’ve

     always

     been

     fascinated

     by

     the

     [

    insert

     industry

     or

     hobby

    ]

    .
    


    I

    ’m

     always

     up

     for

     a

     challenge

     and

     I

     love

     being

     someone

     that

     others

     can

     look

     up

     to

     and

     follow

    .

     I

    ’m

     always

     trying

     to

     learn

     more

     about

     this

     city

     and

     I

    ’m

     always

     eager

     to

     share

     my

     knowledge

     with

     others

    .

     I

    ’m

     always

     looking

     for

     new

    
    
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

     Love

    "

     due

     to

     its

     rich

     cultural

     heritage

     and

     romantic

     history

    .

     The

     city

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     a

     major

     cultural

    ,

     financial

    ,

     and

     political

     center

     in

     Europe

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

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     France

    's

     capital

     city

     is

     a

     bustling

     and

     modern

     met

    ropolis

     with

     a

     rich

     cultural

     and

     artistic

     heritage

    .

     The

     city

     is

     also

     home

     to

     many

     international

     institutions

    ,

     including

     the

     French

     Academy

     of

     Sciences

     and

     the

     European

     Organization

     for

     Nuclear

     Research

     (

    C

    ERN

    ).

     Paris

     is

     a

     city

     of

     contrasts

     and

     diversity

    ,

     with

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     potential

    .

     Here

     are

     some

     potential

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

     automate

     a

     wide

     range

     of

     tasks

    ,

     from

     manufacturing

     and

     transportation

     to

     customer

     service

     and

     healthcare

    .
    


    2

    .

     Improved

     accuracy

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     analyze

     and

     interpret

     data

    ,

     leading

     to

     greater

     accuracy

     and

     precision

     in

     decision

    -making

    .
    


    3

    .

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     personalize

     interactions

     with

     users

    ,

     based

     on

     their

     interests

    ,

     behavior

    ,

     and

     preferences

    .
    


    4

    .

     Autonomous

     agents

    :

     AI

     will

     continue

     to

     become

     more

     capable

     of

     performing

     tasks

     on

     their

     own

    ,

     without

     human

     intervention

    .
    


    5

    .

     Cyber

    security

    :

     AI

     will

     continue

    



```python
llm.shutdown()
```
