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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]


    2026-05-20 04:24:45,886 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 04:24:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.36it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.66it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 13.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 20.20it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 20.20it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 20.20it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 20.20it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 20.20it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 20.20it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 20.20it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 20.20it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 20.20it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 20.20it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.18it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 35.66it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 35.66it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 35.66it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 35.66it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 35.66it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 35.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.21 GB):   3%|▎         | 2/58 [00:00<00:03, 15.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.20 GB):   3%|▎         | 2/58 [00:00<00:03, 15.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.20 GB):   3%|▎         | 2/58 [00:00<00:03, 15.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.20 GB):   3%|▎         | 2/58 [00:00<00:03, 15.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.20 GB):   9%|▊         | 5/58 [00:00<00:04, 12.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.20 GB):   9%|▊         | 5/58 [00:00<00:04, 12.41it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=55.19 GB):   9%|▊         | 5/58 [00:00<00:04, 12.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.18 GB):   9%|▊         | 5/58 [00:00<00:04, 12.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.18 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.18 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.70it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.18 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.18 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.17 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.17 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.60it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.17 GB):  21%|██        | 12/58 [00:00<00:03, 12.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.17 GB):  21%|██        | 12/58 [00:00<00:03, 12.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.17 GB):  21%|██        | 12/58 [00:01<00:03, 12.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.17 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.16 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.84it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.16 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.16 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.15 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.15 GB):  28%|██▊       | 16/58 [00:01<00:03, 11.69it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=55.15 GB):  31%|███       | 18/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.15 GB):  31%|███       | 18/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.15 GB):  31%|███       | 18/58 [00:01<00:03, 11.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.15 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.13 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.62it/s]

    Capturing num tokens (num_tokens=960 avail_mem=55.14 GB):  34%|███▍      | 20/58 [00:01<00:03, 11.62it/s] Capturing num tokens (num_tokens=960 avail_mem=55.14 GB):  38%|███▊      | 22/58 [00:01<00:03, 11.96it/s]Capturing num tokens (num_tokens=896 avail_mem=55.14 GB):  38%|███▊      | 22/58 [00:01<00:03, 11.96it/s]Capturing num tokens (num_tokens=832 avail_mem=55.14 GB):  38%|███▊      | 22/58 [00:01<00:03, 11.96it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.14 GB):  41%|████▏     | 24/58 [00:01<00:02, 12.10it/s]Capturing num tokens (num_tokens=768 avail_mem=55.13 GB):  41%|████▏     | 24/58 [00:01<00:02, 12.10it/s]Capturing num tokens (num_tokens=704 avail_mem=55.13 GB):  41%|████▏     | 24/58 [00:02<00:02, 12.10it/s]Capturing num tokens (num_tokens=704 avail_mem=55.13 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.13it/s]Capturing num tokens (num_tokens=640 avail_mem=55.13 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.13it/s]

    Capturing num tokens (num_tokens=576 avail_mem=55.13 GB):  45%|████▍     | 26/58 [00:02<00:02, 12.13it/s]Capturing num tokens (num_tokens=576 avail_mem=55.13 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.22it/s]Capturing num tokens (num_tokens=512 avail_mem=55.11 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.22it/s]Capturing num tokens (num_tokens=480 avail_mem=55.13 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.22it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.13 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=448 avail_mem=55.12 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=416 avail_mem=55.12 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.01it/s]Capturing num tokens (num_tokens=416 avail_mem=55.12 GB):  55%|█████▌    | 32/58 [00:02<00:02, 11.99it/s]Capturing num tokens (num_tokens=384 avail_mem=54.30 GB):  55%|█████▌    | 32/58 [00:02<00:02, 11.99it/s]

    Capturing num tokens (num_tokens=352 avail_mem=54.29 GB):  55%|█████▌    | 32/58 [00:02<00:02, 11.99it/s]Capturing num tokens (num_tokens=352 avail_mem=54.29 GB):  59%|█████▊    | 34/58 [00:02<00:02, 11.94it/s]Capturing num tokens (num_tokens=320 avail_mem=54.29 GB):  59%|█████▊    | 34/58 [00:02<00:02, 11.94it/s]Capturing num tokens (num_tokens=288 avail_mem=54.29 GB):  59%|█████▊    | 34/58 [00:02<00:02, 11.94it/s]

    Capturing num tokens (num_tokens=288 avail_mem=54.29 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.74it/s]Capturing num tokens (num_tokens=256 avail_mem=54.28 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.74it/s]Capturing num tokens (num_tokens=240 avail_mem=54.28 GB):  62%|██████▏   | 36/58 [00:02<00:01, 12.74it/s]Capturing num tokens (num_tokens=240 avail_mem=54.28 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.51it/s]Capturing num tokens (num_tokens=224 avail_mem=54.28 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.51it/s]Capturing num tokens (num_tokens=208 avail_mem=54.27 GB):  66%|██████▌   | 38/58 [00:03<00:01, 13.51it/s]

    Capturing num tokens (num_tokens=208 avail_mem=54.27 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.55it/s]Capturing num tokens (num_tokens=192 avail_mem=54.27 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.55it/s]Capturing num tokens (num_tokens=176 avail_mem=54.27 GB):  69%|██████▉   | 40/58 [00:03<00:01, 14.55it/s]Capturing num tokens (num_tokens=176 avail_mem=54.27 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.58it/s]Capturing num tokens (num_tokens=160 avail_mem=54.27 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.58it/s]Capturing num tokens (num_tokens=144 avail_mem=54.26 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.58it/s]Capturing num tokens (num_tokens=128 avail_mem=54.26 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.58it/s]

    Capturing num tokens (num_tokens=128 avail_mem=54.26 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.77it/s]Capturing num tokens (num_tokens=112 avail_mem=54.26 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.77it/s]Capturing num tokens (num_tokens=96 avail_mem=54.25 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.77it/s] Capturing num tokens (num_tokens=80 avail_mem=54.25 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.77it/s]Capturing num tokens (num_tokens=80 avail_mem=54.25 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.76it/s]Capturing num tokens (num_tokens=64 avail_mem=54.25 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.76it/s]Capturing num tokens (num_tokens=48 avail_mem=54.24 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.76it/s]Capturing num tokens (num_tokens=32 avail_mem=54.24 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.76it/s]

    Capturing num tokens (num_tokens=32 avail_mem=54.24 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=28 avail_mem=54.23 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=24 avail_mem=54.23 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=20 avail_mem=54.23 GB):  88%|████████▊ | 51/58 [00:03<00:00, 22.01it/s]Capturing num tokens (num_tokens=20 avail_mem=54.23 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.53it/s]Capturing num tokens (num_tokens=16 avail_mem=54.23 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.53it/s]Capturing num tokens (num_tokens=12 avail_mem=54.22 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.53it/s]Capturing num tokens (num_tokens=8 avail_mem=54.22 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.53it/s] Capturing num tokens (num_tokens=4 avail_mem=54.21 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.53it/s]

    Capturing num tokens (num_tokens=4 avail_mem=54.21 GB): 100%|██████████| 58/58 [00:03<00:00, 26.97it/s]Capturing num tokens (num_tokens=4 avail_mem=54.21 GB): 100%|██████████| 58/58 [00:03<00:00, 15.02it/s]


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
    Generated text:  Tom.
    I am a doctor who specializes in mental health. I want to talk to you about anxiety.
    Your first question is: "What is anxiety? "
    Anxiety is a state of worry, fear, or nervousness that can make you feel nervous, anxious, or uneasy. It can be caused by a variety of factors, including stress, fear of failure, or concerns about the future. Anxiety can affect anyone, but it's particularly common in people who have a family history of mental health conditions or who live in stressful environments. Anxiety can also be triggered by certain situations, such as being on a bus or in a car.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. ( )
    A. True
    B. False
    C. Cannot be determined
    To determine whether the statement "The president of the United States is a person" is true, false, or cannot be determined, let's analyze the information step by step.
    
    1. **Understanding the President of the United States**:
       - The president of the United States is the head of state and the head of government of the country.
       - The president is also the commander-in-chief of the military and the commander-in-chief of the armed forces in the United States.
    
    2. **Classification of the President**:
       - The president
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. ____ A. Paris B. Lyon C. Paris
    Answer: A
    
    When 2 liters of water is mixed with 1 liter of alcohol, the mixture has a density of 0.787 g/mL. What is the density of the mixture?
    A. 0.945 g/mL
    B. 0.890 g/mL
    C. 0.787 g/mL
    D. 0.680 g/mL
    Answer: A
    
    Which of the following statements about the relationship between geometric elements and their geometric coordinates in a spatial rectangular coordinate system is incorrect?
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  evolving rapidly, and while researchers have been pushing the boundaries of what is possible, the question of ethics and the responsibility of developers is increasingly pressing.
    From a regulatory perspective, governments have been investigating how to respond to the ethical concerns surrounding AI, with a focus on the impact on privacy and data protection. The European Union (EU) has implemented the General Data Protection Regulation (GDPR), which introduces new rules for the protection of personal data in a digital environment. The European Commission has also launched the European Commission’s AI Ethics Unit to ensure that the development of AI will respect the principles of the GDPR.
    The European Commission has also introduced the European


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


    Generated text:  [Name] and I am a [occupation] with [number] years of experience in [industry]. I am a [type of person] who is always [positive trait]. I am [character trait] and I am always [positive trait]. I am [character trait] and I am always [positive trait]. I am [character trait] and I am always [positive trait]. I am [character trait] and I am always [positive trait]. I am [character trait] and I am always [positive trait]. I am [character trait] and I am always [positive trait]. I am [character trait] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city of people, with a diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to the needs of humans.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even greater use of AI in healthcare, with more sophisticated
    


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
    Generated text:  __________. I am a ___________. I have been with the ____________ for ___________. I enjoy ___________. I am ___________. I love to ___________. I have a lot of ___________ in my life. I am ___________. I am a ___________ and I enjoy ___________. I will do my best to ___________. If you have any questions or need any information, please feel free to ask me. Thank you for having me! (Feel free to add any additional information about the character's personality or any other relevant details.) Oh, I see. Is there anything I can do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Therefore, the answer is Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  quite exciting and unpredictable, with many possibilities and potential challenges that are yet to be fully realized. Here are some possible future trends in AI:
    
    1. Increased integration with other technologies: As AI continues to advance, we can expect to see a greater integration of AI with other technologies like sensors, cameras, and other machine learning algorithms. This will allow for even more advanced AI capabilities and a wider range of applications.
    
    2. Increased personalization: With the help of AI, we can create more personalized experiences for users. AI can analyze user data and identify patterns that can help tailor the user experience to their needs.
    
    3. Increased autonomous vehicles:


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

     fictional

     character

    's

     name

    ],

     and

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     profession

     or

     age

    ].

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

     love

     to

     read

     and

     listen

     to

     music

    .

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     hobby

     or

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

     try

     new

     things

    .

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     open

     to

     new

     challenges

     and

     adventures

    .


    Any

     other

     information

     you

    'd

     like

     me

     to

     include

     in

     my

     self

    -int

    roduction

    ?

     


    What

    's

     your

     favorite

     type

     of

     music

     or

     book

     to

     read

    ?

     Is

     there

     any

     book

     you

     particularly

     enjoy

     reading

     or

     listening

     to

     music

    ?

     


    What

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     a

     bustling

     city

     with

     a

     rich

     history

    ,

     architecture

    ,

     and

     vibrant

     culture

    .

     It

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Lou

    vre

     Museum

    .

     The

     city

     also

     has

     a

     diverse

     population

     of

     over

     

    2

     million

     people

    ,

     making

     it

     the

     largest

     city

     in

     France

    .

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     visitors

     from

     all

     over

     the

     world

    ,

     offering

     a

     mix

     of

     historical

     attractions

    ,

     art

    ,

     and

     food

    .

     Its

     famous

     fashion

     scene

    ,

     including

     cout

    ure

     stores

     like

     Chanel

    ,

     cout

    ure

     designer

     L

    VM

    H

    ,

     and

     luxury

     tail

    oring

     houses

    ,

     is

     also

     a

     popular

     highlight

    .

     Paris

     is

     known

     for

     its

     cultural

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     and

     it

     is

     difficult

     to

     predict

     exactly

     what

     trends

     will

     emerge

    .

     However

    ,

     there

     are

     some

     general

     trends

     that

     seem

     likely

     to

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

    .

     Personal

    ized

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     become

     more

     personalized

    .

     This

     means

     that

     machines

     will

     be

     able

     to

     learn

     from

     data

     and

     provide

     personalized

     recommendations

    ,

     such

     as

     fashion

     advice

    ,

     book

     recommendations

    ,

     or

     even

     restaurant

     suggestions

    .

     This

     can

     lead

     to

     a

     more

     efficient

     and

     effective

     use

     of

     resources

    ,

     as

     machines

     can

     be

     trained

     to

     provide

     tailored

     recommendations

     for

     each

     user

    .
    


    2

    .

     Autonomous

     AI

    :

     Autonomous

     AI

     refers

     to

     machines

     that

     can

     operate

     without

     human

     intervention

    ,

     such

     as

    



```python
llm.shutdown()
```
