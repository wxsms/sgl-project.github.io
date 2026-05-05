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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.65it/s]


    2026-05-05 21:34:47,038 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 21:34:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:57,  4.16s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:57,  4.16s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:57,  4.16s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.72it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.30it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.30it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.30it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.30it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.30it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.10it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.10it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.10it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.10it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.10it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 14.93it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 14.93it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 19.17it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 19.17it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 19.17it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 19.17it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 19.17it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 22.85it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 22.85it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 22.85it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 22.85it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 22.85it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 22.85it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 27.56it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 27.56it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 27.56it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 27.56it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 27.56it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.16it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.16it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.16it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.16it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.16it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 30.16it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 34.28it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 35.61it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 35.61it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 35.61it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 35.61it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 35.61it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 35.61it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 37.73it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 37.73it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 37.73it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 40.71it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 40.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.73 GB):   2%|▏         | 1/58 [00:00<00:08,  6.75it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.70 GB):   2%|▏         | 1/58 [00:00<00:08,  6.75it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.70 GB):   3%|▎         | 2/58 [00:00<00:08,  6.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.69 GB):   3%|▎         | 2/58 [00:00<00:08,  6.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.69 GB):   5%|▌         | 3/58 [00:00<00:07,  7.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.68 GB):   5%|▌         | 3/58 [00:00<00:07,  7.04it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.68 GB):   7%|▋         | 4/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.68 GB):   7%|▋         | 4/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.68 GB):   9%|▊         | 5/58 [00:00<00:06,  7.58it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.67 GB):   9%|▊         | 5/58 [00:00<00:06,  7.58it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.67 GB):  10%|█         | 6/58 [00:00<00:06,  8.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.66 GB):  10%|█         | 6/58 [00:00<00:06,  8.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.66 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.66 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=52.66 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.66 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.65 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.65 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.65 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.34it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.65 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.65 GB):  21%|██        | 12/58 [00:01<00:04,  9.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.64 GB):  21%|██        | 12/58 [00:01<00:04,  9.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.64 GB):  21%|██        | 12/58 [00:01<00:04,  9.80it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.64 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.64 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.63 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.63 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.63 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.60it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=52.63 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.63 GB):  31%|███       | 18/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.62 GB):  31%|███       | 18/58 [00:01<00:03, 10.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.62 GB):  31%|███       | 18/58 [00:01<00:03, 10.88it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=52.62 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.60 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.66it/s]Capturing num tokens (num_tokens=960 avail_mem=52.62 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.66it/s] Capturing num tokens (num_tokens=960 avail_mem=52.62 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.78it/s]Capturing num tokens (num_tokens=896 avail_mem=52.61 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.78it/s]Capturing num tokens (num_tokens=832 avail_mem=52.61 GB):  38%|███▊      | 22/58 [00:02<00:02, 12.78it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.61 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.58it/s]Capturing num tokens (num_tokens=768 avail_mem=52.61 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.58it/s]Capturing num tokens (num_tokens=704 avail_mem=52.60 GB):  41%|████▏     | 24/58 [00:02<00:02, 13.58it/s]Capturing num tokens (num_tokens=704 avail_mem=52.60 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.42it/s]Capturing num tokens (num_tokens=640 avail_mem=52.60 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.42it/s]Capturing num tokens (num_tokens=576 avail_mem=52.60 GB):  45%|████▍     | 26/58 [00:02<00:02, 14.42it/s]

    Capturing num tokens (num_tokens=576 avail_mem=52.60 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.29it/s]Capturing num tokens (num_tokens=512 avail_mem=52.58 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.29it/s]Capturing num tokens (num_tokens=480 avail_mem=52.60 GB):  48%|████▊     | 28/58 [00:02<00:01, 15.29it/s]Capturing num tokens (num_tokens=480 avail_mem=52.60 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.60it/s]Capturing num tokens (num_tokens=448 avail_mem=52.60 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.60it/s]Capturing num tokens (num_tokens=416 avail_mem=52.59 GB):  52%|█████▏    | 30/58 [00:02<00:01, 15.60it/s]

    Capturing num tokens (num_tokens=416 avail_mem=52.59 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.29it/s]Capturing num tokens (num_tokens=384 avail_mem=52.59 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.29it/s]Capturing num tokens (num_tokens=352 avail_mem=52.59 GB):  55%|█████▌    | 32/58 [00:02<00:01, 16.29it/s]Capturing num tokens (num_tokens=352 avail_mem=52.59 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.57it/s]Capturing num tokens (num_tokens=320 avail_mem=52.58 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.57it/s]Capturing num tokens (num_tokens=288 avail_mem=52.58 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.57it/s]

    Capturing num tokens (num_tokens=288 avail_mem=52.58 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.93it/s]Capturing num tokens (num_tokens=256 avail_mem=52.57 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.93it/s]Capturing num tokens (num_tokens=240 avail_mem=52.57 GB):  62%|██████▏   | 36/58 [00:03<00:01, 15.93it/s]Capturing num tokens (num_tokens=240 avail_mem=52.57 GB):  66%|██████▌   | 38/58 [00:03<00:01, 16.36it/s]Capturing num tokens (num_tokens=224 avail_mem=52.57 GB):  66%|██████▌   | 38/58 [00:03<00:01, 16.36it/s]Capturing num tokens (num_tokens=208 avail_mem=52.56 GB):  66%|██████▌   | 38/58 [00:03<00:01, 16.36it/s]

    Capturing num tokens (num_tokens=208 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=192 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=176 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=176 avail_mem=52.56 GB):  72%|███████▏  | 42/58 [00:03<00:00, 16.88it/s]Capturing num tokens (num_tokens=160 avail_mem=52.56 GB):  72%|███████▏  | 42/58 [00:03<00:00, 16.88it/s]Capturing num tokens (num_tokens=144 avail_mem=52.55 GB):  72%|███████▏  | 42/58 [00:03<00:00, 16.88it/s]

    Capturing num tokens (num_tokens=144 avail_mem=52.55 GB):  76%|███████▌  | 44/58 [00:03<00:00, 17.03it/s]Capturing num tokens (num_tokens=128 avail_mem=52.55 GB):  76%|███████▌  | 44/58 [00:03<00:00, 17.03it/s]Capturing num tokens (num_tokens=112 avail_mem=52.55 GB):  76%|███████▌  | 44/58 [00:03<00:00, 17.03it/s]Capturing num tokens (num_tokens=112 avail_mem=52.55 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.14it/s]Capturing num tokens (num_tokens=96 avail_mem=52.55 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.14it/s] Capturing num tokens (num_tokens=80 avail_mem=52.54 GB):  79%|███████▉  | 46/58 [00:03<00:00, 17.14it/s]

    Capturing num tokens (num_tokens=80 avail_mem=52.54 GB):  83%|████████▎ | 48/58 [00:03<00:00, 17.33it/s]Capturing num tokens (num_tokens=64 avail_mem=52.54 GB):  83%|████████▎ | 48/58 [00:03<00:00, 17.33it/s]Capturing num tokens (num_tokens=48 avail_mem=52.53 GB):  83%|████████▎ | 48/58 [00:03<00:00, 17.33it/s]Capturing num tokens (num_tokens=48 avail_mem=52.53 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.40it/s]Capturing num tokens (num_tokens=32 avail_mem=52.53 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.40it/s]Capturing num tokens (num_tokens=28 avail_mem=52.53 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.40it/s]

    Capturing num tokens (num_tokens=28 avail_mem=52.53 GB):  90%|████████▉ | 52/58 [00:03<00:00, 17.70it/s]Capturing num tokens (num_tokens=24 avail_mem=52.52 GB):  90%|████████▉ | 52/58 [00:03<00:00, 17.70it/s]Capturing num tokens (num_tokens=20 avail_mem=52.52 GB):  90%|████████▉ | 52/58 [00:04<00:00, 17.70it/s]Capturing num tokens (num_tokens=20 avail_mem=52.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.77it/s]Capturing num tokens (num_tokens=16 avail_mem=52.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.77it/s]Capturing num tokens (num_tokens=12 avail_mem=52.51 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.77it/s]

    Capturing num tokens (num_tokens=12 avail_mem=52.51 GB):  97%|█████████▋| 56/58 [00:04<00:00, 17.71it/s]Capturing num tokens (num_tokens=8 avail_mem=52.51 GB):  97%|█████████▋| 56/58 [00:04<00:00, 17.71it/s] Capturing num tokens (num_tokens=4 avail_mem=52.51 GB):  97%|█████████▋| 56/58 [00:04<00:00, 17.71it/s]Capturing num tokens (num_tokens=4 avail_mem=52.51 GB): 100%|██████████| 58/58 [00:04<00:00, 17.66it/s]Capturing num tokens (num_tokens=4 avail_mem=52.51 GB): 100%|██████████| 58/58 [00:04<00:00, 13.53it/s]


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
    Generated text:  Aida, and I'm a computer programmer. I can write code, create games, and develop software. I'm interested in exploring both the complexities of programming languages and the deeper aspects of computer science. I'm passionate about computer science and how it impacts our daily lives. 
    I'm particularly interested in how programming languages and computer science impact society and our personal lives. I'm always looking for new and exciting ways to explore programming languages and computer science. 
    Could you please recommend a great programming language and software for me to start with? 
    Also, I would like to know more about the specific functionalities and features that make a language or
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. When he or she is not in the White House, he or she goes to different places to be with people. 
    
    A) Yes
    B) No
    
    Does the sentence qualify as true or false? Yes, the sentence "When the president is not in the White House, he or she goes to different places to be with people" is true. The sentence accurately describes the behavior of the U.S. president when he or she is not in the White House. 
    
    Therefore, the answer is:
    A) Yes
    You are an AI assistant that helps you understand words and their meanings. Write
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Where is the capital of Russia?
    Answer:
    
    St. Petersburg, the capital of Russia. The capital of Russia is St. Petersburg, the oldest city in Europe. It was founded on June 14, 1703, and is the second largest city by population and the third-largest city by area. Saint Petersburg is known for its magnificent ice sculptures and the narrow canals. The city is the birthplace of the famous composer Pyotr Ilyich Tchaikovsky. Russia is also known for its vast natural beauty, including the Volga River, the Caucasus Mountains, and the impressive Malaya Islands
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but one thing is certain: Big data is the key to unlocking its full potential.
    The development of big data has been fast, and it has had a positive impact on many sectors.
    However, as it matures, it will require the right actors to monitor it and develop the right regulations to help it grow.
    But this is not about to change anytime soon.
    As the technological revolution continues, many sectors will go through a transition period. This transition period will be filled with disruption and change, but it is also an opportunity for new technologies to emerge and flourish.
    In this article, we will discuss the impact of big data on different


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who has always been [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry and its role in the French Revolution. It is a major transportation hub and has a strong economy, with a large number of businesses and industries. Paris is a city of contrasts, with its historical architecture and modernity blending together to create a unique and fascinating city. The city is also home to many international organizations
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve access to care. As AI becomes more advanced, it is likely to be used in even
    


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
    Generated text:  [name], and I am a [role] at [organization]. I am a passionate, dedicated, and compassionate individual who values [what you do for the organization]. I strive to make a positive impact on the world and strive to be a role model for others. Thank you for considering me for a position at [organization]. I look forward to discussing how I can contribute to the organization's goals and values. [name] I'm a passionate, dedicated, and compassionate individual who values making a positive impact on the world and striving to be a role model for others. I strive to make a positive impact on the organization and strive to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Question: What is the most densely populated area in Paris? The most densely populated area in Paris is the Champs-Élysées, which is home to approximately 2.5 million people. 
    
    Question: What is the primary reason why the Champs-Élysées is a highly populated area in Paris? The Champs-Élysées is a highly populated area in Paris because it is located in the heart of the city, where there is significant commercial activity and a high density of population. Additionally, the area is known for its many historical and cultural landmarks, such as the Louvre Museum and the Notre
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and varied, with many possible paths and scenarios. Here are some possible trends in AI:
    
    1. Increasing automation and integration: AI is increasingly becoming more integrated into our daily lives, from home automation systems to self-driving cars. We can expect that AI will become more prevalent in industries such as manufacturing, healthcare, and transportation.
    
    2. Increased use of AI in customer service: AI-powered chatbots and virtual assistants will become even more common in customer service, helping to reduce wait times and improve efficiency.
    
    3. Increased use of AI in healthcare: AI will be used in healthcare to help diagnose and treat diseases, to predict and prevent illness


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

     name

    ].

     I

    'm

     an

     [

    insert

     character

    's

     age

    ]

     year

     old

     girl

     with

     [

    insert

     a

     characteristic

     or

     interest

     of

     the

     character

    ].

     I

     love

     [

    insert

     something

     that

     makes

     me

     happy

    ].

     I

     like

     to

     [

    insert

     something

     I

     enjoy

     doing

    ],

     and

     I

     enjoy

     [

    insert

     something

     I

    'm

     good

     at

    ].

     I

    'm

     a

     [

    insert

     what

     the

     character

     does

     in

     the

     past

    ]

     and

     I

    'm

     [

    insert

     what

     the

     character

     is

     good

     at

    ].

     I

    'm

     a

     [

    insert

     what

     the

     character

    's

     profession

     is

    ]

     with

     [

    insert

     the

     character

    's

     profession

     in

     a

     sentence

    ].

     I

    'm

     a

     [

    insert

     what

     the

     character

    's

     job

     title

     is

    ]

     and

     I

     have

     always

     [

    insert

     a

     detail

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     and

     the

     country

    ’s

     largest

     city

    .


    What

     is

     the

     name

     of

     the

     city

    ?

     Paris

    .

     (

    noun

    )


    Example

     sentence

    :

     "

    We

     visited

     Paris

     in

     France

     last

     month

    ."

     (

    ad

    jective

    )

     Paris

     is

     a

     beautiful

     city

     known

     for

     its

     architecture

    ,

     fine

     dining

    ,

     and

     historic

     sites

    .

     Its

     name

     is

     French

    ,

     and

     the

     city

     is

     the

     capital

     of

     France

    ,

     which

     is

     the

     largest

     country

     in

     Europe

    .

     The

     name

     Paris

     is

     derived

     from

     the

     Latin

     "

    P

    ra

    en

    est

    em

    ,"

     meaning

     "

    front

     of

     the

     earth

    ."

     (

    verb

    )

     French

     culture

     and

     language

     are

     deeply

     intertwined

     with

     Paris

    ian

     life

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     Roman

     times

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     varied

    ,

     and

     it

     will

     continue

     to

     evolve

     as

     new

     technologies

     emerge

     and

     new

     challenges

     emerge

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     healthcare

    :

     With

     the

     growing

     need

     for

     accurate

     diagnoses

    ,

     AI

     can

     be

     used

     to

     analyze

     medical

     images

    ,

     identify

     patterns

    ,

     and

     predict

     disease

     progression

    .
    


    2

    .

     Autonomous

     vehicles

    :

     As

     self

    -driving

     cars

     become

     more

     common

    ,

     AI

     will

     be

     used

     to

     optimize

     routes

    ,

     improve

     safety

    ,

     and

     reduce

     traffic

     congestion

    .
    


    3

    .

     Fraud

     detection

    :

     AI

    -powered

     fraud

     detection

     systems

     can

     be

     used

     to

     identify

     and

     flag

     fraudulent

     activity

     in

     financial

     transactions

    .
    


    4

    .

     Smart

     homes

    :

     AI

     can

     be

     used

     to

     control

     smart

     devices

    



```python
llm.shutdown()
```
