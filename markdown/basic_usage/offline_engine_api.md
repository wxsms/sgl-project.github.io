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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 19:12:13] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.32it/s]


    2026-04-26 19:12:19,519 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 19:12:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]

    Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:19,  2.57it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.43it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.43it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.43it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.43it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.71it/s]

    Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.36it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:03, 11.36it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 16.55it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 24.82it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 34.28it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 43.04it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 52.21it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 52.21it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 52.21it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 52.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.77 GB):   2%|▏         | 1/58 [00:00<00:08,  6.66it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.73 GB):   2%|▏         | 1/58 [00:00<00:08,  6.66it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=71.73 GB):   3%|▎         | 2/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.74 GB):   3%|▎         | 2/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.74 GB):   5%|▌         | 3/58 [00:00<00:06,  7.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.75 GB):   5%|▌         | 3/58 [00:00<00:06,  7.98it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:06,  8.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.76 GB):   7%|▋         | 4/58 [00:00<00:06,  8.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   7%|▋         | 4/58 [00:00<00:06,  8.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):  10%|█         | 6/58 [00:00<00:05,  9.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):  10%|█         | 6/58 [00:00<00:05,  9.91it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.76 GB):  10%|█         | 6/58 [00:00<00:05,  9.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.76 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.78 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.77 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.77 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.77 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.90it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.80 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.80 GB):  21%|██        | 12/58 [00:01<00:03, 14.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.87 GB):  21%|██        | 12/58 [00:01<00:03, 14.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.84 GB):  21%|██        | 12/58 [00:01<00:03, 14.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.78 GB):  21%|██        | 12/58 [00:01<00:03, 14.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.78 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.78 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.08it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.78 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.77 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.77 GB):  31%|███       | 18/58 [00:01<00:02, 19.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.76 GB):  31%|███       | 18/58 [00:01<00:02, 19.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.81 GB):  31%|███       | 18/58 [00:01<00:02, 19.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.75 GB):  31%|███       | 18/58 [00:01<00:02, 19.82it/s]Capturing num tokens (num_tokens=960 avail_mem=71.79 GB):  31%|███       | 18/58 [00:01<00:02, 19.82it/s] Capturing num tokens (num_tokens=960 avail_mem=71.79 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.90it/s]Capturing num tokens (num_tokens=896 avail_mem=71.78 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.90it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.78 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.90it/s]Capturing num tokens (num_tokens=768 avail_mem=71.76 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.90it/s]Capturing num tokens (num_tokens=704 avail_mem=71.74 GB):  38%|███▊      | 22/58 [00:01<00:01, 23.90it/s]Capturing num tokens (num_tokens=704 avail_mem=71.74 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.69it/s]Capturing num tokens (num_tokens=640 avail_mem=71.73 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.69it/s]Capturing num tokens (num_tokens=576 avail_mem=71.74 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.69it/s]Capturing num tokens (num_tokens=512 avail_mem=71.71 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.69it/s]Capturing num tokens (num_tokens=480 avail_mem=71.71 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.69it/s]Capturing num tokens (num_tokens=480 avail_mem=71.71 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.16it/s]Capturing num tokens (num_tokens=448 avail_mem=71.72 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.16it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.71 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.16it/s]Capturing num tokens (num_tokens=384 avail_mem=71.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.16it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.16it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.02it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.02it/s]Capturing num tokens (num_tokens=288 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.02it/s]Capturing num tokens (num_tokens=256 avail_mem=71.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.02it/s]Capturing num tokens (num_tokens=240 avail_mem=71.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.02it/s]Capturing num tokens (num_tokens=240 avail_mem=71.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=224 avail_mem=71.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.74it/s]

    Capturing num tokens (num_tokens=208 avail_mem=71.63 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=192 avail_mem=71.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=176 avail_mem=71.61 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=176 avail_mem=71.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=160 avail_mem=71.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=144 avail_mem=71.60 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=128 avail_mem=71.59 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=112 avail_mem=71.58 GB):  72%|███████▏  | 42/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=112 avail_mem=71.58 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.98it/s]Capturing num tokens (num_tokens=96 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.98it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=71.56 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.98it/s]Capturing num tokens (num_tokens=64 avail_mem=71.55 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.98it/s]Capturing num tokens (num_tokens=48 avail_mem=71.54 GB):  79%|███████▉  | 46/58 [00:02<00:00, 35.98it/s]Capturing num tokens (num_tokens=48 avail_mem=71.54 GB):  86%|████████▌ | 50/58 [00:02<00:00, 36.81it/s]Capturing num tokens (num_tokens=32 avail_mem=71.55 GB):  86%|████████▌ | 50/58 [00:02<00:00, 36.81it/s]Capturing num tokens (num_tokens=28 avail_mem=71.53 GB):  86%|████████▌ | 50/58 [00:02<00:00, 36.81it/s]Capturing num tokens (num_tokens=24 avail_mem=71.53 GB):  86%|████████▌ | 50/58 [00:02<00:00, 36.81it/s]Capturing num tokens (num_tokens=20 avail_mem=71.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 36.81it/s]Capturing num tokens (num_tokens=20 avail_mem=71.52 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.65it/s]Capturing num tokens (num_tokens=16 avail_mem=71.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.65it/s]

    Capturing num tokens (num_tokens=12 avail_mem=71.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.65it/s]Capturing num tokens (num_tokens=8 avail_mem=71.49 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.65it/s] Capturing num tokens (num_tokens=4 avail_mem=71.48 GB):  93%|█████████▎| 54/58 [00:02<00:00, 37.65it/s]Capturing num tokens (num_tokens=4 avail_mem=71.48 GB): 100%|██████████| 58/58 [00:02<00:00, 38.15it/s]Capturing num tokens (num_tokens=4 avail_mem=71.48 GB): 100%|██████████| 58/58 [00:02<00:00, 24.72it/s]


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
    Generated text: : [Your Name]
    I would like to request a [loan amount] from [bank name]. I would like to take out [loan type] and [loan duration] to help [specific purpose]. Please provide [loan details] below.
    
    Loan Amount: [loan amount]
    Loan Type: [loan type]
    Loan Duration: [loan duration]
    Purpose: [specific purpose]
    
    You will need to confirm the loan details by signing below. If you have any questions or concerns, please do not hesitate to reach out.
    
    Signature: ____________________________
    Date: _____________________________
    Loan Details: ________________________
    [bank name] ____________________________
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  34 years older than the president of Brazil. The president of Brazil is 2 times older than the president of France. If the president of France is currently 20 years old, how old would the president of Brazil be in 5 years?
    
    To determine the age of the president of Brazil in 5 years, we need to follow a step-by-step approach:
    
    1. Identify the current age of the president of France.
    2. Determine the current age of the president of Brazil.
    3. Calculate the president of Brazil's age in 5 years.
    
    First, we know that the president of France is currently 20
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris  
    B: Lyon  
    C: Paris, Lyon  
    D: Besançon
    To determine the capital of France, let's first identify the countries that are located in France:
    
    1. France is a country, not a city.
    2. The capital of a country is the city that is the main location where the country's government and most important institutions are based.
    
    Now, let's list the capital cities of the countries:
    - The capital of Switzerland is Basel.
    - The capital of Belgium is Brussels.
    - The capital of Germany is Berlin.
    - The capital of France is Paris.
    - The capital of Luxembourg is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the public good, and that future is growing increasingly complex and varied, and the question of its role and purpose is difficult to pin down. As in any mature discipline, the pursuit of the future of AI involves competing visions of the future of AI, some of which are better than others, and the evaluation of these visions is an important part of the discipline. Unfortunately, the promise of the future of AI seems to be divided along lines of competing visions and competing models of an increasingly complex future.
    Why does the author mention the term "mature discipline"? The author mentions the term "mature discipline" to emphasize the complexity of the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French language. The city is also home to many famous French artists, writers, and musicians. Paris is a popular tourist destination and a major hub for international business and diplomacy. It is a major cultural and economic center in Europe and a major hub for international trade. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing ethical AI that is designed to minimize harm and maximize benefits. This could involve developing AI that is designed to be transparent, accountable, and accountable, and that is designed to be used in a way that is consistent with the values of society.
    
    2. Greater use of AI in healthcare: AI is already being
    


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
    Generated text:  [Your Name], and I'm a [brief description of your character, such as "artistic", "intrepid", "funny", "brave", "charismatic", etc. ].
    [Your Name]. Welcome to my world. I am [Your Name] and I am a member of [character's team]. I come from [country], [city], or [location], and I have always been [some description, such as "fun-loving", "excitable", "tough", "relatable", "inspirational", etc. ]. I am [role], and I love to [something, such
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and has a rich history dating back to its founding as a Roman colony in the 1st century BC. It is the second largest city in the European Union and the most populous city in France by population. The city is known for its beautiful architecture, cultural attractions, and livable urban environment. Paris is also home to many of France's most famous landmarks, including the Eiffel Tower and the Louvre Museum.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and rapidly evolving. Here are some possible trends that we can expect to see in the near and long term:
    
    1. Increased automation: As AI technology continues to advance, we can expect to see a significant increase in automation in various industries. This will lead to the creation of new jobs and increase productivity. However, it will also lead to job displacement, so there will be an increased focus on skills training and retraining programs.
    
    2. Personalized AI: AI will become more capable of understanding and responding to personalization. This will be achieved through the use of machine learning and natural language processing. Personalized AI will allow machines to


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

     I

     am

     a

     [

    profession

    ],

     and

     I

    'm

     a

     bit

     of

     a

     trouble

    maker

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     Sure

    ,

     I

    'm

     a

     self

    -pro

    claimed

     trouble

    maker

     who

     likes

     to

     get

     into

     trouble

     with

     the

     law

    .

     I

     enjoy

     solving

     problems

     and

     helping

     others

    ,

     but

     I

    'm

     not

     afraid

     to

     take

     risks

     and

     go

     for

     it

    .

     What

     brings

     you

     to

     this

     profession

    ?

     I

    'm

     looking

     for

     a

     problem

     to

     solve

    ,

     to

     make

     a

     difference

     in

     the

     world

    .

     What

     are

     some

     of

     your

     favorite

     hobbies

     or

     activities

    ?

     I

     love

     reading

     and

     spending

     time

     with

     my

     family

    .

     I

     also

     love

     painting

     and

     creating

     art

    .

     What

     advice

     would

     you

     give

     to

     someone

     who

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     France

    ,

     and

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    .
    


    France

    's

     capital

     city

     is

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    .

     The

     city

    's

     unique

     history

    ,

     culture

    ,

     and

     architecture

     are

     celebrated

     throughout

     the

     country

     and

     attract

     millions

     of

     visitors

     every

     year

    .

     Paris

     is

     known

     for

     its

     art

    ,

     music

    ,

     food

    ,

     and

     fashion

    ,

     and

     it

     is

     also

     home

     to

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

     many

     other

     famous

     landmarks

    .

     Despite

     its

     fame

    ,

     Paris

     remains

     a

     peaceful

     and

     peaceful

     city

    ,

     and

     its

     people

     are

     friendly

     and

     welcoming

     to

     visitors

     of

     all

     backgrounds

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     combination

     of

     advancements

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

     that

     may

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Integration

     with

     Human

     Expert

    ise

    :

     AI

     systems

     may

     become

     more

     integrated

     with

     human

     expertise

     and

     cognitive

     abilities

    ,

     enabling

     them

     to

     make

     decisions

     that

     are

     more

     human

    -like

    .

     This

     integration

     could

     lead

     to

     more

     effective

     and

     ethical

     use

     of

     AI

    ,

     and

     potentially

     improve

     the

     quality

     of

     human

     decision

    -making

    .
    


    2

    .

     Increased

     Efficiency

     and

     Energy

     Efficiency

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     may

     become

     more

     energy

    -efficient

     and

     less

     energy

    -intensive

    .

     This

     could

     lead

     to

     significant

     reductions

     in

     carbon

     emissions

     and

     increased

     efficiency

     in

     various

     industries

    .
    


    3

    .

     Adv

    ancements

     in

    



```python
llm.shutdown()
```
