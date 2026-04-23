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
    [2026-04-23 08:59:30] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]


    2026-04-23 08:59:35,175 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 08:59:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.43it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.43it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.43it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.43it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.43it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.43it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.43it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.43it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.43it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.43it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.85it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:03<00:00, 25.21it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s] 

    Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 42.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.79 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   7%|▋         | 4/58 [00:00<00:03, 15.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   7%|▋         | 4/58 [00:00<00:03, 15.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   7%|▋         | 4/58 [00:00<00:03, 15.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 15.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 15.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 15.09it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.41it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.28it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.51it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.51it/s]

    Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.51it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.51it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.51it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.51it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.66it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.66it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.66it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.66it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.66it/s]

    Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.72it/s]

    Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.16it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.16it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.28it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.25it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.25it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 33.54it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 27.85it/s]


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
    Generated text:  Emma and I’m a Senior Writer at the UK Library. I’ve also worked as a regular contributor to various publications, including Library Journal, The Libraries, and the Library Management Association. I’m a speaker, blogger, and manager of several online and offline projects. Most of my work is focused on libraries, but I also share the knowledge of information and how to use technology in various ways. I have a deep understanding of how libraries work and how they help people to access information and knowledge. I’m very involved in the development of the library experience. I believe strongly in the importance of library services, and I value the work and dedication
    ===============================
    Prompt: The president of the United States is
    Generated text:  an important state of the United States. With 46 states, it's important for the president to be able to deal with the nation's political, social, and economic issues. The president is the head of government of the United States and the commander-in-chief of the armed forces. This makes the president the key player in the nation's politics and the nation's leadership. The president has a lot of power and responsibility to ensure the stability of the country. He or she can make major decisions on the economy, foreign policy, and national security.
    What are the duties of the president of the United States? (If the question is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is located in which country? France is located in Europe, and Paris is the capital of France. Europe consists of several countries, but France is one of them. France is an island country located on the Mediterranean Sea, and it is divided into two parts: the north, which is the French region of the Mediterranean Sea, and the south, which is the region of the Atlantic Ocean. Paris is the capital of France, and it is located in France, which is an island country located on the Mediterranean Sea.
    ===============================
    Prompt: The future of AI is
    Generated text:  the future of data management and analytics, and it requires an even more innovative approach than the past.
    The following will be covered in this article:
    • The definition of AI and its importance in the industry
    • The current landscape of AI in the data management and analytics space
    • The importance of innovation in the AI industry
    • An overview of the challenges that the industry is facing
    • What is the future of AI and data management in the industry?
    • The future of AI in the data management and analytics space
    • The future of AI in the industry
    • Conclusion
    AI is an increasingly important technology, and the industry is rapidly


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and diverse culture. It is located in the south of France and is the largest city in the country. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also known for its food, fashion, and music scenes. Paris is a popular tourist destination and is home to many world-renowned museums, theaters, and art galleries. It is a cultural and intellectual center of Europe and a major economic and financial hub. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ensuring that AI systems are developed and used in ways that are fair, transparent, and ethical.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI
    


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
    Generated text:  [Name] and I am a [Age] year old student at [School Name]. I am currently in my [Occupation/Field] degree program, pursuing [Your Major] major. I am passionate about [What I Love to Do/What I Want to Do] and I am always looking for opportunities to learn new things. I enjoy [What I Enjoy Doing], and I always strive to be the best version of myself. [Name] is a young person who is eager to grow and excel in their chosen field. They are determined to achieve their goals and make a positive impact on the world around them. [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south of the country, and is the largest and most populous city in the country. 
    
    Fact: The capital of France is Paris, located in the south of the country, and is the largest and most populous city in the country. 
    
    City: Paris, located in the south of the country, is the capital of France and the largest and most populous city in the country. 
    
    This statement concisely and accurately summarizes the key facts about Paris's location, size, and role as the capital city of France. It provides a clear and concise understanding of the city's status within the broader context of France and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising with many possibilities to come. Here are some potential trends that could shape the AI landscape in the years to come:
    
    1. **Machine Learning and Deep Learning**: These are the two main areas of focus in AI research. While machine learning has already achieved remarkable success in various applications, deep learning is poised to become even more powerful. Deep learning techniques are designed to handle large datasets and intricate patterns in data that traditional machine learning models struggle with. This could lead to breakthroughs in areas like image recognition, natural language processing, and even autonomous driving.
    
    2. **Quantum Computing**: Quantum computers have the potential to significantly accelerate the


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

    ].

     I

     am

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     and

     I

     currently

     live

     in

     [

    Current

     Location

    ].

     I

     am

     an

     [

    Occup

    ation

    ]

     who

     is

     always

     [

    Big

    gest

     Habit

    ].

     How

     are

     you

    ?

     I

    ’m

     not

     really

     sure

     about

     this

    ,

     I

     don

    ’t

     know

     how

     to

     begin

     a

     conversation

    .

     How

     do

     you

     start

    ?

     To

     start

     a

     conversation

    ,

     I

     would

     say

     something

     like

    :

     “

    Hey

    !

     How

     are

     you

    ?”

     I

    ’d

     then

     try

     to

     get

     to

     know

     you

     and

     learn

     about

     you

    .

     Would

     you

     be

     interested

     in

     learning

     more

     about

     yourself?

     I

    ’d

     ask

     you

     what

     you

     do

    ,

     what

     you

     enjoy

    ,

     what

     you

     like

     to

     do

    ,

     or

     any

     other

    
    
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

     and

     is

     known

     for

     its

     historical

    ,

     cultural

    ,

     and

     artistic

     attractions

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

     E

    iff

    el

     Tower

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

     a

     major

     international

     center

     for

     business

    ,

     finance

    ,

     and

     media

    .

     Its

     nickname

     “

    La

     Ville

     Bl

    anche

    ”

     refers

     to

     its

     white

     marble

     buildings

     and

     its

     reputation

     for

     being

     a

     lively

     and

     exciting

     place

     to

     live

     and

     work

    .

     France

    ’s

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

     Europe

     and

     known

     for

     its

     historical

    ,

     cultural

    ,

     and

     artistic

     attractions

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

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     certainly

     bright

    ,

     with

     many

     possibilities

     and

     advancements

     to

     come

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     could

     shape

     the

     future

     of

     the

     technology

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     the

     ethical

     implications

     of

     AI

     continue

     to

     emerge

    ,

     there

     will

     likely

     be

     greater

     focus

     on

     ensuring

     that

     AI

     is

     used

     in

     ways

     that

     benefit

     society

     as

     a

     whole

    .

     This

     could

     involve

     developing

     more

     transparent

     and

     accountable

     AI

     systems

    ,

     as

     well

     as

     more

     ethical

     guidelines

     for

     how

     AI

     is

     used

    .
    


    2

    .

     Deep

     learning

     and

     machine

     learning

    :

     The

     ability

     to

     process

     and

     analyze

     large

     amounts

     of

     data

     using

     deep

     learning

     and

     machine

     learning

     techniques

     is

     likely

     to

     become

     even

     more

     prominent

     in

     the

     future

    .

     This

     could

     lead

    



```python
llm.shutdown()
```
