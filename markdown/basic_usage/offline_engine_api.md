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
    [2026-04-26 03:44:18] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.54it/s]


    2026-04-26 03:44:23,686 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 03:44:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:02<00:13,  3.89it/s]Compiling num tokens (num_tokens=2816):  12%|█▏        | 7/58 [00:03<00:13,  3.89it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:04,  9.49it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 14.89it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]

    Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 21.96it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 30.39it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 37.08it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.51it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 49.71it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 49.71it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   7%|▋         | 4/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   7%|▋         | 4/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 14.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.75 GB):  10%|█         | 6/58 [00:00<00:03, 14.90it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 14.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.04it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.55it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.89it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.61it/s]Capturing num tokens (num_tokens=960 avail_mem=118.72 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.61it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.61it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.61it/s]Capturing num tokens (num_tokens=768 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.61it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.71 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.07it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.07it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.07it/s]Capturing num tokens (num_tokens=576 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.07it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.07it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.07it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.06it/s]

    Capturing num tokens (num_tokens=320 avail_mem=118.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=320 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  60%|██████    | 35/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  60%|██████    | 35/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.64it/s]Capturing num tokens (num_tokens=192 avail_mem=118.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.64it/s]Capturing num tokens (num_tokens=176 avail_mem=118.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.64it/s]Capturing num tokens (num_tokens=160 avail_mem=118.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.64it/s]

    Capturing num tokens (num_tokens=144 avail_mem=118.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.64it/s]Capturing num tokens (num_tokens=144 avail_mem=118.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=128 avail_mem=118.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=112 avail_mem=118.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=96 avail_mem=118.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.79it/s] Capturing num tokens (num_tokens=80 avail_mem=118.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=80 avail_mem=118.12 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.88it/s]Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.88it/s]

    Capturing num tokens (num_tokens=48 avail_mem=117.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.88it/s]Capturing num tokens (num_tokens=32 avail_mem=117.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.88it/s]Capturing num tokens (num_tokens=28 avail_mem=117.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.88it/s]Capturing num tokens (num_tokens=28 avail_mem=117.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=20 avail_mem=117.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=16 avail_mem=117.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=12 avail_mem=117.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.19it/s]

    Capturing num tokens (num_tokens=12 avail_mem=117.91 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.69it/s]Capturing num tokens (num_tokens=8 avail_mem=117.91 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.69it/s] Capturing num tokens (num_tokens=4 avail_mem=117.91 GB):  97%|█████████▋| 56/58 [00:02<00:00, 32.69it/s]Capturing num tokens (num_tokens=4 avail_mem=117.91 GB): 100%|██████████| 58/58 [00:02<00:00, 28.53it/s]


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
    Generated text:  Alex and I'm a non-profit marketing and analytics firm that specializes in helping small businesses grow their online presence. I've been with the firm for four years now and have had a lot of experience helping businesses grow online through SEO, PPC, social media marketing, and lead generation.
    Today, I'm going to share some of the key things that I think small businesses should be thinking about when it comes to digital marketing.
    First and foremost, it's important for small businesses to have a solid plan for online presence. This includes developing a clear brand image, creating a strong on-page SEO strategy, and creating a content strategy. It's also
    ===============================
    Prompt: The president of the United States is
    Generated text:  very proud of his country and wants to promote unity among its diverse people. In order to do so, he decided to hold a national unity festival. He also wants to include a special event where everyone can come together for a day of celebration and joy. He also wants to make sure that everyone can participate in the festival in a way that is inclusive and respectful of all cultures and backgrounds.
    
    What kind of festival would the president of the United States likely organize? The president of the United States would likely organize a "National Unity Day" festival. This type of festival would be inclusive and respectful of all cultures and backgrounds, as it would celebrate
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Answer this question based on the passage: where is paris located in france?
    
    Available options:
     +city
     +country
     +state
     +region
    The answer is Paris, which is located in France. Based on the passage, Paris is the capital city of France. Therefore, the correct answer is:
    
    +city
    ===============================
    Prompt: The future of AI is
    Generated text:  decentralized. But is it decentralized enough?
    
    Is there anything more surprising than seeing one person’s hologram appear on the screen of another person’s computer? Or watching a hologram appear on a screen of a hologram? Or watching a hologram appearing on a hologram? Or, in the past, just watching a hologram appear on a screen of a hologram? Or watching a hologram appear on a screen of a screen of a hologram? Or watching a hologram appear on a screen of a screen of a screen of a hologram?
    
    You get the point. I’m not sure if I could have imagined it,


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number] years of experience in [Field]. I'm a [Skill] with [Number]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for business, finance, and tourism in France. Paris is a popular tourist destination and a cultural hub for the French people. The city is home to many museums, theaters, and other cultural institutions. It is also known for its cuisine,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some possible future trends in AI include:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations and tasks more effectively. This could lead to more efficient and effective decision-making in various industries.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [Name] and I'm a [occupation] with [number of years] of experience. I've always been fascinated by [one thing or two things] and I strive to do my best, no matter what. If you're looking for a professional or a hobby, I'd be happy to help with [one or more projects]. My [occupation] background will help me in my [occupation] role. What would you like to know about me? [Name]: Hi there! I'm [Name] and I'm a professional [occupation], with [number of years] of experience. I've always been fascinated by [one
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a city located in the center of the country.
    
    Paris is a city in the south of France, known for its medieval architecture, the Eiffel Tower, and its numerous museums and cultural attractions. It is one of the most visited cities in the world and is considered one of the most important and populous cities in the world. Paris is also home to the Louvre Museum, the Notre-Dame Cathedral, and the Champs-Élysées. The city is a cultural center, known for its art, music, and cuisine, and has been the seat of power and government for more than a thousand years. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  dynamic, and there are many potential developments that could shape its course. Here are some possible future trends in AI:
    
    1. Advancements in Machine Learning: Machine learning is the most exciting area of AI right now, with many new techniques and algorithms being developed to improve accuracy and efficiency in AI systems.
    
    2. Increased Use of AI in Healthcare: AI is already being used to improve healthcare outcomes, from better disease diagnosis to more effective drug development. As AI technology continues to improve, we can expect it to play an even greater role in healthcare in the future.
    
    3. AI Ethics and Bias: As AI systems become more integrated into our lives


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

     am

     a

     [

    Type

    ]

     [

    Background

    ].

     I

     have

     [

    Number

    ]

     years

     of

     experience

     in

     [

    Field

    ],

     and

     I

     am

     [

    Type

    ]

     [

    Background

    ].

     I

     am

     passionate

     about

     [

    Your

     Passion

    ],

     and

     I

     am

     always

     seeking

     to

     learn

     new

     things

     and

     improve

     my

     skills

    .

     [

    Type

    ]

     [

    Background

    ]

     [

    Your

     Profession

    ],

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     grow

     and

     develop

     myself

    .

     I

     am

     a

     [

    Type

    ]

     [

    Background

    ]

     [

    Your

     Profession

    ],

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     grow

     and

     develop

     myself

    .

     I

     am

     [

    Type

    ]

     [

    Background

    ],

     and

     I

     am

     always

     seeking

     to

     learn

     new

     things

     and

     improve

     my

     skills

    .

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     also

     a

     UNESCO

     World

     Heritage

     site

     and

     one

     of

     the

     most

     important

     cities

     in

     Europe

    .

     The

     city

     is

     known

     for

     its

     historical

     architecture

    ,

     beautiful

     parks

    ,

     and

     rich

     cultural

     scene

    .

     It

     is

     the

     birth

    place

     of

     many

     famous

     figures

     such

     as

     Napoleon

     and

     Victor

     Hugo

    .

     The

     city

     also

     hosts

     numerous

     major

     events

     and

     festivals

     throughout

     the

     year

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     center

     in

     Europe

    .

     It

     is

     the

     second

     most

     visited

     city

     in

     the

     world

     after

     New

     York

     City

    .

     The

     city

     is

     known

     for

     its

     skyline

    ,

     many

     museums

     and

     art

     galleries

    ,

     and

     food

     and

     wine

     culture

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     possibilities

    ,

     and

     here

     are

     some

     potential

     trends

     that

     we

     can

     expect

     to

     see

    :
    


    1

    .

     Increased

     efficiency

     and

     accuracy

     in

     natural

     language

     processing

    :

     With

     the

     ongoing

     advancements

     in

     machine

     learning

     and

     deep

     learning

    ,

     we

     can

     expect

     to

     see

     improvements

     in

     natural

     language

     processing

    ,

     including

     more

     accurate

     translations

    ,

     better

     understanding

     of

     human

     language

    ,

     and

     faster

     processing

     of

     text

     and

     speech

     data

    .
    


    2

    .

     Enhanced

     autonomous

     decision

    -making

    :

     AI

     is

     becoming

     more

     capable

     of

     making

     autonomous

     decisions

     based

     on

     data

     analysis

     and

     pattern

     recognition

    .

     This

     means

     that

     machines

     will

     be

     able

     to

     make

     decisions

     that

     are

     not

     only

     accurate

     but

     also

     fair

     and

     ethical

    ,

     which

     will

     be

     more

     widely

     adopted

     in

     various

     industries

    .
    


    3

    



```python
llm.shutdown()
```
