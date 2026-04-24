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
    [2026-04-24 02:21:09] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.11it/s]


    2026-04-24 02:21:13,761 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 02:21:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:19,  3.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:19,  3.50s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:19,  3.50s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:52,  1.05it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:52,  1.05it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:03<00:52,  1.05it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:03<00:52,  1.05it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:20,  2.56it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:20,  2.56it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:20,  2.56it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:20,  2.56it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:20,  2.56it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:09,  5.16it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:09,  5.16it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:09,  5.16it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:09,  5.16it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:09,  5.16it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:09,  5.16it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:04,  9.15it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:04,  9.15it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:04,  9.15it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:04,  9.15it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04,  9.15it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04,  9.15it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:04,  9.15it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:02, 16.00it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 25.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 42.88it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 48.32it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 48.32it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 48.32it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 48.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.24 GB):   2%|▏         | 1/58 [00:00<00:06,  8.94it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.16 GB):   2%|▏         | 1/58 [00:00<00:06,  8.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.20 GB):   2%|▏         | 1/58 [00:00<00:06,  8.94it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=117.20 GB):   5%|▌         | 3/58 [00:00<00:04, 13.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.18 GB):   5%|▌         | 3/58 [00:00<00:04, 13.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.19 GB):   5%|▌         | 3/58 [00:00<00:04, 13.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.19 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.17 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.16 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.15 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=117.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=117.16 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.16 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.15 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.14 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.14 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.13 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.73it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=117.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=960 avail_mem=117.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.98it/s] Capturing num tokens (num_tokens=896 avail_mem=117.09 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.98it/s]Capturing num tokens (num_tokens=832 avail_mem=117.09 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.98it/s]Capturing num tokens (num_tokens=768 avail_mem=117.08 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.98it/s]Capturing num tokens (num_tokens=768 avail_mem=117.08 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.62it/s]Capturing num tokens (num_tokens=704 avail_mem=117.08 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.62it/s]Capturing num tokens (num_tokens=640 avail_mem=117.07 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.62it/s]

    Capturing num tokens (num_tokens=576 avail_mem=117.06 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.62it/s]Capturing num tokens (num_tokens=512 avail_mem=117.05 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.62it/s]Capturing num tokens (num_tokens=512 avail_mem=117.05 GB):  50%|█████     | 29/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=480 avail_mem=117.06 GB):  50%|█████     | 29/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=448 avail_mem=117.06 GB):  50%|█████     | 29/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=416 avail_mem=117.05 GB):  50%|█████     | 29/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=384 avail_mem=117.05 GB):  50%|█████     | 29/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=384 avail_mem=117.05 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=352 avail_mem=117.03 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=320 avail_mem=117.03 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.07it/s]

    Capturing num tokens (num_tokens=288 avail_mem=117.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=256 avail_mem=117.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=256 avail_mem=117.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=240 avail_mem=117.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=224 avail_mem=117.01 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=208 avail_mem=117.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=192 avail_mem=117.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=192 avail_mem=117.00 GB):  71%|███████   | 41/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=176 avail_mem=116.99 GB):  71%|███████   | 41/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=160 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.61it/s]

    Capturing num tokens (num_tokens=144 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=128 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=112 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=112 avail_mem=116.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=96 avail_mem=116.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.41it/s] Capturing num tokens (num_tokens=80 avail_mem=116.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=64 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=48 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=32 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=32 avail_mem=116.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.62it/s]Capturing num tokens (num_tokens=28 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.62it/s]

    Capturing num tokens (num_tokens=24 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.62it/s]Capturing num tokens (num_tokens=20 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.62it/s]Capturing num tokens (num_tokens=16 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.62it/s]Capturing num tokens (num_tokens=16 avail_mem=116.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=12 avail_mem=116.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=8 avail_mem=116.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.41it/s] Capturing num tokens (num_tokens=4 avail_mem=116.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=4 avail_mem=116.94 GB): 100%|██████████| 58/58 [00:01<00:00, 30.05it/s]


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
    Generated text:  Tim. I used to live in Switzerland. I know a lot of Swiss people, but I didn't know much about Switzerland.
    
    I am writing a memo about a trip I had in Switzerland. The trip was very successful, but I wanted to share my experiences with others. I would like you to write a memo with the information you found about Switzerland and my experience.
    
    Sure, I'd be happy to help you write a memo about your trip to Switzerland. Can you please provide me with the information you found about Switzerland and your experience? 
    
    Additionally, could you please suggest some tips for someone planning a trip to Switzerland? Any specific things
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the 2020 presidential election as the basis for a presidential proclamation. It would be an easy victory if he could use the election results to support his presidential proposal, but it would also be difficult to get people to agree with him on the policy. It is based on the passage what can be inferred about the president?
    
    I. He has no chance of winning the election;
    
    II. He is not interested in winning the election;
    
    III. He is not a good leader;
    
    IV. He is not a good candidate;;
    
    V. He is not a good president;;
    
    VI. He is not
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Nice
    C. Milan
    D. London
    Answer:
    
    A
    
    When a single-phase ground fault occurs in a system, the fault current flows through the ______ of the circuit.
    A. Insulation
    B. Conductor
    C. Insulation and conductor
    D. Grounding
    Answer:
    
    B
    
    According to the current "Railway Technical Management Regulations", the interval time for the use of wireless communication equipment for shunting operations is specified as ____.
    A. 1 minute
    B. 2 minutes
    C. 3 minutes
    D. 4 minutes
    Answer:
    
    B
    
    According
    ===============================
    Prompt: The future of AI is
    Generated text:  here. In the next five years, the demand for AI technology is expected to grow at a steady rate, with annual growth rates expected to increase from the current rate of 28% to 46% by 2025, and from 28% to 51% by 2035. The demand for AI technology is expected to increase significantly in the following five years, with annual growth rates expected to increase from the current rate of 28% to 46% by 2025, and from 28% to 51% by 203


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Parliament Building. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is also known for its cuisine, including its famous French fries and croissants. Paris is a vibrant and dynamic city that is a must-visit for anyone interested in French culture and history. 
    
    The French capital is also home to many other notable landmarks, including the Palace of Versailles, the Ch
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in various industries.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical and responsible AI. This will include considerations of privacy, bias, and transparency
    


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
    Generated text:  [insert your name], and I'm a [insert your occupation] who loves to [insert something you do or write about that you enjoy]. I'm constantly motivated to learn and grow, and I'm always looking for opportunities to inspire others. Whether it's through reading, writing, or simply engaging in conversations with people, I'm always looking for ways to connect with others and contribute to their lives. Whether you're a friend, family member, or someone else I meet, I'm always ready to learn more about you and what makes you special. So, if you have any questions or want to connect, please let me know!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, beautiful architecture, and vibrant culture.
    
    The statement is: **Paris, the capital city of France, is known for its rich history, beautiful architecture, and vibrant culture.** 
    
    This concise statement summarizes the key points about Paris in a single sentence, encapsulating its reputation as a city with a fascinating history, picturesque architecture, and lively atmosphere. It's a brief yet informative description of one of France's most iconic cities. 
    
    If you need a more detailed answer, I can provide additional context or discuss Paris in more depth. Let me know if you'd like me to expand on any aspect of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly complex and diverse, with potential developments and innovations in various areas. Here are some of the potential future trends in AI:
    
    1. Increased automation and efficiency: AI is already enabling many automation and efficiency improvements in industries such as manufacturing, healthcare, and transportation. As AI continues to improve and become more sophisticated, we can expect to see even greater automation in these industries.
    
    2. Enhanced cognitive functions: AI is expected to continue evolving and improving in ways that will enhance our ability to think and reason. This could include advancements in areas such as language translation, image recognition, and decision-making.
    
    3. Improved privacy and security:


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

     a

     [

    profession

     or

     role

    ].

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    occupation

    ].

     My

     [

    area

     of

     expertise

    ]

     is

     [

    specific

     area

     of

     expertise

    ].

     I

     have

     always

     been

     passionate

     about

     [

    why

     I

     love

     my

     work

    ].

     I

     am

     a

     [

    person

    ality

     type

    ].

     I

     am

     [

    add

     a

     few

     sentences

     that

     highlight

     your

     unique

     qualities

    ].

     I

     love

     [

    thing

     I

     enjoy

     doing

    ].


    Sure

    ,

     here

    's

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    :
    


    "

    Hi

     there

    !

     I

    'm

     [

    Name

    ]

     from

     [

    Location

    ],

     an

     experienced

     [

    profession

     or

     role

    ]

     with

     [

    number

    ]

     years

     of

     experience

    .

     My

     [

    area

     of

     expertise

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    
    The

     statement

     is

     factual

     because

     it

     provides

     a

     clear

     and

     precise

     description

     of

     the

     location

     of

     the

     capital

     city

     of

     France

    ,

     stating

     that

     it

     is

     the

     capital

     and

     having

     a

     definite

     location

    .

     No

     additional

     information

     or

     context

     is

     needed

     to

     confirm

     or

     deny

     this

     statement

    ,

     as

     it

     is

     a

     well

    -known

     and

     widely

     recognized

     fact

    .

     
    


    The

     statement

     is

     also

     concise

     and

     easy

     to

     understand

    ,

     making

     it

     an

     appropriate

     answer

     to

     the

     question

    .

     It

     avoids

     any

     unnecessary

     details

     or

     explanations

    ,

     and

     provides

     a

     clear

     and

     comprehensive

     overview

     of

     the

     capital

     city

    's

     location

     in

     relation

     to

     France

    's

     overall

     political

     and

     cultural

     framework

    .

     
    


    Therefore

    ,

     the

     given

     statement

     is

     fact

    ually

     correct

     and

     appropriate

     for

     answering

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     trends

     that

     will

     shape

     the

     development

     of

     this

     technology

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     AI

     in

     the

     future

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

     AI

     systems

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     an

     increased

     focus

     on

     ethical

     considerations

    ,

     such

     as

     ensuring

     that

     AI

     systems

     are

     used

     to

     benefit

     society

     as

     a

     whole

     rather

     than

     just

     for

     profit

    .

     This

     may

     lead

     to

     increased

     regulation

     and

     oversight

     of

     AI

     systems

    .
    


    2

    .

     More

     autonomous

     systems

    :

     In

     addition

     to

     autonomous

     vehicles

    ,

     there

     is

     a

     growing

     trend

     towards

     more

     autonomous

     systems

     in

     fields

     such

     as

     healthcare

     and

     transportation

    .

     These

    



```python
llm.shutdown()
```
