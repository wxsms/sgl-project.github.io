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
    [2026-04-20 23:16:51] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.39it/s]


    2026-04-20 23:16:56,361 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 23:16:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]

    Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:12,  4.13it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:12,  4.13it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:12,  4.13it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:12,  4.13it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:12,  4.13it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  7.18it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.68it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.68it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.68it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.68it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.68it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.68it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.68it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]

    Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 18.93it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 26.74it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]

    Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 34.32it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 41.46it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 47.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.61 GB):   3%|▎         | 2/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.24 GB):   3%|▎         | 2/58 [00:00<00:03, 17.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.24 GB):   3%|▎         | 2/58 [00:00<00:03, 17.12it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.24 GB):   7%|▋         | 4/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.06 GB):   7%|▋         | 4/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.05 GB):   7%|▋         | 4/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.05 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.06 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.05 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.05 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.05 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.05 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.05 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.04 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.70it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=118.04 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.04 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.04 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.04 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.03 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.03 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 21.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 21.96it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=118.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 21.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 21.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.01 GB):  28%|██▊       | 16/58 [00:00<00:01, 21.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.99 GB):  28%|██▊       | 16/58 [00:00<00:01, 21.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.99 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.37it/s]Capturing num tokens (num_tokens=960 avail_mem=118.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.37it/s] Capturing num tokens (num_tokens=896 avail_mem=118.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.37it/s]Capturing num tokens (num_tokens=832 avail_mem=118.00 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.37it/s]Capturing num tokens (num_tokens=768 avail_mem=118.00 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.37it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.00 GB):  43%|████▎     | 25/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=704 avail_mem=118.88 GB):  43%|████▎     | 25/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=640 avail_mem=118.88 GB):  43%|████▎     | 25/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.22it/s]

    Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 25.70it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 25.70it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 25.70it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 25.70it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 25.70it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 25.70it/s]

    Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.13it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.13it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.13it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.13it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.13it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.13it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.77it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.32it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.32it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.32it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.32it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.32it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.32it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 38.33it/s]

    Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 38.33it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 38.33it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:02<00:00, 39.99it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:02<00:00, 27.78it/s]


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
    Generated text:  Anny and I am a 3rd year Medical student at the University of Edinburgh. I have always been a fan of basketball and have always enjoyed staying up all night playing it. I hope to continue learning and growing as a player on the field and as a basketball coach at the university as I am excited to help support the growth of the team. My goal is to help my fellow student athletes become the best they can be. My father is also a professional basketball player and I am very much influenced by his approach and mentality. Basketball is a very fun and energetic sport that is loved by many, but also requires a lot of skill
    ===============================
    Prompt: The president of the United States is
    Generated text:  considering whether to exercise his veto power on a bill that requires a certain percentage of federal tax revenue to be collected through a payroll tax. The president has gathered data on the previous year's tax revenue for the Federal Government and the percentage of the revenue that would be collected through the payroll tax.
    
    a. If the President decides to veto the bill, what would be the effect on the aggregate demand for government goods and services and on the price level?
    
    b. What would be the effect on the aggregate supply of government goods and services and on the price level?
    
    c. If the President decides to approve the bill, what would be the effect on
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Many Parisians are women, and many of them are not married. The Parisian law states that women who are not married cannot be married in France. It has been a practice for Parisian women for many years.
    
    This law is not enforced effectively. The Parisian women have too many problems in their personal lives and in their economic lives. There are also a lot of cases of women being married to other women. All of this causes an enormous problem for the Parisian law.
    
    Finally, the Parisian law has been brought to a halt by the Supreme Court of Paris. It allows all women to marry. This change is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the realm of n-gons
    
    The last interview in the 2020-2021 academic year provided a thoughtful look at the world of AI and what it will be like in the next decade. It also provided a look at the tech companies that will be revolutionizing our lives with artificial intelligence. That’s where we are, in 2021, discussing the future of AI. Here’s what I learned.
    
    AI is a booming industry. Although there are always challenges, the technology is advancing faster than ever before. The biggest challenge has been the effort that has been put into the way that AI is


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, music, and fashion, and is home to many world-renowned museums and cultural institutions. Paris is a bustling metropolis with a rich history and diverse population, making it a popular tourist destination. The city is known for its French cuisine, including its famous croissants and its iconic French fries. Paris is also home to the French Parliament and the Eiffel Tower, which is considered one of the most iconic structures in the world. Overall, Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a greater emphasis on privacy and security, with more stringent regulations and controls in place to protect user data.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs, and it is likely to continue to play a larger role in this area in the future.
    
    4. Increased use of
    


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
    Generated text:  [Your Name] and I am a/an [Your Occupation/Title] with [Your Most Recent Experience/Role]. My [Your Most Significant Achievement] was [Your Achievement Summary], which was [Your Achievement Description]. If you could call me anything, it would be [Your Name], but I will always be [Your Character Name] and always strive to do [Your Character’s Best] in everything I do. I am always willing to learn and grow from my experiences, whether it be through travel, learning a new skill, or simply meeting new people. I love to explore and immerse myself in new cultures, learning about different
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is an international center of politics, culture, and business, and is known for its iconic landmarks such as the Eiffel Tower and Louvre Museum. Paris is also known for its vibrant and diverse city life, including its cafes, bistros, and nightclubs, as well as its French cuisine and wine. The city has a rich history dating back to the Roman Empire, and is home to numerous museums, galleries, and theaters. Paris is an attractive destination for tourists and locals alike, and is often referred to as the "City of a Thousand Whys" for its extensive library of books, articles, and articles
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends, including:
    
      1. Increased reliance on AI for natural language processing and chatbots: As AI becomes more sophisticated and able to understand and respond to natural language, it is likely to become even more ubiquitous in our daily lives. This includes chatbots and virtual assistants that can provide information, answer questions, and even make phone calls.
      2. Greater integration with other technologies: AI is already being integrated into many other technologies, such as drones, self-driving cars, and smart homes. As AI continues to advance, it is likely to be integrated more deeply into other technologies, such as


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

    Your

     Name

    ]

     and

     I

     am

     [

    Your

     Age

    ].

     I

     am

     a

     [

    Your

     Major

     or

     Occupation

    ].

     I

     am

     [

    Your

     Location

    ].

     I

     love

     [

    Your

     Passion

     or

     Hobby

    ].

     I

     like

     to

     [

    Your

     Hobby

    ].

     I

     have

     [

    Your

     Educational

     Level

    ],

     [

    Your

     Degree

    ],

     or

     [

    Your

     Degree

     Program

    ]

     and

     I

     have

     [

    Your

     Lif

    elong

     Interest

     or

     Hobby

    ].

     I

     have

     [

    Your

     Work

     Experience

    ],

     [

    Your

     Projects

    ],

     [

    Your

     Skills

    ],

     or

     [

    Your

     Professional

     Development

    ].

     I

     have

     [

    Your

     Leadership

     Skills

    ],

     [

    Your

     Team

    work

     Skills

    ],

     or

     [

    Your

     Business

     Skills

    ].

     I

     have

     a

     [

    Your

     Character

     Trait

     or

     Character

    ]

     and

     I

     am

     [

    Your

     Personality

    ].

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     seat

     of

     the

     French

     government

    ,

     as

     well

     as

     being

     the

     country

    's

     most

     popular

     tourist

     destination

    .

     The

     city

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     home

     to

     the

     French

     Parliament

    ,

     the

     national

     anthem

    ,

     and

     the

     city

    's

     vibrant

     fashion

     scene

    .

     Despite

     its

     size

    ,

     Paris

     is

     a

     culturally

     diverse

     city

     with

     a

     rich

     history

     and

     lively

     atmosphere

    .

     It

     has

     a

     population

     of

     over

     

    1

     million

     residents

     and

     is

     known

     for

     its

     sophisticated

     architecture

    ,

     high

    -end

     fashion

    ,

     and

     culinary

     scene

    .

     Its

     annual

     tourism

     industry

     accounts

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    ,

     as

     well

     as

     continued

     development

     in

     areas

     such

     as

     robotics

     and

     autonomous

     systems

    .

     It

     is

     also

     possible

     that

     AI

     will

     continue

     to

     be

     used

     in

     a

     variety

     of

     innovative

     ways

    ,

     such

     as

     in

     the

     development

     of

     new

     medical

     treatments

     and

     treatments

     for

     environmental

     issues

    ,

     and

     in

     the

     development

     of

     new

     technologies

     for

     communication

     and

     transportation

    .

     Additionally

    ,

     the

     integration

     of

     AI

     into

     human

     society

     may

     continue

     to

     grow

    ,

     with

     new

     applications

     being

     developed

     and

     existing

     ones

     being

     expanded

     in

     new

     and

     innovative

     ways

    .

     Finally

    ,

     the

     pace

     and

     scale

     of

     AI

     development

     will

     likely

     continue

     to

     increase

    ,

     as

     companies

    



```python
llm.shutdown()
```
