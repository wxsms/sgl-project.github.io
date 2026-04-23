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
    [2026-04-23 11:40:26] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.77it/s]


    2026-04-23 11:40:32,896 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 11:40:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:36,  3.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:36,  3.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:36,  3.79s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:56,  1.03s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:56,  1.03s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:03<00:56,  1.03s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<00:56,  1.03s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:21,  2.37it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:21,  2.37it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:21,  2.37it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:21,  2.37it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:11,  4.13it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.80it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.80it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.80it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.80it/s]

    Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.80it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:04,  8.80it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 13.88it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 13.88it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 13.88it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 13.88it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 13.88it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:02, 13.88it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 18.87it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 18.87it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 18.87it/s]

    Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 18.87it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 18.87it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 22.32it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 28.93it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 28.93it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 28.93it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 28.93it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 28.93it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:00, 28.93it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 32.04it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 32.04it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 32.04it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 32.04it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 35.70it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 35.70it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 35.70it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 35.70it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 35.70it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 35.70it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 38.43it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 38.43it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 38.43it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 38.43it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 38.43it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 38.43it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 39.85it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 39.85it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 39.85it/s] 

    Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 39.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.74 GB):   2%|▏         | 1/58 [00:00<00:09,  6.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.31 GB):   2%|▏         | 1/58 [00:00<00:09,  6.23it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.31 GB):   3%|▎         | 2/58 [00:00<00:09,  5.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.36 GB):   3%|▎         | 2/58 [00:00<00:09,  5.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.36 GB):   5%|▌         | 3/58 [00:00<00:08,  6.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.80 GB):   5%|▌         | 3/58 [00:00<00:08,  6.60it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.80 GB):   7%|▋         | 4/58 [00:00<00:07,  7.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.30 GB):   7%|▋         | 4/58 [00:00<00:07,  7.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.30 GB):   9%|▊         | 5/58 [00:00<00:06,  7.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.83 GB):   9%|▊         | 5/58 [00:00<00:06,  7.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.30 GB):   9%|▊         | 5/58 [00:00<00:06,  7.85it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=59.30 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.86 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.29 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.29 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.89 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.18it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.28 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.28 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.92 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.28 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.28 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.94 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.38it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=59.27 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.27 GB):  26%|██▌       | 15/58 [00:01<00:02, 14.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.97 GB):  26%|██▌       | 15/58 [00:01<00:02, 14.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.27 GB):  26%|██▌       | 15/58 [00:01<00:02, 14.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.27 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.06 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.13it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=59.02 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.25 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.03 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.24it/s]Capturing num tokens (num_tokens=960 avail_mem=59.25 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.24it/s] Capturing num tokens (num_tokens=960 avail_mem=59.25 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.20it/s]Capturing num tokens (num_tokens=896 avail_mem=59.24 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.20it/s]

    Capturing num tokens (num_tokens=832 avail_mem=59.09 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.20it/s]Capturing num tokens (num_tokens=768 avail_mem=59.23 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.20it/s]Capturing num tokens (num_tokens=768 avail_mem=59.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.62it/s]Capturing num tokens (num_tokens=704 avail_mem=59.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.62it/s]Capturing num tokens (num_tokens=640 avail_mem=59.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.62it/s]Capturing num tokens (num_tokens=576 avail_mem=59.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.62it/s]

    Capturing num tokens (num_tokens=576 avail_mem=59.22 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=512 avail_mem=59.20 GB):  48%|████▊     | 28/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=480 avail_mem=59.12 GB):  48%|████▊     | 28/58 [00:02<00:01, 21.15it/s]Capturing num tokens (num_tokens=448 avail_mem=59.21 GB):  48%|████▊     | 28/58 [00:02<00:01, 21.15it/s]Capturing num tokens (num_tokens=448 avail_mem=59.21 GB):  53%|█████▎    | 31/58 [00:02<00:01, 23.27it/s]Capturing num tokens (num_tokens=416 avail_mem=59.20 GB):  53%|█████▎    | 31/58 [00:02<00:01, 23.27it/s]Capturing num tokens (num_tokens=384 avail_mem=59.20 GB):  53%|█████▎    | 31/58 [00:02<00:01, 23.27it/s]Capturing num tokens (num_tokens=352 avail_mem=59.19 GB):  53%|█████▎    | 31/58 [00:02<00:01, 23.27it/s]

    Capturing num tokens (num_tokens=352 avail_mem=59.19 GB):  59%|█████▊    | 34/58 [00:02<00:00, 24.32it/s]Capturing num tokens (num_tokens=320 avail_mem=59.18 GB):  59%|█████▊    | 34/58 [00:02<00:00, 24.32it/s]Capturing num tokens (num_tokens=288 avail_mem=59.18 GB):  59%|█████▊    | 34/58 [00:02<00:00, 24.32it/s]Capturing num tokens (num_tokens=256 avail_mem=59.17 GB):  59%|█████▊    | 34/58 [00:02<00:00, 24.32it/s]Capturing num tokens (num_tokens=256 avail_mem=59.17 GB):  64%|██████▍   | 37/58 [00:02<00:00, 24.74it/s]Capturing num tokens (num_tokens=240 avail_mem=59.16 GB):  64%|██████▍   | 37/58 [00:02<00:00, 24.74it/s]Capturing num tokens (num_tokens=224 avail_mem=59.16 GB):  64%|██████▍   | 37/58 [00:02<00:00, 24.74it/s]Capturing num tokens (num_tokens=208 avail_mem=59.15 GB):  64%|██████▍   | 37/58 [00:02<00:00, 24.74it/s]

    Capturing num tokens (num_tokens=192 avail_mem=59.14 GB):  64%|██████▍   | 37/58 [00:02<00:00, 24.74it/s]Capturing num tokens (num_tokens=192 avail_mem=59.14 GB):  71%|███████   | 41/58 [00:02<00:00, 26.51it/s]Capturing num tokens (num_tokens=176 avail_mem=59.14 GB):  71%|███████   | 41/58 [00:02<00:00, 26.51it/s]Capturing num tokens (num_tokens=160 avail_mem=59.13 GB):  71%|███████   | 41/58 [00:02<00:00, 26.51it/s]Capturing num tokens (num_tokens=144 avail_mem=59.12 GB):  71%|███████   | 41/58 [00:02<00:00, 26.51it/s]Capturing num tokens (num_tokens=144 avail_mem=59.12 GB):  76%|███████▌  | 44/58 [00:02<00:00, 27.25it/s]Capturing num tokens (num_tokens=128 avail_mem=59.12 GB):  76%|███████▌  | 44/58 [00:02<00:00, 27.25it/s]Capturing num tokens (num_tokens=112 avail_mem=59.11 GB):  76%|███████▌  | 44/58 [00:02<00:00, 27.25it/s]Capturing num tokens (num_tokens=96 avail_mem=59.10 GB):  76%|███████▌  | 44/58 [00:02<00:00, 27.25it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=59.10 GB):  76%|███████▌  | 44/58 [00:02<00:00, 27.25it/s]Capturing num tokens (num_tokens=80 avail_mem=59.10 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.83it/s]Capturing num tokens (num_tokens=64 avail_mem=59.10 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.83it/s]Capturing num tokens (num_tokens=48 avail_mem=59.10 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.83it/s]Capturing num tokens (num_tokens=32 avail_mem=59.09 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.83it/s]Capturing num tokens (num_tokens=28 avail_mem=59.08 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.83it/s]Capturing num tokens (num_tokens=28 avail_mem=59.08 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.14it/s]Capturing num tokens (num_tokens=24 avail_mem=59.07 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.14it/s]Capturing num tokens (num_tokens=20 avail_mem=59.07 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.14it/s]

    Capturing num tokens (num_tokens=16 avail_mem=59.06 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.14it/s]Capturing num tokens (num_tokens=12 avail_mem=59.05 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.14it/s]Capturing num tokens (num_tokens=12 avail_mem=59.05 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.42it/s]Capturing num tokens (num_tokens=8 avail_mem=59.03 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.42it/s] Capturing num tokens (num_tokens=4 avail_mem=59.02 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.42it/s]Capturing num tokens (num_tokens=4 avail_mem=59.02 GB): 100%|██████████| 58/58 [00:02<00:00, 19.58it/s]


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
    Generated text:  Roshan and I was born in 1960 in Surat, Gujarat, India. I have an MA in English literature from the University of California at San Diego, USA, and a PhD in Creative Writing from the University of Pennsylvania. I am the Founder, Editor-in-chief and Author of the award-winning quarterly literary magazine, MINT (Mumbai International Novel). I am also the Founder of the award-winning literary magazine, the foreign language magazine, Metamorphoses and the award-winning film and book festival, Film It. I founded the literary journal, Art and the City, the foreign language journal, Swar
    ===============================
    Prompt: The president of the United States is
    Generated text:  a cabinet member of the executive branch. The president, like other Cabinet members, has significant authority over policy making, but also has considerable influence over policy implementation. The presidency also sits at the apex of the executive branch, making decisions that affect all branches of government, and its influence has moved to include a significant role in national security.
    An important characteristic of the presidency is that it is a ceremonial position, appointed by the head of the federal government and elected for a specific term. Under the Constitution, the president is not bound by the rules of the Senate or the Supreme Court, but all presidential appointments are made by the president. Under federal
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Spain is Madrid. The capital of Italy is Rome.
    
    Q: Is there a capital of the United States?
    
    Pick from:
     - no
     - yes
    no
    The United States does not have a capital city. The states of the United States are the states of the previous American colonies. They have their own capital cities. Washington, D.C. is the capital of the United States. The capital of Spain is Madrid, not the capital of the United States. The capital of France is Paris, not the capital of the United States. The capital of Italy is Rome, not the capital of the United States.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the user. Find out how it can influence and benefit your work. From personalized content and customer support to AI-driven training, machine learning algorithms and predictive analytics can be used to dramatically improve productivity and efficiency. From enhancing efficiency in business processes to transforming customer service and marketing, AI can help businesses become more agile and competitive.
    In the world of business, the adoption of AI has become an integral part of how companies operate. AI technology, powered by machine learning and deep learning algorithms, can help businesses make informed decisions, automate tasks, and improve overall efficiency. From personalizing content and customer support to predictive analytics and machine learning


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [reason for being at the company]. I'm always looking for ways to [what I enjoy doing at the company]. I'm [how I like to be treated]. I'm [what I'm looking for in a partner]. I'm [what I'm looking for in a team]. I'm [what I'm looking for in a friend]. I'm [what I'm looking for in a mentor]. I'm [what I'm looking for in a mentor]. I'm [what I'm looking for in a mentor]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and a major economic center in Europe. It is home to many famous museums, including the Louvre and the Musée d'Orsay. The city is also known for its food
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This integration could lead to more sophisticated and adaptive AI systems that can learn from and adapt to new situations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and
    


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
    Generated text:  [insert fictional character's name]. I am a [insert fictional character's profession or occupation]. I enjoy [insert fictional character's hobby or interest], and I am constantly learning and growing in my field. I'm passionate about [insert fictional character's personal value or trait], and I strive to create positive impact in the world. I am a [insert fictional character's age], [insert fictional character's gender], [insert fictional character's nationality], [insert fictional character's ethnicity], and [insert fictional character's accent]. I come from a [insert fictional character's hometown, city, or country]. I am grateful for [insert fictional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city and the seat of government of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a significant cultural center and home to numerous museums, theaters, and art galleries, and is a major tourist destination. Its rich history, art, and cuisine has made it a popular tourist destination in France. Paris is often referred to as the "City of Light" and is home to many of France's cultural institutions and landmarks. The French capital, Paris, is a popular tourist destination known for its iconic landmarks, art museums
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be driven by several trends that could shape the development of this field in the coming decades:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more accurate and efficient, it may become more widely used in medical imaging, drug discovery, and personalized medicine.
    
    2. Improved transparency and explainability of AI models: As AI models become more sophisticated, there may be a need for more transparency and explainability in their decision-making processes. This could lead to more widespread adoption of AI in fields such as law enforcement and criminal justice.
    
    3. Increased focus on


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

     [

    age

    ]

     years

     old

    .

     I

     have

     always

     had

     a

     strong

     sense

     of

     curiosity

     and

     a

     love

     for

     learning

    .

     I

     am

     an

     avid

     reader

     and

     have

     a

     knack

     for

     understanding

     the

     world

     around

     me

     through

     observation

    .

     I

     enjoy

     spending time

     with

     my

     family

     and

     friends

    ,

     but

     I

     also

     enjoy

     pursuing

     new

     experiences

     and

     taking

     risks

    .

     I

     am

     always

     up

     for

     a

     challenge

     and

     love

     to

     learn

     something

     new

     every

     day

    .

     In

     my

     free

     time

    ,

     I

     like

     to

     enjoy

     a

     good

     book

    ,

     travel

    ,

     and

     engage

     in

     art

    .

     I

    'm

     looking

     forward

     to

     meeting

     you

    !

     G

    reetings

    ,

     how

     can

     I

     assist

     you

     today

    ?

     How

     can

     I

     assist

     you

     today

    ?

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Detailed

     instructions

    :
    


    1

    .

     Select

     the

     specific

     French

     city

     as

     the

     target

     of

     your

     statement

    .


    2

    .

     Highlight

     the

     capital

     city

    's

     most

     notable

     architectural

     features

     or

     landmarks

     in

     your

     sentence

    .


    3

    .

     Craft

     a

     comparative

     statement

     comparing

     Paris

     to

     other

     French

     cities

     or

     neighboring

     countries

     using

     the

     target

     city

    's

     location

     relative

     to

     other

     French

     cities

     and

     its

     proximity

     to

     other

     French

     regions

    .
    


    Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     and

     culture

    .

     Its

     skyline

     is

     filled

     with

     towering

     skys

    crap

    ers

     and

     buildings

     that

     reflect

     the

     city

    's

     industrial

     heritage

    .

     The

     city

     boasts

     a

     diverse

     range

     of

     architectural

     styles

     and

     is

     home

     to

     several

     world

    -ren

    owned

     museums

    ,

     art

     galleries

    ,

     and

     historical

     sites

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     and

     possibilities

    .

     Here

     are

     a

     few

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     automation

    :

     The

     rise

     of

     automation

     will

     allow

     machines

     to

     perform

     tasks

     that

     are

     currently

     done

     by

     humans

    ,

     such

     as

     data

     analysis

    ,

     coding

    ,

     and

     equipment

     maintenance

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     will

     continue

     to

     improve

     at

     its

     current

     rate

    ,

     leading

     to

     more

     accurate

     diagnoses

     and

     treatment

     plans

     for

     patients

    .

     AI

     will

     also

     become

     more

     personalized

    ,

     allowing

     healthcare

     providers

     to

     create

     individual

    ized

     treatment

     plans

     for

     each

     patient

    .
    


    3

    .

     AI

     in

     finance

    :

     AI

     will

     continue

     to

     revolution

    ize

     finance

    ,

     with

     the

     ability

     to

     analyze

     large

     amounts

     of

     data

     and

     make

     predictions

     about

     future

     trends

    .

     This

     could

    



```python
llm.shutdown()
```
