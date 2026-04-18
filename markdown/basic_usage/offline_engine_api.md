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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 03:06:52] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.50it/s]


    2026-04-18 03:06:57,984 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-18 03:06:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.76it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.76it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.76it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.76it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.76it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.76it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.76it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.76it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.76it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.76it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.90it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.01 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.85 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=117.55 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.55 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.54 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.75it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=117.54 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.54 GB):  22%|██▏       | 13/58 [00:00<00:04, 10.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.54 GB):  22%|██▏       | 13/58 [00:00<00:04, 10.81it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=117.53 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.53 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.53 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.75it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=117.53 GB):  26%|██▌       | 15/58 [00:01<00:04,  9.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.53 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.52 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.65it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=117.52 GB):  29%|██▉       | 17/58 [00:01<00:04,  9.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.52 GB):  33%|███▎      | 19/58 [00:01<00:04,  9.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.52 GB):  33%|███▎      | 19/58 [00:01<00:04,  9.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.49 GB):  33%|███▎      | 19/58 [00:01<00:04,  9.63it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=117.49 GB):  36%|███▌      | 21/58 [00:01<00:03,  9.98it/s]Capturing num tokens (num_tokens=960 avail_mem=117.51 GB):  36%|███▌      | 21/58 [00:01<00:03,  9.98it/s] Capturing num tokens (num_tokens=896 avail_mem=117.51 GB):  36%|███▌      | 21/58 [00:01<00:03,  9.98it/s]Capturing num tokens (num_tokens=896 avail_mem=117.51 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.23it/s]Capturing num tokens (num_tokens=832 avail_mem=117.50 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.23it/s]

    Capturing num tokens (num_tokens=768 avail_mem=117.50 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.23it/s]

    Capturing num tokens (num_tokens=768 avail_mem=117.50 GB):  43%|████▎     | 25/58 [00:02<00:03,  8.65it/s]Capturing num tokens (num_tokens=704 avail_mem=117.49 GB):  43%|████▎     | 25/58 [00:02<00:03,  8.65it/s]Capturing num tokens (num_tokens=640 avail_mem=117.49 GB):  43%|████▎     | 25/58 [00:02<00:03,  8.65it/s]Capturing num tokens (num_tokens=640 avail_mem=117.49 GB):  47%|████▋     | 27/58 [00:02<00:03,  9.30it/s]Capturing num tokens (num_tokens=576 avail_mem=117.49 GB):  47%|████▋     | 27/58 [00:02<00:03,  9.30it/s]

    Capturing num tokens (num_tokens=512 avail_mem=117.48 GB):  47%|████▋     | 27/58 [00:02<00:03,  9.30it/s]Capturing num tokens (num_tokens=512 avail_mem=117.48 GB):  50%|█████     | 29/58 [00:02<00:02, 10.14it/s]Capturing num tokens (num_tokens=480 avail_mem=117.49 GB):  50%|█████     | 29/58 [00:02<00:02, 10.14it/s]Capturing num tokens (num_tokens=448 avail_mem=117.49 GB):  50%|█████     | 29/58 [00:02<00:02, 10.14it/s]

    Capturing num tokens (num_tokens=448 avail_mem=117.49 GB):  53%|█████▎    | 31/58 [00:02<00:02, 10.62it/s]Capturing num tokens (num_tokens=416 avail_mem=117.49 GB):  53%|█████▎    | 31/58 [00:02<00:02, 10.62it/s]Capturing num tokens (num_tokens=384 avail_mem=117.46 GB):  53%|█████▎    | 31/58 [00:02<00:02, 10.62it/s]

    Capturing num tokens (num_tokens=384 avail_mem=117.46 GB):  57%|█████▋    | 33/58 [00:03<00:02, 10.43it/s]Capturing num tokens (num_tokens=352 avail_mem=117.20 GB):  57%|█████▋    | 33/58 [00:03<00:02, 10.43it/s]Capturing num tokens (num_tokens=320 avail_mem=117.44 GB):  57%|█████▋    | 33/58 [00:03<00:02, 10.43it/s]

    Capturing num tokens (num_tokens=320 avail_mem=117.44 GB):  60%|██████    | 35/58 [00:03<00:02, 10.10it/s]Capturing num tokens (num_tokens=288 avail_mem=117.44 GB):  60%|██████    | 35/58 [00:03<00:02, 10.10it/s]Capturing num tokens (num_tokens=256 avail_mem=117.44 GB):  60%|██████    | 35/58 [00:03<00:02, 10.10it/s]

    Capturing num tokens (num_tokens=256 avail_mem=117.44 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.04it/s]Capturing num tokens (num_tokens=240 avail_mem=117.44 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.04it/s]Capturing num tokens (num_tokens=224 avail_mem=117.24 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.04it/s]Capturing num tokens (num_tokens=224 avail_mem=117.24 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.49it/s]Capturing num tokens (num_tokens=208 avail_mem=117.43 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.49it/s]

    Capturing num tokens (num_tokens=192 avail_mem=117.42 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.49it/s]Capturing num tokens (num_tokens=192 avail_mem=117.42 GB):  71%|███████   | 41/58 [00:03<00:01, 10.71it/s]Capturing num tokens (num_tokens=176 avail_mem=117.42 GB):  71%|███████   | 41/58 [00:03<00:01, 10.71it/s]

    Capturing num tokens (num_tokens=160 avail_mem=117.41 GB):  71%|███████   | 41/58 [00:03<00:01, 10.71it/s]Capturing num tokens (num_tokens=160 avail_mem=117.41 GB):  74%|███████▍  | 43/58 [00:04<00:01,  9.26it/s]Capturing num tokens (num_tokens=144 avail_mem=117.28 GB):  74%|███████▍  | 43/58 [00:04<00:01,  9.26it/s]

    Capturing num tokens (num_tokens=128 avail_mem=117.28 GB):  74%|███████▍  | 43/58 [00:04<00:01,  9.26it/s]Capturing num tokens (num_tokens=128 avail_mem=117.28 GB):  78%|███████▊  | 45/58 [00:04<00:01,  8.35it/s]Capturing num tokens (num_tokens=112 avail_mem=117.28 GB):  78%|███████▊  | 45/58 [00:04<00:01,  8.35it/s]Capturing num tokens (num_tokens=96 avail_mem=117.38 GB):  78%|███████▊  | 45/58 [00:04<00:01,  8.35it/s] Capturing num tokens (num_tokens=80 avail_mem=117.38 GB):  78%|███████▊  | 45/58 [00:04<00:01,  8.35it/s]Capturing num tokens (num_tokens=80 avail_mem=117.38 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.24it/s]Capturing num tokens (num_tokens=64 avail_mem=117.37 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.24it/s]

    Capturing num tokens (num_tokens=48 avail_mem=117.36 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.24it/s]Capturing num tokens (num_tokens=32 avail_mem=117.35 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.24it/s]Capturing num tokens (num_tokens=32 avail_mem=117.35 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.29it/s]Capturing num tokens (num_tokens=28 avail_mem=117.35 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.29it/s]Capturing num tokens (num_tokens=24 avail_mem=117.34 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.29it/s]Capturing num tokens (num_tokens=20 avail_mem=117.33 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.29it/s]Capturing num tokens (num_tokens=16 avail_mem=117.31 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.29it/s]Capturing num tokens (num_tokens=16 avail_mem=117.31 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.16it/s]Capturing num tokens (num_tokens=12 avail_mem=117.32 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.16it/s]Capturing num tokens (num_tokens=8 avail_mem=117.32 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.16it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=117.31 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.16it/s]Capturing num tokens (num_tokens=4 avail_mem=117.31 GB): 100%|██████████| 58/58 [00:04<00:00, 12.07it/s]


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
    Generated text:  James. I am a 10th grader at our school. I have a big family. My mom is a doctor, my dad is a teacher, and my older brother is a student. My parents are very strict with me. They are very strict with me because my brother is very naughty. I like to play on the computer when I am not at home. I think it is fun to play games on the computer. Sometimes I play computer games with my brother. We play for an hour and a half each day. I also love ice cream. When I have a nice weather, I like to take my brother to
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He has to deal with many important things. Now, let's take a look at the President's schedule and see what he does each day. Let's break down the president's schedule into small parts and describe each part:
    
    1. He starts his day with a quick breakfast, which can be a few minutes long. This is a fast start for the day, so he won't need to eat much.
    
    2. He wakes up at 7:00 AM, which is when the first news of the day comes in. The president looks at the news and takes a few minutes to decide what to do for the
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Madrid
    D. Rome
    Answer:
    A
    
    A 55-year-old male patient suddenly experienced severe abdominal pain, cold sweats, and rapid breathing after being caught in a high-speed car accident. He was admitted to the hospital in an emergency. Physical examination: pulse rate 120 beats/min, blood pressure 12/8 kPa, pale face, cold extremities, decreased bowel sounds. The most likely diagnosis for this patient is ____.
    A. Liver rupture
    B. Pancreatic injury
    C. Duodenal injury
    D. Ga
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but a lot of that is still murky
    
    By Enderly 
    
      *Originally published in The Information, 2022-11-03
    
      *Updated on November 20, 2022
    
      *Based on data provided by iResearch and other sources
    
      *This piece was updated to include additional references and a discussion of diverse AI approaches, an important issue that was not covered in the original article.
    
      *This article is not intended to be a comprehensive review of all aspects of AI. The views expressed are based on the author's personal experience, which may not cover


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" or "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also known for its rich history, art, and culture, and is a major tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is also home to many international organizations and institutions, including the French Academy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, privacy, and transparency. AI developers will need to be more mindful of how their technology is used and ensure that it is used in a way that is fair and responsible.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As
    


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
    Generated text:  __________ and I'm a/an _________. What can you tell me about yourself? You're welcome, it's a pleasure to meet you. What is your profession or occupation? I'm a/an ____________. What is your greatest accomplishment so far? I have achieved ___________.
    You're welcome, my dear, and I'm grateful for your interest in me. How can I assist you further? I'm here to listen, and I'll do my best to provide you with information about yourself and your profession. Is there anything specific I can help you with? I'm always here to assist, so please don't
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Sentence A: "Paris is the capital city of France."
    Sentence B: "Paris is the capital of France."
    Sentence A and B are equivalent statements.
    
    Is this a valid response?
    
    a) No
    b) Yes
    b) Yes
    
    Both sentences A and B are equivalent statements. Sentence A accurately conveys that Paris is indeed the capital city of France. Sentence B is also a correct statement that accurately describes Paris as the capital city of France. The statement is straightforward, clear, and does not contain any grammatical or stylistic errors. Therefore, the correct answer is yes, the response is valid. However,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by increased automation, intelligence, and integration with other technologies. Some possible future trends include:
    
    1. Increased automation: AI is becoming increasingly capable of performing tasks that are currently performed by humans. This will lead to increased automation in industries such as manufacturing, transportation, and healthcare.
    
    2. Intelligence: AI will continue to become more intelligent and capable of learning from experience. This will lead to the development of more advanced and sophisticated AI systems that can perform tasks that were previously thought to be beyond the capabilities of humans.
    
    3. Integration with other technologies: AI will continue to integrate with other technologies such as big data, machine learning


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

    ],

     and

     I

    'm

     a

     professional

     copy

    writer

    ,

     writer

    ,

     and

     content

     creator

    .

     I

     have

     a

     knack

     for

     crafting

     compelling

     content

     that

     can

     help

     businesses

     boost

     their

     marketing

     efforts

    .

     My

     work

     ranges

     from

     SEO

     to

     social

     media

     marketing

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     create

     impactful

     content

     that

     reson

    ates

     with

     my

     clients

    .

     Ready

     to

     take

     on

     the

     world

     with

     you

    ?

     Let

    's

     collaborate

     to

     make

     your

     business

     a

     success

    !

     

    🌟

    💼

    💼

     #

    Content

    M

    akers

     #

    Copy

    writers

     #

    Content

    Creator

     #

    Marketing

    Pro

     #

    S

    ME

    s

     #

    Creat

    ivity

     

    🌟

    💼

    💼

     #

    Self

    Intro

    
    


    Hey

     there

    ,

    !

     

    🙌

    Hey

     there

    ,

    !

    
    
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

     the

     European

     Union

     and

     the

     second

     largest

     in

     the

     world

     by

     population

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     access

     to

     France

    's

     natural

     beauty

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     the

     official

     language

     of

     France

     and

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     the

     Lou

    vre

     and

     the

     Ch

    amps

    -E

    lys

    ées

    .

     It

     is

     also

     the

     birth

    place

     of

     many

     French

     artists

    ,

     writers

    ,

     and

     musicians

    .

     France

    's

     capital

     city

     is

     a

     major

     hub

     for

     business

    ,

     politics

    ,

     and

     culture

    ,

     and

     attracts

     millions

     of

     visitors

     each

     year

    .

     Paris

     is

     a

    
    
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

    ,

     including

    :
    


    1

    .

     Increased

     use

     of

     AI

     in healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     even

     greater

     use

     of

     AI

     in

     this

     area

    .
    


    2

    .

     Greater

     reliance

     on

     AI

     in

     manufacturing

    :

     AI

     is

     already

     being

     used

     to

     optimize

     manufacturing

     processes

     and

     improve

     the

     quality

     of

     products

    .

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     even

     greater

     use

     of

     AI

     in

     manufacturing

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     to

     identify

     fraud

     and

     improve

     risk

     management

    .

     As

     technology

     continues

     to

    



```python
llm.shutdown()
```
