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
    [2026-04-17 05:07:30] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.06it/s]


    2026-04-17 05:07:35,740 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 05:07:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:43,  2.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:43,  2.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:43,  2.88s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:43,  1.26it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:43,  1.26it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:03<00:43,  1.26it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:03<00:43,  1.26it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:17,  3.03it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:17,  3.03it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:17,  3.03it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:17,  3.03it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:17,  3.03it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:03<00:17,  3.03it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]

    Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:06,  6.78it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 13.24it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 13.24it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]

    Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 38.51it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 44.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.68 GB):   3%|▎         | 2/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.48 GB):   3%|▎         | 2/58 [00:00<00:04, 11.47it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.48 GB):   3%|▎         | 2/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.48 GB):   7%|▋         | 4/58 [00:00<00:04, 12.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.60 GB):   7%|▋         | 4/58 [00:00<00:04, 12.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.67 GB):   7%|▋         | 4/58 [00:00<00:04, 12.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.67 GB):  10%|█         | 6/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.67 GB):  10%|█         | 6/58 [00:00<00:03, 14.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.66 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.65 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.64 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.54 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.06it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=118.54 GB):  21%|██        | 12/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.54 GB):  21%|██        | 12/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.57 GB):  21%|██        | 12/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.56 GB):  21%|██        | 12/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.60 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.58 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.25it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=118.58 GB):  31%|███       | 18/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.43 GB):  31%|███       | 18/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.42 GB):  31%|███       | 18/58 [00:01<00:01, 22.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.40 GB):  31%|███       | 18/58 [00:01<00:01, 22.73it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=117.40 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.42it/s]Capturing num tokens (num_tokens=960 avail_mem=117.41 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.42it/s] Capturing num tokens (num_tokens=896 avail_mem=117.41 GB):  36%|███▌      | 21/58 [00:01<00:02, 16.42it/s]Capturing num tokens (num_tokens=896 avail_mem=117.41 GB):  40%|███▉      | 23/58 [00:01<00:02, 14.56it/s]Capturing num tokens (num_tokens=832 avail_mem=117.40 GB):  40%|███▉      | 23/58 [00:01<00:02, 14.56it/s]

    Capturing num tokens (num_tokens=768 avail_mem=117.38 GB):  40%|███▉      | 23/58 [00:01<00:02, 14.56it/s]Capturing num tokens (num_tokens=768 avail_mem=117.38 GB):  43%|████▎     | 25/58 [00:01<00:02, 13.54it/s]Capturing num tokens (num_tokens=704 avail_mem=117.40 GB):  43%|████▎     | 25/58 [00:01<00:02, 13.54it/s]Capturing num tokens (num_tokens=640 avail_mem=117.39 GB):  43%|████▎     | 25/58 [00:01<00:02, 13.54it/s]

    Capturing num tokens (num_tokens=640 avail_mem=117.39 GB):  47%|████▋     | 27/58 [00:01<00:02, 12.71it/s]Capturing num tokens (num_tokens=576 avail_mem=117.39 GB):  47%|████▋     | 27/58 [00:01<00:02, 12.71it/s]Capturing num tokens (num_tokens=512 avail_mem=117.37 GB):  47%|████▋     | 27/58 [00:01<00:02, 12.71it/s]

    Capturing num tokens (num_tokens=512 avail_mem=117.37 GB):  50%|█████     | 29/58 [00:02<00:03,  9.03it/s]Capturing num tokens (num_tokens=480 avail_mem=117.38 GB):  50%|█████     | 29/58 [00:02<00:03,  9.03it/s]

    Capturing num tokens (num_tokens=448 avail_mem=117.38 GB):  50%|█████     | 29/58 [00:02<00:03,  9.03it/s]Capturing num tokens (num_tokens=448 avail_mem=117.38 GB):  53%|█████▎    | 31/58 [00:02<00:03,  8.16it/s]Capturing num tokens (num_tokens=416 avail_mem=117.37 GB):  53%|█████▎    | 31/58 [00:02<00:03,  8.16it/s]Capturing num tokens (num_tokens=384 avail_mem=117.37 GB):  53%|█████▎    | 31/58 [00:02<00:03,  8.16it/s]

    Capturing num tokens (num_tokens=384 avail_mem=117.37 GB):  57%|█████▋    | 33/58 [00:02<00:02,  8.82it/s]Capturing num tokens (num_tokens=352 avail_mem=117.36 GB):  57%|█████▋    | 33/58 [00:02<00:02,  8.82it/s]Capturing num tokens (num_tokens=320 avail_mem=117.35 GB):  57%|█████▋    | 33/58 [00:02<00:02,  8.82it/s]Capturing num tokens (num_tokens=320 avail_mem=117.35 GB):  60%|██████    | 35/58 [00:02<00:02,  9.31it/s]Capturing num tokens (num_tokens=288 avail_mem=117.35 GB):  60%|██████    | 35/58 [00:02<00:02,  9.31it/s]

    Capturing num tokens (num_tokens=256 avail_mem=117.34 GB):  60%|██████    | 35/58 [00:02<00:02,  9.31it/s]Capturing num tokens (num_tokens=256 avail_mem=117.34 GB):  64%|██████▍   | 37/58 [00:02<00:02, 10.08it/s]Capturing num tokens (num_tokens=240 avail_mem=117.34 GB):  64%|██████▍   | 37/58 [00:02<00:02, 10.08it/s]Capturing num tokens (num_tokens=224 avail_mem=117.33 GB):  64%|██████▍   | 37/58 [00:03<00:02, 10.08it/s]

    Capturing num tokens (num_tokens=224 avail_mem=117.33 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.57it/s]Capturing num tokens (num_tokens=208 avail_mem=117.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.57it/s]Capturing num tokens (num_tokens=192 avail_mem=117.32 GB):  67%|██████▋   | 39/58 [00:03<00:01, 10.57it/s]Capturing num tokens (num_tokens=192 avail_mem=117.32 GB):  71%|███████   | 41/58 [00:03<00:01, 10.96it/s]Capturing num tokens (num_tokens=176 avail_mem=117.31 GB):  71%|███████   | 41/58 [00:03<00:01, 10.96it/s]

    Capturing num tokens (num_tokens=160 avail_mem=117.30 GB):  71%|███████   | 41/58 [00:03<00:01, 10.96it/s]Capturing num tokens (num_tokens=160 avail_mem=117.30 GB):  74%|███████▍  | 43/58 [00:03<00:01, 11.60it/s]Capturing num tokens (num_tokens=144 avail_mem=117.30 GB):  74%|███████▍  | 43/58 [00:03<00:01, 11.60it/s]Capturing num tokens (num_tokens=128 avail_mem=117.29 GB):  74%|███████▍  | 43/58 [00:03<00:01, 11.60it/s]

    Capturing num tokens (num_tokens=128 avail_mem=117.29 GB):  78%|███████▊  | 45/58 [00:03<00:01, 11.97it/s]Capturing num tokens (num_tokens=112 avail_mem=117.29 GB):  78%|███████▊  | 45/58 [00:03<00:01, 11.97it/s]Capturing num tokens (num_tokens=96 avail_mem=117.28 GB):  78%|███████▊  | 45/58 [00:03<00:01, 11.97it/s] Capturing num tokens (num_tokens=96 avail_mem=117.28 GB):  81%|████████  | 47/58 [00:03<00:00, 13.09it/s]Capturing num tokens (num_tokens=80 avail_mem=117.27 GB):  81%|████████  | 47/58 [00:03<00:00, 13.09it/s]Capturing num tokens (num_tokens=64 avail_mem=117.27 GB):  81%|████████  | 47/58 [00:03<00:00, 13.09it/s]

    Capturing num tokens (num_tokens=64 avail_mem=117.27 GB):  84%|████████▍ | 49/58 [00:03<00:00, 13.75it/s]Capturing num tokens (num_tokens=48 avail_mem=117.26 GB):  84%|████████▍ | 49/58 [00:03<00:00, 13.75it/s]Capturing num tokens (num_tokens=32 avail_mem=117.25 GB):  84%|████████▍ | 49/58 [00:03<00:00, 13.75it/s]Capturing num tokens (num_tokens=32 avail_mem=117.25 GB):  88%|████████▊ | 51/58 [00:03<00:00, 14.65it/s]Capturing num tokens (num_tokens=28 avail_mem=117.24 GB):  88%|████████▊ | 51/58 [00:03<00:00, 14.65it/s]Capturing num tokens (num_tokens=24 avail_mem=117.24 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.65it/s]Capturing num tokens (num_tokens=20 avail_mem=117.23 GB):  88%|████████▊ | 51/58 [00:04<00:00, 14.65it/s]

    Capturing num tokens (num_tokens=20 avail_mem=117.23 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.30it/s]Capturing num tokens (num_tokens=16 avail_mem=117.23 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.30it/s]Capturing num tokens (num_tokens=12 avail_mem=117.22 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.30it/s]Capturing num tokens (num_tokens=8 avail_mem=117.22 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.30it/s] Capturing num tokens (num_tokens=4 avail_mem=117.21 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.30it/s]Capturing num tokens (num_tokens=4 avail_mem=117.21 GB): 100%|██████████| 58/58 [00:04<00:00, 21.71it/s]Capturing num tokens (num_tokens=4 avail_mem=117.21 GB): 100%|██████████| 58/58 [00:04<00:00, 13.70it/s]


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
    Generated text:  Alexander, I'm 26 years old. I'm a professional financial planner and I work as an investment advisor for the Philadelphia Stock Exchange, where I work as a member of the Investment Committee. I also work as an advisor to a variety of clients, including individuals, families, and corporations, and have a passion for learning about the financial markets and the innovative financial products being developed.
    As an investment advisor for the Philadelphia Stock Exchange, I help clients optimize their portfolios and strategies to maximize their returns. My job involves working closely with traders and other financial professionals to develop and execute investment strategies that align with the client's goals and risk tolerance
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ________.
    A. government official
    B. person in charge of the country
    C. person who governs the country
    D. political leader
    Answer: A
    
    When an elevator is operating, the door panel of the car is driven by the elevator motor, which is controlled by the ___.
    A. Passenger Information System
    B. Cabin Operating Panel
    C. Control Panel
    D. Signal System
    Answer: B
    
    For the product launch of a new brand, the following activities are carried out: ① A sales promotion campaign is launched to promote the product ② An internal advertising campaign is launched to promote
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The cost of living in Paris is 2.2 times that of London. If you want to visit a nearby city with a similar cost of living but lower cost of living in London, you should travel to which city? Choose the answer between "England" and "Scotland".
    To determine which city you should visit to find a lower cost of living while still being near Paris, let's start by calculating the cost of living in London and then find the closest city to Paris with a similar cost of living.
    
    1. **Cost of Living in London:**
       - Generally, London costs around $30,000
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of programmers, not the data. (Read the article's summary and provide your answer.) The article discusses the potential of AI to drive significant advancements in various fields, and the importance of programmers in this development. The author suggests that programmers should be encouraged to explore and harness the power of AI, rather than shying away from the opportunities it presents.
    
    The article also highlights the potential negative consequences of AI development, such as job displacement, privacy concerns, and the potential for bias in algorithms. However, the author argues that these risks are outweighed by the benefits of AI, and that programmers can mitigate these risks by ensuring that


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and is home to many famous landmarks and cultural institutions. It is a popular tourist destination and a major economic center in Europe. The city is known for its cuisine, fashion, and art, and is home to many international organizations and cultural institutions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the creation of new jobs, but it will also create new opportunities for people to work in areas such as data analysis, software development, and robotics.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a growing concern about the ethical implications
    


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
    Generated text:  [insert name here], and I'm a [insert occupation here], with [insert relevant experience here]. I'm a skilled problem solver, and my ability to think creatively and solve complex issues is unmatched. I have a natural affinity for problems, and I enjoy using my analytical mind to find solutions. I'm always looking for new challenges and opportunities to grow and learn, and I strive to be a valuable asset to the team. I'm a team player and enjoy collaborating with others, and I'm always looking for ways to improve my skills and knowledge in my field. I believe in the importance of lifelong learning and am always eager to expand
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
    
    The capital of France is Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. 
    
    This statement provides a brief summary of the capital city's location, notable landmarks, and cultural attractions within France. While not exhaustive, it encapsulates the primary focus of Paris as the capital of France. 
    
    To elaborate, Paris, officially known as the "Metropolis" or "Pays-De-France" in French,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and has the potential to transform a wide range of industries and applications. Here are some potential trends that could shape AI in the coming years:
    
    1. Increased AI integration with human behavior: As AI becomes more integrated with human behavior, we may see more opportunities for AI to learn from and adapt to human behavior. This could result in more natural and intuitive AI that can interact with humans in more meaningful ways.
    
    2. Development of more powerful AI: AI developers are working to develop more powerful and flexible AI systems that can learn from a wide range of data and make better-informed decisions. This could lead to more efficient and effective AI


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

    'm

     a

     skilled

     [

    specific

     skill

     or

     specialty

    ],

     and

     I

    'm

     eager

     to

     learn

     more

     about

     you

     and

     your

     journey

    .
    


    How

     can

     I

     assist

     you

     today

    ?

     


    [

    Name

    ]

     is

     a

     highly

     skilled

     [

    specific

     skill

     or

     specialty

    ],

     passionate

     about

     [

    add

     a

     brief

     explanation

     of

     the

     character

    's

     interest

     or

     passion

    ].

     I

     look

     forward

     to

     hearing

     about

     your

     journey

     and

     what

     challenges

     you

    're

     facing

    .

     How

     can

     I

     assist

     you

     today

    ?
    


    What

     do

     you

     love

     about

     [

    specific

     skill

     or

     specialty

    ]?

     


    As

     a

     skilled

     [

    specific

     skill

     or

     specialty

    ],

     I

     love

     [

    add

     a

     brief

     explanation

     of

     the

     character

    's

     interest

     or

     passion

    ].

     I

    'm

     always

     eager

     to

     learn

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     fact

    ually

     accurate

     and

     provides

     clear

     information

     about

     the

     capital

     city

     of

     France

    .

     However

    ,

     it

     is

     important

     to

     note

     that

     a

     statement

     about

     the

     capital

     of

     a

     specific

     country

     can

     vary

     based

     on

     the

     specific

     region

     or

     region

     that

     is

     being

     discussed

    .

     Therefore

    ,

     in

     this

     case

    ,

     the

     statement

     about

     Paris

     is

     an

     appropriate

     and

     complete

     answer

     to

     the

     question

    .

     
    


    If

     there

     is

     any

     specific

     information

     or

     context

     related

     to

     Paris

     that

     would

     be

     helpful

     for

     a

     response

    ,

     please

     let

     me

     know

    .

     Otherwise

    ,

     I

     can

     provide

     a

     complete

     and

     accurate

     statement

     about

     the

     capital

     city

    .

     
    


    For

     any

     other

     specific

     question

     about

     France

     or

     its

     capital

    ,

     please

     let

     me

     know

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     rapidly

     evolving

     and

     diverse

     field

    ,

     with

     new

     technologies

     and

     applications

     constantly

     emerging

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     are

     currently

     being

     explored

     and

     that

     could

     shape

     the

     future

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     AI

     will

     continue

     to

     become

     more

     personalized

    ,

     allowing

     machines

     to

     learn

     from

     the

     data

     they

     collect

     and

     provide

     more

     accurate

     and

     relevant

     recommendations

    .

     This

     will

     be

     particularly

     important

     in

     industries

     such

     as

     customer

     service

     and

     healthcare

    ,

     where

     personalized

     recommendations

     can

     save

     time

     and

     reduce

     errors

    .
    


    2

    .

     Autonomous

     Vehicles

    :

     AI

     will

     continue

     to

     play

     a

     critical

     role

     in

     the

     development

     of

     autonomous

     vehicles

    ,

     which

     will

     be

     able

     to

     navigate

     roads

     and

     highways

     on

     their

     own

    .

     This

     will

     require

    



```python
llm.shutdown()
```
