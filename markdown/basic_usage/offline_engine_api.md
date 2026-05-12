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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.50it/s]


    2026-05-12 01:24:58,465 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 01:24:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:46,  3.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:46,  3.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:46,  3.98s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:59,  1.08s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:59,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:59,  1.08s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:17,  2.91it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  5.03it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  5.03it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  5.03it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  5.03it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  7.56it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.52it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.52it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.52it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.52it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:03, 11.52it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 15.43it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:01, 20.53it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:01, 20.53it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:01, 20.53it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:01, 20.53it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:01, 20.53it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 24.27it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 24.27it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 24.27it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 24.27it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 24.27it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 24.27it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 29.34it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 33.32it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 33.32it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 33.32it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 33.32it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 33.32it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 33.32it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 34.69it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 34.69it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 34.69it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 34.69it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 34.69it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 34.69it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 40.95it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 40.95it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 40.95it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 40.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.46 GB):   2%|▏         | 1/58 [00:00<00:09,  5.95it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   2%|▏         | 1/58 [00:00<00:09,  5.95it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:08,  6.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:08,  6.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.61it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.94it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.05it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.37 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.57it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.34 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.34 GB):  36%|███▌      | 21/58 [00:01<00:02, 17.92it/s]Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 17.92it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  36%|███▌      | 21/58 [00:01<00:02, 17.92it/s]Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  40%|███▉      | 23/58 [00:02<00:01, 18.35it/s]Capturing num tokens (num_tokens=832 avail_mem=53.35 GB):  40%|███▉      | 23/58 [00:02<00:01, 18.35it/s]Capturing num tokens (num_tokens=768 avail_mem=53.35 GB):  40%|███▉      | 23/58 [00:02<00:01, 18.35it/s]Capturing num tokens (num_tokens=768 avail_mem=53.35 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.25it/s]Capturing num tokens (num_tokens=704 avail_mem=53.34 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.25it/s]

    Capturing num tokens (num_tokens=640 avail_mem=53.34 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.25it/s]Capturing num tokens (num_tokens=640 avail_mem=53.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=576 avail_mem=53.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=512 avail_mem=53.32 GB):  47%|████▋     | 27/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=512 avail_mem=53.32 GB):  50%|█████     | 29/58 [00:02<00:01, 18.58it/s]Capturing num tokens (num_tokens=480 avail_mem=53.34 GB):  50%|█████     | 29/58 [00:02<00:01, 18.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=53.34 GB):  50%|█████     | 29/58 [00:02<00:01, 18.58it/s]Capturing num tokens (num_tokens=448 avail_mem=53.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.32it/s]Capturing num tokens (num_tokens=416 avail_mem=53.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.32it/s]Capturing num tokens (num_tokens=384 avail_mem=53.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 17.32it/s]Capturing num tokens (num_tokens=384 avail_mem=53.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.71it/s]Capturing num tokens (num_tokens=352 avail_mem=53.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.71it/s]

    Capturing num tokens (num_tokens=320 avail_mem=53.32 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.71it/s]Capturing num tokens (num_tokens=320 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:01, 18.27it/s]Capturing num tokens (num_tokens=288 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:01, 18.27it/s]Capturing num tokens (num_tokens=256 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:01, 18.27it/s]Capturing num tokens (num_tokens=256 avail_mem=53.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=240 avail_mem=53.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.41it/s]

    Capturing num tokens (num_tokens=224 avail_mem=53.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.41it/s]Capturing num tokens (num_tokens=224 avail_mem=53.31 GB):  67%|██████▋   | 39/58 [00:02<00:01, 15.89it/s]Capturing num tokens (num_tokens=208 avail_mem=52.80 GB):  67%|██████▋   | 39/58 [00:02<00:01, 15.89it/s]

    Capturing num tokens (num_tokens=192 avail_mem=52.80 GB):  67%|██████▋   | 39/58 [00:03<00:01, 15.89it/s]Capturing num tokens (num_tokens=192 avail_mem=52.80 GB):  71%|███████   | 41/58 [00:03<00:01, 14.44it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:01, 14.44it/s]Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:01, 14.44it/s]Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  74%|███████▍  | 43/58 [00:03<00:00, 15.53it/s]Capturing num tokens (num_tokens=144 avail_mem=52.79 GB):  74%|███████▍  | 43/58 [00:03<00:00, 15.53it/s]

    Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  74%|███████▍  | 43/58 [00:03<00:00, 15.53it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  74%|███████▍  | 43/58 [00:03<00:00, 15.53it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.89it/s]Capturing num tokens (num_tokens=96 avail_mem=52.78 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.89it/s] Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.89it/s]

    Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  83%|████████▎ | 48/58 [00:03<00:00, 17.15it/s]Capturing num tokens (num_tokens=64 avail_mem=52.77 GB):  83%|████████▎ | 48/58 [00:03<00:00, 17.15it/s]Capturing num tokens (num_tokens=48 avail_mem=52.77 GB):  83%|████████▎ | 48/58 [00:03<00:00, 17.15it/s]Capturing num tokens (num_tokens=48 avail_mem=52.77 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.64it/s]Capturing num tokens (num_tokens=32 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.64it/s]Capturing num tokens (num_tokens=28 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.64it/s]

    Capturing num tokens (num_tokens=24 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:03<00:00, 17.64it/s]Capturing num tokens (num_tokens=24 avail_mem=52.76 GB):  91%|█████████▏| 53/58 [00:03<00:00, 18.75it/s]Capturing num tokens (num_tokens=20 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:03<00:00, 18.75it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  91%|█████████▏| 53/58 [00:03<00:00, 18.75it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  95%|█████████▍| 55/58 [00:03<00:00, 18.84it/s]Capturing num tokens (num_tokens=12 avail_mem=52.75 GB):  95%|█████████▍| 55/58 [00:03<00:00, 18.84it/s]Capturing num tokens (num_tokens=8 avail_mem=52.74 GB):  95%|█████████▍| 55/58 [00:03<00:00, 18.84it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=52.74 GB):  95%|█████████▍| 55/58 [00:03<00:00, 18.84it/s]Capturing num tokens (num_tokens=4 avail_mem=52.74 GB): 100%|██████████| 58/58 [00:03<00:00, 19.05it/s]Capturing num tokens (num_tokens=4 avail_mem=52.74 GB): 100%|██████████| 58/58 [00:03<00:00, 14.54it/s]


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
    Generated text:  Kim. I'm 14 years old. I have a dog named Max. I have to take care of him every day. He can talk to people. I love him. He's very clever and friendly. I have a favorite color, but it's not red. My favorite color is blue. I have a pet, like you. How can you tell the difference between a dog and a cat? I'm curious. What do you think? What's your favorite color? What do you think of your pet? It's okay to be a little funny about things. As a mother, I'm always trying to give my
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to raise the income tax or the estate tax. He has a choice between raising both taxes or keeping both taxes the same. In the first scenario, raising the income tax by $200 billion will increase GDP by 4% and raising the estate tax by $200 billion will increase GDP by 5%. In the second scenario, raising the income tax by $200 billion will increase GDP by 3% and raising the estate tax by $200 billion will increase GDP by 5%.
    
    Does the president have a clear optimal strategy? To determine whether the president has a clear optimal strategy
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. Paris
    
    The capital of France is Paris. Paris is the capital city of France. It is located in the north of the country and is the third largest city in France by population. It is also the capital of a province of France called the Île-de-France, which covers much of the south of the country. Paris is famous for its architecture, museums, and food, and it is home to many of the country's most famous landmarks and attractions. It is also the home of the European Parliament and other important government institutions. The city is known for its beautiful streets, vibrant culture, and annual summer festivals like
    ===============================
    Prompt: The future of AI is
    Generated text:  bleak – not because it’s bad, but because it’s not well understood. MIT’s own Sebastian Thrun talks about the future of AI in the comments section here.
    The future of AI is bleak – not because it’s bad, but because it’s not well understood. Sebastian Thrun, the inventor of the first artificial intelligence to achieve superhuman intelligence and who is the president of the European Council for Innovation and Technology (ECIT) says that while AI is rapidly becoming pervasive in our daily lives, it is still in its very early stages of development. He believes that, to reach its full potential, AI needs to be understood and


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Quarter. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art scene. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, with a diverse population of over 10 million people
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and there is a lot of potential for further development in this area. AI-powered diagnostic tools, chatbots
    


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
    Generated text:  [Name] and I'm a [job title] with [years] years of experience in the [field]. I'm passionate about [occupation] because [reason for passion]. I believe in [value] and strive to [step one of my values]. I have a [quantity] of [skill], and I enjoy [reason for skill]. I'm organized, detail-oriented, and have a strong work ethic. I'm constantly learning and growing, and I believe in [ultimate goal]. How do you feel about yourself? I'm confident, organized, and I'm always striving to improve. I love working with others, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, art, and cuisine. It is a city of bridges, towers, and stunning architecture, and is the third largest city in the European Union by population. Paris is home to the Eiffel Tower, Louvre Museum, and the Notre-Dame Cathedral, among many other famous landmarks. Its cuisine is a reflection of its rich history and cultural influences, with dishes such as croissants, boudin, and crêpes being popular. The city also plays a key role in the French economy and is a major center for art, music, and theater. Paris is the cultural and economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field, with numerous potential developments and trends shaping its future trajectory. Here are some possible trends in AI that could play a significant role in shaping the future of technology and society:
    
    1. Autonomous vehicles: Autonomous vehicles are already being developed, with some companies like Tesla and Waymo starting to offer self-driving cars. As the technology improves and becomes more widely adopted, we can expect to see more widespread adoption of autonomous vehicles, with the potential to dramatically reduce traffic accidents and increase safety for all drivers.
    
    2. Medical imaging: AI is already being used in medical imaging, helping to diagnose diseases and abnormalities more accurately and quickly than human


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

     male

     with

     [

    occupation

    ]

     and

     [

    physical

     characteristics

    ].

     I

     am

     [

    gender

    ].

     I

     have

     [

    abilities

    ]

     and

     [

    character

    istics

    ].

     I

     enjoy

     [

    my

     hobby

     or

     interest

    ].

     I

     like

     to

     [

    sp

    end

     time

     with

     friends

     or

     family

    ],

     [

    engage

     in

     any

     hobbies

     or

     sports

    ],

     [

    watch

     movies

     or

     listen

     to

     music

    ],

     etc

    .

     I

     am

     [

    friendly

    ,

     reserved

    ,

     outgoing

    ,

     or

     intro

    verted

    ],

     [

    pol

    ite

    ,

     casual

    ,

     or

     convers

    ational

    ],

     and

     [

    gener

    ous

    ,

     helpful

    ,

     or

     empath

    etic

    ].

     I

     am

     [

    easy

    -going

    ,

     energetic

    ,

     or

     driven

    ],

     [

    ad

    ap

    table

    ,

     resilient

    ,

     or

     flexible

    
    
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

     the

     second

    -largest

     city

     in

     the

     world

     by

     population

    ,

     after

     New

     York

     City

    .

     
    


    The

     city

    's

     skyline

     is

     impressive

     with

     tall

     buildings

     such

     as

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

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     known

     for

     its

     vibrant

     arts

     and

     culture

     scene

    ,

     including

     the

     Op

    éra

    ,

     the

     Mus

    ée

     Rod

    in

    ,

     and

     the

     Luxembourg

     Gardens

    .

     It

     is

     also

     renowned

     for

     its

     beautiful

     beaches

    ,

     including

     the

     Se

    ine

    -S

    aint

    -D

    enis

     and

     Les

     Invalid

    es

     beaches

    .

     The

     French

     language

     is

     the

     official

     language

     of

     France

     and

     is

     spoken

     by

     millions

     of

     people

     throughout

     the

     country

    .

     
    


    Paris

     is

     also

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     trends

     and

     developments

     that

     will

     shape

     how

     we

     use

     and

     interact

     with

     the

     technology

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     As

     AI

     continues

     to

     improve

     its

     ability

     to

     understand

     and

     process

     large

     amounts

     of

     data

    ,

     it

     is

     expected

     to

     become

     more

     personalized

     in

     its

     recommendations

     and

     services

    .

     This

     will

     enable

     businesses

     to

     offer

     more

     relevant

     products

     and

     services

     to

     individual

     customers

    ,

     leading

     to

     increased

     customer

     satisfaction

     and

     loyalty

    .
    


    2

    .

     Artificial

     General

     Intelligence

     (

    AG

    I

    ):

     This

     is

     the

     ultimate

     goal

     of

     AI

    ,

     where

     machines

     can

     think

    ,

     learn

    ,

     and

     innovate

     without

     being

     explicitly

     programmed

    .

     AG

    I

     has

     the

     potential

     to

     revolution

    ize

    



```python
llm.shutdown()
```
