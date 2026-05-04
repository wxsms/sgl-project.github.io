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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.49it/s]


    2026-05-04 04:03:53,787 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 04:03:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.82it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.82it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.82it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.82it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.82it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.65it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.65it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.65it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.65it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.65it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.65it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.76it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.76it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.76it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.76it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.76it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 21.72it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 21.72it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 21.72it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 21.72it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 21.72it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 23.88it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 23.88it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 23.88it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 23.88it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 23.88it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 25.82it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 25.82it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 25.82it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 25.82it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 25.51it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 25.51it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 25.51it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 25.51it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 25.70it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 25.70it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 25.70it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 25.70it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 25.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 25.68it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 27.52it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 27.52it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 27.52it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 27.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   2%|▏         | 1/58 [00:00<00:09,  5.80it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   2%|▏         | 1/58 [00:00<00:09,  5.80it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   3%|▎         | 2/58 [00:00<00:09,  5.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   3%|▎         | 2/58 [00:00<00:09,  5.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:08,  6.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:08,  6.33it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:08,  6.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:08,  6.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:07,  6.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.89 GB):   9%|▊         | 5/58 [00:00<00:07,  6.82it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.89 GB):  10%|█         | 6/58 [00:00<00:07,  7.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.88 GB):  10%|█         | 6/58 [00:00<00:07,  7.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:01<00:06,  7.67it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.86 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.86 GB):  22%|██▏       | 13/58 [00:01<00:02, 18.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.22 GB):  22%|██▏       | 13/58 [00:01<00:02, 18.33it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=71.21 GB):  22%|██▏       | 13/58 [00:01<00:02, 18.33it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.21 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.21 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.21 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.20 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.20 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.20 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.20 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.18 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.73it/s]Capturing num tokens (num_tokens=960 avail_mem=71.19 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.73it/s] Capturing num tokens (num_tokens=896 avail_mem=70.65 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.73it/s]

    Capturing num tokens (num_tokens=896 avail_mem=70.65 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.01it/s]Capturing num tokens (num_tokens=832 avail_mem=71.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.01it/s]Capturing num tokens (num_tokens=768 avail_mem=71.15 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.01it/s]Capturing num tokens (num_tokens=704 avail_mem=70.69 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.01it/s]Capturing num tokens (num_tokens=704 avail_mem=70.69 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=640 avail_mem=71.14 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.99it/s]

    Capturing num tokens (num_tokens=576 avail_mem=70.71 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=512 avail_mem=71.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=512 avail_mem=71.12 GB):  50%|█████     | 29/58 [00:01<00:01, 20.44it/s]Capturing num tokens (num_tokens=480 avail_mem=71.14 GB):  50%|█████     | 29/58 [00:01<00:01, 20.44it/s]Capturing num tokens (num_tokens=448 avail_mem=71.12 GB):  50%|█████     | 29/58 [00:01<00:01, 20.44it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.13 GB):  50%|█████     | 29/58 [00:02<00:01, 20.44it/s]Capturing num tokens (num_tokens=416 avail_mem=71.13 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.55it/s]Capturing num tokens (num_tokens=384 avail_mem=70.79 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.55it/s]Capturing num tokens (num_tokens=352 avail_mem=71.12 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.55it/s]Capturing num tokens (num_tokens=320 avail_mem=71.12 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.55it/s]Capturing num tokens (num_tokens=320 avail_mem=71.12 GB):  60%|██████    | 35/58 [00:02<00:01, 20.45it/s]Capturing num tokens (num_tokens=288 avail_mem=70.83 GB):  60%|██████    | 35/58 [00:02<00:01, 20.45it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.11 GB):  60%|██████    | 35/58 [00:02<00:01, 20.45it/s]Capturing num tokens (num_tokens=240 avail_mem=71.10 GB):  60%|██████    | 35/58 [00:02<00:01, 20.45it/s]Capturing num tokens (num_tokens=240 avail_mem=71.10 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.13it/s]Capturing num tokens (num_tokens=224 avail_mem=70.87 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.13it/s]Capturing num tokens (num_tokens=208 avail_mem=71.06 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.13it/s]Capturing num tokens (num_tokens=192 avail_mem=71.09 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.13it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.09 GB):  71%|███████   | 41/58 [00:02<00:00, 21.82it/s]Capturing num tokens (num_tokens=176 avail_mem=71.08 GB):  71%|███████   | 41/58 [00:02<00:00, 21.82it/s]Capturing num tokens (num_tokens=160 avail_mem=71.08 GB):  71%|███████   | 41/58 [00:02<00:00, 21.82it/s]Capturing num tokens (num_tokens=144 avail_mem=70.89 GB):  71%|███████   | 41/58 [00:02<00:00, 21.82it/s]Capturing num tokens (num_tokens=144 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.35it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.35it/s]Capturing num tokens (num_tokens=112 avail_mem=71.04 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.35it/s]Capturing num tokens (num_tokens=96 avail_mem=71.04 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.35it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=71.04 GB):  81%|████████  | 47/58 [00:02<00:00, 24.34it/s]Capturing num tokens (num_tokens=80 avail_mem=71.04 GB):  81%|████████  | 47/58 [00:02<00:00, 24.34it/s]Capturing num tokens (num_tokens=64 avail_mem=71.03 GB):  81%|████████  | 47/58 [00:02<00:00, 24.34it/s]Capturing num tokens (num_tokens=48 avail_mem=71.02 GB):  81%|████████  | 47/58 [00:02<00:00, 24.34it/s]Capturing num tokens (num_tokens=48 avail_mem=71.02 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.99it/s]Capturing num tokens (num_tokens=32 avail_mem=70.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.99it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.00 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.99it/s]Capturing num tokens (num_tokens=24 avail_mem=70.98 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.99it/s]Capturing num tokens (num_tokens=20 avail_mem=70.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.99it/s]Capturing num tokens (num_tokens=20 avail_mem=70.99 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.58it/s]Capturing num tokens (num_tokens=16 avail_mem=70.99 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.58it/s]Capturing num tokens (num_tokens=12 avail_mem=70.98 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.58it/s]Capturing num tokens (num_tokens=8 avail_mem=70.97 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.58it/s] Capturing num tokens (num_tokens=4 avail_mem=70.96 GB):  93%|█████████▎| 54/58 [00:03<00:00, 24.58it/s]

    Capturing num tokens (num_tokens=4 avail_mem=70.96 GB): 100%|██████████| 58/58 [00:03<00:00, 27.78it/s]Capturing num tokens (num_tokens=4 avail_mem=70.96 GB): 100%|██████████| 58/58 [00:03<00:00, 18.65it/s]


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
    Generated text:  Peter and I have a pet parrot. I have a very sweet and adorable pet parrot named Echo. I have been taking care of her for about a year now. I love to share my thoughts about my pet parrot with you.
    My parrot is quite lovely and has a very sweet personality. She is also quite smart and has a good memory. She is quite vocal, so we have had some fun getting along with her. She has also shown her interest in toys and objects, and I love to see this.
    She has been taking care of her food and water for about a year now and I have seen her chewing
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for re-election. He has 5 potential voters, each with a different probability of supporting him in a vote. Each voter's potential support can be modeled by the probability \( p_i \), where \( p_i \) is the probability of voter \( i \) supporting the president. The goal is to maximize the total potential support, which can be modeled as the sum of the probabilities of all voters supporting him. The president's goal is to maximize his total potential support, but he can only vote for a maximum of 4 different voters. What is the best strategy for the president to maximize his total potential support? To determine
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, where the city-state of France is located. The city is situated in the northeastern part of the country, along the Seine River. The city is the capital of the department of Paris. It is a major international financial center, and its economy is based on banking, insurance, and financial services.
    Is there an answer to this question (If it cannot be answered, return "Unanswerable"). No. Paris, the capital of France, is a major international financial center. It is located along the Seine River and is the capital of the department of Paris. It is a major international financial center and has an economy based
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable, but some researchers believe that a new generation of artificial intelligence will continue to develop.
    This is a new generation of AI, it's called "Chromax." The first generation of artificial intelligence was created by the Turing test, which was the first test of artificial intelligence. The test is designed to determine if a machine can exhibit human-like intelligence. The test is a set of questions that the machine is asked to answer. If the machine can answer the questions correctly, it is considered to be intelligent. The second generation of artificial intelligence was created by DeepMind, which was created by a company called DeepMind. DeepMind's AI


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Occupation] who has always been passionate about [What interests you about your occupation]. I'm always looking for new challenges and opportunities to grow and learn. I'm always eager to learn and improve myself. I'm a [What do you like to do for fun?] who loves to [What is something you enjoy doing in your free time?] I'm a [What is your favorite hobby?] who enjoys [What is something you like to do with friends?] I'm a [What is your favorite thing about [Occupation
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is also the seat of the French government and home to many of the country's cultural and political institutions. Paris is a bustling metropolis with a rich history dating back to the Roman Empire and the French Revolution. The city is known for its fashion, art, and cuisine, and is a major tourist destination. It is also home to many famous landmarks such as the Notre-Dame Cathedral and the Arc de Triomphe. Paris is a vibrant and dynamic city that continues to evolve and grow, with a strong sense of identity and a rich cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. AI ethics and privacy concerns: As AI technology becomes more advanced, there will be increasing concerns about its impact on society. This could include issues such as bias in AI algorithms
    


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
    Generated text:  [Name] and I am a [profession]. I'm excited to meet you and learn about your interests and passions. Let's connect! [Tell about your experience and skills that make you unique. Are you a natural public speaker, an expert in a particular field, or do you have a particular niche in the industry? How do you approach problem-solving and decision-making? How do you handle stress and maintain a healthy work-life balance? Let me know if you have any questions or concerns before we begin the conversation. I look forward to hearing from you! [Tell any additional relevant information that can help the listener understand my personality and personality
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To address your question, here is a concise factual statement about Paris: 
    
    Paris, the capital of France, is the largest city in Europe, known for its rich history, breathtaking architecture, and vibrant culture. Its iconic landmarks include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris also hosts numerous world-class museums, theaters, and festivals throughout the year. The city is an important transportation hub, with many roads and rail lines connecting its many neighborhoods. 
    
    To confirm this statement, I would need more specific information about what aspects of Paris you would like to highlight or expand upon, such
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and growing. Here are some possible future trends:
    
    1. Increased use of AI in healthcare: As AI technology improves, it can be used to diagnose and treat diseases with greater accuracy and speed. In the future, AI may play a more significant role in healthcare by improving patient outcomes, reducing costs, and making healthcare more accessible.
    
    2. AI-powered agriculture: AI can be used to monitor crop health and yield, predict weather patterns, and optimize planting and harvesting schedules. This could lead to increased crop yields and reduced waste, while also reducing the need for chemical inputs.
    
    3. AI in finance: AI can be used to identify fraud


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

    'm

     a

    /an

     [

    Occup

    ation

    ]

     and

     I

    'm

     currently

     [

    Age

    ].

     I

     grew

     up

     in

     [

    City

    ].

     I

    've

     always

     been

     [

    tal

    ent

     or

     passion

    ].

     I

    've

     traveled

     the

     world

     and

     traveled

     through

     languages

    ,

     cultures

    ,

     and

     [

    learning

     or

     experience

    ].

     I

    'm

     fluent

     in

     [

    languages

    ],

     and

     I

    'm

     [

    able

     to

     speak

     or

     write

    ].


    Wow

    ,

     that

    's

     quite

     a

     mouth

    ful

    ,

     but

     can

     you

     tell

     me

     more

     about

     your

     background

     and

     how

     you

     got

     started

     in

     the

     field

    ?

     What

     is

     your

     passion

     or

     talent

     that

     led

     you

     to

     this

     occupation

    ,

     and

     what

     language

     do

     you

     speak

     flu

    ently

    ?

     Can

     you

     also

     tell

     me

     about

     any

     unique

     experiences

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Cer

    isy

    ne

    ."
    


    The

     city

     is

     renowned

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

     It

     also

     boasts

     one

     of

     the

     oldest

     universities

     in

     the

     world

    ,

     the

     Sor

    bon

    ne

    ,

     and

     has

     a

     rich

     cultural

     heritage

     dating

     back

     over

     

    5

    0

    0

     years

    .

     Paris

     is

     also

     known

     for

     its

     annual

     festival

     of

     lights

    ,

     the

     Op

    éra

    ,

     and

     its

     historic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     a

     popular

     destination

     for

     tourists

     and

     has

     a

     vibrant

     cultural

     scene

    ,

     with

     many

     museums

     and

     galleries

    .

     With

     its

     historical

     architecture

    ,

     vibrant

     life

    ,

     and

     culture

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     exciting

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Increased

     efficiency

    :

     AI

     is

     expected

     to

     improve

     efficiency

     in

     industries

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    .

     It

     can

     automate

     repetitive

     tasks

     and

     optimize

     processes

    ,

     leading

     to

     increased

     productivity

     and

     cost

     savings

    .
    


    2

    .

     Autonomous

     vehicles

    :

     As

     autonomous

     vehicles

     become

     more

     sophisticated

    ,

     they

     are

     likely

     to

     become

     more

     common

    ,

     reducing

     the

     need

     for

     human

     drivers

     and

     increasing

     safety

    .

     AI

    -powered

     systems

     will

     be

     able

     to

     detect

     and

     avoid

     obstacles

    ,

     navigate

     roads

    ,

     and

     make

     decision

    -making

     based

     on

     real

    -time

     data

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

     can

     be

     used

     to

     develop

     personalized

     treatment

     plans

     for

     patients

     based

     on

     their

     genetic

     makeup

    ,

    



```python
llm.shutdown()
```
