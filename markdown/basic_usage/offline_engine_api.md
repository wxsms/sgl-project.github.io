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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:07,  1.22s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:07,  1.22s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:07,  1.22s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.62it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.62it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.62it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.95it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.95it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.95it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.95it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.95it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.71it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.71it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.71it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.71it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.59it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.59it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.59it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.59it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.59it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.59it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:01, 20.27it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:00, 27.71it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 34.04it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 41.00it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 45.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.59it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.59it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:06,  7.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   5%|▌         | 3/58 [00:00<00:06,  7.90it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:06,  8.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   7%|▋         | 4/58 [00:00<00:06,  8.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.42 GB):   9%|▊         | 5/58 [00:00<00:06,  8.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):   9%|▊         | 5/58 [00:00<00:06,  8.71it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.40 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.06it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.39 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.28it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 13.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.72it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=53.36 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.36 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.34 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.78it/s]Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.78it/s] Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.78it/s]

    Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.78it/s]Capturing num tokens (num_tokens=832 avail_mem=53.35 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.78it/s]Capturing num tokens (num_tokens=768 avail_mem=53.35 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.78it/s]Capturing num tokens (num_tokens=704 avail_mem=53.34 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.78it/s]Capturing num tokens (num_tokens=704 avail_mem=53.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.75it/s]Capturing num tokens (num_tokens=640 avail_mem=53.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.75it/s]Capturing num tokens (num_tokens=576 avail_mem=53.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.75it/s]Capturing num tokens (num_tokens=512 avail_mem=53.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.75it/s]

    Capturing num tokens (num_tokens=512 avail_mem=53.32 GB):  50%|█████     | 29/58 [00:01<00:01, 21.59it/s]Capturing num tokens (num_tokens=480 avail_mem=53.34 GB):  50%|█████     | 29/58 [00:01<00:01, 21.59it/s]Capturing num tokens (num_tokens=448 avail_mem=53.34 GB):  50%|█████     | 29/58 [00:02<00:01, 21.59it/s]Capturing num tokens (num_tokens=416 avail_mem=53.34 GB):  50%|█████     | 29/58 [00:02<00:01, 21.59it/s]Capturing num tokens (num_tokens=416 avail_mem=53.34 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.51it/s]Capturing num tokens (num_tokens=384 avail_mem=53.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.51it/s]Capturing num tokens (num_tokens=352 avail_mem=53.33 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.51it/s]

    Capturing num tokens (num_tokens=320 avail_mem=53.32 GB):  55%|█████▌    | 32/58 [00:02<00:01, 22.51it/s]Capturing num tokens (num_tokens=320 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:00, 23.01it/s]Capturing num tokens (num_tokens=288 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:00, 23.01it/s]Capturing num tokens (num_tokens=256 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:00, 23.01it/s]Capturing num tokens (num_tokens=240 avail_mem=53.32 GB):  60%|██████    | 35/58 [00:02<00:00, 23.01it/s]Capturing num tokens (num_tokens=240 avail_mem=53.32 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.03it/s]Capturing num tokens (num_tokens=224 avail_mem=53.31 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.03it/s]

    Capturing num tokens (num_tokens=208 avail_mem=53.31 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.03it/s]Capturing num tokens (num_tokens=192 avail_mem=53.31 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.03it/s]Capturing num tokens (num_tokens=192 avail_mem=53.31 GB):  71%|███████   | 41/58 [00:02<00:00, 23.43it/s]Capturing num tokens (num_tokens=176 avail_mem=53.30 GB):  71%|███████   | 41/58 [00:02<00:00, 23.43it/s]Capturing num tokens (num_tokens=160 avail_mem=53.30 GB):  71%|███████   | 41/58 [00:02<00:00, 23.43it/s]Capturing num tokens (num_tokens=144 avail_mem=53.30 GB):  71%|███████   | 41/58 [00:02<00:00, 23.43it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.30 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.40it/s]Capturing num tokens (num_tokens=128 avail_mem=53.29 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.40it/s]Capturing num tokens (num_tokens=112 avail_mem=53.29 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.40it/s]Capturing num tokens (num_tokens=96 avail_mem=53.29 GB):  76%|███████▌  | 44/58 [00:02<00:00, 23.40it/s] Capturing num tokens (num_tokens=96 avail_mem=53.29 GB):  81%|████████  | 47/58 [00:02<00:00, 23.80it/s]Capturing num tokens (num_tokens=80 avail_mem=53.28 GB):  81%|████████  | 47/58 [00:02<00:00, 23.80it/s]Capturing num tokens (num_tokens=64 avail_mem=53.28 GB):  81%|████████  | 47/58 [00:02<00:00, 23.80it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  81%|████████  | 47/58 [00:02<00:00, 23.80it/s]Capturing num tokens (num_tokens=48 avail_mem=53.28 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.04it/s]Capturing num tokens (num_tokens=32 avail_mem=53.27 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.04it/s]Capturing num tokens (num_tokens=28 avail_mem=53.27 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.04it/s]Capturing num tokens (num_tokens=24 avail_mem=53.27 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.04it/s]Capturing num tokens (num_tokens=24 avail_mem=53.27 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.17it/s]Capturing num tokens (num_tokens=20 avail_mem=53.26 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.17it/s]

    Capturing num tokens (num_tokens=16 avail_mem=53.26 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.17it/s]Capturing num tokens (num_tokens=12 avail_mem=53.26 GB):  91%|█████████▏| 53/58 [00:03<00:00, 24.17it/s]Capturing num tokens (num_tokens=12 avail_mem=53.26 GB):  97%|█████████▋| 56/58 [00:03<00:00, 24.83it/s]Capturing num tokens (num_tokens=8 avail_mem=53.25 GB):  97%|█████████▋| 56/58 [00:03<00:00, 24.83it/s] Capturing num tokens (num_tokens=4 avail_mem=53.25 GB):  97%|█████████▋| 56/58 [00:03<00:00, 24.83it/s]Capturing num tokens (num_tokens=4 avail_mem=53.25 GB): 100%|██████████| 58/58 [00:03<00:00, 18.46it/s]


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
    Generated text:  Rita. I'm a member of the Communications Department at the University of California, Davis. My job is to help people understand the signals on a television channel, radio channel, or other form of media. That's what I do all day, every day, and it doesn't get any easier. I answer the phone, I set up my equipment, I help people understand what the television or radio broadcast is telling them, and I make sure they have proper manners and etiquette on the air. If I have to do any of those things, I do it. Because I have no choice. I have to do it for my job and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president is elected by the members of Congress. The members of Congress are made up of the members of both houses of Congress. Each member of Congress represents a district. The President has the power to appoint 25 cabinet members and the governor of each state. How many members of Congress does it take to represent 100 states? The president is elected by the members of Congress, and each member of Congress represents a district. Since there are 50 states, and each state needs a representative, it would take 50 representatives to represent 100 states. However, the president also
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is situated in the center of the European continent and is located on the banks of the Seine River. Its area is 164 km² and its population is 2, 461, 000 as of 2008. The capital of France is located in the heart of Paris, which is surrounded by a network of streets called the Marais. The Marais area has long been known as the most desirable residential area in the city and remains the heart of Paris. The Marais is a residential area near the axis of the city. This area is known for its unique architectural
    ===============================
    Prompt: The future of AI is
    Generated text:  a bit cloudy due to the fact that the term is still new and some might think that it’s too young to know how to handle it, but this is not the case. What do you think about the potential of AI? What are the pros and cons of AI? What are the technological challenges that arise from AI and how are these being solved? What are the ethical implications of AI and how are they being addressed?
    You can share your thoughts and experiences in the comments below. Let’s explore the future of AI together.
    For more on the topic, check out this article on the future of AI. It’s a great read and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to learn more about your career and what you do. What do you do for a living? [Name] is a [job title] at [company name], and I'm excited to learn more about your career and what you do. What do you do for a living? [Name] is a [job title] at [company name], and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the annual Eiffel Tower Festival. It is also the birthplace of French writer Charles Dickens. Paris is a major cultural and economic center, with a rich history dating back to the Roman Empire and the medieval period. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is home to many famous landmarks, including the Louvre Museum and the Arc de Triomphe. Paris is a popular tourist destination, with millions of visitors annually. It is also known for its cuisine,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and responsible development. This could lead to more stringent regulations and guidelines for
    


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
    Generated text:  [Name], and I am a [Type of Person] with [Number of Years Experience in [Industry/Field]]. I have [Number of Projects Completed] projects under my belt, and I have successfully [Achievements in the industry]. I believe that [Name] is a fantastic fit for your role, and I look forward to working with you. Let’s make something special together! [Name]: [Job Title] [Name]: [Job Title]
    Hey there! I'm [Name] and I'm a [Type of Person] with [Number of Years Experience in [Industry/Field]]. I've got [Number of Projects
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A. True B. False
    A. True
    
    Paris is the capital city of France and is known for its beautiful Eiffel Tower, romantic canals, and diverse cultural scene. The French government is based in the city, and the city is the economic and cultural center of France. Paris is also known as the City of Light, referring to its blindingly bright sunlight that illuminates its buildings. It is a world-renowned tourist destination and has a rich history and culture. Despite its size, Paris is the third largest city in France, and the population has grown significantly over the years. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of key trends, including:
    
    1. Increased focus on ethical AI: As ethical concerns grow more important, there may be a greater push towards developing AI systems that are more aligned with ethical principles, such as minimizing bias and ensuring transparency.
    
    2. Continued development of quantum computing: Quantum computers have the potential to revolutionize AI by breaking through the limitations of classical computers, allowing for faster and more efficient algorithms. 
    
    3. Integration of AI with other technologies: AI is already being integrated into a wide range of other technologies, such as autonomous vehicles, smart homes, and smart cities. It is likely that this integration will


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

    /an

     [

    Your

     Profession

    ]

     with

     a

     passion

     for

     [

    Your

     Profession

    ]

     and

     [

    Your

     Profession

    's

     Profession

    ].

     I

     am

     [

    Your

     Profession

    ]

     and

     I

     am

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     How

     can

     I

     help

     you

     today

    ?


    I

     am

     the

     founder

     of

     the

     [

    Your

     Company

    ]

     and

     I

     am

     passionate

     about

     creating

     a

     positive

     impact

     on

     the

     world

    .

     My

     goal

     is

     to

     use

     my

     skills

     and

     knowledge

     to

     help

     people

     and

     make

     the

     world

     a

     better

     place

    .

     I

     am

     always

     looking

     for

     new

     ways

     to

     innovate

     and

     improve

    ,

     and

     I

     am

     always

     eager

     to

     learn

     and

     grow

    .

     What

     exc

    ites

     me

     the

     most

     about

     my

     career

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Task

    :

     Write

     a

     concise

     factual

     statement

     about

     France

    ’s

     capital

     city

    .

     Provide

     your

     answer

     in

     French

    .
    


    Les

     capit

    ales

     de

     l

    '

    Al

    lem

    agne

     sont

     Berlin

     et

     Paris

    .

     Comment

     faire

     ?

     Dans

     cette

     question

    ,

     vous

     devez

     :
    


    1

    .

     Ident

    ifie

    z

     le

     nombre

     exact

     de

     capit

    ales

     de

     l

    '

    Al

    lem

    agne

    .


    2

    .

     Ident

    ifie

    z

     le

     nombre

     exact

     de

     capit

    ales

     de

     l

    '

    Al

    lem

    agne

     act

    uelles

    .


    3

    .

     A

    fin

     de

     faire

     une

     question

    ,

     vous

     devez

     adapter

     la

     réponse

     pour

     qu

    'elle

     soit

     question

    née

     en

     français

    .
    


    Solution

     :


    Les

     capit

    ales

     de

     l

    '

    Al

    lem

    agne

     sont

     

    3

     :

     Berlin

    ,

     Paris

    ,

     et

     Erf

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     but

     also

     some

     potential

     downs

    ides

    .

     One

     of

     the

     most

     promising

     areas

     for

     future

     developments

     in

     AI

     is

     the

     development

     of

     more

     advanced

     self

    -driving

     cars

    .

     Companies

     like

     Google

    ,

     Tesla

    ,

     and

     Uber

     are

     already

     making

     significant

     strides

     in

     developing

     self

    -driving

     cars

    ,

     with

     many

     autom

    akers

     also

     investing

     heavily

     in

     research

     and

     development

     to

     make

     them

     more

     practical

     and

     cost

    -effective

    .
    


    Another

     area

     of

     potential

     future

     AI

     developments

     is

     in

     the

     field

     of

     healthcare

    .

     AI

     is

     already

     being

     used

     to

     analyze

     medical

     data

     and

     help

     doctors

     make

     more

     accurate

     diagnoses

     and

     treatment

     plans

    .

     In

     the

     future

    ,

     we

     may

     see

     even

     more

     sophisticated

     AI

     systems

     that

     can

     analyze

     medical

     images

    ,

     analyze

     medical

     records

    ,

    



```python
llm.shutdown()
```
