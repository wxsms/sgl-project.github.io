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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.44it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.44it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.44it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.44it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.44it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.62it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.32it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.32it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.32it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 12.32it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 12.32it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 16.22it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 16.22it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 16.22it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 16.22it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.22it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 16.22it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 21.00it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 29.97it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 29.97it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 29.97it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 29.97it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 29.97it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 29.97it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 41.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 48.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.54it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:05,  9.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:04, 10.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:04, 10.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):   9%|▊         | 5/58 [00:00<00:04, 10.82it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.43it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.93it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 12.69it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 13.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.36it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.49it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.49it/s] Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.49it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.49it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:01<00:01, 18.87it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  41%|████▏     | 24/58 [00:01<00:01, 18.87it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 18.87it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 18.87it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.67it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.67it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.67it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 19.67it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.38it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.38it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.38it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  52%|█████▏    | 30/58 [00:02<00:01, 20.38it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.02it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.02it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.02it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  57%|█████▋    | 33/58 [00:02<00:01, 21.02it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.29it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.29it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.29it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  62%|██████▏   | 36/58 [00:02<00:01, 21.29it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.60it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.31it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.31it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.31it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.31it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.66it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.66it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.66it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:02<00:00, 22.66it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.05it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.05it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.05it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.05it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.20it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.20it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 23.20it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  88%|████████▊ | 51/58 [00:03<00:00, 23.20it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.33it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.33it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.33it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 23.33it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:03<00:00, 23.40it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:03<00:00, 23.40it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 17.77it/s]


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
    Generated text:  Ariel and I'm a hardworking teen who is learning to do sports. In the coming weeks, I want to improve my running speed and endurance. Can you please give me tips on how to improve my running speed and endurance?
    
    Certainly! Improving your running speed and endurance involves a combination of training, nutrition, and proper technique. Here are some tips to help you:
    
    ### 1. **Nutrition**
    - **Carbohydrates**: Focus on carbohydrates, which are the body's main source of energy. Include a variety of fruits, vegetables, and whole grains.
    - **Protein**: Proteins are also essential for muscle growth
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considering whether to hold a third term. If the president were to hold a third term, he would have to serve an additional four years. The current president served two years, and the person he replaced served four years. How many more years in total would the president have served if the president were to hold a third term? To determine the total number of years the president would have served if he were to hold a third term, we need to consider the additional years he would serve in the upcoming three years.
    
    1. The president served two years in the current term.
    2. The person he replaced served four years in the current term
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. London
    C. Rome
    D. Berlin
    Answer:
    
    A
    
    In the "Recycling Classroom" activity, students learn that the best way to save metal is to recycle. The title of this activity is ____
    A. I Save a Metal, I Save a Life
    B. I Save a Metal, I Save a Future
    C. I Save a Metal, I Save a Time
    D. I Save a Metal, I Save a Good
    Answer:
    
    B
    
    Which of the following sentences is free of grammatical errors? A. The school organizes some voluntary labor activities, helping
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but one thing is certain: it is changing the way we live and work. Here are some potential changes that may occur:
    
    1. AI will become more ubiquitous - As AI becomes more advanced and more accessible, it will become more integrated into our lives. This will include everything from smart homes to virtual assistants and chatbots.
    
    2. AI will be used for more complex tasks - As AI becomes more sophisticated, it will be able to perform a wider range of tasks than ever before. This will include things like medical diagnostics, security, and even art and design.
    
    3. AI will be used for more precise and accurate results -


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your personality or skills]. And what's your background? I have [insert a brief description of your education or work experience]. And what's your favorite hobby or activity? I enjoy [insert a brief description of your favorite activity]. And what's your favorite book or movie? I love [insert a brief description of your favorite book or movie]. And what's your favorite place to go? I love [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose" (the Rose City). It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is a cultural, historical, and artistic center, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a major financial center and a major transportation hub, with many major airports and train stations. The city is home to many museums, art galleries, and theaters, and is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, leading to increased automation of tasks that were previously done by humans. This could lead to job losses in some sectors, but also create new opportunities for automation and innovation.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be an increased need for privacy and security measures to protect the data they collect. This could lead to new regulations and standards for AI systems, as
    


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
    Generated text:  [Your Name] and I'm a/an [Your Profession] [Your occupation or field of expertise] with over [Number] years of experience in [Field of Study]. I have a strong foundation in [Specific Skill or Knowledge], and I enjoy exploring new ideas and learning new things. I'm always looking for ways to improve and grow as a professional. If you need help, feedback, or any resources, I'm here to assist. [Your Name], feel free to introduce yourself! 📚💼✨
    
    Hey there, fellow professionals! I'm [Your Name], the master of my own word! 🤩✨
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is a bustling and diverse city known for its rich history and cultural heritage. The city is also known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, as well as its famous neighborhoods such as the Latin Quarter and the Eiffeline. Paris is a city of contrasts and influences from all over the world, making it a fascinating place to explore. It is also home to many world-renowned art museums, such as the Musée d'Orsay and the Musée d'Art Moderne, which attract millions of visitors annually. Overall, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve significant advances in areas such as deep learning and reinforcement learning, which could enable machines to perform increasingly complex tasks and learn from a wide range of data. Additionally, there may be an increased focus on developing ethical considerations and ensuring that AI is used responsibly and in the public interest. AI-powered systems may also become more accessible and affordable, making them more widely available to individuals and industries. Finally, it's possible that AI will continue to evolve and adapt, becoming increasingly capable of performing tasks that are currently beyond the capabilities of human beings. Overall, the future of AI is likely to involve further innovations and developments, but with the potential


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

     computer

     scientist

     with

     a

     passion

     for

     exploring

     the

     unknown

    .

     My

     work

     is

     primarily

     focused

     on

     designing

     and

     building

     complex

     systems

    ,

     but

     I

     also

     enjoy

     teaching

     and

     learning

     new

     technologies

    .

     I

    'm

     always

     eager

     to

     learn

    ,

     try

     new

     things

    ,

     and

     challenge

     myself

    .

     What

     other

     role

     or

     project

     do

     you

     think

     I

     could

     be

     part

     of

    ?

     [

    Name

    ]

     Please

     describe

     your

     role

     or

     project

     and

     how

     it

     might

     have

     been

     influenced

     by

     your

     interest

     in

     computers

    .

     Hello

    ,

     my

     name

     is

     [

    Name

    ].

     I

    'm

     a

     computer

     scientist

     with

     a

     passion

     for

     exploring

     the

     unknown

    .

     My

     work

     is

     primarily

     focused

     on

     designing

     and

     building

     complex

     systems

    ,

     but

     I

     also

     enjoy

     teaching

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     the

     seat

     of

     government

     of

     the

     country

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     of

     France

    ,

     located

     on

     the

     Î

    le

     de

     France

     in

     the

     south

     of

     the

     country

    .

     It

     is

     one

     of

     the

     most

     important

     cities

     in

     Europe

     and

     is

     famous

     for

     its

     classical

     architecture

    ,

     museums

    ,

     and

     fashion

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     landmarks

    ,

     including

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

     also

     known

     for

     its

     culinary

     scene

    ,

     where

     the

     city

    's

     famous

     dishes

     like

     be

    ign

    ets

    ,

     cro

    iss

    ants

    ,

     and

     b

    oul

    anger

    ie

     are

     enjoyed

     by

     millions

     of

     visitors

     each

     year

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     shaped

     by

     several

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increased

     integration

     of

     AI

     into

     various

     industries

     and

     services

    :

     As

     AI

     becomes

     more

     integrated

     into

     various

     industries

     and

     services

    ,

     we

     are

     likely

     to

     see

     a

     shift

     towards

     more

     personalized

     and

     automated

     solutions

    ,

     such

     as

     chat

    bots

     and

     virtual

     assistants

    .

     This

     will

     lead

     to

     greater

     efficiency

     and

     cost

     savings

     for

     businesses

    .


     

     

    2

    .

     Continued

     development

     of

     AI

     technologies

     and

     algorithms

    :

     AI

     is

     likely

     to

     continue

     to

     be

     developed

     and

     refined

    ,

     with

     new

     algorithms

     and

     techniques

     being

     integrated

     into

     AI

     systems

    .

     This

     will

     likely

     lead

     to

     greater

     accuracy

     and

     efficiency

     in

     AI

     applications

    .


     

     

    3

    .

     Increased

     focus

     on

     ethical

     and

     responsible

     AI

    :

    



```python
llm.shutdown()
```
