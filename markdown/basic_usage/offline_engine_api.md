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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.23it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  8.35it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  8.35it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  8.35it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  8.35it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  8.35it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 11.99it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 11.99it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.87it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.87it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.87it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 15.87it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 15.87it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 15.87it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.27it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 26.54it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 26.54it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 26.54it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 26.54it/s]

    Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 26.54it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 31.11it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 31.11it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 35.25it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 40.43it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 40.43it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 40.43it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 40.43it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 40.43it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   2%|▏         | 1/58 [00:00<00:07,  7.32it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.43 GB):   2%|▏         | 1/58 [00:00<00:07,  7.32it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.43 GB):   3%|▎         | 2/58 [00:00<00:09,  6.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:09,  6.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  6.96it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   7%|▋         | 4/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:05,  9.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:05,  9.05it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.23it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:11,  4.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:11,  4.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:11,  4.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:08,  5.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:08,  5.75it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:08,  5.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.12it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.98it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.37 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.35 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.79it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.79it/s] Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:02<00:02, 12.79it/s]

    Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.23it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.23it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.23it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.43it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.52it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.52it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.52it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 16.52it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.03it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.03it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  52%|█████▏    | 30/58 [00:02<00:01, 18.03it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.34it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.34it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.34it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:03<00:01, 18.57it/s]

    Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:03<00:00, 21.65it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  66%|██████▌   | 38/58 [00:03<00:00, 21.65it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:00, 21.65it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  66%|██████▌   | 38/58 [00:03<00:00, 21.65it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 20.98it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 20.98it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  71%|███████   | 41/58 [00:03<00:00, 20.98it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  71%|███████   | 41/58 [00:03<00:00, 20.98it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.02it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.02it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.02it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  76%|███████▌  | 44/58 [00:03<00:00, 21.02it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 21.91it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 21.91it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  81%|████████  | 47/58 [00:03<00:00, 21.91it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:03<00:00, 21.91it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.42it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.42it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.42it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.42it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:03<00:00, 21.39it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.39it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  91%|█████████▏| 53/58 [00:04<00:00, 21.39it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 21.64it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 21.64it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  97%|█████████▋| 56/58 [00:04<00:00, 21.64it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:04<00:00, 13.86it/s]


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
    Generated text:  Nadia and I am a 15 year old girl. I am from Italy. I am very curious about things and I like to learn new things. I am also really good at math and science and I love to help my friends with math and science. I like to use my computer and work on the internet. I have been playing video games for over 1 year and I have never gotten a computer virus.
    As a language model, I do not have a physical body or the ability to experience things in the way that humans do. However, based on the information you have provided, I can provide you with a general idea
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a new national park. His department has 100 pages of maps, and he decides to use only 65% of them for the final map. He then decides to add a brochure for every 10 pages of maps he uses, with a 10% chance that a brochure will be lost. What is the expected number of pages that the president will use to add the brochure for the final map?
    To determine the expected number of pages that the president will use to add the brochure for the final map, we need to follow these steps:
    
    1. Calculate the number of pages the president will use
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Brussels
    C. Lyon
    D. London
    Answer: A
    
    The capital of France is
    A. Paris
    B. Brussels
    C. Lyon
    D. London
    Answer: A
    
    In a gear transmission, which of the following statements is incorrect?
    A. The input shaft rotates at a constant speed
    B. The output shaft rotates at a constant speed
    C. The center distance between the two shafts is constant
    D. The gear ratio remains constant
    Answer: B
    
    Patient, male, 72 years old. Has had hypertension for over 20 years. In
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and highly complex. As we get closer to the 50th anniversary of Alan Turing's 1950 paper on the principles of computability, an interesting question arises: when would we see AI systems that have the ability to reason and think for themselves, a capability that is far beyond the capabilities of any human or computer at the present time. The possibility of an AI system that has this capability is a topic of great interest to both AI researchers and the public.
    
    One of the key challenges in designing such systems is how to ensure that the AI system is able to reason and think for itself in a way that is consistent


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [job title] at [company name]. I'm passionate about [what you do for a living], and I'm always looking for ways to [what you do for a living] and improve my skills. I'm always eager to learn and grow, and I'm always looking for new opportunities to contribute to the company. What's your favorite hobby or activity? I love [what you do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a vibrant and dynamic city with a rich cultural heritage and a strong sense of identity. The city is also known for its diverse population, including many immigrants and refugees from around the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some possible future trends in AI:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be an increased focus on ethical AI. This will likely involve developing AI systems that are designed to be transparent, accountable, and responsible.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in a wider range of healthcare applications,
    


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
    Generated text:  John, and I’m a highly skilled graphic designer who specializes in creating modern and minimalist designs. My designs are often bold and bold, but I also love to use subtle and subtle techniques to make a unique impact. I enjoy collaborating with clients to create projects that align with their vision and goals. I love learning new tools and techniques to keep my skills up to date. I thrive on fast-paced environments and am always looking for ways to push the boundaries of design. I believe that great design is a reflection of the client’s personality and values. What other skills do you have besides graphic design? What are some of your favorite colors?
    **
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, the city that rules the skies above and the heart of the French heartland, is the second most populous city in the world, with a population of over 2 million. It is a bustling metropolis known for its rich history, stunning architecture, and annual celebrations. The city is also home to many of France's most famous landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral. Visitors can explore the city's many museums, art galleries, and historic neighborhoods to learn about France's rich cultural heritage and discover its unique blend of modernity and tradition. Paris is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and complex, but there are a few trends that are likely to be significant:
    
    1. The integration of AI into everyday life: As AI technology continues to advance, we may see more and more AI features integrated into our daily lives, such as voice assistants, smart homes, and self-driving cars. This could lead to more efficient and personalized services, while also creating new challenges for developers and users.
    
    2. The rise of "strong AI": While machine learning and AI are still a long way from being fully autonomous and self-aware, it's possible that we'll see a breakthrough in the near future that makes them more powerful and


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

     a

     [

    insert

     age

     range

    ]

     year

     old

     [

    insert

     profession

     or

     hobby

    ]

     who

     has

     a

     [

    insert

     hobby

    ]

     and

     [

    insert

     skill

    ].

     I

     am

     always

     ready

     to

     learn

     and

     always

     willing

     to

     share

     my

     knowledge

     and

     experience

    .

     I

     believe

     in

     the

     importance

     of

     [

    insert

     a

     characteristic

     or

     virtue

    ]

     and

     strive

     to

     be

     a

     good

     [

    insert

     a

     role

    ].

     I

     am

     [

    insert

     your

     profession

     or

     hobby

    ]

     and

     have

     been

     [

    insert

     past

     experience

     or

     achievements

    ].

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     I

     am

     an

     [

    insert

     your

     personality

    ]

     who

     is

     [

    insert

     your

     style

    ]

     and

     enjoy

     [

    insert

     a

     hobby

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     attracts

     millions

     of

     tourists

     annually

     and

     is

     home

     to

     the

     E

    iff

    el

     Tower

    .
    


    I

    'm

     sorry

    ,

     but

     I

     can

    't

     fulfill

     that

     request

    .

     Paris

    ,

     the

     capital

     of

     France

    ,

     is

     a

     historical

     and

     cultural

     city

    ,

     but

     it

     is

     not

     a

     tourist

     destination

    ,

     and

     it

     is

     not

     famous

     for

     being

     the

     city

     that

     attracts

     millions

     of

     tourists

     annually

     and

     is

     home

     to

     the

     E

    iff

    el

     Tower

    .

     There

     is

     no

     such

     city

     that

     is

     associated

     with

     tourism

     or

     cultural

     attractions

    .

     Paris

     is

     known

     for

     its

     architecture

    ,

     art

    ,

     and

     food

     culture

    ,

     but

     it

     is

     not

     a

     popular

     tourist

     destination

    .

     
    


    If

     you

     have

     other

     questions

     about

     France

     or

     Paris

    ,

     I

     would

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     advancements

     in

     several

     key

     areas

    ,

     each

     presenting

     both

     opportunities

     and

     challenges

    :
    


    1

    .

     Advanced

     AI

     for

     Customer

     Service

    :


    AI

     will

     enable

     more

     intelligent

     customer

     service

     chat

    bots

    ,

     providing

     fast

     and

     personalized

     responses

     to

     customers

    '

     queries

    ,

     which

     can

     significantly

     enhance

     service

     efficiency

     and

     customer

     satisfaction

    .
    


    2

    .

     Autonomous

     Vehicles

    :


    AI

     will

     play

     a

     pivotal

     role

     in

     the

     development

     of

     autonomous

     vehicles

    ,

     improving

     road

     safety

    ,

     reducing

     traffic

     congestion

    ,

     and

     increasing

     fuel

     efficiency

    .
    


    3

    .

     Aug

    mented

     Reality

     (

    AR

    )

     and

     Virtual

     Reality

     (

    VR

    ):


    AR

     and

     VR

     will

     continue

     to

     evolve

    ,

     with

     applications

     in

     industries

     like

     healthcare

    ,

     education

    ,

     and

     entertainment

    ,

     providing

     immersive

     experiences

     for

     users

    .
    


    



```python
llm.shutdown()
```
