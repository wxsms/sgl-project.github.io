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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:46,  3.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:46,  3.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:46,  3.98s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.74it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.74it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.74it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.78it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:11,  4.10it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.45it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  9.08it/s]

    Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  9.08it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.89it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.89it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.89it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 12.89it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 12.89it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.68it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.68it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.68it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.68it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.68it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.47it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.47it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.47it/s]

    Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.47it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.47it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.47it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 28.76it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 28.76it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 28.76it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 28.76it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 28.76it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 28.76it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 32.16it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 34.54it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 34.54it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 34.54it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 34.54it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 34.54it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 34.54it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 36.16it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 36.16it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 36.16it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 36.16it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 36.16it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 36.16it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 38.97it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 38.97it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 38.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.78 GB):   2%|▏         | 1/58 [00:00<00:08,  6.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.77 GB):   2%|▏         | 1/58 [00:00<00:08,  6.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.77 GB):   3%|▎         | 2/58 [00:00<00:09,  5.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.78 GB):   3%|▎         | 2/58 [00:00<00:09,  5.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.78 GB):   5%|▌         | 3/58 [00:00<00:09,  5.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.80 GB):   5%|▌         | 3/58 [00:00<00:09,  5.93it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.80 GB):   7%|▋         | 4/58 [00:00<00:08,  6.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.81 GB):   7%|▋         | 4/58 [00:00<00:08,  6.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.81 GB):   9%|▊         | 5/58 [00:00<00:08,  6.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.82 GB):   9%|▊         | 5/58 [00:00<00:08,  6.28it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.82 GB):  10%|█         | 6/58 [00:00<00:07,  6.58it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.99 GB):  10%|█         | 6/58 [00:00<00:07,  6.58it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.99 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.98 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.87it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.98 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.98 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.98 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.35it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.97 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.96 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.96 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.96 GB):  21%|██        | 12/58 [00:01<00:05,  8.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.95 GB):  21%|██        | 12/58 [00:01<00:05,  8.73it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=55.95 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.94 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.94 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.93 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.04it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.93 GB):  24%|██▍       | 14/58 [00:01<00:04,  9.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.93 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.92 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.91 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.47it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=55.91 GB):  31%|███       | 18/58 [00:02<00:04,  9.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.91 GB):  31%|███       | 18/58 [00:02<00:04,  9.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.89 GB):  31%|███       | 18/58 [00:02<00:04,  9.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.89 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.89 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.05it/s]

    Capturing num tokens (num_tokens=960 avail_mem=55.90 GB):  34%|███▍      | 20/58 [00:02<00:03, 10.05it/s] Capturing num tokens (num_tokens=960 avail_mem=55.90 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.32it/s]Capturing num tokens (num_tokens=896 avail_mem=55.89 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.32it/s]Capturing num tokens (num_tokens=832 avail_mem=55.88 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.32it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.88 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.62it/s]Capturing num tokens (num_tokens=768 avail_mem=55.88 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.62it/s]Capturing num tokens (num_tokens=704 avail_mem=55.87 GB):  41%|████▏     | 24/58 [00:02<00:03, 10.62it/s]Capturing num tokens (num_tokens=704 avail_mem=55.87 GB):  45%|████▍     | 26/58 [00:02<00:02, 10.84it/s]Capturing num tokens (num_tokens=640 avail_mem=55.87 GB):  45%|████▍     | 26/58 [00:02<00:02, 10.84it/s]

    Capturing num tokens (num_tokens=576 avail_mem=55.85 GB):  45%|████▍     | 26/58 [00:03<00:02, 10.84it/s]Capturing num tokens (num_tokens=576 avail_mem=55.85 GB):  48%|████▊     | 28/58 [00:03<00:02, 10.98it/s]Capturing num tokens (num_tokens=512 avail_mem=55.85 GB):  48%|████▊     | 28/58 [00:03<00:02, 10.98it/s]Capturing num tokens (num_tokens=480 avail_mem=55.87 GB):  48%|████▊     | 28/58 [00:03<00:02, 10.98it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.87 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.14it/s]Capturing num tokens (num_tokens=448 avail_mem=55.86 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.14it/s]Capturing num tokens (num_tokens=416 avail_mem=55.86 GB):  52%|█████▏    | 30/58 [00:03<00:02, 11.14it/s]Capturing num tokens (num_tokens=416 avail_mem=55.86 GB):  55%|█████▌    | 32/58 [00:03<00:02, 11.69it/s]Capturing num tokens (num_tokens=384 avail_mem=55.85 GB):  55%|█████▌    | 32/58 [00:03<00:02, 11.69it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.84 GB):  55%|█████▌    | 32/58 [00:03<00:02, 11.69it/s]Capturing num tokens (num_tokens=352 avail_mem=55.84 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.88it/s]Capturing num tokens (num_tokens=320 avail_mem=55.83 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.88it/s]Capturing num tokens (num_tokens=288 avail_mem=55.83 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.88it/s]

    Capturing num tokens (num_tokens=288 avail_mem=55.83 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.18it/s]Capturing num tokens (num_tokens=256 avail_mem=55.82 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.18it/s]Capturing num tokens (num_tokens=240 avail_mem=55.81 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.18it/s]Capturing num tokens (num_tokens=240 avail_mem=55.81 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.43it/s]Capturing num tokens (num_tokens=224 avail_mem=55.81 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.43it/s]

    Capturing num tokens (num_tokens=208 avail_mem=55.80 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.43it/s]Capturing num tokens (num_tokens=208 avail_mem=55.80 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.53it/s]Capturing num tokens (num_tokens=192 avail_mem=55.80 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.53it/s]Capturing num tokens (num_tokens=176 avail_mem=55.79 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.53it/s]Capturing num tokens (num_tokens=176 avail_mem=55.79 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.67it/s]Capturing num tokens (num_tokens=160 avail_mem=55.78 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.67it/s]

    Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  72%|███████▏  | 42/58 [00:04<00:01, 13.67it/s]Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:04<00:00, 14.70it/s]Capturing num tokens (num_tokens=128 avail_mem=55.77 GB):  76%|███████▌  | 44/58 [00:04<00:00, 14.70it/s]Capturing num tokens (num_tokens=112 avail_mem=55.77 GB):  76%|███████▌  | 44/58 [00:04<00:00, 14.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.77 GB):  79%|███████▉  | 46/58 [00:04<00:00, 14.51it/s]Capturing num tokens (num_tokens=96 avail_mem=55.76 GB):  79%|███████▉  | 46/58 [00:04<00:00, 14.51it/s] Capturing num tokens (num_tokens=80 avail_mem=55.75 GB):  79%|███████▉  | 46/58 [00:04<00:00, 14.51it/s]Capturing num tokens (num_tokens=80 avail_mem=55.75 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.28it/s]Capturing num tokens (num_tokens=64 avail_mem=55.75 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.28it/s]Capturing num tokens (num_tokens=48 avail_mem=55.74 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.28it/s]Capturing num tokens (num_tokens=32 avail_mem=55.73 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.28it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.72 GB):  83%|████████▎ | 48/58 [00:04<00:00, 15.28it/s]Capturing num tokens (num_tokens=28 avail_mem=55.72 GB):  90%|████████▉ | 52/58 [00:04<00:00, 20.26it/s]Capturing num tokens (num_tokens=24 avail_mem=55.72 GB):  90%|████████▉ | 52/58 [00:04<00:00, 20.26it/s]Capturing num tokens (num_tokens=20 avail_mem=55.71 GB):  90%|████████▉ | 52/58 [00:04<00:00, 20.26it/s]Capturing num tokens (num_tokens=16 avail_mem=55.71 GB):  90%|████████▉ | 52/58 [00:04<00:00, 20.26it/s]Capturing num tokens (num_tokens=12 avail_mem=55.70 GB):  90%|████████▉ | 52/58 [00:04<00:00, 20.26it/s]

    Capturing num tokens (num_tokens=12 avail_mem=55.70 GB):  97%|█████████▋| 56/58 [00:04<00:00, 20.57it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:04<00:00, 20.57it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:04<00:00, 20.57it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:04<00:00, 11.71it/s]


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
    Generated text:  Ryan and I am currently a sophomore at the University of Chicago majoring in Biochemistry. I have a great interest in both math and biology. I am a natural verbal thinker and have great problem solving skills that make me an excellent candidate for this role. I excel in both verbal and quantitative reasoning and am able to communicate effectively and clearly with both levels of proficiency.
    There is a common belief that people are too busy to learn math and biology. For most people, the roles that math and biology can play in a career can make them very busy. There are a variety of ways to bring math and biology to the table, such as teaching
    ===============================
    Prompt: The president of the United States is
    Generated text:  a government official who is the head of the executive branch of the federal government, the highest governing body of the United States. He or she is also known as the first in line to the presidency. Presidents are elected by the people of the United States, and they serve 4-year terms. Presidents are subject to impeachment, and they may have to resign if they are found to have committed serious crimes.
    
    Based on this passage, who becomes president in the united states? Let's think fast. slow first. Then, let's formulate the best answer.
    
    Based on the information provided in the passage, the president of the United States becomes president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris was founded in 843. Paris was the capital of France for over a thousand years. The name Paris means "peace". The site of the city of Paris is on the Île de la Cité. In the center of Paris, where we find the famous Eiffel Tower and the Louvre, there is the site of the ancient city of Paris. There is a museum and the most famous building of Paris which is the Arc de Triomphe. The Eiffel Tower has been a symbol of Paris for almost 100 years. The Louvre is also one of the most famous museums
    ===============================
    Prompt: The future of AI is
    Generated text:  one of cooperation. The partners of the AI2019 Conference agreed to cooperate on a program of joint research, and they decided to make a special category for organizations with a broad and open-minded approach and a strong commitment to the unity of nations.
    
    Please answer the following question:
    
    There are 300 participants at the conference. What is the maximum number of organizations that can be selected for the special category if each organization can be selected at most once?
    
    To determine the maximum number of organizations that can be selected for the special category, we need to ensure that each organization is included at most once, and there must be at least


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


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is the seat of government, administration, and culture for the country. Paris is known for its rich history, art, and cuisine, and is a major tourist destination. The city is also home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and captivate people around the world. The city is also home to many important organizations and institutions, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to disease diagnosis and prediction. As AI technology continues to improve, we can expect to see even more advanced applications in healthcare.
    
    2. AI in manufacturing: AI is already being used to optimize production processes, reduce costs, and improve quality. As AI technology continues to evolve, we can expect to see even more applications in manufacturing.
    
    3. AI in finance: AI is already being used
    


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
    Generated text:  [insert character's name]. I'm a [insert age, gender, occupation, etc.]. I currently live in [insert location]. I am [insert profession]. I enjoy [insert hobby or interest that interests me]. I am passionate about [insert why I am passionate about it]. I enjoy [insert why I am passionate about it]. I am a [insert trait or quality that sets me apart]. I am a [insert personality type]. I am a [insert something about what I believe or think]. I am a [insert what I really like to do]. I am a [insert my best piece of advice]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest and most populous city in France, located in the Île de la Cité area of the Loire river valley in the north-central part of the country. It is the cultural, financial, industrial, and intellectual heart of the French Republic and is a major center for French politics, journalism, music, fashion, and much more. The city is also home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. Paris is the world’s 6th most populous city and the fourth most populous metropolitan area, after New York City, Los
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly likely to involve a wide range of technological advancements and applications. Here are some potential trends that could shape the AI landscape in the coming years:
    
    1. Increased automation and automation: AI will continue to advance and become more integrated into various industries, leading to an increasing number of automated tasks and processes. This will require a shift in human roles and responsibilities, potentially leading to job displacement in some areas.
    
    2. Biometric security: The use of biometric security methods such as facial recognition and fingerprints will become more prevalent, with AI being used to analyze and identify these features. This could lead to more secure and efficient authentication systems, but also


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

    /an

     [

    occupation

    ].

     I

    'm

     a

    /an

     [

    profession

    ].

     I

     enjoy

     [

    reason

     for

     love

    /h

    obby

    ]

     and

     I

    'm

     always

     looking

     for

     [

    goal

    ].

     I

     can

     adapt

     to

     any

     situation

    ,

     but

     I

    'm

     most

     comfortable

     in

     [

    location

    ].

     I

    'm

     [

    age

    ]

     years

     old

     and

     I

     have

     a

     [

    degree

    /

    education

    ]

     in

     [

    field

     of

     study

    ].

     I

     am

     passionate

     about

     [

    reason

     for

     passion

    ],

     and

     I

     believe

     that

     [

    reason

     for

     passion

    ]

     will

     help

     me

     achieve

     my

     goals

    .

     I

    'm

     a

    /an

     [

    skill

    /

    ability

    ]

     who

     is

     always

     looking

     for

     new

     challenges

    .

     I

    'm

     [

    person

    ality

    ].

     I

     have

     [

    number

     of

     years

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     capital

     of

     France

    ?

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

     and

     continues

     to

     evolve

     rapidly

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

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     it

    's

     becoming

     increasingly

     important

     to

     consider

     the

     ethical

     implications

     of

     their

     decisions

     and

     actions

    .

     This

     includes

     issues

     such

     as

     bias

    ,

     transparency

    ,

     accountability

    ,

     and

     the

     potential

     for

     misuse

    .
    


    2

    .

     Deep

     learning

     and

     neural

     networks

    :

     Deep

     learning

    ,

     a

     subset

     of

     machine

     learning

     that

     uses

     neural

     networks

     to

     perform

     tasks

     such

     as

     image

     recognition

     and

     natural

     language

     processing

    ,

     is

     set

     to

     become

     more

     powerful

     and

     efficient

     in

     the

     years

     ahead

    .
    


    3

    .

     AI

    -powered

     healthcare

    :

     AI

     can

     be

     used

     to

     improve

     patient

     care

     in

     healthcare

    ,

    



```python
llm.shutdown()
```
