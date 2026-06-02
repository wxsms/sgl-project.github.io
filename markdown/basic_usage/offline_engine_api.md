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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:20,  2.54it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:20,  2.54it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.54it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  6.06it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  6.06it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:07,  6.06it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:07,  6.06it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  8.58it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  8.58it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:05,  8.58it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:05,  8.58it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:05,  8.58it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 12.34it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 12.34it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 12.34it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 12.34it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.08it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 22.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 22.29it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 22.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 22.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 22.29it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 22.29it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 27.92it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 27.92it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 27.92it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 27.92it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 27.92it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:06<00:00, 27.92it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:06<00:00, 27.92it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:06<00:00, 33.94it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:06<00:00, 38.46it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 43.05it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 43.05it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 43.05it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 43.05it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 43.05it/s] 

    Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 43.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.87 GB):   2%|▏         | 1/58 [00:00<00:08,  6.70it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.84 GB):   2%|▏         | 1/58 [00:00<00:08,  6.70it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.84 GB):   3%|▎         | 2/58 [00:00<00:09,  6.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.83 GB):   3%|▎         | 2/58 [00:00<00:09,  6.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.83 GB):   5%|▌         | 3/58 [00:00<00:08,  6.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.83 GB):   5%|▌         | 3/58 [00:00<00:08,  6.65it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.83 GB):   7%|▋         | 4/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.83 GB):   7%|▋         | 4/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.83 GB):   9%|▊         | 5/58 [00:00<00:07,  7.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.82 GB):   9%|▊         | 5/58 [00:00<00:07,  7.18it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.82 GB):  10%|█         | 6/58 [00:00<00:06,  7.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.82 GB):  10%|█         | 6/58 [00:00<00:06,  7.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.81 GB):  10%|█         | 6/58 [00:00<00:06,  7.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.81 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.81 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.68it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.81 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.81 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.80 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.80 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.46it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.80 GB):  21%|██        | 12/58 [00:01<00:04, 10.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.80 GB):  21%|██        | 12/58 [00:01<00:04, 10.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.80 GB):  21%|██        | 12/58 [00:01<00:04, 10.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.80 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.79 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.08it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.79 GB):  24%|██▍       | 14/58 [00:01<00:03, 11.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.79 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.79 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.78 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.12it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=55.78 GB):  31%|███       | 18/58 [00:01<00:03, 12.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.78 GB):  31%|███       | 18/58 [00:01<00:03, 12.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.78 GB):  31%|███       | 18/58 [00:01<00:03, 12.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.76 GB):  31%|███       | 18/58 [00:01<00:03, 12.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.76 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.92it/s]Capturing num tokens (num_tokens=960 avail_mem=55.77 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.92it/s] Capturing num tokens (num_tokens=896 avail_mem=55.77 GB):  36%|███▌      | 21/58 [00:01<00:02, 15.92it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.77 GB):  36%|███▌      | 21/58 [00:02<00:02, 15.92it/s]Capturing num tokens (num_tokens=832 avail_mem=55.77 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=768 avail_mem=55.76 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=704 avail_mem=55.76 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.20it/s]Capturing num tokens (num_tokens=704 avail_mem=55.76 GB):  45%|████▍     | 26/58 [00:02<00:01, 16.97it/s]Capturing num tokens (num_tokens=640 avail_mem=55.76 GB):  45%|████▍     | 26/58 [00:02<00:01, 16.97it/s]

    Capturing num tokens (num_tokens=576 avail_mem=55.76 GB):  45%|████▍     | 26/58 [00:02<00:01, 16.97it/s]Capturing num tokens (num_tokens=576 avail_mem=55.76 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.10it/s]Capturing num tokens (num_tokens=512 avail_mem=55.74 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.10it/s]Capturing num tokens (num_tokens=480 avail_mem=55.76 GB):  48%|████▊     | 28/58 [00:02<00:01, 17.10it/s]Capturing num tokens (num_tokens=480 avail_mem=55.76 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.16it/s]Capturing num tokens (num_tokens=448 avail_mem=55.75 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.16it/s]

    Capturing num tokens (num_tokens=416 avail_mem=55.75 GB):  52%|█████▏    | 30/58 [00:02<00:01, 17.16it/s]Capturing num tokens (num_tokens=416 avail_mem=55.75 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.67it/s]Capturing num tokens (num_tokens=384 avail_mem=55.75 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.67it/s]Capturing num tokens (num_tokens=352 avail_mem=55.74 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.67it/s]Capturing num tokens (num_tokens=320 avail_mem=55.74 GB):  55%|█████▌    | 32/58 [00:02<00:01, 17.67it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.74 GB):  60%|██████    | 35/58 [00:02<00:01, 18.57it/s]Capturing num tokens (num_tokens=288 avail_mem=55.74 GB):  60%|██████    | 35/58 [00:02<00:01, 18.57it/s]Capturing num tokens (num_tokens=256 avail_mem=55.73 GB):  60%|██████    | 35/58 [00:02<00:01, 18.57it/s]Capturing num tokens (num_tokens=256 avail_mem=55.73 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.92it/s]Capturing num tokens (num_tokens=240 avail_mem=55.73 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.92it/s]Capturing num tokens (num_tokens=224 avail_mem=55.73 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.92it/s]

    Capturing num tokens (num_tokens=208 avail_mem=55.72 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.92it/s]Capturing num tokens (num_tokens=208 avail_mem=55.72 GB):  69%|██████▉   | 40/58 [00:02<00:00, 19.11it/s]Capturing num tokens (num_tokens=192 avail_mem=55.72 GB):  69%|██████▉   | 40/58 [00:02<00:00, 19.11it/s]Capturing num tokens (num_tokens=176 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:03<00:00, 19.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.32 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.51it/s]Capturing num tokens (num_tokens=160 avail_mem=58.32 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.51it/s]Capturing num tokens (num_tokens=144 avail_mem=58.32 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.51it/s]Capturing num tokens (num_tokens=128 avail_mem=58.32 GB):  72%|███████▏  | 42/58 [00:03<00:01, 15.51it/s]Capturing num tokens (num_tokens=128 avail_mem=58.32 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.51it/s]Capturing num tokens (num_tokens=112 avail_mem=58.31 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.51it/s]Capturing num tokens (num_tokens=96 avail_mem=58.31 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.51it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.31 GB):  78%|███████▊  | 45/58 [00:03<00:00, 17.51it/s]Capturing num tokens (num_tokens=80 avail_mem=58.31 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.69it/s]Capturing num tokens (num_tokens=64 avail_mem=58.30 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.69it/s]Capturing num tokens (num_tokens=48 avail_mem=58.30 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.69it/s]Capturing num tokens (num_tokens=32 avail_mem=58.30 GB):  83%|████████▎ | 48/58 [00:03<00:00, 19.69it/s]Capturing num tokens (num_tokens=32 avail_mem=58.30 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.65it/s]Capturing num tokens (num_tokens=28 avail_mem=58.29 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.65it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.29 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.65it/s]Capturing num tokens (num_tokens=20 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:03<00:00, 20.65it/s]Capturing num tokens (num_tokens=20 avail_mem=58.28 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.64it/s]Capturing num tokens (num_tokens=16 avail_mem=58.28 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.64it/s]Capturing num tokens (num_tokens=12 avail_mem=58.28 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.64it/s]Capturing num tokens (num_tokens=8 avail_mem=58.28 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.64it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.28 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.07it/s]Capturing num tokens (num_tokens=4 avail_mem=58.27 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.07it/s]Capturing num tokens (num_tokens=4 avail_mem=58.27 GB): 100%|██████████| 58/58 [00:03<00:00, 15.17it/s]


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
    Generated text:  Vladimir.
    I'm a software developer and I'm passionate about my work.
    I have been working in the field for over a decade and am currently working on a project for my client that I'm very excited about. Can you tell me about the project and the goals you have for it? And do you have any advice for someone who is looking for a job in this industry?
    Certainly! Can you provide more details about the project you are working on and the goals you have for it? Also, what kind of job opportunities are available in the field of software development? I am here to provide support and guidance to anyone in need. 
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ( ) position, and the vice president is a ( ) position.
    A. Standing B. Position C. Official D. Military
    Answer:
    
    AC
    
    Which of the following are energy sources?
    A. Coal
    B. Natural gas
    C. Oil
    D. Hydroelectric power
    E. Solar energy
    Answer:
    
    ABCE
    
    Which of the following groups of words are all nouns?
    A. Horse, trunk
    B. Butterfly, life
    C. House, breath
    D. Spring, day
    Answer:
    
    A and B
    
    Please select the one that does not belong to the same category as the others:
    A.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. How does the urban design of Paris contribute to its cultural significance?
    The urban design of Paris has a significant impact on its cultural significance in several ways:
    
    1. Functionality: The urban design of Paris is designed to function efficiently, with a focus on maximizing green spaces and reducing traffic congestion. This design philosophy has helped to create a livable and walkable city that is well-suited for the needs of its citizens.
    
    2. Visual appeal: The urban design of Paris is a masterpiece of design, with a combination of architectural styles and functional elements that create a visually stunning and sophisticated cityscape. This design philosophy has helped to create
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about automation, but also about integration. AI is about integration of different kinds of AI across the world, such as AI that supports different kinds of devices, AI that supports different kinds of sensors, and so on. The idea behind integration of AI is to create a unified machine that can respond to any kind of problem, no matter what kind of problem is faced. AI is not just about creating a machine that can do the job, but it is also about creating a machine that can be integrated with other machines in order to create new problems. It is the responsibility of the developers to ensure that AI is integrated in the right way


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many world-renowned museums, including the Musée d'Orsay and the Musée d'Art Moderne. Paris is a bustling city with a rich history and a diverse population, making it a popular tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. Its status as the capital of France has made it a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that were previously done by humans. This could lead to the creation of more efficient and cost-effective systems that can perform a wide range of tasks, from manufacturing to healthcare.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be an increasing need to protect the privacy and security of personal data. This could lead to the development
    


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
    Generated text:  [Name] and I am [Age], a [job title]. I am [career path]. My main focus is [career path], and I enjoy [career path]. I was born and raised in [city or country], and I have always been [general sense of person]. I have a passion for [interest or hobby], and I enjoy [career path] because [reason why]. I am always looking for new challenges, and I am eager to learn and grow as a [career path]. I have always been [general sense of person], and I have always been [general sense of person]. I am a [general sense
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, Notre Dame Cathedral, and lively vibrant culture. It is also a popular tourist destination, hosting many cultural events and festivals throughout the year. The city is home to several international organizations such as UNESCO and the French Academy of Sciences, and is a major commercial and financial center in Europe. It's known for its rich history, beautiful architecture, and lively nightlife. Paris is a city of contrasts, and the city's cultural and historical sites are a must-visit for anyone visiting. It's a city that's steeped in history, and continues to thrive as a major cultural and business center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and full of promise, but it also presents several challenges. One of the most significant trends is the integration of AI into everyday life, from the devices we use every day to the autonomous vehicles that could change the way we move and interact with the world around us. AI is also expected to play an increasingly important role in healthcare, as AI-powered tools can analyze data and help doctors diagnose and treat illnesses more accurately and efficiently.
    
    Another trend is the increasing focus on ethical AI. AI systems are often designed and implemented with a focus on maximizing profit and minimizing human error, which can lead to unintended consequences if not properly managed. The development of


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

     [

    Occup

    ation

    ]

     by

     day

    ,

     [

    Prof

    ession

    ]

     by

     night

    .

     I

     enjoy

     [

    Just

    ify

     your

     occupation

     or

     profession

     here

    .

    ].

     In

     my

     spare

     time

    ,

     I

     like

     to

     [

    What

     I

     do

     outside

     of

     work

     or

     play

    ?

    ].

     What

    's

     your

     personality

     like

    ,

     and

     what

     do

     you

     think

     makes

     you

     unique

    ?

     That

    's

     all

     I

     can

     say

     right

     now

    ,

     but

     I

    'll

     be

     back

     with

     more

     information

     in

     the

     future

    .

     Good

    night

    !

     To

     start

    ,

     I

     would

     like

     to

     ask

     what

     your

     name

     is

     and

     what

     profession

     you

     have

    .


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    Occup

    ation

    ]

     by

     day

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     where

     the

     French

     Revolution

     took

     place

    .

     
    


    The

     French

     Revolution

     began

     in

     Paris

     in

     the

     early

     

    1

    8

    th

     century

    .

     The

     Revolution

     ended

     with

     the

     death

     of

     King

     Louis

     XVI

     and

     the

     establishment

     of

     the

     French

     Republic

     in

     

    1

    7

    9

    2

    .

     In

     

    1

    7

    9

    3

    ,

     the

     French

     and

     the

     French

     Revolution

     were

     severely

     impacted

     by

     a

     series

     of

     bombings

     in

     the

     city

    .

     The

     city

    's

     political

     climate

     remained

     tense

     and

     the

     Revolution

    's

     impact

     on

     Paris

     remained

     felt

     for

     years

     to

     come

    .

     
    


    The

     French

     Revolution

     took

     place

     in

     

    1

    7

    8

    9

     and

     lasted

     for

     seven

     years

    .

     The

     Revolution

     had

     a

     significant

     impact

     on

     the

     French

     people

    ,

     as

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     highly

     interconnected

     and

     dynamic

     one

    ,

     driven

     by

     ongoing

     technological

     advancements

    ,

     changes

     in

     societal

     norms

    ,

     and

     the

     ongoing

     evolution

     of

     the

     human

     workforce

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     collaboration

     and

     communication

     between

     humans

     and

     machines

    :

     As

     AI

     continues

     to

     improve

     and

     become

     more

     integrated

     with

     other

     technologies

    ,

     it

     is

     likely

     to

     play

     a

     more

     significant

     role

     in

     human

     decision

    -making

     and

     communication

    .

     This

     could

     lead

     to

     new

     ways

     of

     working

    ,

     such

     as

     AI

    -ass

    isted

     virtual

     teams

     and

     virtual

     reality

    -based

     collaboration

    .
    


    2

    .

     Improved

     emotional

     intelligence

     and

     empathy

    :

     As

     AI

     continues

     to

     learn

     and

     adapt

    ,

     it

     is

     likely

     to

     become

     more

     capable

     of

     recognizing

     and

     responding

    



```python
llm.shutdown()
```
