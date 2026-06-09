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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.61it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.61it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.61it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.61it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.55it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.55it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.92it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.92it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.92it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.92it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.92it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.47it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.47it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.85it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.85it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.85it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.85it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.85it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 19.85it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 24.59it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 30.64it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 36.39it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 36.39it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 38.18it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 38.18it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 38.18it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 38.18it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 38.18it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 38.18it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 35.77it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 35.77it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 35.77it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 35.77it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 35.77it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 35.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 37.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.94 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.91 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:07,  7.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   5%|▌         | 3/58 [00:00<00:07,  7.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:06,  8.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   7%|▋         | 4/58 [00:00<00:06,  8.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:06,  8.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.90 GB):   9%|▊         | 5/58 [00:00<00:06,  8.67it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):   9%|▊         | 5/58 [00:00<00:06,  8.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.89 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.54it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=52.88 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.88 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.87 GB):  19%|█▉        | 11/58 [00:01<00:04, 11.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.87 GB):  19%|█▉        | 11/58 [00:01<00:04, 11.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  19%|█▉        | 11/58 [00:01<00:04, 11.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.87 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.87 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.86 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.86 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.86 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.65it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=52.86 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.86 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.85 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.85 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.85 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.85 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.74it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=52.83 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.74it/s]Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.74it/s] Capturing num tokens (num_tokens=960 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]Capturing num tokens (num_tokens=896 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]Capturing num tokens (num_tokens=832 avail_mem=52.84 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]Capturing num tokens (num_tokens=768 avail_mem=52.83 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]

    Capturing num tokens (num_tokens=768 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.23it/s]Capturing num tokens (num_tokens=704 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.23it/s]Capturing num tokens (num_tokens=640 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.23it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.23it/s]Capturing num tokens (num_tokens=576 avail_mem=52.83 GB):  48%|████▊     | 28/58 [00:02<00:01, 18.26it/s]Capturing num tokens (num_tokens=512 avail_mem=52.81 GB):  48%|████▊     | 28/58 [00:02<00:01, 18.26it/s]

    Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  48%|████▊     | 28/58 [00:02<00:01, 18.26it/s]Capturing num tokens (num_tokens=480 avail_mem=52.83 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.11it/s]Capturing num tokens (num_tokens=448 avail_mem=52.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.11it/s]

    Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.11it/s]Capturing num tokens (num_tokens=416 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.35it/s]Capturing num tokens (num_tokens=384 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.35it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  55%|█████▌    | 32/58 [00:02<00:02, 10.35it/s]Capturing num tokens (num_tokens=352 avail_mem=52.82 GB):  59%|█████▊    | 34/58 [00:02<00:02,  8.90it/s]Capturing num tokens (num_tokens=320 avail_mem=52.81 GB):  59%|█████▊    | 34/58 [00:02<00:02,  8.90it/s]

    Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  59%|█████▊    | 34/58 [00:03<00:02,  8.90it/s]Capturing num tokens (num_tokens=288 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:03<00:02,  8.46it/s]Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  62%|██████▏   | 36/58 [00:03<00:02,  8.46it/s]

    Capturing num tokens (num_tokens=256 avail_mem=52.81 GB):  64%|██████▍   | 37/58 [00:03<00:02,  8.36it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  64%|██████▍   | 37/58 [00:03<00:02,  8.36it/s]Capturing num tokens (num_tokens=240 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:02,  8.01it/s]Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  66%|██████▌   | 38/58 [00:03<00:02,  8.01it/s]

    Capturing num tokens (num_tokens=224 avail_mem=52.80 GB):  67%|██████▋   | 39/58 [00:03<00:02,  8.14it/s]Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  67%|██████▋   | 39/58 [00:03<00:02,  8.14it/s]Capturing num tokens (num_tokens=208 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:02,  8.46it/s]Capturing num tokens (num_tokens=192 avail_mem=52.79 GB):  69%|██████▉   | 40/58 [00:03<00:02,  8.46it/s]

    Capturing num tokens (num_tokens=192 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:01,  8.76it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  71%|███████   | 41/58 [00:03<00:01,  8.76it/s]Capturing num tokens (num_tokens=176 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:03<00:01,  9.05it/s]Capturing num tokens (num_tokens=160 avail_mem=52.79 GB):  72%|███████▏  | 42/58 [00:03<00:01,  9.05it/s]Capturing num tokens (num_tokens=144 avail_mem=52.78 GB):  72%|███████▏  | 42/58 [00:03<00:01,  9.05it/s]

    Capturing num tokens (num_tokens=144 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:04<00:01,  9.82it/s]Capturing num tokens (num_tokens=128 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:04<00:01,  9.82it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  76%|███████▌  | 44/58 [00:04<00:01,  9.82it/s]Capturing num tokens (num_tokens=112 avail_mem=52.78 GB):  79%|███████▉  | 46/58 [00:04<00:01, 10.67it/s]Capturing num tokens (num_tokens=96 avail_mem=52.77 GB):  79%|███████▉  | 46/58 [00:04<00:01, 10.67it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  79%|███████▉  | 46/58 [00:04<00:01, 10.67it/s]Capturing num tokens (num_tokens=80 avail_mem=52.77 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.70it/s]Capturing num tokens (num_tokens=64 avail_mem=52.76 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.70it/s]Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  83%|████████▎ | 48/58 [00:04<00:00, 11.70it/s]

    Capturing num tokens (num_tokens=48 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.16it/s]Capturing num tokens (num_tokens=32 avail_mem=52.76 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.16it/s]Capturing num tokens (num_tokens=28 avail_mem=52.75 GB):  86%|████████▌ | 50/58 [00:04<00:00, 12.16it/s]Capturing num tokens (num_tokens=28 avail_mem=52.75 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.64it/s]Capturing num tokens (num_tokens=24 avail_mem=52.75 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.64it/s]Capturing num tokens (num_tokens=20 avail_mem=52.75 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.64it/s]Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.64it/s]

    Capturing num tokens (num_tokens=16 avail_mem=52.75 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.06it/s]Capturing num tokens (num_tokens=12 avail_mem=52.74 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.06it/s]Capturing num tokens (num_tokens=8 avail_mem=52.74 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.06it/s] Capturing num tokens (num_tokens=4 avail_mem=52.73 GB):  95%|█████████▍| 55/58 [00:04<00:00, 17.06it/s]Capturing num tokens (num_tokens=4 avail_mem=52.73 GB): 100%|██████████| 58/58 [00:04<00:00, 19.31it/s]Capturing num tokens (num_tokens=4 avail_mem=52.73 GB): 100%|██████████| 58/58 [00:04<00:00, 11.88it/s]


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
    Generated text:  Tobi, and I'm a graphic designer and illustrator in Portland, Oregon. I specialize in brand identity and logo design, and have worked with clients including the NBA, Hyundai, and Samsung.
    I'm currently working on a branding project for a tech startup. I want to create a new logo that captures the essence of the startup and its unique selling proposition. Can you help me design a logo that effectively represents the brand?
    Sure, I'd be happy to help you design a logo for your tech startup! To get started, can you provide me with some more information about the company and its mission statement? This will help me create a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political position, and the Senate is the legislative branch of government. The first sitting president of the United States was George Washington. By the 1850s, the United States was divided into three states. In 1861, the final issue of tension over slavery began. The following year, the Civil War broke out. After the war, the president of the United States had to seek a second term. He also faced a series of difficulties while in office, including the 1865 election of Abraham Lincoln as the new president. Since then, the president of the United States is the incumbent. The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Yes, I'm familiar with that. Paris is the capital of France, also known as the "city of love" and the "city of light." It is located in the Loire Valley and is the largest city in the European Union by population. The city is famous for its rich history, beautiful architecture, and beautiful parks, including the Palace of Versailles, the Louvre Museum, and the Champs-Élysées. Paris is a major tourist destination and a popular destination for film, fashion, and other cultural events. Its historic center, the famous Eiffel Tower, and its lively nightlife are also popular
    ===============================
    Prompt: The future of AI is
    Generated text:  changing, and it’s important for businesses to understand it and how to prepare for it.
    AI is a term that is often used to describe all forms of intelligent technology, including machine learning, natural language processing, and robotics. However, in the context of artificial intelligence, there are two main types: the human-machine interaction, and the artificial intelligence. The human-machine interaction is the technology that enables the interaction between humans and machines, while the artificial intelligence is the technology that enables machines to learn and perform tasks on their own.
    In the human-machine interaction, the AI system learns and adapts to the human needs and provides them with a more


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and is home to many famous French artists, writers, and musicians. The city is also known for its rich history and cultural heritage, including its medieval architecture and its role in the French Revolution. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage, making it a popular destination for tourists and locals alike.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in our daily lives. This could lead to the creation of more efficient and cost-effective systems, as well as the creation of new jobs that require little or no human intervention.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, we can expect to see increased concerns about privacy and security. This could lead to new regulations
    


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
    Generated text:  [Name], and I'm a [Job Title] with [Company Name]. I'm a professional with a high level of [Skill or Ability], and I'm always looking for new challenges and opportunities. I have a keen eye for detail and am always looking for ways to improve my work. I'm a team player, and I thrive in collaboration with others. I have a love for creativity and a passion for learning, and I'm always eager to keep up with the latest trends and technologies. I'm a problem solver who is always looking for ways to enhance the quality of my work. I'm always looking for ways to improve my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Please confirm, even if it's just a link to a source, if it's possible to create an FAQ type page on this site, linking to the full statement and URL from which the page can be accessed.
    
    Sure, please just give me the link. I would prefer not to create a new page just for this. 
    
    Here are some examples:
    - A FAQ for a country
    - A FAQ for a city
    - A FAQ for a company
    - A FAQ for a service
    - A FAQ for a product
    - A FAQ for a music artist
    - A FAQ for a sports team
    - A FAQ
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and difficult to predict, but it is likely to continue to evolve rapidly. Here are some possible future trends in AI:
    
    1. Increased automation and efficiency: With the increasing demand for automation, AI will become more prevalent in manufacturing, transportation, and other industries. This will lead to increased efficiency and cost savings for businesses.
    
    2. Greater integration of AI: AI will become more integrated with other technologies such as sensors, blockchain, and IoT, leading to more advanced machine learning algorithms and deeper insights into human behavior.
    
    3. AI-powered healthcare: AI will play a significant role in healthcare, with more personalized treatment plans based on individual patient data


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

    Character

    's

     name

    ].

     I

     am

     a

     [

    age

    ]

     year

     old

     [

    profession

    ].

     I

    'm

     a

     [

    occupation

    ].

     I

     love

     [

    reason

     for

     love

     or

     passion

    ].

     I

    'm

     passionate

     about

     [

    what

     interests

     you

    ]

     because

     [

    exc

    use

     for

     the

     interest

    ].

     I

     enjoy

     [

    activities

     that

     interest

     me

    ],

     which

     make

     me

     [

    your

    self

    ].

     I

     spend

     my

     days

     [

    activities

    ],

     which

     make

     me

     [

    your

    self

    ].

     If

     you

     want

     to

     talk

     to

     me

    ,

     please

     call

     me

     [

    character

    's

     name

    ].

     I

     am

     a

     [

    character

    's

     profession

    ].

     I

     look

     forward

     to

     meeting

     you

     all

    .

     Thank

     you

    .

     How

     old

     is

     the

     character

     you

     are

     addressing

    ?

     And

     what

     profession

     do

     they

     have

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     history

     and

     cultural

     heritage

    ,

     including

     its

     historic

     center

     and

     many

     notable

     museums

     and

     institutions

    .

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

     and

     a

     vibrant

     nightlife

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

     art

     and

     architecture

    ,

     including

     the

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     a

     cosm

    opolitan

     city

     with

     a

     strong

     sense

     of

     French

     identity

     and

     a

     reputation

     for

     innovation

     and

     creativity

    .

     Its

     picturesque

     landscapes

    ,

     delicious

     cuisine

    ,

     and

     vibrant

     culture

     make

     it

     an

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     emerging

     trends

     and

     advancements

    .

     Here

     are

     some

     possible

     trends

     in

     the

     AI

     industry

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Medical

     Diagnosis

    :

     AI

     can

     be

     used

     to

     improve

     the

     accuracy

     of

     medical

     diagnoses

     by

     analyzing

     large

     amounts

     of

     medical

     data

     and

     identifying

     patterns

     that

     may

     have

     been

     missed

     by

     human

     doctors

    .

     This

     can

     lead

     to

     earlier

     detection

     of

     diseases

     and

     more

     effective

     treatment

     options

    .
    


    2

    .

     Enhanced

     Personal

    ized

     Medicine

    :

     AI

     can

     be

     used

     to

     analyze

     genetic

     data

     and

     individual

     health

     information

     to

     develop

     personalized

     treatment

     plans

    .

     This

     can

     lead

     to

     more

     effective

     and

     efficient

     treatments

     for

     individual

     patients

    .
    


    3

    .

     Autonomous

     and

     Self

    -

    Driving

     Vehicles

    :

     AI

     is

     already

     being

     used

    



```python
llm.shutdown()
```
