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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.50it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.50it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.50it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:20,  2.50it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.66it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.66it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  6.66it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.66it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.66it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.30it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.30it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.30it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.30it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 12.85it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 12.85it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 17.00it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 17.00it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 17.00it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 17.00it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 17.00it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.02it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.02it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.02it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.02it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.02it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.02it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:00, 26.79it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 33.28it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 33.28it/s]

    Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:00, 33.28it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:00, 33.28it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:06<00:00, 33.28it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:06<00:00, 33.28it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:06<00:00, 33.28it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 38.92it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]

    Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 43.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.46 GB):   2%|▏         | 1/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.43 GB):   2%|▏         | 1/58 [00:00<00:07,  7.53it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.43 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  8.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.08it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  8.08it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.41 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.24it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 11.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 11.50it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 11.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.86it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.37 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.22it/s]

    Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.22it/s] Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.68it/s]Capturing num tokens (num_tokens=896 avail_mem=58.36 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.68it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.68it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.68it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.33it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.33it/s]

    Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.33it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.33it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.91it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.91it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.91it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.91it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.08it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.08it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.08it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 22.08it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.79it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.79it/s]Capturing num tokens (num_tokens=288 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.79it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.79it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.10it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.10it/s]Capturing num tokens (num_tokens=224 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.10it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.10it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 23.90it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 23.90it/s]Capturing num tokens (num_tokens=176 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 23.90it/s]

    Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:02<00:00, 23.90it/s]Capturing num tokens (num_tokens=160 avail_mem=58.31 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.13it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.13it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.13it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.13it/s]Capturing num tokens (num_tokens=112 avail_mem=58.30 GB):  79%|███████▉  | 46/58 [00:02<00:00, 24.32it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 24.32it/s] Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 24.32it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  79%|███████▉  | 46/58 [00:02<00:00, 24.32it/s]Capturing num tokens (num_tokens=64 avail_mem=58.29 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.62it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.62it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.62it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.62it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.78it/s]Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.78it/s]Capturing num tokens (num_tokens=20 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.78it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  90%|████████▉ | 52/58 [00:03<00:00, 24.78it/s]Capturing num tokens (num_tokens=16 avail_mem=58.27 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.32it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.32it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.32it/s] Capturing num tokens (num_tokens=4 avail_mem=58.26 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.32it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:03<00:00, 25.77it/s]Capturing num tokens (num_tokens=4 avail_mem=58.26 GB): 100%|██████████| 58/58 [00:03<00:00, 18.30it/s]


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
    Generated text:  Rina and I am a student at George Mason University in Washington, D. C. I am studying Statistics and Economics and have been using Excel for some time now. I am interested in learning more about data visualization and I am interested in learning how to use Excel to generate graphs and charts. Can you please provide me with information on the various methods to create charts and graphs in Excel, the different chart types available in Excel, and the different ways to add annotations, labels, and data labels to charts in Excel? Additionally, I would like to know about any statistical software that can be used with Excel and how it can be used for
    ===============================
    Prompt: The president of the United States is
    Generated text:  a government official, and the position is filled by the head of the executive branch. Which position would a president be in charge of if the president was a teacher? Select the statement that correctly identifies the position of the president in this scenario.
    
    A. A teacher is in charge of the president.
    B. A teacher is in charge of the United States government.
    C. A teacher is in charge of the president of the United States.
    D. A teacher is in charge of the executive branch.
    E. A teacher is in charge of the Supreme Court.
    To determine the correct statement, let's analyze each option step by step:
    
    A.
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    B. Brussels
    C. London
    D. Rome
    The capital of France is:
    
    A. Paris
    
    Paris is the capital city of France, located in the Paris region in the south of the country. It is known for its rich history, beautiful architecture, and vibrant culture, making it the most famous and iconic city in France. Other cities in the French region include:
    
    B. Brussels (capital of Belgium)
    C. London (capital of the United Kingdom)
    D. Rome (capital of Italy)
    
    None of the other cities listed are the capital of France. Therefore, the correct answer is A. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. By 2023, the world will have more than 1 billion people who will be using AI and machine learning to improve their lives. This will be a key driver for the rapid development of AI and machine learning. The speed and the scope of AI and machine learning are increasing at a rapid pace, and they have the potential to completely transform how we live our lives. That is why it is important for people to understand what AI and machine learning is and how it can impact our lives.
    
    AI and machine learning can be broken down into different components. One of the most important components of AI and machine learning is algorithms


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Skill or Hobby] enthusiast. I love [What I like to do] and I am always looking for new challenges and opportunities to learn and grow. I am a [Favorite Book, Movie, or Sport] lover. I am passionate about [What I believe in] and I am always striving to make the world a better place. I am a [Personality trait or quality] person. I am [What I am most proud of]. I am [What I am most excited about]. I am [What I am most proud of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many world-renowned museums, theaters, and other cultural institutions. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Paris is a major transportation hub, with many major highways, railroads, and airports, and is a popular tourist destination. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to finance. This will lead to increased automation of tasks, which will require more human oversight and control.
    
    2. AI ethics: As AI becomes more integrated into our daily lives, there will be a growing need for ethical guidelines and regulations to ensure that AI is used in a responsible and beneficial way.
    
    3. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes,
    


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
    Generated text:  [Name], and I'm an AI. I've been here for over 5 years, and I'm here to assist you with all your questions and concerns. I'm here to help you learn, and I'm here to keep you entertained. So, if you have any questions or concerns, please feel free to ask, and I'll do my best to help you. Thank you for asking! 📚✨📚 #MyAIName #AskMeAnything #OpenToHelp #LifeWithAI #ExpertSupporter #TechTalker #TechSavvy #TechSavvyChat #ChatWithTechExpert #Tech
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    In short, Paris is the largest and most populous city in Europe, the world's most-visited city, and the birthplace of the French Revolution. Paris is known for its rich history, diverse culture, iconic landmarks, and artistic legacy. The city is also known for its cuisine, fashion, and wine, making it a global city with a rich cultural heritage. It is home to a number of famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many more. 
    
    As the political and cultural center of France, Paris plays an important role in the country's history and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several trends that are currently emerging and are expected to continue in the coming years. Some of these trends include:
    
    1. Increased Use of AI in Healthcare: With the increasing demand for accurate and personalized health care, AI is expected to play a significant role in the healthcare industry. Medical professionals will be able to use AI algorithms to analyze large amounts of data and provide better diagnosis and treatment options. AI will also be used to develop personalized medicine and to predict disease outbreaks.
    
    2. Advanced Autonomous Vehicles: Self-driving cars and other autonomous vehicles are expected to become more prevalent in the future. These vehicles will be able to navigate


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

    'm

     a

     [

    Your

     Profession

     or

     Occupation

    ]

     who

     has

     a

     passion

     for

     [

    Your

     Interest

     or

     Hobby

    ].

     I

     believe

     that

     my

     skills

     and

     experience

     are

     invaluable

     in

     [

    Your

     Profession

     or

     Hobby

    ].

     I

     am

     confident

     in

     my

     ability

     to

     [

    Your

     Goal

     or

     Achievement

    ],

     and

     I

     am

     eager

     to

     [

    Your

     Next

     Step

     or

     Challenge

    ].

     I

     am

     excited

     to

     learn

     more

     about

     you

     and

     how

     I

     can

     contribute

     to

     your

     success

     and

     growth

    .

     [

    Your

     Name

    ]

     Self

    -int

    roduction

    :

     I

     am

     [

    Your

     Name

    ]

     and

     I

     am

     a

     [

    Your

     Profession

     or

     Occupation

    ]

     with

     a

     passion

     for

     [

    Your

     Interest

     or

     Hobby

    ].

     I

     believe

     that

     my

     skills

     and

     experience

     are

     invaluable

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     commonly

     referred

     to

     as

     "

    the

     city

     of

     a

     thousand

     islands

    "

     due

     to

     its

     extensive

     network

     of

     can

    als

     and

     water

    ways

    .

     It

     is

     the

     seat

     of

     the

     Government

     and

     the

     largest

     city

     in

     France

    ,

     with

     a

     population

     of

     around

     

    1

    .

    8

     million

     inhabitants

    .

     Paris

     is

     also

     known

     for

     its

     rich

     culture

    ,

     art

    ,

     and

     cuisine

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     The

     city

     has

     a

     long

     and

     rich

     history

    ,

     which

     dates

     back

     to

     the

     Roman

     Empire

     and

     has

     continued

     to

     be

     a

     center

     of

     culture

     and

     power

     for

     centuries

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     is

     ranked

     as

     the

     

    9

    th

     most

     expensive

     city

     in

     the

     world

    ,

     although

     this

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     unpredictable

     and

     variable

    .

     Here

     are

     some

     potential

     trends

     that

     could

     affect

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     AI

     will

     become

     more

     accessible

     and

     affordable

    :

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

     AI

     become

     more

     accessible

     to

     a

     wider

     range

     of

     people

    .

     This

     could

     include

     more

     widespread

     adoption

     of

     AI

     in

     everyday

     life

    ,

     such

     as

     in

     self

    -driving

     cars

     and

     voice

    -

    activated

     devices

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     into

     various

     industries

    :

     AI

     is

     already

     being

     used

     in

     many

     industries

    ,

     and

     we

     can

     expect

     to

     see

     it

     increasingly

     integrated

     into

     other

     areas

    .

     This

     could

     include

     a

     wider

     range

     of

     industries

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
