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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:45,  1.88s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:45,  1.88s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:04<01:45,  1.88s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:41,  1.31it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:41,  1.31it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:41,  1.31it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:23,  2.26it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:14,  3.42it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:14,  3.42it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:14,  3.42it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:09,  4.84it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:09,  4.84it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:09,  4.84it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  6.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  6.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:07,  6.49it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  8.25it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  8.25it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  8.25it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:05,  8.25it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.38it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.38it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.48it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.48it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.03it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.03it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.03it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.03it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:01, 20.43it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 20.43it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:06<00:01, 22.59it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:06<00:01, 22.59it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:06<00:01, 22.59it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:06<00:01, 22.59it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:00, 24.21it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:00, 24.21it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:00, 24.21it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:06<00:00, 24.21it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:00, 24.34it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:00, 24.34it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:00, 24.34it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:00, 24.34it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:00, 24.34it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 26.61it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 26.61it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 26.61it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 26.61it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 26.34it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 26.34it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 26.34it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 26.34it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 26.34it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 27.32it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 27.32it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 27.32it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 27.32it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 27.89it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 27.89it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 27.89it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 27.89it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 28.00it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 28.00it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 28.00it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 28.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=40.96 GB):   2%|▏         | 1/58 [00:00<00:12,  4.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.93 GB):   2%|▏         | 1/58 [00:00<00:12,  4.41it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=40.93 GB):   3%|▎         | 2/58 [00:00<00:12,  4.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.93 GB):   3%|▎         | 2/58 [00:00<00:12,  4.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.93 GB):   5%|▌         | 3/58 [00:00<00:11,  4.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.92 GB):   5%|▌         | 3/58 [00:00<00:11,  4.93it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.92 GB):   7%|▋         | 4/58 [00:00<00:10,  5.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.92 GB):   7%|▋         | 4/58 [00:00<00:10,  5.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.92 GB):   9%|▊         | 5/58 [00:00<00:09,  5.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.92 GB):   9%|▊         | 5/58 [00:00<00:09,  5.36it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=40.92 GB):  10%|█         | 6/58 [00:01<00:09,  5.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.91 GB):  10%|█         | 6/58 [00:01<00:09,  5.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.91 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.90 GB):  12%|█▏        | 7/58 [00:01<00:08,  5.95it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=40.90 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.90 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.90 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.77it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.90 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.77it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=40.90 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.89 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.89 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.89 GB):  19%|█▉        | 11/58 [00:01<00:06,  7.37it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=40.89 GB):  21%|██        | 12/58 [00:01<00:05,  7.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.89 GB):  21%|██        | 12/58 [00:01<00:05,  7.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.89 GB):  21%|██        | 12/58 [00:01<00:05,  7.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.89 GB):  24%|██▍       | 14/58 [00:02<00:04, 10.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.88 GB):  24%|██▍       | 14/58 [00:02<00:04, 10.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.88 GB):  24%|██▍       | 14/58 [00:02<00:04, 10.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.87 GB):  24%|██▍       | 14/58 [00:02<00:04, 10.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.87 GB):  24%|██▍       | 14/58 [00:02<00:04, 10.27it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=40.87 GB):  24%|██▍       | 14/58 [00:02<00:04, 10.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.87 GB):  33%|███▎      | 19/58 [00:02<00:01, 19.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.87 GB):  33%|███▎      | 19/58 [00:02<00:01, 19.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.85 GB):  33%|███▎      | 19/58 [00:02<00:01, 19.98it/s]Capturing num tokens (num_tokens=960 avail_mem=40.86 GB):  33%|███▎      | 19/58 [00:02<00:01, 19.98it/s] Capturing num tokens (num_tokens=896 avail_mem=40.86 GB):  33%|███▎      | 19/58 [00:02<00:01, 19.98it/s]Capturing num tokens (num_tokens=832 avail_mem=40.85 GB):  33%|███▎      | 19/58 [00:02<00:01, 19.98it/s]Capturing num tokens (num_tokens=832 avail_mem=40.85 GB):  41%|████▏     | 24/58 [00:02<00:01, 27.21it/s]Capturing num tokens (num_tokens=768 avail_mem=40.85 GB):  41%|████▏     | 24/58 [00:02<00:01, 27.21it/s]Capturing num tokens (num_tokens=704 avail_mem=40.85 GB):  41%|████▏     | 24/58 [00:02<00:01, 27.21it/s]Capturing num tokens (num_tokens=640 avail_mem=40.84 GB):  41%|████▏     | 24/58 [00:02<00:01, 27.21it/s]Capturing num tokens (num_tokens=576 avail_mem=40.84 GB):  41%|████▏     | 24/58 [00:02<00:01, 27.21it/s]

    Capturing num tokens (num_tokens=512 avail_mem=40.83 GB):  41%|████▏     | 24/58 [00:02<00:01, 27.21it/s]Capturing num tokens (num_tokens=512 avail_mem=40.83 GB):  50%|█████     | 29/58 [00:02<00:00, 32.67it/s]Capturing num tokens (num_tokens=480 avail_mem=40.84 GB):  50%|█████     | 29/58 [00:02<00:00, 32.67it/s]Capturing num tokens (num_tokens=448 avail_mem=40.84 GB):  50%|█████     | 29/58 [00:02<00:00, 32.67it/s]Capturing num tokens (num_tokens=416 avail_mem=40.84 GB):  50%|█████     | 29/58 [00:02<00:00, 32.67it/s]Capturing num tokens (num_tokens=384 avail_mem=40.84 GB):  50%|█████     | 29/58 [00:02<00:00, 32.67it/s]Capturing num tokens (num_tokens=384 avail_mem=40.84 GB):  57%|█████▋    | 33/58 [00:02<00:00, 31.87it/s]Capturing num tokens (num_tokens=352 avail_mem=39.70 GB):  57%|█████▋    | 33/58 [00:02<00:00, 31.87it/s]

    Capturing num tokens (num_tokens=320 avail_mem=39.69 GB):  57%|█████▋    | 33/58 [00:02<00:00, 31.87it/s]Capturing num tokens (num_tokens=288 avail_mem=39.69 GB):  57%|█████▋    | 33/58 [00:02<00:00, 31.87it/s]Capturing num tokens (num_tokens=256 avail_mem=40.79 GB):  57%|█████▋    | 33/58 [00:02<00:00, 31.87it/s]

    Capturing num tokens (num_tokens=256 avail_mem=40.79 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.83it/s]Capturing num tokens (num_tokens=240 avail_mem=40.79 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.83it/s]Capturing num tokens (num_tokens=224 avail_mem=39.80 GB):  64%|██████▍   | 37/58 [00:03<00:01, 18.83it/s]

    Capturing num tokens (num_tokens=208 avail_mem=39.79 GB):  64%|██████▍   | 37/58 [00:03<00:01, 18.83it/s]Capturing num tokens (num_tokens=208 avail_mem=39.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.14it/s]Capturing num tokens (num_tokens=192 avail_mem=39.79 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.14it/s]

    Capturing num tokens (num_tokens=176 avail_mem=40.77 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.14it/s]Capturing num tokens (num_tokens=160 avail_mem=39.85 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.14it/s]

    Capturing num tokens (num_tokens=160 avail_mem=39.85 GB):  74%|███████▍  | 43/58 [00:03<00:01, 10.68it/s]Capturing num tokens (num_tokens=144 avail_mem=39.85 GB):  74%|███████▍  | 43/58 [00:03<00:01, 10.68it/s]Capturing num tokens (num_tokens=128 avail_mem=39.84 GB):  74%|███████▍  | 43/58 [00:03<00:01, 10.68it/s]

    Capturing num tokens (num_tokens=128 avail_mem=39.84 GB):  78%|███████▊  | 45/58 [00:04<00:01,  9.64it/s]Capturing num tokens (num_tokens=112 avail_mem=40.76 GB):  78%|███████▊  | 45/58 [00:04<00:01,  9.64it/s]Capturing num tokens (num_tokens=96 avail_mem=39.90 GB):  78%|███████▊  | 45/58 [00:04<00:01,  9.64it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=39.90 GB):  81%|████████  | 47/58 [00:04<00:01,  8.88it/s]Capturing num tokens (num_tokens=80 avail_mem=39.90 GB):  81%|████████  | 47/58 [00:04<00:01,  8.88it/s]Capturing num tokens (num_tokens=64 avail_mem=40.75 GB):  81%|████████  | 47/58 [00:04<00:01,  8.88it/s]

    Capturing num tokens (num_tokens=64 avail_mem=40.75 GB):  84%|████████▍ | 49/58 [00:04<00:01,  8.44it/s]Capturing num tokens (num_tokens=48 avail_mem=39.96 GB):  84%|████████▍ | 49/58 [00:04<00:01,  8.44it/s]Capturing num tokens (num_tokens=32 avail_mem=39.96 GB):  84%|████████▍ | 49/58 [00:04<00:01,  8.44it/s]

    Capturing num tokens (num_tokens=32 avail_mem=39.96 GB):  88%|████████▊ | 51/58 [00:04<00:00,  7.83it/s]Capturing num tokens (num_tokens=28 avail_mem=39.95 GB):  88%|████████▊ | 51/58 [00:04<00:00,  7.83it/s]Capturing num tokens (num_tokens=28 avail_mem=39.95 GB):  90%|████████▉ | 52/58 [00:05<00:00,  7.89it/s]Capturing num tokens (num_tokens=24 avail_mem=40.74 GB):  90%|████████▉ | 52/58 [00:05<00:00,  7.89it/s]

    Capturing num tokens (num_tokens=24 avail_mem=40.74 GB):  91%|█████████▏| 53/58 [00:05<00:00,  7.74it/s]Capturing num tokens (num_tokens=20 avail_mem=40.01 GB):  91%|█████████▏| 53/58 [00:05<00:00,  7.74it/s]Capturing num tokens (num_tokens=20 avail_mem=40.01 GB):  93%|█████████▎| 54/58 [00:05<00:00,  7.55it/s]Capturing num tokens (num_tokens=16 avail_mem=40.01 GB):  93%|█████████▎| 54/58 [00:05<00:00,  7.55it/s]

    Capturing num tokens (num_tokens=16 avail_mem=40.01 GB):  95%|█████████▍| 55/58 [00:05<00:00,  7.63it/s]Capturing num tokens (num_tokens=12 avail_mem=40.73 GB):  95%|█████████▍| 55/58 [00:05<00:00,  7.63it/s]Capturing num tokens (num_tokens=12 avail_mem=40.73 GB):  97%|█████████▋| 56/58 [00:05<00:00,  7.57it/s]Capturing num tokens (num_tokens=8 avail_mem=40.07 GB):  97%|█████████▋| 56/58 [00:05<00:00,  7.57it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=40.07 GB):  98%|█████████▊| 57/58 [00:05<00:00,  7.25it/s]Capturing num tokens (num_tokens=4 avail_mem=40.06 GB):  98%|█████████▊| 57/58 [00:05<00:00,  7.25it/s]Capturing num tokens (num_tokens=4 avail_mem=40.06 GB): 100%|██████████| 58/58 [00:05<00:00,  7.57it/s]Capturing num tokens (num_tokens=4 avail_mem=40.06 GB): 100%|██████████| 58/58 [00:05<00:00,  9.91it/s]


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
    Generated text:  Sam and I am from New York City. I have always loved horses since I was a child and I had been riding horses since I was 6 years old. I am currently 25 years old and I have been a horse trainer for over 15 years. I have been successful at getting my horses to perform tricks and I have also helped horses with physical problems. I have also worked with horses in therapy and have assisted with training horses with anxiety. I have also learned many new things about horses and I hope to one day become a horse doctor or a horse surgeon. I hope to have the opportunity to train horses to fly
    ===============================
    Prompt: The president of the United States is
    Generated text:  a federal executive officer. The position is responsible for guiding the country's policies in the United States. In the United States, the president is a member of the United States legislative branch of government. The position of president has been held since 1787. The country has had two presidents since then. The current president is Donald Trump. On the first day of his presidency, the president called the United States to a "day of national mourning". The president chose to honor this by appointing two uncles to the United States Supreme Court. One of the uncles is Benjamin Franklin, who has been a member of the United States
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lisbon
    C. London
    D. Rome
    
    To determine the capital of France, let's review the options provided:
    
    A. Paris
    B. Lisbon
    C. London
    D. Rome
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, and as the science of artificial intelligence has advanced, so too has the world around us. One of the most promising areas of AI is the use of deep learning, which is a type of machine learning that involves training computers to perform tasks that would typically require human intelligence. Deep learning has the potential to revolutionize a wide range of industries, from healthcare to finance, and it has already shown promise in a variety of applications.
    One of the key applications of deep learning is in the field of image recognition. Deep learning algorithms can be trained on large amounts of data, and they can then be used to identify patterns and features


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm confident in my abilities and I'm eager to [action or goal]. I'm [age] years old, and I'm [gender] and [race]. I'm [occupation] and I have [number of years] of experience in the industry. I'm [reason for passion], and I'm always looking for ways to [action or goal]. I'm confident in my abilities and I'm eager to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous croissants and its many traditional French dishes. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is both
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more efficient and capable of performing tasks that were previously done by humans. This could lead to a significant increase in automation in various industries, including manufacturing, transportation, and healthcare.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing concern about its impact on society. This includes issues such as bias in AI algorithms, privacy concerns, and the potential
    


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
    Generated text:  [insert name] and I am a [insert occupation] and a [insert profession]. I have always been a strong believer in the idea that [insert reason for belief], and have always been passionate about [insert area of interest, such as travel, music, or science]. I am always looking to expand my horizons and learn more about the world around me. I am also a great leader and have worked in a variety of roles, including [insert roles, such as team leader, consultant, or mentor], and have a genuine passion for helping others succeed.
    As an [insert occupation] with a passion for [insert field of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and the world's 10th-largest city by population. Paris is the cultural and intellectual center of France and one of the world's most visited cities. It has been the capital of France since 1830 and continues to be so today. Paris is known for its landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and the Parc des Expositions. The city is also home to many museums, theaters, and art galleries. It is a popular tourist destination for people from around the world. Paris is known for its art,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a variety of trends and technologies that will continue to evolve and transform the field. Some of the key trends that are expected to drive the future of AI include:
    
    1. Increased use of machine learning and deep learning: AI systems are becoming more capable of learning and adapting to new data, which is driving the development of more advanced machine learning and deep learning models.
    
    2. Improved efficiency and accuracy: AI systems are becoming more accurate and efficient at performing tasks that were previously difficult or time-consuming. This is likely to continue as AI continues to be used in new and innovative ways.
    
    3. Greater integration of AI into our


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

    name

    ]

     and

     I

     am

     [

    age

    ].

     I

     am

     a

     [

    type

     of

     character

    ]

     (

    e

    .g

    .

     character

    ,

     villain

    ,

     etc

    .)

     who

     have

     been

     [

    reason

     for

     existence

    ]

     and

     have

     been

     able

     to

     [

    major

     achievement

     or

     outcome

    ].

     I

     am

     a

     [

    gender

     or

     race

    ]

     (

    e

    .g

    .

     white

    ,

     Asian

    ,

     etc

    .)

     and

     have

     been

     [

    reason

     for

     being

    ].

     I

     am

     [

    ability

     or

     skill

    ]

     and

     have

     been

     [

    reason

     for

     being

    ].

     I

     am

     [

    role

     in

     the

     world

    ]

     (

    e

    .g

    .

     protagonist

    ,

     antagonist

    ,

     etc

    .)

     and

     have

     been

     [

    reason

     for

     being

    ].

     I

     am

     [

    occupation

    ]

     and

     have

     been

     [

    reason

     for

     being

    ].

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     its

     historical

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     famous

     for

     its

     annual

     cout

    ure

     fashion

     show

     and

     its

     rich

     culture

     and

     cuisine

    .

     Paris

     is

     a

     major

     cultural

     hub

     with

     many

     important

     universities

     and

     museums

    .

     The

     French

     capital

     is

     known

     for

     its

     vibrant

     nightlife

     and

     is

     home

     to

     many

     famous

     landmarks

     and

     museums

    .

     It

     is

     often

     referred

     to

     as

     "

    the

     city

     of

     lights

    "

     and

     is

     one

     of

     the

     most

     important

     cities

     in

     Europe

    .

     In

     

    2

    0

    1

    9

    ,

     Paris

     saw

     an

     estimated

     population

     of

     around

     

    2

    .

    1

     million

     people

    .

     Paris

     is

     also

     a

     popular

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     highly

     dynamic

     and

     unpredictable

    ,

     with

     potential

     areas

     of

     innovation

     and

     progress

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

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     development

     of

     advanced

     AI

     algorithms

     and

     machine

     learning

    ,

     there

     is

     an

     increased

     potential

     for

     AI

     to

     be

     used

     in

     healthcare

    .

     AI

     can

     be

     used

     to

     analyze

     medical

     records

    ,

     predict

     patient

     outcomes

    ,

     and

     develop

     personalized

     treatment

     plans

    .

     This

     could

     lead

     to

     more

     accurate

     diagnoses

     and

     potentially

     save

     lives

    .
    


    2

    .

     AI

     integration

     with

     everyday

     products

    :

     The

     integration

     of

     AI

     into

     everyday

     products

     and

     services

     is

     becoming

     increasingly

     common

    .

     For

     example

    ,

     many

     smart

     home

     devices

     use

     AI

     algorithms

     to

     learn

     and

     adapt

     to

     user

    



```python
llm.shutdown()
```
