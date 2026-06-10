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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=56.38 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=56.38 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=120 avail_mem=56.28 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=112 avail_mem=56.28 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=104 avail_mem=56.28 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it]Capturing batches (bs=96 avail_mem=56.28 GB):   5%|▌         | 1/20 [00:01<00:21,  1.12s/it] Capturing batches (bs=96 avail_mem=56.28 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.13it/s]Capturing batches (bs=88 avail_mem=56.28 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.13it/s]Capturing batches (bs=80 avail_mem=56.28 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.13it/s]Capturing batches (bs=72 avail_mem=56.28 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.13it/s]

    Capturing batches (bs=64 avail_mem=56.28 GB):  25%|██▌       | 5/20 [00:01<00:02,  5.13it/s]Capturing batches (bs=64 avail_mem=56.28 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.58it/s]Capturing batches (bs=56 avail_mem=56.27 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.58it/s]Capturing batches (bs=48 avail_mem=56.27 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.58it/s]Capturing batches (bs=40 avail_mem=56.27 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.58it/s]Capturing batches (bs=32 avail_mem=56.27 GB):  45%|████▌     | 9/20 [00:01<00:01,  9.58it/s]Capturing batches (bs=32 avail_mem=56.27 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.74it/s]Capturing batches (bs=24 avail_mem=56.27 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.74it/s]Capturing batches (bs=16 avail_mem=56.27 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.74it/s]

    Capturing batches (bs=12 avail_mem=56.27 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.74it/s]Capturing batches (bs=12 avail_mem=56.27 GB):  80%|████████  | 16/20 [00:01<00:00, 15.66it/s]Capturing batches (bs=8 avail_mem=56.24 GB):  80%|████████  | 16/20 [00:01<00:00, 15.66it/s] Capturing batches (bs=4 avail_mem=56.24 GB):  80%|████████  | 16/20 [00:01<00:00, 15.66it/s]Capturing batches (bs=2 avail_mem=56.24 GB):  80%|████████  | 16/20 [00:01<00:00, 15.66it/s]Capturing batches (bs=1 avail_mem=56.24 GB):  80%|████████  | 16/20 [00:01<00:00, 15.66it/s]Capturing batches (bs=1 avail_mem=56.24 GB): 100%|██████████| 20/20 [00:01<00:00, 19.86it/s]Capturing batches (bs=1 avail_mem=56.24 GB): 100%|██████████| 20/20 [00:01<00:00, 11.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:04<00:01, 18.29it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=28):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=24):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=20):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=16):  71%|███████   | 41/58 [00:04<00:00, 27.27it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 44.41it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 44.41it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 44.41it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 44.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.57 GB):   3%|▎         | 2/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.57 GB):   3%|▎         | 2/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.56 GB):   3%|▎         | 2/58 [00:00<00:03, 17.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.56 GB):   3%|▎         | 2/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.56 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.56 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.55 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.55 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.55 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.54 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.69it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.54 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.54 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.54 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.53 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.53 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.53 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.52 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.52 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.52 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.72it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.51 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.51 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.51 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=960 avail_mem=76.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.37it/s] Capturing num tokens (num_tokens=896 avail_mem=76.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=896 avail_mem=76.50 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.67it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.50 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=768 avail_mem=76.49 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=704 avail_mem=76.49 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=640 avail_mem=76.49 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=576 avail_mem=76.49 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=576 avail_mem=76.49 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.98it/s]Capturing num tokens (num_tokens=512 avail_mem=76.47 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.98it/s]Capturing num tokens (num_tokens=480 avail_mem=76.49 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.98it/s]Capturing num tokens (num_tokens=448 avail_mem=76.49 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=416 avail_mem=76.48 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.98it/s]Capturing num tokens (num_tokens=384 avail_mem=76.48 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.98it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.48 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=352 avail_mem=76.48 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=320 avail_mem=76.47 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=288 avail_mem=76.47 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=256 avail_mem=76.47 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=240 avail_mem=76.46 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=240 avail_mem=76.46 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=224 avail_mem=76.46 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=208 avail_mem=76.46 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=192 avail_mem=76.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=176 avail_mem=76.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=160 avail_mem=76.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.91it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.45 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=144 avail_mem=76.45 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=128 avail_mem=76.44 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=112 avail_mem=76.44 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=96 avail_mem=76.44 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s] Capturing num tokens (num_tokens=80 avail_mem=76.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=80 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=64 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=48 avail_mem=76.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=32 avail_mem=76.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=28 avail_mem=76.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.60it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=24 avail_mem=76.42 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=20 avail_mem=76.41 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=16 avail_mem=76.41 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=12 avail_mem=76.41 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=8 avail_mem=76.40 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.12it/s] Capturing num tokens (num_tokens=4 avail_mem=76.40 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=4 avail_mem=76.40 GB): 100%|██████████| 58/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=4 avail_mem=76.40 GB): 100%|██████████| 58/58 [00:01<00:00, 35.97it/s]


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
    Generated text:  Shiela and I'm a Senior Computer Science and Engineering major at the University of Florida. I've always been passionate about math and computer science and I have been working on various coding and programming projects for over a decade now.\nI can play music, make music, and read music. I have a serious passion for music and I love playing instruments like the violin and guitar. I also love writing stories and I have been reading books for a long time.\nI have a lot of long-distance relationships and I enjoy talking to people, even strangers. I enjoy going on long drives and trying new foods and cultures. I love cooking and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is like the boss of the whole country. He or she helps to make important decisions about how to run the country. He or she makes the rules and laws. The president is also the most powerful person in the world. He or she is the leader of the country. Most of the people in the country want him or her to stay in power for as long as they can. The president is very busy, so it is hard for him or her to take time for himself. He or she works very hard every day and has a lot of problems. As a result, he
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Lille
    B. Lyon
    C. Paris
    D. Nancy
    Answer: C
    
    Female, 28 years old. She has had high fever, limb weakness, and gum bleeding for 2 days. Physical examination: T39.8°C, heart rate 126 beats/min, respiratory rate 24 breaths/min, alert, mild tenderness in the right lower quadrant, mild tenderness in the left lower quadrant, no rebound tenderness or muscle guarding, bowel sounds 10/minute, ECG shows ST-segment elevation in leads II, III, aVF
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the individuals who design, develop, and operate it. So, with the impending introduction of the virtual reality gaming market, players should be aware of the importance of immersion in the virtual environment. The VR gaming market is predicted to grow at a significant rate in the next few years, and it’s high time that we take measures to ensure that our gaming environment is immersive, compelling, and engaging.
    
    ### 1. Understanding the Importance of Immersion in VR Gaming
    
    The first step towards creating an immersive VR gaming experience is to understand the importance of immersion. Immersion refers to the way that a user feels connected to the


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
    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and is home to many famous landmarks and attractions. It is also a major center for French politics and culture. The city is known for its rich history and cultural heritage, and is a major hub for international trade and diplomacy. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its diverse population,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI that can learn from human behavior and adapt to new situations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of AI.
    
    3. Increased use of AI
    


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
    Generated text:  [Name], and I'm [Age]. I'm a [职业] with a [才能] and [技能] that I'm passionate about. I'm always looking for new challenges and learning new things. How can I get to know you better? Let me know if you're interested in [job title], [field], or [specialization]. If you're interested in [job title], please tell me about [responsibilities]. If you're interested in [field], please tell me about [research], [methods], or [tools]. If you're interested in [specialization], please tell me about [details]. If you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital city of France and serves as the administrative and cultural center of the country. It is known for its rich history, stunning architecture, vibrant culture, and annual fashion week. The city is also famous for its numerous museums, parks, and landmarks such as the Eiffel Tower and Notre-Dame Cathedral. Paris is a popular tourist destination and host to numerous international events and festivals. Its unique blend of old and new, and its cosmopolitan atmosphere make it a fascinating and vibrant city. It is one of the world’s most visited cities and is considered one of the most beautiful cities in the world. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve rapidly, with many exciting developments and trends in the coming years. Some of the potential trends in AI include:
    
    1. AI-AI integration: This is expected to become more common as AI becomes more sophisticated, and there will be more integration between AI systems. This will allow for more powerful and intelligent AI systems that can perform tasks more efficiently and effectively.
    
    2. AI ethics: There is a growing concern about the ethical implications of AI, and it is expected that AI systems will need to be developed and used with care. There will be more efforts to ensure that AI is used to benefit society as a whole,


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

     Age

    ]

     year

     old

    ,

     [

    Your

     Profession

    ]

     and

     [

    Your

     Job

     Title

    ]

     in

     the

     [

    Your

     Company

     Name

    ]

     team

    .

     I

    'm

     [

    Your

     Qual

    ifications

    ]

     and

     I

     am

     passionate

     about

     [

    Your

     Field

     of

     Interest

    ]

     and

     my

     work

     is

     dedicated

     to

     [

    Your

     Mission

    ].

     I

     love

     to

     [

    Your

     Hab

    its

    /

    Activities

    ].

     I

     am

     [

    Your

     Personality

     Traits

    ]

     and

     I

     am

     always

     looking

     for

     [

    Your

     Strength

    s

    /

    Weak

    ness

    es

    ]

     in

     order

     to

     improve

     myself

    .

     I

     am

     [

    Your

     Mot

    ivation

    ]

     and

     I

     always

     aim

     to

     [

    Your

     Achie

    vements

    /

    Goals

    ].

     I

     am

     very

     [

    Your

     Personality

     Type

    ]

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     atmosphere

     and

     many

     cultural

     institutions

     that

     celebrate

     love

     and

     romance

    .

     
    


    This

     statement

     encaps

    ulates

     Paris

    's

     status

     as

     the

     largest

     city

     in

     Europe

     and

     a

     global

     met

    ropolis

    ,

     particularly

     known

     for

     its

     romantic

     attractions

    ,

     cultural

     institutions

    ,

     and

     its

     passionate

     and

     eclectic

     population

    .

     
    


    For

     a

     more

     detailed

     summary

    ,

     it

     could

     be

     stated

    :

     "

    Paris

    ,

     the

     largest

     city

     in

     Europe

     and

     a

     global

     met

    ropolis

    ,

     is

     renowned

     for

     its

     romantic

     atmosphere

    ,

     cultural

     institutions

     that

     celebrate

     love

     and

     romance

    ,

     and

     its

     passionate

     and

     eclectic

     population

    ."

     
    


    This

     statement

     touches

     on

     Paris

    's

     historical

     significance

    ,

     its

     cultural

     achievements

    ,

     and

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     full

     of

     possibilities

    ,

     and

     there

     are

     many

     different

     trends

     that

     are

     likely

     to

     shape

     it

    .

     Some

     of

     the

     most

     likely

     trends

     include

    :
    


    1

    .

     Self

    -learning

     and

     autonomy

    :

     As

     AI

     systems

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

     capable

     of

     self

    -learning

     and

     making

     decisions

     on

     their

     own

    .

     This

     could

     mean

     more

     autonomous

     robots

     and

     self

    -driving

     cars

    ,

     as

     well

     as

     more

     complex

     AI

     systems

     that

     can

     make

     decisions

     based

     on

     context

     and

     knowledge

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     improve

     patient

     care

     in

     healthcare

    ,

     from

     analyzing

     medical

     images

     to

     predicting

     disease

     risks

    .

     As

     AI

     continues

     to

     improve

     and

     become

     more

     integrated

     into

     healthcare

    ,

     we

     may

     see

    



```python
llm.shutdown()
```
