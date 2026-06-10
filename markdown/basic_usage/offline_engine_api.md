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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.38it/s]


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=55.89 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=55.89 GB):   5%|▌         | 1/20 [00:01<00:23,  1.23s/it]Capturing batches (bs=120 avail_mem=55.79 GB):   5%|▌         | 1/20 [00:01<00:23,  1.23s/it]Capturing batches (bs=112 avail_mem=55.79 GB):   5%|▌         | 1/20 [00:01<00:23,  1.23s/it]Capturing batches (bs=104 avail_mem=55.79 GB):   5%|▌         | 1/20 [00:01<00:23,  1.23s/it]Capturing batches (bs=104 avail_mem=55.79 GB):  20%|██        | 4/20 [00:01<00:04,  3.82it/s]Capturing batches (bs=96 avail_mem=55.79 GB):  20%|██        | 4/20 [00:01<00:04,  3.82it/s] Capturing batches (bs=88 avail_mem=55.79 GB):  20%|██        | 4/20 [00:01<00:04,  3.82it/s]Capturing batches (bs=80 avail_mem=55.79 GB):  20%|██        | 4/20 [00:01<00:04,  3.82it/s]

    Capturing batches (bs=80 avail_mem=55.79 GB):  35%|███▌      | 7/20 [00:01<00:01,  7.14it/s]Capturing batches (bs=72 avail_mem=55.79 GB):  35%|███▌      | 7/20 [00:01<00:01,  7.14it/s]Capturing batches (bs=64 avail_mem=55.78 GB):  35%|███▌      | 7/20 [00:01<00:01,  7.14it/s]Capturing batches (bs=56 avail_mem=55.78 GB):  35%|███▌      | 7/20 [00:01<00:01,  7.14it/s]Capturing batches (bs=56 avail_mem=55.78 GB):  50%|█████     | 10/20 [00:01<00:00, 10.51it/s]Capturing batches (bs=48 avail_mem=55.78 GB):  50%|█████     | 10/20 [00:01<00:00, 10.51it/s]Capturing batches (bs=40 avail_mem=55.78 GB):  50%|█████     | 10/20 [00:01<00:00, 10.51it/s]Capturing batches (bs=32 avail_mem=55.78 GB):  50%|█████     | 10/20 [00:01<00:00, 10.51it/s]

    Capturing batches (bs=32 avail_mem=55.78 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.76it/s]Capturing batches (bs=24 avail_mem=55.78 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.76it/s]Capturing batches (bs=16 avail_mem=55.78 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.76it/s]Capturing batches (bs=12 avail_mem=55.77 GB):  65%|██████▌   | 13/20 [00:01<00:00, 13.76it/s]Capturing batches (bs=12 avail_mem=55.77 GB):  80%|████████  | 16/20 [00:01<00:00, 15.67it/s]Capturing batches (bs=8 avail_mem=55.77 GB):  80%|████████  | 16/20 [00:01<00:00, 15.67it/s] Capturing batches (bs=4 avail_mem=55.77 GB):  80%|████████  | 16/20 [00:01<00:00, 15.67it/s]

    Capturing batches (bs=2 avail_mem=55.77 GB):  80%|████████  | 16/20 [00:01<00:00, 15.67it/s]Capturing batches (bs=1 avail_mem=55.77 GB):  80%|████████  | 16/20 [00:01<00:00, 15.67it/s]Capturing batches (bs=1 avail_mem=55.77 GB): 100%|██████████| 20/20 [00:01<00:00, 19.85it/s]Capturing batches (bs=1 avail_mem=55.77 GB): 100%|██████████| 20/20 [00:01<00:00, 10.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.67it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 15.67it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 22.05it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=20):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=16):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]

    Compiling num tokens (num_tokens=12):  74%|███████▍  | 43/58 [00:05<00:00, 30.19it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 46.82it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 46.82it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 46.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.45 GB):   3%|▎         | 2/58 [00:00<00:03, 15.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.45 GB):   3%|▎         | 2/58 [00:00<00:03, 15.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.44 GB):   3%|▎         | 2/58 [00:00<00:03, 15.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.44 GB):   7%|▋         | 4/58 [00:00<00:03, 16.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.44 GB):   7%|▋         | 4/58 [00:00<00:03, 16.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.44 GB):   7%|▋         | 4/58 [00:00<00:03, 16.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.42 GB):   7%|▋         | 4/58 [00:00<00:03, 16.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.42 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.42 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.42 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.17it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.42 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.42 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.41 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.41 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.41 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.41 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.41 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.72it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=55.40 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.40 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.39 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.37 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=960 avail_mem=55.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.72it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=55.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=832 avail_mem=55.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=768 avail_mem=55.37 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=768 avail_mem=55.37 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.24it/s]Capturing num tokens (num_tokens=704 avail_mem=55.37 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.24it/s]Capturing num tokens (num_tokens=640 avail_mem=55.37 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.24it/s]Capturing num tokens (num_tokens=576 avail_mem=55.36 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.24it/s]Capturing num tokens (num_tokens=512 avail_mem=55.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.24it/s]Capturing num tokens (num_tokens=480 avail_mem=55.37 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.24it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.37 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=448 avail_mem=55.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=416 avail_mem=55.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=384 avail_mem=55.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=352 avail_mem=55.35 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=352 avail_mem=55.35 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=320 avail_mem=55.35 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=288 avail_mem=55.35 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=256 avail_mem=55.34 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=240 avail_mem=55.34 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.65it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.34 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=224 avail_mem=55.34 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=208 avail_mem=55.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=192 avail_mem=55.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=176 avail_mem=55.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=176 avail_mem=55.33 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.48it/s]Capturing num tokens (num_tokens=160 avail_mem=55.33 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.48it/s]Capturing num tokens (num_tokens=144 avail_mem=55.33 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.48it/s]Capturing num tokens (num_tokens=128 avail_mem=55.32 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.48it/s]Capturing num tokens (num_tokens=112 avail_mem=55.32 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.48it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.32 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=96 avail_mem=55.32 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.57it/s] Capturing num tokens (num_tokens=80 avail_mem=55.31 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=64 avail_mem=55.31 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=48 avail_mem=55.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=48 avail_mem=55.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=32 avail_mem=55.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=28 avail_mem=55.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=24 avail_mem=55.29 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.29 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=20 avail_mem=55.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=16 avail_mem=55.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=12 avail_mem=55.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=8 avail_mem=55.28 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s] Capturing num tokens (num_tokens=4 avail_mem=55.28 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=4 avail_mem=55.28 GB): 100%|██████████| 58/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=4 avail_mem=55.28 GB): 100%|██████████| 58/58 [00:01<00:00, 31.36it/s]


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
    Generated text:  Joseph and I am a product manager for a leading digital marketing company. We are developing a product that will allow people to easily create and share short, high-quality videos. We are currently working on the final design and the next step in the process is to gather feedback on the product design. We are not confident that we have a good understanding of the intended use of the product. We have asked a few people who know about digital marketing to help us gather this information. 
    
    The questions are:
    
    1. How does the product create social proof for the videos?
    2. How does the product help users to understand what they are creating?
    3
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a new tax on the wealthiest 0.01% of Americans. If the tax is to be shared evenly among the 400,000 wealthiest Americans, how much would each person have to pay in tax?
    To determine how much each person in the wealthiest 0.01% of Americans would have to pay in tax, we need to follow these steps:
    
    1. **Identify the total wealth of the wealthiest 0.01% of Americans:**
       - The wealthiest 0.01% of Americans comprise 400,000 individuals.
       - Therefore, the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    答案: A
    
    柴油机的种类很多，按用途分为两大类：动力性、经济性、操纵性、机动性和安全性。其中动力性主要指柴油机的____
    A. 动力性和经济性
    B. 动力性和操纵性
    C. 动力性和安全性
    答案: A
    
    不能用于诊断颈椎病的检查方法是
    A. 检查颈椎及后头部的X线平片
    B. 测量颈椎前后径和横径
    C. 肌电图
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but also with a lot of risks, said Jairus Cavalcanti, CEO of Voss.ai.
    Voss.ai, a leading global provider of AI platform and software, announced its acquisition of DeepMind Vision, a global leader in artificial intelligence.
    The acquisition of DeepMind Vision, a global leader in artificial intelligence, has become a reality, and Voss.ai, a leading global provider of AI platform and software, announced its acquisition of DeepMind Vision, a global leader in artificial intelligence.
    Founded in 2014, DeepMind Vision has been building deep learning systems for over 15 years, and it


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that was founded in 789 AD and is the largest city in the European Union. It is also the seat of the French government, the French parliament, and the headquarters of the French Foreign Ministry. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in Europe. The city is known for its fashion industry, art, and cuisine. It is also home to many cultural institutions such as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there will be a greater emphasis on developing ethical AI that is designed to minimize harm and maximize benefits.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, from self-driving cars to smart homes. As more companies and governments invest in AI, it is likely that we will see more integration of AI with other technologies, such
    


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
    Generated text:  [Character Name], and I'm a [job title] at [organization name]. I have [short biography or significant accomplishment/achievements] at [organization name], and I enjoy [a brief personal trait or quality]. I'm always up for a challenge and have a passion for [a sport, hobby, etc.]. What's your name, and what brings you to this moment in time? [Character Name] is passionate about [job title] and enjoys [job title] work. I have always been drawn to the challenge of [job title] because [reason why] and I am excited to help [organization name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city in the country and the third largest in Europe.
    
    The capital of France is Paris, which is the largest city in the country and the third largest in Europe. 
    
    This information is concise, factual, and provides the essential facts about the capital city's importance and size in France. It does not contain any additional context or information beyond the key points. A potential advantage of this statement is its brevity, which can be useful for quick reference or publication. However, it is important to note that this statement is not entirely accurate, as Paris is not the only city in France, and there are other
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are many possible trends that are shaping its direction. Here are some of the most likely future trends in AI:
    
    1. Integration with human decision-making: AI is becoming more integrated with human decision-making processes, allowing it to make more accurate and informed decisions.
    
    2. Advancements in machine learning: Machine learning is becoming more powerful and accurate, allowing AI to solve more complex problems and make faster decisions.
    
    3. Increased focus on ethical considerations: There is a growing need for ethical considerations when it comes to AI, as it can have unintended consequences if not properly designed and used.
    
    4. Integration with other technologies: AI


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

     software

     engineer

     with

     over

     

    1

    0

     years

     of

     experience

     in

     web

     development

    ,

     design

    ,

     and

     project

     management

    .

     I

    'm

     fluent

     in

     multiple

     programming

     languages

     and

     enjoy

     solving

     complex

     problems

     with

     creativity

     and

     innovation

    .

     I

    'm

     a

     highly

     organized

     and

     detail

    -oriented

     person

    ,

     and

     I

     believe

     that

     collaboration

     with

     other

     team

     members

     is

     essential

     for

     the

     success

     of

     any

     project

    .

     I

    'm

     a

     team

     player

     who

     thr

    ives

     in

     fast

    -paced

     environments

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     technologies

    .

     I

    'm

     looking

     forward

     to

     starting

     a

     new

     chapter

     in

     my

     career

     and

     I

    'm

     excited

     to

     learn

     more

     about

     your

     company

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Why

     is

     Paris

     considered

     the

     capital

     of

     France

    ?

     Paris

     is

     the

     capital

     of

     France

     due

     to

     its

     unique

     geographical

     location

    ,

     which

     gives

     it

     strategic

     advantages

     in

     terms

     of

     natural

     resources

    ,

     transportation

    ,

     and

     communication

    .

     As

     the

     oldest

     capital

     of

     France

    ,

     Paris

     has

     a

     rich

     and

     complex

     history

     that

     continues

     to

     influence

     the

     city

    's

     culture

    ,

     society

    ,

     and

     economy

    .

     The

     city

    's

     status

     as

     a

     major

     economic

    ,

     cultural

    ,

     and

     political

     center

     of

     Europe

     and

     the

     world

     is

     a

     testament

     to

     its

     importance

     in

     French

     history

     and

     culture

    .

     Additionally

    ,

     Paris

     has

     been

     a

     favorite

     tourist

     destination

     since

     its

     founding

    ,

     attracting

     millions

     of

     visitors

     each

     year

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     op

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     is

     shaped

     by

     a

     complex

     inter

    play

     of

     technological

     and

     societal

     factors

    .

     However

    ,

     several

     trends

     are

     expected

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     Integration

    :

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     allowing

     us

     to

     automate

     routine

     tasks

     and

     enhance

     productivity

    .

     This

     trend

     is

     likely

     to

     result

     in

     more

     efficient

     and

     cost

    -effective

     AI

     applications

    ,

     but

     it

     also

     raises

     concerns

     about

     job

     displacement

     and

     the

     need

     for

     re

    training

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     and

     become

     more

     personalized

    ,

     enabling

     more

     accurate

     predictions

     and

     recommendations

    .

     This

     trend

     is

     likely

     to

     lead

     to

     more

     personalized

     services

     and

     experiences

    ,

     but

     it

     also

     raises

     concerns

     about

    



```python
llm.shutdown()
```
